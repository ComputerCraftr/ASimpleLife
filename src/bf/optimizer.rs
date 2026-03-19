use std::collections::BTreeMap;

use super::ir::BfIr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Char,
    Number,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellSign {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodegenOpts {
    pub io_mode: IoMode,
    pub cell_bits: u32,
    pub input_bits: Option<u32>,
    pub output_bits: Option<u32>,
    pub cell_sign: CellSign,
}

pub fn optimize(program: Vec<BfIr>) -> Vec<BfIr> {
    let program = rewrite_ir_bottom_up(program, &canonicalize_loop_body);
    cleanup_lowering_ir(program)
}

// --- Pass 1: normalize_sequence ---

fn normalize_distribute_targets(targets: &[(isize, i32)]) -> BfIr {
    let mut merged = BTreeMap::<isize, i32>::new();
    for &(offset, coeff) in targets {
        if offset == 0 || coeff == 0 {
            continue;
        }
        *merged.entry(offset).or_insert(0) += coeff;
    }
    let targets = merged
        .into_iter()
        .filter_map(|(offset, coeff)| (coeff != 0).then_some((offset, coeff)))
        .collect::<Vec<_>>();
    if targets.is_empty() {
        BfIr::Clear
    } else {
        BfIr::Distribute { targets }
    }
}

fn normalize_flat_node(node: BfIr) -> Option<BfIr> {
    match node {
        BfIr::Add(0) | BfIr::MovePtr(0) => None,
        BfIr::Distribute { targets } => Some(normalize_distribute_targets(&targets)),
        other => Some(other),
    }
}

fn push_normalized_node(out: &mut Vec<BfIr>, node: BfIr) {
    match node {
        BfIr::Add(0) | BfIr::MovePtr(0) => {}
        BfIr::Add(delta) => match out.last_mut() {
            Some(BfIr::Add(prev)) => {
                *prev += delta;
                if *prev == 0 {
                    out.pop();
                }
            }
            Some(BfIr::Clear) if delta != 0 => {
                out.pop();
                out.push(BfIr::Add(delta));
            }
            Some(BfIr::Clear) => {}
            _ => out.push(BfIr::Add(delta)),
        },
        BfIr::MovePtr(delta) => match out.last_mut() {
            Some(BfIr::MovePtr(prev)) => {
                *prev += delta;
                if *prev == 0 {
                    out.pop();
                }
            }
            _ => out.push(BfIr::MovePtr(delta)),
        },
        BfIr::Clear => match out.last() {
            Some(BfIr::Add(_)) | Some(BfIr::Clear) => {
                out.pop();
                out.push(BfIr::Clear);
            }
            Some(BfIr::Distribute { .. }) => {}
            _ => out.push(BfIr::Clear),
        },
        BfIr::Distribute { targets } => match out.last() {
            Some(BfIr::Clear) | Some(BfIr::Distribute { .. }) => {}
            _ => out.push(BfIr::Distribute { targets }),
        },
        other => out.push(other),
    }
}

fn normalize_sequence_once(nodes: &mut Vec<BfIr>) -> bool {
    let before = nodes.clone();
    let mut merged = Vec::with_capacity(nodes.len());
    for node in nodes.drain(..) {
        if let Some(node) = normalize_flat_node(node) {
            push_normalized_node(&mut merged, node);
        }
    }
    let changed = before != merged;
    *nodes = merged;
    changed
}

fn normalize_sequence(nodes: &mut Vec<BfIr>) {
    while normalize_sequence_once(nodes) {}
}

// --- Pass 2: try_summarize ---

fn try_summarize_clear_like_loop(body: &[BfIr]) -> Option<BfIr> {
    match body {
        [BfIr::Clear] => Some(BfIr::Clear),
        [BfIr::Add(delta)] if *delta == -1 || *delta == 1 => Some(BfIr::Clear),
        _ => None,
    }
}

struct LoopSummary {
    ptr_end: isize,
    deltas: BTreeMap<isize, i32>,
}

fn summarize_simple_loop(body: &[BfIr]) -> Option<LoopSummary> {
    let mut ptr = 0isize;
    let mut deltas = BTreeMap::<isize, i32>::new();
    for op in body {
        match *op {
            BfIr::Add(n) => {
                *deltas.entry(ptr).or_insert(0) += n;
            }
            BfIr::MovePtr(n) => {
                ptr += n;
            }
            _ => return None,
        }
    }
    deltas.retain(|_, v| *v != 0);
    Some(LoopSummary {
        ptr_end: ptr,
        deltas,
    })
}

fn try_summarize_affine_transfer_loop(body: &[BfIr]) -> Option<BfIr> {
    if let [BfIr::Distribute { targets }] = body {
        return Some(normalize_distribute_targets(targets));
    }

    let summary = summarize_simple_loop(body)?;
    if summary.ptr_end != 0 {
        return None;
    }

    // This summary is intentionally conservative. We only fold loops whose
    // source cell changes by exactly +/-1 per iteration, which is the sound
    // boundary for the compiler's unsigned modular tape model.
    let src_delta = *summary.deltas.get(&0).unwrap_or(&0);

    let coeff_sign = match src_delta {
        -1 => 1i32,
        1 => -1i32,
        _ => return None,
    };

    let mut targets = Vec::new();
    for (&offset, &coeff) in &summary.deltas {
        if offset == 0 {
            continue;
        }
        let scaled = coeff * coeff_sign;
        if scaled != 0 {
            targets.push((offset, scaled));
        }
    }

    if targets.is_empty() {
        Some(BfIr::Clear)
    } else {
        Some(normalize_distribute_targets(&targets))
    }
}

fn is_guarded_diverge_like(body: &[BfIr]) -> bool {
    if body.len() != 1 {
        return false;
    }

    let mut cur = &body[0];
    loop {
        match cur {
            BfIr::Diverge => return true,
            BfIr::Loop(inner) if inner.len() == 1 => {
                cur = &inner[0];
            }
            _ => return false,
        }
    }
}

fn try_recognize_guarded_diverge_loop(body: &[BfIr]) -> Option<BfIr> {
    (body.is_empty() || is_guarded_diverge_like(body)).then_some(BfIr::Loop(vec![BfIr::Diverge]))
}

fn try_summarize_loop_body(body: &[BfIr]) -> Option<BfIr> {
    try_summarize_clear_like_loop(body).or_else(|| try_summarize_affine_transfer_loop(body))
}

// --- Pass 3: canonicalize_loop ---
// Canonicalize an already-rewritten loop body. Structure traversal happens in
// the shared bottom-up rewriter; loop policy lives here.
fn canonicalize_loop_body(mut body: Vec<BfIr>) -> BfIr {
    normalize_sequence(&mut body);

    if let Some(diverge) = try_recognize_guarded_diverge_loop(&body) {
        return diverge;
    }

    match try_summarize_loop_body(&body) {
        Some(summary) => summary,
        None => BfIr::Loop(body),
    }
}

#[cfg(test)]
pub(super) fn canonicalize_loop(body: Vec<BfIr>) -> BfIr {
    canonicalize_loop_body(rewrite_ir_bottom_up(body, &canonicalize_loop_body))
}

// --- Pass 4: lowering cleanup ---
// This pass is semantics-blind and only reduces redundant IR surface area after
// structural loop rewriting has finished.
fn normalize_loop_for_lowering(mut body: Vec<BfIr>) -> BfIr {
    normalize_sequence(&mut body);
    try_recognize_guarded_diverge_loop(&body).unwrap_or(BfIr::Loop(body))
}

fn cleanup_lowering_ir(program: Vec<BfIr>) -> Vec<BfIr> {
    rewrite_ir_bottom_up(program, &normalize_loop_for_lowering)
}

// --- Bottom-up structural rewrite ---

enum RewriteFrame {
    Seq {
        input: std::vec::IntoIter<BfIr>,
        out: Vec<BfIr>,
    },
    LoopFinalize,
}

fn rewrite_ir_bottom_up(
    program: Vec<BfIr>,
    rewrite_loop: &impl Fn(Vec<BfIr>) -> BfIr,
) -> Vec<BfIr> {
    let mut stack = vec![RewriteFrame::Seq {
        input: program.into_iter(),
        out: Vec::new(),
    }];
    let mut completed: Option<Vec<BfIr>> = None;

    while let Some(frame) = stack.last_mut() {
        match frame {
            RewriteFrame::Seq { input, out } => {
                let Some(node) = input.next() else {
                    let mut result = std::mem::take(out);
                    normalize_sequence(&mut result);
                    stack.pop();
                    completed = Some(result);
                    continue;
                };
                match node {
                    BfIr::Loop(body) => {
                        stack.push(RewriteFrame::LoopFinalize);
                        stack.push(RewriteFrame::Seq {
                            input: body.into_iter(),
                            out: Vec::new(),
                        });
                    }
                    BfIr::Add(0) | BfIr::MovePtr(0) => {}
                    other => out.push(other),
                }
            }
            RewriteFrame::LoopFinalize => {
                let body = completed.take().unwrap();
                stack.pop();
                let RewriteFrame::Seq { out, .. } = stack.last_mut().unwrap() else {
                    unreachable!();
                };
                out.push(rewrite_loop(body));
            }
        }
    }
    completed.unwrap_or_default()
}
