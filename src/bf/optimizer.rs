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

pub(super) enum CanonResult {
    Summary(BfIr),
    Loop(Vec<BfIr>),
}

enum CanonFrame {
    Seq {
        input: std::vec::IntoIter<BfIr>,
        out: Vec<BfIr>,
    },
    LoopFinalize,
}

pub fn optimize(program: Vec<BfIr>) -> Vec<BfIr> {
    optimize_sequence(program)
}

// --- Pass 1: normalize_sequence ---

fn merge_adjacent(nodes: &mut Vec<BfIr>) {
    let mut merged = Vec::with_capacity(nodes.len());
    for node in nodes.drain(..) {
        match (merged.last_mut(), node) {
            (Some(BfIr::Add(a)), BfIr::Add(b)) => {
                *a += b;
                if *a == 0 {
                    merged.pop();
                }
            }
            (Some(BfIr::MovePtr(a)), BfIr::MovePtr(b)) => {
                *a += b;
                if *a == 0 {
                    merged.pop();
                }
            }
            (_, BfIr::Add(0)) | (_, BfIr::MovePtr(0)) => {}
            (_, n) => merged.push(n),
        }
    }
    *nodes = merged;
}

// --- Pass 2: try_summarize ---

fn summarize_loop(body: &[BfIr]) -> Option<BfIr> {
    match body {
        [BfIr::Clear] => Some(BfIr::Clear),
        [BfIr::Distribute { targets }] => Some(BfIr::Distribute {
            targets: targets.clone(),
        }),
        [BfIr::Add(delta)] if *delta == -1 || *delta == 1 => Some(BfIr::Clear),
        _ => recognize_affine_loop(body),
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

fn recognize_affine_loop(body: &[BfIr]) -> Option<BfIr> {
    let summary = summarize_simple_loop(body)?;
    if summary.ptr_end != 0 {
        return None;
    }

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
        Some(BfIr::Distribute { targets })
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

// --- Pass 3: canonicalize_loop ---
// Canonicalize nested loops bottom-up, then summarize the current loop body if
// the outer loop itself matches a sound summary rule.

pub(super) fn canonicalize_loop(body: Vec<BfIr>) -> CanonResult {
    let mut stack = vec![CanonFrame::Seq {
        input: body.into_iter(),
        out: Vec::new(),
    }];
    let mut completed: Option<Vec<BfIr>> = None;

    while let Some(frame) = stack.last_mut() {
        match frame {
            CanonFrame::Seq { input, out } => {
                let Some(node) = input.next() else {
                    let mut result = std::mem::take(out);
                    merge_adjacent(&mut result);
                    stack.pop();
                    completed = Some(result);
                    continue;
                };

                match node {
                    BfIr::Loop(inner) => {
                        stack.push(CanonFrame::LoopFinalize);
                        stack.push(CanonFrame::Seq {
                            input: inner.into_iter(),
                            out: Vec::new(),
                        });
                    }
                    BfIr::Add(0) | BfIr::MovePtr(0) => {}
                    other => out.push(other),
                }
            }
            CanonFrame::LoopFinalize => {
                let inner = completed
                    .take()
                    .expect("canonicalized child loop body must exist");
                stack.pop();
                let CanonFrame::Seq { out, .. } = stack
                    .last_mut()
                    .expect("child loop must have a parent sequence")
                else {
                    unreachable!();
                };

                if inner.is_empty() || is_guarded_diverge_like(&inner) {
                    out.push(BfIr::Loop(vec![BfIr::Diverge]));
                } else {
                    match summarize_loop(&inner) {
                        Some(summary) => out.push(summary),
                        None => out.push(BfIr::Loop(inner)),
                    }
                }
            }
        }
    }

    let mut current = completed.unwrap_or_default();
    merge_adjacent(&mut current);

    if current.is_empty() || is_guarded_diverge_like(&current) {
        CanonResult::Loop(vec![BfIr::Diverge])
    } else {
        match summarize_loop(&current) {
            Some(summary) => CanonResult::Summary(summary),
            None => CanonResult::Loop(current),
        }
    }
}

// --- Stack orchestration ---

enum OptimizeFrame {
    Seq {
        input: std::vec::IntoIter<BfIr>,
        out: Vec<BfIr>,
    },
    LoopFinalize,
}

fn optimize_sequence(program: Vec<BfIr>) -> Vec<BfIr> {
    let mut stack = vec![OptimizeFrame::Seq {
        input: program.into_iter(),
        out: Vec::new(),
    }];
    let mut completed: Option<Vec<BfIr>> = None;

    while let Some(frame) = stack.last_mut() {
        match frame {
            OptimizeFrame::Seq { input, out } => {
                let Some(node) = input.next() else {
                    let mut result = std::mem::take(out);
                    merge_adjacent(&mut result);
                    stack.pop();
                    completed = Some(result);
                    continue;
                };
                match node {
                    BfIr::Loop(body) => {
                        stack.push(OptimizeFrame::LoopFinalize);
                        stack.push(OptimizeFrame::Seq {
                            input: body.into_iter(),
                            out: Vec::new(),
                        });
                    }
                    BfIr::Add(0) | BfIr::MovePtr(0) => {}
                    other => out.push(other),
                }
            }
            OptimizeFrame::LoopFinalize => {
                let body = completed.take().unwrap();
                stack.pop();
                let OptimizeFrame::Seq { out, .. } = stack.last_mut().unwrap() else {
                    unreachable!();
                };
                match canonicalize_loop(body) {
                    CanonResult::Summary(s) => out.push(s),
                    CanonResult::Loop(b) => out.push(BfIr::Loop(b)),
                }
            }
        }
    }
    completed.unwrap_or_default()
}
