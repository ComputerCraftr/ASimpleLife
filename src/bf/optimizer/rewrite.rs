use super::super::ir::BfIr;
use super::normalize::normalize_sequence;
use super::semantics::OptimizerSemantics;
use super::summaries::try_summarize_loop_body;
use crate::RequiredExt;

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

fn try_recognize_guarded_diverge_loop(body: &[BfIr]) -> Option<Vec<BfIr>> {
    (body.is_empty() || is_guarded_diverge_like(body))
        .then_some(vec![BfIr::Loop(vec![BfIr::Diverge])])
}

// --- Pass 3: canonicalize_loop ---
// Canonicalize an already-rewritten loop body. Structure traversal happens in
// the shared bottom-up rewriter; loop policy lives here.
pub(super) fn canonicalize_loop_body(
    mut body: Vec<BfIr>,
    semantics: OptimizerSemantics,
) -> Vec<BfIr> {
    normalize_sequence(&mut body, semantics);

    if let Some(diverge) = try_recognize_guarded_diverge_loop(&body) {
        return diverge;
    }

    match try_summarize_loop_body(&body, semantics) {
        Some(summary) => summary,
        None => vec![BfIr::Loop(body)],
    }
}

// --- Bottom-up structural rewrite ---

enum RewriteFrame {
    Seq {
        input: std::vec::IntoIter<BfIr>,
        out: Vec<BfIr>,
    },
    LoopFinalize,
}

pub(super) fn rewrite_ir_bottom_up(
    program: Vec<BfIr>,
    semantics: OptimizerSemantics,
    rewrite_loop: &impl Fn(Vec<BfIr>) -> Vec<BfIr>,
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
                    normalize_sequence(&mut result, semantics);
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
                let body = completed.take().or_invariant("required value");
                stack.pop();
                let RewriteFrame::Seq { out, .. } = stack.last_mut().or_invariant("required value")
                else {
                    crate::invariant_failure!();
                };
                out.extend(rewrite_loop(body));
            }
        }
    }
    completed.unwrap_or_default()
}
