#[cfg(test)]
use crate::RequiredExt;
mod normalize;
mod rewrite;
mod semantics;
mod ssa;
mod summaries;

use super::ir::BfIr;
use normalize::normalize_sequence;
use rewrite::{canonicalize_loop_body, rewrite_ir_bottom_up};
use semantics::OptimizerSemantics;
pub use semantics::{CellSign, CodegenOpts, CodegenOptsError, IoMode, MAX_CELL_BITS};

pub fn optimize_with_opts(program: Vec<BfIr>, opts: CodegenOpts) -> Vec<BfIr> {
    // Invalid configurations cannot authorize algebraic rewrites. The codegen
    // boundary reports the typed validation error instead of clamping widths.
    if opts.validate().is_err() {
        return program;
    }
    let semantics = OptimizerSemantics::from_opts(opts);
    let mut program = rewrite_ir_bottom_up(program, semantics, &|body| {
        canonicalize_loop_body(body, semantics)
    });
    normalize_sequence(&mut program, semantics);
    program
}

#[cfg(test)]
pub(super) fn canonicalize_loop(body: Vec<BfIr>) -> BfIr {
    let nodes = canonicalize_loop_with_semantics(body, OptimizerSemantics::default_unsigned());
    assert_eq!(
        nodes.len(),
        1,
        "canonicalize_loop expected single node result"
    );
    nodes.into_iter().next().or_invariant("required value")
}

#[cfg(test)]
fn canonicalize_loop_with_semantics(body: Vec<BfIr>, semantics: OptimizerSemantics) -> Vec<BfIr> {
    canonicalize_loop_body(
        rewrite_ir_bottom_up(body, semantics, &|loop_body| {
            canonicalize_loop_body(loop_body, semantics)
        }),
        semantics,
    )
}
