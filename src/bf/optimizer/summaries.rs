use super::super::ir::{BfIr, ShiftDir};
use super::normalize::normalize_distribute_targets_with_preserve;
use super::semantics::OptimizerSemantics;
use super::ssa::{build_loop_ssa, solve_loop_effect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PolynomialTerm {
    Affine { coeff: i32 },
    Shift { amount: u32, dir: ShiftDir },
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PolynomialSummary {
    pub(super) src: crate::bf::BfOffset,
    pub(super) dst: crate::bf::BfOffset,
    pub(super) preserve_src: bool,
    pub(super) set_dst: bool,
    pub(super) term: PolynomialTerm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ArithmeticSummary {
    Linear {
        targets: Vec<(crate::bf::BfOffset, i32)>,
        preserve_src: bool,
    },
    Polynomial(PolynomialSummary),
}

fn lower_polynomial_summary(summary: PolynomialSummary) -> BfIr {
    match summary.term {
        PolynomialTerm::Affine { coeff } => BfIr::Affine {
            src: summary.src,
            dst: summary.dst,
            coeff,
            preserve_src: summary.preserve_src,
            set_dst: summary.set_dst,
        },
        PolynomialTerm::Shift { amount, dir } => BfIr::Shift {
            src: summary.src,
            dst: summary.dst,
            amount,
            dir,
            preserve_src: summary.preserve_src,
            set_dst: summary.set_dst,
        },
        PolynomialTerm::Square => BfIr::Square {
            src: summary.src,
            dst: summary.dst,
            preserve_src: summary.preserve_src,
            set_dst: summary.set_dst,
        },
    }
}

pub(super) fn lower_arithmetic_summary(
    summary: ArithmeticSummary,
    semantics: OptimizerSemantics,
) -> Option<BfIr> {
    match summary {
        ArithmeticSummary::Linear {
            targets,
            preserve_src,
        } => normalize_distribute_targets_with_preserve(&targets, preserve_src, semantics),
        ArithmeticSummary::Polynomial(summary) => Some(lower_polynomial_summary(summary)),
    }
}

pub(super) fn polynomial_summary_from_single_source_node(
    node: &BfIr,
    rewritten_src: crate::bf::BfOffset,
    preserve_src: bool,
) -> Option<PolynomialSummary> {
    match node {
        BfIr::Affine {
            dst,
            coeff,
            set_dst,
            ..
        } => Some(PolynomialSummary {
            src: rewritten_src,
            dst: *dst,
            preserve_src,
            set_dst: *set_dst,
            term: PolynomialTerm::Affine { coeff: *coeff },
        }),
        BfIr::Shift {
            dst,
            amount,
            dir,
            set_dst,
            ..
        } => Some(PolynomialSummary {
            src: rewritten_src,
            dst: *dst,
            preserve_src,
            set_dst: *set_dst,
            term: PolynomialTerm::Shift {
                amount: *amount,
                dir: *dir,
            },
        }),
        BfIr::Square { dst, set_dst, .. } => Some(PolynomialSummary {
            src: rewritten_src,
            dst: *dst,
            preserve_src,
            set_dst: *set_dst,
            term: PolynomialTerm::Square,
        }),
        _ => None,
    }
}

pub(super) fn nested_preserving_linear_summary(
    targets: &[(crate::bf::BfOffset, i32)],
    temp_off: crate::bf::BfOffset,
    restore_dst: crate::bf::BfOffset,
) -> Option<ArithmeticSummary> {
    if restore_dst != -temp_off {
        return None;
    }

    let mut preserved_targets = Vec::new();
    let mut temp_matches = false;
    for &(offset, coeff) in targets {
        if offset == temp_off && coeff == 1 {
            temp_matches = true;
        } else {
            preserved_targets.push((offset, coeff));
        }
    }

    temp_matches.then_some(ArithmeticSummary::Linear {
        targets: preserved_targets,
        preserve_src: true,
    })
}

// --- Pass 2: try_summarize ---

fn try_summarize_clear_like_loop(body: &[BfIr]) -> Option<Vec<BfIr>> {
    match body {
        [BfIr::Clear] => Some(vec![BfIr::Clear]),
        [BfIr::Add(delta)] if *delta == -1 || *delta == 1 => Some(vec![BfIr::Clear]),
        _ => None,
    }
}

fn try_summarize_scan_loop(body: &[BfIr]) -> Option<Vec<BfIr>> {
    match body {
        [BfIr::MovePtr(stride)] if *stride != 0 => Some(vec![BfIr::Scan { stride: *stride }]),
        _ => None,
    }
}

fn try_summarize_direct_rich_loop(
    body: &[BfIr],
    semantics: OptimizerSemantics,
) -> Option<Vec<BfIr>> {
    match body {
        [
            BfIr::Distribute {
                targets,
                preserve_src,
            },
        ] if !preserve_src
            && targets
                .iter()
                .all(|(offset, _)| !semantics.offsets_alias_on_supported_tape(0, *offset))
            && semantics.has_no_wrapped_offset_aliases(
                std::iter::once(0).chain(targets.iter().map(|(offset, _)| *offset)),
            ) =>
        {
            Some(vec![normalize_distribute_targets_with_preserve(
                targets, false, semantics,
            )?])
        }
        [
            BfIr::Affine {
                src: 0,
                dst,
                preserve_src: false,
                set_dst: false,
                ..
            },
        ] if !semantics.offsets_alias_on_supported_tape(0, *dst) => Some(vec![body[0].clone()]),
        [
            BfIr::Shift {
                src: 0,
                dst,
                preserve_src: false,
                set_dst: false,
                ..
            },
        ] if !semantics.offsets_alias_on_supported_tape(0, *dst) => Some(vec![body[0].clone()]),
        [
            BfIr::Square {
                src: 0,
                dst,
                preserve_src: false,
                set_dst: false,
                ..
            },
        ] if !semantics.offsets_alias_on_supported_tape(0, *dst) => Some(vec![body[0].clone()]),
        [
            BfIr::MulAdd {
                lhs: 0,
                rhs,
                dst,
                preserve_lhs: false,
                set_dst: false,
                ..
            },
        ] if !semantics.offsets_alias_on_supported_tape(0, *rhs)
            && !semantics.offsets_alias_on_supported_tape(0, *dst)
            && semantics.has_no_wrapped_offset_aliases([0, *rhs, *dst]) =>
        {
            Some(vec![body[0].clone()])
        }
        _ => None,
    }
}

fn try_summarize_ssa_loop(body: &[BfIr], semantics: OptimizerSemantics) -> Option<Vec<BfIr>> {
    let model = build_loop_ssa(body)?;
    let solved = solve_loop_effect(&model, semantics)?;
    debug_assert!(solved.summary.validate().is_ok());
    debug_assert!(solved.summary.lower_to_ir().is_ok());
    Some(solved.nodes)
}

pub(super) fn try_summarize_loop_body(
    body: &[BfIr],
    semantics: OptimizerSemantics,
) -> Option<Vec<BfIr>> {
    try_summarize_clear_like_loop(body)
        .or_else(|| try_summarize_scan_loop(body))
        .or_else(|| try_summarize_direct_rich_loop(body, semantics))
        .or_else(|| try_summarize_ssa_loop(body, semantics))
}
