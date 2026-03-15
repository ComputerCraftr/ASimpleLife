use crate::RequiredExt;
use std::collections::{BTreeMap, VecDeque};

use super::super::ir::{BfIr, ShiftDir};
use super::semantics::OptimizerSemantics;
use super::summaries::{
    ArithmeticSummary, lower_arithmetic_summary, nested_preserving_linear_summary,
    polynomial_summary_from_single_source_node,
};

fn node_cost(node: &BfIr) -> usize {
    match node {
        BfIr::Clear | BfIr::ClearAt { .. } => 0,
        BfIr::Shift { .. } => 1,
        BfIr::Affine { .. } => 2,
        BfIr::Distribute { .. } => 3,
        BfIr::Square { .. } => 4,
        BfIr::MulAdd { .. } => 5,
        _ => 6,
    }
}

fn choose_cheapest_node(mut candidates: Vec<BfIr>) -> BfIr {
    debug_assert!(!candidates.is_empty());
    let mut best = candidates.remove(0);
    for candidate in candidates {
        if node_cost(&candidate) < node_cost(&best) {
            best = candidate;
        }
    }
    best
}

fn single_target_transfer_summary(
    offset: crate::bf::BfOffset,
    coeff: i32,
    preserve_src: bool,
    semantics: OptimizerSemantics,
) -> BfIr {
    let mut candidates = vec![BfIr::Affine {
        src: 0,
        dst: offset,
        coeff,
        preserve_src,
        set_dst: false,
    }];

    if let Some(amount) = semantics.shift_amount_for_coeff(coeff) {
        candidates.push(BfIr::Shift {
            src: 0,
            dst: offset,
            amount,
            dir: ShiftDir::Left,
            preserve_src,
            set_dst: false,
        });
    }

    choose_cheapest_node(candidates)
}

pub(super) fn normalize_distribute_targets_with_preserve(
    targets: &[(crate::bf::BfOffset, i32)],
    preserve_src: bool,
    semantics: OptimizerSemantics,
) -> Option<BfIr> {
    let mut merged = BTreeMap::<crate::bf::BfOffset, i32>::new();
    for &(offset, coeff) in targets {
        if offset == 0 || coeff == 0 {
            continue;
        }
        let entry = merged.entry(offset).or_insert(0);
        let Some(combined) = entry.checked_add(coeff) else {
            return Some(BfIr::Distribute {
                targets: targets.to_vec(),
                preserve_src,
            });
        };
        *entry = combined;
    }
    let targets = merged
        .into_iter()
        .filter_map(|(offset, coeff)| (coeff != 0).then_some((offset, coeff)))
        .collect::<Vec<_>>();
    match targets.as_slice() {
        [] => {
            if preserve_src {
                None
            } else {
                Some(BfIr::Clear)
            }
        }
        &[(offset, coeff)] => Some(single_target_transfer_summary(
            offset,
            coeff,
            preserve_src,
            semantics,
        )),
        _ => Some(BfIr::Distribute {
            targets,
            preserve_src,
        }),
    }
}

fn normalize_flat_node_with_change(
    node: BfIr,
    semantics: OptimizerSemantics,
) -> (Option<BfIr>, bool) {
    fn normalize_in_place_affine(
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        coeff: i32,
        preserve_src: bool,
        set_dst: bool,
        semantics: OptimizerSemantics,
    ) -> BfIr {
        if src == dst
            && preserve_src
            && set_dst
            && let Some(amount) = semantics.shift_amount_for_coeff(coeff)
            && amount != 0
        {
            return choose_cheapest_node(vec![
                BfIr::Affine {
                    src,
                    dst,
                    coeff,
                    preserve_src: true,
                    set_dst: true,
                },
                BfIr::Shift {
                    src,
                    dst,
                    amount,
                    dir: ShiftDir::Left,
                    preserve_src: true,
                    set_dst: true,
                },
            ]);
        }

        if src == dst
            && preserve_src
            && !set_dst
            && let Some(scale_coeff) = semantics.in_place_scale_coeff(coeff)
            && let Some(amount) = semantics.shift_amount_for_coeff(scale_coeff)
        {
            return choose_cheapest_node(vec![
                BfIr::Affine {
                    src,
                    dst,
                    coeff,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::Shift {
                    src,
                    dst,
                    amount,
                    dir: ShiftDir::Left,
                    preserve_src: true,
                    set_dst: true,
                },
            ]);
        }

        BfIr::Affine {
            src,
            dst,
            coeff,
            preserve_src: preserve_src || src == dst,
            set_dst,
        }
    }

    match node {
        BfIr::Add(0) | BfIr::MovePtr(0) => (None, true),
        BfIr::Distribute {
            targets,
            preserve_src,
        } => {
            let normalized =
                normalize_distribute_targets_with_preserve(&targets, preserve_src, semantics);
            let changed = !matches!(
                &normalized,
                Some(BfIr::Distribute {
                    targets: normalized_targets,
                    preserve_src: normalized_preserve_src,
                }) if *normalized_targets == targets && *normalized_preserve_src == preserve_src
            );
            (normalized, changed)
        }
        BfIr::Affine {
            src,
            dst,
            coeff,
            preserve_src,
            set_dst,
        } => {
            let normalized = if coeff == 0 && set_dst && src == dst {
                Some(if dst == 0 {
                    BfIr::Clear
                } else {
                    BfIr::ClearAt { offset: dst }
                })
            } else if coeff == 0 && !set_dst {
                if preserve_src || src != dst {
                    None
                } else {
                    Some(if dst == 0 {
                        BfIr::Clear
                    } else {
                        BfIr::ClearAt { offset: dst }
                    })
                }
            } else {
                Some(normalize_in_place_affine(
                    src,
                    dst,
                    coeff,
                    preserve_src,
                    set_dst,
                    semantics,
                ))
            };
            let changed = !matches!(
                &normalized,
                Some(BfIr::Affine {
                    src: normalized_src,
                    dst: normalized_dst,
                    coeff: normalized_coeff,
                    preserve_src: normalized_preserve_src,
                    set_dst: normalized_set_dst,
                }) if *normalized_src == src
                    && *normalized_dst == dst
                    && *normalized_coeff == coeff
                    && *normalized_preserve_src == preserve_src
                    && *normalized_set_dst == set_dst
            );
            (normalized, changed)
        }
        BfIr::Shift {
            src,
            dst,
            amount,
            dir,
            preserve_src,
            set_dst,
        } => {
            let normalized = Some(if amount == 0 {
                BfIr::Affine {
                    src,
                    dst,
                    coeff: 1,
                    preserve_src: preserve_src || src == dst,
                    set_dst,
                }
            } else {
                BfIr::Shift {
                    src,
                    dst,
                    amount,
                    dir,
                    preserve_src: preserve_src || src == dst,
                    set_dst,
                }
            });
            let changed = !matches!(
                &normalized,
                Some(BfIr::Shift {
                    src: normalized_src,
                    dst: normalized_dst,
                    amount: normalized_amount,
                    dir: normalized_dir,
                    preserve_src: normalized_preserve_src,
                    set_dst: normalized_set_dst,
                }) if *normalized_src == src
                    && *normalized_dst == dst
                    && *normalized_amount == amount
                    && *normalized_dir == dir
                    && *normalized_preserve_src == preserve_src
                    && *normalized_set_dst == set_dst
            );
            (normalized, changed)
        }
        BfIr::Square {
            src,
            dst,
            preserve_src,
            set_dst,
        } => {
            let normalized = Some(BfIr::Square {
                src,
                dst,
                preserve_src: preserve_src || src == dst,
                set_dst,
            });
            let changed = !matches!(
                &normalized,
                Some(BfIr::Square {
                    src: normalized_src,
                    dst: normalized_dst,
                    preserve_src: normalized_preserve_src,
                    set_dst: normalized_set_dst,
                }) if *normalized_src == src
                    && *normalized_dst == dst
                    && *normalized_preserve_src == preserve_src
                    && *normalized_set_dst == set_dst
            );
            (normalized, changed)
        }
        BfIr::MulAdd {
            lhs,
            rhs,
            dst,
            preserve_lhs,
            preserve_rhs,
            set_dst,
        } => {
            if lhs == rhs {
                (
                    Some(BfIr::Square {
                        src: lhs,
                        dst,
                        preserve_src: preserve_lhs && preserve_rhs,
                        set_dst,
                    }),
                    true,
                )
            } else {
                let normalized = Some(BfIr::MulAdd {
                    lhs,
                    rhs,
                    dst,
                    preserve_lhs: preserve_lhs || lhs == dst,
                    preserve_rhs: preserve_rhs || rhs == dst,
                    set_dst,
                });
                let changed = !matches!(
                    &normalized,
                    Some(BfIr::MulAdd {
                        lhs: normalized_lhs,
                        rhs: normalized_rhs,
                        dst: normalized_dst,
                        preserve_lhs: normalized_preserve_lhs,
                        preserve_rhs: normalized_preserve_rhs,
                        set_dst: normalized_set_dst,
                    }) if *normalized_lhs == lhs
                        && *normalized_rhs == rhs
                        && *normalized_dst == dst
                        && *normalized_preserve_lhs == preserve_lhs
                        && *normalized_preserve_rhs == preserve_rhs
                        && *normalized_set_dst == set_dst
                );
                (normalized, changed)
            }
        }
        other => (Some(other), false),
    }
}

fn normalize_flat_node(node: BfIr, semantics: OptimizerSemantics) -> Option<BfIr> {
    normalize_flat_node_with_change(node, semantics).0
}

fn clears_current_cell(node: &BfIr) -> bool {
    match node {
        BfIr::Clear => true,
        BfIr::ClearAt { .. } => false,
        BfIr::Distribute { preserve_src, .. } => !preserve_src,
        BfIr::Affine {
            src, preserve_src, ..
        }
        | BfIr::Shift {
            src, preserve_src, ..
        }
        | BfIr::Square {
            src, preserve_src, ..
        } => *src == 0 && !preserve_src,
        BfIr::MulAdd {
            lhs,
            rhs,
            preserve_lhs,
            preserve_rhs,
            ..
        } => (*lhs == 0 && !preserve_lhs) || (*rhs == 0 && !preserve_rhs),
        _ => false,
    }
}

fn noop_if_current_zero(node: &BfIr) -> bool {
    match node {
        BfIr::Distribute { .. } => true,
        BfIr::Affine { src, set_dst, .. }
        | BfIr::Shift { src, set_dst, .. }
        | BfIr::Square { src, set_dst, .. } => *src == 0 && !set_dst,
        BfIr::MulAdd {
            lhs, rhs, set_dst, ..
        } => !set_dst && (*lhs == 0 || *rhs == 0),
        _ => false,
    }
}

fn push_normalized_node(out: &mut Vec<BfIr>, node: BfIr) -> bool {
    if out.last().is_some_and(clears_current_cell) && noop_if_current_zero(&node) {
        return false;
    }

    match node {
        BfIr::Add(0) | BfIr::MovePtr(0) => false,
        BfIr::Add(delta) => match out.last_mut() {
            Some(BfIr::Add(prev)) => {
                if let Some(combined) = prev.checked_add(delta) {
                    *prev = combined;
                    if *prev == 0 {
                        out.pop();
                    }
                    true
                } else {
                    out.push(BfIr::Add(delta));
                    false
                }
            }
            Some(BfIr::Clear) => {
                out.push(BfIr::Add(delta));
                false
            }
            _ => {
                out.push(BfIr::Add(delta));
                false
            }
        },
        BfIr::MovePtr(delta) => match out.last_mut() {
            Some(BfIr::MovePtr(prev)) => {
                if let Some(combined) = prev.checked_add(delta) {
                    *prev = combined;
                    if *prev == 0 {
                        out.pop();
                    }
                    true
                } else {
                    out.push(BfIr::MovePtr(delta));
                    false
                }
            }
            _ => {
                out.push(BfIr::MovePtr(delta));
                false
            }
        },
        BfIr::Clear => match out.last() {
            Some(BfIr::Add(_)) | Some(BfIr::Clear) => {
                out.pop();
                out.push(BfIr::Clear);
                true
            }
            Some(BfIr::Distribute { .. })
            | Some(BfIr::Affine { .. })
            | Some(BfIr::Shift { .. })
            | Some(BfIr::Square { .. })
            | Some(BfIr::MulAdd { .. }) => false,
            _ => {
                out.push(BfIr::Clear);
                false
            }
        },
        BfIr::ClearAt { offset } => {
            if matches!(out.last(), Some(BfIr::ClearAt { offset: prior }) if *prior == offset) {
                false
            } else {
                out.push(BfIr::ClearAt { offset });
                false
            }
        }
        BfIr::Distribute {
            targets,
            preserve_src,
        } => match out.last() {
            Some(BfIr::Clear) | Some(BfIr::Distribute { .. }) | Some(BfIr::Affine { .. }) => false,
            _ => {
                out.push(BfIr::Distribute {
                    targets,
                    preserve_src,
                });
                false
            }
        },
        BfIr::Affine { .. } | BfIr::Shift { .. } | BfIr::Square { .. } | BfIr::MulAdd { .. } => {
            out.push(node);
            false
        }
        other => {
            out.push(other);
            false
        }
    }
}

fn normalize_sequence_once(nodes: &mut Vec<BfIr>, semantics: OptimizerSemantics) -> bool {
    let mut merged = Vec::with_capacity(nodes.len());
    let mut changed = false;
    for node in nodes.drain(..) {
        let (node, node_changed) = normalize_flat_node_with_change(node, semantics);
        changed |= node_changed;
        if let Some(node) = node {
            changed |= push_normalized_node(&mut merged, node);
        }
    }
    let (merged, pattern_changed) = reduce_sequence_patterns(merged, semantics);
    changed |= pattern_changed;
    *nodes = merged;
    changed
}

pub(super) fn normalize_sequence(nodes: &mut Vec<BfIr>, semantics: OptimizerSemantics) {
    let original = nodes.clone();
    let pass_budget = nodes.len().saturating_mul(2).max(8);
    for _ in 0..pass_budget {
        if !normalize_sequence_once(nodes, semantics) {
            return;
        }
    }
    *nodes = original;
}

fn reduce_sequence_patterns(nodes: Vec<BfIr>, semantics: OptimizerSemantics) -> (Vec<BfIr>, bool) {
    fn preserving_unit_copy(node: &BfIr) -> Option<(crate::bf::BfOffset, crate::bf::BfOffset)> {
        match node {
            BfIr::Affine {
                src,
                dst,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            } => Some((*src, *dst)),
            _ => None,
        }
    }

    fn arithmetic_offsets(node: &BfIr) -> Option<Vec<crate::bf::BfOffset>> {
        match node {
            BfIr::Affine { src, dst, .. } | BfIr::Shift { src, dst, .. } => Some(vec![*src, *dst]),
            BfIr::Square { src, dst, .. } => Some(vec![*src, *dst]),
            BfIr::MulAdd { lhs, rhs, dst, .. } => Some(vec![*lhs, *rhs, *dst]),
            _ => None,
        }
    }

    fn temporary_roles_are_distinct(
        semantics: OptimizerSemantics,
        src: crate::bf::BfOffset,
        temp: crate::bf::BfOffset,
        operation: &BfIr,
    ) -> bool {
        let Some(mut offsets) = arithmetic_offsets(operation) else {
            return false;
        };
        offsets.extend([src, temp]);
        semantics.has_no_wrapped_offset_aliases(offsets)
    }

    fn rewrite_muladd_through_preserved_copy(
        node: &BfIr,
        src: crate::bf::BfOffset,
        temp: crate::bf::BfOffset,
        semantics: OptimizerSemantics,
    ) -> Option<BfIr> {
        let BfIr::MulAdd {
            lhs,
            rhs,
            dst,
            preserve_lhs,
            preserve_rhs,
            set_dst,
        } = *node
        else {
            return None;
        };

        let lhs_uses_temp = lhs == temp;
        let rhs_uses_temp = rhs == temp;
        if !lhs_uses_temp && !rhs_uses_temp {
            return None;
        }
        if !semantics.has_no_wrapped_offset_aliases([src, temp, lhs, rhs, dst]) {
            return None;
        }

        normalize_flat_node(
            BfIr::MulAdd {
                lhs: if lhs_uses_temp { src } else { lhs },
                rhs: if rhs_uses_temp { src } else { rhs },
                dst,
                preserve_lhs: if lhs_uses_temp { true } else { preserve_lhs },
                preserve_rhs: if rhs_uses_temp { true } else { preserve_rhs },
                set_dst,
            },
            semantics,
        )
    }

    fn try_rewrite_temp_clear_window(
        out: &mut Vec<BfIr>,
        window: &[BfIr],
        semantics: OptimizerSemantics,
    ) -> bool {
        if let (
            Some((src, temp)),
            op,
            BfIr::MovePtr(move_to_temp),
            BfIr::Clear,
            BfIr::MovePtr(move_back),
        ) = (
            preserving_unit_copy(&window[0]),
            &window[1],
            &window[2],
            &window[3],
            &window[4],
        ) && temp == *move_to_temp
            && *move_back == -*move_to_temp
        {
            if !temporary_roles_are_distinct(semantics, src, temp, op) {
                return false;
            }
            return match op {
                BfIr::Affine {
                    src: op_src,
                    preserve_src: false,
                    ..
                } if *op_src == src => {
                    if let Some(summary) =
                        polynomial_summary_from_single_source_node(op, src, false)
                    {
                        out.push(
                            lower_arithmetic_summary(
                                ArithmeticSummary::Polynomial(summary),
                                semantics,
                            )
                            .or_invariant("required value"),
                        );
                        true
                    } else {
                        false
                    }
                }
                BfIr::Affine {
                    src: op_src,
                    preserve_src: false,
                    ..
                } if *op_src == temp => {
                    if let Some(summary) = polynomial_summary_from_single_source_node(op, src, true)
                    {
                        out.push(
                            lower_arithmetic_summary(
                                ArithmeticSummary::Polynomial(summary),
                                semantics,
                            )
                            .or_invariant("required value"),
                        );
                        true
                    } else {
                        false
                    }
                }
                BfIr::Shift {
                    src: op_src,
                    preserve_src: false,
                    ..
                } if *op_src == src => {
                    if let Some(summary) =
                        polynomial_summary_from_single_source_node(op, src, false)
                    {
                        out.push(
                            lower_arithmetic_summary(
                                ArithmeticSummary::Polynomial(summary),
                                semantics,
                            )
                            .or_invariant("required value"),
                        );
                        true
                    } else {
                        false
                    }
                }
                BfIr::Shift {
                    src: op_src,
                    preserve_src: false,
                    ..
                } if *op_src == temp => {
                    if let Some(summary) = polynomial_summary_from_single_source_node(op, src, true)
                    {
                        out.push(
                            lower_arithmetic_summary(
                                ArithmeticSummary::Polynomial(summary),
                                semantics,
                            )
                            .or_invariant("required value"),
                        );
                        true
                    } else {
                        false
                    }
                }
                BfIr::Square {
                    src: op_src,
                    preserve_src: false,
                    ..
                } if *op_src == src => {
                    if let Some(summary) =
                        polynomial_summary_from_single_source_node(op, src, false)
                    {
                        out.push(
                            lower_arithmetic_summary(
                                ArithmeticSummary::Polynomial(summary),
                                semantics,
                            )
                            .or_invariant("required value"),
                        );
                        true
                    } else {
                        false
                    }
                }
                BfIr::Square {
                    src: op_src,
                    preserve_src: false,
                    ..
                } if *op_src == temp => {
                    if let Some(summary) = polynomial_summary_from_single_source_node(op, src, true)
                    {
                        out.push(
                            lower_arithmetic_summary(
                                ArithmeticSummary::Polynomial(summary),
                                semantics,
                            )
                            .or_invariant("required value"),
                        );
                        true
                    } else {
                        false
                    }
                }
                BfIr::MulAdd { .. } => {
                    rewrite_muladd_through_preserved_copy(op, src, temp, semantics)
                        .map(|rewritten| {
                            out.push(rewritten);
                        })
                        .is_some()
                }
                _ => false,
            };
        }
        false
    }

    fn try_combine_in_place_scaling_pair(
        pair: &[BfIr],
        semantics: OptimizerSemantics,
    ) -> Option<BfIr> {
        match (&pair[0], &pair[1]) {
            (
                BfIr::Affine {
                    src: left_src,
                    dst: left_dst,
                    coeff: left_coeff,
                    preserve_src: true,
                    set_dst: true,
                },
                BfIr::Affine {
                    src: right_src,
                    dst: right_dst,
                    coeff: right_coeff,
                    preserve_src: true,
                    set_dst: true,
                },
            ) if *left_src == *left_dst && *right_src == *right_dst && *left_src == *right_src => {
                semantics
                    .multiply_coefficients(*left_coeff, *right_coeff)
                    .and_then(|coeff| {
                        normalize_flat_node(
                            BfIr::Affine {
                                src: *left_src,
                                dst: *left_dst,
                                coeff,
                                preserve_src: true,
                                set_dst: true,
                            },
                            semantics,
                        )
                    })
            }
            (
                BfIr::Shift {
                    src: left_src,
                    dst: left_dst,
                    amount: left_amount,
                    dir: left_dir,
                    preserve_src: true,
                    set_dst: true,
                },
                BfIr::Shift {
                    src: right_src,
                    dst: right_dst,
                    amount: right_amount,
                    dir: right_dir,
                    preserve_src: true,
                    set_dst: true,
                },
            ) if *left_src == *left_dst
                && *right_src == *right_dst
                && *left_src == *right_src
                && *left_dir == *right_dir =>
            {
                left_amount
                    .checked_add(*right_amount)
                    .map(|amount| BfIr::Shift {
                        src: *left_src,
                        dst: *left_dst,
                        amount,
                        dir: *left_dir,
                        preserve_src: true,
                        set_dst: true,
                    })
            }
            _ => None,
        }
    }

    fn try_rewrite_preserved_copy_pair(
        out: &mut Vec<BfIr>,
        pair: &[BfIr],
        semantics: OptimizerSemantics,
    ) -> bool {
        if let (Some((src, temp)), op) = (preserving_unit_copy(&pair[0]), &pair[1]) {
            if !temporary_roles_are_distinct(semantics, src, temp, op) {
                return false;
            }
            return match op {
                BfIr::Affine {
                    src: op_src,
                    dst,
                    coeff,
                    preserve_src: false,
                    set_dst,
                } if *op_src == temp => {
                    out.push(BfIr::Affine {
                        src,
                        dst: *dst,
                        coeff: *coeff,
                        preserve_src: true,
                        set_dst: *set_dst,
                    });
                    true
                }
                BfIr::Shift {
                    src: op_src,
                    dst,
                    amount,
                    dir,
                    preserve_src: false,
                    set_dst,
                } if *op_src == temp => {
                    out.push(BfIr::Shift {
                        src,
                        dst: *dst,
                        amount: *amount,
                        dir: *dir,
                        preserve_src: true,
                        set_dst: *set_dst,
                    });
                    true
                }
                BfIr::Square {
                    src: op_src,
                    dst,
                    preserve_src: false,
                    set_dst,
                } if *op_src == temp => {
                    out.push(BfIr::Square {
                        src,
                        dst: *dst,
                        preserve_src: true,
                        set_dst: *set_dst,
                    });
                    true
                }
                _ => false,
            };
        }
        false
    }

    fn preserved_copy_node(src: crate::bf::BfOffset, temp: crate::bf::BfOffset) -> BfIr {
        BfIr::Affine {
            src,
            dst: temp,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        }
    }

    fn try_rewrite_preserved_copy_muladd_pair(
        out: &mut Vec<BfIr>,
        pair: &[BfIr],
        semantics: OptimizerSemantics,
    ) -> bool {
        if let (Some((src, temp)), BfIr::MulAdd { .. }) = (preserving_unit_copy(&pair[0]), &pair[1])
            && let Some(rewritten) =
                rewrite_muladd_through_preserved_copy(&pair[1], src, temp, semantics)
        {
            out.push(preserved_copy_node(src, temp));
            out.push(rewritten);
            return true;
        }
        false
    }

    let mut out = Vec::with_capacity(nodes.len());
    let mut pending: VecDeque<BfIr> = nodes.into();
    let mut changed = false;
    while !pending.is_empty() {
        let window = pending.make_contiguous();
        if window.len() >= 5 && try_rewrite_temp_clear_window(&mut out, &window[..5], semantics) {
            pending.drain(..5);
            changed = true;
            continue;
        }

        if window.len() >= 2
            && let Some(combined) = try_combine_in_place_scaling_pair(&window[..2], semantics)
        {
            out.push(combined);
            pending.drain(..2);
            changed = true;
            continue;
        }

        if window.len() >= 2 && try_rewrite_preserved_copy_pair(&mut out, &window[..2], semantics) {
            pending.drain(..2);
            changed = true;
            continue;
        }

        if window.len() >= 2
            && try_rewrite_preserved_copy_muladd_pair(&mut out, &window[..2], semantics)
        {
            pending.drain(..2);
            changed = true;
            continue;
        }

        if window.len() >= 4
            && let (
                BfIr::Distribute {
                    targets,
                    preserve_src: false,
                },
                BfIr::MovePtr(temp_off),
                BfIr::Affine {
                    src: 0,
                    dst: restore_dst,
                    coeff: 1,
                    preserve_src: false,
                    set_dst: false,
                },
                BfIr::MovePtr(back),
            ) = (&window[0], &window[1], &window[2], &window[3])
        {
            let distinct_roles = semantics.has_no_wrapped_offset_aliases(
                std::iter::once(0)
                    .chain(std::iter::once(*temp_off))
                    .chain(targets.iter().map(|(offset, _)| *offset)),
            );
            if distinct_roles
                && *back == -*temp_off
                && *restore_dst == -*temp_off
                && let Some(summary) =
                    nested_preserving_linear_summary(targets, *temp_off, *restore_dst)
                        .and_then(|summary| lower_arithmetic_summary(summary, semantics))
            {
                out.push(summary);
                pending.drain(..4);
                changed = true;
                continue;
            }
        }
        out.push(pending.pop_front().or_invariant("required value"));
    }
    (out, changed)
}
