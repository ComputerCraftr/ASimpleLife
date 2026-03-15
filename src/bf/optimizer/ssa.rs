use std::collections::{BTreeMap, BTreeSet};

use super::super::ir::BfIr;
use super::super::summary::{
    LoopId, LoopSummary, OffsetOp, SummaryProvenance, normalize_offset_body,
};
use super::normalize::normalize_distribute_targets_with_preserve;
use super::semantics::OptimizerSemantics;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct MemorySlotId(pub(super) crate::bf::BfOffset);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SsaValueId(pub(super) usize);

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SsaValueKind {
    InputCell(MemorySlotId),
    AddConst {
        base: SsaValueId,
        delta: i32,
    },
    MultiplyConst {
        base: SsaValueId,
        coeff: i32,
    },
    AddScaledSource {
        base: SsaValueId,
        source: MemorySlotId,
        coeff: i32,
    },
    Square {
        source: MemorySlotId,
    },
    Product {
        lhs: MemorySlotId,
        rhs: MemorySlotId,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GuardRecurrence {
    AddConst(i32),
    Affine { mul: i32, add: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopSideEffects {
    Pure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ProductTerm {
    pub(super) rhs_slot: MemorySlotId,
    pub(super) dst_slot: MemorySlotId,
    pub(super) coeff: i32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LoopSsaModel {
    pub(super) ptr_end: crate::bf::BfOffset,
    pub(super) guard_slot: MemorySlotId,
    pub(super) guard_recurrence: GuardRecurrence,
    pub(super) slot_deltas: BTreeMap<MemorySlotId, i32>,
    pub(super) product_terms: Vec<ProductTerm>,
    pub(super) touched_slots: Vec<MemorySlotId>,
    pub(super) source_slots: BTreeSet<MemorySlotId>,
    pub(super) write_slots: BTreeSet<MemorySlotId>,
    pub(super) values: Vec<SsaValueKind>,
    pub(super) side_effects: LoopSideEffects,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SolvedLoopEffect {
    pub(super) nodes: Vec<BfIr>,
    pub(super) summary: LoopSummary,
}

pub(super) fn build_loop_ssa(body: &[BfIr]) -> Option<LoopSsaModel> {
    let normalized = normalize_offset_body(body);
    if !normalized.has_io
        && normalized.net_pointer_delta == 0
        && normalized
            .ops
            .iter()
            .all(|op| matches!(op, OffsetOp::AddAt { .. }))
    {
        let mut deltas = BTreeMap::new();
        let mut values = vec![SsaValueKind::InputCell(MemorySlotId(0))];
        for op in normalized.ops {
            let OffsetOp::AddAt { offset, delta } = op else {
                crate::invariant_failure!();
            };
            let slot = MemorySlotId(offset);
            let base = SsaValueId(values.len().saturating_sub(1));
            values.push(SsaValueKind::AddConst { base, delta });
            let entry = deltas.entry(slot).or_insert(0_i32);
            *entry = entry.checked_add(delta)?;
        }
        deltas.retain(|_, delta| *delta != 0);
        let guard_delta = *deltas.get(&MemorySlotId(0)).unwrap_or(&0);
        let touched_slots = normalized
            .touched_offsets
            .iter()
            .copied()
            .map(MemorySlotId)
            .collect::<Vec<_>>();
        let source_slots = normalized
            .read_offsets
            .into_iter()
            .map(MemorySlotId)
            .collect();
        let write_slots = normalized
            .write_offsets
            .into_iter()
            .map(MemorySlotId)
            .collect();
        return Some(LoopSsaModel {
            ptr_end: 0,
            guard_slot: MemorySlotId(0),
            guard_recurrence: GuardRecurrence::AddConst(guard_delta),
            slot_deltas: deltas,
            product_terms: Vec::new(),
            touched_slots,
            source_slots,
            write_slots,
            values,
            side_effects: LoopSideEffects::Pure,
        });
    }

    let mut ptr: crate::bf::BfOffset = 0;
    let mut index = 0usize;
    let mut deltas = BTreeMap::<MemorySlotId, i32>::new();
    let mut product_terms = Vec::<ProductTerm>::new();
    let mut touched_slots = BTreeSet::<MemorySlotId>::new();
    let mut source_slots = BTreeSet::<MemorySlotId>::new();
    let mut write_slots = BTreeSet::<MemorySlotId>::new();
    let mut values = vec![SsaValueKind::InputCell(MemorySlotId(0))];

    while let Some(op) = body.get(index) {
        match *op {
            BfIr::Add(delta) => {
                let slot = MemorySlotId(ptr);
                touched_slots.insert(slot);
                source_slots.insert(slot);
                write_slots.insert(slot);
                let base = SsaValueId(values.len().saturating_sub(1));
                values.push(SsaValueKind::AddConst { base, delta });
                let entry = deltas.entry(slot).or_insert(0_i32);
                *entry = entry.checked_add(delta)?;
                index += 1;
            }
            BfIr::MovePtr(delta) => {
                ptr = ptr.checked_add(delta)?;
                index += 1;
            }
            BfIr::Affine {
                src,
                dst,
                coeff,
                preserve_src: true,
                set_dst: false,
            } => {
                let src_slot = MemorySlotId(ptr.checked_add(src)?);
                let dst_slot = MemorySlotId(ptr.checked_add(dst)?);
                if src_slot == dst_slot
                    || src_slot == MemorySlotId(0)
                    || dst_slot == MemorySlotId(0)
                {
                    return None;
                }
                touched_slots.insert(src_slot);
                touched_slots.insert(dst_slot);
                source_slots.insert(src_slot);
                write_slots.insert(dst_slot);
                let base = SsaValueId(values.len().saturating_sub(1));
                values.push(SsaValueKind::AddScaledSource {
                    base,
                    source: src_slot,
                    coeff,
                });
                product_terms.push(ProductTerm {
                    rhs_slot: src_slot,
                    dst_slot,
                    coeff,
                });
                index += 1;
            }
            BfIr::Shift {
                src,
                dst,
                amount,
                dir: super::super::ir::ShiftDir::Left,
                preserve_src: true,
                set_dst: false,
            } => {
                let coeff = i32::checked_shl(1, amount)?;
                let src_slot = MemorySlotId(ptr.checked_add(src)?);
                let dst_slot = MemorySlotId(ptr.checked_add(dst)?);
                if src_slot == dst_slot
                    || src_slot == MemorySlotId(0)
                    || dst_slot == MemorySlotId(0)
                {
                    return None;
                }
                touched_slots.insert(src_slot);
                touched_slots.insert(dst_slot);
                source_slots.insert(src_slot);
                write_slots.insert(dst_slot);
                let base = SsaValueId(values.len().saturating_sub(1));
                values.push(SsaValueKind::AddScaledSource {
                    base,
                    source: src_slot,
                    coeff,
                });
                product_terms.push(ProductTerm {
                    rhs_slot: src_slot,
                    dst_slot,
                    coeff,
                });
                index += 1;
            }
            BfIr::Distribute {
                ref targets,
                preserve_src: true,
            } => {
                let src_slot = MemorySlotId(ptr);
                if src_slot == MemorySlotId(0) {
                    return None;
                }
                touched_slots.insert(src_slot);
                source_slots.insert(src_slot);
                for &(offset, coeff) in targets {
                    let dst_slot = MemorySlotId(ptr.checked_add(offset)?);
                    if dst_slot == src_slot || dst_slot == MemorySlotId(0) {
                        return None;
                    }
                    touched_slots.insert(dst_slot);
                    write_slots.insert(dst_slot);
                    let base = SsaValueId(values.len().saturating_sub(1));
                    values.push(SsaValueKind::AddScaledSource {
                        base,
                        source: src_slot,
                        coeff,
                    });
                    product_terms.push(ProductTerm {
                        rhs_slot: src_slot,
                        dst_slot,
                        coeff,
                    });
                }
                index += 1;
            }
            BfIr::Distribute {
                ref targets,
                preserve_src: false,
            } => {
                let Some(
                    [
                        BfIr::MovePtr(temp_off),
                        BfIr::Affine {
                            src: 0,
                            dst: restore_dst,
                            coeff: 1,
                            preserve_src: false,
                            set_dst: false,
                        },
                        BfIr::MovePtr(back),
                    ],
                ) = body.get(index + 1..index + 4)
                else {
                    return None;
                };
                if *restore_dst != -*temp_off {
                    return None;
                }
                let summary = nested_preserving_linear_terms(targets, *temp_off, *restore_dst)?;
                let src_slot = MemorySlotId(ptr);
                if src_slot == MemorySlotId(0) {
                    return None;
                }
                touched_slots.insert(src_slot);
                source_slots.insert(src_slot);
                for &(offset, coeff) in &summary {
                    let dst_slot = MemorySlotId(ptr.checked_add(offset)?);
                    if dst_slot == src_slot || dst_slot == MemorySlotId(0) {
                        return None;
                    }
                    touched_slots.insert(dst_slot);
                    write_slots.insert(dst_slot);
                    let base = SsaValueId(values.len().saturating_sub(1));
                    values.push(SsaValueKind::AddScaledSource {
                        base,
                        source: src_slot,
                        coeff,
                    });
                    product_terms.push(ProductTerm {
                        rhs_slot: src_slot,
                        dst_slot,
                        coeff,
                    });
                }
                ptr = ptr.checked_add(*temp_off)?.checked_add(*back)?;
                index += 4;
            }
            BfIr::Input | BfIr::Output | BfIr::Scan { .. } | BfIr::Diverge | BfIr::Loop(_) => {
                return None;
            }
            BfIr::Clear
            | BfIr::ClearAt { .. }
            | BfIr::Shift { .. }
            | BfIr::Affine { .. }
            | BfIr::Square { .. }
            | BfIr::MulAdd { .. } => return None,
        }
    }

    deltas.retain(|_, delta| *delta != 0);
    let guard_delta = *deltas.get(&MemorySlotId(0)).unwrap_or(&0);
    for term in &product_terms {
        if term.rhs_slot == MemorySlotId(0)
            || write_slots.contains(&term.rhs_slot)
            || deltas.contains_key(&term.rhs_slot)
        {
            return None;
        }
    }

    Some(LoopSsaModel {
        ptr_end: ptr,
        guard_slot: MemorySlotId(0),
        guard_recurrence: GuardRecurrence::AddConst(guard_delta),
        slot_deltas: deltas,
        product_terms,
        touched_slots: touched_slots.into_iter().collect(),
        source_slots,
        write_slots,
        values,
        side_effects: LoopSideEffects::Pure,
    })
}

pub(super) fn solve_loop_effect(
    model: &LoopSsaModel,
    semantics: OptimizerSemantics,
) -> Option<SolvedLoopEffect> {
    if model.ptr_end != 0 || !matches!(model.side_effects, LoopSideEffects::Pure) {
        return None;
    }
    if !model.product_terms.is_empty() && !semantics.supports_muladd() {
        return None;
    }
    if !semantics.has_no_wrapped_offset_aliases(model.touched_slots.iter().map(|slot| slot.0)) {
        return None;
    }

    let guard_delta = match model.guard_recurrence {
        GuardRecurrence::AddConst(delta) => delta,
        GuardRecurrence::Affine { .. } => return None,
    };
    if guard_delta == 0 {
        return None;
    }

    let inverse = semantics.multiplicative_inverse(-guard_delta)?;
    let mut linear_targets = Vec::new();
    for (&slot, &delta) in &model.slot_deltas {
        if slot == model.guard_slot {
            continue;
        }
        let scaled = semantics.wrap_coeff_to_i32(i128::from(delta) * inverse)?;
        if scaled != 0 {
            linear_targets.push((slot.0, scaled));
        }
    }

    let mut nodes = Vec::new();
    if !linear_targets.is_empty() {
        nodes.push(normalize_distribute_targets_with_preserve(
            &linear_targets,
            !model.product_terms.is_empty(),
            semantics,
        )?);
    }

    let product_len = model.product_terms.len();
    for (index, term) in model.product_terms.iter().enumerate() {
        let scaled = semantics.wrap_coeff_to_i32(i128::from(term.coeff) * inverse)?;
        if scaled != 1 {
            return None;
        }
        nodes.push(BfIr::MulAdd {
            lhs: model.guard_slot.0,
            rhs: term.rhs_slot.0,
            dst: term.dst_slot.0,
            preserve_lhs: index + 1 != product_len,
            preserve_rhs: true,
            set_dst: false,
        });
    }

    if nodes.is_empty() {
        nodes.push(BfIr::Clear);
    }

    let summary =
        LoopSummary::from_ir_nodes(LoopId::default(), SummaryProvenance::Static, &nodes).ok()?;
    Some(SolvedLoopEffect { nodes, summary })
}

fn nested_preserving_linear_terms(
    targets: &[(crate::bf::BfOffset, i32)],
    temp_off: crate::bf::BfOffset,
    restore_dst: crate::bf::BfOffset,
) -> Option<Vec<(crate::bf::BfOffset, i32)>> {
    if restore_dst != -temp_off {
        return None;
    }

    let mut preserved_targets = Vec::new();
    let mut temp_matches = false;
    for &(offset, coeff) in targets {
        if offset == temp_off && coeff == 1 {
            temp_matches = true;
        } else if coeff != 0 {
            preserved_targets.push((offset, coeff));
        }
    }

    temp_matches.then_some(preserved_targets)
}
