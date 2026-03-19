use super::*;

impl EmitterEngine {
    pub(super) fn transfer(&mut self, id: NodeId) -> SymbolicTransfer {
        if let Some(transfer) = self.transfers.get(&id) {
            return transfer.clone();
        }
        let transfer = match self.interner.get(id).clone() {
            NodeKind::Add(delta) => {
                let mut effects = BTreeMap::new();
                effects.insert(0, SymbolicCellEffect::AddConst(delta as i64));
                SymbolicTransfer {
                    ptr_delta: 0,
                    effects,
                    reads: BTreeSet::from([0]),
                    may_input: false,
                    may_output: false,
                    may_diverge: false,
                    unknown: false,
                }
            }
            NodeKind::Move(delta) => SymbolicTransfer {
                ptr_delta: delta,
                effects: BTreeMap::new(),
                reads: BTreeSet::new(),
                may_input: false,
                may_output: false,
                may_diverge: false,
                unknown: false,
            },
            NodeKind::Input => SymbolicTransfer {
                may_input: true,
                unknown: true,
                ..SymbolicTransfer::identity()
            },
            NodeKind::Output => SymbolicTransfer {
                reads: BTreeSet::from([0]),
                may_output: true,
                unknown: true,
                ..SymbolicTransfer::identity()
            },
            NodeKind::Clear => {
                let mut effects = BTreeMap::new();
                effects.insert(0, SymbolicCellEffect::Clear);
                SymbolicTransfer {
                    ptr_delta: 0,
                    effects,
                    reads: BTreeSet::from([0]),
                    may_input: false,
                    may_output: false,
                    may_diverge: false,
                    unknown: false,
                }
            }
            NodeKind::Distribute(targets) => {
                let mut effects = BTreeMap::new();
                effects.insert(0, SymbolicCellEffect::Clear);
                for &(offset, coeff) in &targets {
                    effects.insert(
                        offset,
                        SymbolicCellEffect::AddScaledSource {
                            source_offset: 0,
                            coeff,
                        },
                    );
                }
                let mut reads = BTreeSet::from([0]);
                for &(offset, _) in &targets {
                    reads.insert(offset);
                }
                SymbolicTransfer {
                    ptr_delta: 0,
                    effects,
                    reads,
                    may_input: false,
                    may_output: false,
                    may_diverge: false,
                    unknown: false,
                }
            }
            NodeKind::Diverge => SymbolicTransfer {
                may_diverge: true,
                unknown: true,
                ..SymbolicTransfer::identity()
            },
            NodeKind::Seq(children) => {
                let mut acc = SymbolicTransfer::identity();
                for child in children {
                    acc = compose_transfer(&acc, &self.transfer(child));
                    if acc.unknown {
                        break;
                    }
                }
                acc
            }
            NodeKind::Loop(_) => SymbolicTransfer::unknown(),
        };
        self.transfers.insert(id, transfer.clone());
        transfer
    }

    pub(super) fn loop_analysis(&mut self, id: NodeId) -> Option<LoopAnalysis> {
        if let Some(analysis) = self.loop_analyses.get(&id) {
            return analysis.clone();
        }
        let analysis = match self.interner.get(id).clone() {
            NodeKind::Loop(body) => {
                let transfer = self.transfer(body);
                if transfer.is_direct_kernel_loop_shape() {
                    if let Some(powered) = powered_loop_analysis(body, &transfer) {
                        Some(LoopAnalysis::Powered(powered))
                    } else {
                        Some(LoopAnalysis::DirectKernel {
                            body,
                            accessed_offsets: transfer.accessed_offsets(),
                        })
                    }
                } else if let Some(window) = transfer.memo_window() {
                    Some(LoopAnalysis::Symbolic { body, window })
                } else {
                    Some(LoopAnalysis::Residual { body })
                }
            }
            _ => None,
        };
        self.loop_analyses.insert(id, analysis.clone());
        analysis
    }
}

pub(super) fn shift_effect(effect: &SymbolicCellEffect, delta: isize) -> SymbolicCellEffect {
    match effect {
        SymbolicCellEffect::AddConst(value) => SymbolicCellEffect::AddConst(*value),
        SymbolicCellEffect::Clear => SymbolicCellEffect::Clear,
        SymbolicCellEffect::AddScaledSource {
            source_offset,
            coeff,
        } => SymbolicCellEffect::AddScaledSource {
            source_offset: *source_offset + delta,
            coeff: *coeff,
        },
        SymbolicCellEffect::Opaque => SymbolicCellEffect::Opaque,
    }
}

pub(super) fn combine_effects(
    left: &SymbolicCellEffect,
    right: &SymbolicCellEffect,
) -> SymbolicCellEffect {
    match (left, right) {
        (_, SymbolicCellEffect::Opaque) | (SymbolicCellEffect::Opaque, _) => {
            SymbolicCellEffect::Opaque
        }
        (SymbolicCellEffect::AddConst(a), SymbolicCellEffect::AddConst(b)) => {
            SymbolicCellEffect::AddConst(a + b)
        }
        (_, SymbolicCellEffect::Clear) => SymbolicCellEffect::Clear,
        (SymbolicCellEffect::Clear, SymbolicCellEffect::AddConst(v)) => {
            SymbolicCellEffect::AddConst(*v)
        }
        _ => SymbolicCellEffect::Opaque,
    }
}

pub(super) fn compose_transfer(left: &SymbolicTransfer, right: &SymbolicTransfer) -> SymbolicTransfer {
    if left.unknown || right.unknown {
        return SymbolicTransfer::unknown();
    }
    let mut effects = left.effects.clone();
    for (offset, effect) in &right.effects {
        let rebased = offset + left.ptr_delta;
        let shifted = shift_effect(effect, left.ptr_delta);
        let merged = if let Some(existing) = effects.get(&rebased) {
            combine_effects(existing, &shifted)
        } else {
            shifted
        };
        effects.insert(rebased, merged);
    }
    let mut reads = left.reads.clone();
    reads.extend(right.reads.iter().map(|offset| offset + left.ptr_delta));
    SymbolicTransfer {
        ptr_delta: left.ptr_delta + right.ptr_delta,
        effects,
        reads,
        may_input: left.may_input || right.may_input,
        may_output: left.may_output || right.may_output,
        may_diverge: left.may_diverge || right.may_diverge,
        unknown: false,
    }
}

fn powered_loop_analysis(body: NodeId, transfer: &SymbolicTransfer) -> Option<PoweredLoopAnalysis> {
    let window = transfer.memo_window()?;
    if !transfer.is_direct_kernel_loop_shape() {
        return None;
    }
    let guard_delta = match transfer.effects.get(&0) {
        Some(SymbolicCellEffect::AddConst(delta)) if *delta != 0 => *delta,
        _ => return None,
    };
    if transfer.effects.values().any(|effect| {
        matches!(
            effect,
            SymbolicCellEffect::Opaque | SymbolicCellEffect::AddScaledSource { .. }
        )
    }) {
        return None;
    }
    let mut powers = Vec::with_capacity(usize::from(SUPER_LOOP_POWER_MAX) + 1);
    let mut power = transfer.clone();
    powers.push(power.clone());
    while powers.len() <= usize::from(SUPER_LOOP_POWER_MAX) {
        power = compose_transfer(&power, &power);
        if !power.is_pure_windowed() || power.has_opaque_effects() {
            break;
        }
        powers.push(power.clone());
    }
    Some(PoweredLoopAnalysis {
        body,
        window,
        guard_offset: 0,
        guard_delta,
        powers,
    })
}
