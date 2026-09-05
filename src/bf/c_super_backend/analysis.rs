use crate::RequiredExt;
use std::collections::{BTreeMap, BTreeSet};

use super::*;

impl EmitterEngine {
    pub(super) fn exact_memo_spec(&mut self, id: NodeId) -> Option<ExactMemoSpec> {
        enum Frame {
            Visit(NodeId),
            Seq {
                id: NodeId,
                next_child: usize,
                ptr_delta: crate::bf::BfOffset,
                offsets: BTreeSet<crate::bf::BfOffset>,
            },
            Loop {
                id: NodeId,
                body: NodeId,
            },
        }

        let mut stack = vec![Frame::Visit(id)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Visit(current) => {
                    if self.exact_specs.contains_key(&current) {
                        continue;
                    }
                    match self.interner.get(current) {
                        NodeKind::Seq(children) if !children.is_empty() => {
                            let child = children[0];
                            stack.push(Frame::Seq {
                                id: current,
                                next_child: 0,
                                ptr_delta: 0,
                                offsets: BTreeSet::new(),
                            });
                            stack.push(Frame::Visit(child));
                            continue;
                        }
                        NodeKind::Loop(body) => {
                            let body = *body;
                            stack.push(Frame::Loop { id: current, body });
                            stack.push(Frame::Visit(body));
                            continue;
                        }
                        _ => {}
                    }

                    let spec = self.exact_memo_spec_from_cached_children(current);
                    self.exact_specs.insert(current, spec);
                }
                Frame::Seq {
                    id: current,
                    next_child,
                    mut ptr_delta,
                    mut offsets,
                } => {
                    let child = match self.interner.get(current) {
                        NodeKind::Seq(children) => children[next_child],
                        _ => crate::invariant_failure!("sequence frame must reference a sequence"),
                    };
                    let Some(child_spec) = self.exact_specs[&child] else {
                        self.exact_specs.insert(current, None);
                        continue;
                    };
                    let Some(window_start) = child_spec.window.start.checked_add(ptr_delta) else {
                        self.exact_specs.insert(current, None);
                        continue;
                    };
                    if let Some(window_tail) = child_spec.window.len.checked_sub(1) {
                        let Some(window_end) =
                            window_start.checked_add(crate::bf::BfOffset::from(window_tail))
                        else {
                            self.exact_specs.insert(current, None);
                            continue;
                        };
                        offsets.extend(window_start..=window_end);
                    }
                    let Some(next_ptr_delta) = ptr_delta.checked_add(child_spec.ptr_delta) else {
                        self.exact_specs.insert(current, None);
                        continue;
                    };
                    ptr_delta = next_ptr_delta;

                    let next_child = next_child + 1;
                    let next = match self.interner.get(current) {
                        NodeKind::Seq(children) => children.get(next_child).copied(),
                        _ => crate::invariant_failure!("sequence frame must reference a sequence"),
                    };
                    if let Some(child) = next {
                        stack.push(Frame::Seq {
                            id: current,
                            next_child,
                            ptr_delta,
                            offsets,
                        });
                        stack.push(Frame::Visit(child));
                    } else {
                        let spec = exact_spec_from_offsets(&offsets, ptr_delta);
                        self.exact_specs.insert(current, spec);
                    }
                }
                Frame::Loop { id: current, body } => {
                    let spec = self.exact_specs[&body].and_then(|body_spec| {
                        (body_spec.ptr_delta == 0).then_some(ExactMemoSpec {
                            window: body_spec.window,
                            ptr_delta: 0,
                        })
                    });
                    self.exact_specs.insert(current, spec);
                }
            }
        }
        self.exact_specs[&id]
    }

    fn exact_memo_spec_from_cached_children(&self, id: NodeId) -> Option<ExactMemoSpec> {
        match self.interner.get(id) {
            NodeKind::Add(_) | NodeKind::Clear => Some(ExactMemoSpec {
                window: MemoWindow { start: 0, len: 1 },
                ptr_delta: 0,
            }),
            NodeKind::ClearAt(offset) => Some(ExactMemoSpec {
                window: MemoWindow {
                    start: *offset,
                    len: 1,
                },
                ptr_delta: 0,
            }),
            NodeKind::Move(delta) => Some(ExactMemoSpec {
                window: MemoWindow { start: 0, len: 0 },
                ptr_delta: *delta,
            }),
            NodeKind::Input | NodeKind::Output | NodeKind::Scan { .. } | NodeKind::Diverge => None,
            NodeKind::Distribute { targets, .. } => {
                let mut offsets = BTreeSet::from([0]);
                offsets.extend(targets.iter().map(|(offset, _)| *offset));
                exact_spec_from_offsets(&offsets, 0)
            }
            NodeKind::Affine { src, dst, .. }
            | NodeKind::Shift { src, dst, .. }
            | NodeKind::Square { src, dst, .. } => {
                exact_spec_from_offsets(&BTreeSet::from([*src, *dst]), 0)
            }
            NodeKind::MulAdd { lhs, rhs, dst, .. } => {
                exact_spec_from_offsets(&BTreeSet::from([*lhs, *rhs, *dst]), 0)
            }
            NodeKind::Seq(children) if children.is_empty() => {
                exact_spec_from_offsets(&BTreeSet::new(), 0)
            }
            NodeKind::Seq(_) | NodeKind::Loop(_) => {
                crate::invariant_failure!("compound nodes are evaluated by traversal frames")
            }
        }
    }

    pub(super) fn transfer(&mut self, id: NodeId) -> &SymbolicTransfer {
        enum Frame {
            Visit(NodeId),
            Seq {
                id: NodeId,
                next_child: usize,
                acc: SymbolicTransfer,
            },
        }

        let mut stack = vec![Frame::Visit(id)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Visit(current) => {
                    if self.transfers.contains_key(&current) {
                        continue;
                    }
                    if let NodeKind::Seq(children) = self.interner.get(current)
                        && let Some(&child) = children.first()
                    {
                        stack.push(Frame::Seq {
                            id: current,
                            next_child: 0,
                            acc: SymbolicTransfer::identity(),
                        });
                        stack.push(Frame::Visit(child));
                        continue;
                    }
                    let mut transfer = if self.compile_work.budget.can_start_transfer() {
                        self.transfer_from_cached_children(current)
                    } else {
                        SymbolicTransfer::unknown()
                    };
                    if !self.compile_work.budget.admit(&transfer) {
                        transfer = SymbolicTransfer::unknown();
                    }
                    self.transfers.insert(current, transfer);
                }
                Frame::Seq {
                    id: current,
                    next_child,
                    acc,
                } => {
                    let child = match self.interner.get(current) {
                        NodeKind::Seq(children) => children[next_child],
                        _ => crate::invariant_failure!("sequence frame must reference a sequence"),
                    };
                    let acc = compose_transfer_owned(
                        acc,
                        &self.transfers[&child],
                        self.polynomial_semantics,
                        &mut self.compile_work,
                    );
                    if acc.unknown {
                        self.transfers.insert(current, acc);
                        continue;
                    }

                    let next_child = next_child + 1;
                    let next = match self.interner.get(current) {
                        NodeKind::Seq(children) => children.get(next_child).copied(),
                        _ => crate::invariant_failure!("sequence frame must reference a sequence"),
                    };
                    if let Some(child) = next {
                        stack.push(Frame::Seq {
                            id: current,
                            next_child,
                            acc,
                        });
                        stack.push(Frame::Visit(child));
                    } else {
                        self.transfers.insert(current, acc);
                    }
                }
            }
        }
        self.transfers.get(&id).or_invariant("required value")
    }

    fn transfer_from_cached_children(&self, id: NodeId) -> SymbolicTransfer {
        let mut transfer = match self.interner.get(id) {
            NodeKind::Add(delta) => {
                let Some(value) = SymbolicPolynomial::input(0).add_constant(i64::from(*delta))
                else {
                    return SymbolicTransfer::unknown();
                };
                polynomial_transfer(BTreeMap::from([(0, value)]))
            }
            NodeKind::Move(delta) => SymbolicTransfer {
                ptr_delta: *delta,
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
                polynomial_transfer(BTreeMap::from([(0, SymbolicPolynomial::zero())]))
            }
            NodeKind::ClearAt(offset) => {
                polynomial_transfer(BTreeMap::from([(*offset, SymbolicPolynomial::zero())]))
            }
            NodeKind::Scan { .. } => SymbolicTransfer::unknown(),
            NodeKind::Distribute {
                targets,
                preserve_src,
            } => {
                if targets.len() > 256 {
                    return SymbolicTransfer::unknown();
                }
                let mut seen = BTreeSet::new();
                if targets
                    .iter()
                    .any(|(offset, _)| *offset == 0 || !seen.insert(*offset))
                {
                    return SymbolicTransfer::unknown();
                }
                let mut effects = BTreeMap::new();
                for &(offset, coeff) in targets {
                    let current = effects
                        .remove(&offset)
                        .unwrap_or_else(|| SymbolicPolynomial::input(offset));
                    let Some(value) = current.add_scaled_input(0, i64::from(coeff)) else {
                        return SymbolicTransfer::unknown();
                    };
                    effects.insert(offset, value);
                }
                if !preserve_src {
                    effects.insert(0, SymbolicPolynomial::zero());
                }
                polynomial_transfer(effects)
            }
            NodeKind::Affine {
                src,
                dst,
                coeff,
                preserve_src,
                set_dst,
            } => {
                if src == dst {
                    return SymbolicTransfer::unknown();
                }
                let base = if *set_dst {
                    SymbolicPolynomial::zero()
                } else {
                    SymbolicPolynomial::input(*dst)
                };
                let Some(value) = base.add_scaled_input(*src, i64::from(*coeff)) else {
                    return SymbolicTransfer::unknown();
                };
                let mut effects = BTreeMap::from([(*dst, value)]);
                if !preserve_src && src != dst {
                    effects.insert(*src, SymbolicPolynomial::zero());
                }
                polynomial_transfer(effects)
            }
            NodeKind::Shift {
                src,
                dst,
                amount,
                dir: ShiftDir::Left,
                preserve_src,
                set_dst,
            } => {
                if src == dst {
                    return SymbolicTransfer::unknown();
                }
                let Some(coeff) = 1_i64.checked_shl(*amount) else {
                    return SymbolicTransfer::unknown();
                };
                let base = if *set_dst {
                    SymbolicPolynomial::zero()
                } else {
                    SymbolicPolynomial::input(*dst)
                };
                let Some(value) = base.add_scaled_input(*src, coeff) else {
                    return SymbolicTransfer::unknown();
                };
                let mut effects = BTreeMap::from([(*dst, value)]);
                if !preserve_src && src != dst {
                    effects.insert(*src, SymbolicPolynomial::zero());
                }
                polynomial_transfer(effects)
            }
            NodeKind::Square {
                src,
                dst,
                preserve_src,
                set_dst,
            } => {
                if src == dst {
                    return SymbolicTransfer::unknown();
                }
                let mut value = if *set_dst {
                    SymbolicPolynomial::zero()
                } else {
                    SymbolicPolynomial::input(*dst)
                };
                if value
                    .add_assign(&SymbolicPolynomial::product(*src, *src))
                    .is_none()
                {
                    return SymbolicTransfer::unknown();
                }
                let mut effects = BTreeMap::from([(*dst, value)]);
                if !preserve_src && src != dst {
                    effects.insert(*src, SymbolicPolynomial::zero());
                }
                polynomial_transfer(effects)
            }
            NodeKind::MulAdd {
                lhs,
                rhs,
                dst,
                preserve_lhs,
                preserve_rhs,
                set_dst,
            } => {
                if lhs == rhs || lhs == dst || rhs == dst {
                    return SymbolicTransfer::unknown();
                }
                let mut value = if *set_dst {
                    SymbolicPolynomial::zero()
                } else {
                    SymbolicPolynomial::input(*dst)
                };
                if value
                    .add_assign(&SymbolicPolynomial::product(*lhs, *rhs))
                    .is_none()
                {
                    return SymbolicTransfer::unknown();
                }
                let mut effects = BTreeMap::from([(*dst, value)]);
                if !preserve_lhs && lhs != dst {
                    effects.insert(*lhs, SymbolicPolynomial::zero());
                }
                if !preserve_rhs && rhs != dst && rhs != lhs {
                    effects.insert(*rhs, SymbolicPolynomial::zero());
                }
                polynomial_transfer(effects)
            }
            NodeKind::Shift { .. } => SymbolicTransfer::unknown(),
            NodeKind::Diverge => SymbolicTransfer {
                may_diverge: true,
                unknown: true,
                ..SymbolicTransfer::identity()
            },
            NodeKind::Seq(children) if children.is_empty() => SymbolicTransfer::identity(),
            NodeKind::Seq(_) => {
                crate::invariant_failure!("non-empty sequences are evaluated by traversal frames")
            }
            NodeKind::Loop(_) => SymbolicTransfer::unknown(),
        };
        if let Some(semantics) = self.polynomial_semantics {
            for polynomial in transfer.effects.values_mut() {
                let Some(normalized) = polynomial.normalized(&semantics) else {
                    return SymbolicTransfer::unknown();
                };
                *polynomial = normalized;
            }
            transfer.ptr_delta = normalized_c_offset(transfer.ptr_delta);
            transfer.reads = transfer
                .effects
                .values()
                .flat_map(SymbolicPolynomial::sources)
                .collect();
        }
        transfer
    }

    pub(super) fn loop_analysis(&mut self, id: NodeId) -> Option<&LoopAnalysis> {
        if self.loop_analyses.contains_key(&id) {
            return self
                .loop_analyses
                .get(&id)
                .and_then(|analysis| analysis.as_ref());
        }
        let body = match self.interner.get(id) {
            NodeKind::Loop(body) => *body,
            _ => {
                self.loop_analyses.insert(id, None);
                return None;
            }
        };
        let exact = self.exact_memo_spec(id);
        let transfer = self.transfer(body);
        let analysis = if transfer.is_direct_kernel_loop_shape() {
            if let Some(exact) = exact {
                Some(LoopAnalysis::ExactMemoPlusDirectKernel { body, exact })
            } else {
                Some(LoopAnalysis::Residual { body })
            }
        } else if let Some(exact) = exact {
            Some(LoopAnalysis::ExactMemoOnly { body, exact })
        } else {
            Some(LoopAnalysis::Residual { body })
        };
        self.loop_analyses.insert(id, analysis);
        self.loop_analyses
            .get(&id)
            .and_then(|analysis| analysis.as_ref())
    }
}

fn polynomial_transfer(
    effects: BTreeMap<crate::bf::BfOffset, SymbolicPolynomial>,
) -> SymbolicTransfer {
    let reads = effects
        .values()
        .flat_map(SymbolicPolynomial::sources)
        .collect();
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

fn exact_spec_from_offsets(
    offsets: &BTreeSet<crate::bf::BfOffset>,
    ptr_delta: crate::bf::BfOffset,
) -> Option<ExactMemoSpec> {
    if offsets.is_empty() {
        return Some(ExactMemoSpec {
            window: MemoWindow { start: 0, len: 0 },
            ptr_delta,
        });
    }
    let min = *offsets.iter().next()?;
    let max = *offsets.iter().next_back()?;
    i32::try_from(min).ok()?;
    i32::try_from(max).ok()?;
    let span = max.checked_sub(min)?.checked_add(1)?;
    if span <= 0 || span > SUPER_MEMO_WINDOW_MAX {
        return None;
    }
    Some(ExactMemoSpec {
        window: MemoWindow {
            start: min,
            len: u8::try_from(span).or_invariant("validated memo span exceeded u8"),
        },
        ptr_delta,
    })
}

pub(super) fn shift_effect(
    effect: &SymbolicPolynomial,
    delta: crate::bf::BfOffset,
) -> Option<SymbolicPolynomial> {
    effect.shifted(delta)
}

#[cfg(test)]
pub(super) fn compose_transfer(
    left: &SymbolicTransfer,
    right: &SymbolicTransfer,
) -> SymbolicTransfer {
    compose_transfer_owned(left.clone(), right, None, &mut CompileWork::default())
}

pub(super) fn compose_transfer_owned(
    mut left: SymbolicTransfer,
    right: &SymbolicTransfer,
    semantics: Option<PolynomialSemantics>,
    work: &mut CompileWork,
) -> SymbolicTransfer {
    if !work.budget.begin_composition() {
        return SymbolicTransfer::unknown();
    }
    work.compositions += 1;
    if left.unknown || right.unknown {
        return SymbolicTransfer::unknown();
    }
    if right.effects.len() > 256 || left.effects.len() > 256 {
        return SymbolicTransfer::unknown();
    }
    let mut updates = Vec::with_capacity(right.effects.len());
    let mut budget = SubstitutionBudget::new();
    for (offset, effect) in &right.effects {
        let Some(rebased) = offset.checked_add(left.ptr_delta) else {
            return SymbolicTransfer::unknown();
        };
        let rebased = if semantics.is_some() {
            normalized_c_offset(rebased)
        } else {
            rebased
        };
        let polynomial = match semantics {
            Some(ref semantics) => effect.shifted_with(left.ptr_delta, semantics),
            None => shift_effect(effect, left.ptr_delta),
        };
        let Some(polynomial) = polynomial else {
            return SymbolicTransfer::unknown();
        };
        let composed = match semantics {
            Some(ref semantics) => {
                polynomial.substitute_with(&left.effects, semantics, &mut budget)
            }
            None => polynomial.substitute(&left.effects),
        };
        let Some(composed) = composed else {
            work.record_substitution(&budget);
            return SymbolicTransfer::unknown();
        };
        updates.push((rebased, composed));
    }
    work.record_substitution(&budget);
    left.effects.extend(updates);
    if left.effects.len() > 256 || left.effects.values().map(|p| p.terms.len()).sum::<usize>() > 256
    {
        return SymbolicTransfer::unknown();
    }
    let reads = left
        .effects
        .values()
        .flat_map(SymbolicPolynomial::sources)
        .collect();
    let result = SymbolicTransfer {
        ptr_delta: match left.ptr_delta.checked_add(right.ptr_delta) {
            Some(delta) if semantics.is_some() => normalized_c_offset(delta),
            Some(delta) => delta,
            None => return SymbolicTransfer::unknown(),
        },
        effects: left.effects,
        reads,
        may_input: left.may_input || right.may_input,
        may_output: left.may_output || right.may_output,
        may_diverge: left.may_diverge || right.may_diverge,
        unknown: false,
    };
    if work.budget.admit(&result) {
        result
    } else {
        SymbolicTransfer::unknown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memo_window_rebases_the_complete_checked_inclusive_range() {
        let mut engine = EmitterEngine::new();
        let first = engine.interner.intern(NodeKind::Clear);
        let last = engine.interner.intern(NodeKind::ClearAt(7));
        let body = engine.interner.intern(NodeKind::Seq(vec![first, last]));
        let shift = engine.interner.intern(NodeKind::Move(-11));
        let root = engine.interner.intern(NodeKind::Seq(vec![shift, body]));

        assert_eq!(
            engine.exact_memo_spec(root),
            Some(ExactMemoSpec {
                window: MemoWindow { start: -11, len: 8 },
                ptr_delta: -11,
            }),
            "rebasing must include both endpoints of the eight-cell window"
        );
    }

    #[test]
    fn memo_window_overflow_rejects_before_visiting_the_next_child() {
        let mut engine = EmitterEngine::new();
        let first = engine.interner.intern(NodeKind::Clear);
        let last = engine.interner.intern(NodeKind::ClearAt(1));
        let body = engine.interner.intern(NodeKind::Seq(vec![first, last]));
        let shift = engine
            .interner
            .intern(NodeKind::Move(crate::bf::BfOffset::MAX));
        let unvisited = engine.interner.intern(NodeKind::Input);
        let root = engine
            .interner
            .intern(NodeKind::Seq(vec![shift, body, unvisited]));

        assert_eq!(engine.exact_memo_spec(root), None);
        assert!(
            !engine.exact_specs.contains_key(&unvisited),
            "overflowing MAX + 1 must reject the range, not continue analysis"
        );
    }

    #[test]
    fn empty_memo_window_at_maximum_pointer_does_not_touch_a_cell() {
        let mut engine = EmitterEngine::new();
        let shift = engine
            .interner
            .intern(NodeKind::Move(crate::bf::BfOffset::MAX));
        let stationary = engine.interner.intern(NodeKind::Move(0));
        let root = engine
            .interner
            .intern(NodeKind::Seq(vec![shift, stationary]));

        assert_eq!(
            engine.exact_memo_spec(root),
            Some(ExactMemoSpec {
                window: MemoWindow { start: 0, len: 0 },
                ptr_delta: crate::bf::BfOffset::MAX,
            }),
            "an empty window must not become an inclusive one-cell range"
        );
    }

    #[test]
    fn deep_dag_analysis_uses_iterative_postorder() {
        const DEPTH: usize = 20_000;

        let mut engine = EmitterEngine::new();
        let leaf = engine.interner.intern(NodeKind::Add(1));
        let mut root = leaf;
        for _ in 0..DEPTH {
            root = engine.interner.intern(NodeKind::Seq(vec![root]));
        }

        assert_eq!(
            engine.exact_memo_spec(root),
            Some(ExactMemoSpec {
                window: MemoWindow { start: 0, len: 1 },
                ptr_delta: 0,
            })
        );
        assert_eq!(engine.exact_specs.len(), DEPTH + 1);

        let transfer = engine.transfer(root);
        assert!(!transfer.unknown);
        assert_eq!(
            transfer.effects.get(&0),
            Some(
                &SymbolicPolynomial::input(0)
                    .add_constant(1)
                    .or_invariant("required value")
            )
        );
        assert_eq!(engine.transfers.len(), DEPTH + 1);
    }

    #[test]
    fn iterative_analysis_preserves_sequence_short_circuit_caching() {
        let mut engine = EmitterEngine::new();
        let unknown = engine.interner.intern(NodeKind::Input);
        let unvisited = engine.interner.intern(NodeKind::Add(1));
        let root = engine
            .interner
            .intern(NodeKind::Seq(vec![unknown, unvisited]));

        assert_eq!(engine.exact_memo_spec(root), None);
        assert!(engine.exact_specs.contains_key(&unknown));
        assert!(engine.exact_specs.contains_key(&root));
        assert!(!engine.exact_specs.contains_key(&unvisited));

        assert!(engine.transfer(root).unknown);
        assert!(engine.transfers.contains_key(&unknown));
        assert!(engine.transfers.contains_key(&root));
        assert!(!engine.transfers.contains_key(&unvisited));
    }

    #[test]
    fn composition_propagates_constants_and_reduces_product_degree() {
        let left = polynomial_transfer(BTreeMap::from([(0, SymbolicPolynomial::constant(5))]));
        let right = polynomial_transfer(BTreeMap::from([(2, SymbolicPolynomial::product(0, 1))]));

        let composed = compose_transfer(&left, &right);

        assert!(!composed.unknown);
        assert_eq!(
            composed.effects[&2].terms,
            BTreeMap::from([(SymbolicMonomial::Linear(1), 5)])
        );
        assert_eq!(composed.effects[&2].degree(), 1);
    }

    #[test]
    fn composition_prunes_an_overwritten_symbolic_write() {
        let left = polynomial_transfer(BTreeMap::from([(1, SymbolicPolynomial::input(0))]));
        let right = polynomial_transfer(BTreeMap::from([(1, SymbolicPolynomial::zero())]));

        let composed = compose_transfer(&left, &right);

        assert_eq!(composed.effects.len(), 1);
        assert!(composed.effects[&1].is_zero());
        assert!(composed.reads.is_empty());
    }

    #[test]
    fn composition_preserves_exact_cubic_terms() {
        let left = polynomial_transfer(BTreeMap::from([(0, SymbolicPolynomial::product(1, 2))]));
        let right = polynomial_transfer(BTreeMap::from([(3, SymbolicPolynomial::product(0, 4))]));

        let composed = compose_transfer(&left, &right);
        assert!(!composed.unknown);
        assert_eq!(composed.effects[&3].degree(), 3);
    }

    #[test]
    fn composition_rejects_pointer_offset_overflow() {
        let left = SymbolicTransfer {
            ptr_delta: crate::bf::BfOffset::MAX,
            ..SymbolicTransfer::identity()
        };
        let right = polynomial_transfer(BTreeMap::from([(1, SymbolicPolynomial::input(1))]));

        assert!(compose_transfer(&left, &right).unknown);
    }

    #[test]
    fn scan_and_output_remain_opaque_to_symbolic_composition() {
        let mut engine = EmitterEngine::new();
        let scan = engine.interner.intern(NodeKind::Scan { stride: 1 });
        let output = engine.interner.intern(NodeKind::Output);

        assert!(engine.transfer(scan).unknown);
        assert!(engine.transfer(output).unknown);
        assert_eq!(engine.exact_memo_spec(scan), None);
        assert_eq!(engine.exact_memo_spec(output), None);
    }
}
