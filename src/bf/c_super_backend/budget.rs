use super::SymbolicTransfer;

/// Bounds symbolic payload/work, not the source IR or the complete compiler RSS.
/// Charges are monotone: discarded candidates do not replenish compilation fuel.
#[derive(Clone, Copy, Debug)]
pub(in crate::bf) struct SymbolicLimits {
    pub(super) compositions: usize,
    pub(super) payload_bytes: usize,
    pub(super) expression_nodes: usize,
    pub(super) evaluation_passes: usize,
}

impl Default for SymbolicLimits {
    fn default() -> Self {
        Self {
            compositions: 100_000,
            payload_bytes: 64 * 1024 * 1024,
            expression_nodes: 1_000_000,
            evaluation_passes: 100_000,
        }
    }
}

#[derive(Debug, Default)]
pub(in crate::bf) struct SymbolicBudget {
    limits: SymbolicLimits,
    compositions: usize,
    payload_bytes: usize,
    expression_nodes: usize,
    evaluation_passes: usize,
    pub(super) rejected: usize,
}

impl SymbolicBudget {
    pub(super) fn new(limits: SymbolicLimits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    pub(super) fn begin_composition(&mut self) -> bool {
        if self.compositions >= self.limits.compositions {
            self.rejected += 1;
            return false;
        }
        self.compositions += 1;
        true
    }

    pub(super) fn can_start_transfer(&self) -> bool {
        self.payload_bytes.saturating_add(128) <= self.limits.payload_bytes
            && self.compositions < self.limits.compositions
    }

    pub(super) fn admit_evaluation(&mut self, transfer: &SymbolicTransfer) -> bool {
        // A selection builds at most 2*N+2 candidate DAGs. Charge two selections
        // for costing plus emission, including cache hits and rejected choices.
        let total = transfer
            .effects
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| self.evaluation_passes.checked_add(n));
        if let Some(total) = total
            && total <= self.limits.evaluation_passes
        {
            self.evaluation_passes = total;
            true
        } else {
            self.rejected += 1;
            false
        }
    }

    pub(super) fn admit(&mut self, transfer: &SymbolicTransfer) -> bool {
        if transfer.unknown {
            return true;
        }
        // Conservative payload allowance includes map/vector metadata, capacity
        // slack, monomial factor arrays and the evaluation DAG's product nodes.
        let mut bytes = 128usize;
        let mut nodes = 0usize;
        let mut add = |count: usize, per_item: usize| -> Option<()> {
            bytes = bytes.checked_add(count.checked_mul(per_item)?)?;
            Some(())
        };
        let estimate = (|| {
            add(transfer.effects.len(), 256)?;
            add(transfer.reads.len(), 64)?;
            for polynomial in transfer.effects.values() {
                add(polynomial.terms.len(), 128)?;
                for monomial in polynomial.terms.keys() {
                    let degree = usize::from(monomial.degree());
                    add(degree, 32)?;
                    nodes = nodes.checked_add(degree.saturating_sub(1))?;
                }
            }
            Some(())
        })();
        let totals = estimate.and_then(|()| {
            Some((
                self.payload_bytes.checked_add(bytes)?,
                self.expression_nodes.checked_add(nodes)?,
            ))
        });
        if let Some((bytes, nodes)) = totals
            && bytes <= self.limits.payload_bytes
            && nodes <= self.limits.expression_nodes
        {
            self.payload_bytes = bytes;
            self.expression_nodes = nodes;
            true
        } else {
            self.rejected += 1;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::symbolic::SymbolicPolynomial;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn compilation_budget_rejects_without_refunding_or_exceeding_limits() {
        let mut budget = SymbolicBudget::new(SymbolicLimits {
            compositions: 2,
            payload_bytes: 800,
            expression_nodes: 1,
            ..SymbolicLimits::default()
        });
        let transfer = SymbolicTransfer {
            effects: BTreeMap::from([(1, SymbolicPolynomial::product(0, 0))]),
            reads: BTreeSet::from([0]),
            ..SymbolicTransfer::identity()
        };
        assert!(budget.admit(&transfer));
        let charged = budget.payload_bytes;
        assert!(!budget.admit(&transfer));
        assert_eq!(
            budget.payload_bytes, charged,
            "rejection must not overshoot or refund retained charges"
        );
        assert!(budget.begin_composition());
        assert!(budget.begin_composition());
        assert!(!budget.begin_composition());
        assert_eq!(budget.compositions, 2);
        assert_eq!(budget.expression_nodes, 1);
    }

    #[test]
    fn factoring_trials_consume_a_bounded_compilation_allowance() {
        let transfer = SymbolicTransfer {
            effects: BTreeMap::from([(1, SymbolicPolynomial::product(0, 0))]),
            ..SymbolicTransfer::identity()
        };
        let mut budget = SymbolicBudget::new(SymbolicLimits {
            evaluation_passes: 8,
            ..SymbolicLimits::default()
        });
        assert!(budget.admit_evaluation(&transfer));
        assert!(!budget.admit_evaluation(&transfer));
        assert_eq!(
            budget.evaluation_passes, 8,
            "rejected trials cannot overshoot"
        );
        assert_eq!(budget.rejected, 1);
    }

    #[test]
    fn region_analysis_does_not_reset_compilation_work_allowance() {
        use crate::bf::{BfIr, CellSign, CodegenOpts, IoMode};
        let opts = CodegenOpts {
            cell_bits: 8,
            cell_sign: CellSign::Unsigned,
            io_mode: IoMode::Number,
            input_bits: None,
            output_bits: None,
        };
        let mut work = super::super::CompileWork {
            budget: SymbolicBudget::new(SymbolicLimits {
                compositions: 2,
                ..SymbolicLimits::default()
            }),
            ..super::super::CompileWork::default()
        };
        let region = [
            BfIr::Square {
                src: 0,
                dst: 1,
                preserve_src: true,
                set_dst: true,
            },
            BfIr::Square {
                src: 1,
                dst: 2,
                preserve_src: true,
                set_dst: true,
            },
        ];
        assert!(super::super::summarize_c_region_with_work(&region, opts, &mut work).is_some());
        assert!(
            super::super::summarize_c_region_with_work(&region, opts, &mut work).is_none(),
            "a new region must not silently replenish the compilation's composition allowance"
        );
        assert_eq!(work.compositions, 2);
    }

    #[test]
    fn exhausted_budget_keeps_loop_on_nonpowered_execution_plan() {
        use crate::bf::{BfIr, CellSign, CodegenOpts, IoMode};
        let opts = CodegenOpts {
            cell_bits: 8,
            cell_sign: CellSign::Unsigned,
            io_mode: IoMode::Number,
            input_bits: None,
            output_bits: None,
        };
        let mut engine = super::super::EmitterEngine::with_symbolic_limits(
            opts,
            SymbolicLimits {
                compositions: 0,
                payload_bytes: 0,
                expression_nodes: 0,
                ..SymbolicLimits::default()
            },
        );
        let root = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-1)])]);
        assert!(
            !matches!(
                engine.plan_node(root),
                super::super::ExecPlan::ExactPoweredLoopMemo { .. }
            ),
            "resource exhaustion must not authorize a powered fast path"
        );
        assert_eq!(engine.compile_work.compositions, 0);
        assert_eq!(engine.compile_work.powers_built, 0);
    }

    #[test]
    fn cached_powers_cannot_bypass_remaining_compilation_allowance() {
        use crate::bf::BfIr;
        let mut engine = super::super::EmitterEngine::new();
        let first = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-2)])]);
        let second = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-1), BfIr::Add(-1)])]);
        engine.loop_analysis(second);
        assert!(matches!(
            engine.plan_node(first),
            super::super::ExecPlan::ExactPoweredLoopMemo { .. }
        ));
        assert!(!engine.power_cache.is_empty());
        engine.compile_work.budget = SymbolicBudget::new(SymbolicLimits {
            payload_bytes: 0,
            ..SymbolicLimits::default()
        });
        assert!(
            !matches!(
                engine.plan_node(second),
                super::super::ExecPlan::ExactPoweredLoopMemo { .. }
            ),
            "a cached table cannot authorize emission after the budget rejects it"
        );
        assert_eq!(engine.compile_work.power_cache_hits, 0);
    }
}
