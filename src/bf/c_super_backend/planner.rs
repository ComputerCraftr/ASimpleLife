use super::*;
use crate::RequiredExt;

impl EmitterEngine {
    pub(super) fn plan_node(&mut self, id: NodeId) -> ExecPlan {
        self.plan_decision(id).plan
    }

    pub(super) fn plan_cost(&mut self, id: NodeId) -> usize {
        self.plan_decision(id).estimated_cost
    }

    pub(super) fn plan_decision(&mut self, id: NodeId) -> PlanDecision {
        if let Some(&decision) = self.plan_decisions.get(&id) {
            return decision;
        }
        let decision = match self.interner.get(id) {
            NodeKind::Input | NodeKind::Output | NodeKind::Scan { .. } | NodeKind::Diverge => {
                PlanDecision {
                    plan: ExecPlan::Primitive,
                    estimated_cost: 1,
                }
            }
            NodeKind::Add(_)
            | NodeKind::Move(_)
            | NodeKind::Clear
            | NodeKind::ClearAt(_)
            | NodeKind::Distribute { .. }
            | NodeKind::Affine { .. }
            | NodeKind::Shift { .. }
            | NodeKind::Square { .. }
            | NodeKind::MulAdd { .. } => match self.exact_memo_spec(id) {
                Some(exact) => PlanDecision {
                    plan: ExecPlan::ExactMemo(exact),
                    estimated_cost: 1 + usize::from(exact.window.len).max(1),
                },
                None => PlanDecision {
                    plan: ExecPlan::Primitive,
                    estimated_cost: 1,
                },
            },
            NodeKind::Loop(_) => self.plan_loop_node(id),
            NodeKind::Seq(children) => {
                let children = children.clone();
                self.plan_seq_node(&children, id)
            }
        };
        self.plan_decisions.insert(id, decision);
        decision
    }

    fn plan_loop_node(&mut self, id: NodeId) -> PlanDecision {
        let analysis = self
            .loop_analysis(id)
            .cloned()
            .or_invariant("loop nodes must produce loop analysis");
        let body = match analysis {
            LoopAnalysis::ExactMemoPlusDirectKernel { body, .. }
            | LoopAnalysis::ExactMemoPlusSymbolicPower {
                powered: PoweredLoopAnalysis { body, .. },
                ..
            }
            | LoopAnalysis::ExactMemoOnly { body, .. }
            | LoopAnalysis::Residual { body } => body,
        };
        let residual_cost = 6 + self.plan_cost(body) * 4;
        match analysis {
            LoopAnalysis::ExactMemoPlusDirectKernel { body, exact } => {
                let exact_cost = 4 + usize::from(exact.window.len) + self.plan_cost(body);
                if exact_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::ExactLoopMemo { body, exact },
                        estimated_cost: exact_cost,
                    }
                } else {
                    PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    }
                }
            }
            LoopAnalysis::ExactMemoPlusSymbolicPower { exact, powered } => {
                // Powers are straight-line guarded summaries; code-size grows with the table,
                // but runtime cost grows only with the set bits in the iteration count.
                let powered_cost =
                    2 + usize::from(exact.window.len) + powered.powers.len().div_ceil(32);
                if powered_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::ExactPoweredLoopMemo {
                            body: powered.body,
                            exact,
                            max_power: u8::try_from(powered.powers.len() - 1)
                                .or_invariant("powered-loop count exceeded u8"),
                        },
                        estimated_cost: powered_cost,
                    }
                } else {
                    PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    }
                }
            }
            LoopAnalysis::ExactMemoOnly { body, exact } => {
                let exact_cost = 4 + usize::from(exact.window.len) + self.plan_cost(body);
                if exact_cost < residual_cost {
                    PlanDecision {
                        plan: ExecPlan::ExactLoopMemo { body, exact },
                        estimated_cost: exact_cost,
                    }
                } else {
                    PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    }
                }
            }
            LoopAnalysis::Residual { .. } => PlanDecision {
                plan: ExecPlan::Residual,
                estimated_cost: residual_cost,
            },
        }
    }

    fn plan_seq_node(&mut self, children: &[NodeId], id: NodeId) -> PlanDecision {
        let residual_cost = 1 + children
            .iter()
            .map(|&child| self.plan_cost(child))
            .sum::<usize>();
        match self.exact_memo_spec(id) {
            Some(exact) => {
                let exact_cost = 2 + usize::from(exact.window.len) + children.len().div_ceil(2);
                if exact_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::ExactMemo(exact),
                        estimated_cost: exact_cost,
                    }
                } else {
                    PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    }
                }
            }
            None => PlanDecision {
                plan: ExecPlan::Residual,
                estimated_cost: residual_cost,
            },
        }
    }
}
