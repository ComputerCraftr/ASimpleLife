use super::*;
use crate::RequiredExt;

impl EmitterEngine {
    pub(in crate::bf) fn plan_node(&mut self, id: NodeId) -> ExecPlan {
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
            NodeKind::Seq(_) => self.plan_seq_node(id),
        };
        self.plan_decisions.insert(id, decision);
        decision
    }

    fn plan_loop_node(&mut self, id: NodeId) -> PlanDecision {
        let body = match self
            .loop_analysis(id)
            .or_invariant("loop nodes must produce loop analysis")
        {
            LoopAnalysis::ExactMemoPlusDirectKernel { body, .. }
            | LoopAnalysis::ExactMemoPlusSymbolicPower {
                powered: PoweredLoopAnalysis { body, .. },
                ..
            }
            | LoopAnalysis::ExactMemoOnly { body, .. }
            | LoopAnalysis::Residual { body } => *body,
        };
        let body_cost = self.plan_cost(body);
        let residual_cost = 6 + body_cost * 4;
        self.prepare_powers(id, body, body_cost);
        match self.loop_analyses[&id]
            .as_ref()
            .or_invariant("loop analysis is cached")
        {
            LoopAnalysis::ExactMemoPlusDirectKernel { body, exact } => {
                let exact_cost = 4 + usize::from(exact.window.len) + body_cost;
                if exact_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::ExactLoopMemo {
                            body: *body,
                            exact: *exact,
                        },
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
                if !powered
                    .powers
                    .iter()
                    .all(|power| self.compile_work.admit_evaluation(power))
                {
                    return PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    };
                }
                let table_cost = if powered.only_drains_guard() {
                    1
                } else {
                    powered
                        .powers
                        .iter()
                        .map(|power| super::powers::transfer_cost(power) + 2)
                        .sum::<usize>()
                };
                let powered_cost = 2 + usize::from(exact.window.len) + table_cost;
                let max_iterations =
                    super::powers::maximum_proven_iterations(self.opts, powered.guard_delta);
                let fallback_work = u128::from(max_iterations)
                    * u128::try_from(body_cost + 1).or_invariant("body cost fits u128");
                let max_power = u8::try_from(powered.powers.len() - 1)
                    .or_invariant("powered-loop count exceeded u8");
                let applications = if powered.only_drains_guard() {
                    1
                } else {
                    max_iterations >> max_power
                };
                let largest_cost = if powered.only_drains_guard() {
                    1
                } else {
                    super::powers::transfer_cost(&powered.powers[usize::from(max_power)]) + 2
                };
                let runtime_work = u128::from(applications)
                    * u128::try_from(largest_cost).or_invariant("power cost fits u128")
                    + u128::try_from(powered_cost).or_invariant("plan cost fits u128");
                if runtime_work <= fallback_work {
                    PlanDecision {
                        plan: ExecPlan::ExactPoweredLoopMemo {
                            body: powered.body,
                            exact: *exact,
                            max_power,
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
                let exact_cost = 4 + usize::from(exact.window.len) + body_cost;
                if exact_cost < residual_cost {
                    PlanDecision {
                        plan: ExecPlan::ExactLoopMemo {
                            body: *body,
                            exact: *exact,
                        },
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

    fn plan_seq_node(&mut self, id: NodeId) -> PlanDecision {
        let len = match self.interner.get(id) {
            NodeKind::Seq(children) => children.len(),
            _ => crate::invariant_failure!("sequence planning requires sequence node"),
        };
        let mut residual_cost = 1;
        for index in 0..len {
            let child = match self.interner.get(id) {
                NodeKind::Seq(children) => children[index],
                _ => crate::invariant_failure!("sequence planning requires sequence node"),
            };
            residual_cost += self.plan_cost(child);
        }
        match self.exact_memo_spec(id) {
            Some(exact) => {
                let exact_cost = 2 + usize::from(exact.window.len) + len.div_ceil(2);
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
