use super::*;

impl EmitterEngine {
    pub(super) fn plan_node(&mut self, id: NodeId) -> ExecPlan {
        self.plan_decision(id).plan
    }

    pub(super) fn plan_cost(&mut self, id: NodeId) -> usize {
        self.plan_decision(id).estimated_cost
    }

    pub(super) fn plan_decision(&mut self, id: NodeId) -> PlanDecision {
        if let Some(decision) = self.plan_decisions.get(&id) {
            return decision.clone();
        }
        let decision = match self.interner.get(id).clone() {
            NodeKind::Add(_)
            | NodeKind::Move(_)
            | NodeKind::Input
            | NodeKind::Output
            | NodeKind::Diverge => PlanDecision {
                plan: ExecPlan::Primitive,
                estimated_cost: 1,
            },
            NodeKind::Clear => PlanDecision {
                plan: ExecPlan::DirectKernel("clear"),
                estimated_cost: 1,
            },
            NodeKind::Distribute(targets) => PlanDecision {
                plan: ExecPlan::DirectKernel("distribute"),
                estimated_cost: 2 + targets.len(),
            },
            NodeKind::Loop(_) => self.plan_loop_node(id),
            NodeKind::Seq(children) => self.plan_seq_node(&children, id),
        };
        self.plan_decisions.insert(id, decision.clone());
        decision
    }

    fn plan_loop_node(&mut self, id: NodeId) -> PlanDecision {
        let analysis = self
            .loop_analysis(id)
            .expect("loop nodes must produce loop analysis");
        let body = match &analysis {
            LoopAnalysis::DirectKernel { body, .. }
            | LoopAnalysis::Powered(PoweredLoopAnalysis { body, .. })
            | LoopAnalysis::Symbolic { body, .. }
            | LoopAnalysis::Residual { body } => *body,
        };
        let residual_cost = 6 + self.plan_cost(body) * 4;
        match analysis {
            LoopAnalysis::DirectKernel {
                accessed_offsets, ..
            } => {
                let direct_cost = 12 + accessed_offsets.len() * 2;
                if direct_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::DirectKernel("loop_kernel"),
                        estimated_cost: direct_cost,
                    }
                } else {
                    PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    }
                }
            }
            LoopAnalysis::Powered(powered) => {
                let powered_cost =
                    2 + usize::from(powered.window.len) + powered.powers.len().div_ceil(4);
                let direct_cost = 12 + powered.powers[0].accessed_offsets().len() * 2;
                if powered_cost < direct_cost && powered_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::PoweredSymbolicLoop {
                            body: powered.body,
                            window: powered.window,
                            max_power: powered.powers.len() as u8 - 1,
                        },
                        estimated_cost: powered_cost,
                    }
                } else if direct_cost <= residual_cost {
                    PlanDecision {
                        plan: ExecPlan::DirectKernel("loop_kernel"),
                        estimated_cost: direct_cost,
                    }
                } else {
                    PlanDecision {
                        plan: ExecPlan::Residual,
                        estimated_cost: residual_cost,
                    }
                }
            }
            LoopAnalysis::Symbolic { body, window } => {
                let symbolic_cost = 10 + usize::from(window.len) * 3 + self.plan_cost(body);
                if symbolic_cost < residual_cost {
                    PlanDecision {
                        plan: ExecPlan::SymbolicLoop { body, window },
                        estimated_cost: symbolic_cost,
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
        let residual_cost = 1 + children.iter().map(|&child| self.plan_cost(child)).sum::<usize>();
        let transfer = self.transfer(id);
        if !transfer.is_pure_windowed() || transfer.ptr_delta != 0 || transfer.has_opaque_effects()
        {
            PlanDecision {
                plan: ExecPlan::Residual,
                estimated_cost: residual_cost,
            }
        } else {
            match transfer.memo_window() {
                Some(window) => {
                    let symbolic_cost = 1
                        + transfer.effects.len()
                        + transfer.reads.len().div_ceil(2)
                        + usize::from(window.len).div_ceil(2);
                    if symbolic_cost <= residual_cost {
                        PlanDecision {
                            plan: ExecPlan::SymbolicMemo(window),
                            estimated_cost: symbolic_cost,
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
}
