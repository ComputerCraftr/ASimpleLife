use super::*;

impl<'a> OracleSession<'a> {
    pub fn advance_runtime_target(
        mut self,
        target_generation: u64,
        mut on_step: Option<OracleStepCallback<'_>>,
    ) -> OracleRuntimeOutcome {
        self.advance_runtime_target_internal(target_generation, &mut on_step, true)
    }

    pub fn advance_runtime_target_hashlife_first(
        mut self,
        target_generation: u64,
        mut on_step: Option<OracleStepCallback<'_>>,
    ) -> OracleRuntimeOutcome {
        self.advance_runtime_target_internal(target_generation, &mut on_step, false)
    }

    fn advance_runtime_target_internal(
        &mut self,
        target_generation: u64,
        on_step: &mut Option<OracleStepCallback<'_>>,
        use_probe_prefix: bool,
    ) -> OracleRuntimeOutcome {
        if !use_probe_prefix
            && !self.is_hashlife_phase()
            && let Some(grid) = self.grid.as_ref()
        {
            self.simulation.load_hashlife_state(grid);
            self.phase = OraclePhase::HashLifeApprox;
        }
        if self.confirmed_emitter_cycle.is_none() {
            self.try_confirm_emitter_cycle();
        }
        if let Some(emitter_cycle) = self.confirmed_emitter_cycle.clone() {
            if target_generation <= ORACLE_RUNTIME_EMITTER_EXACT_METRICS_MAX_GENERATION {
                if self.generation < target_generation {
                    let jump = target_generation.saturating_sub(self.generation);
                    self.advance_by(jump);
                }
                let metrics = self.current_state_shape();
                return OracleRuntimeOutcome {
                    classification: Classification::LikelyInfinite {
                        reason: "emitter_cycle",
                        detected_at: emitter_cycle.first_seen,
                    },
                    final_generation: target_generation,
                    population: metrics.population,
                    bounds_span: metrics.bounds_span,
                };
            }
            if let Some(callback) = on_step.as_deref_mut() {
                let baseline_generation = emitter_cycle.model.baseline_generation;
                if baseline_generation > self.generation {
                    callback(
                        OracleStepPlan {
                            generation: self.generation,
                            step_span: baseline_generation.saturating_sub(self.generation),
                            backend: SimulationBackend::HashLife,
                        },
                        self.current_state_shape(),
                    );
                }
                callback(
                    OracleStepPlan {
                        generation: target_generation,
                        step_span: 0,
                        backend: SimulationBackend::HashLife,
                    },
                    OracleStateMetrics {
                        population: emitter_runtime_population(
                            &emitter_cycle.model,
                            target_generation,
                        ),
                        bounds_span: emitter_runtime_bounds_span(
                            &emitter_cycle.model,
                            target_generation,
                        ),
                    },
                );
            }
            return OracleRuntimeOutcome {
                classification: Classification::LikelyInfinite {
                    reason: "emitter_cycle",
                    detected_at: emitter_cycle.first_seen,
                },
                final_generation: target_generation,
                population: emitter_runtime_population(&emitter_cycle.model, target_generation),
                bounds_span: emitter_runtime_bounds_span(&emitter_cycle.model, target_generation),
            };
        }
        self.advance_runtime_metadata_to_target(target_generation, on_step, use_probe_prefix)
    }

    pub(super) fn advance_by(&mut self, step_span: u64) {
        self.last_step_span = step_span;
        if step_span <= 1 && (self.is_hashlife_phase() || self.simulation.hashlife_loaded()) {
            self.simulation.advance_hashlife_root(1);
            self.grid = None;
            self.phase = OraclePhase::HashLifeApprox;
        } else if step_span <= 1 {
            let current_grid = self.take_or_sample_grid();
            self.grid =
                Some(step_grid_with_changes_and_memo(&current_grid, &mut self.exact_memo).0);
            self.exact_memo.maybe_collect_transition_caches();
            self.phase = OraclePhase::ExactConfirmation;
        } else {
            if !self.is_hashlife_phase() {
                let current_grid = self.ensure_sampled_grid().clone();
                self.simulation.load_hashlife_state(&current_grid);
            }
            self.simulation.advance_hashlife_root(step_span);
            self.grid = None;
            self.phase = OraclePhase::HashLifeApprox;
        }
        self.generation += step_span;
    }

    pub(super) fn extinction_classification(&self) -> Classification {
        if self.last_step_span <= 1 {
            Classification::DiesOut {
                at_generation: self.generation,
            }
        } else {
            Classification::Unknown {
                simulated: self.generation,
            }
        }
    }

    fn apply_cycle_skip(&mut self, generation_skip: u64, dx: Coord, dy: Coord) {
        if generation_skip == 0 {
            return;
        }
        self.generation = self.generation.saturating_add(generation_skip);
        if dx == 0 && dy == 0 {
            return;
        }
        if self.is_hashlife_phase() {
            self.simulation.shift_hashlife_origin(dx, dy);
            self.grid = None;
        } else {
            let current_grid = self
                .grid
                .take()
                .expect("translated cycle skip requires a materialized grid");
            self.grid = Some(current_grid.translated(dx, dy));
        }
    }

    pub(super) fn land_confirmed_cycle_to_target(
        &mut self,
        target_generation: u64,
        cycle: ConfirmedCycle,
    ) -> OracleAdvanceOutcome {
        if cycle.period > 0 && self.generation < target_generation {
            let remaining = target_generation - self.generation;
            let skip_cycles = remaining / cycle.period;
            if skip_cycles > 0 {
                let cycle_count = Coord::try_from(skip_cycles).expect("cycle skip exceeded Coord");
                self.apply_cycle_skip(
                    skip_cycles * cycle.period,
                    cycle
                        .delta
                        .0
                        .checked_mul(cycle_count)
                        .expect("cycle x overflow"),
                    cycle
                        .delta
                        .1
                        .checked_mul(cycle_count)
                        .expect("cycle y overflow"),
                );
            }
        }

        while self.generation < target_generation {
            self.advance_by(1);
        }

        let classification = if cycle.delta == (0, 0) {
            Classification::Repeats {
                period: cycle.period,
                first_seen: cycle.first_seen,
            }
        } else {
            Classification::Spaceship {
                period: cycle.period,
                first_seen: cycle.first_seen,
                delta: cycle.delta,
                detected_at: self.generation,
            }
        };

        OracleAdvanceOutcome {
            classification,
            final_generation: self.generation,
            grid: self.take_or_sample_grid(),
        }
    }

    fn runtime_outcome_for_confirmed_cycle(
        &mut self,
        target_generation: u64,
        cycle: ConfirmedCycle,
    ) -> OracleRuntimeOutcome {
        if cycle.period > 0 && self.generation < target_generation {
            let remaining = target_generation - self.generation;
            let skip_cycles = remaining / cycle.period;
            if skip_cycles > 0 {
                let cycle_count = Coord::try_from(skip_cycles).expect("cycle skip exceeded Coord");
                self.apply_cycle_skip(
                    skip_cycles * cycle.period,
                    cycle
                        .delta
                        .0
                        .checked_mul(cycle_count)
                        .expect("cycle x overflow"),
                    cycle
                        .delta
                        .1
                        .checked_mul(cycle_count)
                        .expect("cycle y overflow"),
                );
            }
        }

        while self.generation < target_generation {
            self.advance_by(1);
        }

        let classification = if cycle.delta == (0, 0) {
            Classification::Repeats {
                period: cycle.period,
                first_seen: cycle.first_seen,
            }
        } else {
            Classification::Spaceship {
                period: cycle.period,
                first_seen: cycle.first_seen,
                delta: cycle.delta,
                detected_at: self.generation,
            }
        };
        let metrics = self.current_state_shape();

        OracleRuntimeOutcome {
            classification,
            final_generation: self.generation,
            population: metrics.population,
            bounds_span: metrics.bounds_span,
        }
    }

    pub(super) fn land_confirmed_emitter_cycle_to_target(
        &mut self,
        target_generation: u64,
        emitter_cycle: &ConfirmedEmitterCycle,
    ) -> OracleAdvanceOutcome {
        if self.generation < emitter_cycle.model.baseline_generation {
            let jump = emitter_cycle
                .model
                .baseline_generation
                .saturating_sub(self.generation);
            self.advance_by(jump);
        }
        OracleAdvanceOutcome {
            classification: Classification::LikelyInfinite {
                reason: "emitter_cycle",
                detected_at: emitter_cycle.first_seen,
            },
            final_generation: target_generation,
            grid: self.take_or_sample_grid(),
        }
    }

    fn advance_runtime_metadata_to_target(
        &mut self,
        target_generation: u64,
        on_step: &mut Option<OracleStepCallback<'_>>,
        use_probe_prefix: bool,
    ) -> OracleRuntimeOutcome {
        let cycle_probe_limit = if use_probe_prefix {
            self.generation
                .saturating_add(cycle_probe_prefix_window(
                    self.current_state_shape().population,
                    self.current_state_shape().bounds_span,
                ))
                .min(target_generation)
        } else {
            self.generation
        };
        let mut next_checkpoint_generation = cycle_probe_limit.saturating_add(1);
        let mut checkpoint_stride = 1_u64;

        while self.generation <= target_generation {
            if let Some(cycle) = self.confirmed_cycle {
                return self.runtime_outcome_for_confirmed_cycle(target_generation, cycle);
            }
            if let Some(emitter_cycle) = self.confirmed_emitter_cycle.clone() {
                return OracleRuntimeOutcome {
                    classification: Classification::LikelyInfinite {
                        reason: "emitter_cycle",
                        detected_at: emitter_cycle.first_seen,
                    },
                    final_generation: target_generation,
                    population: emitter_runtime_population(&emitter_cycle.model, target_generation),
                    bounds_span: emitter_runtime_bounds_span(
                        &emitter_cycle.model,
                        target_generation,
                    ),
                };
            }

            if self.should_sample_state(
                target_generation,
                cycle_probe_limit,
                next_checkpoint_generation,
            ) {
                if self.is_hashlife_phase() {
                    let population = self.simulation.hashlife_population().unwrap_or(0);
                    if population == 0 {
                        return OracleRuntimeOutcome {
                            classification: self.extinction_classification(),
                            final_generation: self.generation,
                            population: 0,
                            bounds_span: 0,
                        };
                    }

                    if let Some(checkpoint) = self.simulation.hashlife_checkpoint().cloned() {
                        let checkpoint_key = checkpoint.signature.key();
                        if !self.checkpoints.contains_key(&checkpoint_key) {
                            self.checkpoints
                                .insert(checkpoint_key, (self.generation, checkpoint.origin));
                        }
                    }

                    if self.generation >= cycle_probe_limit {
                        checkpoint_stride = checkpoint_stride.saturating_mul(2).min(1 << 20);
                        next_checkpoint_generation = self
                            .generation
                            .saturating_add(checkpoint_stride)
                            .min(target_generation);
                    }
                } else if let Some(outcome) = self.advance_checkpoint(target_generation) {
                    let bounds_span = outcome
                        .grid
                        .bounds()
                        .map(|bounds| bounds_dimensions(bounds).2)
                        .unwrap_or(0);
                    return OracleRuntimeOutcome {
                        classification: outcome.classification,
                        final_generation: outcome.final_generation,
                        population: outcome.grid.population(),
                        bounds_span,
                    };
                }
            }

            if self.generation == target_generation {
                break;
            }

            let remaining = target_generation.saturating_sub(self.generation);
            let plan = if self.generation < cycle_probe_limit {
                OracleStepPlan {
                    generation: self.generation,
                    step_span: 1,
                    backend: SimulationBackend::SimdChunk,
                }
            } else {
                self.plan_target_step(remaining)
            };
            let plan = OracleStepPlan {
                step_span: plan.step_span.min(remaining),
                ..plan
            };
            if let Some(callback) = on_step.as_deref_mut() {
                callback(plan, self.current_state_shape());
            }
            self.advance_by(plan.step_span);
            if let Some(callback) = on_step.as_deref_mut() {
                callback(
                    OracleStepPlan {
                        generation: self.generation,
                        step_span: 0,
                        backend: plan.backend,
                    },
                    self.current_state_shape(),
                );
            }
        }

        let metrics = self.current_state_shape();
        OracleRuntimeOutcome {
            classification: Classification::LikelyInfinite {
                reason: "oracle_generation_limit",
                detected_at: target_generation,
            },
            final_generation: self.generation,
            population: metrics.population,
            bounds_span: metrics.bounds_span,
        }
    }
}
