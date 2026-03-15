use super::*;
use crate::RequiredExt;

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
            if let Err(error) = self
                .simulation
                .try_load_hashlife_state_at_generation(grid, self.generation)
            {
                self.advance_failure = Some(conversion_failure_at(error, self.generation));
                return self.runtime_failure_outcome();
            }
            self.phase = OraclePhase::HashLifeApprox;
        }
        if self.confirmed_emitter_cycle.is_none() {
            self.try_confirm_emitter_cycle();
        }
        if let Some(emitter_cycle) = self.confirmed_emitter_cycle.clone() {
            if target_generation <= ORACLE_RUNTIME_EMITTER_EXACT_METRICS_MAX_GENERATION {
                if self.generation < target_generation {
                    let jump = target_generation.saturating_sub(self.generation);
                    if !self.advance_by(jump) {
                        return self.runtime_failure_outcome();
                    }
                }
                if !self.ensure_runtime_session_state() {
                    return self.runtime_failure_outcome();
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
                    state: OracleRuntimeState::RetainedHashLife,
                    failure: None,
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
                state: OracleRuntimeState::Modeled,
                failure: None,
            };
        }
        self.advance_runtime_metadata_to_target(target_generation, on_step, use_probe_prefix)
    }

    pub(super) fn advance_by(&mut self, step_span: u64) -> bool {
        self.last_step_span = 0;
        if step_span <= 1 && self.is_hashlife_phase() {
            let result = self.simulation.advance_hashlife_root(1);
            self.grid = None;
            self.phase = OraclePhase::HashLifeApprox;
            match result {
                Ok(stats) => {
                    self.generation = stats.reached_generation;
                    self.last_step_span = stats.completed_generations;
                }
                Err(error) => {
                    self.generation = error.reached_generation();
                    self.last_step_span = error.completed_generations();
                    self.advance_failure = Some(error);
                    return false;
                }
            }
        } else if step_span <= 1 {
            let current_grid = self.take_or_sample_grid();
            self.grid =
                Some(step_grid_with_changes_and_memo(&current_grid, &mut self.exact_memo).0);
            self.exact_memo.maybe_collect_transition_caches();
            self.phase = OraclePhase::ExactConfirmation;
            let Some(reached_generation) = self.generation.checked_add(step_span) else {
                self.advance_failure = Some(HashLifeAdvanceError::GenerationOverflow {
                    starting_generation: self.generation,
                    requested_delta: step_span,
                    completed_generations: 0,
                    reached_generation: self.generation,
                });
                return false;
            };
            self.generation = reached_generation;
            self.last_step_span = step_span;
        } else {
            if !self.is_hashlife_phase() {
                let current_grid = self.ensure_sampled_grid().clone();
                if let Err(error) = self
                    .simulation
                    .try_load_hashlife_state_at_generation(&current_grid, self.generation)
                {
                    self.advance_failure = Some(conversion_failure_at(error, self.generation));
                    return false;
                }
            }
            let result = self.simulation.advance_hashlife_root(step_span);
            self.grid = None;
            self.phase = OraclePhase::HashLifeApprox;
            match result {
                Ok(stats) => {
                    self.generation = stats.reached_generation;
                    self.last_step_span = stats.completed_generations;
                }
                Err(error) => {
                    self.generation = error.reached_generation();
                    self.last_step_span = error.completed_generations();
                    self.advance_failure = Some(error);
                    return false;
                }
            }
        }
        self.assert_hashlife_generation_aligned();
        true
    }

    fn ensure_runtime_session_state(&mut self) -> bool {
        if self.is_hashlife_phase() {
            self.assert_hashlife_generation_aligned();
            return true;
        }
        let Some(grid) = self.grid.as_ref() else {
            return true;
        };
        if let Err(error) = self
            .simulation
            .try_load_hashlife_state_at_generation(grid, self.generation)
        {
            self.advance_failure = Some(conversion_failure_at(error, self.generation));
            return false;
        }
        self.grid = None;
        self.phase = OraclePhase::HashLifeApprox;
        self.assert_hashlife_generation_aligned();
        true
    }

    fn assert_hashlife_generation_aligned(&self) {
        if !self.is_hashlife_phase() {
            return;
        }
        assert!(
            self.simulation.hashlife_loaded(),
            "HashLife oracle phase requires a loaded simulation session"
        );
        assert_eq!(
            self.generation,
            self.simulation.hashlife_generation(),
            "oracle and HashLife session generations diverged"
        );
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

    fn apply_cycle_skip(&mut self, generation_skip: u64, dx: Coord, dy: Coord) -> bool {
        if generation_skip == 0 {
            return true;
        }
        if self.is_hashlife_phase() {
            match self.simulation.skip_hashlife_generations(generation_skip) {
                Ok(stats) => self.generation = stats.reached_generation,
                Err(error) => {
                    self.generation = error.reached_generation();
                    self.advance_failure = Some(error);
                    return false;
                }
            }
            if (dx != 0 || dy != 0) && self.simulation.shift_hashlife_origin(dx, dy).is_err() {
                return false;
            }
            self.grid = None;
        } else {
            let Some(reached_generation) = self.generation.checked_add(generation_skip) else {
                self.advance_failure = Some(HashLifeAdvanceError::GenerationOverflow {
                    starting_generation: self.generation,
                    requested_delta: generation_skip,
                    completed_generations: 0,
                    reached_generation: self.generation,
                });
                return false;
            };
            self.generation = reached_generation;
            if dx != 0 || dy != 0 {
                let current_grid = self
                    .grid
                    .take()
                    .or_invariant("translated cycle skip requires a materialized grid");
                self.grid = Some(current_grid.translated(dx, dy));
            }
        }
        true
    }

    fn runtime_failure_outcome(&mut self) -> OracleRuntimeOutcome {
        let metrics = self.current_state_shape();
        OracleRuntimeOutcome {
            classification: Classification::Unknown {
                simulated: self.generation,
            },
            final_generation: self.generation,
            population: metrics.population,
            bounds_span: metrics.bounds_span,
            state: if self.is_hashlife_phase() {
                OracleRuntimeState::RetainedHashLife
            } else {
                OracleRuntimeState::Modeled
            },
            failure: self.advance_failure,
        }
    }

    fn retain_projected_constellation(
        &mut self,
        target_generation: u64,
        projected: BitGrid,
        on_step: &mut Option<OracleStepCallback<'_>>,
    ) -> OracleRuntimeOutcome {
        let remaining = target_generation.saturating_sub(self.generation);
        if let Some(callback) = on_step.as_deref_mut() {
            callback(
                OracleStepPlan {
                    generation: self.generation,
                    step_span: remaining,
                    backend: SimulationBackend::HashLife,
                },
                self.current_state_shape(),
            );
        }
        let population = projected.population();
        let bounds_span = projected
            .bounds()
            .map(|bounds| bounds_dimensions(bounds).2)
            .unwrap_or(0);
        if let Err(error) = self
            .simulation
            .try_load_hashlife_state_at_generation(&projected, target_generation)
        {
            self.advance_failure = Some(conversion_failure_at(error, self.generation));
            return self.runtime_failure_outcome();
        }
        self.generation = target_generation;
        self.grid = None;
        self.phase = OraclePhase::HashLifeApprox;
        self.last_step_span = remaining;
        OracleRuntimeOutcome {
            classification: Classification::Unknown {
                simulated: target_generation,
            },
            final_generation: target_generation,
            population,
            bounds_span,
            state: OracleRuntimeState::RetainedHashLife,
            failure: None,
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
                let cycle_count =
                    Coord::try_from(skip_cycles).or_invariant("cycle skip exceeded Coord");
                if !self.apply_cycle_skip(
                    skip_cycles * cycle.period,
                    cycle
                        .delta
                        .0
                        .checked_mul(cycle_count)
                        .or_invariant("cycle x overflow"),
                    cycle
                        .delta
                        .1
                        .checked_mul(cycle_count)
                        .or_invariant("cycle y overflow"),
                ) {
                    return OracleAdvanceOutcome {
                        classification: Classification::Unknown {
                            simulated: self.generation,
                        },
                        final_generation: self.generation,
                        grid: self.take_or_sample_grid(),
                    };
                }
            }
        }

        while self.generation < target_generation {
            if !self.advance_by(1) {
                break;
            }
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
                detected_at: cycle.detected_at,
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
                let cycle_count =
                    Coord::try_from(skip_cycles).or_invariant("cycle skip exceeded Coord");
                if !self.apply_cycle_skip(
                    skip_cycles * cycle.period,
                    cycle
                        .delta
                        .0
                        .checked_mul(cycle_count)
                        .or_invariant("cycle x overflow"),
                    cycle
                        .delta
                        .1
                        .checked_mul(cycle_count)
                        .or_invariant("cycle y overflow"),
                ) {
                    return self.runtime_failure_outcome();
                }
            }
        }

        while self.generation < target_generation {
            if !self.advance_by(1) {
                return self.runtime_failure_outcome();
            }
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
                detected_at: cycle.detected_at,
            }
        };
        if !self.ensure_runtime_session_state() {
            return self.runtime_failure_outcome();
        }
        let metrics = self.current_state_shape();

        OracleRuntimeOutcome {
            classification,
            final_generation: self.generation,
            population: metrics.population,
            bounds_span: metrics.bounds_span,
            state: OracleRuntimeState::RetainedHashLife,
            failure: None,
        }
    }

    pub(super) fn land_confirmed_emitter_cycle_to_target(
        &mut self,
        target_generation: u64,
        emitter_cycle: &ConfirmedEmitterCycle,
    ) -> OracleAdvanceOutcome {
        if self.generation < target_generation {
            let _ = self.advance_by(target_generation.saturating_sub(self.generation));
        }
        let reached_target = self.generation == target_generation;
        OracleAdvanceOutcome {
            classification: if reached_target {
                Classification::LikelyInfinite {
                    reason: "emitter_cycle",
                    detected_at: emitter_cycle.first_seen,
                }
            } else {
                Classification::Unknown {
                    simulated: self.generation,
                }
            },
            final_generation: self.generation,
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
        let mut periodic_probe_generation = 64_u64;
        let mut hashlife_checkpoint_probes = 0_usize;

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
                    state: OracleRuntimeState::Modeled,
                    failure: None,
                };
            }

            if self.should_sample_state(
                target_generation,
                cycle_probe_limit,
                next_checkpoint_generation,
            ) {
                if self.is_hashlife_phase() {
                    let population = self.simulation.hashlife_population_count();
                    if population.is_some_and(PopulationCount::is_zero) {
                        let observed_classification = self.extinction_classification();
                        let remaining = target_generation.saturating_sub(self.generation);
                        if remaining != 0 {
                            if let Some(callback) = on_step.as_deref_mut() {
                                callback(
                                    OracleStepPlan {
                                        generation: self.generation,
                                        step_span: remaining,
                                        backend: SimulationBackend::HashLife,
                                    },
                                    OracleStateMetrics::default(),
                                );
                            }
                            let Ok(stats) = self.simulation.skip_hashlife_generations(remaining)
                            else {
                                return self.runtime_failure_outcome();
                            };
                            self.generation = stats.reached_generation;
                            self.assert_hashlife_generation_aligned();
                        }
                        let classification = match observed_classification {
                            Classification::Unknown { .. } => Classification::Unknown {
                                simulated: target_generation,
                            },
                            exact => exact,
                        };
                        return OracleRuntimeOutcome {
                            classification,
                            final_generation: target_generation,
                            population: 0,
                            bounds_span: 0,
                            state: OracleRuntimeState::RetainedHashLife,
                            failure: None,
                        };
                    }

                    if let Some(checkpoint) = self.simulation.hashlife_checkpoint().cloned() {
                        hashlife_checkpoint_probes = hashlife_checkpoint_probes.saturating_add(1);
                        if self.prepare_hashlife_checkpoint(&checkpoint) {
                            // Root-level changes invalidate retained identities. Restart the
                            // adjacent probes so late stabilizations and oscillators are observed
                            // before another coarse jump changes the checkpoint epoch again.
                            hashlife_checkpoint_probes = 1;
                        }
                        if self.checkpoints.contains_key(&checkpoint.identity) {
                            if let Some(cycle) =
                                self.observe_repeated_hashlife_checkpoint(&checkpoint)
                            {
                                self.confirmed_cycle = Some(cycle);
                                return self
                                    .runtime_outcome_for_confirmed_cycle(target_generation, cycle);
                            }
                        } else {
                            let remaining = target_generation.saturating_sub(self.generation);
                            let projected = (self.generation >= 64
                                && checkpoint.population
                                    <= u128::from(ORACLE_CONSTELLATION_MAX_POPULATION))
                            .then(|| {
                                self.simulation.sample_hashlife_state_grid(
                                    GridExtractionPolicy::FullGridIfUnder {
                                        max_population: u128::from(
                                            ORACLE_CONSTELLATION_MAX_POPULATION,
                                        ),
                                        max_chunks: ORACLE_CONSTELLATION_MAX_CHUNKS,
                                        max_bounds_span: ORACLE_CONSTELLATION_MAX_SPAN,
                                    },
                                )
                            })
                            .transpose()
                            .ok()
                            .flatten()
                            .and_then(|sample| {
                                constellation::project_periodic_constellation(&sample, remaining)
                            });
                            if let Some(projected) = projected {
                                return self.retain_projected_constellation(
                                    target_generation,
                                    projected,
                                    on_step,
                                );
                            }
                            self.record_hashlife_checkpoint(checkpoint);
                        }
                    }

                    if self.generation >= cycle_probe_limit
                        && hashlife_checkpoint_probes >= ORACLE_HASHLIFE_CHECKPOINT_PROBE_COUNT
                    {
                        while periodic_probe_generation <= self.generation {
                            periodic_probe_generation = periodic_probe_generation.saturating_mul(2);
                        }
                        next_checkpoint_generation =
                            periodic_probe_generation.min(target_generation);
                    } else if self.generation >= cycle_probe_limit {
                        next_checkpoint_generation = self
                            .generation
                            .saturating_add(ORACLE_HASHLIFE_CHECKPOINT_PROBE_STEP)
                            .min(target_generation);
                    }
                } else {
                    if let Some(outcome) = self.advance_checkpoint(target_generation) {
                        let bounds_span = outcome
                            .grid
                            .bounds()
                            .map(|bounds| bounds_dimensions(bounds).2)
                            .unwrap_or(0);
                        if let Err(error) = self.simulation.try_load_hashlife_state_at_generation(
                            &outcome.grid,
                            outcome.final_generation,
                        ) {
                            self.advance_failure =
                                Some(conversion_failure_at(error, self.generation));
                            return self.runtime_failure_outcome();
                        }
                        self.phase = OraclePhase::HashLifeApprox;
                        self.grid = None;
                        return OracleRuntimeOutcome {
                            classification: outcome.classification,
                            final_generation: outcome.final_generation,
                            population: outcome.grid.population(),
                            bounds_span,
                            state: OracleRuntimeState::RetainedHashLife,
                            failure: None,
                        };
                    }
                    if self.generation >= cycle_probe_limit {
                        let remaining = target_generation.saturating_sub(self.generation);
                        let projected = self.grid.as_ref().and_then(|grid| {
                            constellation::project_periodic_constellation(grid, remaining)
                        });
                        if let Some(projected) = projected {
                            return self.retain_projected_constellation(
                                target_generation,
                                projected,
                                on_step,
                            );
                        }
                        next_checkpoint_generation =
                            periodic_probe_generation.min(target_generation);
                    }
                }
            }

            if self.generation == target_generation {
                break;
            }

            let remaining = target_generation.saturating_sub(self.generation);
            let mut plan = if self.generation < cycle_probe_limit {
                OracleStepPlan {
                    generation: self.generation,
                    step_span: 1,
                    backend: SimulationBackend::SimdChunk,
                }
            } else if self.is_hashlife_phase() || self.simulation.hashlife_loaded() {
                self.plan_runtime_hashlife_step(remaining, hashlife_checkpoint_probes)
            } else {
                self.plan_target_step(remaining)
            };
            if matches!(plan.backend, SimulationBackend::HashLife)
                && next_checkpoint_generation > self.generation
            {
                plan.step_span = plan
                    .step_span
                    .min(next_checkpoint_generation - self.generation);
            }
            let plan = OracleStepPlan {
                step_span: plan.step_span.min(remaining),
                ..plan
            };
            if let Some(callback) = on_step.as_deref_mut() {
                callback(plan, self.current_state_shape());
            }
            if !self.advance_by(plan.step_span) {
                break;
            }
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

        if self.advance_failure.is_some() {
            return self.runtime_failure_outcome();
        }
        if !self.ensure_runtime_session_state() {
            return self.runtime_failure_outcome();
        }
        let metrics = self.current_state_shape();
        OracleRuntimeOutcome {
            classification: Classification::Unknown {
                simulated: target_generation,
            },
            final_generation: self.generation,
            population: metrics.population,
            bounds_span: metrics.bounds_span,
            state: OracleRuntimeState::RetainedHashLife,
            failure: None,
        }
    }
}

fn conversion_failure_at(error: HashLifeConversionError, generation: u64) -> HashLifeAdvanceError {
    match error {
        HashLifeConversionError::MemoryBudgetExceeded {
            retained_bytes,
            limit_bytes,
            ..
        } => HashLifeAdvanceError::MemoryBudgetExceeded {
            starting_generation: generation,
            requested_delta: 0,
            completed_generations: 0,
            requested_generation: generation,
            reached_generation: generation,
            allocated_bytes: retained_bytes,
            limit_bytes,
        },
        HashLifeConversionError::AllocationFailed { requested_bytes } => {
            HashLifeAdvanceError::AllocationFailed {
                starting_generation: generation,
                requested_delta: 0,
                completed_generations: 0,
                reached_generation: generation,
                requested_bytes,
            }
        }
        HashLifeConversionError::NodeIdExhausted => HashLifeAdvanceError::NodeIdExhausted {
            starting_generation: generation,
            requested_delta: 0,
            completed_generations: 0,
            reached_generation: generation,
        },
        HashLifeConversionError::CanonicalReferenceExhausted => {
            HashLifeAdvanceError::CanonicalReferenceExhausted {
                starting_generation: generation,
                requested_delta: 0,
                completed_generations: 0,
                reached_generation: generation,
            }
        }
        HashLifeConversionError::CoordinateRangeExceeded { .. }
        | HashLifeConversionError::Snapshot(_) => HashLifeAdvanceError::CoordinateRangeExceeded {
            starting_generation: generation,
            requested_delta: 0,
            completed_generations: 0,
            reached_generation: generation,
            required_level: crate::hashlife::MAX_COORD_ROOT_LEVEL + 1,
        },
    }
}
