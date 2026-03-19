use super::*;

const ORACLE_HASHLIFE_EARLY_LOWBIT_MAX_STEP: u64 = 1 << 19;
const ORACLE_HASHLIFE_EARLY_LOWBIT_REORDER_MAX_GENERATIONS: u64 = 1 << 31;

impl<'a> OracleSession<'a> {
    pub(super) fn runtime_hashlife_lowbit_step(&self, remaining: u64) -> Option<u64> {
        let lowest_set_bit = remaining & remaining.wrapping_neg();
        (lowest_set_bit > 0
            && remaining <= ORACLE_HASHLIFE_EARLY_LOWBIT_REORDER_MAX_GENERATIONS
            && lowest_set_bit <= ORACLE_HASHLIFE_EARLY_LOWBIT_MAX_STEP
            && remaining > ORACLE_HASHLIFE_EARLY_LOWBIT_MAX_STEP)
            .then_some(lowest_set_bit)
    }

    pub(super) fn plan_step(
        &mut self,
        generation_limit: u64,
        nominal_generation_limit: u64,
    ) -> OracleStepPlan {
        let step_span = continuation_step_span(
            self.current_state_shape(),
            self.generation,
            generation_limit,
            nominal_generation_limit,
            self.simulation,
            self.is_hashlife_phase(),
        );
        OracleStepPlan {
            generation: self.generation,
            step_span,
            backend: self.planned_backend_for_shape(step_span),
        }
    }

    pub(super) fn plan_target_step(&mut self, remaining: u64) -> OracleStepPlan {
        let shape = self.current_state_shape();
        let safe_hashlife_jump = max_hashlife_safe_jump_from_span(shape.bounds_span);
        let exact_suffix = target_exact_suffix_window(shape.population, shape.bounds_span);
        let step_span = if remaining <= exact_suffix {
            1
        } else if self.is_hashlife_phase() || self.simulation.hashlife_loaded() {
            largest_power_of_two_leq(
                remaining
                    .saturating_sub(exact_suffix)
                    .min(safe_hashlife_jump)
                    .max(1),
            )
        } else {
            match self.simulation.planned_backend_from_session_metrics(
                shape.population,
                shape.bounds_span,
                remaining,
            ) {
                SimulationBackend::SimdChunk => 1,
                SimulationBackend::HybridSegmented => hybrid_target_prefix_generations(
                    shape.population,
                    remaining.saturating_sub(exact_suffix).max(1),
                ),
                SimulationBackend::HashLife => largest_power_of_two_leq(
                    remaining
                        .saturating_sub(exact_suffix)
                        .min(safe_hashlife_jump)
                        .max(1),
                ),
            }
        };
        OracleStepPlan {
            generation: self.generation,
            step_span,
            backend: self.planned_backend_for_shape(step_span),
        }
    }

    pub(super) fn plan_runtime_hashlife_step(&mut self, remaining: u64) -> OracleStepPlan {
        let shape = self.current_state_shape();
        let safe_hashlife_jump = max_hashlife_safe_jump_from_span(shape.bounds_span).max(1);
        let step_span = if let Some(lowbit) = self.runtime_hashlife_lowbit_step(remaining) {
            lowbit
        } else {
            remaining.min(safe_hashlife_jump).max(1)
        };
        OracleStepPlan {
            generation: self.generation,
            step_span,
            backend: SimulationBackend::HashLife,
        }
    }

    pub(super) fn planned_backend_for_shape(&mut self, step_span: u64) -> SimulationBackend {
        if step_span <= 1 {
            SimulationBackend::SimdChunk
        } else if self.is_hashlife_phase() || self.simulation.hashlife_loaded() {
            SimulationBackend::HashLife
        } else {
            let shape = self.current_state_shape();
            self.simulation.planned_backend_from_session_metrics(
                shape.population,
                shape.bounds_span,
                step_span,
            )
        }
    }

    pub(super) fn current_state_shape(&mut self) -> OracleStateMetrics {
        if self.is_hashlife_phase() || self.simulation.hashlife_loaded() {
            let population = self
                .simulation
                .hashlife_population()
                .map(|count| usize::try_from(count).expect("hashlife population exceeded usize"))
                .unwrap_or(0);
            let bounds_span = self
                .simulation
                .hashlife_bounds()
                .map(|bounds| bounds_dimensions(bounds).2)
                .unwrap_or(0);
            OracleStateMetrics {
                population,
                bounds_span,
            }
        } else {
            let grid = self
                .grid
                .as_ref()
                .expect("oracle exact phase should have a grid");
            OracleStateMetrics {
                population: grid.population(),
                bounds_span: grid
                    .bounds()
                    .map(|bounds| bounds_dimensions(bounds).2)
                    .unwrap_or(0),
            }
        }
    }

    pub(super) fn should_sample_state(
        &self,
        target_generation: u64,
        cycle_probe_limit: u64,
        next_checkpoint_generation: u64,
    ) -> bool {
        !self.is_hashlife_phase()
            || self.generation <= cycle_probe_limit
            || self.generation == target_generation
            || self.generation >= next_checkpoint_generation
    }

    pub(super) fn ensure_sampled_grid(&mut self) -> &BitGrid {
        if self.grid.is_none() {
            self.simulation
                .record_hashlife_oracle_confirmation_materialization();
            self.grid = Some(
                self.simulation
                    .sample_hashlife_state_grid(confirmation_full_grid_policy())
                    .expect(
                        "hashlife state exceeded safe extraction bounds for exact confirmation",
                    ),
            );
        }
        self.grid
            .as_ref()
            .expect("oracle sampled grid should be available")
    }

    pub(super) fn take_or_sample_grid(&mut self) -> BitGrid {
        if self.grid.is_none() {
            self.ensure_sampled_grid();
        }
        self.grid
            .take()
            .expect("oracle should have a final sampled grid")
    }

    pub(super) fn is_hashlife_phase(&self) -> bool {
        matches!(self.phase, OraclePhase::HashLifeApprox)
    }
}
