use super::*;

impl<'a> OracleSession<'a> {
    pub(super) fn classify_hashlife_checkpoint(&mut self) -> Option<Classification> {
        let population = self.simulation.hashlife_population_count()?;
        if population.is_zero() {
            return Some(self.extinction_classification());
        }

        let recurrence = self.observe_recurrence();
        if let Some(classification) = self.classification_for_recurrence(recurrence) {
            return Some(classification);
        }

        self.try_confirm_emitter_cycle();
        self.confirmed_emitter_cycle
            .as_ref()
            .map(|emitter_cycle| Classification::LikelyInfinite {
                reason: "emitter_cycle",
                detected_at: emitter_cycle.first_seen,
            })
    }

    pub(super) fn advance_checkpoint(
        &mut self,
        target_generation: u64,
    ) -> Option<OracleAdvanceOutcome> {
        if self.is_hashlife_phase() {
            let population = self.simulation.hashlife_population_count()?;
            if population.is_zero() {
                return Some(OracleAdvanceOutcome {
                    classification: self.extinction_classification(),
                    final_generation: self.generation,
                    grid: self.take_or_sample_grid(),
                });
            }

            let recurrence = self.observe_recurrence();
            if self.classification_for_recurrence(recurrence).is_some() {
                let cycle = self.confirmed_cycle?;
                return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
            }

            self.try_confirm_emitter_cycle();
            return None;
        }

        if self.ensure_sampled_grid().is_empty() {
            return Some(OracleAdvanceOutcome {
                classification: Classification::DiesOut {
                    at_generation: self.generation,
                },
                final_generation: self.generation,
                grid: self.take_or_sample_grid(),
            });
        }

        let recurrence = self.observe_recurrence();
        if self.classification_for_recurrence(recurrence).is_some() {
            let cycle = self.confirmed_cycle?;
            return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
        }
        None
    }

    pub(super) fn try_confirm_emitter_cycle(&mut self) {
        if self.confirmed_emitter_cycle.is_some() || self.generation != 0 {
            return;
        }
        let metrics = self.current_state_shape();
        if metrics.population < 30 || metrics.bounds_span < 32 {
            return;
        }
        self.emitter_cycle_candidate = Some(EmitterCycleCandidate {
            first_seen: self.generation,
        });
        let Some(grid) = self.grid.as_ref() else {
            return;
        };
        if let Some(model) = build_emitter_macro_model(Some(grid), self.generation) {
            let Some(candidate) = self.emitter_cycle_candidate else {
                return;
            };
            self.confirmed_emitter_cycle = Some(ConfirmedEmitterCycle {
                first_seen: candidate.first_seen,
                model,
            });
        }
    }
}
