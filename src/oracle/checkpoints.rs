use super::*;
use crate::RequiredExt;

impl<'a> OracleSession<'a> {
    pub(super) fn prepare_hashlife_checkpoint(
        &mut self,
        checkpoint: &HashLifeStateCheckpoint,
    ) -> bool {
        let epoch_changed = self
            .checkpoints
            .keys()
            .next()
            .is_some_and(|retained| !retained.same_epoch(checkpoint.identity));
        if epoch_changed {
            self.checkpoints.clear();
        }
        epoch_changed
    }

    pub(super) fn record_hashlife_checkpoint(&mut self, checkpoint: HashLifeStateCheckpoint) {
        self.prepare_hashlife_checkpoint(&checkpoint);
        while self.checkpoints.len() >= ORACLE_MAX_HASHLIFE_CHECKPOINTS {
            let oldest_generation = self
                .checkpoints
                .values()
                .map(|(generation, _)| *generation)
                .min()
                .or_invariant("nonempty checkpoint set should have an oldest generation");
            self.checkpoints
                .retain(|_, (generation, _)| *generation != oldest_generation);
        }
        self.checkpoints.insert(
            checkpoint.identity,
            (checkpoint.generation, checkpoint.origin),
        );
    }

    pub(super) fn classify_hashlife_checkpoint(&mut self) -> Option<Classification> {
        let population = self.simulation.hashlife_population_count()?;
        if population.is_zero() {
            return Some(self.extinction_classification());
        }

        let checkpoint = self.simulation.hashlife_checkpoint().cloned()?;
        self.prepare_hashlife_checkpoint(&checkpoint);

        if self.checkpoints.contains_key(&checkpoint.identity) {
            if let Some(cycle) = self.observe_repeated_hashlife_checkpoint(&checkpoint) {
                self.confirmed_cycle = Some(cycle);
                return Some(if cycle.delta == (0, 0) {
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
                });
            }
            return None;
        }

        self.try_confirm_emitter_cycle();
        if let Some(emitter_cycle) = self.confirmed_emitter_cycle.as_ref() {
            return Some(Classification::LikelyInfinite {
                reason: "emitter_cycle",
                detected_at: emitter_cycle.first_seen,
            });
        }

        self.record_hashlife_checkpoint(checkpoint);
        None
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

            let checkpoint = self.simulation.hashlife_checkpoint().cloned()?;
            self.prepare_hashlife_checkpoint(&checkpoint);
            if self.checkpoints.contains_key(&checkpoint.identity) {
                if let Some(cycle) = self.observe_repeated_hashlife_checkpoint(&checkpoint) {
                    self.confirmed_cycle = Some(cycle);
                    return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
                }
                return None;
            }

            self.record_hashlife_checkpoint(checkpoint);
            return None;
        }

        let (signature, origin, is_empty) = {
            let grid = self.ensure_sampled_grid();
            let (signature, origin) = normalize(grid);
            (signature, origin, grid.is_empty())
        };
        if is_empty {
            return Some(OracleAdvanceOutcome {
                classification: Classification::DiesOut {
                    at_generation: self.generation,
                },
                final_generation: self.generation,
                grid: self.take_or_sample_grid(),
            });
        }

        if let Some(&(first_seen, first_origin)) = self.seen.get(&signature) {
            let period = self.generation - first_seen;
            let dx = origin.0 - first_origin.0;
            let dy = origin.1 - first_origin.1;
            let cycle = ConfirmedCycle {
                period,
                first_seen,
                delta: (dx, dy),
                detected_at: self.generation,
            };
            self.confirmed_cycle = Some(cycle);
            return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
        }

        self.seen.insert(signature, (self.generation, origin));
        None
    }

    pub(super) fn try_confirm_emitter_cycle(&mut self) {
        if self.confirmed_emitter_cycle.is_some() {
            return;
        }
        if self.generation != 0 {
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
            let candidate = self
                .emitter_cycle_candidate
                .or_invariant("emitter cycle candidate should exist");
            self.confirmed_emitter_cycle = Some(ConfirmedEmitterCycle {
                first_seen: candidate.first_seen,
                model,
            });
        }
    }

    pub(super) fn observe_repeated_hashlife_checkpoint(
        &self,
        checkpoint: &HashLifeStateCheckpoint,
    ) -> Option<ConfirmedCycle> {
        let &(first_seen, first_origin) = self.checkpoints.get(&checkpoint.identity)?;

        let observed = ConfirmedCycle {
            period: self.generation - first_seen,
            first_seen,
            delta: (
                checkpoint.origin.0 - first_origin.0,
                checkpoint.origin.1 - first_origin.1,
            ),
            detected_at: self.generation,
        };
        Some(observed)
    }
}
