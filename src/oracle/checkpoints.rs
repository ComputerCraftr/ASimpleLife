use super::*;

impl<'a> OracleSession<'a> {
    pub(super) fn classify_hashlife_checkpoint(
        &mut self,
        limits: &ClassificationLimits,
    ) -> Option<Classification> {
        let population = self.simulation.hashlife_population().unwrap_or(0);
        if population == 0 {
            return Some(self.extinction_classification());
        }

        let checkpoint = self.simulation.hashlife_checkpoint().cloned()?;

        if checkpoint.population > limits.max_population as u64 {
            return Some(Classification::LikelyInfinite {
                reason: "population_growth",
                detected_at: self.generation,
            });
        }

        let (width, height, _) = bounds_dimensions(checkpoint.bounds);
        if width > limits.max_bounding_box || height > limits.max_bounding_box {
            return Some(Classification::LikelyInfinite {
                reason: "expanding_bounds",
                detected_at: self.generation,
            });
        }

        let checkpoint_key = checkpoint.signature.key();
        if self.checkpoints.contains_key(&checkpoint_key) {
            if let Some(cycle) =
                self.observe_repeated_hashlife_checkpoint(&checkpoint, checkpoint_key)
            {
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
                        detected_at: self.generation,
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

        self.checkpoints
            .insert(checkpoint_key, (self.generation, checkpoint.origin));
        None
    }

    pub(super) fn advance_checkpoint(
        &mut self,
        target_generation: u64,
    ) -> Option<OracleAdvanceOutcome> {
        if self.is_hashlife_phase() {
            let population = self.simulation.hashlife_population().unwrap_or(0);
            if population == 0 {
                return Some(OracleAdvanceOutcome {
                    classification: self.extinction_classification(),
                    final_generation: self.generation,
                    grid: self.take_or_sample_grid(),
                });
            }

            let checkpoint = self.simulation.hashlife_checkpoint().cloned()?;
            let checkpoint_key = checkpoint.signature.key();
            if self.checkpoints.contains_key(&checkpoint_key) {
                if let Some(cycle) =
                    self.observe_repeated_hashlife_checkpoint(&checkpoint, checkpoint_key)
                {
                    self.confirmed_cycle = Some(cycle);
                    return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
                }
                return None;
            }

            self.checkpoints
                .insert(checkpoint_key, (self.generation, checkpoint.origin));
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
            };
            self.confirmed_cycle = Some(cycle);
            return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
        }

        self.seen.insert(signature, (self.generation, origin));
        None
    }

    fn confirm_current_exact_signature(&mut self) -> Option<(NormalizedGridSignature, Cell)> {
        let previous_phase = self.phase;
        self.phase = OraclePhase::ExactConfirmation;
        let grid = self.ensure_sampled_grid();
        let (signature, origin) = normalize(grid);
        if previous_phase == OraclePhase::HashLifeApprox {
            self.grid = None;
        }
        self.phase = previous_phase;
        Some((signature, origin))
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
                .expect("emitter cycle candidate should exist");
            self.confirmed_emitter_cycle = Some(ConfirmedEmitterCycle {
                first_seen: candidate.first_seen,
                model,
            });
        }
    }

    fn observe_repeated_hashlife_checkpoint(
        &mut self,
        checkpoint: &HashLifeCheckpoint,
        checkpoint_key: HashLifeCheckpointKey,
    ) -> Option<ConfirmedCycle> {
        let (confirmed_signature, confirmed_origin) = self.confirm_current_exact_signature()?;
        if !checkpoint
            .signature
            .matches_normalized(&confirmed_signature)
            || checkpoint.origin != confirmed_origin
        {
            return None;
        }

        if let Some(cycle) = self.confirm_checkpoint_cycle_candidate(
            checkpoint_key,
            &confirmed_signature,
            confirmed_origin,
        ) {
            return Some(cycle);
        }

        self.checkpoint_cycle_candidate = Some(CheckpointCycleCandidate {
            key: checkpoint_key,
            generation: self.generation,
            origin: confirmed_origin,
            signature: confirmed_signature,
        });
        None
    }

    fn confirm_checkpoint_cycle_candidate(
        &self,
        checkpoint_key: HashLifeCheckpointKey,
        confirmed_signature: &NormalizedGridSignature,
        confirmed_origin: Cell,
    ) -> Option<ConfirmedCycle> {
        let candidate = self.checkpoint_cycle_candidate.as_ref()?;
        if candidate.key != checkpoint_key || candidate.signature != *confirmed_signature {
            return None;
        }

        Some(ConfirmedCycle {
            period: self.generation - candidate.generation,
            first_seen: candidate.generation,
            delta: (
                confirmed_origin.0 - candidate.origin.0,
                confirmed_origin.1 - candidate.origin.1,
            ),
        })
    }
}
