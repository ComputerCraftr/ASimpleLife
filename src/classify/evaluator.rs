//! Shared evidence decisions; adapters own stepping and capture resources.
use super::*;

pub(crate) struct EvidenceEvaluator {
    pub(crate) recurrence: ExactRecurrenceTracker,
    limits: ClassificationLimits,
    generation_limit: u64,
    metrics: [(usize, Coord, Coord, Coord, Coord, Coord); PERSISTENT_EXPANSION_WINDOW * 2 + 1],
    samples: usize,
}

impl EvidenceEvaluator {
    pub(crate) fn new(
        recurrence: ExactRecurrenceTracker,
        limits: &ClassificationLimits,
        generation_limit: u64,
    ) -> Self {
        Self {
            recurrence,
            limits: limits.clone(),
            generation_limit,
            metrics: [(0, 0, 0, 0, 0, 0); PERSISTENT_EXPANSION_WINDOW * 2 + 1],
            samples: 0,
        }
    }

    pub(crate) fn observe(
        &mut self,
        generation: u64,
        observation: Result<Observation, crate::recurrence::RecurrenceUnavailable>,
        empty: bool,
        grid: Option<&BitGrid>,
    ) -> Option<ClassificationReport> {
        if empty {
            return Some(ClassificationReport::from_legacy(
                &Classification::DiesOut {
                    at_generation: generation,
                },
            ));
        }
        if let ObserveOutcome::Repeated(certificate) = self.recurrence.observe_result(observation) {
            let (dx, dy) = certificate.delta();
            if let Some(delta) = Coord::try_from(dx).ok().zip(Coord::try_from(dy).ok()) {
                let result = if delta == (0, 0) {
                    Classification::Repeats {
                        period: certificate.period(),
                        first_seen: certificate.first_seen(),
                    }
                } else {
                    Classification::Spaceship {
                        period: certificate.period(),
                        first_seen: certificate.first_seen(),
                        delta,
                        detected_at: generation,
                    }
                };
                return Some(ClassificationReport::from_legacy(&result));
            }
        }
        if let Some(grid) = grid {
            self.generation_limit = self.generation_limit.max(effective_generation_limit(
                &self.limits,
                grid.population(),
                grid.bounds(),
            ));
            if let Some((x0, y0, x1, y1)) = grid.bounds() {
                let (_, _, span) = bounds_dimensions((x0, y0, x1, y1));
                if self.samples == self.metrics.len() {
                    self.metrics.rotate_left(1);
                    self.samples -= 1;
                }
                self.metrics[self.samples] = (grid.population(), x0, x1, y0, y1, span);
                self.samples += 1;
                if let Some(result) = detect_persistent_expansion(
                    generation,
                    &self.metrics[..self.samples],
                    grid,
                    &self.limits,
                ) {
                    return Some(ClassificationReport::from_legacy(&result));
                }
            }
        } else {
            self.samples = 0;
        }
        if generation >= self.generation_limit {
            if let Some(next) = settling_extension_limit(
                &self.limits,
                self.generation_limit,
                &self.metrics[..self.samples],
            ) {
                self.generation_limit = next;
            }
            if generation >= self.generation_limit {
                return Some(ClassificationReport::from_legacy(
                    &Classification::Unknown {
                        simulated: generation,
                    },
                ));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    #[test]
    fn horizon_reports_the_last_observed_state_without_an_extra_step() {
        let limits = ClassificationLimits { max_generations: 0 };
        let mut evidence =
            EvidenceEvaluator::new(ExactRecurrenceTracker::new(Lineage::fresh()), &limits, 0);
        let report = evidence
            .observe(
                0,
                Err(crate::recurrence::RecurrenceUnavailable::WitnessLimit),
                false,
                None,
            )
            .or_invariant("horizon report");
        assert_eq!(report.observed_through, 0);
        assert_eq!(report.outcome, ClassificationOutcome::Unresolved);
    }
}
