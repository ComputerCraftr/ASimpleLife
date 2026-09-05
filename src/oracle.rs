use std::sync::OnceLock;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::classify::{Classification, ClassificationCheckpoint};
use crate::engine::{SimulationBackend, SimulationSession};
use crate::generators::pattern_by_name;
use crate::hashlife::{
    GridExtractionPolicy, HASHLIFE_FULL_GRID_MAX_CHUNKS, HASHLIFE_FULL_GRID_MAX_POPULATION,
    HashLifeAdvanceError, HashLifeConversionError, PopulationCount,
};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};
use crate::recurrence::{ExactRecurrenceTracker, Observation, ObserveOutcome, PeriodicCertificate};

mod checkpoints;
mod constellation;
mod patterns;
mod policy;
mod runtime;
mod session;

use patterns::{
    build_emitter_macro_model, emitter_runtime_bounds_span, emitter_runtime_population,
};
use policy::{
    confirmation_full_grid_policy, continuation_step_span, cycle_probe_prefix_window,
    hybrid_target_prefix_generations, largest_power_of_two_leq, max_hashlife_safe_jump_from_span,
    target_exact_suffix_window,
};

type OracleStepCallback<'a> = &'a mut dyn FnMut(OracleStepPlan, OracleStateMetrics);

const ORACLE_HASHLIFE_MIN_JUMP_BUDGET: u64 = 1_024;
const ORACLE_HYBRID_SEGMENT_MAX_STEP: u64 = 64;
const ORACLE_HASHLIFE_CHECKPOINT_PROBE_COUNT: usize = 8;
const ORACLE_HASHLIFE_CHECKPOINT_PROBE_STEP: u64 = 1;
const ORACLE_SMALL_EXACT_POPULATION: usize = 64;
const ORACLE_MEDIUM_EXACT_POPULATION: usize = 256;
const ORACLE_LARGE_EXACT_POPULATION: usize = 2_048;
const ORACLE_TARGET_SUFFIX_MAX_POPULATION: usize = 10_000;
const ORACLE_SMALL_EXACT_SPAN: Coord = 32;
const ORACLE_MEDIUM_EXACT_SPAN: Coord = 64;
const ORACLE_LARGE_EXACT_SPAN: Coord = 256;
const ORACLE_TARGET_SUFFIX_MAX_SPAN: Coord = 4_096;
const ORACLE_SMALL_EXACT_WINDOW: u64 = 256;
const ORACLE_METHUSELAH_EXACT_WINDOW: u64 = 2_048;
const ORACLE_MEDIUM_EXACT_WINDOW: u64 = 128;
const ORACLE_LARGE_EXACT_WINDOW: u64 = 32;
const ORACLE_MIN_EXACT_WINDOW: u64 = 8;
const ORACLE_RUNTIME_EMITTER_EXACT_METRICS_MAX_GENERATION: u64 = 100_000;
const ORACLE_CONSTELLATION_MAX_POPULATION: u64 = 4_096;
const ORACLE_CONSTELLATION_MAX_CHUNKS: usize = 4_096;
const ORACLE_CONSTELLATION_MAX_SPAN: Coord = 1_000_000;

fn bounds_dimensions(bounds: (Coord, Coord, Coord, Coord)) -> (Coord, Coord, Coord) {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;
    (width, height, width.max(height))
}

#[derive(Clone, Copy, Debug)]
pub struct OracleStepPlan {
    pub generation: u64,
    pub step_span: u64,
    pub backend: SimulationBackend,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OracleStateMetrics {
    pub population: usize,
    pub bounds_span: Coord,
}

#[derive(Clone, Debug)]
pub struct OracleAdvanceOutcome {
    pub classification: Classification,
    pub final_generation: u64,
    pub grid: BitGrid,
}

#[derive(Clone, Debug)]
pub struct OracleRuntimeOutcome {
    pub classification: Classification,
    pub final_generation: u64,
    pub population: usize,
    pub bounds_span: Coord,
    pub state: OracleRuntimeState,
    pub failure: Option<HashLifeAdvanceError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleRuntimeState {
    RetainedHashLife,
    Modeled,
}

#[derive(Clone, Copy, Debug)]
struct ConfirmedCycle {
    certificate: PeriodicCertificate,
}

#[derive(Clone, Debug)]
struct EmitterMacroModel {
    baseline_generation: u64,
    baseline_glider_count: u64,
    core_population_by_phase: [usize; 30],
    core_bounds_by_phase: [(Coord, Coord, Coord, Coord); 30],
    oldest_glider_origin: Cell,
    oldest_glider_phase: u8,
}

#[derive(Clone, Copy, Debug)]
struct EmitterCycleCandidate {
    first_seen: u64,
}

#[derive(Clone, Debug)]
struct ConfirmedEmitterCycle {
    first_seen: u64,
    model: EmitterMacroModel,
}

#[derive(Debug)]
pub struct OracleSession<'a> {
    grid: Option<BitGrid>,
    generation: u64,
    recurrence: ExactRecurrenceTracker,
    confirmed_cycle: Option<ConfirmedCycle>,
    emitter_cycle_candidate: Option<EmitterCycleCandidate>,
    confirmed_emitter_cycle: Option<ConfirmedEmitterCycle>,
    simulation: &'a mut SimulationSession,
    exact_memo: Memo,
    phase: OraclePhase,
    last_step_span: u64,
    advance_failure: Option<HashLifeAdvanceError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OraclePhase {
    ExactGrid,
    HashLifeApprox,
    ExactConfirmation,
}

impl<'a> OracleSession<'a> {
    pub fn new(grid: BitGrid, generation: u64, simulation: &'a mut SimulationSession) -> Self {
        let lineage = simulation.recurrence_lineage();
        Self {
            grid: Some(grid),
            generation,
            recurrence: ExactRecurrenceTracker::new(lineage),
            confirmed_cycle: None,
            emitter_cycle_candidate: None,
            confirmed_emitter_cycle: None,
            simulation,
            exact_memo: Memo::default(),
            phase: OraclePhase::ExactGrid,
            last_step_span: 0,
            advance_failure: None,
        }
    }

    pub fn from_hashlife_state(
        generation: u64,
        simulation: &'a mut SimulationSession,
    ) -> Result<Self, &'static str> {
        if !simulation.hashlife_loaded() {
            return Err("HashLife-backed oracle requires a loaded simulation session");
        }
        if generation != simulation.hashlife_generation() {
            return Err("oracle and HashLife session generations must be aligned");
        }
        Ok(Self {
            grid: None,
            generation,
            recurrence: ExactRecurrenceTracker::new(simulation.recurrence_lineage()),
            confirmed_cycle: None,
            emitter_cycle_candidate: None,
            confirmed_emitter_cycle: None,
            simulation,
            exact_memo: Memo::default(),
            phase: OraclePhase::HashLifeApprox,
            last_step_span: 0,
            advance_failure: None,
        })
    }

    pub(crate) fn from_classification_checkpoint(
        checkpoint: ClassificationCheckpoint,
        simulation: &'a mut SimulationSession,
    ) -> Self {
        Self {
            grid: Some(checkpoint.grid),
            generation: checkpoint.generation,
            recurrence: checkpoint.recurrence,
            confirmed_cycle: None,
            emitter_cycle_candidate: None,
            confirmed_emitter_cycle: None,
            simulation,
            exact_memo: Memo::default(),
            phase: OraclePhase::ExactGrid,
            last_step_span: 0,
            advance_failure: None,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn classify_continuation(
        mut self,
        generation_limit: u64,
        nominal_generation_limit: u64,
    ) -> Classification {
        while self.generation <= generation_limit {
            if let Some(emitter_cycle) = self.confirmed_emitter_cycle.as_ref() {
                return Classification::LikelyInfinite {
                    reason: "emitter_cycle",
                    detected_at: emitter_cycle.first_seen,
                };
            }
            if self.is_hashlife_phase() {
                if let Some(classification) = self.classify_hashlife_checkpoint() {
                    return classification;
                }
            } else if let Some(classification) = self.classify_exact_state() {
                return classification;
            }
            let plan = self.plan_step(generation_limit, nominal_generation_limit);
            if !self.advance_by(plan.step_span) {
                break;
            }
        }

        Classification::Unknown {
            simulated: generation_limit,
        }
    }

    pub fn advance_to_target(
        mut self,
        target_generation: u64,
        mut on_step: Option<OracleStepCallback<'_>>,
    ) -> OracleAdvanceOutcome {
        let cycle_probe_limit = self
            .generation
            .saturating_add(cycle_probe_prefix_window(
                self.current_state_shape().population,
                self.current_state_shape().bounds_span,
            ))
            .min(target_generation);
        let mut next_checkpoint_generation = cycle_probe_limit.saturating_add(1);
        let mut checkpoint_stride = 1_u64;
        while self.generation <= target_generation {
            if let Some(cycle) = self.confirmed_cycle {
                return self.land_confirmed_cycle_to_target(target_generation, cycle);
            }
            if let Some(emitter_cycle) = self.confirmed_emitter_cycle.clone() {
                return self
                    .land_confirmed_emitter_cycle_to_target(target_generation, &emitter_cycle);
            }

            if self.should_sample_state(
                target_generation,
                cycle_probe_limit,
                next_checkpoint_generation,
            ) {
                if let Some(outcome) = self.advance_checkpoint(target_generation) {
                    return outcome;
                }
                if self.generation >= cycle_probe_limit && self.is_hashlife_phase() {
                    checkpoint_stride = checkpoint_stride.saturating_mul(2).min(1 << 20);
                    next_checkpoint_generation = self
                        .generation
                        .saturating_add(checkpoint_stride)
                        .min(target_generation);
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

        OracleAdvanceOutcome {
            classification: Classification::Unknown {
                simulated: target_generation,
            },
            final_generation: self.generation,
            grid: self.take_or_sample_grid(),
        }
    }

    fn classify_exact_state(&mut self) -> Option<Classification> {
        let is_empty = {
            let grid = self.ensure_sampled_grid();
            grid.is_empty()
        };
        if is_empty {
            return Some(Classification::DiesOut {
                at_generation: self.generation,
            });
        }

        let outcome = self.observe_recurrence();
        self.classification_for_recurrence(outcome)
    }

    fn observe_recurrence(&mut self) -> ObserveOutcome {
        let simulation_lineage = self.simulation.recurrence_lineage();
        if self.is_hashlife_phase() && self.recurrence.lineage() != simulation_lineage {
            self.recurrence.reset(simulation_lineage);
        }
        let lineage = if self.is_hashlife_phase() {
            simulation_lineage
        } else {
            self.recurrence.lineage()
        };
        let observation = if self.is_hashlife_phase() {
            self.simulation.recurrence_observation()
        } else {
            Observation::from_grid(lineage, self.generation, self.ensure_sampled_grid())
        };
        self.recurrence.observe_result(observation)
    }

    fn classification_for_recurrence(&mut self, outcome: ObserveOutcome) -> Option<Classification> {
        let ObserveOutcome::Repeated(certificate) = outcome else {
            return None;
        };
        let delta = certificate.delta();
        let delta = (
            Coord::try_from(delta.0).ok()?,
            Coord::try_from(delta.1).ok()?,
        );
        self.confirmed_cycle = Some(ConfirmedCycle { certificate });
        Some(if delta == (0, 0) {
            Classification::Repeats {
                period: certificate.period(),
                first_seen: certificate.first_seen(),
            }
        } else {
            Classification::Spaceship {
                period: certificate.period(),
                first_seen: certificate.first_seen(),
                delta,
                detected_at: certificate.detected_at(),
            }
        })
    }
}
