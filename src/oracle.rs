use std::collections::HashMap;
use std::sync::OnceLock;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::classify::{Classification, ClassificationLimits};
use crate::engine::{SimulationBackend, SimulationSession};
use crate::generators::pattern_by_name;
use crate::hashlife::{
    GridExtractionPolicy, HASHLIFE_FULL_GRID_MAX_CHUNKS, HASHLIFE_FULL_GRID_MAX_POPULATION,
    HashLifeCheckpoint, HashLifeCheckpointKey,
};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};

mod patterns;
mod policy;
mod checkpoints;
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

type SeenStates = HashMap<NormalizedGridSignature, (u64, Cell)>;
type CheckpointStates = HashMap<HashLifeCheckpointKey, (u64, Cell)>;
type OracleStepCallback<'a> = &'a mut dyn FnMut(OracleStepPlan, OracleStateMetrics);

const ORACLE_HASHLIFE_MIN_JUMP_BUDGET: u64 = 1_024;
const ORACLE_HYBRID_SEGMENT_MAX_STEP: u64 = 64;
const ORACLE_SMALL_EXACT_POPULATION: usize = 64;
const ORACLE_MEDIUM_EXACT_POPULATION: usize = 256;
const ORACLE_LARGE_EXACT_POPULATION: usize = 2_048;
const ORACLE_TARGET_SUFFIX_MAX_POPULATION: usize = 10_000;
const ORACLE_SMALL_EXACT_SPAN: Coord = 32;
const ORACLE_MEDIUM_EXACT_SPAN: Coord = 64;
const ORACLE_LARGE_EXACT_SPAN: Coord = 256;
const ORACLE_TARGET_SUFFIX_MAX_SPAN: Coord = 4_096;
const ORACLE_SMALL_EXACT_WINDOW: u64 = 256;
const ORACLE_MEDIUM_EXACT_WINDOW: u64 = 128;
const ORACLE_LARGE_EXACT_WINDOW: u64 = 32;
const ORACLE_MIN_EXACT_WINDOW: u64 = 8;
const ORACLE_RUNTIME_EMITTER_EXACT_METRICS_MAX_GENERATION: u64 = 100_000;

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
}

#[derive(Clone, Copy, Debug)]
struct ConfirmedCycle {
    period: u64,
    first_seen: u64,
    delta: (Coord, Coord),
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

#[derive(Clone, Debug)]
struct CheckpointCycleCandidate {
    key: HashLifeCheckpointKey,
    generation: u64,
    origin: Cell,
    signature: NormalizedGridSignature,
}

#[derive(Debug)]
pub struct OracleSession<'a> {
    grid: Option<BitGrid>,
    generation: u64,
    seen: SeenStates,
    checkpoints: CheckpointStates,
    checkpoint_cycle_candidate: Option<CheckpointCycleCandidate>,
    confirmed_cycle: Option<ConfirmedCycle>,
    emitter_cycle_candidate: Option<EmitterCycleCandidate>,
    confirmed_emitter_cycle: Option<ConfirmedEmitterCycle>,
    simulation: &'a mut SimulationSession,
    exact_memo: Memo,
    phase: OraclePhase,
    last_step_span: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OraclePhase {
    ExactGrid,
    HashLifeApprox,
    ExactConfirmation,
}

impl<'a> OracleSession<'a> {
    pub fn new(
        grid: BitGrid,
        generation: u64,
        seen: SeenStates,
        simulation: &'a mut SimulationSession,
    ) -> Self {
        Self {
            grid: Some(grid),
            generation,
            seen,
            checkpoints: HashMap::new(),
            checkpoint_cycle_candidate: None,
            confirmed_cycle: None,
            emitter_cycle_candidate: None,
            confirmed_emitter_cycle: None,
            simulation,
            exact_memo: Memo::default(),
            phase: OraclePhase::ExactGrid,
            last_step_span: 0,
        }
    }

    pub fn from_hashlife_state(generation: u64, simulation: &'a mut SimulationSession) -> Self {
        Self {
            grid: None,
            generation,
            seen: HashMap::new(),
            checkpoints: HashMap::new(),
            checkpoint_cycle_candidate: None,
            confirmed_cycle: None,
            emitter_cycle_candidate: None,
            confirmed_emitter_cycle: None,
            simulation,
            exact_memo: Memo::default(),
            phase: OraclePhase::HashLifeApprox,
            last_step_span: 0,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn classify_continuation(
        mut self,
        generation_limit: u64,
        nominal_generation_limit: u64,
        limits: &ClassificationLimits,
    ) -> Classification {
        while self.generation <= generation_limit {
            if let Some(emitter_cycle) = self.confirmed_emitter_cycle.as_ref() {
                return Classification::LikelyInfinite {
                    reason: "emitter_cycle",
                    detected_at: emitter_cycle.first_seen,
                };
            }
            if self.is_hashlife_phase() {
                if let Some(classification) = self.classify_hashlife_checkpoint(limits) {
                    return classification;
                }
            } else if let Some(classification) = self.classify_exact_state(limits) {
                return classification;
            }
            let plan = self.plan_step(generation_limit, nominal_generation_limit);
            self.advance_by(plan.step_span);
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
            classification: Classification::LikelyInfinite {
                reason: "oracle_generation_limit",
                detected_at: target_generation,
            },
            final_generation: self.generation,
            grid: self.take_or_sample_grid(),
        }
    }

    fn classify_exact_state(&mut self, limits: &ClassificationLimits) -> Option<Classification> {
        let (signature, origin, population, bounds, is_empty) = {
            let grid = self.ensure_sampled_grid();
            let (signature, origin) = normalize(grid);
            (
                signature,
                origin,
                grid.population(),
                grid.bounds(),
                grid.is_empty(),
            )
        };
        if is_empty {
            return Some(Classification::DiesOut {
                at_generation: self.generation,
            });
        }

        if let Some(&(first_seen, first_origin)) = self.seen.get(&signature) {
            let period = self.generation - first_seen;
            let dx = origin.0 - first_origin.0;
            let dy = origin.1 - first_origin.1;
            self.confirmed_cycle = Some(ConfirmedCycle {
                period,
                first_seen,
                delta: (dx, dy),
            });
            return Some(if dx == 0 && dy == 0 {
                Classification::Repeats { period, first_seen }
            } else {
                Classification::Spaceship {
                    period,
                    first_seen,
                    delta: (dx, dy),
                    detected_at: self.generation,
                }
            });
        }

        if population > limits.max_population {
            return Some(Classification::LikelyInfinite {
                reason: "population_growth",
                detected_at: self.generation,
            });
        }

        if let Some(bounds) = bounds {
            let (width, height, _) = bounds_dimensions(bounds);
            if width > limits.max_bounding_box || height > limits.max_bounding_box {
                return Some(Classification::LikelyInfinite {
                    reason: "expanding_bounds",
                    detected_at: self.generation,
                });
            }
        }

        self.seen.insert(signature, (self.generation, origin));
        None
    }

}
