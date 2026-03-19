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

    fn advance_by(&mut self, step_span: u64) {
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

    fn extinction_classification(&self) -> Classification {
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

    fn land_confirmed_cycle_to_target(
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

    fn land_confirmed_emitter_cycle_to_target(
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

    fn classify_hashlife_checkpoint(
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

    fn advance_checkpoint(&mut self, target_generation: u64) -> Option<OracleAdvanceOutcome> {
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

    fn try_confirm_emitter_cycle(&mut self) {
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
                let outcome = self.land_confirmed_cycle_to_target(target_generation, cycle);
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
                        if self.checkpoints.contains_key(&checkpoint_key) {
                            if let Some(cycle) = self
                                .observe_repeated_hashlife_checkpoint(&checkpoint, checkpoint_key)
                            {
                                self.confirmed_cycle = Some(cycle);
                                continue;
                            }
                        }
                        self.checkpoints
                            .insert(checkpoint_key, (self.generation, checkpoint.origin));
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
