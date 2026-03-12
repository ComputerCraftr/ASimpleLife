use std::collections::HashMap;
use std::sync::OnceLock;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::classify::{Classification, ClassificationLimits};
use crate::engine::{SimulationBackend, SimulationSession};
use crate::generators::pattern_by_name;
use crate::hashlife::{GridExtractionPolicy, HashLifeCheckpointKey};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};

type SeenStates = HashMap<NormalizedGridSignature, (u64, Cell)>;
type CheckpointStates = HashMap<HashLifeCheckpointKey, (u64, Cell)>;
type OracleStepCallback<'a> = &'a mut dyn FnMut(OracleStepPlan, OracleStateMetrics);

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
            self.grid = Some(translate_grid(&current_grid, dx, dy));
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

    fn plan_step(
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

    fn plan_target_step(&mut self, remaining: u64) -> OracleStepPlan {
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

    fn planned_backend_for_shape(&mut self, step_span: u64) -> SimulationBackend {
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

    fn current_state_shape(&mut self) -> OracleStateMetrics {
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

    fn should_sample_state(
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

    fn ensure_sampled_grid(&mut self) -> &BitGrid {
        if self.grid.is_none() {
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

    fn take_or_sample_grid(&mut self) -> BitGrid {
        if self.grid.is_none() {
            self.ensure_sampled_grid();
        }
        self.grid
            .take()
            .expect("oracle should have a final sampled grid")
    }

    fn is_hashlife_phase(&self) -> bool {
        matches!(self.phase, OraclePhase::HashLifeApprox)
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
            let (confirmed_signature, confirmed_origin) = self.confirm_current_exact_signature()?;
            if !checkpoint
                .signature
                .matches_normalized(&confirmed_signature)
                || checkpoint.origin != confirmed_origin
            {
                return None;
            }
            if let Some(candidate) = self.checkpoint_cycle_candidate.as_ref()
                && candidate.key == checkpoint_key
                && candidate.signature == confirmed_signature
            {
                let period = self.generation - candidate.generation;
                let dx = confirmed_origin.0 - candidate.origin.0;
                let dy = confirmed_origin.1 - candidate.origin.1;
                self.confirmed_cycle = Some(ConfirmedCycle {
                    period,
                    first_seen: candidate.generation,
                    delta: (dx, dy),
                });
                return Some(if dx == 0 && dy == 0 {
                    Classification::Repeats {
                        period,
                        first_seen: candidate.generation,
                    }
                } else {
                    Classification::Spaceship {
                        period,
                        first_seen: candidate.generation,
                        delta: (dx, dy),
                        detected_at: self.generation,
                    }
                });
            }
            self.checkpoint_cycle_candidate = Some(CheckpointCycleCandidate {
                key: checkpoint_key,
                generation: self.generation,
                origin: confirmed_origin,
                signature: confirmed_signature,
            });
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
                let (confirmed_signature, confirmed_origin) =
                    self.confirm_current_exact_signature()?;
                if !checkpoint
                    .signature
                    .matches_normalized(&confirmed_signature)
                    || checkpoint.origin != confirmed_origin
                {
                    return None;
                }
                if let Some(candidate) = self.checkpoint_cycle_candidate.as_ref()
                    && candidate.key == checkpoint_key
                    && candidate.signature == confirmed_signature
                {
                    let period = self.generation - candidate.generation;
                    let dx = confirmed_origin.0 - candidate.origin.0;
                    let dy = confirmed_origin.1 - candidate.origin.1;
                    let cycle = ConfirmedCycle {
                        period,
                        first_seen: candidate.generation,
                        delta: (dx, dy),
                    };
                    self.confirmed_cycle = Some(cycle);
                    return Some(self.land_confirmed_cycle_to_target(target_generation, cycle));
                }
                self.checkpoint_cycle_candidate = Some(CheckpointCycleCandidate {
                    key: checkpoint_key,
                    generation: self.generation,
                    origin: confirmed_origin,
                    signature: confirmed_signature,
                });
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
                            let (confirmed_signature, confirmed_origin) = self
                                .confirm_current_exact_signature()
                                .expect("runtime repeat confirmation should succeed");
                            if checkpoint
                                .signature
                                .matches_normalized(&confirmed_signature)
                                && checkpoint.origin == confirmed_origin
                            {
                                if let Some(candidate) = self.checkpoint_cycle_candidate.as_ref()
                                    && candidate.key == checkpoint_key
                                    && candidate.signature == confirmed_signature
                                {
                                    let period = self.generation - candidate.generation;
                                    let dx = confirmed_origin.0 - candidate.origin.0;
                                    let dy = confirmed_origin.1 - candidate.origin.1;
                                    self.confirmed_cycle = Some(ConfirmedCycle {
                                        period,
                                        first_seen: candidate.generation,
                                        delta: (dx, dy),
                                    });
                                    continue;
                                }
                                self.checkpoint_cycle_candidate = Some(CheckpointCycleCandidate {
                                    key: checkpoint_key,
                                    generation: self.generation,
                                    origin: confirmed_origin,
                                    signature: confirmed_signature,
                                });
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
}

fn confirmation_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: 250_000,
        max_chunks: 100_000,
        max_bounds_span: Coord::MAX,
    }
}

fn continuation_step_span(
    current: OracleStateMetrics,
    generation: u64,
    generation_limit: u64,
    nominal_generation_limit: u64,
    simulation: &mut SimulationSession,
    hashlife_active: bool,
) -> u64 {
    if generation < nominal_generation_limit {
        return 1;
    }

    let remaining = generation_limit.saturating_sub(generation);
    if remaining <= 1 {
        return 1;
    }

    if hashlife_active && current.population > 0 {
        let safe_hashlife_jump = max_hashlife_safe_jump_from_span(current.bounds_span);
        if safe_hashlife_jump >= 2 {
            return largest_power_of_two_leq(remaining.min(safe_hashlife_jump));
        }
    }

    let exact_tail = oracle_exact_tail_window(current.population, current.bounds_span);
    if remaining <= exact_tail {
        return 1;
    }
    let jump_budget = remaining - exact_tail;
    let safe_hashlife_jump = max_hashlife_safe_jump_from_span(current.bounds_span);

    if jump_budget >= 1_024 && safe_hashlife_jump >= 2 {
        return largest_power_of_two_leq(jump_budget.min(safe_hashlife_jump));
    }

    match simulation.planned_backend_from_session_metrics(
        current.population,
        current.bounds_span,
        remaining,
    ) {
        SimulationBackend::SimdChunk => 1,
        SimulationBackend::HybridSegmented => jump_budget.min(64),
        SimulationBackend::HashLife => {
            if safe_hashlife_jump >= 2 {
                largest_power_of_two_leq(jump_budget.min(safe_hashlife_jump))
            } else {
                1
            }
        }
    }
}

fn oracle_exact_tail_window(population: usize, span: Coord) -> u64 {
    if population <= 64 && span <= 32 {
        256
    } else if population <= 256 && span <= 64 {
        128
    } else {
        64
    }
}

fn target_exact_suffix_window(population: usize, span: Coord) -> u64 {
    if population > 10_000 || span > 4_096 {
        0
    } else if population <= 64 && span <= 32 {
        256
    } else if population <= 256 && span <= 64 {
        128
    } else if population <= 2_048 && span <= 256 {
        32
    } else {
        8
    }
}

fn cycle_probe_prefix_window(population: usize, span: Coord) -> u64 {
    if population <= 64 && span <= 32 {
        64
    } else if population <= 256 && span <= 64 {
        16
    } else {
        0
    }
}

fn largest_power_of_two_leq(value: u64) -> u64 {
    debug_assert!(value > 0);
    1_u64 << (63 - value.leading_zeros())
}

fn max_hashlife_safe_jump_from_span(span: Coord) -> u64 {
    if span <= 0 {
        return 1;
    }
    let raw_max_jump = (((Coord::MAX as i128) - (2 * span as i128) - 8) / 4).max(1) as u64;
    let mut jump = 1_u64 << (63 - raw_max_jump.leading_zeros());
    while jump > 1 && required_root_size_for_jump(span as u64, jump) > Coord::MAX as u64 {
        jump >>= 1;
    }
    jump
}

fn hybrid_target_prefix_generations(population: usize, generations: u64) -> u64 {
    let prefix = (population as u64)
        .saturating_div(32)
        .clamp(16, 64)
        .min(generations);
    prefix.max(1).min(generations)
}

fn required_root_size_for_jump(span: u64, jump: u64) -> u64 {
    (2 * span + 4 * (jump + 2))
        .max((4 * jump) + 4)
        .max(4)
        .next_power_of_two()
}

fn translate_grid(grid: &BitGrid, dx: Coord, dy: Coord) -> BitGrid {
    let mut cells = Vec::with_capacity(grid.population());
    cells.extend(grid.live_cells().into_iter().map(|(x, y)| {
        (
            x.checked_add(dx).expect("grid translation x overflow"),
            y.checked_add(dy).expect("grid translation y overflow"),
        )
    }));
    BitGrid::from_cells(&cells)
}

fn build_emitter_macro_model(seed: Option<&BitGrid>, generation: u64) -> Option<EmitterMacroModel> {
    if generation != 0 {
        return None;
    }
    let seed = seed?;
    let gosper = pattern_by_name("gosper_glider_gun")?;
    if normalize(seed).0 != normalize(&gosper).0 {
        return None;
    }

    const PERIOD: u64 = 30;
    const BASELINE_GENERATION: u64 = 300;
    let baseline_grid = advance_exact_steps(seed, BASELINE_GENERATION);
    let next_grid = advance_exact_steps(seed, BASELINE_GENERATION + PERIOD);
    let baseline_state = extract_gosper_state(&baseline_grid)?;
    let next_state = extract_gosper_state(&next_grid)?;
    if normalize(&baseline_state.core).0 != normalize(&next_state.core).0 {
        return None;
    }
    if next_state.gliders.len() != baseline_state.gliders.len() + 1 {
        return None;
    }

    let mut core_population_by_phase = [0_usize; 30];
    let mut core_bounds_by_phase = [(0_i64, 0_i64, 0_i64, 0_i64); 30];
    let mut core_phase = baseline_state.core.clone();
    for residual in 0..30 {
        core_population_by_phase[residual] = core_phase.population();
        core_bounds_by_phase[residual] = core_phase.bounds().unwrap_or((0, 0, 0, 0));
        core_phase = step_grid_with_changes_and_memo(&core_phase, &mut Memo::default()).0;
    }

    let oldest_glider = baseline_state
        .gliders
        .iter()
        .max_by_key(|glider| (glider.origin.0, glider.origin.1))
        .copied()?;

    Some(EmitterMacroModel {
        baseline_generation: BASELINE_GENERATION,
        baseline_glider_count: u64::try_from(baseline_state.gliders.len())
            .expect("gosper glider count exceeded u64"),
        core_population_by_phase,
        core_bounds_by_phase,
        oldest_glider_origin: oldest_glider.origin,
        oldest_glider_phase: oldest_glider.phase,
    })
}

fn emitter_runtime_population(model: &EmitterMacroModel, target_generation: u64) -> usize {
    let delta = target_generation.saturating_sub(model.baseline_generation);
    let emitted = delta / 30;
    let residual = usize::try_from(delta % 30).expect("gosper residual exceeded usize");
    model.core_population_by_phase[residual]
        + usize::try_from((model.baseline_glider_count + emitted) * 5)
            .expect("gosper runtime population exceeded usize")
}

fn emitter_runtime_bounds_span(model: &EmitterMacroModel, target_generation: u64) -> Coord {
    let delta = target_generation.saturating_sub(model.baseline_generation);
    let residual = delta % 120;
    let cycles = delta / 120;
    let phase = usize::from(model.oldest_glider_phase);
    let table = glider_runtime_table();
    let glider_bounds =
        table[phase][usize::try_from(residual).expect("glider residual exceeded usize")];
    let cycle_shift = Coord::try_from(cycles)
        .expect("gosper cycle count exceeded Coord")
        .checked_mul(30)
        .expect("gosper cycle shift overflow");
    let glider_max_x = model
        .oldest_glider_origin
        .0
        .checked_add(cycle_shift)
        .and_then(|origin_x| origin_x.checked_add(glider_bounds.2))
        .expect("gosper glider max x overflow");
    let glider_max_y = model
        .oldest_glider_origin
        .1
        .checked_add(cycle_shift)
        .and_then(|origin_y| origin_y.checked_add(glider_bounds.3))
        .expect("gosper glider max y overflow");
    let residual_index = usize::try_from(delta % 30).expect("gosper residual exceeded usize");
    let core_bounds = model.core_bounds_by_phase[residual_index];
    let width = glider_max_x
        .max(core_bounds.2)
        .checked_sub(core_bounds.0)
        .and_then(|span| span.checked_add(1))
        .expect("gosper runtime width overflow");
    let height = glider_max_y
        .max(core_bounds.3)
        .checked_sub(core_bounds.1)
        .and_then(|span| span.checked_add(1))
        .expect("gosper runtime height overflow");
    width.max(height)
}

#[derive(Clone, Copy, Debug)]
struct GosperGliderInstance {
    origin: Cell,
    phase: u8,
}

#[derive(Clone, Debug)]
struct GosperExactState {
    core: BitGrid,
    gliders: Vec<GosperGliderInstance>,
}

fn extract_gosper_state(grid: &BitGrid) -> Option<GosperExactState> {
    let mut core_cells = crop_grid_region(grid, 0, 0, 36, 9).live_cells();
    let field = exclude_rect(grid, 0, 0, 36, 9);
    let variants = canonical_glider_variants();
    let mut gliders = Vec::new();
    for component in connected_components(&field) {
        let component_grid = BitGrid::from_cells(&component);
        let (signature, origin) = normalize(&component_grid);
        let Some(phase) = variants
            .iter()
            .position(|variant| *variant == signature)
            .and_then(|index| u8::try_from(index).ok())
        else {
            core_cells.extend(component);
            continue;
        };
        gliders.push(GosperGliderInstance { origin, phase });
    }
    if gliders.is_empty() {
        return None;
    }
    Some(GosperExactState {
        core: BitGrid::from_cells(&core_cells),
        gliders,
    })
}

fn advance_exact_steps(seed: &BitGrid, generations: u64) -> BitGrid {
    let mut grid = seed.clone();
    let mut memo = Memo::default();
    for _ in 0..generations {
        grid = step_grid_with_changes_and_memo(&grid, &mut memo).0;
    }
    grid
}

fn crop_grid_region(
    grid: &BitGrid,
    min_x: Coord,
    min_y: Coord,
    width: Coord,
    height: Coord,
) -> BitGrid {
    let max_x = min_x + width - 1;
    let max_y = min_y + height - 1;
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|(x, y)| *x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y)
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn exclude_rect(
    grid: &BitGrid,
    min_x: Coord,
    min_y: Coord,
    width: Coord,
    height: Coord,
) -> BitGrid {
    let max_x = min_x + width - 1;
    let max_y = min_y + height - 1;
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|(x, y)| !(*x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y))
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn connected_components(grid: &BitGrid) -> Vec<Vec<Cell>> {
    let live = grid.live_cells();
    let mut remaining = live
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut components = Vec::new();
    while let Some(&start) = remaining.iter().next() {
        let mut queue = std::collections::VecDeque::from([start]);
        let mut component = Vec::new();
        remaining.remove(&start);
        while let Some((x, y)) = queue.pop_front() {
            component.push((x, y));
            for ny in (y - 1)..=(y + 1) {
                for nx in (x - 1)..=(x + 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    if remaining.remove(&(nx, ny)) {
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

fn canonical_glider_variants() -> &'static [NormalizedGridSignature; 4] {
    static VARIANTS: OnceLock<[NormalizedGridSignature; 4]> = OnceLock::new();
    VARIANTS.get_or_init(|| {
        let mut variants = Vec::with_capacity(4);
        let mut grid = pattern_by_name("glider").expect("glider pattern should exist");
        let mut memo = Memo::default();
        for _ in 0..4 {
            variants.push(normalize(&grid).0);
            grid = step_grid_with_changes_and_memo(&grid, &mut memo).0;
        }
        [
            variants[0].clone(),
            variants[1].clone(),
            variants[2].clone(),
            variants[3].clone(),
        ]
    })
}

type GliderBoundsTable = [[(Coord, Coord, Coord, Coord); 120]; 4];

fn glider_runtime_table() -> &'static GliderBoundsTable {
    static TABLE: OnceLock<GliderBoundsTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[(0, 0, 0, 0); 120]; 4];
        let mut phase_grid = pattern_by_name("glider").expect("glider pattern should exist");
        let variants = canonical_glider_variants();
        for phase in 0..4 {
            let phase_signature = &variants[phase];
            let mut memo = Memo::default();
            for residual in 0..120 {
                if residual == 0 {
                    let bounds = phase_grid.bounds().expect("glider should be non-empty");
                    table[phase][residual] = bounds;
                } else {
                    phase_grid = step_grid_with_changes_and_memo(&phase_grid, &mut memo).0;
                    let bounds = phase_grid.bounds().expect("glider should be non-empty");
                    table[phase][residual] = bounds;
                }
            }
            phase_grid = BitGrid::from_cells(&phase_signature.cells);
        }
        table
    })
}
