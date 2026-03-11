use std::collections::HashMap;
use std::mem;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::classify::{Classification, ClassificationLimits};
use crate::engine::{SimulationBackend, SimulationSession};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};

type SeenStates = HashMap<NormalizedGridSignature, (u64, Cell)>;
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

#[derive(Debug)]
pub struct OracleSession<'a> {
    grid: Option<BitGrid>,
    generation: u64,
    seen: SeenStates,
    simulation: &'a mut SimulationSession,
    exact_memo: Memo,
    hashlife_active: bool,
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
            simulation,
            exact_memo: Memo::default(),
            hashlife_active: false,
        }
    }

    pub fn sampled_grid(&mut self) -> &BitGrid {
        self.ensure_sampled_grid()
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
                return Classification::DiesOut {
                    at_generation: self.generation,
                };
            }

            if let Some(&(first_seen, first_origin)) = self.seen.get(&signature) {
                let period = self.generation - first_seen;
                let dx = origin.0 - first_origin.0;
                let dy = origin.1 - first_origin.1;
                return if dx == 0 && dy == 0 {
                    Classification::Repeats { period, first_seen }
                } else {
                    Classification::Spaceship {
                        period,
                        first_seen,
                        delta: (dx, dy),
                        detected_at: self.generation,
                    }
                };
            }

            if population > limits.max_population {
                return Classification::LikelyInfinite {
                    reason: "population_growth",
                    detected_at: self.generation,
                };
            }

            if let Some(bounds) = bounds {
                let (width, height, _) = bounds_dimensions(bounds);
                if width > limits.max_bounding_box || height > limits.max_bounding_box {
                    return Classification::LikelyInfinite {
                        reason: "expanding_bounds",
                        detected_at: self.generation,
                    };
                }
            }

            self.seen.insert(signature, (self.generation, origin));
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
            .saturating_add(cycle_probe_prefix_window(self.current_metrics().population, self.current_metrics().bounds_span))
            .min(target_generation);
        let mut next_checkpoint_generation = cycle_probe_limit.saturating_add(1);
        let mut checkpoint_stride = 1_u64;
        while self.generation <= target_generation {
            if self.should_sample_state(target_generation, cycle_probe_limit, next_checkpoint_generation) {
                let (signature, origin, is_empty) = {
                    let grid = self.ensure_sampled_grid();
                    let (signature, origin) = normalize(grid);
                    (signature, origin, grid.is_empty())
                };
                if is_empty {
                    return OracleAdvanceOutcome {
                        classification: Classification::DiesOut {
                            at_generation: self.generation,
                        },
                        final_generation: self.generation,
                        grid: mem::take(&mut self.grid).expect("oracle sampled grid missing on extinction"),
                    };
                }

                if let Some(&(first_seen, first_origin)) = self.seen.get(&signature) {
                    let period = self.generation - first_seen;
                    let dx = origin.0 - first_origin.0;
                    let dy = origin.1 - first_origin.1;
                    let classification = if dx == 0 && dy == 0 {
                        if period > 0 && self.generation < target_generation {
                            let remaining = target_generation - self.generation;
                            let skip = (remaining / period) * period;
                            if skip > 0 {
                                self.apply_cycle_skip(skip, 0, 0);
                                if self.generation == target_generation {
                                    return OracleAdvanceOutcome {
                                        classification: Classification::Repeats { period, first_seen },
                                        final_generation: self.generation,
                                        grid: self.take_or_sample_grid(),
                                    };
                                }
                            }
                        }
                        Classification::Repeats { period, first_seen }
                    } else {
                        if period > 0 && self.generation < target_generation {
                            let remaining = target_generation - self.generation;
                            let skip_cycles = remaining / period;
                            if skip_cycles > 0 {
                                let cycle_count =
                                    Coord::try_from(skip_cycles).expect("cycle skip exceeded Coord");
                                self.apply_cycle_skip(
                                    skip_cycles * period,
                                    dx.checked_mul(cycle_count)
                                        .expect("spaceship cycle x overflow"),
                                    dy.checked_mul(cycle_count)
                                        .expect("spaceship cycle y overflow"),
                                );
                                if self.generation == target_generation {
                                    return OracleAdvanceOutcome {
                                        classification: Classification::Spaceship {
                                            period,
                                            first_seen,
                                            delta: (dx, dy),
                                            detected_at: self.generation,
                                        },
                                        final_generation: self.generation,
                                        grid: self.take_or_sample_grid(),
                                    };
                                }
                            }
                        }
                        Classification::Spaceship {
                            period,
                            first_seen,
                            delta: (dx, dy),
                            detected_at: self.generation,
                        }
                    };
                    return OracleAdvanceOutcome {
                        classification,
                        final_generation: self.generation,
                        grid: self.take_or_sample_grid(),
                    };
                }

                self.seen.insert(signature, (self.generation, origin));
                if self.generation >= cycle_probe_limit {
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
                callback(plan, self.current_metrics());
            }
            self.advance_by(plan.step_span);
            if let Some(callback) = on_step.as_deref_mut() {
                callback(
                    OracleStepPlan {
                        generation: self.generation,
                        step_span: 0,
                        backend: plan.backend,
                    },
                    self.current_metrics(),
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

    fn advance_by(&mut self, step_span: u64) {
        if step_span <= 1 {
            let current_grid = self
                .grid
                .take()
                .expect("exact oracle stepping requires a materialized grid");
            self.grid = Some(step_grid_with_changes_and_memo(&current_grid, &mut self.exact_memo).0);
            self.hashlife_active = false;
        } else {
            if !self.hashlife_active {
                let current_grid = self.ensure_sampled_grid().clone();
                self.simulation.load_hashlife_state(&current_grid);
                self.hashlife_active = true;
            }
            self.simulation.advance_hashlife_root(step_span);
            self.grid = None;
        }
        self.generation += step_span;
    }

    fn apply_cycle_skip(&mut self, generation_skip: u64, dx: Coord, dy: Coord) {
        if generation_skip == 0 {
            return;
        }
        self.generation = self.generation.saturating_add(generation_skip);
        if dx == 0 && dy == 0 {
            return;
        }
        if self.hashlife_active {
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

    fn plan_step(&mut self, generation_limit: u64, nominal_generation_limit: u64) -> OracleStepPlan {
        let step_span = continuation_step_span(self.current_state_shape(), self.generation, generation_limit, nominal_generation_limit, self.simulation, self.hashlife_active);
        OracleStepPlan {
            generation: self.generation,
            step_span,
            backend: self.planned_backend_for_shape(step_span),
        }
    }

    fn plan_target_step(&mut self, remaining: u64) -> OracleStepPlan {
        let shape = self.current_state_shape();
        let safe_hashlife_jump = max_hashlife_safe_jump_from_span(shape.bounds_span);
        let step_span = if self.hashlife_active || self.simulation.hashlife_loaded() {
            largest_power_of_two_leq(remaining.min(safe_hashlife_jump).max(1))
        } else {
            match self.simulation.planned_backend_from_metrics(shape.population, shape.bounds_span, remaining) {
                SimulationBackend::SimdChunk => 1,
                SimulationBackend::HybridSegmented => hybrid_target_prefix_generations(shape.population, remaining),
                SimulationBackend::HashLife => largest_power_of_two_leq(remaining.min(safe_hashlife_jump).max(1)),
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
        } else if self.hashlife_active || self.simulation.hashlife_loaded() {
            SimulationBackend::HashLife
        } else {
            let shape = self.current_state_shape();
            self.simulation
                .planned_backend_from_metrics(shape.population, shape.bounds_span, step_span)
        }
    }

    fn current_state_shape(&mut self) -> OracleStateMetrics {
        if self.hashlife_active || self.simulation.hashlife_loaded() {
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

    fn current_metrics(&mut self) -> OracleStateMetrics {
        self.current_state_shape()
    }

    fn should_sample_state(
        &self,
        target_generation: u64,
        cycle_probe_limit: u64,
        next_checkpoint_generation: u64,
    ) -> bool {
        self.generation <= cycle_probe_limit
            || self.generation == target_generation
            || !self.hashlife_active
            || self.generation >= next_checkpoint_generation
    }

    fn ensure_sampled_grid(&mut self) -> &BitGrid {
        if self.grid.is_none() {
            self.grid = Some(
                self.simulation
                    .sample_hashlife_state_grid()
                    .expect("hashlife state should be sampleable on demand")
                    .clone(),
            );
        }
        self.grid
            .as_ref()
            .expect("oracle sampled grid should be available")
    }

    fn take_or_sample_grid(&mut self) -> BitGrid {
        if self.grid.is_none() {
            let _ = self.ensure_sampled_grid();
        }
        self.grid
            .take()
            .expect("oracle should have a final sampled grid")
    }
}

fn continuation_step_span(
    current: OracleStateMetrics,
    generation: u64,
    generation_limit: u64,
    nominal_generation_limit: u64,
    simulation: &SimulationSession,
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

    match simulation.planned_backend_from_metrics(current.population, current.bounds_span, remaining) {
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
