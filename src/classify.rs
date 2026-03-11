use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::benchmark::effective_generation_limit;
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};

fn bounds_dimensions(bounds: (Coord, Coord, Coord, Coord)) -> (Coord, Coord, Coord) {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;
    (width, height, width.max(height))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    DiesOut {
        at_generation: u64,
    },
    Repeats {
        period: u64,
        first_seen: u64,
    },
    Spaceship {
        period: u64,
        first_seen: u64,
        delta: Cell,
        detected_at: u64,
    },
    LikelyInfinite {
        reason: &'static str,
        detected_at: u64,
    },
    Unknown {
        simulated: u64,
    },
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DiesOut { at_generation } => write!(f, "dies_out@{at_generation}"),
            Self::Repeats { period, first_seen } => {
                write!(f, "repeats(period={period}, first_seen={first_seen})")
            }
            Self::Spaceship {
                period,
                first_seen,
                delta,
                detected_at,
            } => write!(
                f,
                "spaceship(period={period}, first_seen={first_seen}, dx={}, dy={}, detected_at={detected_at})",
                delta.0, delta.1
            ),
            Self::LikelyInfinite {
                reason,
                detected_at,
            } => write!(f, "likely_infinite({reason}, gen={detected_at})"),
            Self::Unknown { simulated } => write!(f, "unknown(after={simulated})"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClassificationLimits {
    pub max_generations: u64,
    pub max_population: usize,
    pub max_bounding_box: Coord,
}

impl Default for ClassificationLimits {
    fn default() -> Self {
        Self {
            max_generations: 512,
            max_population: 20_000,
            max_bounding_box: Coord::MAX,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ClassificationCheckpoint {
    pub generation: u64,
    pub grid: BitGrid,
    pub seen: HashMap<NormalizedGridSignature, (u64, Cell)>,
}

pub fn classify_seed(
    seed: &BitGrid,
    limits: &ClassificationLimits,
    memo: &mut Memo,
) -> Classification {
    let (seed_signature, _) = normalize(seed);
    if let Some(cached) = memo.get_classification(&seed_signature) {
        return cached;
    }

    let (result, _) = predict_seed_with_checkpoint(seed, limits, memo);
    memo.insert_classification(seed_signature, result.clone());
    result
}

pub(crate) fn predict_seed_with_checkpoint(
    seed: &BitGrid,
    limits: &ClassificationLimits,
    memo: &mut Memo,
) -> (Classification, ClassificationCheckpoint) {
    run_classification_from_state(
        seed.clone(),
        HashMap::new(),
        0,
        effective_generation_limit(limits, seed.population(), seed.bounds()),
        limits,
        memo,
    )
}

fn run_classification_from_state(
    mut grid: BitGrid,
    mut seen: HashMap<NormalizedGridSignature, (u64, Cell)>,
    mut generation: u64,
    mut generation_limit: u64,
    limits: &ClassificationLimits,
    memo: &mut Memo,
) -> (Classification, ClassificationCheckpoint) {
    let mut metrics_history: Vec<(usize, Coord, Coord, Coord, Coord, Coord)> = Vec::new();

    while generation <= generation_limit {
        let (signature, origin) = normalize(&grid);

        if grid.is_empty() {
            return (
                Classification::DiesOut {
                    at_generation: generation,
                },
                ClassificationCheckpoint {
                    generation,
                    grid,
                    seen,
                },
            );
        }

        if let Some(&(first_seen, first_origin)) = seen.get(&signature) {
            let period = generation - first_seen;
            let dx = origin.0 - first_origin.0;
            let dy = origin.1 - first_origin.1;
            let result = if dx == 0 && dy == 0 {
                Classification::Repeats { period, first_seen }
            } else {
                Classification::Spaceship {
                    period,
                    first_seen,
                    delta: (dx, dy),
                    detected_at: generation,
                }
            };
            return (
                result,
                ClassificationCheckpoint {
                    generation,
                    grid,
                    seen,
                },
            );
        }

        if grid.population() > limits.max_population {
            return (
                Classification::LikelyInfinite {
                    reason: "population_growth",
                    detected_at: generation,
                },
                ClassificationCheckpoint {
                    generation,
                    grid,
                    seen,
                },
            );
        }

        if let Some(bounds) = grid.bounds() {
            let (min_x, min_y, max_x, max_y) = bounds;
            let (width, height, span) = bounds_dimensions(bounds);
            metrics_history.push((
                grid.population(),
                min_x,
                max_x,
                min_y,
                max_y,
                span,
            ));
            if width > limits.max_bounding_box || height > limits.max_bounding_box {
                return (
                    Classification::LikelyInfinite {
                        reason: "expanding_bounds",
                        detected_at: generation,
                    },
                    ClassificationCheckpoint {
                        generation,
                        grid,
                        seen,
                    },
                );
            }

            if let Some(result) =
                detect_persistent_expansion(generation, &metrics_history, &grid, limits)
            {
                return (
                    result,
                    ClassificationCheckpoint {
                        generation,
                        grid,
                        seen,
                    },
                );
            }
        }

        seen.insert(signature.clone(), (generation, origin));
        grid = step_grid_with_changes_and_memo(&grid, memo).0;
        generation += 1;

        if generation > generation_limit {
            let mut next_limit =
                effective_generation_limit(limits, grid.population(), grid.bounds());
            if next_limit <= generation_limit
                && let Some(settling_limit) =
                    settling_extension_limit(limits, generation_limit, &metrics_history)
            {
                next_limit = settling_limit;
            }
            generation_limit = next_limit;
        }
    }

    (
        Classification::Unknown {
            simulated: generation_limit,
        },
        ClassificationCheckpoint {
            generation,
            grid,
            seen,
        },
    )
}

fn settling_extension_limit(
    limits: &ClassificationLimits,
    generation_limit: u64,
    metrics_history: &[(usize, Coord, Coord, Coord, Coord, Coord)],
) -> Option<u64> {
    const MAX_SETTLING_POPULATION: usize = 256;
    const MAX_SETTLING_SPAN: Coord = 64;
    const MAX_WIDE_SETTLING_POPULATION: usize = 16;
    const MAX_WIDE_SETTLING_SPAN: Coord = 256;
    const MIN_EXTENSION_LIMIT: u64 = 512;
    const MAX_EXTENSION_LIMIT: u64 = 1024;

    if limits.max_generations < 256 || generation_limit >= MAX_EXTENSION_LIMIT {
        return None;
    }

    let &(current_population, _, _, _, _, max_span) = metrics_history.last()?;
    let bounded_small_pattern =
        current_population <= MAX_SETTLING_POPULATION && max_span <= MAX_SETTLING_SPAN;
    let bounded_wide_tiny_pattern =
        current_population <= MAX_WIDE_SETTLING_POPULATION && max_span <= MAX_WIDE_SETTLING_SPAN;

    if !bounded_small_pattern && !bounded_wide_tiny_pattern {
        return None;
    }

    Some(
        limits
            .max_generations
            .saturating_mul(2)
            .clamp(MIN_EXTENSION_LIMIT, MAX_EXTENSION_LIMIT),
    )
}

fn detect_persistent_expansion(
    generation: u64,
    metrics_history: &[(usize, Coord, Coord, Coord, Coord, Coord)],
    grid: &BitGrid,
    limits: &ClassificationLimits,
) -> Option<Classification> {
    const BURN_IN: u64 = 288;
    const WINDOW: usize = 32;
    const MIN_POPULATION_GROWTH_PER_WINDOW: usize = 1;
    const MIN_PERSISTENT_EXPANSION_SPAN: Coord = 64;
    const MIN_HEURISTIC_HORIZON: u64 = 512;

    if limits.max_generations < MIN_HEURISTIC_HORIZON {
        return None;
    }

    if generation < BURN_IN || metrics_history.len() <= WINDOW * 2 {
        return None;
    }

    let (old_population, old_min_x, old_max_x, old_min_y, old_max_y, _) =
        metrics_history[metrics_history.len() - (WINDOW * 2) - 1];
    let (mid_population, mid_min_x, mid_max_x, mid_min_y, mid_max_y, _) =
        metrics_history[metrics_history.len() - WINDOW - 1];
    let (
        current_population,
        current_min_x,
        current_max_x,
        current_min_y,
        current_max_y,
        current_span,
    ) = metrics_history[metrics_history.len() - 1];

    if current_span < MIN_PERSISTENT_EXPANSION_SPAN {
        return None;
    }

    let monotone_population = current_population >= mid_population
        && mid_population >= old_population
        && current_population.saturating_sub(mid_population) >= MIN_POPULATION_GROWTH_PER_WINDOW
        && mid_population.saturating_sub(old_population) >= MIN_POPULATION_GROWTH_PER_WINDOW;

    let width_growth_1 = (mid_max_x - mid_min_x) - (old_max_x - old_min_x);
    let width_growth_2 = (current_max_x - current_min_x) - (mid_max_x - mid_min_x);
    let height_growth_1 = (mid_max_y - mid_min_y) - (old_max_y - old_min_y);
    let height_growth_2 = (current_max_y - current_min_y) - (mid_max_y - mid_min_y);

    let x_positive_front = edge_advances(
        mid_max_x - old_max_x,
        current_max_x - mid_max_x,
        old_min_x - mid_min_x,
        mid_min_x - current_min_x,
        height_growth_1,
        height_growth_2,
    );
    let x_negative_front = edge_advances(
        old_min_x - mid_min_x,
        mid_min_x - current_min_x,
        mid_max_x - old_max_x,
        current_max_x - mid_max_x,
        height_growth_1,
        height_growth_2,
    );
    let y_positive_front = edge_advances(
        mid_max_y - old_max_y,
        current_max_y - mid_max_y,
        old_min_y - mid_min_y,
        mid_min_y - current_min_y,
        width_growth_1,
        width_growth_2,
    );
    let y_negative_front = edge_advances(
        old_min_y - mid_min_y,
        mid_min_y - current_min_y,
        mid_max_y - old_max_y,
        current_max_y - mid_max_y,
        width_growth_1,
        width_growth_2,
    );

    let fronts = FrontierDirections {
        pos_x: x_positive_front,
        neg_x: x_negative_front,
        pos_y: y_positive_front,
        neg_y: y_negative_front,
    };

    if monotone_population
        && fronts.any()
        && (has_detached_frontier_glider(grid, &fronts) || has_trailing_blinker_ash(grid, &fronts))
    {
        return Some(Classification::LikelyInfinite {
            reason: "persistent_expansion",
            detected_at: generation,
        });
    }

    None
}

#[derive(Clone, Copy)]
struct FrontierDirections {
    pos_x: bool,
    neg_x: bool,
    pos_y: bool,
    neg_y: bool,
}

impl FrontierDirections {
    fn any(self) -> bool {
        self.pos_x || self.neg_x || self.pos_y || self.neg_y
    }
}

fn edge_advances(
    prior_front: Coord,
    recent_front: Coord,
    prior_back: Coord,
    recent_back: Coord,
    prior_orthogonal: Coord,
    recent_orthogonal: Coord,
) -> bool {
    const MIN_EDGE_ADVANCE_PER_WINDOW: Coord = 6;
    const MAX_OPPOSITE_EDGE_DRIFT: Coord = 4;
    const MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW: Coord = 8;

    prior_front >= MIN_EDGE_ADVANCE_PER_WINDOW
        && recent_front >= MIN_EDGE_ADVANCE_PER_WINDOW
        && prior_back <= MAX_OPPOSITE_EDGE_DRIFT
        && recent_back <= MAX_OPPOSITE_EDGE_DRIFT
        && prior_orthogonal <= MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW
        && recent_orthogonal <= MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW
}

fn has_detached_frontier_glider(grid: &BitGrid, fronts: &FrontierDirections) -> bool {
    const GLIDER_CELLS: usize = 5;
    const MAX_COMPONENT_SPAN: Coord = 4;
    const FRONT_MARGIN: Coord = 3;
    const MIN_GAP_FROM_MAIN: Coord = 8;
    const MIN_FRONTIER_GLIDERS: usize = 1;

    let Some((global_min_x, global_min_y, global_max_x, global_max_y)) = grid.bounds() else {
        return false;
    };
    let components = connected_components(grid);
    if components.len() < 2 {
        return false;
    }

    let main = components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let (main_min_x, main_min_y, main_max_x, main_max_y) = component_bounds(&main);

    let frontier_gliders = components
        .into_iter()
        .filter(|component| {
            if component == &main || component.len() != GLIDER_CELLS {
                return false;
            }
            let (min_x, min_y, max_x, max_y) = component_bounds(component);
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > MAX_COMPONENT_SPAN || height > MAX_COMPONENT_SPAN {
                return false;
            }

            if !matches_glider(component) {
                return false;
            }

            let near_front = (fronts.pos_x && global_max_x - max_x <= FRONT_MARGIN)
                || (fronts.neg_x && min_x - global_min_x <= FRONT_MARGIN)
                || (fronts.pos_y && global_max_y - max_y <= FRONT_MARGIN)
                || (fronts.neg_y && min_y - global_min_y <= FRONT_MARGIN);

            let separated = max_x < main_min_x - MIN_GAP_FROM_MAIN
                || min_x > main_max_x + MIN_GAP_FROM_MAIN
                || max_y < main_min_y - MIN_GAP_FROM_MAIN
                || min_y > main_max_y + MIN_GAP_FROM_MAIN;

            near_front && separated
        })
        .count();

    frontier_gliders >= MIN_FRONTIER_GLIDERS
}

fn has_trailing_blinker_ash(grid: &BitGrid, fronts: &FrontierDirections) -> bool {
    const BLINKER_CELLS: usize = 3;
    const MAX_COMPONENT_SPAN: Coord = 3;
    const TRAIL_MARGIN: Coord = 6;
    const MIN_GAP_FROM_MAIN: Coord = 8;
    const MIN_TRAILING_BLINKERS: usize = 2;

    let Some((global_min_x, global_min_y, global_max_x, global_max_y)) = grid.bounds() else {
        return false;
    };
    let components = connected_components(grid);
    if components.len() < 3 {
        return false;
    }

    let main = components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let (main_min_x, main_min_y, main_max_x, main_max_y) = component_bounds(&main);

    let trailing_blinkers = components
        .into_iter()
        .filter(|component| {
            if component == &main || component.len() != BLINKER_CELLS {
                return false;
            }
            let (min_x, min_y, max_x, max_y) = component_bounds(component);
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > MAX_COMPONENT_SPAN || height > MAX_COMPONENT_SPAN {
                return false;
            }
            if !matches_blinker(component) {
                return false;
            }

            let near_trail = (fronts.pos_x && min_x - global_min_x <= TRAIL_MARGIN)
                || (fronts.neg_x && global_max_x - max_x <= TRAIL_MARGIN)
                || (fronts.pos_y && min_y - global_min_y <= TRAIL_MARGIN)
                || (fronts.neg_y && global_max_y - max_y <= TRAIL_MARGIN);

            let separated = max_x < main_min_x - MIN_GAP_FROM_MAIN
                || min_x > main_max_x + MIN_GAP_FROM_MAIN
                || max_y < main_min_y - MIN_GAP_FROM_MAIN
                || min_y > main_max_y + MIN_GAP_FROM_MAIN;

            near_trail && separated
        })
        .count();

    trailing_blinkers >= MIN_TRAILING_BLINKERS
}

fn matches_glider(component: &[Cell]) -> bool {
    let normalized = normalize(&BitGrid::from_cells(component)).0.cells;
    glider_variants()
        .iter()
        .any(|variant| variant == &normalized)
}

fn matches_blinker(component: &[Cell]) -> bool {
    let normalized = normalize(&BitGrid::from_cells(component)).0.cells;
    blinker_variants()
        .iter()
        .any(|variant| variant == &normalized)
}

fn glider_variants() -> Vec<Vec<Cell>> {
    type Transform = fn(Coord, Coord) -> Cell;
    let phases = [
        vec![(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
        vec![(0, 0), (2, 0), (1, 1), (2, 1), (1, 2)],
        vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)],
        vec![(1, 0), (0, 1), (1, 1), (2, 1), (0, 2)],
    ];
    let transforms: [Transform; 8] = [
        |x, y| (x, y),
        |x, y| (x, -y),
        |x, y| (-x, y),
        |x, y| (-x, -y),
        |x, y| (y, x),
        |x, y| (y, -x),
        |x, y| (-y, x),
        |x, y| (-y, -x),
    ];
    let mut variants = Vec::new();

    for phase in phases {
        for transform in transforms {
            let transformed = phase
                .iter()
                .map(|&(x, y)| transform(x, y))
                .collect::<Vec<_>>();
            let normalized = normalize(&BitGrid::from_cells(&transformed)).0.cells;
            if !variants.iter().any(|existing| existing == &normalized) {
                variants.push(normalized);
            }
        }
    }

    variants
}

fn blinker_variants() -> Vec<Vec<Cell>> {
    type Transform = fn(Coord, Coord) -> Cell;
    let phases = [vec![(0, 0), (1, 0), (2, 0)], vec![(0, 0), (0, 1), (0, 2)]];
    let transforms: [Transform; 8] = [
        |x, y| (x, y),
        |x, y| (x, -y),
        |x, y| (-x, y),
        |x, y| (-x, -y),
        |x, y| (y, x),
        |x, y| (y, -x),
        |x, y| (-y, x),
        |x, y| (-y, -x),
    ];
    let mut variants = Vec::new();

    for phase in phases {
        for transform in transforms {
            let transformed = phase
                .iter()
                .map(|&(x, y)| transform(x, y))
                .collect::<Vec<_>>();
            let normalized = normalize(&BitGrid::from_cells(&transformed)).0.cells;
            if !variants.iter().any(|existing| existing == &normalized) {
                variants.push(normalized);
            }
        }
    }

    variants
}

fn connected_components(grid: &BitGrid) -> Vec<Vec<Cell>> {
    let mut remaining = grid.live_cells().into_iter().collect::<HashSet<_>>();
    let mut components = Vec::new();

    while let Some(&start) = remaining.iter().next() {
        let mut queue = VecDeque::from([start]);
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

fn component_bounds(component: &[Cell]) -> (Coord, Coord, Coord, Coord) {
    let mut min_x = component[0].0;
    let mut max_x = component[0].0;
    let mut min_y = component[0].1;
    let mut max_y = component[0].1;
    for &(x, y) in component.iter().skip(1) {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x, max_y)
}
