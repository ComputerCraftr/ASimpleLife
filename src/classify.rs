use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::bitgrid::BitGrid;
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    DiesOut {
        at_generation: usize,
    },
    Repeats {
        period: usize,
        first_seen: usize,
    },
    Spaceship {
        period: usize,
        first_seen: usize,
        delta: (i32, i32),
        detected_at: usize,
    },
    LikelyInfinite {
        reason: &'static str,
        detected_at: usize,
    },
    Unknown {
        simulated: usize,
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
    pub max_generations: usize,
    pub max_population: usize,
    pub max_bounding_box: i32,
}

impl Default for ClassificationLimits {
    fn default() -> Self {
        Self {
            max_generations: 512,
            max_population: 20_000,
            max_bounding_box: 512,
        }
    }
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

    let mut seen: HashMap<NormalizedGridSignature, (usize, (i32, i32))> = HashMap::new();
    let mut grid = seed.clone();
    let mut metrics_history: Vec<(usize, i32, i32, i32, i32, i32)> = Vec::new();

    for generation in 0..=limits.max_generations {
        let (signature, origin) = normalize(&grid);

        if grid.is_empty() {
            let result = Classification::DiesOut {
                at_generation: generation,
            };
            memo.insert_classification(seed_signature, result.clone());
            return result;
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
            memo.insert_classification(seed_signature, result.clone());
            return result;
        }

        if grid.population() > limits.max_population {
            let result = Classification::LikelyInfinite {
                reason: "population_growth",
                detected_at: generation,
            };
            memo.insert_classification(seed_signature, result.clone());
            return result;
        }

        if let Some((min_x, min_y, max_x, max_y)) = grid.bounds() {
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            metrics_history.push((
                grid.population(),
                min_x,
                max_x,
                min_y,
                max_y,
                width.max(height),
            ));
            if width > limits.max_bounding_box || height > limits.max_bounding_box {
                let result = Classification::LikelyInfinite {
                    reason: "expanding_bounds",
                    detected_at: generation,
                };
                memo.insert_classification(seed_signature, result.clone());
                return result;
            }

            if let Some(result) = detect_persistent_expansion(generation, &metrics_history, &grid) {
                memo.insert_classification(seed_signature, result.clone());
                return result;
            }
        }

        seen.insert(signature.clone(), (generation, origin));
        grid = next_grid_with_memo(&signature, &grid, memo);
    }

    let result = Classification::Unknown {
        simulated: limits.max_generations,
    };
    memo.insert_classification(seed_signature, result.clone());
    result
}

fn detect_persistent_expansion(
    generation: usize,
    metrics_history: &[(usize, i32, i32, i32, i32, i32)],
    grid: &BitGrid,
) -> Option<Classification> {
    const BURN_IN: usize = 288;
    const WINDOW: usize = 32;
    const MIN_POPULATION_GROWTH_PER_WINDOW: usize = 1;

    if generation < BURN_IN || metrics_history.len() <= WINDOW * 2 {
        return None;
    }

    let (old_population, old_min_x, old_max_x, old_min_y, old_max_y, _) =
        metrics_history[metrics_history.len() - (WINDOW * 2) - 1];
    let (mid_population, mid_min_x, mid_max_x, mid_min_y, mid_max_y, _) =
        metrics_history[metrics_history.len() - WINDOW - 1];
    let (current_population, current_min_x, current_max_x, current_min_y, current_max_y, _) =
        metrics_history[metrics_history.len() - 1];

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
    prior_front: i32,
    recent_front: i32,
    prior_back: i32,
    recent_back: i32,
    prior_orthogonal: i32,
    recent_orthogonal: i32,
) -> bool {
    const MIN_EDGE_ADVANCE_PER_WINDOW: i32 = 6;
    const MAX_OPPOSITE_EDGE_DRIFT: i32 = 4;
    const MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW: i32 = 8;

    prior_front >= MIN_EDGE_ADVANCE_PER_WINDOW
        && recent_front >= MIN_EDGE_ADVANCE_PER_WINDOW
        && prior_back <= MAX_OPPOSITE_EDGE_DRIFT
        && recent_back <= MAX_OPPOSITE_EDGE_DRIFT
        && prior_orthogonal <= MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW
        && recent_orthogonal <= MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW
}

fn has_detached_frontier_glider(grid: &BitGrid, fronts: &FrontierDirections) -> bool {
    const GLIDER_CELLS: usize = 5;
    const MAX_COMPONENT_SPAN: i32 = 4;
    const FRONT_MARGIN: i32 = 3;
    const MIN_GAP_FROM_MAIN: i32 = 8;
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
    const MAX_COMPONENT_SPAN: i32 = 3;
    const TRAIL_MARGIN: i32 = 6;
    const MIN_GAP_FROM_MAIN: i32 = 8;
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

fn matches_glider(component: &[(i32, i32)]) -> bool {
    let normalized = normalize(&BitGrid::from_cells(component)).0.cells;
    glider_variants()
        .iter()
        .any(|variant| variant == &normalized)
}

fn matches_blinker(component: &[(i32, i32)]) -> bool {
    let normalized = normalize(&BitGrid::from_cells(component)).0.cells;
    blinker_variants()
        .iter()
        .any(|variant| variant == &normalized)
}

fn glider_variants() -> Vec<Vec<(i32, i32)>> {
    type Transform = fn(i32, i32) -> (i32, i32);
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

fn blinker_variants() -> Vec<Vec<(i32, i32)>> {
    type Transform = fn(i32, i32) -> (i32, i32);
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

fn connected_components(grid: &BitGrid) -> Vec<Vec<(i32, i32)>> {
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

fn component_bounds(component: &[(i32, i32)]) -> (i32, i32, i32, i32) {
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

fn next_grid_with_memo(
    _signature: &NormalizedGridSignature,
    current: &BitGrid,
    memo: &mut Memo,
) -> BitGrid {
    step_grid_with_changes_and_memo(current, memo).0
}
