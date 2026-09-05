use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
pub(crate) mod analysis;
mod evaluator;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};
use crate::recurrence::{ExactRecurrenceTracker, Lineage, Observation, ObserveOutcome};

const SETTLING_MAX_POPULATION: usize = 256;
const SETTLING_MAX_SPAN: Coord = 64;
const SETTLING_WIDE_MAX_POPULATION: usize = 16;
const SETTLING_WIDE_MAX_SPAN: Coord = 256;
const SETTLING_ULTRA_WIDE_MAX_POPULATION: usize = 16;
const SETTLING_ULTRA_WIDE_MAX_SPAN: Coord = 1_024;
const SETTLING_MIN_EXTENSION_LIMIT: u64 = 512;
const SETTLING_MAX_EXTENSION_LIMIT: u64 = 1_024;
const SETTLING_ULTRA_WIDE_MAX_EXTENSION_LIMIT: u64 = 20_000;

const PERSISTENT_EXPANSION_BURN_IN: u64 = 2_048;
const PERSISTENT_EXPANSION_EMITTER_BURN_IN: u64 = 256;
const PERSISTENT_EXPANSION_WINDOW: usize = 32;
const PERSISTENT_EXPANSION_MIN_POPULATION_GROWTH_PER_WINDOW: usize = 1;
const PERSISTENT_EXPANSION_MIN_SPAN: Coord = 64;
const PERSISTENT_EXPANSION_MIN_HEURISTIC_HORIZON: u64 = 512;
const PERSISTENT_EXPANSION_MIN_EMITTER_SPAN: Coord = 128;
const PERSISTENT_EXPANSION_MIN_EMITTER_SCALE_POPULATION: usize = 512;

const FRONTIER_MIN_EDGE_ADVANCE_PER_WINDOW: Coord = 6;
const FRONTIER_MAX_OPPOSITE_EDGE_DRIFT: Coord = 4;
const FRONTIER_MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW: Coord = 8;

const GLIDER_COMPONENT_CELLS: usize = 5;
const GLIDER_MAX_COMPONENT_SPAN: Coord = 4;
const GLIDER_FRONT_MARGIN: Coord = 3;
const DETACHED_PATTERN_MIN_GAP_FROM_MAIN: Coord = 8;

const BLINKER_COMPONENT_CELLS: usize = 3;
const BLINKER_MAX_COMPONENT_SPAN: Coord = 3;
const BLINKER_TRAIL_MARGIN: Coord = 6;

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

/// Stable categorical result for classifier consumers that do not need the
/// legacy result's presentation-oriented payload layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassificationOutcome {
    Extinct,
    StillLife,
    Oscillator,
    Spaceship,
    Emitter,
    Puffer,
    Expanding,
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassificationCertainty {
    Exact,
    Heuristic,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassificationGrowthKind {
    Emitter,
    Puffer,
    PersistentExpansion,
    Other(&'static str),
}

impl ClassificationGrowthKind {
    fn from_legacy(reason: &'static str) -> Self {
        match reason {
            "emitter" => Self::Emitter,
            "puffer" => Self::Puffer,
            "persistent_expansion" => Self::PersistentExpansion,
            other => Self::Other(other),
        }
    }

    fn legacy_reason(self) -> &'static str {
        match self {
            Self::Emitter => "emitter",
            Self::Puffer => "puffer",
            Self::PersistentExpansion => "persistent_expansion",
            Self::Other(reason) => reason,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassificationEvidence {
    Extinction {
        at_generation: u64,
    },
    Recurrence {
        period: u64,
        first_seen: u64,
        displacement: Cell,
        detected_at: u64,
    },
    PersistentGrowth {
        kind: ClassificationGrowthKind,
        detected_at: u64,
    },
    Horizon {
        simulated: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassificationReport {
    pub outcome: ClassificationOutcome,
    pub certainty: ClassificationCertainty,
    pub observed_through: u64,
    pub evidence: ClassificationEvidence,
}

impl ClassificationReport {
    #[must_use]
    pub fn from_legacy(classification: &Classification) -> Self {
        match classification {
            Classification::DiesOut { at_generation } => Self {
                outcome: ClassificationOutcome::Extinct,
                certainty: ClassificationCertainty::Exact,
                observed_through: *at_generation,
                evidence: ClassificationEvidence::Extinction {
                    at_generation: *at_generation,
                },
            },
            Classification::Repeats { period, first_seen } => {
                let detected_at = first_seen.saturating_add(*period);
                Self {
                    outcome: if *period == 1 {
                        ClassificationOutcome::StillLife
                    } else {
                        ClassificationOutcome::Oscillator
                    },
                    certainty: ClassificationCertainty::Exact,
                    observed_through: detected_at,
                    evidence: ClassificationEvidence::Recurrence {
                        period: *period,
                        first_seen: *first_seen,
                        displacement: (0, 0),
                        detected_at,
                    },
                }
            }
            Classification::Spaceship {
                period,
                first_seen,
                delta,
                detected_at,
            } => Self {
                outcome: ClassificationOutcome::Spaceship,
                certainty: ClassificationCertainty::Exact,
                observed_through: *detected_at,
                evidence: ClassificationEvidence::Recurrence {
                    period: *period,
                    first_seen: *first_seen,
                    displacement: *delta,
                    detected_at: *detected_at,
                },
            },
            Classification::LikelyInfinite {
                reason,
                detected_at,
            } => {
                let kind = ClassificationGrowthKind::from_legacy(reason);
                Self {
                    outcome: match kind {
                        ClassificationGrowthKind::Emitter => ClassificationOutcome::Emitter,
                        ClassificationGrowthKind::Puffer => ClassificationOutcome::Puffer,
                        _ => ClassificationOutcome::Expanding,
                    },
                    certainty: ClassificationCertainty::Heuristic,
                    observed_through: *detected_at,
                    evidence: ClassificationEvidence::PersistentGrowth {
                        kind,
                        detected_at: *detected_at,
                    },
                }
            }
            Classification::Unknown { simulated } => Self {
                outcome: ClassificationOutcome::Unresolved,
                certainty: ClassificationCertainty::Inconclusive,
                observed_through: *simulated,
                evidence: ClassificationEvidence::Horizon {
                    simulated: *simulated,
                },
            },
        }
    }

    #[must_use]
    pub fn to_legacy(&self) -> Classification {
        match &self.evidence {
            ClassificationEvidence::Extinction { at_generation } => Classification::DiesOut {
                at_generation: *at_generation,
            },
            ClassificationEvidence::Recurrence {
                period,
                first_seen,
                displacement,
                detected_at: _,
            } if *displacement == (0, 0) => Classification::Repeats {
                period: *period,
                first_seen: *first_seen,
            },
            ClassificationEvidence::Recurrence {
                period,
                first_seen,
                displacement,
                detected_at,
            } => Classification::Spaceship {
                period: *period,
                first_seen: *first_seen,
                delta: *displacement,
                detected_at: *detected_at,
            },
            ClassificationEvidence::PersistentGrowth { kind, detected_at } => {
                Classification::LikelyInfinite {
                    reason: kind.legacy_reason(),
                    detected_at: *detected_at,
                }
            }
            ClassificationEvidence::Horizon { simulated } => Classification::Unknown {
                simulated: *simulated,
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClassificationCache {
    exact: HashMap<NormalizedGridSignature, ClassificationReport>,
}

impl ClassificationCache {
    #[must_use]
    pub fn get(&self, signature: &NormalizedGridSignature) -> Option<ClassificationReport> {
        self.exact.get(signature).cloned()
    }

    pub fn insert(&mut self, signature: NormalizedGridSignature, report: ClassificationReport) {
        if report.certainty == ClassificationCertainty::Exact {
            self.exact.insert(signature, report);
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.exact.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exact.is_empty()
    }
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
}

#[must_use]
pub fn effective_generation_limit(
    limits: &ClassificationLimits,
    population: usize,
    bounds: Option<(Coord, Coord, Coord, Coord)>,
) -> u64 {
    const SMALL_PATTERN_POPULATION: usize = 64;
    const SMALL_PATTERN_SPAN: Coord = 24;
    const MIN_EXTENDED_LIMIT: u64 = 1_024;
    let Some((min_x, min_y, max_x, max_y)) = bounds else {
        return limits.max_generations;
    };
    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;
    if population <= SMALL_PATTERN_POPULATION
        && width <= SMALL_PATTERN_SPAN
        && height <= SMALL_PATTERN_SPAN
    {
        return limits.max_generations.max(MIN_EXTENDED_LIMIT);
    }
    limits.max_generations
}

impl Default for ClassificationLimits {
    fn default() -> Self {
        Self {
            max_generations: 512,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ClassificationCheckpoint {
    pub generation: u64,
    pub grid: BitGrid,
    pub recurrence: ExactRecurrenceTracker,
}

pub fn classify_seed(
    seed: &BitGrid,
    limits: &ClassificationLimits,
    memo: &mut Memo,
) -> Classification {
    classify_seed_report(seed, limits, memo).to_legacy()
}

pub fn classify_seed_report(
    seed: &BitGrid,
    limits: &ClassificationLimits,
    transition_memo: &mut Memo,
) -> ClassificationReport {
    let (result, _) = predict_seed_with_checkpoint(seed, limits, transition_memo);
    ClassificationReport::from_legacy(&result)
}

pub fn classify_seed_report_cached(
    seed: &BitGrid,
    limits: &ClassificationLimits,
    transition_memo: &mut Memo,
    classification_cache: &mut ClassificationCache,
) -> ClassificationReport {
    let (seed_signature, _) = normalize(seed);
    if let Some(cached) = classification_cache.get(&seed_signature) {
        return cached;
    }

    let result = classify_seed_report(seed, limits, transition_memo);
    classification_cache.insert(seed_signature, result.clone());
    result
}

pub(crate) fn predict_seed_with_checkpoint(
    seed: &BitGrid,
    limits: &ClassificationLimits,
    memo: &mut Memo,
) -> (Classification, ClassificationCheckpoint) {
    run_classification_from_state(
        seed.clone(),
        ExactRecurrenceTracker::new(Lineage::fresh()),
        0,
        effective_generation_limit(limits, seed.population(), seed.bounds()),
        limits,
        memo,
    )
}

fn run_classification_from_state(
    mut grid: BitGrid,
    recurrence: ExactRecurrenceTracker,
    mut generation: u64,
    generation_limit: u64,
    limits: &ClassificationLimits,
    memo: &mut Memo,
) -> (Classification, ClassificationCheckpoint) {
    let mut evidence = evaluator::EvidenceEvaluator::new(recurrence, limits, generation_limit);
    loop {
        let observation = Observation::from_grid(evidence.recurrence.lineage(), generation, &grid);
        if let Some(report) =
            evidence.observe(generation, observation, grid.is_empty(), Some(&grid))
        {
            return (
                report.to_legacy(),
                ClassificationCheckpoint {
                    generation,
                    grid,
                    recurrence: evidence.recurrence,
                },
            );
        }
        grid = step_grid_with_changes_and_memo(&grid, memo).0;
        // The evaluator returns a horizon result before the u64 boundary.
        generation += 1;
    }
}

fn settling_extension_limit(
    limits: &ClassificationLimits,
    generation_limit: u64,
    metrics_history: &[(usize, Coord, Coord, Coord, Coord, Coord)],
) -> Option<u64> {
    if limits.max_generations < 256 {
        return None;
    }

    let &(current_population, _, _, _, _, max_span) = metrics_history.last()?;
    let bounded_small_pattern =
        current_population <= SETTLING_MAX_POPULATION && max_span <= SETTLING_MAX_SPAN;
    let bounded_wide_tiny_pattern =
        current_population <= SETTLING_WIDE_MAX_POPULATION && max_span <= SETTLING_WIDE_MAX_SPAN;
    let bounded_ultra_wide_tiny_pattern = current_population <= SETTLING_ULTRA_WIDE_MAX_POPULATION
        && max_span <= SETTLING_ULTRA_WIDE_MAX_SPAN;

    if !bounded_small_pattern && !bounded_wide_tiny_pattern && !bounded_ultra_wide_tiny_pattern {
        return None;
    }

    if bounded_ultra_wide_tiny_pattern && generation_limit < SETTLING_ULTRA_WIDE_MAX_EXTENSION_LIMIT
    {
        return Some(limits.max_generations.saturating_mul(32).clamp(
            SETTLING_MIN_EXTENSION_LIMIT,
            SETTLING_ULTRA_WIDE_MAX_EXTENSION_LIMIT,
        ));
    }

    if generation_limit >= SETTLING_MAX_EXTENSION_LIMIT {
        return None;
    }

    Some(
        limits
            .max_generations
            .saturating_mul(2)
            .clamp(SETTLING_MIN_EXTENSION_LIMIT, SETTLING_MAX_EXTENSION_LIMIT),
    )
}

fn detect_persistent_expansion(
    generation: u64,
    metrics_history: &[(usize, Coord, Coord, Coord, Coord, Coord)],
    grid: &BitGrid,
    limits: &ClassificationLimits,
) -> Option<Classification> {
    if limits.max_generations < PERSISTENT_EXPANSION_MIN_HEURISTIC_HORIZON {
        return None;
    }

    if generation < PERSISTENT_EXPANSION_EMITTER_BURN_IN
        || metrics_history.len() <= PERSISTENT_EXPANSION_WINDOW * 2
    {
        return None;
    }

    let (old_population, old_min_x, old_max_x, old_min_y, old_max_y, _) =
        metrics_history[metrics_history.len() - (PERSISTENT_EXPANSION_WINDOW * 2) - 1];
    let (mid_population, mid_min_x, mid_max_x, mid_min_y, mid_max_y, _) =
        metrics_history[metrics_history.len() - PERSISTENT_EXPANSION_WINDOW - 1];
    let (
        current_population,
        current_min_x,
        current_max_x,
        current_min_y,
        current_max_y,
        current_span,
    ) = metrics_history[metrics_history.len() - 1];

    if current_span < PERSISTENT_EXPANSION_MIN_SPAN {
        return None;
    }

    let monotone_population = current_population >= mid_population
        && mid_population >= old_population
        && current_population.saturating_sub(mid_population)
            >= PERSISTENT_EXPANSION_MIN_POPULATION_GROWTH_PER_WINDOW
        && mid_population.saturating_sub(old_population)
            >= PERSISTENT_EXPANSION_MIN_POPULATION_GROWTH_PER_WINDOW;

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

    let frontier_gliders = count_detached_frontier_gliders(grid, &fronts);
    let trailing_blinkers = count_trailing_blinker_ash(grid, &fronts);
    let detached_gliders = count_detached_gliders_anywhere(grid);
    let detached_blinkers = count_detached_blinkers_anywhere(grid);
    let confirmed_detached_emitter_signal = current_span >= PERSISTENT_EXPANSION_MIN_EMITTER_SPAN
        && (frontier_gliders >= 3
            || trailing_blinkers >= 3
            || detached_gliders >= 3
            || detached_blinkers >= 3);

    if generation >= PERSISTENT_EXPANSION_EMITTER_BURN_IN
        && monotone_population
        && fronts.any()
        && confirmed_detached_emitter_signal
    {
        return Some(Classification::LikelyInfinite {
            reason: "persistent_expansion",
            detected_at: generation,
        });
    }

    let emitter_scale_population =
        current_population >= PERSISTENT_EXPANSION_MIN_EMITTER_SCALE_POPULATION;
    let confirmed_emitter_signal =
        emitter_scale_population && (frontier_gliders >= 2 || trailing_blinkers >= 2);

    if generation >= PERSISTENT_EXPANSION_BURN_IN
        && monotone_population
        && fronts.any()
        && confirmed_emitter_signal
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
    prior_front >= FRONTIER_MIN_EDGE_ADVANCE_PER_WINDOW
        && recent_front >= FRONTIER_MIN_EDGE_ADVANCE_PER_WINDOW
        && prior_back <= FRONTIER_MAX_OPPOSITE_EDGE_DRIFT
        && recent_back <= FRONTIER_MAX_OPPOSITE_EDGE_DRIFT
        && prior_orthogonal <= FRONTIER_MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW
        && recent_orthogonal <= FRONTIER_MAX_ORTHOGONAL_SPAN_GROWTH_PER_WINDOW
}

fn count_detached_frontier_gliders(grid: &BitGrid, fronts: &FrontierDirections) -> usize {
    let Some((global_min_x, global_min_y, global_max_x, global_max_y)) = grid.bounds() else {
        return 0;
    };
    let components = connected_components(grid);
    if components.len() < 2 {
        return 0;
    }

    let main = components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let (main_min_x, main_min_y, main_max_x, main_max_y) = component_bounds(&main);

    components
        .into_iter()
        .filter(|component| {
            if component == &main || component.len() != GLIDER_COMPONENT_CELLS {
                return false;
            }
            let (min_x, min_y, max_x, max_y) = component_bounds(component);
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > GLIDER_MAX_COMPONENT_SPAN || height > GLIDER_MAX_COMPONENT_SPAN {
                return false;
            }

            if !matches_glider(component) {
                return false;
            }

            let near_front = (fronts.pos_x && global_max_x - max_x <= GLIDER_FRONT_MARGIN)
                || (fronts.neg_x && min_x - global_min_x <= GLIDER_FRONT_MARGIN)
                || (fronts.pos_y && global_max_y - max_y <= GLIDER_FRONT_MARGIN)
                || (fronts.neg_y && min_y - global_min_y <= GLIDER_FRONT_MARGIN);

            let separated = max_x < main_min_x - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_x > main_max_x + DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || max_y < main_min_y - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_y > main_max_y + DETACHED_PATTERN_MIN_GAP_FROM_MAIN;

            near_front && separated
        })
        .count()
}

fn count_trailing_blinker_ash(grid: &BitGrid, fronts: &FrontierDirections) -> usize {
    let Some((global_min_x, global_min_y, global_max_x, global_max_y)) = grid.bounds() else {
        return 0;
    };
    let components = connected_components(grid);
    if components.len() < 3 {
        return 0;
    }

    let main = components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let (main_min_x, main_min_y, main_max_x, main_max_y) = component_bounds(&main);

    components
        .into_iter()
        .filter(|component| {
            if component == &main || component.len() != BLINKER_COMPONENT_CELLS {
                return false;
            }
            let (min_x, min_y, max_x, max_y) = component_bounds(component);
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > BLINKER_MAX_COMPONENT_SPAN || height > BLINKER_MAX_COMPONENT_SPAN {
                return false;
            }
            if !matches_blinker(component) {
                return false;
            }

            let near_trail = (fronts.pos_x && min_x - global_min_x <= BLINKER_TRAIL_MARGIN)
                || (fronts.neg_x && global_max_x - max_x <= BLINKER_TRAIL_MARGIN)
                || (fronts.pos_y && min_y - global_min_y <= BLINKER_TRAIL_MARGIN)
                || (fronts.neg_y && global_max_y - max_y <= BLINKER_TRAIL_MARGIN);

            let separated = max_x < main_min_x - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_x > main_max_x + DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || max_y < main_min_y - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_y > main_max_y + DETACHED_PATTERN_MIN_GAP_FROM_MAIN;

            near_trail && separated
        })
        .count()
}

fn count_detached_gliders_anywhere(grid: &BitGrid) -> usize {
    let components = connected_components(grid);
    if components.len() < 2 {
        return 0;
    }

    let main = components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let (main_min_x, main_min_y, main_max_x, main_max_y) = component_bounds(&main);

    components
        .into_iter()
        .filter(|component| {
            if component == &main || component.len() != GLIDER_COMPONENT_CELLS {
                return false;
            }
            let (min_x, min_y, max_x, max_y) = component_bounds(component);
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > GLIDER_MAX_COMPONENT_SPAN || height > GLIDER_MAX_COMPONENT_SPAN {
                return false;
            }
            let separated = max_x < main_min_x - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_x > main_max_x + DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || max_y < main_min_y - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_y > main_max_y + DETACHED_PATTERN_MIN_GAP_FROM_MAIN;

            separated && matches_glider(component)
        })
        .count()
}

fn count_detached_blinkers_anywhere(grid: &BitGrid) -> usize {
    let components = connected_components(grid);
    if components.len() < 2 {
        return 0;
    }

    let main = components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let (main_min_x, main_min_y, main_max_x, main_max_y) = component_bounds(&main);

    components
        .into_iter()
        .filter(|component| {
            if component == &main || component.len() != BLINKER_COMPONENT_CELLS {
                return false;
            }
            let (min_x, min_y, max_x, max_y) = component_bounds(component);
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > BLINKER_MAX_COMPONENT_SPAN || height > BLINKER_MAX_COMPONENT_SPAN {
                return false;
            }
            let separated = max_x < main_min_x - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_x > main_max_x + DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || max_y < main_min_y - DETACHED_PATTERN_MIN_GAP_FROM_MAIN
                || min_y > main_max_y + DETACHED_PATTERN_MIN_GAP_FROM_MAIN;

            separated && matches_blinker(component)
        })
        .count()
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
