use std::collections::HashMap;
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
    let mut metrics_history: Vec<(usize, i32, i32)> = Vec::new();

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
                Classification::LikelyInfinite {
                    reason: "spaceship",
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
            metrics_history.push((grid.population(), width, height));
            if width > limits.max_bounding_box || height > limits.max_bounding_box {
                let result = Classification::LikelyInfinite {
                    reason: "expanding_bounds",
                    detected_at: generation,
                };
                memo.insert_classification(seed_signature, result.clone());
                return result;
            }

            if let Some(result) = detect_persistent_expansion(generation, &metrics_history) {
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
    metrics_history: &[(usize, i32, i32)],
) -> Option<Classification> {
    const BURN_IN: usize = 180;
    const WINDOW: usize = 60;
    const MIN_AXIS_GROWTH: i32 = 12;

    if generation < BURN_IN || metrics_history.len() <= WINDOW {
        return None;
    }

    let (past_population, past_width, past_height) =
        metrics_history[metrics_history.len() - WINDOW - 1];
    let (current_population, current_width, current_height) =
        metrics_history[metrics_history.len() - 1];
    let width_growth = current_width - past_width;
    let height_growth = current_height - past_height;
    let population_growth = current_population.saturating_sub(past_population);

    if (width_growth >= MIN_AXIS_GROWTH || height_growth >= MIN_AXIS_GROWTH)
        && population_growth > 0
    {
        return Some(Classification::LikelyInfinite {
            reason: "persistent_expansion",
            detected_at: generation,
        });
    }

    None
}

fn next_grid_with_memo(
    _signature: &NormalizedGridSignature,
    current: &BitGrid,
    memo: &mut Memo,
) -> BitGrid {
    step_grid_with_changes_and_memo(current, memo).0
}
