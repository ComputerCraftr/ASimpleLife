use crate::bitgrid::BitGrid;
use crate::hashlife::HashLifeOracle;
use crate::life::step_grid_with_chunk_changes_and_memo;
use crate::memo::Memo;

const SIMD_GENERATION_LIMIT: u64 = 64;
const SIMD_POPULATION_LIMIT: usize = 512;
const SIMD_SPAN_LIMIT: i32 = 64;
const HASHLIFE_GENERATION_LIMIT: u64 = 512;
const HASHLIFE_DENSITY_LIMIT: f64 = 0.18;
const HYBRID_PREFIX_LIMIT: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimulationEngine {
    SimdChunk,
    HashLife,
    HybridSegmented,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvanceStats {
    pub engine: SimulationEngine,
    pub simd_generations: u64,
    pub hashlife_generations: u64,
}

#[derive(Clone, Debug)]
pub struct AdvanceResult {
    pub grid: BitGrid,
    pub stats: AdvanceStats,
}

pub fn select_engine(grid: &BitGrid, generations: u64) -> SimulationEngine {
    if generations == 0 || grid.is_empty() {
        return SimulationEngine::SimdChunk;
    }

    let population = grid.population();
    let span = grid
        .bounds()
        .map(|(min_x, min_y, max_x, max_y)| (max_x - min_x + 1).max(max_y - min_y + 1))
        .unwrap_or(0);
    let density = if span <= 0 {
        0.0
    } else {
        population as f64 / ((span as i64 * span as i64) as f64)
    };

    if generations <= SIMD_GENERATION_LIMIT
        || (generations <= HASHLIFE_GENERATION_LIMIT
            && population <= SIMD_POPULATION_LIMIT
            && span <= SIMD_SPAN_LIMIT)
    {
        return SimulationEngine::SimdChunk;
    }

    if generations >= HASHLIFE_GENERATION_LIMIT && (density <= HASHLIFE_DENSITY_LIMIT || span >= 96)
    {
        return SimulationEngine::HashLife;
    }

    SimulationEngine::HybridSegmented
}

pub fn advance_grid(grid: &BitGrid, generations: u64) -> AdvanceResult {
    let engine = select_engine(grid, generations);
    match engine {
        SimulationEngine::SimdChunk => AdvanceResult {
            grid: advance_simd(grid, generations),
            stats: AdvanceStats {
                engine,
                simd_generations: generations,
                hashlife_generations: 0,
            },
        },
        SimulationEngine::HashLife => {
            let mut oracle = HashLifeOracle::default();
            AdvanceResult {
                grid: oracle.advance(grid, generations),
                stats: AdvanceStats {
                    engine,
                    simd_generations: 0,
                    hashlife_generations: generations,
                },
            }
        }
        SimulationEngine::HybridSegmented => {
            let simd_generations = hybrid_prefix_generations(grid, generations);
            let simd_grid = advance_simd(grid, simd_generations);
            let remaining = generations.saturating_sub(simd_generations);
            let final_grid = if remaining == 0 {
                simd_grid
            } else {
                let mut oracle = HashLifeOracle::default();
                oracle.advance(&simd_grid, remaining)
            };
            AdvanceResult {
                grid: final_grid,
                stats: AdvanceStats {
                    engine,
                    simd_generations,
                    hashlife_generations: remaining,
                },
            }
        }
    }
}

fn hybrid_prefix_generations(grid: &BitGrid, generations: u64) -> u64 {
    let population = grid.population() as u64;
    let prefix = population
        .saturating_div(32)
        .clamp(16, HYBRID_PREFIX_LIMIT)
        .min(generations);
    prefix.max(1).min(generations)
}

fn advance_simd(grid: &BitGrid, generations: u64) -> BitGrid {
    if generations == 0 || grid.is_empty() {
        return grid.clone();
    }

    let mut memo = Memo::default();
    let mut current = grid.clone();
    for _ in 0..generations {
        current = step_grid_with_chunk_changes_and_memo(&current, &mut memo).0;
    }
    current
}
