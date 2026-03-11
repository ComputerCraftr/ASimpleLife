use crate::bitgrid::{BitGrid, Coord};
use crate::hashlife::HashLifeSession;
use crate::life::step_grid_with_chunk_changes_and_memo;
use crate::memo::Memo;

const SIMD_GENERATION_LIMIT: u64 = 64;
const SIMD_POPULATION_LIMIT: usize = 512;
const SIMD_SPAN_LIMIT: Coord = 64;
const HASHLIFE_GENERATION_LIMIT: u64 = 512;
const HASHLIFE_DENSITY_LIMIT: f64 = 0.18;
const HYBRID_PREFIX_LIMIT: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimulationBackend {
    SimdChunk,
    HashLife,
    HybridSegmented,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvanceStats {
    pub backend: SimulationBackend,
    pub simd_generations: u64,
    pub hashlife_generations: u64,
}

#[derive(Clone, Debug)]
pub struct AdvanceResult {
    pub grid: BitGrid,
    pub stats: AdvanceStats,
}

#[derive(Debug, Default)]
pub struct SimulationSession {
    simd_memo: Memo,
    hashlife_session: HashLifeSession,
    preferred_backend: Option<SimulationBackend>,
}

pub fn select_backend(grid: &BitGrid, generations: u64) -> SimulationBackend {
    if generations == 0 || grid.is_empty() {
        return SimulationBackend::SimdChunk;
    }

    let population = grid.population();
    let span = grid
        .bounds()
        .map(|(min_x, min_y, max_x, max_y)| (max_x - min_x + 1).max(max_y - min_y + 1))
        .unwrap_or(0);
    let density = if span <= 0 {
        0.0
    } else {
        population as f64 / ((span * span) as f64)
    };

    if generations <= SIMD_GENERATION_LIMIT
        || (generations <= HASHLIFE_GENERATION_LIMIT
            && population <= SIMD_POPULATION_LIMIT
            && span <= SIMD_SPAN_LIMIT)
    {
        return SimulationBackend::SimdChunk;
    }

    if generations >= HASHLIFE_GENERATION_LIMIT && (density <= HASHLIFE_DENSITY_LIMIT || span >= 96)
    {
        return SimulationBackend::HashLife;
    }

    SimulationBackend::HybridSegmented
}

pub fn advance_grid(grid: &BitGrid, generations: u64) -> AdvanceResult {
    let mut session = SimulationSession::default();
    let result = session.advance(grid, generations);
    session.finish();
    result
}

impl SimulationSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&mut self, grid: &BitGrid, generations: u64) -> AdvanceResult {
        let backend = self.planned_backend(grid, generations);
        let result = match backend {
        SimulationBackend::SimdChunk => AdvanceResult {
            grid: self.advance_simd_chunk(grid, generations),
            stats: AdvanceStats {
                backend,
                simd_generations: generations,
                hashlife_generations: 0,
            },
        },
        SimulationBackend::HashLife => AdvanceResult {
            grid: {
                self.load_hashlife_state(grid);
                self.advance_hashlife_root(generations);
                self.sample_hashlife_state_grid()
                    .expect("hashlife state should be sampleable after advance")
                    .clone()
            },
            stats: AdvanceStats {
                backend,
                simd_generations: 0,
                hashlife_generations: generations,
            },
        },
        SimulationBackend::HybridSegmented => {
            let simd_generations = hybrid_prefix_generations(grid, generations);
            let simd_grid = self.advance_simd_chunk(grid, simd_generations);
            let remaining = generations.saturating_sub(simd_generations);
            let final_grid = if remaining == 0 {
                simd_grid
            } else {
                self.load_hashlife_state(&simd_grid);
                self.advance_hashlife_root(remaining);
                self.sample_hashlife_state_grid()
                    .expect("hashlife state should be sampleable after hybrid advance")
                    .clone()
            };
            AdvanceResult {
                grid: final_grid,
                stats: AdvanceStats {
                    backend,
                    simd_generations,
                    hashlife_generations: remaining,
                },
            }
        }
        };
        self.preferred_backend = Some(backend);
        result
    }

    pub fn finish(&mut self) {
        self.hashlife_session.finish();
    }

    pub fn load_hashlife_state(&mut self, grid: &BitGrid) {
        self.hashlife_session.load_grid(grid);
        self.preferred_backend = Some(SimulationBackend::HashLife);
    }

    pub fn advance_hashlife_root(&mut self, generations: u64) {
        self.hashlife_session.advance_root(generations);
        self.preferred_backend = Some(SimulationBackend::HashLife);
    }

    pub fn hashlife_loaded(&self) -> bool {
        self.hashlife_session.is_loaded()
    }

    pub fn hashlife_generation(&self) -> u64 {
        self.hashlife_session.generation()
    }

    pub fn hashlife_population(&self) -> Option<u64> {
        self.hashlife_session.population()
    }

    pub fn hashlife_bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)> {
        self.hashlife_session.bounds()
    }

    pub fn shift_hashlife_origin(&mut self, dx: Coord, dy: Coord) {
        self.hashlife_session.shift_origin(dx, dy);
    }

    pub fn sample_hashlife_state_grid(&mut self) -> Option<&BitGrid> {
        self.hashlife_session.sample_grid()
    }

    pub fn planned_backend_from_metrics(
        &self,
        population: usize,
        span: Coord,
        generations: u64,
    ) -> SimulationBackend {
        if generations == 0 || population == 0 {
            return SimulationBackend::SimdChunk;
        }

        let density = if span <= 0 {
            0.0
        } else {
            population as f64 / ((span * span) as f64)
        };

        match self.preferred_backend {
            Some(SimulationBackend::HashLife) if self.hashlife_loaded() && generations > 1 => {
                return SimulationBackend::HashLife;
            }
            Some(SimulationBackend::HybridSegmented)
                if self.hashlife_loaded() && generations > HYBRID_PREFIX_LIMIT =>
            {
                return SimulationBackend::HashLife;
            }
            Some(SimulationBackend::SimdChunk) if generations <= SIMD_GENERATION_LIMIT => {
                return SimulationBackend::SimdChunk;
            }
            _ => {}
        }

        if generations <= SIMD_GENERATION_LIMIT
            || (generations <= HASHLIFE_GENERATION_LIMIT
                && population <= SIMD_POPULATION_LIMIT
                && span <= SIMD_SPAN_LIMIT)
        {
            return SimulationBackend::SimdChunk;
        }

        if generations >= HASHLIFE_GENERATION_LIMIT
            && (density <= HASHLIFE_DENSITY_LIMIT || span >= 96)
        {
            return SimulationBackend::HashLife;
        }

        SimulationBackend::HybridSegmented
    }

    pub fn planned_backend(&self, grid: &BitGrid, generations: u64) -> SimulationBackend {
        let span = grid
            .bounds()
            .map(|(min_x, min_y, max_x, max_y)| (max_x - min_x + 1).max(max_y - min_y + 1))
            .unwrap_or(0);
        self.planned_backend_from_metrics(grid.population(), span, generations)
    }

    fn advance_simd_chunk(&mut self, grid: &BitGrid, generations: u64) -> BitGrid {
        if grid.is_empty() {
            return BitGrid::empty();
        }
        if generations == 0 {
            return grid.clone();
        }

        let mut current = None::<BitGrid>;
        for _ in 0..generations {
            let next =
                step_grid_with_chunk_changes_and_memo(current.as_ref().unwrap_or(grid), &mut self.simd_memo)
                    .0;
            current = Some(next);
        }
        current.unwrap_or_else(BitGrid::empty)
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
