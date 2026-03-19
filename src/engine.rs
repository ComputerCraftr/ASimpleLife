use crate::bitgrid::{BitGrid, Coord};
use crate::hashlife::{
    GridExtractionError, GridExtractionPolicy, HashLifeCheckpoint, HashLifeSession,
    HashLifeSnapshotError,
};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};
use std::collections::HashMap;
use std::fmt;

const SIMD_GENERATION_LIMIT: u64 = 64;
const SIMD_POPULATION_LIMIT: usize = 512;
const SIMD_SPAN_LIMIT: Coord = 64;
const HASHLIFE_GENERATION_LIMIT: u64 = 512;
const HASHLIFE_DENSITY_LIMIT: f64 = 0.18;
const HYBRID_PREFIX_LIMIT: u64 = 64;
const EXACT_REPEAT_SKIP_GENERATION_LIMIT: u64 = 4_096;

type SeenStates = HashMap<NormalizedGridSignature, (u64, (Coord, Coord))>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimulationBackend {
    SimdChunk,
    HashLife,
    HybridSegmented,
}

impl fmt::Display for SimulationBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SimdChunk => "simd_chunk",
            Self::HashLife => "hashlife",
            Self::HybridSegmented => "hybrid_segmented",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvanceStats {
    pub backend: SimulationBackend,
    pub simd_generations: u64,
    pub hashlife_generations: u64,
    pub repeat_skip_events: u64,
    pub repeat_skip_generations: u64,
}

#[derive(Debug, Default)]
pub struct SimulationSession {
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

impl SimulationSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finish(&mut self) {
        self.hashlife_session.finish();
    }

    pub fn load_hashlife_state(&mut self, grid: &BitGrid) {
        self.hashlife_session.load_grid(grid);
        self.preferred_backend = Some(SimulationBackend::HashLife);
    }

    pub fn advance_hashlife_root(&mut self, generations: u64) -> AdvanceStats {
        self.hashlife_session.advance_root(generations);
        self.preferred_backend = Some(SimulationBackend::HashLife);
        AdvanceStats {
            backend: SimulationBackend::HashLife,
            simd_generations: 0,
            hashlife_generations: generations,
            repeat_skip_events: 0,
            repeat_skip_generations: 0,
        }
    }

    pub fn hashlife_loaded(&self) -> bool {
        self.hashlife_session.is_loaded()
    }

    pub fn load_hashlife_snapshot(&mut self, snapshot: &str) -> Result<(), HashLifeSnapshotError> {
        self.hashlife_session.load_snapshot_string(snapshot)?;
        self.preferred_backend = Some(SimulationBackend::HashLife);
        Ok(())
    }

    pub fn export_hashlife_snapshot(&mut self) -> Option<String> {
        self.hashlife_session.export_snapshot_string()
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

    pub fn hashlife_checkpoint(&mut self) -> Option<&HashLifeCheckpoint> {
        self.hashlife_session.signature_checkpoint()
    }

    pub fn shift_hashlife_origin(&mut self, dx: Coord, dy: Coord) {
        self.hashlife_session.shift_origin(dx, dy);
    }

    pub fn sample_hashlife_state_grid(
        &mut self,
        policy: GridExtractionPolicy,
    ) -> Result<BitGrid, GridExtractionError> {
        self.hashlife_session.extract_grid(policy)
    }

    pub fn sample_hashlife_state_region(
        &mut self,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Option<BitGrid> {
        self.hashlife_session
            .sample_region(min_x, min_y, max_x, max_y)
    }

    pub fn advance_simd_chunk_exact(
        &mut self,
        grid: &BitGrid,
        generations: u64,
    ) -> (BitGrid, AdvanceStats) {
        if generations == 0 {
            return (
                grid.clone(),
                AdvanceStats {
                    backend: SimulationBackend::SimdChunk,
                    simd_generations: 0,
                    hashlife_generations: 0,
                    repeat_skip_events: 0,
                    repeat_skip_generations: 0,
                },
            );
        }

        let mut current = grid.clone();
        let mut memo = Memo::default();
        let mut seen: SeenStates = HashMap::new();
        let mut generation = 0_u64;
        let mut repeat_skip_events = 0_u64;
        let mut repeat_skip_generations = 0_u64;

        while generation < generations {
            let (signature, origin) = normalize(&current);
            if let Some(&(first_seen, first_origin)) = seen.get(&signature) {
                let period = generation - first_seen;
                if period > 0 {
                    let remaining = generations - generation;
                    let skip_cycles = remaining / period;
                    if skip_cycles > 0 {
                        let dx = origin.0 - first_origin.0;
                        let dy = origin.1 - first_origin.1;
                        if dx == 0 && dy == 0 {
                            let skipped = skip_cycles * period;
                            generation += skipped;
                            repeat_skip_events += 1;
                            repeat_skip_generations += skipped;
                            continue;
                        }
                        let cycle_count =
                            Coord::try_from(skip_cycles).expect("simd repeat skip exceeded Coord");
                        current = current.translated(
                            dx.checked_mul(cycle_count).expect("simd repeat x overflow"),
                            dy.checked_mul(cycle_count).expect("simd repeat y overflow"),
                        );
                        let skipped = skip_cycles * period;
                        generation += skipped;
                        repeat_skip_events += 1;
                        repeat_skip_generations += skipped;
                        continue;
                    }
                }
            }
            seen.insert(signature, (generation, origin));
            current = step_grid_with_changes_and_memo(&current, &mut memo).0;
            memo.maybe_collect_transition_caches();
            generation += 1;
        }

        self.preferred_backend = Some(SimulationBackend::SimdChunk);
        (
            current,
            AdvanceStats {
                backend: SimulationBackend::SimdChunk,
                simd_generations: generations,
                hashlife_generations: 0,
                repeat_skip_events,
                repeat_skip_generations,
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn hashlife_sample_materializations(&self) -> usize {
        self.hashlife_session.sample_materializations()
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

    pub fn planned_backend_from_session_metrics(
        &mut self,
        population: usize,
        span: Coord,
        generations: u64,
    ) -> SimulationBackend {
        if generations == 0 || population == 0 {
            return SimulationBackend::SimdChunk;
        }

        if self.hashlife_loaded()
            && generations > 1
            && (self.hashlife_checkpoint().is_some()
                || matches!(
                    self.preferred_backend,
                    Some(SimulationBackend::HashLife | SimulationBackend::HybridSegmented)
                ))
        {
            return SimulationBackend::HashLife;
        }

        self.planned_backend_from_metrics(population, span, generations)
    }
}

pub fn should_use_exact_simd_repeat_skip(grid: &BitGrid, generations: u64) -> bool {
    if generations == 0 {
        return true;
    }
    if generations > EXACT_REPEAT_SKIP_GENERATION_LIMIT {
        return false;
    }
    matches!(
        select_backend(grid, generations),
        SimulationBackend::SimdChunk
    )
}
