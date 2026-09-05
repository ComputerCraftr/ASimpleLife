use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Coord};
use crate::hashlife::{
    GridExtractionError, GridExtractionPolicy, HashLifeAdvanceError, HashLifeAllocationFailure,
    HashLifeConversionError, HashLifeLimits, HashLifeMaterializationError, HashLifeSession,
    PopulationCount,
};
use crate::life::{CellStepWorkspace, step_grid_state_only_with_workspace};
use crate::memo::Memo;
use crate::recurrence::{
    ExactRecurrenceTracker, Lineage, Observation, ObserveOutcome, PeriodicCertificate,
    RecurrenceUnavailable,
};
use std::fmt;
use std::io::{Cursor, Read, Write};

#[cfg(test)]
mod recurrence_tests;

const HYBRID_PREFIX_LIMIT: u64 = 64;
const EXACT_REPEAT_SKIP_GENERATION_LIMIT: u64 = 4_096;
const CELL_PROBES_PER_CHUNK_GENERATION: u128 = 81;
const HASHLIFE_SETUP_BASE_WORK: u128 = 4_096;
const HASHLIFE_WORK_PER_CHUNK_LEVEL: u128 = 16;

#[derive(Debug, Default)]
enum SimulationAuthority {
    #[default]
    Unloaded,
    Cell {
        grid: BitGrid,
        generation: u64,
    },
    HashLife,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimulationConversionError {
    HashLife(HashLifeConversionError),
    Extraction(GridExtractionError),
    NoAuthoritativeState,
    PartialExtractionCannotBecomeAuthoritative,
}

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
    pub requested_generations: u64,
    pub completed_generations: u64,
    pub starting_generation: u64,
    pub reached_generation: u64,
    pub simd_generations: u64,
    pub hashlife_generations: u64,
    pub repeat_skip_events: u64,
    pub repeat_skip_generations: u64,
}

#[derive(Debug, Default)]
pub struct SimulationSession {
    hashlife_session: HashLifeSession,
    preferred_backend: Option<SimulationBackend>,
    authority: SimulationAuthority,
    state_revision: u64,
    cell_memo: Memo,
    cell_workspace: CellStepWorkspace,
    recurrence_lineage: Lineage,
    cell_recurrence: Option<ExactRecurrenceTracker>,
}

pub fn select_backend(grid: &BitGrid, generations: u64) -> SimulationBackend {
    if generations == 0 || grid.is_empty() {
        return SimulationBackend::SimdChunk;
    }

    let population = grid.population();
    let span = grid
        .bounds()
        .map(|(min_x, min_y, max_x, max_y)| {
            let width = i128::from(max_x) - i128::from(min_x) + 1;
            let height = i128::from(max_y) - i128::from(min_y) + 1;
            u128::try_from(width.max(height)).or_invariant("grid span should be positive")
        })
        .unwrap_or(0);
    planned_backend_for_work(population, grid.chunk_count(), span, generations)
}

fn planned_backend_for_work(
    population: usize,
    occupied_chunks: usize,
    span: u128,
    generations: u64,
) -> SimulationBackend {
    if generations <= 1 || population == 0 {
        return SimulationBackend::SimdChunk;
    }
    let chunks = occupied_chunks.max(1) as u128;
    let span = span.max(1);
    let root_level = u128::from(span.next_power_of_two().trailing_zeros().max(2));
    let cell_work = chunks
        .saturating_mul(u128::from(generations))
        .saturating_mul(CELL_PROBES_PER_CHUNK_GENERATION);
    let hashlife_work = HASHLIFE_SETUP_BASE_WORK.saturating_add(
        chunks
            .saturating_mul(root_level)
            .saturating_mul(HASHLIFE_WORK_PER_CHUNK_LEVEL),
    );

    // A 20% band avoids backend churn when the coarse estimates are close.
    if cell_work.saturating_mul(5) < hashlife_work.saturating_mul(4) {
        SimulationBackend::SimdChunk
    } else if hashlife_work.saturating_mul(5) < cell_work.saturating_mul(4) {
        SimulationBackend::HashLife
    } else {
        SimulationBackend::HybridSegmented
    }
}

impl SimulationSession {
    pub(crate) fn capture_analysis(
        &self,
        region: Option<crate::bitgrid::Bounds>,
        limits: crate::hashlife::session::capture::CaptureLimits,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<
        crate::hashlife::session::capture::OwnedDag,
        crate::hashlife::session::capture::CaptureError,
    > {
        self.hashlife_session
            .capture_analysis(region, limits, cancelled)
    }
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_hashlife_limits(limits: HashLifeLimits) -> Self {
        Self {
            hashlife_session: HashLifeSession::with_limits(limits),
            ..Self::default()
        }
    }

    pub fn set_hashlife_limits(&mut self, limits: HashLifeLimits) {
        self.hashlife_session.set_limits(limits);
    }

    pub fn configure_hashlife_allocation_failure(
        &mut self,
        failure: Option<HashLifeAllocationFailure>,
    ) {
        self.hashlife_session.configure_allocation_failure(failure);
    }

    pub fn finish(&mut self) {
        self.hashlife_session.finish();
    }

    pub fn load_cell_state(&mut self, grid: BitGrid, generation: u64) {
        self.hashlife_session.unload();
        self.authority = SimulationAuthority::Cell { grid, generation };
        self.recurrence_lineage = Lineage::fresh();
        self.preferred_backend = Some(SimulationBackend::SimdChunk);
        self.state_revision = self.state_revision.wrapping_add(1);
    }

    pub fn cell_state(&self) -> Option<(&BitGrid, u64)> {
        match &self.authority {
            SimulationAuthority::Cell { grid, generation } => Some((grid, *generation)),
            SimulationAuthority::Unloaded | SimulationAuthority::HashLife => None,
        }
    }

    pub fn try_convert_to_hashlife(&mut self) -> Result<(), SimulationConversionError> {
        let (grid, generation) = match &self.authority {
            SimulationAuthority::Cell { grid, generation } => (grid.clone(), *generation),
            SimulationAuthority::HashLife => return Ok(()),
            SimulationAuthority::Unloaded => {
                return Err(SimulationConversionError::NoAuthoritativeState);
            }
        };
        self.hashlife_session
            .try_load_grid_at_generation(&grid, generation)
            .map_err(SimulationConversionError::HashLife)?;
        self.preferred_backend = Some(SimulationBackend::HashLife);
        self.authority = SimulationAuthority::HashLife;
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(())
    }

    pub fn try_convert_to_cell(
        &mut self,
        policy: GridExtractionPolicy,
    ) -> Result<(), SimulationConversionError> {
        // Clipped samples are inspection data, never a replacement universe.
        if !matches!(policy, GridExtractionPolicy::FullGridIfUnder { .. }) {
            return Err(SimulationConversionError::PartialExtractionCannotBecomeAuthoritative);
        }
        if matches!(self.authority, SimulationAuthority::Cell { .. }) {
            return Ok(());
        }
        if matches!(self.authority, SimulationAuthority::Unloaded) {
            return Err(SimulationConversionError::NoAuthoritativeState);
        }
        let generation = self.hashlife_session.generation();
        let candidate = self
            .hashlife_session
            .try_extract_grid_for_conversion(policy)
            .map_err(|error| match error {
                HashLifeMaterializationError::Conversion(error) => {
                    SimulationConversionError::HashLife(error)
                }
                HashLifeMaterializationError::Extraction(error) => {
                    SimulationConversionError::Extraction(error)
                }
            })?;
        self.hashlife_session.unload();
        self.authority = SimulationAuthority::Cell {
            grid: candidate,
            generation,
        };
        self.preferred_backend = Some(SimulationBackend::SimdChunk);
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(())
    }

    pub fn try_load_hashlife_state(
        &mut self,
        grid: &BitGrid,
    ) -> Result<(), HashLifeConversionError> {
        self.hashlife_session.try_load_grid(grid)?;
        self.recurrence_lineage = Lineage::fresh();
        self.preferred_backend = Some(SimulationBackend::HashLife);
        self.authority = SimulationAuthority::HashLife;
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(())
    }

    pub(crate) fn try_load_hashlife_state_at_generation(
        &mut self,
        grid: &BitGrid,
        generation: u64,
    ) -> Result<(), HashLifeConversionError> {
        self.hashlife_session
            .try_load_grid_at_generation(grid, generation)?;
        self.recurrence_lineage = Lineage::fresh();
        self.preferred_backend = Some(SimulationBackend::HashLife);
        self.authority = SimulationAuthority::HashLife;
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(())
    }

    pub fn advance_hashlife_root(
        &mut self,
        generations: u64,
    ) -> Result<AdvanceStats, HashLifeAdvanceError> {
        self.advance_hashlife_root_controlled(generations, None)
    }

    pub(crate) fn advance_hashlife_root_controlled(
        &mut self,
        generations: u64,
        cancelled: Option<(
            &std::sync::Arc<std::sync::atomic::AtomicBool>,
            &std::sync::Arc<std::sync::atomic::AtomicBool>,
        )>,
    ) -> Result<AdvanceStats, HashLifeAdvanceError> {
        let starting_generation = self.hashlife_session.generation();
        let result = match cancelled {
            Some((token, stop)) => {
                self.hashlife_session
                    .advance_root_interruptible(generations, token, stop)
            }
            None => self.hashlife_session.advance_root(generations),
        };
        // Failed requests may have committed earlier segments. Revision follows
        // authoritative progress, never only the success/error category.
        if self.hashlife_session.generation() != starting_generation {
            self.state_revision = self.state_revision.wrapping_add(1);
        }
        let advanced = result?;
        self.preferred_backend = Some(SimulationBackend::HashLife);
        self.authority = SimulationAuthority::HashLife;
        Ok(AdvanceStats {
            backend: SimulationBackend::HashLife,
            requested_generations: advanced.requested_generations,
            completed_generations: advanced.completed_generations,
            starting_generation: advanced.starting_generation,
            reached_generation: advanced.reached_generation,
            simd_generations: 0,
            hashlife_generations: advanced.completed_generations,
            repeat_skip_events: 0,
            repeat_skip_generations: 0,
        })
    }

    pub fn hashlife_loaded(&self) -> bool {
        matches!(self.authority, SimulationAuthority::HashLife) && self.hashlife_session.is_loaded()
    }

    pub fn load_hashlife_snapshot(
        &mut self,
        snapshot: &str,
    ) -> Result<(), HashLifeConversionError> {
        self.load_hashlife_snapshot_reader(Cursor::new(snapshot.as_bytes()))
    }

    pub fn load_hashlife_snapshot_reader(
        &mut self,
        reader: impl Read,
    ) -> Result<(), HashLifeConversionError> {
        self.hashlife_session.load_snapshot_reader(reader)?;
        self.recurrence_lineage = Lineage::fresh();
        self.preferred_backend = Some(SimulationBackend::HashLife);
        self.authority = SimulationAuthority::HashLife;
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(())
    }

    pub fn load_hashlife_snapshot_owned(
        &mut self,
        snapshot: &crate::hashlife::OwnedHashLifeSnapshot,
    ) -> Result<(), HashLifeConversionError> {
        self.hashlife_session.load_snapshot_owned(snapshot)?;
        self.recurrence_lineage = Lineage::fresh();
        self.preferred_backend = Some(SimulationBackend::HashLife);
        self.authority = SimulationAuthority::HashLife;
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(())
    }

    pub fn export_hashlife_snapshot(
        &mut self,
    ) -> Result<Option<String>, crate::hashlife::HashLifeSnapshotError> {
        let Some(snapshot) = self.export_hashlife_snapshot_owned()? else {
            return Ok(None);
        };
        String::from_utf8(snapshot.into_bytes())
            .map(Some)
            .map_err(|_| {
                crate::hashlife::HashLifeSnapshotError::new("snapshot writer emitted invalid UTF-8")
            })
    }

    pub fn export_hashlife_snapshot_owned(
        &mut self,
    ) -> Result<
        Option<crate::hashlife::OwnedHashLifeSnapshot>,
        crate::hashlife::HashLifeSnapshotError,
    > {
        let mut bytes = Vec::new();
        if !self.write_hashlife_snapshot(&mut bytes)? {
            return Ok(None);
        }
        Ok(Some(crate::hashlife::OwnedHashLifeSnapshot::from_bytes(
            bytes,
        )))
    }

    pub fn write_hashlife_snapshot(
        &mut self,
        writer: &mut impl Write,
    ) -> Result<bool, crate::hashlife::HashLifeSnapshotError> {
        match &self.authority {
            SimulationAuthority::Unloaded => Ok(false),
            SimulationAuthority::HashLife => self.hashlife_session.write_snapshot(writer),
            SimulationAuthority::Cell { grid, generation } => {
                let mut snapshot_session =
                    HashLifeSession::with_limits(self.hashlife_session.limits());
                snapshot_session
                    .try_load_grid_at_generation(grid, *generation)
                    .map_err(|error| {
                        crate::hashlife::HashLifeSnapshotError::new(format!(
                            "snapshot cell conversion failed: {error:?}"
                        ))
                    })?;
                snapshot_session.write_snapshot(writer)
            }
        }
    }

    pub fn hashlife_generation(&self) -> u64 {
        self.hashlife_session.generation()
    }

    pub fn hashlife_population_count(&self) -> Option<PopulationCount> {
        self.hashlife_session.population_count()
    }

    pub fn hashlife_bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)> {
        self.hashlife_session.bounds()
    }

    pub fn hashlife_checkpoint(&mut self) -> Option<&crate::hashlife::HashLifeStateCheckpoint> {
        self.hashlife_session.signature_checkpoint()
    }

    pub(crate) const fn recurrence_lineage(&self) -> Lineage {
        self.recurrence_lineage
    }

    pub(crate) fn recurrence_observation(&mut self) -> Result<Observation, RecurrenceUnavailable> {
        match &self.authority {
            SimulationAuthority::Unloaded => Err(RecurrenceUnavailable::WitnessLimit),
            SimulationAuthority::Cell { grid, generation } => {
                Observation::from_grid(self.recurrence_lineage, *generation, grid)
            }
            SimulationAuthority::HashLife => self
                .hashlife_session
                .try_recurrence_observation(self.recurrence_lineage),
        }
    }

    pub(crate) fn apply_hashlife_recurrence(
        &mut self,
        certificate: PeriodicCertificate,
        target_generation: u64,
    ) -> Result<Option<crate::hashlife::SessionAdvanceStats>, RecurrenceUnavailable> {
        if !certificate.matches_lineage(self.recurrence_lineage) || !self.hashlife_loaded() {
            return Err(RecurrenceUnavailable::LineageMismatch);
        }
        let Some(skip) = certificate.checked_power(self.hashlife_generation(), target_generation)
        else {
            return Ok(None);
        };
        let advanced = self.hashlife_session.try_apply_recurrence_skip(skip)?;
        self.state_revision = self.state_revision.wrapping_add(1);
        Ok(Some(advanced))
    }

    pub fn shift_hashlife_origin(
        &mut self,
        dx: Coord,
        dy: Coord,
    ) -> Result<(), crate::hashlife::HashLifeGeometryError> {
        self.hashlife_session.shift_origin(dx, dy)
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

    pub(crate) fn inspect_viewport_region(
        &self,
        bounds: (i128, i128, i128, i128),
        remaining: &mut usize,
    ) -> Option<bool> {
        self.hashlife_session
            .viewport_region_occupied(bounds, remaining)
    }

    pub(crate) fn viewport_root_bounds(&self) -> Option<crate::bitgrid::Bounds> {
        self.hashlife_session.viewport_root_bounds()
    }

    pub(crate) fn inspect_viewport_neighborhood(
        &self,
        tile: (Coord, Coord),
        remaining: &mut usize,
    ) -> Option<[u64; 9]> {
        self.hashlife_session.viewport_neighborhood(tile, remaining)
    }

    pub(crate) fn record_hashlife_oracle_confirmation_materialization(&mut self) {
        self.hashlife_session
            .record_oracle_confirmation_materialization();
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
                    requested_generations: 0,
                    completed_generations: 0,
                    starting_generation: 0,
                    reached_generation: 0,
                    simd_generations: 0,
                    hashlife_generations: 0,
                    repeat_skip_events: 0,
                    repeat_skip_generations: 0,
                },
            );
        }

        let mut current = grid.clone();
        // This compatibility entrypoint accepts an arbitrary grid rather than
        // the session's state, so every private batch starts a new lineage.
        let lineage = Lineage::fresh();
        let recurrence = self
            .cell_recurrence
            .get_or_insert_with(|| ExactRecurrenceTracker::new(lineage));
        recurrence.reset(lineage);
        let mut generation = 0_u64;
        let mut repeat_skip_events = 0_u64;
        let mut repeat_skip_generations = 0_u64;

        while generation < generations {
            if let ObserveOutcome::Repeated(certificate) =
                recurrence.observe_result(Observation::from_grid(lineage, generation, &current))
                && let Some(skip) = certificate.checked_power(generation, generations)
                && let Ok(candidate) = skip.try_translate_grid(&current)
            {
                // Both the translated candidate and the checked power exist
                // before publishing any progress, even within this private batch.
                current = candidate;
                generation += skip.committed_generations();
                repeat_skip_events += 1;
                repeat_skip_generations += skip.committed_generations();
                continue;
            }
            current = step_grid_state_only_with_workspace(
                &current,
                &mut self.cell_memo,
                &mut self.cell_workspace,
            );
            self.cell_memo.maybe_collect_transition_caches();
            generation += 1;
        }

        // Keep the prior authority intact until the complete cell result exists.
        // Allocation failure currently aborts the operation, but no recoverable path
        // may publish a partially converted representation.
        self.hashlife_session.unload();
        self.authority = SimulationAuthority::Cell {
            grid: current.clone(),
            generation: generations,
        };
        self.recurrence_lineage = lineage;
        self.state_revision = self.state_revision.wrapping_add(1);
        self.preferred_backend = Some(SimulationBackend::SimdChunk);
        (
            current,
            AdvanceStats {
                backend: SimulationBackend::SimdChunk,
                requested_generations: generations,
                completed_generations: generations,
                starting_generation: 0,
                reached_generation: generations,
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

    #[cfg(test)]
    pub(crate) fn hashlife_runtime_stats(&self) -> crate::hashlife::HashLifeRuntimeStats {
        self.hashlife_session.runtime_stats()
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

        match self.preferred_backend {
            Some(SimulationBackend::HashLife) if self.hashlife_loaded() && generations > 1 => {
                return SimulationBackend::HashLife;
            }
            Some(SimulationBackend::HybridSegmented)
                if self.hashlife_loaded() && generations > HYBRID_PREFIX_LIMIT =>
            {
                return SimulationBackend::HashLife;
            }
            _ => {}
        }
        planned_backend_for_work(
            population,
            population.div_ceil(64),
            u128::try_from(span.max(1)).or_invariant("positive span should fit u128"),
            generations,
        )
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
