use super::geometry::{HashLifeGeometryError, MAX_COORD_ROOT_LEVEL, RootGeometry};
use super::memory::{
    AllocationGate, HashLifeAllocationClass, HashLifeMemoryBudget, wide_allocated_bytes,
};
use super::session_types::*;
use super::{
    EngineAllocationFailure, GridExtractionError, GridExtractionPolicy,
    HASHLIFE_FULL_GRID_MAX_CHUNKS, HASHLIFE_FULL_GRID_MAX_POPULATION, HashLifeEngine,
    HashLifeSnapshotError, HashLifeStateCheckpoint, HashLifeStateIdentity, NodeId, PopulationCount,
};
use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Coord};
use std::io::{Cursor, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};

mod cancellation;
pub(crate) mod capture;
mod collection;
mod conversion;
mod inspection;
mod recurrence;

const MAX_COORD_STEP_EXP: u32 = MAX_COORD_ROOT_LEVEL - 2;

#[derive(Clone, Copy)]
struct PublishedSessionState {
    current_root: Option<NodeId>,
    current_origin_x: Coord,
    current_origin_y: Coord,
    current_generation: u64,
    root_is_centered: bool,
    sampled_bounds: Option<Option<(Coord, Coord, Coord, Coord)>>,
    sampled_checkpoint: Option<Option<HashLifeStateCheckpoint>>,
    checkpoint_epoch: u64,
    checkpoint_level: Option<u32>,
}

#[derive(Debug)]
pub struct HashLifeSession {
    engine: HashLifeEngine,
    active_run: bool,
    previous_root: Option<NodeId>,
    current_root: Option<NodeId>,
    current_origin_x: Coord,
    current_origin_y: Coord,
    current_generation: u64,
    root_is_centered: bool,
    sampled_bounds: Option<Option<(Coord, Coord, Coord, Coord)>>,
    sampled_checkpoint: Option<Option<HashLifeStateCheckpoint>>,
    checkpoint_session: u64,
    checkpoint_epoch: u64,
    checkpoint_level: Option<u32>,
    limits: HashLifeLimits,
    memory_budget: HashLifeMemoryBudget,
    allocation_gate: AllocationGate,
    #[cfg(test)]
    sample_materializations: usize,
    #[cfg(test)]
    checkpoint_profile: HashLifeCheckpointProfile,
}

impl Default for HashLifeSession {
    fn default() -> Self {
        static NEXT_CHECKPOINT_SESSION: AtomicU64 = AtomicU64::new(1);

        let engine = HashLifeEngine {
            allocation_hard_limit: DEFAULT_HARD_MEMORY_BYTES,
            ..HashLifeEngine::default()
        };
        let retained = wide_allocated_bytes(engine.allocated_bytes());
        let mut memory_budget = HashLifeMemoryBudget::new(DEFAULT_HARD_MEMORY_BYTES);
        memory_budget.sync_retained(retained);
        Self {
            engine,
            active_run: false,
            previous_root: None,
            current_root: None,
            current_origin_x: 0,
            current_origin_y: 0,
            current_generation: 0,
            root_is_centered: false,
            sampled_bounds: None,
            sampled_checkpoint: None,
            checkpoint_session: NEXT_CHECKPOINT_SESSION.fetch_add(1, Ordering::Relaxed),
            checkpoint_epoch: 0,
            checkpoint_level: None,
            limits: HashLifeLimits::default(),
            memory_budget,
            allocation_gate: AllocationGate::default(),
            #[cfg(test)]
            sample_materializations: 0,
            #[cfg(test)]
            checkpoint_profile: HashLifeCheckpointProfile::default(),
        }
    }
}

impl HashLifeSession {
    pub fn new() -> Self {
        Self::with_id_capacity_headroom(usize::MAX, usize::MAX)
    }

    pub fn with_limits(limits: HashLifeLimits) -> Self {
        let mut session = Self::default();
        session.set_limits(limits);
        session
    }

    pub fn limits(&self) -> HashLifeLimits {
        self.limits
    }

    pub fn set_limits(&mut self, limits: HashLifeLimits) {
        self.limits = HashLifeLimits {
            soft_memory_bytes: limits.soft_memory_bytes.min(limits.hard_memory_bytes),
            hard_memory_bytes: limits.hard_memory_bytes,
        };
        self.memory_budget
            .set_hard_limit(self.limits.hard_memory_bytes);
        self.engine.allocation_hard_limit = self.limits.hard_memory_bytes;
    }

    pub fn allocated_bytes(&self) -> u128 {
        wide_allocated_bytes(self.engine.allocated_bytes())
    }

    pub fn execution_stats(&self) -> HashLifeExecutionStats {
        let materialization = self.engine.stats.materialization;
        HashLifeExecutionStats {
            allocated_bytes: wide_allocated_bytes(self.engine.allocated_bytes()),
            nodes: self.engine.node_count(),
            arena_epoch: self.engine.arena_epoch,
            gc_runs: self.engine.stats.gc.gc_runs,
            dependency_stalls: self.engine.stats.scheduler.dependency_stalls,
            materializations: materialization.session_full_grid_materializations
                + materialization.embedded_result_full_extractions
                + materialization.clipped_viewport_extractions
                + materialization.checkpoint_cell_materializations
                + materialization.oracle_confirmation_materializations,
            candidate_lanes: self.engine.stats.simd.kernel.candidate_lanes,
            portable_vector_lanes: self.engine.stats.simd.kernel.portable_vector_lanes,
            vectorized_structural_lanes: self.engine.stats.simd.kernel.vectorized_structural_lanes,
            scalar_fallback_lanes: self.engine.stats.simd.kernel.scalar_fallback_lanes,
            native_avx2_lanes: self.engine.stats.simd.kernel.native_avx2_lanes,
            native_neon_lanes: self.engine.stats.simd.kernel.native_neon_lanes,
            d4_candidate_lanes: self.engine.stats.simd.kernel.d4_candidate_lanes,
            native_d4_candidate_lanes: self.engine.stats.simd.kernel.native_d4_candidate_lanes,
            native_d4_prefix_compare_lanes: self
                .engine
                .stats
                .simd
                .kernel
                .native_d4_prefix_compare_lanes,
            native_d4_exact_winner_lanes: self
                .engine
                .stats
                .simd
                .kernel
                .native_d4_exact_winner_lanes,
            swar_control_groups: self.engine.stats.simd.kernel.swar_control_groups,
            native_avx2_control_groups: self.engine.stats.simd.kernel.native_avx2_control_groups,
            native_neon_control_groups: self.engine.stats.simd.kernel.native_neon_control_groups,
        }
    }

    fn ensure_active_run(&mut self) {
        if self.active_run {
            return;
        }
        self.previous_root = self.engine.begin_persistent_run();
        self.active_run = true;
    }

    fn published_state(&self) -> PublishedSessionState {
        PublishedSessionState {
            current_root: self.current_root,
            current_origin_x: self.current_origin_x,
            current_origin_y: self.current_origin_y,
            current_generation: self.current_generation,
            root_is_centered: self.root_is_centered,
            sampled_bounds: self.sampled_bounds,
            sampled_checkpoint: self.sampled_checkpoint,
            checkpoint_epoch: self.checkpoint_epoch,
            checkpoint_level: self.checkpoint_level,
        }
    }

    fn restore_published_state(&mut self, state: PublishedSessionState) {
        self.current_root = state.current_root;
        self.current_origin_x = state.current_origin_x;
        self.current_origin_y = state.current_origin_y;
        self.current_generation = state.current_generation;
        self.root_is_centered = state.root_is_centered;
        self.sampled_bounds = state.sampled_bounds;
        self.sampled_checkpoint = state.sampled_checkpoint;
        self.checkpoint_epoch = state.checkpoint_epoch;
        self.checkpoint_level = state.checkpoint_level;
    }

    fn finish_active_run(&mut self) {
        if !self.active_run {
            return;
        }
        let arena_epoch = self.engine.arena_epoch;
        self.engine.finish_persistent_run(
            self.previous_root,
            self.current_root,
            self.limits.hard_memory_bytes,
        );
        if self.engine.arena_epoch != arena_epoch {
            self.current_root = self.engine.retained_roots.last().copied();
            self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
            self.clear_cached_samples();
        }
        self.active_run = false;
        self.previous_root = None;
    }

    pub fn generation(&self) -> u64 {
        self.current_generation
    }

    pub fn is_loaded(&self) -> bool {
        self.current_root.is_some()
    }

    pub(crate) fn unload(&mut self) {
        self.active_run = false;
        self.previous_root = None;
        self.engine.retained_roots.clear();
        self.current_root = None;
        self.current_generation = 0;
        self.current_origin_x = 0;
        self.current_origin_y = 0;
        self.root_is_centered = false;
        self.clear_cached_samples();
    }

    pub fn population_count(&self) -> Option<PopulationCount> {
        self.current_root
            .map(|root| self.engine.node_columns.population_count(root))
    }

    pub fn origin(&self) -> Option<(Coord, Coord)> {
        self.current_root
            .map(|_| (self.current_origin_x, self.current_origin_y))
    }

    pub fn bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)> {
        self.try_bounds().ok().flatten()
    }

    pub fn try_bounds(
        &mut self,
    ) -> Result<Option<(Coord, Coord, Coord, Coord)>, HashLifeGeometryError> {
        if let Some(bounds) = self.sampled_bounds {
            return Ok(bounds);
        }
        let Some(root) = self.current_root else {
            return Ok(None);
        };
        RootGeometry::new(
            self.engine.node_columns.level(root),
            self.current_origin_x,
            self.current_origin_y,
        )?;
        let bounds = self
            .engine
            .node_bounds(root, self.current_origin_x, self.current_origin_y);
        self.sampled_bounds = Some(bounds);
        Ok(bounds)
    }

    pub fn shift_origin(&mut self, dx: Coord, dy: Coord) -> Result<(), HashLifeGeometryError> {
        if self.current_root.is_none() {
            return Ok(());
        }
        let origin_x = self
            .current_origin_x
            .checked_add(dx)
            .ok_or(HashLifeGeometryError::CoordinateRangeExceeded { axis: "x" })?;
        let origin_y = self
            .current_origin_y
            .checked_add(dy)
            .ok_or(HashLifeGeometryError::CoordinateRangeExceeded { axis: "y" })?;
        let root = self.current_root.or_invariant("loaded root must exist");
        RootGeometry::new(self.engine.node_columns.level(root), origin_x, origin_y)?;
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.clear_cached_samples();
        Ok(())
    }

    pub fn advance_root(
        &mut self,
        generations: u64,
    ) -> Result<SessionAdvanceStats, HashLifeAdvanceError> {
        let starting_generation = self.current_generation;
        let Some(_) = self.current_root else {
            return Err(HashLifeAdvanceError::NotLoaded {
                starting_generation,
                requested_delta: generations,
                completed_generations: 0,
                reached_generation: starting_generation,
            });
        };
        if generations == 0 {
            return Ok(SessionAdvanceStats {
                requested_generations: 0,
                completed_generations: 0,
                starting_generation,
                reached_generation: starting_generation,
            });
        }
        let requested_generation = self.current_generation.checked_add(generations).ok_or(
            HashLifeAdvanceError::GenerationOverflow {
                starting_generation,
                requested_delta: generations,
                completed_generations: 0,
                reached_generation: starting_generation,
            },
        )?;
        self.collect_before_allocation();
        let allocated_bytes = wide_allocated_bytes(self.engine.allocated_bytes());
        if allocated_bytes > self.limits.hard_memory_bytes {
            return Err(HashLifeAdvanceError::MemoryBudgetExceeded {
                starting_generation,
                requested_delta: generations,
                completed_generations: 0,
                requested_generation,
                reached_generation: starting_generation,
                allocated_bytes,
                limit_bytes: self.limits.hard_memory_bytes,
            });
        }
        self.ensure_active_run();
        let mut remaining = generations;
        while remaining != 0 {
            self.collect_before_allocation();
            if self
                .allocation_gate
                .check(HashLifeAllocationClass::ArenaGrowth, 1)
                .is_err()
            {
                return Err(HashLifeAdvanceError::AllocationFailed {
                    starting_generation,
                    requested_delta: generations,
                    completed_generations: self.current_generation - starting_generation,
                    reached_generation: self.current_generation,
                    requested_bytes: 1,
                });
            }
            let segment_root = self.current_root;
            let segment_origin_x = self.current_origin_x;
            let segment_origin_y = self.current_origin_y;
            let segment_generation = self.current_generation;
            let segment_centered = self.root_is_centered;
            let saved_segment = (
                segment_root,
                segment_origin_x,
                segment_origin_y,
                segment_generation,
                segment_centered,
            );
            self.engine
                .begin_allocation_transaction(self.limits.hard_memory_bytes);
            // Apply low powers before large expanding jumps. F^a and F^b commute for
            // one deterministic transition, while this order keeps early roots compact.
            let desired_step_exp = remaining.trailing_zeros().min(MAX_COORD_STEP_EXP);
            if let Err(required_level) = self.crop_empty_outer_quadrants() {
                if let Some(error) = self.take_segment_allocation_error(
                    saved_segment,
                    starting_generation,
                    generations,
                ) {
                    return Err(error);
                }
                self.restore_segment_state(
                    segment_root,
                    segment_origin_x,
                    segment_origin_y,
                    segment_generation,
                    segment_centered,
                );
                return Err(self.coordinate_error(
                    generations,
                    starting_generation,
                    required_level,
                ));
            }
            if let Err(required_level) = self.ensure_centered_capacity(desired_step_exp) {
                if let Some(error) = self.take_segment_allocation_error(
                    saved_segment,
                    starting_generation,
                    generations,
                ) {
                    return Err(error);
                }
                self.restore_segment_state(
                    segment_root,
                    segment_origin_x,
                    segment_origin_y,
                    segment_generation,
                    segment_centered,
                );
                return Err(self.coordinate_error(
                    generations,
                    starting_generation,
                    required_level,
                ));
            }
            if let Some(error) =
                self.take_segment_allocation_error(saved_segment, starting_generation, generations)
            {
                return Err(error);
            }
            let root = self
                .current_root
                .or_invariant("hashlife session root disappeared");
            let level = self.engine.node_columns.level(root);
            let step_exp = desired_step_exp.min(level.saturating_sub(2));
            let step = 1_u64 << step_exp;
            let Some(root_size) = 1_i64.checked_shl(level) else {
                self.restore_segment_state(
                    segment_root,
                    segment_origin_x,
                    segment_origin_y,
                    segment_generation,
                    segment_centered,
                );
                return Err(self.coordinate_error(generations, starting_generation, level));
            };
            let advanced = self.engine.advance_pow2(root, step_exp);
            if let Some(error) =
                self.take_segment_allocation_error(saved_segment, starting_generation, generations)
            {
                return Err(error);
            }
            let staged_origin_x = self.current_origin_x.checked_add(root_size / 4);
            let staged_origin_y = self.current_origin_y.checked_add(root_size / 4);
            let staged_generation = self.current_generation.checked_add(step);
            let (Some(staged_origin_x), Some(staged_origin_y), Some(staged_generation)) =
                (staged_origin_x, staged_origin_y, staged_generation)
            else {
                self.restore_segment_state(
                    segment_root,
                    segment_origin_x,
                    segment_origin_y,
                    segment_generation,
                    segment_centered,
                );
                return Err(self.coordinate_error(generations, starting_generation, level));
            };
            let allocated_bytes = wide_allocated_bytes(self.engine.allocated_bytes());
            if allocated_bytes > self.limits.hard_memory_bytes {
                self.restore_segment_state(
                    segment_root,
                    segment_origin_x,
                    segment_origin_y,
                    segment_generation,
                    segment_centered,
                );
                return Err(HashLifeAdvanceError::MemoryBudgetExceeded {
                    starting_generation,
                    requested_delta: generations,
                    completed_generations: segment_generation - starting_generation,
                    requested_generation,
                    reached_generation: segment_generation,
                    allocated_bytes,
                    limit_bytes: self.limits.hard_memory_bytes,
                });
            }

            self.current_origin_x = staged_origin_x;
            self.current_origin_y = staged_origin_y;
            self.current_root = Some(advanced);
            self.current_generation = staged_generation;
            self.root_is_centered = false;
            self.clear_cached_samples();
            remaining -= step;
            let mut allocated_bytes = allocated_bytes;
            if allocated_bytes > self.limits.soft_memory_bytes {
                let arena_epoch = self.engine.arena_epoch;
                self.current_root = self.engine.maybe_collect_active_run(
                    self.current_root,
                    self.limits.soft_memory_bytes,
                    self.limits.hard_memory_bytes,
                );
                if self.engine.arena_epoch != arena_epoch {
                    self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
                    self.previous_root = self.engine.retained_roots.last().copied();
                }
                allocated_bytes = wide_allocated_bytes(self.engine.allocated_bytes());
            }
            if allocated_bytes > self.limits.hard_memory_bytes {
                return Err(HashLifeAdvanceError::MemoryBudgetExceeded {
                    starting_generation,
                    requested_delta: generations,
                    completed_generations: self.current_generation - starting_generation,
                    requested_generation,
                    reached_generation: self.current_generation,
                    allocated_bytes,
                    limit_bytes: self.limits.hard_memory_bytes,
                });
            }
        }
        let arena_epoch = self.engine.arena_epoch;
        self.current_root = self.engine.maybe_collect_active_run(
            self.current_root,
            self.limits.soft_memory_bytes,
            self.limits.hard_memory_bytes,
        );
        if self.engine.arena_epoch != arena_epoch {
            self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
            self.previous_root = self.engine.retained_roots.last().copied();
        }
        let allocated_bytes = wide_allocated_bytes(self.engine.allocated_bytes());
        if allocated_bytes > self.limits.hard_memory_bytes {
            return Err(HashLifeAdvanceError::MemoryBudgetExceeded {
                starting_generation,
                requested_delta: generations,
                completed_generations: self.current_generation - starting_generation,
                requested_generation,
                reached_generation: self.current_generation,
                allocated_bytes,
                limit_bytes: self.limits.hard_memory_bytes,
            });
        }
        debug_assert_eq!(self.current_generation, requested_generation);
        Ok(SessionAdvanceStats {
            requested_generations: generations,
            completed_generations: self.current_generation - starting_generation,
            starting_generation,
            reached_generation: self.current_generation,
        })
    }

    fn coordinate_error(
        &self,
        requested_delta: u64,
        starting_generation: u64,
        required_level: u32,
    ) -> HashLifeAdvanceError {
        HashLifeAdvanceError::CoordinateRangeExceeded {
            starting_generation,
            requested_delta,
            completed_generations: self.current_generation - starting_generation,
            reached_generation: self.current_generation,
            required_level,
        }
    }

    fn restore_segment_state(
        &mut self,
        root: Option<NodeId>,
        origin_x: Coord,
        origin_y: Coord,
        generation: u64,
        centered: bool,
    ) {
        self.current_root = root;
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = generation;
        self.root_is_centered = centered;
        self.clear_cached_samples();
    }

    fn take_segment_allocation_error(
        &mut self,
        saved: (Option<NodeId>, Coord, Coord, u64, bool),
        starting_generation: u64,
        requested_delta: u64,
    ) -> Option<HashLifeAdvanceError> {
        self.engine.poll_advance_cancellation();
        let failure = self.engine.take_allocation_failure()?;
        let (root, origin_x, origin_y, generation, centered) = saved;
        self.restore_segment_state(root, origin_x, origin_y, generation, centered);
        self.engine.clear_transient_state(false);
        let completed_generations = generation - starting_generation;
        Some(match failure {
            EngineAllocationFailure::Cancelled => HashLifeAdvanceError::Cancelled {
                starting_generation,
                requested_delta,
                completed_generations,
                reached_generation: generation,
            },
            EngineAllocationFailure::Allocation { requested_bytes } => {
                HashLifeAdvanceError::AllocationFailed {
                    starting_generation,
                    requested_delta,
                    completed_generations,
                    reached_generation: generation,
                    requested_bytes,
                }
            }
            EngineAllocationFailure::NodeIdExhausted => HashLifeAdvanceError::NodeIdExhausted {
                starting_generation,
                requested_delta,
                completed_generations,
                reached_generation: generation,
            },
            EngineAllocationFailure::CanonicalReferenceExhausted => {
                HashLifeAdvanceError::CanonicalReferenceExhausted {
                    starting_generation,
                    requested_delta,
                    completed_generations,
                    reached_generation: generation,
                }
            }
        })
    }

    fn crop_empty_outer_quadrants(&mut self) -> Result<(), u32> {
        let Some(mut root) = self.current_root else {
            return Ok(());
        };
        let Some((min_x, min_y, max_x, max_y)) =
            self.engine
                .node_bounds(root, self.current_origin_x, self.current_origin_y)
        else {
            return Ok(());
        };

        loop {
            let level = self.engine.node_columns.level(root);
            if level == 0 {
                break;
            }
            let child_size = 1_i64.checked_shl(level - 1).ok_or(level)?;
            let split_x = self.current_origin_x.checked_add(child_size).ok_or(level)?;
            let split_y = self.current_origin_y.checked_add(child_size).ok_or(level)?;
            let column = if max_x < split_x {
                0
            } else if min_x >= split_x {
                1
            } else {
                break;
            };
            let row = if max_y < split_y {
                0
            } else if min_y >= split_y {
                1
            } else {
                break;
            };
            root = self.engine.node_columns.quadrants(root)[row * 2 + column];
            if column != 0 {
                self.current_origin_x = self
                    .current_origin_x
                    .checked_add(child_size)
                    .or_invariant("HashLife cropped origin x overflow");
            }
            if row != 0 {
                self.current_origin_y = self
                    .current_origin_y
                    .checked_add(child_size)
                    .or_invariant("HashLife cropped origin y overflow");
            }
        }

        if self.current_root != Some(root) {
            self.current_root = Some(root);
            self.root_is_centered = false;
            self.clear_cached_samples();
        }
        Ok(())
    }

    fn ensure_centered_capacity(&mut self, desired_step_exp: u32) -> Result<(), u32> {
        let Some(mut root) = self.current_root else {
            return Ok(());
        };
        let initial_level = self.engine.node_columns.level(root);
        let initial_size = 1_i64.checked_shl(initial_level).ok_or(initial_level)?;
        let growth_margin = 1_i64
            .checked_shl(desired_step_exp)
            .ok_or(desired_step_exp)?;
        let origin_x = i128::from(self.current_origin_x);
        let origin_y = i128::from(self.current_origin_y);
        let input_size = i128::from(initial_size);
        let growth_margin = i128::from(growth_margin);
        let needs_live_margin = self
            .engine
            .node_bounds(root, self.current_origin_x, self.current_origin_y)
            .is_some_and(|(min_x, min_y, max_x, max_y)| {
                i128::from(min_x) - origin_x <= growth_margin
                    || i128::from(min_y) - origin_y <= growth_margin
                    || origin_x + input_size - 1 - i128::from(max_x) <= growth_margin
                    || origin_y + input_size - 1 - i128::from(max_y) <= growth_margin
            });
        let minimum_input_level = initial_level
            .checked_add(1 + u32::from(needs_live_margin))
            .ok_or(u32::MAX)?;
        let required_level = desired_step_exp
            .checked_add(2)
            .ok_or(u32::MAX)?
            .max(minimum_input_level);
        if required_level > MAX_COORD_ROOT_LEVEL {
            return Err(required_level);
        }

        loop {
            let level = self.engine.node_columns.level(root);
            let needs_expansion = !self.root_is_centered
                || level < desired_step_exp + 2
                || level < minimum_input_level;
            if !needs_expansion {
                break;
            }
            root = self.center_expand_root(root)?;
            self.current_root = Some(root);
            self.root_is_centered = true;
            self.clear_cached_samples();
        }
        Ok(())
    }

    fn center_expand_root(&mut self, root: NodeId) -> Result<NodeId, u32> {
        let level = self.engine.node_columns.level(root);
        if level >= MAX_COORD_ROOT_LEVEL {
            return Err(level + 1);
        }
        let child_size = if level == 0 {
            1
        } else {
            1_i64.checked_shl(level - 1).ok_or(level)?
        };
        let origin_x = self
            .current_origin_x
            .checked_sub(child_size)
            .ok_or(level + 1)?;
        let origin_y = self
            .current_origin_y
            .checked_sub(child_size)
            .ok_or(level + 1)?;
        RootGeometry::new(level + if level == 0 { 2 } else { 1 }, origin_x, origin_y)
            .map_err(|_| level + 1)?;
        let expanded = self.engine.centered_shell(root);
        if self.engine.allocation_failed() {
            return Err(level + 1);
        }
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        Ok(expanded)
    }

    pub fn signature_checkpoint(&mut self) -> Option<&HashLifeStateCheckpoint> {
        self.try_signature_checkpoint().ok().flatten()
    }

    pub fn try_signature_checkpoint(
        &mut self,
    ) -> Result<Option<&HashLifeStateCheckpoint>, HashLifeGeometryError> {
        if self.sampled_checkpoint.is_none() {
            let Some(current_root) = self.current_root else {
                return Ok(None);
            };
            #[cfg(test)]
            {
                self.checkpoint_profile.metadata_reads += 1;
            }
            let current_level = self.engine.node_columns.level(current_root);
            let target_level = match self.checkpoint_level {
                Some(level) if current_level <= level => level,
                Some(_) => {
                    self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
                    current_level
                }
                None => current_level
                    .checked_add(1)
                    .filter(|level| *level <= MAX_COORD_ROOT_LEVEL)
                    .unwrap_or(current_level),
            };
            self.checkpoint_level = Some(target_level);

            let saved_origin = (self.current_origin_x, self.current_origin_y);
            let mut root = current_root;
            while self.engine.node_columns.level(root) < target_level {
                let Ok(expanded) = self.center_expand_root(root) else {
                    root = current_root;
                    self.current_origin_x = saved_origin.0;
                    self.current_origin_y = saved_origin.1;
                    self.checkpoint_level = Some(current_level);
                    break;
                };
                root = expanded;
            }
            let origin = (self.current_origin_x, self.current_origin_y);
            self.current_origin_x = saved_origin.0;
            self.current_origin_y = saved_origin.1;

            let level = self.engine.node_columns.level(root);
            let geometry = RootGeometry::new(level, origin.0, origin.1)?;
            let root_span = Coord::try_from(geometry.level.span()).map_err(|_| {
                HashLifeGeometryError::CoordinateRangeExceeded { axis: "checkpoint" }
            })?;
            self.sampled_checkpoint = Some(Some(HashLifeStateCheckpoint {
                generation: self.current_generation,
                origin,
                identity: HashLifeStateIdentity {
                    session: self.checkpoint_session,
                    epoch: self.checkpoint_epoch,
                    root: u64::from(root),
                    level,
                },
                population: self.engine.node_columns.population(root),
                root_span,
            }));
        }
        Ok(self.sampled_checkpoint.as_ref().and_then(Option::as_ref))
    }

    pub fn sample_grid(&mut self) -> Result<BitGrid, GridExtractionError> {
        self.extract_grid(default_full_grid_policy())
    }

    pub fn export_snapshot_string(&mut self) -> Result<Option<String>, HashLifeSnapshotError> {
        let Some(snapshot) = self.export_snapshot_owned()? else {
            return Ok(None);
        };
        String::from_utf8(snapshot.into_bytes())
            .map(Some)
            .map_err(|_| HashLifeSnapshotError::new("snapshot writer emitted invalid UTF-8"))
    }

    pub fn export_snapshot_owned(
        &mut self,
    ) -> Result<Option<super::OwnedHashLifeSnapshot>, HashLifeSnapshotError> {
        let mut bytes = Vec::new();
        if !self.write_snapshot(&mut bytes)? {
            return Ok(None);
        }
        Ok(Some(super::OwnedHashLifeSnapshot { bytes }))
    }

    pub fn write_snapshot(
        &mut self,
        writer: &mut impl Write,
    ) -> Result<bool, HashLifeSnapshotError> {
        let Some(root) = self.current_root else {
            return Ok(false);
        };
        let estimated_bytes = (self.engine.node_count() as u128)
            .saturating_mul(256)
            .saturating_add(1_024);
        self.allocation_gate
            .check(HashLifeAllocationClass::SnapshotExport, estimated_bytes)
            .map_err(|_| HashLifeSnapshotError::allocation(estimated_bytes))?;
        self.engine
            .begin_allocation_transaction(self.limits.hard_memory_bytes);
        let exported = self.engine.write_snapshot(
            root,
            self.current_origin_x,
            self.current_origin_y,
            self.current_generation,
            writer,
        );
        let failure = self.engine.take_allocation_failure();
        if let Some(failure) = failure {
            let requested_bytes = match failure {
                EngineAllocationFailure::Allocation { requested_bytes } => requested_bytes,
                EngineAllocationFailure::Cancelled
                | EngineAllocationFailure::NodeIdExhausted
                | EngineAllocationFailure::CanonicalReferenceExhausted => 0,
            };
            return Err(HashLifeSnapshotError::allocation(requested_bytes));
        }
        exported.map(|()| true)
    }

    pub fn extract_grid(
        &mut self,
        policy: GridExtractionPolicy,
    ) -> Result<BitGrid, GridExtractionError> {
        let root = self.current_root.ok_or(GridExtractionError::NotLoaded)?;
        #[cfg(test)]
        {
            self.sample_materializations += 1;
        }
        let grid =
            self.engine
                .node_to_grid(root, self.current_origin_x, self.current_origin_y, policy)?;
        if matches!(policy, GridExtractionPolicy::FullGridIfUnder { .. }) {
            self.engine
                .stats
                .materialization
                .session_full_grid_materializations += 1;
        }
        Ok(grid)
    }

    pub fn sample_region(
        &mut self,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Option<BitGrid> {
        let root = self.current_root?;
        self.engine
            .stats
            .materialization
            .clipped_viewport_extractions += 1;
        Some(self.engine.node_to_grid_clipped(
            root,
            self.current_origin_x,
            self.current_origin_y,
            (min_x, min_y, max_x, max_y),
        ))
    }

    pub fn finish(&mut self) {
        self.finish_active_run();
        self.engine.retained_roots.clear();
        self.current_root = None;
        self.current_generation = 0;
        self.checkpoint_level = None;
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
    }

    #[cfg(test)]
    pub(crate) fn sample_materializations(&self) -> usize {
        self.sample_materializations
    }

    #[cfg(test)]
    pub(crate) fn checkpoint_profile(&self) -> HashLifeCheckpointProfile {
        self.checkpoint_profile
    }

    #[cfg(test)]
    pub(crate) fn runtime_stats(&self) -> super::HashLifeRuntimeStats {
        self.engine.runtime_stats()
    }

    pub(crate) fn record_oracle_confirmation_materialization(&mut self) {
        self.engine
            .stats
            .materialization
            .oracle_confirmation_materializations += 1;
    }

    fn clear_cached_samples(&mut self) {
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
    }
}

fn default_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: u128::from(HASHLIFE_FULL_GRID_MAX_POPULATION),
        max_chunks: HASHLIFE_FULL_GRID_MAX_CHUNKS,
        max_bounds_span: Coord::MAX,
    }
}
