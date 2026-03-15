use super::super::memory::{AllocationBoundaryError, HashLifeAllocationFailure, checked_capacity};
use super::super::{EngineAllocationFailure, EngineIdCapacity};
use super::*;

impl HashLifeSession {
    /// Reads one universe cell directly from the current DAG without materialization.
    pub fn sample_cell(&self, x: Coord, y: Coord) -> Option<bool> {
        let root = self.current_root?;
        Some(
            self.engine
                .node_cell_alive(root, self.current_origin_x, self.current_origin_y, x, y),
        )
    }

    pub(crate) fn with_id_capacity_headroom(
        node_headroom: usize,
        canonical_headroom: usize,
    ) -> Self {
        let mut session = Self::default();
        session.engine.id_capacity = EngineIdCapacity {
            node_count: session
                .engine
                .node_count()
                .saturating_add(node_headroom)
                .min(EngineIdCapacity::FULL.node_count),
            canonical_count: session
                .engine
                .canonical_caches
                .shape_intern
                .len()
                .saturating_add(canonical_headroom)
                .min(EngineIdCapacity::FULL.canonical_count),
        };
        session
    }

    pub fn configure_allocation_failure(&mut self, failure: Option<HashLifeAllocationFailure>) {
        self.allocation_gate.configure(failure);
    }

    pub(crate) fn check_allocation_gate(
        &mut self,
        class: HashLifeAllocationClass,
        requested_bytes: u128,
    ) -> Result<(), HashLifeConversionError> {
        self.allocation_gate
            .check(class, requested_bytes)
            .map_err(|error| {
                conversion_budget_error(
                    error,
                    wide_allocated_bytes(self.engine.allocated_bytes()),
                    requested_bytes,
                    self.limits,
                )
            })
    }

    pub fn try_load_grid(&mut self, grid: &BitGrid) -> Result<(), HashLifeConversionError> {
        self.try_load_grid_at_generation(grid, 0)
    }

    pub(crate) fn try_load_grid_at_generation(
        &mut self,
        grid: &BitGrid,
        generation: u64,
    ) -> Result<(), HashLifeConversionError> {
        let published = self.published_state();
        let retained_before = wide_allocated_bytes(self.engine.allocated_bytes());
        self.memory_budget.sync_retained(retained_before);
        let estimated_nodes = u128::try_from(grid.chunk_count())
            .unwrap_or(u128::MAX)
            .saturating_mul(64)
            .saturating_add(64);
        let candidate_bytes = estimated_nodes.saturating_mul(64);
        self.check_allocation_gate(HashLifeAllocationClass::Embed, candidate_bytes)?;
        checked_capacity::<NodeId>(estimated_nodes).map_err(|error| {
            HashLifeConversionError::AllocationFailed {
                requested_bytes: allocation_error_bytes(error, candidate_bytes),
            }
        })?;
        self.memory_budget
            .reserve_candidate(candidate_bytes)
            .map_err(|error| {
                conversion_budget_error(error, retained_before, candidate_bytes, self.limits)
            })?;
        self.ensure_active_run();
        self.engine
            .begin_allocation_transaction(self.limits.hard_memory_bytes);
        let (root, origin_x, origin_y) = match self.engine.try_embed_grid_state(grid) {
            Ok(embedded) => embedded,
            Err(error) => {
                self.memory_budget.release_candidate(candidate_bytes);
                self.memory_budget
                    .sync_retained(wide_allocated_bytes(self.engine.allocated_bytes()));
                self.restore_published_state(published);
                if let Some(failure) = self.engine.take_allocation_failure() {
                    self.engine.clear_transient_state(false);
                    return Err(conversion_engine_failure(failure));
                }
                return Err(HashLifeConversionError::CoordinateRangeExceeded { axis: error.axis });
            }
        };
        if let Some(failure) = self.engine.take_allocation_failure() {
            self.memory_budget.release_candidate(candidate_bytes);
            self.memory_budget
                .sync_retained(wide_allocated_bytes(self.engine.allocated_bytes()));
            self.restore_published_state(published);
            self.engine.clear_transient_state(false);
            return Err(conversion_engine_failure(failure));
        }
        let retained_after = wide_allocated_bytes(self.engine.allocated_bytes());
        self.memory_budget
            .commit_replacement(retained_before, retained_after, candidate_bytes);
        if retained_after > self.limits.hard_memory_bytes {
            self.restore_published_state(published);
            return Err(HashLifeConversionError::MemoryBudgetExceeded {
                retained_bytes: retained_after,
                candidate_bytes,
                limit_bytes: self.limits.hard_memory_bytes,
            });
        }
        self.current_root = Some(root);
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = generation;
        self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
        self.checkpoint_level = None;
        self.root_is_centered = false;
        self.sampled_bounds = Some(grid.bounds());
        self.sampled_checkpoint = None;
        Ok(())
    }

    pub fn load_snapshot_string(&mut self, snapshot: &str) -> Result<(), HashLifeConversionError> {
        let published = self.published_state();
        let retained_before = wide_allocated_bytes(self.engine.allocated_bytes());
        self.memory_budget.sync_retained(retained_before);
        let candidate_bytes = u128::try_from(snapshot.len())
            .unwrap_or(u128::MAX)
            .saturating_mul(4);
        self.check_allocation_gate(HashLifeAllocationClass::SnapshotImport, candidate_bytes)?;
        checked_capacity::<u8>(candidate_bytes).map_err(|error| {
            HashLifeConversionError::AllocationFailed {
                requested_bytes: allocation_error_bytes(error, candidate_bytes),
            }
        })?;
        self.memory_budget
            .reserve_candidate(candidate_bytes)
            .map_err(|error| {
                conversion_budget_error(error, retained_before, candidate_bytes, self.limits)
            })?;
        self.ensure_active_run();
        self.engine
            .begin_allocation_transaction(self.limits.hard_memory_bytes);
        let imported = self.engine.import_snapshot_string(snapshot);
        let (root, origin_x, origin_y, generation) = match imported {
            Ok(imported) => imported,
            Err(error) => {
                self.memory_budget.release_candidate(candidate_bytes);
                self.memory_budget
                    .sync_retained(wide_allocated_bytes(self.engine.allocated_bytes()));
                self.restore_published_state(published);
                if let Some(failure) = self.engine.take_allocation_failure() {
                    self.engine.clear_transient_state(false);
                    return Err(conversion_engine_failure(failure));
                }
                if let Some(requested_bytes) = error.allocation_bytes() {
                    return Err(HashLifeConversionError::AllocationFailed { requested_bytes });
                }
                return Err(HashLifeConversionError::Snapshot(error));
            }
        };
        if let Some(failure) = self.engine.take_allocation_failure() {
            self.memory_budget.release_candidate(candidate_bytes);
            self.memory_budget
                .sync_retained(wide_allocated_bytes(self.engine.allocated_bytes()));
            self.restore_published_state(published);
            self.engine.clear_transient_state(false);
            return Err(conversion_engine_failure(failure));
        }
        let retained_after = wide_allocated_bytes(self.engine.allocated_bytes());
        self.memory_budget
            .commit_replacement(retained_before, retained_after, candidate_bytes);
        if retained_after > self.limits.hard_memory_bytes {
            self.restore_published_state(published);
            return Err(HashLifeConversionError::MemoryBudgetExceeded {
                retained_bytes: retained_after,
                candidate_bytes,
                limit_bytes: self.limits.hard_memory_bytes,
            });
        }
        self.current_root = Some(root);
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = generation;
        self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
        self.checkpoint_level = None;
        self.root_is_centered = false;
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
        Ok(())
    }

    pub(crate) fn try_extract_grid_for_conversion(
        &mut self,
        policy: GridExtractionPolicy,
    ) -> Result<BitGrid, HashLifeMaterializationError> {
        let retained = wide_allocated_bytes(self.engine.allocated_bytes());
        self.memory_budget.sync_retained(retained);
        let candidate_bytes = self
            .population_count()
            .map(PopulationCount::lower_bound)
            .unwrap_or(0)
            .saturating_mul(32)
            .max(1_024);
        self.check_allocation_gate(HashLifeAllocationClass::Materialize, candidate_bytes)
            .map_err(HashLifeMaterializationError::Conversion)?;
        checked_capacity::<u8>(candidate_bytes).map_err(|error| {
            HashLifeMaterializationError::Conversion(HashLifeConversionError::AllocationFailed {
                requested_bytes: allocation_error_bytes(error, candidate_bytes),
            })
        })?;
        self.memory_budget
            .reserve_candidate(candidate_bytes)
            .map_err(|error| {
                HashLifeMaterializationError::Conversion(conversion_budget_error(
                    error,
                    retained,
                    candidate_bytes,
                    self.limits,
                ))
            })?;
        let candidate = match self.extract_grid(policy) {
            Ok(candidate) => candidate,
            Err(error) => {
                self.memory_budget.release_candidate(candidate_bytes);
                return Err(HashLifeMaterializationError::Extraction(error));
            }
        };
        self.memory_budget.release_candidate(candidate_bytes);
        let actual_candidate = wide_allocated_bytes(candidate.allocated_bytes());
        if retained.saturating_add(actual_candidate) > self.limits.hard_memory_bytes {
            return Err(HashLifeMaterializationError::Conversion(
                HashLifeConversionError::MemoryBudgetExceeded {
                    retained_bytes: retained,
                    candidate_bytes: actual_candidate,
                    limit_bytes: self.limits.hard_memory_bytes,
                },
            ));
        }
        Ok(candidate)
    }
}

fn allocation_error_bytes(error: AllocationBoundaryError, fallback: u128) -> u128 {
    match error {
        AllocationBoundaryError::BudgetExceeded { requested, .. } => requested,
        AllocationBoundaryError::ByteCountOverflow | AllocationBoundaryError::CapacityOverflow => {
            fallback
        }
    }
}

fn conversion_budget_error(
    error: AllocationBoundaryError,
    retained_bytes: u128,
    candidate_bytes: u128,
    limits: HashLifeLimits,
) -> HashLifeConversionError {
    match error {
        AllocationBoundaryError::BudgetExceeded { .. } => {
            HashLifeConversionError::MemoryBudgetExceeded {
                retained_bytes,
                candidate_bytes,
                limit_bytes: limits.hard_memory_bytes,
            }
        }
        AllocationBoundaryError::ByteCountOverflow | AllocationBoundaryError::CapacityOverflow => {
            HashLifeConversionError::AllocationFailed {
                requested_bytes: candidate_bytes,
            }
        }
    }
}

fn conversion_engine_failure(failure: EngineAllocationFailure) -> HashLifeConversionError {
    match failure {
        EngineAllocationFailure::Allocation { requested_bytes } => {
            HashLifeConversionError::AllocationFailed { requested_bytes }
        }
        EngineAllocationFailure::NodeIdExhausted => HashLifeConversionError::NodeIdExhausted,
        EngineAllocationFailure::CanonicalReferenceExhausted => {
            HashLifeConversionError::CanonicalReferenceExhausted
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredErrorExt;

    #[test]
    fn unrepresentable_full_coordinate_span_fails_without_replacing_authoritative_state() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0)]))
            .or_invariant("baseline grid should load");
        let before = session
            .export_snapshot_string()
            .or_invariant("baseline snapshot should export");

        let extreme = BitGrid::from_cells(&[(Coord::MIN, 0), (Coord::MAX, 0)]);
        let error = session
            .try_load_grid(&extreme)
            .error_or_invariant("full coordinate span should fail typed conversion");
        assert!(
            matches!(
                error,
                HashLifeConversionError::CoordinateRangeExceeded { .. }
            ),
            "full coordinate span returned wrong error: {error:?}"
        );
        assert_eq!(session.generation(), 0, "failure changed generation");
        assert_eq!(
            session
                .export_snapshot_string()
                .or_invariant("authoritative state should remain exportable"),
            before,
            "failure replaced authoritative root"
        );
    }

    #[test]
    fn recursive_node_growth_failure_returns_typed_zero_progress_and_preserves_root() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]))
            .or_invariant("blinker should load");
        let before = session
            .export_snapshot_string()
            .or_invariant("baseline snapshot should export");
        session.configure_allocation_failure(Some(HashLifeAllocationFailure {
            class: HashLifeAllocationClass::ArenaGrowth,
            ordinal: 1,
        }));

        let error = session
            .advance_root(1)
            .error_or_invariant("full mandatory arena should reject recursive growth");

        assert!(
            matches!(
                error,
                HashLifeAdvanceError::AllocationFailed {
                    completed_generations: 0,
                    reached_generation: 0,
                    ..
                }
            ),
            "recursive allocation failure returned wrong progress: {error:?}"
        );
        assert_eq!(session.generation(), 0);
        assert_eq!(
            session
                .export_snapshot_string()
                .or_invariant("failed session should remain exportable"),
            before,
            "failed recursive growth replaced the authoritative root"
        );
    }

    #[test]
    fn grid_conversion_reports_node_id_exhaustion_without_using_the_remap_sentinel() {
        let mut session = HashLifeSession::with_id_capacity_headroom(0, usize::MAX);
        let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]);

        let error = session
            .try_load_grid(&grid)
            .error_or_invariant("bounded node-ID space should reject new nodes");

        assert_eq!(error, HashLifeConversionError::NodeIdExhausted);
        assert!(!session.is_loaded(), "failed conversion published a root");
        assert!(
            session.engine.node_columns.remap(NodeId::MAX).is_none(),
            "reserved remap sentinel became a valid node ID"
        );
    }

    #[test]
    fn grid_conversion_reports_canonical_reference_exhaustion() {
        let mut session = HashLifeSession::with_id_capacity_headroom(usize::MAX, 0);
        let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]);

        let error = session
            .try_load_grid(&grid)
            .error_or_invariant("bounded canonical-reference space should reject new shapes");

        assert_eq!(error, HashLifeConversionError::CanonicalReferenceExhausted);
        assert!(!session.is_loaded(), "failed conversion published a root");
    }

    #[test]
    fn advancement_reports_node_id_exhaustion_with_zero_uncommitted_progress() {
        let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]);
        let (session, error) = (0..512)
            .find_map(|node_headroom| {
                let mut session =
                    HashLifeSession::with_id_capacity_headroom(node_headroom, usize::MAX);
                session.try_load_grid(&grid).ok()?;
                let error = session.advance_root(1).err()?;
                matches!(error, HashLifeAdvanceError::NodeIdExhausted { .. })
                    .then_some((session, error))
            })
            .or_invariant("bounded ID probe should reach advancement exhaustion");

        assert_eq!(
            error,
            HashLifeAdvanceError::NodeIdExhausted {
                starting_generation: 0,
                requested_delta: 1,
                completed_generations: 0,
                reached_generation: 0,
            }
        );
        assert_eq!(session.generation(), 0, "failed segment committed progress");
    }
}
