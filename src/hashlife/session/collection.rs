use super::*;

impl HashLifeSession {
    /// Reclaim only at a committed boundary, before a transaction reserves new storage.
    pub(super) fn collect_before_allocation(&mut self) {
        let epoch = self.engine.arena_epoch;
        self.current_root = self.engine.maybe_collect_active_run(
            self.current_root,
            self.limits.soft_memory_bytes,
            self.limits.hard_memory_bytes,
        );
        if self.engine.arena_epoch != epoch {
            self.previous_root = self.current_root;
            self.checkpoint_epoch = self.checkpoint_epoch.wrapping_add(1);
            self.clear_cached_samples();
        }
        self.memory_budget.sync_retained(self.allocated_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_session_parser_uses_remaining_budget_not_whole_limit() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0), (1, 0), (0, 1), (1, 1)]))
            .or_invariant("authoritative fixture loads");
        let before = session
            .export_snapshot_string()
            .or_invariant("fixture exports");
        session.engine.release_optional_cache_storage();
        session.engine.last_gc_nodes = session.engine.node_count();
        let limits = session.limits();
        session.set_limits(HashLifeLimits {
            hard_memory_bytes: session.allocated_bytes() + 1_024,
            ..limits
        });
        let error = session.load_snapshot_string(before.as_deref().or_invariant("loaded snapshot"));
        assert!(
            matches!(
                error,
                Err(super::super::super::HashLifeConversionError::AllocationFailed { .. })
            ),
            "parser must reject inadequate remaining scratch budget: {error:?}"
        );
        session.set_limits(limits);
        assert_eq!(
            session
                .export_snapshot_string()
                .or_invariant("retained state exports"),
            before,
            "failed import changed authoritative semantics"
        );
    }

    fn add_unreachable_patterns(engine: &mut HashLifeEngine) {
        for bits in 1..8_192_u32 {
            let children = std::array::from_fn::<_, 4, _>(|quadrant| {
                let cells = std::array::from_fn::<_, 4, _>(|cell| {
                    if bits & (1 << (quadrant * 4 + cell)) == 0 {
                        engine.dead_leaf
                    } else {
                        engine.live_leaf
                    }
                });
                engine.join(cells[0], cells[1], cells[2], cells[3])
            });
            engine.join(children[0], children[1], children[2], children[3]);
        }
    }

    #[test]
    fn replacement_publishes_the_new_gc_root_for_grid_and_snapshot_loads() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0)]))
            .or_invariant("initial state loads");
        let first = session.current_root;
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0), (1, 0), (0, 1)]))
            .or_invariant("replacement loads");
        assert_ne!(
            session.current_root, first,
            "fixture must change structural identity"
        );
        assert_eq!(
            session.engine.retained_roots.last().copied(),
            session.current_root,
            "grid replacement left the old GC root published"
        );
        let snapshot =
            super::super::super::snapshot::serialize_grid(&BitGrid::from_cells(&[(8, 8)]))
                .or_invariant("replacement snapshot serializes");
        session
            .load_snapshot_string(&snapshot)
            .or_invariant("snapshot replacement loads");
        assert_eq!(
            session.engine.retained_roots.last().copied(),
            session.current_root,
            "snapshot replacement left the old GC root published"
        );
        let retained = session.engine.mark_live_nodes();
        assert!(retained >= 2);
        assert!(
            session
                .engine
                .node_columns
                .is_marked(session.current_root.or_invariant("loaded root").index())
        );
        assert_eq!(session.sample_cell(8, 8), Some(true));
    }

    #[test]
    fn cache_only_pressure_releases_storage_without_repeated_marking() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0)]))
            .or_invariant("fixture loads");
        session.engine.last_gc_nodes = session.engine.node_count();
        session
            .engine
            .result_caches
            .bounds
            .try_reserve(16_384)
            .or_invariant("cache fixture reserves");
        let bytes = session.allocated_bytes();
        session.set_limits(HashLifeLimits {
            soft_memory_bytes: bytes / 2,
            hard_memory_bytes: bytes + 1,
        });
        let marks = session.engine.stats.gc.gc_mark_batches;
        let epoch = session.engine.arena_epoch;
        session.collect_before_allocation();
        assert!(
            session.allocated_bytes() < bytes,
            "cache-only pressure retained backing storage"
        );
        session.collect_before_allocation();
        assert_eq!(
            session.engine.stats.gc.gc_mark_batches, marks,
            "unchanged DAG was traversed for cache pressure"
        );
        assert_eq!(session.engine.arena_epoch, epoch);
        assert_eq!(session.sample_cell(0, 0), Some(true));
    }

    #[test]
    fn advancement_reclaims_dead_segments_before_rejecting_lowered_hard_limit() {
        let block = BitGrid::from_cells(&[(0, 0), (1, 0), (0, 1), (1, 1)]);
        let mut session = HashLifeSession::new();
        session.try_load_grid(&block).or_invariant("block loads");
        add_unreachable_patterns(&mut session.engine);
        let bytes = session.allocated_bytes();
        let epoch = session.engine.arena_epoch;
        session.set_limits(HashLifeLimits {
            soft_memory_bytes: bytes / 2,
            hard_memory_bytes: bytes - 1,
        });

        let result = session
            .advance_root(1)
            .or_invariant("reclaimable arena must not strand advancement");
        assert_eq!(result.completed_generations, 1);
        assert!(
            session.engine.arena_epoch > epoch,
            "pressure did not repack dead nodes"
        );
        assert!(session.allocated_bytes() <= session.limits().hard_memory_bytes);
        assert_eq!(session.population_count(), Some(PopulationCount::Exact(4)));
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(
                session.sample_cell(x, y),
                Some(true),
                "repacking changed block at ({x},{y})"
            );
        }
    }

    #[test]
    fn unload_retires_authoritative_root_and_pressure_reclaims_it_before_reload() {
        let mut session = HashLifeSession::new();
        let grid = BitGrid::from_cells(&[(10, 10), (11, 10), (12, 10)]);
        session.try_load_grid(&grid).or_invariant("fixture loads");
        session.unload();
        assert!(
            session.engine.retained_roots.is_empty(),
            "unload kept the old universe pinned"
        );
        add_unreachable_patterns(&mut session.engine);
        let bytes = session.allocated_bytes();
        session.set_limits(HashLifeLimits {
            soft_memory_bytes: bytes / 2,
            hard_memory_bytes: bytes - 1,
        });
        session
            .try_load_grid(&BitGrid::from_cells(&[(7, 8)]))
            .or_invariant("replacement must reuse reclaimed storage");
        assert_eq!(session.sample_cell(7, 8), Some(true));
        assert_eq!(session.population_count(), Some(PopulationCount::Exact(1)));
        assert!(
            session.engine.node_count() < 100,
            "unloaded or unreachable nodes survived pressure GC"
        );
    }
}
