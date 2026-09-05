use super::arena::NodeColumns;
use super::cache_lifecycle::CacheStorageLifecycle;
use super::{CanonicalNodeRef, HashLifeEngine, NodeId};
use crate::RequiredExt;
use crate::cache_policy::{
    HASHLIFE_GC_MIN_NODES, HASHLIFE_GC_MIN_RECLAIM, HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER,
    hashlife_gc_reason,
};

impl HashLifeEngine {
    fn dynamic_total_hot_budget(&self) -> usize {
        let canonical_entries = self.canonical_cache_entries().max(1);
        let transient_entries = self.transient_cache_entries().max(canonical_entries);
        let observed_before = self
            .stats
            .gc
            .gc_canonical_cache_entries_before
            .max(canonical_entries);
        let min_budget = (HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER / 1024).max(64);
        let max_budget = (HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER / 4).max(min_budget);
        (observed_before / 4 + transient_entries / 16).clamp(min_budget, max_budget)
    }

    pub(super) fn rebalance_hot_canonical_budgets(&mut self) {
        let total_budget = self.dynamic_total_hot_budget();
        let packed_weight = self
            .stats
            .canonical_cache
            .canonical_packed_cache_hits
            .max(1);
        let oriented_weight = self
            .stats
            .canonical_cache
            .canonical_oriented_cache_hits
            .max(1);
        let direct_parent_weight = (self.stats.canonical_cache.direct_parent_cached_result_hits
            + self.stats.canonical_cache.direct_parent_winner_hits)
            .max(1);
        let total_weight = packed_weight + oriented_weight + direct_parent_weight;

        self.canonical_caches.hot_packed_budget =
            (total_budget * packed_weight / total_weight).max(1);
        self.canonical_caches.hot_oriented_budget =
            (total_budget * oriented_weight / total_weight).max(1);
        self.canonical_caches.hot_direct_parent_budget =
            (total_budget * direct_parent_weight / total_weight).max(1);

        let assigned = self.canonical_caches.hot_packed_budget
            + self.canonical_caches.hot_oriented_budget
            + self.canonical_caches.hot_direct_parent_budget;
        if assigned < total_budget {
            self.canonical_caches.hot_direct_parent_budget += total_budget - assigned;
        }

        self.trim_hot_canonical_caches_to_budget();
    }

    fn trim_hot_canonical_caches_to_budget(&mut self) {
        if self.canonical_caches.hot_packed.len() > self.canonical_caches.hot_packed_budget {
            self.canonical_caches.hot_packed.clear();
        }
        if self.canonical_caches.hot_oriented.len() > self.canonical_caches.hot_oriented_budget {
            self.canonical_caches.hot_oriented.clear();
        }
        if self.canonical_caches.hot_direct_parent.len()
            > self.canonical_caches.hot_direct_parent_budget
        {
            self.canonical_caches.hot_direct_parent.clear();
        }
    }

    pub(super) fn canonical_cache_entries(&self) -> usize {
        self.canonical_caches.packed.len()
            + self.canonical_caches.hot_packed.len()
            + self.canonical_caches.oriented.len()
            + self.canonical_caches.hot_oriented.len()
            + self.canonical_caches.direct_parent.len()
            + self.canonical_caches.hot_direct_parent.len()
            + self.result_caches.structural_fast_path.len()
            + self.result_caches.packed_structural_fast_path.len()
            + self.canonical_caches.symmetry_refs.len()
    }

    pub(super) fn transient_cache_entries(&self) -> usize {
        self.result_caches.jump.len()
            + self.result_caches.root.len()
            + self.result_caches.overlap.len()
            + self.result_caches.oriented.len()
            + self.result_caches.shells.len()
            + self.result_caches.bounds.len()
            + self.canonical_caches.packed.len()
            + self.canonical_caches.hot_packed.len()
            + self.canonical_caches.oriented.len()
            + self.canonical_caches.hot_oriented.len()
            + self.canonical_caches.direct_parent.len()
            + self.canonical_caches.hot_direct_parent.len()
            + self.result_caches.structural_fast_path.len()
            + self.result_caches.packed_structural_fast_path.len()
            + self.transform_state.canonical_cache.len()
            + self.transform_state.intern.len()
            + self.result_caches.materialized_packed.len()
            + self.embed_layout_cache.len()
            + self.future_state.result_len()
    }

    pub(super) fn transient_cache_pressure_entries(&self) -> usize {
        self.result_caches.jump.len()
            + self.result_caches.root.len()
            + self.result_caches.overlap.len()
            + self.result_caches.oriented.len()
            + self.result_caches.shells.len()
            + self.result_caches.bounds.len()
            + self.canonical_caches.packed.len()
            + self.canonical_caches.oriented.len()
            + self.canonical_caches.direct_parent.len()
            + self.result_caches.structural_fast_path.len()
            + self.result_caches.packed_structural_fast_path.len()
            + self.transform_state.canonical_cache.len()
            + self.transform_state.intern.len()
            + self.result_caches.materialized_packed.len()
            + self.embed_layout_cache.len()
            + self.future_state.result_len()
    }

    pub(super) fn initialize_runtime_state(&mut self) {
        let dead_shape = self.intern_canonical_shape(super::CanonicalStructKey::leaf(false));
        let live_shape = self.intern_canonical_shape(super::CanonicalStructKey::leaf(true));
        debug_assert_eq!(
            (dead_shape, live_shape),
            (CanonicalNodeRef::DEAD, CanonicalNodeRef::LIVE)
        );
        self.dead_leaf = self.intern_leaf(false);
        self.live_leaf = self.intern_leaf(true);
        self.empty_by_level.push(self.dead_leaf);
        self.reset_packed_transform_state();
    }

    pub(super) fn clear_transient_state(&mut self, preserve_hot_canonical: bool) {
        self.rebalance_hot_canonical_budgets();
        self.result_caches
            .apply_lifecycle(CacheStorageLifecycle::RetainCapacity);
        self.embed_layout_cache.clear();
        self.clear_packed_transform_state();
        self.canonical_caches.apply_lifecycle(
            CacheStorageLifecycle::RetainCapacity,
            preserve_hot_canonical,
        );
        self.clear_future_results();
    }

    pub(super) fn release_optional_cache_storage(&mut self) {
        self.result_caches
            .apply_lifecycle(CacheStorageLifecycle::ReleaseStorage);
        self.canonical_caches
            .apply_lifecycle(CacheStorageLifecycle::ReleaseStorage, false);
        self.embed_layout_cache = std::collections::HashMap::new();
        self.release_packed_transform_state();
        self.release_future_state();
    }

    pub(super) fn gc_reason(
        &self,
        previous_root: Option<NodeId>,
        current_root: Option<NodeId>,
    ) -> &'static str {
        hashlife_gc_reason(
            previous_root != current_root,
            self.node_count(),
            self.last_gc_nodes,
        )
    }

    pub(super) fn maybe_garbage_collect_with_budget(
        &mut self,
        reason: &'static str,
        hard_memory_bytes: u128,
    ) {
        if !self.at_gc_safepoint() {
            self.stats.gc.gc_reason = if self.scheduler_active {
                "scheduler_active"
            } else {
                "transaction_active"
            };
            self.stats.gc.gc_skips += 1;
            return;
        }
        if reason == "skip" {
            self.stats.gc.gc_reason = "skip";
            self.stats.gc.gc_skips += 1;
            if self.transient_cache_pressure_entries() >= HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER {
                self.stats.gc.gc_skipped_with_transient_growth += 1;
            }
            return;
        }

        self.stats.gc.gc_runs += 1;
        self.stats.gc.gc_transient_pressure_entries_before =
            self.transient_cache_pressure_entries();
        self.stats.gc.gc_canonical_cache_entries_before = self.canonical_cache_entries();
        let live_nodes = self.mark_live_nodes();
        self.stats.gc.nodes_before_mark = self.node_count();
        self.stats.gc.nodes_after_mark = live_nodes;
        let reclaimable = self.node_count().saturating_sub(live_nodes);
        let ordinary_compaction = self.node_count() >= HASHLIFE_GC_MIN_NODES
            && reclaimable >= HASHLIFE_GC_MIN_RECLAIM
            && reclaimable * 4 >= self.node_count();
        let pressure_compaction = reason == "budget_pressure"
            && (reclaimable != 0 || self.canonical_caches.shapes.len() > live_nodes);
        let should_compact = ordinary_compaction || pressure_compaction;

        if should_compact && self.can_repack_mandatory_indexes(live_nodes) {
            self.stats.gc.gc_reason = "compacted";
            self.stats.gc.nodes_before_compact = self.node_count();
            self.compact_marked_nodes();
            self.stats.gc.nodes_after_compact = self.node_count();
            self.last_gc_nodes = self.node_count();
        } else {
            self.stats.gc.gc_reason = if reason == "root_changed" {
                "root_changed_mark_only"
            } else {
                reason
            };
            self.stats.gc.nodes_before_compact = self.node_count();
            self.stats.gc.nodes_after_compact = self.node_count();
            self.last_gc_nodes = self.node_count();
            self.filter_caches_to_live_nodes();
            self.clear_packed_transform_state();
        }
        let release_threshold = hard_memory_bytes.saturating_sub(hard_memory_bytes / 5);
        if super::memory::wide_allocated_bytes(self.allocated_bytes()) >= release_threshold {
            self.release_optional_cache_storage();
        }
    }

    fn can_repack_mandatory_indexes(&self, live_nodes: usize) -> bool {
        self.at_gc_safepoint()
            && !self.allocation_failed()
            && self.intern.can_rebuild_without_allocation(live_nodes)
            && self
                .canonical_caches
                .shape_intern
                .can_rebuild_without_allocation(live_nodes)
            && self.canonical_caches.shapes.capacity() >= live_nodes
    }

    pub(in crate::hashlife) fn filter_caches_to_live_nodes(&mut self) {
        let columns = &self.node_columns;
        self.intern.retain_for_gc(|key, node| {
            node_is_live(node, columns) && packed_node_is_live(key, columns)
        });
        self.result_caches
            .jump
            .retain(|_, result| packed_node_is_live(result.packed, columns));
        self.result_caches
            .root
            .retain(|_, result| packed_node_is_live(result.packed, columns));
        self.result_caches
            .overlap
            .retain(|_, nodes| nodes.into_iter().all(|node| node_is_live(node, columns)));
        self.result_caches.oriented.retain(|key, value| {
            packed_node_is_live(key.packed, columns) && packed_node_is_live(value, columns)
        });
        self.result_caches
            .materialized_packed
            .retain(|key, node| packed_node_is_live(key, columns) && node_is_live(node, columns));
        self.result_caches
            .structural_fast_path
            .retain(|node, identity| {
                node_is_live(node, columns) && packed_node_is_live(identity.packed, columns)
            });
        self.result_caches
            .packed_structural_fast_path
            .retain(|key, identity| {
                packed_node_is_live(key, columns) && packed_node_is_live(identity.packed, columns)
            });
        self.result_caches
            .shells
            .retain(|key, shell| node_is_live(key.node, columns) && node_is_live(shell, columns));
        self.result_caches
            .bounds
            .retain(|node, _| node_is_live(node, columns));

        self.canonical_caches.node.retain(|node, identity| {
            node_is_live(node, columns) && packed_node_is_live(identity.packed, columns)
        });
        self.canonical_caches.packed.retain(|key, identity| {
            packed_node_is_live(key, columns) && packed_node_is_live(identity.packed, columns)
        });
        self.canonical_caches.hot_packed.retain(|key, identity| {
            packed_node_is_live(key, columns) && packed_node_is_live(identity.packed, columns)
        });
        self.canonical_caches.oriented.retain(|key, identity| {
            packed_node_is_live(key.packed, columns)
                && packed_node_is_live(identity.packed, columns)
        });
        self.canonical_caches.hot_oriented.retain(|key, identity| {
            packed_node_is_live(key.packed, columns)
                && packed_node_is_live(identity.packed, columns)
        });
        self.canonical_caches
            .direct_parent
            .retain(|_, identity| packed_node_is_live(identity.packed, columns));
        self.canonical_caches
            .hot_direct_parent
            .retain(|_, identity| packed_node_is_live(identity.packed, columns));
        // Orientation records never root shapes. Active filtering retains owned
        // capacity; the separate hard-pressure path releases their backing store.
        self.canonical_caches.symmetry_refs.reset();
        self.filter_future_results_to_live_nodes();
    }

    pub(super) fn mark_live_nodes(&mut self) -> usize {
        let node_count = self.node_count();
        self.node_columns.clear_marks();
        for root in self
            .empty_by_level
            .iter()
            .chain(self.retained_roots.iter())
            .copied()
            .chain([self.dead_leaf, self.live_leaf])
        {
            self.node_columns.mark(root);
        }

        for index in (0..node_count).rev() {
            if !self.node_columns.is_marked(index) {
                continue;
            }
            self.stats.gc.gc_mark_batches += 1;
            let node = NodeId::try_from(index)
                .or_invariant("HashLife arena exceeded u32 capacity during marking");
            if self.node_columns.level(node) == 0 {
                continue;
            }
            for child in self.node_columns.quadrants(node) {
                debug_assert!(
                    child.precedes(node),
                    "HashLife child must precede parent during GC"
                );
                self.node_columns.mark(child);
            }
        }

        self.node_columns.marked_count()
    }

    pub(super) fn record_retained_root(&mut self, root: NodeId) {
        if let Some(retained) = self.retained_roots.first_mut() {
            *retained = root;
        } else {
            // One slot is reserved with the engine; root publication cannot grow it.
            self.retained_roots.push(root);
        }
    }

    pub(super) fn compact_marked_nodes(&mut self) {
        assert!(
            self.at_gc_safepoint(),
            "HashLife arena repacking requires an explicit quiescent safepoint"
        );
        let old_len = self.node_columns.len();
        let live_nodes = self.node_columns.marked_count();
        assert!(
            self.can_repack_mandatory_indexes(live_nodes),
            "mandatory HashLife indexes must be prevalidated before arena repacking"
        );
        // Future results contain weak arena IDs. Discard the registry and all
        // dependent caches before any ID is rewritten.
        self.prepare_future_for_shape_rebuild();
        self.node_columns.clear_remap();
        let mut live = 0_usize;
        let mut old_idx = 0_usize;
        while old_idx < old_len {
            self.stats.gc.gc_remap_batches += 1;
            let batch_end = (old_idx + 8).min(old_len);
            for current_idx in old_idx..batch_end {
                if !self.node_columns.is_marked(current_idx) {
                    continue;
                }
                let remapped_node = NodeId::try_from(live)
                    .or_invariant("HashLife compacted arena exceeded u32 capacity");
                self.node_columns.set_remap(current_idx, remapped_node);
                self.node_columns.copy_node(current_idx, live);
                if self.node_columns.level(remapped_node) == 0 {
                    let population = self.node_columns.population_stat(remapped_node);
                    self.node_columns.set_fingerprint(
                        remapped_node,
                        crate::hashing::hash_leaf_population(population.lo),
                    );
                } else {
                    let remapped_children =
                        self.node_columns.quadrants(remapped_node).map(|child| {
                            self.node_columns
                                .remap(child)
                                .or_invariant("live child must have a compaction remap")
                        });
                    self.node_columns
                        .set_quadrants(remapped_node, remapped_children);
                    debug_assert!(
                        self.node_columns
                            .quadrants(remapped_node)
                            .into_iter()
                            .all(|child| child.precedes(remapped_node)),
                        "stable arena packing must preserve child-before-parent topology"
                    );
                    self.node_columns.set_fingerprint(
                        remapped_node,
                        crate::hashing::hash_u64_words_with_level(
                            self.node_columns.level(remapped_node),
                            self.node_columns.quadrants(remapped_node).map(u64::from),
                        ),
                    );
                }
                live += 1;
            }
            old_idx = batch_end;
        }
        self.node_columns.set_len_after_compaction(live);
        self.arena_epoch = self
            .arena_epoch
            .checked_add(1)
            .or_invariant("HashLife arena epoch overflow");
        self.rebuild_canonical_shapes();
        self.intern.clear();
        for node_id in 0..self.node_count() {
            let node_id = NodeId::try_from(node_id)
                .or_invariant("HashLife rebuilt arena exceeded u32 capacity");
            let key = if self.node_columns.level(node_id) == 0 {
                HashLifeEngine::packed_leaf_key(self.node_columns.population(node_id) == 1)
            } else {
                self.node_columns.packed_key(node_id)
            };
            if self
                .intern
                .try_insert_with_fingerprint(key, self.node_columns.fingerprint(node_id), node_id)
                .is_err()
            {
                crate::invariant_failure!(
                    "prevalidated mandatory intern capacity changed during arena repacking"
                );
            }
        }
        let columns = &self.node_columns;
        self.retained_roots.retain_mut(|root| {
            let Some(remapped) = columns.remap(*root) else {
                return false;
            };
            *root = remapped;
            true
        });
        self.empty_by_level.retain_mut(|empty| {
            let Some(remapped) = columns.remap(*empty) else {
                return false;
            };
            *empty = remapped;
            true
        });
        self.dead_leaf = self
            .node_columns
            .remap(self.dead_leaf)
            .or_invariant("dead leaf must survive compaction");
        self.live_leaf = self
            .node_columns
            .remap(self.live_leaf)
            .or_invariant("live leaf must survive compaction");

        self.node_columns.release_tail_segments();
        self.node_columns.clear_marks();
        self.node_columns.clear_remap();

        self.discard_epoch_bound_state();
    }

    /// No cache containing an arena-local node reference may survive an epoch change.
    /// Mandatory structural indexes are rebuilt above from compacted node semantics;
    /// all acceleration state is complete-or-discarded.
    fn discard_epoch_bound_state(&mut self) {
        self.result_caches
            .apply_lifecycle(CacheStorageLifecycle::RetainCapacity);
        self.canonical_caches
            .apply_lifecycle(CacheStorageLifecycle::RetainCapacity, false);
        self.embed_layout_cache.clear();
        self.clear_packed_transform_state();
    }
}

fn node_is_live(node: NodeId, columns: &NodeColumns) -> bool {
    columns.is_marked(node.index())
}

pub(super) fn packed_node_is_live_for_cache(
    node: super::PackedNodeKey,
    columns: &NodeColumns,
) -> bool {
    node.level == 0
        || node
            .children
            .into_iter()
            .all(|child| node_is_live(child, columns))
}

fn packed_node_is_live(node: super::PackedNodeKey, columns: &NodeColumns) -> bool {
    packed_node_is_live_for_cache(node, columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashlife::PopulationStat;

    #[test]
    fn pressure_repack_is_allocation_free_and_preserves_structural_uniqueness() {
        let mut engine = HashLifeEngine::default();
        let d = engine.dead_leaf;
        let l = engine.live_leaf;
        engine.join(d, l, d, l); // Unreachable structure below normal GC thresholds.
        let retained = engine.join(l, l, d, l);
        engine.record_retained_root(retained);
        let children = [CanonicalNodeRef::LIVE; 4];
        let shape = engine.canonical_parent_key(1, children);
        engine.intern_canonical_shape(shape); // Shape-only historical identity.
        engine.release_packed_transform_state();
        let bytes = engine.allocated_bytes();
        let epoch = engine.arena_epoch;
        engine.allocation_hard_limit = bytes as u128;

        engine.maybe_garbage_collect_with_budget("budget_pressure", bytes as u128);

        assert_eq!(
            engine.arena_epoch,
            epoch + 1,
            "small dead DAG was stranded by size thresholds"
        );
        assert!(
            engine.allocated_bytes() <= bytes,
            "repacking allocated new storage"
        );
        assert!(
            engine.transform_state.nodes.is_empty(),
            "GC eagerly recreated transform leaves"
        );
        assert_eq!(engine.take_allocation_failure(), None);
        assert_eq!(engine.node_count(), 3);
        assert_eq!(
            engine.canonical_caches.shapes.len(),
            3,
            "historical shapes survived epoch rebuild"
        );
        let root = engine.retained_roots[0];
        let children = engine.node_columns.quadrants(root);
        let identity = engine.node_columns.identity_ref(root);
        for _ in 0..8 {
            assert_eq!(
                engine.join(children[0], children[1], children[2], children[3]),
                root,
                "exact structure received a duplicate node after index rebuild"
            );
            let key = engine.canonical_parent_key(
                1,
                children.map(|child| engine.node_columns.identity_ref(child)),
            );
            assert_eq!(
                engine.intern_canonical_shape(key),
                identity,
                "exact structure received a duplicate canonical identity after rebuild"
            );
        }
        assert_eq!(engine.node_count(), 3);
        assert_eq!(engine.canonical_caches.shapes.len(), 3);
    }

    #[test]
    fn root_publication_and_transient_cleanup_do_not_allocate() {
        let mut engine = HashLifeEngine::default();
        let capacity = engine.retained_roots.capacity();
        for root in [engine.live_leaf, engine.dead_leaf, engine.live_leaf] {
            engine.record_retained_root(root);
            assert_eq!(engine.retained_roots.as_slice(), &[root]);
            assert_eq!(
                engine.retained_roots.capacity(),
                capacity,
                "root replacement grew storage"
            );
        }
        engine.result_caches.bounds.insert(
            engine.live_leaf,
            super::super::RelativeBounds {
                min_x: 0,
                min_y: 0,
                max_x: 0,
                max_y: 0,
            },
        );
        engine.release_packed_transform_state();
        let bytes = engine.allocated_bytes();
        engine.clear_transient_state(false);
        assert_eq!(engine.result_caches.bounds.len(), 0);
        assert_eq!(
            engine.allocated_bytes(),
            bytes,
            "cleanup seeded transform storage"
        );
        assert!(engine.transform_state.nodes.is_empty());
    }

    #[test]
    fn shell_and_bounds_entries_contribute_to_gc_pressure() {
        let mut engine = HashLifeEngine::default();
        let baseline = engine.transient_cache_pressure_entries();
        let shell_key = super::super::ShellKey {
            node: engine.dead_leaf,
            target_level: 1,
        };
        engine
            .result_caches
            .shells
            .insert(shell_key, engine.dead_leaf);
        engine.result_caches.bounds.insert(
            engine.live_leaf,
            super::super::RelativeBounds {
                min_x: 0,
                min_y: 0,
                max_x: 0,
                max_y: 0,
            },
        );

        assert_eq!(
            engine.transient_cache_pressure_entries(),
            baseline + 2,
            "shell and bounds allocations must both increase GC pressure accounting"
        );
    }

    #[test]
    fn garbage_collection_skips_non_safepoint_scheduler_state() {
        let mut engine = HashLifeEngine::default();
        let node_count = engine.node_count();
        engine.scheduler_active = true;

        engine.maybe_garbage_collect_with_budget("active", u128::MAX);

        assert_eq!(engine.node_count(), node_count);
        assert_eq!(engine.stats.gc.gc_runs, 0);
        assert_eq!(engine.stats.gc.gc_skips, 1);
        assert_eq!(engine.stats.gc.gc_reason, "scheduler_active");
    }

    #[test]
    fn garbage_collection_skips_active_allocation_transaction() {
        let mut engine = HashLifeEngine::default();
        let node_count = engine.node_count();
        engine.begin_allocation_transaction(u128::MAX);

        engine.maybe_garbage_collect_with_budget("active", u128::MAX);

        assert_eq!(engine.node_count(), node_count);
        assert_eq!(engine.stats.gc.gc_runs, 0);
        assert_eq!(engine.stats.gc.gc_skips, 1);
        assert_eq!(engine.stats.gc.gc_reason, "transaction_active");
        assert!(engine.take_allocation_failure().is_none());
    }

    #[test]
    fn repack_does_not_start_without_mandatory_index_capacity() {
        let mut engine = HashLifeEngine::default();
        for _ in 0..super::super::arena::NODE_SEGMENT_LEN {
            engine
                .node_columns
                .try_reserve_nodes(1)
                .or_invariant("test arena segment allocation failed");
            engine.push_node(
                0,
                PopulationStat::exact(0),
                NodeId::ZERO,
                NodeId::ZERO,
                NodeId::ZERO,
                NodeId::ZERO,
            );
        }
        engine.record_retained_root(engine.dead_leaf);
        engine.canonical_caches.shape_intern.release_storage();
        let nodes_before = engine.node_count();
        let epoch_before = engine.arena_epoch;

        engine.maybe_garbage_collect_with_budget("root_changed", u128::MAX);

        assert_eq!(engine.node_count(), nodes_before);
        assert_eq!(engine.arena_epoch, epoch_before);
        assert_eq!(engine.stats.gc.nodes_after_compact, nodes_before);
    }

    #[test]
    fn quiescent_repack_reuses_arena_capacity_without_second_arena() {
        let mut engine = HashLifeEngine::default();
        for _ in 0..super::super::arena::NODE_SEGMENT_LEN * 4 {
            engine
                .node_columns
                .try_reserve_nodes(1)
                .or_invariant("test arena segment allocation failed");
            engine.push_node(
                0,
                PopulationStat::exact(0),
                NodeId::ZERO,
                NodeId::ZERO,
                NodeId::ZERO,
                NodeId::ZERO,
            );
        }
        let capacity_before = engine.node_columns.capacity();
        let segments_before = engine.node_columns.segment_count();
        let bytes_before = engine.allocated_bytes();
        let epoch_before = engine.arena_epoch;

        let live = engine.mark_live_nodes();
        engine.compact_marked_nodes();

        assert_eq!(
            live, 2,
            "only canonical dead/live leaves should remain reachable"
        );
        assert_eq!(engine.node_count(), 2);
        assert!(segments_before >= 4, "fixture did not span enough segments");
        assert_eq!(
            engine.node_columns.segment_count(),
            1,
            "dead tail segments were not physically released"
        );
        assert!(
            engine.node_columns.capacity() <= capacity_before,
            "quiescent compaction must not grow segmented arena capacity"
        );
        assert!(
            engine.allocated_bytes() <= bytes_before,
            "in-place repacking unexpectedly grew retained storage: before={bytes_before} after={}",
            engine.allocated_bytes()
        );
        assert_eq!(engine.arena_epoch, epoch_before + 1);
        let resumed = engine.join(
            engine.dead_leaf,
            engine.live_leaf,
            engine.live_leaf,
            engine.dead_leaf,
        );
        assert!(
            engine.live_leaf.precedes(resumed),
            "mandatory interning did not resume from reclaimed segment capacity"
        );
        for parent in 0..engine.node_count() {
            let parent = NodeId::try_from(parent).or_invariant("test node id");
            if engine.node_columns.level(parent) != 0 {
                assert!(
                    engine
                        .node_columns
                        .quadrants(parent)
                        .into_iter()
                        .all(|child| child.precedes(parent)),
                    "repacked parent {parent:?} violates child-before-parent topology"
                );
            }
        }
    }

    #[test]
    fn active_mark_and_filter_preserve_all_owned_capacities() {
        let mut engine = HashLifeEngine::default();
        engine.record_retained_root(engine.dead_leaf);
        let bytes_before = engine.allocated_bytes();

        engine.mark_live_nodes();
        engine.filter_caches_to_live_nodes();
        engine.reset_packed_transform_state();

        assert_eq!(
            engine.allocated_bytes(),
            bytes_before,
            "active mark/filter changed owned capacity"
        );
    }

    #[test]
    fn mandatory_node_growth_is_rejected_before_publication_at_hard_limit() {
        let mut engine = HashLifeEngine::default();
        while engine.node_count() < engine.node_columns.capacity() {
            let id = NodeId::try_from(engine.node_count()).or_invariant("test node id");
            engine.push_node(0, PopulationStat::exact(0), id, id, id, id);
        }
        let nodes_before = engine.node_count();
        engine.begin_allocation_transaction(super::super::memory::wide_allocated_bytes(
            engine.allocated_bytes(),
        ));

        let result = engine.join(
            engine.dead_leaf,
            engine.live_leaf,
            engine.live_leaf,
            engine.dead_leaf,
        );

        assert_eq!(
            result, engine.dead_leaf,
            "failed join must remain unpublished"
        );
        assert_eq!(engine.node_count(), nodes_before);
        assert!(
            engine.take_allocation_failure().is_some(),
            "mandatory growth rejection was not propagated to the transaction"
        );
    }

    #[test]
    fn scheduler_workspace_is_rejected_before_allocation_at_hard_limit() {
        let mut engine = HashLifeEngine::default();
        let quadrant = engine.join(
            engine.live_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
        );
        let root = engine.join(quadrant, quadrant, quadrant, quadrant);
        let retained = super::super::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);

        let result = engine.advance_pow2(root, 0);

        assert_eq!(result, engine.dead_leaf);
        assert_eq!(
            super::super::memory::wide_allocated_bytes(engine.allocated_bytes()),
            retained,
            "scheduler failure allocated outside the preflighted transaction"
        );
        assert!(
            engine.take_allocation_failure().is_some(),
            "scheduler workspace growth bypassed the active hard limit"
        );
        assert!(!engine.scheduler_active);
    }

    #[test]
    fn hard_pressure_releases_transform_backing_storage_and_reinitializes_lazily() {
        let mut engine = HashLifeEngine::default();
        engine.transform_state.nodes.reserve(16_384);
        engine.transform_state.materialized.reserve(16_384);
        engine.transform_state.packed_roots.reserve(16_384);
        let bytes_before = engine.allocated_bytes();

        engine.release_optional_cache_storage();

        let bytes_after = engine.allocated_bytes();
        assert!(
            bytes_after < bytes_before,
            "hard-pressure release retained transform capacity: before={bytes_before} after={bytes_after}"
        );
        let leaf = super::super::PackedNodeKey::new(0, [super::super::NodeId::ZERO; 4]);
        assert_eq!(
            engine.transform_packed_node_key(leaf, crate::symmetry::D4Symmetry::Identity),
            super::super::PackedTransformId::ZERO,
            "released transform state did not reinitialize its canonical leaves"
        );
    }
}
