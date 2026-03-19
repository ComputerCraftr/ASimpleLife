use super::{
    CanonicalNodeIdentity, DirectCanonicalParentKey, FlatTable, HashLifeEngine, NodeColumns,
    NodeId, PackedNodeKey, PackedSymmetryKey,
};
use crate::cache_policy::{
    HASHLIFE_GC_MIN_NODES, HASHLIFE_GC_MIN_RECLAIM, HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER,
    hashlife_gc_reason,
};

const HASHLIFE_MAX_RETAINED_ROOTS: usize = 1;

impl HashLifeEngine {
    fn remap_packed_node_key(remap: &[NodeId], packed: PackedNodeKey) -> Option<PackedNodeKey> {
        if packed.level == 0 {
            Some(packed)
        } else {
            let mut remapped_children = [0; 4];
            for (index, child) in packed.children.into_iter().enumerate() {
                let child_index = child as usize;
                if child_index >= remap.len() {
                    return None;
                }
                let remapped = remap[child_index];
                if remapped == NodeId::MAX {
                    return None;
                }
                remapped_children[index] = remapped;
            }
            Some(PackedNodeKey::new(packed.level, remapped_children))
        }
    }

    fn remap_canonical_node_identity(
        remap: &[NodeId],
        canonical: CanonicalNodeIdentity,
    ) -> Option<CanonicalNodeIdentity> {
        Some(CanonicalNodeIdentity {
            packed: Self::remap_packed_node_key(remap, canonical.packed)?,
            ..canonical
        })
    }

    fn dynamic_total_hot_budget(&self) -> usize {
        let canonical_entries = self.canonical_cache_entries().max(1);
        let transient_entries = self.transient_cache_entries().max(canonical_entries);
        let observed_before = self
            .stats
            .gc_canonical_cache_entries_before
            .max(canonical_entries);
        let min_budget = (HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER / 1024).max(64);
        let max_budget = (HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER / 4).max(min_budget);
        (observed_before / 4 + transient_entries / 16).clamp(min_budget, max_budget)
    }

    pub(super) fn rebalance_hot_canonical_budgets(&mut self) {
        let total_budget = self.dynamic_total_hot_budget();
        let packed_weight = self.stats.canonical_packed_cache_hits.max(1);
        let oriented_weight = self.stats.canonical_oriented_cache_hits.max(1);
        let direct_parent_weight = (self.stats.direct_parent_cached_result_hits
            + self.stats.direct_parent_winner_hits)
            .max(1);
        let total_weight = packed_weight + oriented_weight + direct_parent_weight;

        self.hot_canonical_packed_budget = (total_budget * packed_weight / total_weight).max(1);
        self.hot_canonical_oriented_budget =
            (total_budget * oriented_weight / total_weight).max(1);
        self.hot_direct_parent_canonical_budget =
            (total_budget * direct_parent_weight / total_weight).max(1);

        let assigned = self.hot_canonical_packed_budget
            + self.hot_canonical_oriented_budget
            + self.hot_direct_parent_canonical_budget;
        if assigned < total_budget {
            self.hot_direct_parent_canonical_budget += total_budget - assigned;
        }

        self.trim_hot_canonical_caches_to_budget();
    }

    fn trim_hot_canonical_caches_to_budget(&mut self) {
        if self.hot_canonical_packed_cache.len() > self.hot_canonical_packed_budget {
            self.hot_canonical_packed_cache.clear();
        }
        if self.hot_canonical_oriented_cache.len() > self.hot_canonical_oriented_budget {
            self.hot_canonical_oriented_cache.clear();
        }
        if self.hot_direct_parent_canonical_cache.len() > self.hot_direct_parent_canonical_budget {
            self.hot_direct_parent_canonical_cache.clear();
        }
    }

    pub(super) fn canonical_cache_entries(&self) -> usize {
        self.canonical_packed_cache.len()
            + self.hot_canonical_packed_cache.len()
            + self.canonical_oriented_cache.len()
            + self.hot_canonical_oriented_cache.len()
            + self.direct_parent_canonical_cache.len()
            + self.hot_direct_parent_canonical_cache.len()
            + self.structural_fast_path_cache.len()
            + self.packed_structural_fast_path_cache.len()
    }

    pub(super) fn transient_cache_entries(&self) -> usize {
        self.jump_cache.len()
            + self.root_result_cache.len()
            + self.overlap_cache.len()
            + self.oriented_result_cache.len()
            + self.canonical_packed_cache.len()
            + self.hot_canonical_packed_cache.len()
            + self.canonical_oriented_cache.len()
            + self.hot_canonical_oriented_cache.len()
            + self.direct_parent_canonical_cache.len()
            + self.hot_direct_parent_canonical_cache.len()
            + self.structural_fast_path_cache.len()
            + self.packed_structural_fast_path_cache.len()
            + self.canonical_transform_cache.len()
            + self.packed_transform_compare_cache.len()
            + self.packed_transform_intern.len()
    }

    pub(super) fn transient_cache_pressure_entries(&self) -> usize {
        self.jump_cache.len()
            + self.root_result_cache.len()
            + self.overlap_cache.len()
            + self.oriented_result_cache.len()
            + self.canonical_packed_cache.len()
            + self.canonical_oriented_cache.len()
            + self.direct_parent_canonical_cache.len()
            + self.structural_fast_path_cache.len()
            + self.packed_structural_fast_path_cache.len()
            + self.canonical_transform_cache.len()
            + self.packed_transform_compare_cache.len()
            + self.packed_transform_intern.len()
    }

    pub(super) fn initialize_runtime_state(&mut self) {
        self.dead_leaf = self.intern_leaf(false);
        self.live_leaf = self.intern_leaf(true);
        self.empty_by_level.push(self.dead_leaf);
        self.reset_packed_transform_state();
    }

    pub(super) fn clear_transient_state(&mut self, preserve_hot_canonical: bool) {
        self.rebalance_hot_canonical_budgets();
        self.jump_cache.clear();
        self.root_result_cache.clear();
        self.overlap_cache.clear();
        #[cfg(test)]
        self.transform_cache.clear();
        self.canonical_transform_cache.clear();
        if !preserve_hot_canonical {
            self.oriented_result_cache.clear();
            self.materialized_packed_result_cache.clear();
        }
        self.reset_packed_transform_state();
        self.canonical_node_cache.clear();
        self.canonical_packed_cache.clear();
        self.canonical_oriented_cache.clear();
        self.direct_parent_canonical_cache.clear();
        if !preserve_hot_canonical {
            self.hot_canonical_packed_cache.clear();
            self.hot_canonical_oriented_cache.clear();
            self.hot_direct_parent_canonical_cache.clear();
        }
        self.structural_fast_path_cache.clear();
        self.packed_structural_fast_path_cache.clear();
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

    pub(super) fn maybe_garbage_collect(&mut self, reason: &'static str) {
        if reason == "skip" {
            self.stats.gc_reason = "skip";
            self.stats.gc_skips += 1;
            if self.transient_cache_pressure_entries() >= HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER {
                self.stats.gc_skipped_with_transient_growth += 1;
            }
            self.clear_transient_state(true);
            return;
        }

        self.stats.gc_runs += 1;
        self.stats.gc_transient_pressure_entries_before = self.transient_cache_pressure_entries();
        self.stats.gc_canonical_cache_entries_before = self.canonical_cache_entries();
        let (marked, live_nodes) = self.mark_live_nodes();
        self.stats.nodes_before_mark = self.node_count();
        self.stats.nodes_after_mark = live_nodes;
        let reclaimable = self.node_count().saturating_sub(live_nodes);
        let should_compact = (self.node_count() >= HASHLIFE_GC_MIN_NODES
            && reclaimable >= HASHLIFE_GC_MIN_RECLAIM
            && (reclaimable * 4 >= self.node_count() || reason != "skip"))
            || self.transient_cache_pressure_entries() >= HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER;

        if should_compact {
            self.stats.gc_reason = "compacted";
            self.stats.nodes_before_compact = self.node_count();
            self.compact_marked_nodes(marked);
            self.stats.nodes_after_compact = self.node_count();
            self.last_gc_nodes = self.node_count();
            self.clear_transient_state(true);
        } else {
            self.stats.gc_reason = if reason == "root_changed" {
                "root_changed_mark_only"
            } else {
                reason
            };
            self.stats.nodes_before_compact = self.node_count();
            self.stats.nodes_after_compact = self.node_count();
            self.last_gc_nodes = live_nodes;
        }
    }

    pub(super) fn mark_live_nodes(&mut self) -> (Vec<u64>, usize) {
        let mut marked = vec![0_u64; self.node_count().div_ceil(64)];
        let mut stack =
            Vec::with_capacity(self.empty_by_level.len() + self.retained_roots.len() + 2);
        stack.extend(self.empty_by_level.iter().copied());
        stack.extend(self.retained_roots.iter().copied());
        stack.push(self.dead_leaf);
        stack.push(self.live_leaf);

        let mut batch = [0_u64; 8];
        while !stack.is_empty() {
            let batch_len = stack.len().min(batch.len());
            self.stats.gc_mark_batches += 1;
            for slot in &mut batch[..batch_len] {
                *slot = stack.pop().expect("stack length already checked");
            }
            for &node_id in &batch[..batch_len] {
                let idx = node_id as usize;
                if node_id == NodeId::MAX || idx >= self.node_count() {
                    continue;
                }
                let word = idx / 64;
                let bit = 1_u64 << (idx % 64);
                if (marked[word] & bit) != 0 {
                    continue;
                }
                marked[word] |= bit;

                if self.node_columns.level(node_id) == 0 {
                    continue;
                }
                let [nw, ne, sw, se] = self.node_columns.quadrants(node_id);
                stack.push(nw);
                stack.push(ne);
                stack.push(sw);
                stack.push(se);
            }
        }

        let live = marked.iter().map(|word| word.count_ones() as usize).sum();
        (marked, live)
    }

    pub(super) fn record_retained_root(&mut self, root: NodeId) {
        if self.retained_roots.last().copied() == Some(root) {
            return;
        }
        self.retained_roots.push(root);
        if self.retained_roots.len() > HASHLIFE_MAX_RETAINED_ROOTS {
            let excess = self.retained_roots.len() - HASHLIFE_MAX_RETAINED_ROOTS;
            self.retained_roots.drain(0..excess);
        }
    }

    pub(super) fn compact_marked_nodes(&mut self, marked: Vec<u64>) {
        let old_hot_canonical_packed = self.hot_canonical_packed_cache.iter().collect::<Vec<_>>();
        let old_hot_canonical_oriented =
            self.hot_canonical_oriented_cache.iter().collect::<Vec<_>>();
        let old_hot_direct_parent = self
            .hot_direct_parent_canonical_cache
            .iter()
            .collect::<Vec<_>>();
        let old_oriented_result_cache = self.oriented_result_cache.iter().collect::<Vec<_>>();
        let old_materialized_packed_result_cache = self
            .materialized_packed_result_cache
            .iter()
            .collect::<Vec<_>>();
        let old_len = self.node_count();
        let old_levels = self.node_columns.levels.clone();
        let old_populations = self.node_columns.populations.clone();
        let old_nws = self.node_columns.nws.clone();
        let old_nes = self.node_columns.nes.clone();
        let old_sws = self.node_columns.sws.clone();
        let old_ses = self.node_columns.ses.clone();

        let mut remap = vec![NodeId::MAX; old_len];
        let live = marked.iter().map(|word| word.count_ones() as usize).sum();
        let mut compacted = NodeColumns::default();
        compacted.reserve(live);
        let mut old_idx = 0_usize;
        while old_idx < old_len {
            self.stats.gc_remap_batches += 1;
            let batch_end = (old_idx + 8).min(old_len);
            for current_idx in old_idx..batch_end {
                let word = current_idx / 64;
                let bit = 1_u64 << (current_idx % 64);
                if (marked[word] & bit) == 0 {
                    continue;
                }
                remap[current_idx] = compacted.len() as NodeId;
                compacted.push(
                    old_levels[current_idx],
                    old_populations[current_idx],
                    old_nws[current_idx],
                    old_nes[current_idx],
                    old_sws[current_idx],
                    old_ses[current_idx],
                    self.node_columns.symmetry_metadata[current_idx],
                );
            }
            old_idx = batch_end;
        }

        for node_idx in 0..compacted.len() {
            if compacted.levels[node_idx] == 0 {
                compacted.fingerprints[node_idx] =
                    crate::hashing::hash_leaf_population(compacted.populations[node_idx]);
                continue;
            }
            compacted.nws[node_idx] = remap[compacted.nws[node_idx] as usize];
            compacted.nes[node_idx] = remap[compacted.nes[node_idx] as usize];
            compacted.sws[node_idx] = remap[compacted.sws[node_idx] as usize];
            compacted.ses[node_idx] = remap[compacted.ses[node_idx] as usize];
            compacted.fingerprints[node_idx] = crate::hashing::hash_u64_words_with_level(
                compacted.levels[node_idx],
                [
                    compacted.nws[node_idx],
                    compacted.nes[node_idx],
                    compacted.sws[node_idx],
                    compacted.ses[node_idx],
                ],
            );
        }

        self.node_columns = compacted;
        self.intern = FlatTable::with_capacity(self.node_count().saturating_mul(2));
        for node_id in 0..self.node_count() {
            let node_id = node_id as NodeId;
            let key = if self.node_columns.level(node_id) == 0 {
                HashLifeEngine::packed_leaf_key(self.node_columns.population(node_id) == 1)
            } else {
                self.node_columns.packed_key(node_id)
            };
            self.intern.insert_with_fingerprint(
                key,
                self.node_columns.fingerprint(node_id),
                node_id,
            );
        }
        self.retained_roots.retain_mut(|root| {
            let index = *root as usize;
            if index >= remap.len() {
                return false;
            }
            let remapped = remap[index];
            if remapped == NodeId::MAX {
                return false;
            }
            *root = remapped;
            true
        });
        self.empty_by_level.retain_mut(|empty| {
            let index = *empty as usize;
            if index >= remap.len() {
                return false;
            }
            let remapped = remap[index];
            if remapped == NodeId::MAX {
                return false;
            }
            *empty = remapped;
            true
        });
        self.dead_leaf = remap
            .get(self.dead_leaf as usize)
            .copied()
            .filter(|&node| node != NodeId::MAX)
            .expect("dead leaf must survive compaction");
        self.live_leaf = remap
            .get(self.live_leaf as usize)
            .copied()
            .filter(|&node| node != NodeId::MAX)
            .expect("live leaf must survive compaction");

        self.hot_canonical_packed_cache =
            FlatTable::with_capacity(old_hot_canonical_packed.len().saturating_mul(2).max(16));
        for (key, value) in old_hot_canonical_packed {
            let (Some(remapped_key), Some(remapped_value)) = (
                Self::remap_packed_node_key(&remap, key),
                Self::remap_canonical_node_identity(&remap, value),
            ) else {
                continue;
            };
            self.hot_canonical_packed_cache
                .insert(remapped_key, remapped_value);
        }

        self.hot_canonical_oriented_cache =
            FlatTable::with_capacity(old_hot_canonical_oriented.len().saturating_mul(2).max(16));
        for (key, value) in old_hot_canonical_oriented {
            let (Some(remapped_packed), Some(remapped_value)) = (
                Self::remap_packed_node_key(&remap, key.packed),
                Self::remap_canonical_node_identity(&remap, value),
            ) else {
                continue;
            };
            self.hot_canonical_oriented_cache.insert(
                PackedSymmetryKey {
                    packed: remapped_packed,
                    symmetry: key.symmetry,
                },
                remapped_value,
            );
        }

        self.hot_direct_parent_canonical_cache =
            FlatTable::with_capacity(old_hot_direct_parent.len().saturating_mul(2).max(16));
        for (key, value) in old_hot_direct_parent {
            let Some(remapped_value) = Self::remap_canonical_node_identity(&remap, value) else {
                continue;
            };
            self.hot_direct_parent_canonical_cache.insert(
                DirectCanonicalParentKey {
                    level: key.level,
                    symmetry: key.symmetry,
                    children: key.children,
                },
                remapped_value,
            );
        }

        self.oriented_result_cache =
            FlatTable::with_capacity(old_oriented_result_cache.len().saturating_mul(2).max(16));
        for (key, value) in old_oriented_result_cache {
            let (Some(remapped_packed), Some(remapped_value)) = (
                Self::remap_packed_node_key(&remap, key.packed),
                Self::remap_packed_node_key(&remap, value),
            ) else {
                continue;
            };
            self.oriented_result_cache.insert(
                PackedSymmetryKey {
                    packed: remapped_packed,
                    symmetry: key.symmetry,
                },
                remapped_value,
            );
        }

        self.materialized_packed_result_cache = FlatTable::with_capacity(
            old_materialized_packed_result_cache
                .len()
                .saturating_mul(2)
                .max(16),
        );
        for (key, value) in old_materialized_packed_result_cache {
            let Some(remapped_key) = Self::remap_packed_node_key(&remap, key) else {
                continue;
            };
            let value_index = value as usize;
            if value_index >= remap.len() {
                continue;
            }
            let remapped_value = remap[value_index];
            if remapped_value == NodeId::MAX {
                continue;
            }
            self.materialized_packed_result_cache
                .insert(remapped_key, remapped_value);
        }
    }
}
