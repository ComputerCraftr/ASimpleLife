use super::{FlatTable, HashLifeEngine, NodeColumns, NodeId};
use crate::cache_policy::{HASHLIFE_GC_MIN_NODES, HASHLIFE_GC_MIN_RECLAIM, hashlife_gc_reason};

const HASHLIFE_MAX_RETAINED_ROOTS: usize = 1;

impl HashLifeEngine {
    pub(super) fn initialize_runtime_state(&mut self) {
        self.dead_leaf = self.intern_leaf(false);
        self.live_leaf = self.intern_leaf(true);
        self.empty_by_level.push(self.dead_leaf);
        self.reset_packed_transform_state();
    }

    pub(super) fn clear_transient_state(&mut self) {
        self.jump_cache.clear();
        self.root_result_cache.clear();
        self.overlap_cache.clear();
        #[cfg(test)]
        self.transform_cache.clear();
        self.canonical_transform_cache.clear();
        self.oriented_result_cache.clear();
        self.packed_transform_compare_cache.clear();
        self.reset_packed_transform_state();
        self.canonical_node_cache.clear();
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
            self.clear_transient_state();
            return;
        }

        self.stats.gc_runs += 1;
        let (marked, live_nodes) = self.mark_live_nodes();
        self.stats.nodes_before_mark = self.node_count();
        self.stats.nodes_after_mark = live_nodes;
        let reclaimable = self.node_count().saturating_sub(live_nodes);
        let should_compact = self.node_count() >= HASHLIFE_GC_MIN_NODES
            && reclaimable >= HASHLIFE_GC_MIN_RECLAIM
            && (reclaimable * 4 >= self.node_count() || reason != "skip");

        if should_compact {
            self.stats.gc_reason = "compacted";
            self.stats.nodes_before_compact = self.node_count();
            self.compact_marked_nodes(marked);
            self.stats.nodes_after_compact = self.node_count();
            self.last_gc_nodes = self.node_count();
            self.clear_transient_state();
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

        for root in &mut self.retained_roots {
            *root = remap[*root as usize];
        }
        for empty in &mut self.empty_by_level {
            *empty = remap[*empty as usize];
        }
        self.dead_leaf = remap[self.dead_leaf as usize];
        self.live_leaf = remap[self.live_leaf as usize];
    }
}
