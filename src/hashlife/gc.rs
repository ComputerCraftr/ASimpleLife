use super::{GC_GROWTH_TRIGGER, GC_MIN_NODES, GC_MIN_RECLAIM, HashLifeOracle, NodeId, NodeKey};

impl HashLifeOracle {
    pub(super) fn initialize_runtime_state(&mut self) {
        self.dead_leaf = self.intern_leaf(false);
        self.live_leaf = self.intern_leaf(true);
        self.empty_by_level.push(self.dead_leaf);
    }

    pub(super) fn clear_transient_state(&mut self) {
        self.jump_cache.clear();
        self.root_result_cache.clear();
        self.overlap_cache.clear();
    }

    pub(super) fn gc_reason(
        &self,
        previous_root: Option<NodeId>,
        current_root: Option<NodeId>,
    ) -> &'static str {
        let root_changed = previous_root != current_root;
        let grew = self.nodes.len().saturating_sub(self.last_gc_nodes) >= GC_GROWTH_TRIGGER;
        if root_changed {
            "root_changed"
        } else if self.nodes.len() >= GC_MIN_NODES {
            "node_threshold"
        } else if grew {
            "growth_threshold"
        } else {
            "skip"
        }
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
        self.stats.nodes_before_mark = self.nodes.len();
        self.stats.nodes_after_mark = live_nodes;
        let reclaimable = self.nodes.len().saturating_sub(live_nodes);
        let should_compact = self.nodes.len() >= GC_MIN_NODES
            && reclaimable >= GC_MIN_RECLAIM
            && (reclaimable * 4 >= self.nodes.len() || reason != "skip");

        if should_compact {
            self.stats.gc_reason = "compacted";
            self.stats.nodes_before_compact = self.nodes.len();
            self.compact_marked_nodes(marked);
            self.stats.nodes_after_compact = self.nodes.len();
            self.last_gc_nodes = self.nodes.len();
        } else {
            self.stats.gc_reason = if reason == "root_changed" {
                "root_changed_mark_only"
            } else {
                reason
            };
            self.stats.nodes_before_compact = self.nodes.len();
            self.stats.nodes_after_compact = self.nodes.len();
            self.last_gc_nodes = live_nodes;
        }

        self.clear_transient_state();
    }

    pub(super) fn mark_live_nodes(&self) -> (Vec<bool>, usize) {
        let mut marked = vec![false; self.nodes.len()];
        let mut stack =
            Vec::with_capacity(self.empty_by_level.len() + self.retained_roots.len() + 2);
        stack.extend(self.empty_by_level.iter().copied());
        stack.extend(self.retained_roots.iter().copied());
        stack.push(self.dead_leaf);
        stack.push(self.live_leaf);

        while let Some(node_id) = stack.pop() {
            let idx = node_id as usize;
            if marked[idx] {
                continue;
            }
            marked[idx] = true;

            let node = self.nodes[idx];
            if node.level == 0 {
                continue;
            }
            stack.push(node.nw);
            stack.push(node.ne);
            stack.push(node.sw);
            stack.push(node.se);
        }

        let live = marked.iter().filter(|&&keep| keep).count();
        (marked, live)
    }

    pub(super) fn record_retained_root(&mut self, root: NodeId) {
        if self.retained_roots.last().copied() == Some(root) {
            return;
        }
        self.retained_roots.push(root);
        const MAX_RETAINED_ROOTS: usize = 1;
        if self.retained_roots.len() > MAX_RETAINED_ROOTS {
            let excess = self.retained_roots.len() - MAX_RETAINED_ROOTS;
            self.retained_roots.drain(0..excess);
        }
    }

    pub(super) fn compact_marked_nodes(&mut self, marked: Vec<bool>) {
        let mut remap = vec![NodeId::MAX; self.nodes.len()];
        let mut compacted = Vec::with_capacity(marked.iter().filter(|&&keep| keep).count());
        for (old_idx, node) in self.nodes.iter().copied().enumerate() {
            if !marked[old_idx] {
                continue;
            }
            remap[old_idx] = compacted.len() as NodeId;
            compacted.push(node);
        }

        for node in &mut compacted {
            if node.level == 0 {
                continue;
            }
            node.nw = remap[node.nw as usize];
            node.ne = remap[node.ne as usize];
            node.sw = remap[node.sw as usize];
            node.se = remap[node.se as usize];
        }

        self.nodes = compacted;
        self.intern.clear();
        for (node_id, node) in self.nodes.iter().copied().enumerate() {
            let key = if node.level == 0 {
                NodeKey::Leaf(node.population == 1)
            } else {
                NodeKey::Internal {
                    level: node.level,
                    nw: node.nw,
                    ne: node.ne,
                    sw: node.sw,
                    se: node.se,
                }
            };
            self.intern.insert(key, node_id as NodeId);
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

#[cfg(test)]
impl HashLifeOracle {
    pub(crate) fn runtime_stats(&self) -> super::HashLifeRuntimeStats {
        super::HashLifeRuntimeStats {
            nodes: self.nodes.len(),
            intern: self.intern.len(),
            empty_levels: self.empty_by_level.len(),
            jump_cache: self.jump_cache.len(),
            retained_roots: self.retained_roots.len(),
            overlap_cache: self.overlap_cache.len(),
            jump_cache_hits: self.stats.jump_cache_hits,
            jump_cache_misses: self.stats.jump_cache_misses,
            root_result_cache_hits: self.stats.root_result_cache_hits,
            root_result_cache_misses: self.stats.root_result_cache_misses,
            overlap_cache_hits: self.stats.overlap_cache_hits,
            overlap_cache_misses: self.stats.overlap_cache_misses,
            gc_runs: self.stats.gc_runs,
            gc_skips: self.stats.gc_skips,
            nodes_before_mark: self.stats.nodes_before_mark,
            nodes_after_mark: self.stats.nodes_after_mark,
            nodes_before_compact: self.stats.nodes_before_compact,
            nodes_after_compact: self.stats.nodes_after_compact,
            jump_cache_before_clear: self.stats.jump_cache_before_clear,
            gc_reason: self.stats.gc_reason,
            builder_partitions: self.stats.builder_partitions,
        }
    }
}
