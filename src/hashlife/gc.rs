use crate::cache_policy::{HASHLIFE_GC_MIN_NODES, HASHLIFE_GC_MIN_RECLAIM, hashlife_gc_reason};
#[cfg(test)]
use crate::bitgrid::BitGrid;
#[cfg(test)]
use crate::symmetry::D4Symmetry as Symmetry;

use super::{FlatTable, HashLifeEngine, NodeColumns, NodeId};

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
        self.packed_symmetry_children_cache.clear();
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

        self.clear_transient_state();
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
            self.intern
                .insert_with_fingerprint(key, self.node_columns.fingerprint(node_id), node_id);
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
impl HashLifeEngine {
    pub(crate) fn verify_overlap_batch_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut nodes = [0; crate::simd_layout::SIMD_BATCH_LANES];
        let mut active = 0;
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) < 2 {
                continue;
            }
            nodes[active] = node;
            active += 1;
            if active == crate::simd_layout::SIMD_BATCH_LANES {
                break;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.push(se);
            stack.push(sw);
            stack.push(ne);
            stack.push(nw);
        }

        if active == 0 {
            return true;
        }

        let batched = self.probe_and_build_overlaps_staged(&nodes, active);
        for lane in 0..active {
            if batched[lane] != self.overlapping_subnodes(nodes[lane]) {
                return false;
            }
        }
        true
    }

    pub(crate) fn verify_canonical_overlap_batch_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut nodes = [0; crate::simd_layout::SIMD_BATCH_LANES];
        let mut canonical_keys =
            [super::CanonicalJumpKey {
                packed: super::PackedNodeKey::new(0, [0; 4]),
                step_exp: 1,
            }; crate::simd_layout::SIMD_BATCH_LANES];
        let mut active = 0;
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) < 2 {
                continue;
            }
            nodes[active] = node;
            canonical_keys[active] = self.canonical_jump_key_packed((node, 1)).0;
            active += 1;
            if active == crate::simd_layout::SIMD_BATCH_LANES {
                break;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.push(se);
            stack.push(sw);
            stack.push(ne);
            stack.push(nw);
        }

        if active == 0 {
            return true;
        }

        let raw = self.probe_and_build_overlaps_staged(&nodes, active);
        let canonical = self.probe_and_build_overlaps_from_canonical_keys_staged(&canonical_keys, active);
        for lane in 0..active {
            if raw[lane] != canonical[lane] {
                return false;
            }
        }
        true
    }

    pub(crate) fn verify_canonical_child_key_batch_parity(&mut self, grid: &crate::bitgrid::BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let overlaps = self.overlapping_subnodes(root);
        let nodes = [
            overlaps[8],
            overlaps[7],
            overlaps[6],
            overlaps[5],
            overlaps[4],
            overlaps[3],
            overlaps[2],
            overlaps[1],
            overlaps[0],
        ];
        let batched = self.discovered_jump_tasks_from_nodes(nodes, 2);
        for lane in 0..9 {
            if batched[lane].key != self.canonical_jump_key_packed((nodes[lane], 2)).0 {
                return false;
            }
        }
        true
    }

    pub(crate) fn duplicate_overlap_batch_dedupe_stats(
        &mut self,
        grid: &crate::bitgrid::BitGrid,
    ) -> (usize, usize) {
        let (root, _, _) = self.embed_grid_state(grid);
        let before = (self.stats.overlap_cache_misses, self.stats.overlap_local_reuse_lanes);
        let mut nodes = [0; crate::simd_layout::SIMD_BATCH_LANES];
        nodes[0] = root;
        nodes[1] = root;
        let overlaps = self.probe_and_build_overlaps_staged(&nodes, 2);
        assert_eq!(overlaps[0], overlaps[1]);
        (
            self.stats.overlap_cache_misses - before.0,
            self.stats.overlap_local_reuse_lanes - before.1,
        )
    }

    pub(crate) fn duplicate_jump_batch_query_stats(
        &mut self,
        grid: &crate::bitgrid::BitGrid,
    ) -> (usize, usize) {
        let (root, _, _) = self.embed_grid_state(grid);
        self.insert_jump_result((root, 0), root);
        let before = (
            self.stats.jump_batch_unique_queries,
            self.stats.jump_batch_reused_queries,
        );
        let results = self.jump_result_batch([root, root, root, root], 0);
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
        assert_eq!(results[2], results[3]);
        (
            self.stats.jump_batch_unique_queries - before.0,
            self.stats.jump_batch_reused_queries - before.1,
        )
    }

    pub(crate) fn verify_packed_jump_cache_roundtrip(
        &mut self,
        grid: &crate::bitgrid::BitGrid,
        step_exp: u32,
    ) -> bool {
        let embedded = self.embed_for_jump(grid, step_exp);
        let root = embedded.root;
        let before_hits = self.stats.packed_cache_result_hits;
        let result = self.advance_pow2(root, step_exp);
        let cached = self.cached_jump_result((root, step_exp));
        cached == Some(result) && self.stats.packed_cache_result_hits > before_hits
    }

    pub(crate) fn duplicate_oriented_result_cache_stats(
        &mut self,
        grid: &crate::bitgrid::BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);
        let before_first = (
            self.stats.packed_cache_result_materializations,
            self.stats.packed_inverse_transform_hits,
        );
        let first = self.materialize_oriented_packed_result(
            packed,
            Symmetry::Identity,
            Symmetry::Rotate90,
        );
        let first_delta = (
            self.stats.packed_cache_result_materializations - before_first.0,
            self.stats.packed_inverse_transform_hits - before_first.1,
        );
        let before_second = (
            self.stats.packed_cache_result_materializations,
            self.stats.packed_inverse_transform_hits,
        );
        let second = self.materialize_oriented_packed_result(
            packed,
            Symmetry::Identity,
            Symmetry::Rotate90,
        );
        let second_delta = (
            self.stats.packed_cache_result_materializations - before_second.0,
            self.stats.packed_inverse_transform_hits - before_second.1,
        );
        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn verify_packed_transform_parity(&mut self, grid: &crate::bitgrid::BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut stack = vec![root];
        let mut checked = 0;
        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) == 0 {
                continue;
            }
            let packed = self.node_columns.packed_key(node);
            for symmetry in crate::symmetry::D4Symmetry::ALL {
                let expected = self.transform_node(node, symmetry);
                let transformed = self.transform_packed_node_key(packed, symmetry);
                let actual = self.materialize_packed_transform_root(transformed);
                if expected != actual {
                    return false;
                }
            }
            checked += 1;
            if checked == crate::simd_layout::SIMD_BATCH_LANES {
                break;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.push(se);
            stack.push(sw);
            stack.push(ne);
            stack.push(nw);
        }
        true
    }

    pub(crate) fn verify_packed_canonicalization_symmetry_parity(
        &mut self,
        grid: &crate::bitgrid::BitGrid,
    ) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut stack = vec![root];
        let mut checked = 0;
        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) == 0 {
                continue;
            }
            let canonical = self.canonicalize_packed_node(node).packed;
            for symmetry in crate::symmetry::D4Symmetry::ALL {
                let transformed = self.transform_node(node, symmetry);
                let transformed_canonical = self.canonicalize_packed_node(transformed).packed;
                if transformed_canonical != canonical {
                    return false;
                }
            }
            checked += 1;
            if checked == crate::simd_layout::SIMD_BATCH_LANES {
                break;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.push(se);
            stack.push(sw);
            stack.push(ne);
            stack.push(nw);
        }
        true
    }

    pub(crate) fn runtime_stats(&self) -> super::HashLifeRuntimeStats {
        super::HashLifeRuntimeStats {
            nodes: self.node_count(),
            intern: self.intern.len(),
            empty_levels: self.empty_by_level.len(),
            jump_cache: self.jump_cache.len(),
            retained_roots: self.retained_roots.len(),
            overlap_cache: self.overlap_cache.len(),
            jump_cache_hits: self.stats.jump_cache_hits,
            symmetric_jump_cache_hits: self.stats.symmetric_jump_cache_hits,
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
            builder_frames: self.stats.builder_frames,
            builder_partitions: self.stats.builder_partitions,
            builder_max_stack: self.stats.builder_max_stack,
            scheduler_tasks: self.stats.scheduler_tasks,
            scheduler_ready_max: self.stats.scheduler_ready_max,
            simd_disabled_fast_exits: self.stats.simd_disabled_fast_exits,
            step0_simd_lanes: self.stats.step0_simd_lanes,
            phase1_simd_lanes: self.stats.phase1_simd_lanes,
            phase2_simd_lanes: self.stats.phase2_simd_lanes,
            step0_simd_batches: self.stats.step0_simd_batches,
            phase1_simd_batches: self.stats.phase1_simd_batches,
            phase2_simd_batches: self.stats.phase2_simd_batches,
            step0_provisional_records: self.stats.step0_provisional_records,
            phase1_provisional_records: self.stats.phase1_provisional_records,
            phase2_provisional_records: self.stats.phase2_provisional_records,
            scalar_commit_lanes: self.stats.scalar_commit_lanes,
            join_shortcut_avoided: self.stats.join_shortcut_avoided,
            dependency_stalls: self.stats.dependency_stalls,
            step0_ready_max: self.stats.step0_ready_max,
            phase1_ready_max: self.stats.phase1_ready_max,
            phase2_ready_max: self.stats.phase2_ready_max,
            canonical_batch_lanes: self.stats.canonical_batch_lanes,
            canonical_batch_batches: self.stats.canonical_batch_batches,
            overlap_prep_lanes: self.stats.overlap_prep_lanes,
            overlap_prep_batches: self.stats.overlap_prep_batches,
            recursive_overlap_batch_lanes: self.stats.recursive_overlap_batch_lanes,
            recursive_overlap_batch_batches: self.stats.recursive_overlap_batch_batches,
            overlap_local_reuse_lanes: self.stats.overlap_local_reuse_lanes,
            cache_probe_batches: self.stats.cache_probe_batches,
            scheduler_probe_batches: self.stats.scheduler_probe_batches,
            symmetry_gate_allowed: self.stats.symmetry_gate_allowed,
            symmetry_gate_blocked: self.stats.symmetry_gate_blocked,
            canonical_node_cache_hits: self.stats.canonical_node_cache_hits,
            canonical_node_cache_misses: self.stats.canonical_node_cache_misses,
            jump_presence_probe_batches: self.stats.jump_presence_probe_batches,
            jump_presence_probe_lanes: self.stats.jump_presence_probe_lanes,
            jump_presence_probe_hits: self.stats.jump_presence_probe_hits,
            jump_batch_unique_queries: self.stats.jump_batch_unique_queries,
            jump_batch_reused_queries: self.stats.jump_batch_reused_queries,
            cached_fingerprint_probes: self.stats.cached_fingerprint_probes,
            recomputed_fingerprint_probes: self.stats.recomputed_fingerprint_probes,
            gc_mark_batches: self.stats.gc_mark_batches,
            gc_remap_batches: self.stats.gc_remap_batches,
            packed_d4_canonicalization_misses: self.stats.packed_d4_canonicalization_misses,
            packed_inverse_transform_hits: self.stats.packed_inverse_transform_hits,
            packed_recursive_transform_hits: self.stats.packed_recursive_transform_hits,
            packed_recursive_transform_misses: self.stats.packed_recursive_transform_misses,
            packed_overlap_outputs_produced: self.stats.packed_overlap_outputs_produced,
            packed_cache_result_lookups: self.stats.packed_cache_result_lookups,
            packed_cache_result_hits: self.stats.packed_cache_result_hits,
            packed_cache_result_materializations: self.stats.packed_cache_result_materializations,
            #[cfg(test)]
            transformed_node_materializations: self.stats.transformed_node_materializations,
        }
    }

    pub(crate) fn diagnostic_summary(&self) -> super::HashLifeDiagnosticSummary {
        let stats = self.runtime_stats();
        let jump_full_total = stats.jump_cache_hits + stats.jump_cache_misses;
        let jump_presence_total = stats.jump_presence_probe_hits
            + stats
                .jump_presence_probe_lanes
                .saturating_sub(stats.jump_presence_probe_hits);
        let overlap_total = stats.overlap_cache_hits + stats.overlap_cache_misses;
        let symmetry_gate_total = stats.symmetry_gate_allowed + stats.symmetry_gate_blocked;
        let canonical_cache_total = stats.canonical_node_cache_hits + stats.canonical_node_cache_misses;
        let packed_cache_total = stats.packed_cache_result_lookups;
        let total_simd_lanes =
            stats.step0_simd_lanes + stats.phase1_simd_lanes + stats.phase2_simd_lanes;
        let total_provisionals = stats.step0_provisional_records
            + stats.phase1_provisional_records
            + stats.phase2_provisional_records;
        let recursive_overlap_total = stats.overlap_prep_batches.max(1);
        let gc_mark_total = stats.nodes_before_mark.max(1);
        let gc_compact_total = stats.nodes_before_compact.max(1);

        super::HashLifeDiagnosticSummary {
            total_nodes: stats.nodes,
            retained_roots: stats.retained_roots,
            nodes_match_intern: stats.nodes == stats.intern,
            dependency_stalls: stats.dependency_stalls,
            jump_full_hit_rate: stats.jump_cache_hits as f64 / jump_full_total.max(1) as f64,
            jump_presence_hit_rate: stats.jump_presence_probe_hits as f64
                / jump_presence_total.max(1) as f64,
            overlap_hit_rate: stats.overlap_cache_hits as f64 / overlap_total.max(1) as f64,
            overlap_local_reuse_rate: stats.overlap_local_reuse_lanes as f64
                / stats.overlap_prep_lanes.max(1) as f64,
            symmetry_gate_allow_rate: stats.symmetry_gate_allowed as f64
                / symmetry_gate_total.max(1) as f64,
            canonical_cache_hit_rate: stats.canonical_node_cache_hits as f64
                / canonical_cache_total.max(1) as f64,
            packed_cache_hit_rate: stats.packed_cache_result_hits as f64
                / packed_cache_total.max(1) as f64,
            symmetry_jump_hits: stats.symmetric_jump_cache_hits,
            simd_lane_coverage: total_simd_lanes as f64 / total_provisionals.max(1) as f64,
            scalar_commit_ratio: stats.scalar_commit_lanes as f64
                / total_provisionals.max(1) as f64,
            probes_per_scheduler_task: stats.scheduler_probe_batches as f64
                / stats.scheduler_tasks.max(1) as f64,
            recursive_overlap_batch_rate: stats.recursive_overlap_batch_batches as f64
                / recursive_overlap_total as f64,
            gc_reclaim_ratio: (stats.nodes_before_mark.saturating_sub(stats.nodes_after_mark))
                as f64
                / gc_mark_total as f64,
            gc_compact_ratio: (stats.nodes_before_compact.saturating_sub(stats.nodes_after_compact))
                as f64
                / gc_compact_total as f64,
            gc_reason: stats.gc_reason,
            gc_runs: stats.gc_runs,
            gc_skips: stats.gc_skips,
            packed_d4_canonicalization_misses: stats.packed_d4_canonicalization_misses,
            packed_inverse_transform_hits: stats.packed_inverse_transform_hits,
            packed_recursive_transform_hits: stats.packed_recursive_transform_hits,
            packed_recursive_transform_misses: stats.packed_recursive_transform_misses,
            packed_overlap_outputs_produced: stats.packed_overlap_outputs_produced,
            packed_cache_result_materializations: stats.packed_cache_result_materializations,
            #[cfg(test)]
            transformed_node_materializations: stats.transformed_node_materializations,
        }
    }

    pub(crate) fn verify_node_fingerprint_invariants(&self) -> bool {
        for node_id in 0..self.node_count() {
            let node_id = node_id as NodeId;
            let packed = self.node_columns.packed_key(node_id);
            if self.node_columns.fingerprint(node_id) != super::FlatKey::fingerprint(&packed) {
                return false;
            }
        }
        true
    }

    pub(crate) fn verify_intern_fingerprint_fast_path_parity(&self) -> bool {
        for node_id in 0..self.node_count() {
            let node_id = node_id as NodeId;
            let key = self.node_columns.packed_key(node_id);
            let fingerprint = self.node_columns.fingerprint(node_id);
            if self.intern.get(&key) != self.intern.get_with_fingerprint(&key, fingerprint) {
                return false;
            }
        }
        true
    }

}
