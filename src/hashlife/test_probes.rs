use crate::bitgrid::BitGrid;
use crate::symmetry::D4Symmetry as Symmetry;

use super::{HashLifeEngine, NodeId, PackedNodeKey, PackedSymmetryKey, Phase2CommitLane};

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
        (0..active).all(|lane| batched[lane] == self.overlapping_subnodes(nodes[lane]))
    }

    pub(crate) fn verify_canonical_overlap_batch_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut nodes = [0; crate::simd_layout::SIMD_BATCH_LANES];
        let mut canonical_keys =
            [super::CanonicalJumpKey::empty(); crate::simd_layout::SIMD_BATCH_LANES];
        let mut canonical_packed =
            [super::PackedNodeKey::new(0, [0; 4]); crate::simd_layout::SIMD_BATCH_LANES];
        let mut active = 0;
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) < 2 {
                continue;
            }
            nodes[active] = node;
            let canonical = self.canonicalize_packed_node(node);
            canonical_keys[active] = super::CanonicalJumpKey {
                structural: canonical.node.structural,
                step_exp: 1,
            };
            canonical_packed[active] = canonical.node.packed;
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
        let mut canonical_nodes = [0; crate::simd_layout::SIMD_BATCH_LANES];
        for lane in 0..active {
            canonical_nodes[lane] = self.materialize_packed_node_key(canonical_packed[lane]);
        }
        let raw = self.probe_and_build_overlaps_staged(&canonical_nodes, active);
        let mut identities = [super::CanonicalNodeIdentity {
            packed: super::PackedNodeKey::new(0, [0; 4]),
            structural: super::CanonicalStructKey::new(0, [0; 4]),
            symmetry: super::Symmetry::Identity,
        }; crate::simd_layout::SIMD_BATCH_LANES];
        let mut fingerprints = [0_u64; crate::simd_layout::SIMD_BATCH_LANES];
        for lane in 0..active {
            identities[lane] = super::CanonicalNodeIdentity {
                packed: canonical_packed[lane],
                structural: canonical_keys[lane].structural,
                symmetry: super::Symmetry::Identity,
            };
            fingerprints[lane] =
                crate::flat_table::FlatKey::fingerprint(&canonical_keys[lane].structural);
        }
        self.stats.cache_probe_batches += 1;
        self.stats.scheduler_probe_batches += 1;
        self.stats.overlap_prep_batches += 1;
        self.stats.packed_overlap_outputs_produced += active;
        let canonical =
            self.probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, active);
        (0..active).all(|lane| raw[lane] == canonical[lane])
    }

    pub(crate) fn verify_canonical_child_key_batch_parity(&mut self, grid: &BitGrid) -> bool {
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
        (0..9).all(|lane| {
            batched[lane].key
                == self
                    .canonical_jump_probe((nodes[lane], 2))
                    .key
        })
    }

    pub(crate) fn duplicate_overlap_batch_dedupe_stats(
        &mut self,
        grid: &BitGrid,
    ) -> (usize, usize) {
        let (root, _, _) = self.embed_grid_state(grid);
        let before = (
            self.stats.overlap_cache_misses,
            self.stats.overlap_local_reuse_lanes,
        );
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

    pub(crate) fn duplicate_jump_batch_query_stats(&mut self, grid: &BitGrid) -> (usize, usize) {
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
        grid: &BitGrid,
        step_exp: u32,
    ) -> bool {
        let embedded = self.embed_for_jump(grid, step_exp);
        let root = embedded.root;
        let before_hits = self.stats.jump_result_cache_hits;
        let result = self.advance_pow2(root, step_exp);
        let cached = self.cached_jump_result((root, step_exp));
        cached == Some(result) && self.stats.jump_result_cache_hits > before_hits
    }

    pub(crate) fn repeated_canonical_result_insertion_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let embedded = self.embed_for_jump(grid, 1);
        let root = embedded.root;
        let result = self.advance_pow2(root, 1);
        let canonical_key = self.canonical_jump_probe((root, 1)).key;

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        self.insert_canonical_jump_result(canonical_key, result);
        let first_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        self.insert_canonical_jump_result(canonical_key, result);
        let second_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        (first_delta, second_delta)
    }

    pub(crate) fn repeated_jump_result_insertion_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let embedded = self.embed_for_jump(grid, 1);
        let root = embedded.root;
        let result = self.advance_pow2(root, 1);

        let before = (
            self.stats.canonical_node_cache_hits,
            self.stats.canonical_node_cache_misses,
        );
        self.insert_jump_result((root, 1), result);
        let first_delta = (
            self.stats.canonical_node_cache_hits - before.0,
            self.stats.canonical_node_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_node_cache_hits,
            self.stats.canonical_node_cache_misses,
        );
        self.insert_jump_result((root, 1), result);
        let second_delta = (
            self.stats.canonical_node_cache_hits - before.0,
            self.stats.canonical_node_cache_misses - before.1,
        );

        (first_delta, second_delta)
    }

    pub(crate) fn duplicate_oriented_result_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.packed_cache_result_materializations,
            self.stats.packed_inverse_transform_hits,
        );
        let first =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let first_delta = (
            self.stats.packed_cache_result_materializations - before.0,
            self.stats.packed_inverse_transform_hits - before.1,
        );

        let before = (
            self.stats.packed_cache_result_materializations,
            self.stats.packed_inverse_transform_hits,
        );
        let second =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let second_delta = (
            self.stats.packed_cache_result_materializations - before.0,
            self.stats.packed_inverse_transform_hits - before.1,
        );

        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn repeated_canonical_packed_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        let first = self.canonicalize_packed_key_for_snapshot(packed);
        let first_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        let second = self.canonicalize_packed_key_for_snapshot(packed);
        let second_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        assert_eq!(first.node, second.node);
        (first_delta, second_delta)
    }

    pub(crate) fn repeated_oriented_canonical_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let rotated = self.transform_node(root, Symmetry::Rotate90);
        let packed = self.node_columns.packed_key(rotated);

        let before = (
            self.stats.canonical_oriented_cache_hits,
            self.stats.canonical_oriented_cache_misses,
        );
        let first = self.canonicalize_packed_under_symmetry(packed, Symmetry::Rotate90);
        let first_delta = (
            self.stats.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_oriented_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_oriented_cache_hits,
            self.stats.canonical_oriented_cache_misses,
        );
        let second = self.canonicalize_packed_under_symmetry(packed, Symmetry::Rotate90);
        let second_delta = (
            self.stats.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_oriented_cache_misses - before.1,
        );

        assert_eq!(first.node, second.node);
        (first_delta, second_delta)
    }

    pub(crate) fn canonical_hot_cache_survives_skip_gc(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        let first = self.canonicalize_packed_key_for_snapshot(packed);
        let populate_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        let _warm = self.canonicalize_packed_key_for_snapshot(packed);
        let warm_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        self.maybe_garbage_collect("skip");

        let protected_entries = (
            self.hot_canonical_packed_cache.len(),
            self.hot_direct_parent_canonical_cache.len(),
        );

        let before = (
            self.stats.canonical_packed_cache_hits,
            self.stats.canonical_packed_cache_misses,
        );
        let second = self.canonicalize_packed_key_for_snapshot(packed);
        let retained_delta = (
            self.stats.canonical_packed_cache_hits - before.0,
            self.stats.canonical_packed_cache_misses - before.1,
        );

        assert_eq!(first.node, second.node);
        assert!(warm_delta.0 > 0);
        assert_eq!(warm_delta.1, 0);
        (populate_delta, protected_entries, retained_delta)
    }

    pub(crate) fn repeated_nonidentity_jump_result_insertion_oriented_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let embedded = self.embed_for_jump(grid, 1);
        let mut source_root = embedded.root;
        for symmetry in Symmetry::ALL.into_iter().skip(1) {
            let candidate = self.transform_node(embedded.root, symmetry);
            if self.canonical_jump_probe((candidate, 1)).node.symmetry != Symmetry::Identity {
                source_root = candidate;
                break;
            }
        }
        assert_ne!(
            self.canonical_jump_probe((source_root, 1)).node.symmetry,
            Symmetry::Identity,
            "expected a non-identity canonical probe for oriented insertion stats"
        );
        let result = self.advance_pow2(source_root, 1);
        self.canonical_oriented_cache.clear();

        let before = (
            self.stats.canonical_oriented_cache_hits,
            self.stats.canonical_oriented_cache_misses,
        );
        self.insert_jump_result((source_root, 1), result);
        let first_delta = (
            self.stats.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_oriented_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_oriented_cache_hits,
            self.stats.canonical_oriented_cache_misses,
        );
        self.insert_jump_result((source_root, 1), result);
        let second_delta = (
            self.stats.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_oriented_cache_misses - before.1,
        );

        (first_delta, second_delta)
    }

    pub(crate) fn identity_packed_canonicalization_avoids_oriented_cache(
        &mut self,
        grid: &BitGrid,
    ) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);
        let before = self.stats.canonical_oriented_cache_lookups;
        let _ = self.canonicalize_packed_key_for_snapshot(packed);
        self.stats.canonical_oriented_cache_lookups == before
    }

    pub(crate) fn repeated_gate_blocked_probe_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);

        let before = (
            self.stats.structural_fast_path_hits,
            self.stats.structural_fast_path_misses,
        );
        let first = self.canonicalize_blocked_jump_node_for_tests(root);
        let first_delta = (
            self.stats.structural_fast_path_hits - before.0,
            self.stats.structural_fast_path_misses - before.1,
        );

        let before = (
            self.stats.structural_fast_path_hits,
            self.stats.structural_fast_path_misses,
        );
        let second = self.canonicalize_blocked_jump_node_for_tests(root);
        let second_delta = (
            self.stats.structural_fast_path_hits - before.0,
            self.stats.structural_fast_path_misses - before.1,
        );

        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn repeated_direct_parent_winner_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);
        let before = (
            self.stats.direct_parent_winner_hits,
            self.stats.symmetry_scan_fallbacks,
        );
        let first = self
            .direct_parent_winner_for_tests(packed, Symmetry::Identity)
            .expect("direct parent winner should exist for non-leaf packed node");
        let first_delta = (
            self.stats.direct_parent_winner_hits - before.0,
            self.stats.symmetry_scan_fallbacks - before.1,
        );

        let before = (
            self.stats.direct_parent_winner_hits,
            self.stats.symmetry_scan_fallbacks,
        );
        let second = self
            .direct_parent_winner_for_tests(packed, Symmetry::Identity)
            .expect("direct parent winner should be cached after warmup");
        let second_delta = (
            self.stats.direct_parent_winner_hits - before.0,
            self.stats.symmetry_scan_fallbacks - before.1,
        );

        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn repeated_direct_parent_cached_result_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.direct_parent_cached_result_hits,
            self.stats.canonical_transform_root_reconstructions,
            self.stats.direct_parent_winner_fallbacks,
        );
        let first = self.canonicalize_packed_direct_for_tests(packed, Symmetry::Identity);
        let first_delta = (
            self.stats.direct_parent_cached_result_hits - before.0,
            self.stats.canonical_transform_root_reconstructions - before.1,
            self.stats.direct_parent_winner_fallbacks - before.2,
        );

        let before = (
            self.stats.direct_parent_cached_result_hits,
            self.stats.canonical_transform_root_reconstructions,
            self.stats.direct_parent_winner_fallbacks,
        );
        let second = self.canonicalize_packed_direct_for_tests(packed, Symmetry::Identity);
        let second_delta = (
            self.stats.direct_parent_cached_result_hits - before.0,
            self.stats.canonical_transform_root_reconstructions - before.1,
            self.stats.direct_parent_winner_fallbacks - before.2,
        );

        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn direct_parent_cache_respects_symmetry_mode(
        &mut self,
        grid: &BitGrid,
    ) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);
        let identity = self.canonicalize_packed_direct_for_tests(packed, Symmetry::Identity);
        let rotated = self.canonicalize_packed_direct_for_tests(packed, Symmetry::Rotate90);

        let mut fresh = HashLifeEngine::default();
        let (fresh_root, _, _) = fresh.embed_grid_state(grid);
        let fresh_packed = fresh.node_columns.packed_key(fresh_root);
        let fresh_identity =
            fresh.canonicalize_packed_direct_for_tests(fresh_packed, Symmetry::Identity);
        let fresh_rotated =
            fresh.canonicalize_packed_direct_for_tests(fresh_packed, Symmetry::Rotate90);

        fresh_identity != fresh_rotated
            && identity == fresh_identity
            && rotated == fresh_rotated
    }

    pub(crate) fn duplicate_phase2_canonicalization_stats(
        &mut self,
        grid: &BitGrid,
    ) -> (usize, usize, usize) {
        let embedded = self.embed_for_jump(grid, 1);
        let root = embedded.root;
        let result = self.advance_pow2(root, 1);
        let packed_input = PackedSymmetryKey {
            packed: self.node_columns.packed_key(result),
            symmetry: Symmetry::Identity,
        };
        let canonical_key = self.canonical_jump_probe((root, 1)).key;
        let mut lanes = [
            Phase2CommitLane {
                key: canonical_key,
                fallback: result,
                result,
                unique_input_index: 0,
                packed_input,
                canonical_entry: PackedSymmetryKey {
                    packed: PackedNodeKey::new(0, [0; 4]),
                    symmetry: Symmetry::Identity,
                },
            };
            4
        ];

        let before = (
            self.stats.canonical_result_unique_inputs,
            self.stats.canonical_result_unique_parent_shapes,
            self.stats.canonical_result_batch_local_reuses,
        );
        self.canonicalize_phase2_commit_lanes(&mut lanes);
        assert!(lanes.windows(2).all(|pair| pair[0].canonical_entry == pair[1].canonical_entry));
        (
            self.stats.canonical_result_unique_inputs - before.0,
            self.stats.canonical_result_unique_parent_shapes - before.1,
            self.stats.canonical_result_batch_local_reuses - before.2,
        )
    }

    pub(crate) fn verify_packed_transform_parity(&mut self, grid: &BitGrid) -> bool {
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
        grid: &BitGrid,
    ) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut stack = vec![root];
        let mut checked = 0;
        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) == 0 {
                continue;
            }
            let canonical = self.canonicalize_packed_node(node);
            for symmetry in crate::symmetry::D4Symmetry::ALL {
                let transformed = self.transform_node(node, symmetry);
                let transformed_canonical = self.canonicalize_packed_node(transformed);
                if transformed_canonical.node.structural != canonical.node.structural {
                    return false;
                }
                let expected = self.materialize_packed_node_key(canonical.node.packed);
                let actual = self.materialize_packed_node_key(transformed_canonical.node.packed);
                let policy = crate::hashlife::GridExtractionPolicy::FullGridIfUnder {
                    max_population: u64::MAX,
                    max_chunks: usize::MAX,
                    max_bounds_span: i64::MAX,
                };
                if crate::normalize::normalize(
                    &self
                        .node_to_grid(expected, 0, 0, policy)
                        .expect("canonical packed test node should materialize"),
                )
                .0
                    != crate::normalize::normalize(
                        &self
                            .node_to_grid(actual, 0, 0, policy)
                            .expect("canonical packed test node should materialize"),
                    )
                    .0
                {
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

    pub(crate) fn verify_packed_transform_root_key_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut stack = vec![root];
        let mut checked = 0;
        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) == 0 {
                continue;
            }
            let packed = self.node_columns.packed_key(node);
            for symmetry in crate::symmetry::D4Symmetry::ALL {
                let transform_id = self.transform_packed_node_key(packed, symmetry);
                let materialized = self.materialize_packed_transform_root(transform_id);
                let expected = self.node_columns.packed_key(materialized);
                let actual = self.materialize_winning_packed_transform_root(transform_id);
                if actual != expected {
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
            canonical_packed_cache_entries: self.canonical_packed_cache.len()
                + self.hot_canonical_packed_cache.len(),
            canonical_oriented_cache_entries: self.canonical_oriented_cache.len()
                + self.hot_canonical_oriented_cache.len(),
            direct_parent_cache_entries: self.direct_parent_canonical_cache.len()
                + self.hot_direct_parent_canonical_cache.len(),
            structural_fast_path_cache_entries: self.structural_fast_path_cache.len(),
            packed_structural_fast_path_cache_entries: self.packed_structural_fast_path_cache.len(),
            oriented_result_cache_entries: self.oriented_result_cache.len(),
            packed_transform_intern_entries: self.packed_transform_intern.len(),
            jump_result_cache_lookups: self.stats.jump_result_cache_lookups,
            jump_result_cache_hits: self.stats.jump_result_cache_hits,
            jump_result_cache_misses: self.stats.jump_result_cache_misses,
            symmetric_jump_result_cache_hits: self.stats.symmetric_jump_result_cache_hits,
            oriented_result_cache_lookups: self.stats.oriented_result_cache_lookups,
            oriented_result_cache_hits: self.stats.oriented_result_cache_hits,
            oriented_result_cache_misses: self.stats.oriented_result_cache_misses,
            root_result_cache_lookups: self.stats.root_result_cache_lookups,
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
            symmetry_gate_canonical_cache_bypasses: self.stats.symmetry_gate_canonical_cache_bypasses,
            structural_fast_path_lookups: self.stats.structural_fast_path_lookups,
            structural_fast_path_hits: self.stats.structural_fast_path_hits,
            structural_fast_path_misses: self.stats.structural_fast_path_misses,
            canonical_result_insert_bypasses: self.stats.canonical_result_insert_bypasses,
            symmetry_aware_result_canonicalization_lookups: self.stats
                .symmetry_aware_result_canonicalization_lookups,
            canonical_node_cache_hits: self.stats.canonical_node_cache_hits,
            canonical_node_cache_misses: self.stats.canonical_node_cache_misses,
            canonical_packed_cache_lookups: self.stats.canonical_packed_cache_lookups,
            canonical_packed_cache_hits: self.stats.canonical_packed_cache_hits,
            canonical_packed_cache_misses: self.stats.canonical_packed_cache_misses,
            canonical_oriented_cache_lookups: self.stats.canonical_oriented_cache_lookups,
            canonical_oriented_cache_hits: self.stats.canonical_oriented_cache_hits,
            canonical_oriented_cache_misses: self.stats.canonical_oriented_cache_misses,
            direct_parent_winner_lookups: self.stats.direct_parent_winner_lookups,
            direct_parent_winner_hits: self.stats.direct_parent_winner_hits,
            direct_parent_winner_misses: self.stats.direct_parent_winner_misses,
            direct_parent_winner_fallbacks: self.stats.direct_parent_winner_fallbacks,
            direct_parent_cached_result_hits: self.stats.direct_parent_cached_result_hits,
            symmetry_scan_fallbacks: self.stats.symmetry_scan_fallbacks,
            canonical_phase2_fallbacks: self.stats.canonical_phase2_fallbacks,
            canonical_result_batch_fallbacks: self.stats.canonical_result_batch_fallbacks,
            canonical_blocked_structural_fallbacks: self.stats
                .canonical_blocked_structural_fallbacks,
            jump_presence_probe_batches: self.stats.jump_presence_probe_batches,
            jump_presence_probe_lanes: self.stats.jump_presence_probe_lanes,
            jump_presence_probe_hits: self.stats.jump_presence_probe_hits,
            jump_batch_unique_queries: self.stats.jump_batch_unique_queries,
            jump_batch_reused_queries: self.stats.jump_batch_reused_queries,
            cached_fingerprint_probes: self.stats.cached_fingerprint_probes,
            recomputed_fingerprint_probes: self.stats.recomputed_fingerprint_probes,
            gc_mark_batches: self.stats.gc_mark_batches,
            gc_remap_batches: self.stats.gc_remap_batches,
            gc_transient_entries_before: self.stats.gc_transient_entries_before,
            gc_canonical_cache_entries_before: self.stats.gc_canonical_cache_entries_before,
            gc_skipped_with_transient_growth: self.stats.gc_skipped_with_transient_growth,
            packed_d4_canonicalization_misses: self.stats.packed_d4_canonicalization_misses,
            packed_inverse_transform_hits: self.stats.packed_inverse_transform_hits,
            packed_recursive_transform_hits: self.stats.packed_recursive_transform_hits,
            packed_recursive_transform_misses: self.stats.packed_recursive_transform_misses,
            packed_overlap_outputs_produced: self.stats.packed_overlap_outputs_produced,
            packed_cache_result_materializations: self.stats.packed_cache_result_materializations,
            session_full_grid_materializations: self.stats.session_full_grid_materializations,
            session_bounded_grid_extractions: self.stats.session_bounded_grid_extractions,
            embedded_result_bounded_extractions: self.stats.embedded_result_bounded_extractions,
            clipped_viewport_extractions: self.stats.clipped_viewport_extractions,
            checkpoint_cell_materializations: self.stats.checkpoint_cell_materializations,
            oracle_confirmation_materializations: self.stats.oracle_confirmation_materializations,
            dense_fallback_invocations: self.stats.dense_fallback_invocations,
            transformed_node_materializations: self.stats.transformed_node_materializations,
        }
    }

    pub(crate) fn diagnostic_summary(&self) -> super::HashLifeDiagnosticSummary {
        let stats = self.runtime_stats();
        let jump_presence_total = stats.jump_presence_probe_hits
            + stats
                .jump_presence_probe_lanes
                .saturating_sub(stats.jump_presence_probe_hits);
        let overlap_total = stats.overlap_cache_hits + stats.overlap_cache_misses;
        let symmetry_gate_total = stats.symmetry_gate_allowed + stats.symmetry_gate_blocked;
        let canonical_cache_total =
            stats.canonical_node_cache_hits + stats.canonical_node_cache_misses;
        let canonical_packed_total =
            stats.canonical_packed_cache_hits + stats.canonical_packed_cache_misses;
        let canonical_oriented_total =
            stats.canonical_oriented_cache_hits + stats.canonical_oriented_cache_misses;
        let direct_parent_total =
            stats.direct_parent_winner_hits + stats.direct_parent_winner_misses;
        let oriented_result_total =
            stats.oriented_result_cache_hits + stats.oriented_result_cache_misses;
        let structural_fast_path_total =
            stats.structural_fast_path_hits + stats.structural_fast_path_misses;
        let total_simd_lanes =
            stats.step0_simd_lanes + stats.phase1_simd_lanes + stats.phase2_simd_lanes;
        let total_provisionals = stats.step0_provisional_records
            + stats.phase1_provisional_records
            + stats.phase2_provisional_records;

        super::HashLifeDiagnosticSummary {
            total_nodes: stats.nodes,
            retained_roots: stats.retained_roots,
            nodes_match_intern: stats.nodes == stats.intern,
            dependency_stalls: stats.dependency_stalls,
            jump_result_hit_rate: stats.jump_result_cache_hits as f64
                / stats.jump_result_cache_lookups.max(1) as f64,
            jump_result_miss_count: stats.jump_result_cache_misses,
            oriented_result_hit_rate: stats.oriented_result_cache_hits as f64
                / oriented_result_total.max(1) as f64,
            root_result_hit_rate: stats.root_result_cache_hits as f64
                / stats.root_result_cache_lookups.max(1) as f64,
            jump_presence_hit_rate: stats.jump_presence_probe_hits as f64
                / jump_presence_total.max(1) as f64,
            overlap_hit_rate: stats.overlap_cache_hits as f64 / overlap_total.max(1) as f64,
            overlap_local_reuse_rate: stats.overlap_local_reuse_lanes as f64
                / stats.overlap_prep_lanes.max(1) as f64,
            symmetry_gate_allow_rate: stats.symmetry_gate_allowed as f64
                / symmetry_gate_total.max(1) as f64,
            symmetry_gate_canonical_cache_bypasses: stats.symmetry_gate_canonical_cache_bypasses,
            structural_fast_path_hit_rate: stats.structural_fast_path_hits as f64
                / structural_fast_path_total.max(1) as f64,
            canonical_cache_hit_rate: stats.canonical_node_cache_hits as f64
                / canonical_cache_total.max(1) as f64,
            canonical_packed_cache_hit_rate: stats.canonical_packed_cache_hits as f64
                / canonical_packed_total.max(1) as f64,
            canonical_oriented_cache_hit_rate: stats.canonical_oriented_cache_hits as f64
                / canonical_oriented_total.max(1) as f64,
            direct_parent_winner_hit_rate: stats.direct_parent_winner_hits as f64
                / direct_parent_total.max(1) as f64,
            direct_parent_cached_result_hits: stats.direct_parent_cached_result_hits,
            direct_parent_winner_fallbacks: stats.direct_parent_winner_fallbacks,
            symmetry_scan_fallbacks: stats.symmetry_scan_fallbacks,
            symmetry_jump_result_hits: stats.symmetric_jump_result_cache_hits,
            simd_lane_coverage: total_simd_lanes as f64 / total_provisionals.max(1) as f64,
            scalar_commit_ratio: stats.scalar_commit_lanes as f64
                / total_provisionals.max(1) as f64,
            probes_per_scheduler_task: stats.scheduler_probe_batches as f64
                / stats.scheduler_tasks.max(1) as f64,
            recursive_overlap_batch_rate: stats.recursive_overlap_batch_batches as f64
                / stats.overlap_prep_batches.max(1) as f64,
            gc_reclaim_ratio: stats
                .nodes_before_mark
                .saturating_sub(stats.nodes_after_mark) as f64
                / stats.nodes_before_mark.max(1) as f64,
            gc_compact_ratio: stats
                .nodes_before_compact
                .saturating_sub(stats.nodes_after_compact) as f64
                / stats.nodes_before_compact.max(1) as f64,
            gc_reason: stats.gc_reason,
            gc_runs: stats.gc_runs,
            gc_skips: stats.gc_skips,
            gc_transient_entries_before: stats.gc_transient_entries_before,
            gc_canonical_cache_entries_before: stats.gc_canonical_cache_entries_before,
            gc_skipped_with_transient_growth: stats.gc_skipped_with_transient_growth,
            canonical_packed_cache_entries: stats.canonical_packed_cache_entries,
            canonical_oriented_cache_entries: stats.canonical_oriented_cache_entries,
            direct_parent_cache_entries: stats.direct_parent_cache_entries,
            structural_fast_path_cache_entries: stats.structural_fast_path_cache_entries,
            packed_structural_fast_path_cache_entries: stats.packed_structural_fast_path_cache_entries,
            oriented_result_cache_entries: stats.oriented_result_cache_entries,
            packed_transform_intern_entries: stats.packed_transform_intern_entries,
            packed_d4_canonicalization_misses: stats.packed_d4_canonicalization_misses,
            packed_inverse_transform_hits: stats.packed_inverse_transform_hits,
            packed_recursive_transform_hits: stats.packed_recursive_transform_hits,
            packed_recursive_transform_misses: stats.packed_recursive_transform_misses,
            packed_overlap_outputs_produced: stats.packed_overlap_outputs_produced,
            packed_cache_result_materializations: stats.packed_cache_result_materializations,
            session_full_grid_materializations: stats.session_full_grid_materializations,
            session_bounded_grid_extractions: stats.session_bounded_grid_extractions,
            embedded_result_bounded_extractions: stats.embedded_result_bounded_extractions,
            clipped_viewport_extractions: stats.clipped_viewport_extractions,
            checkpoint_cell_materializations: stats.checkpoint_cell_materializations,
            oracle_confirmation_materializations: stats.oracle_confirmation_materializations,
            dense_fallback_invocations: stats.dense_fallback_invocations,
            transformed_node_materializations: stats.transformed_node_materializations,
        }
    }

    pub(crate) fn verify_node_fingerprint_invariants(&self) -> bool {
        (0..self.node_count()).all(|i| {
            let id = i as NodeId;
            self.node_columns.fingerprint(id)
                == super::FlatKey::fingerprint(&self.node_columns.packed_key(id))
        })
    }

    pub(crate) fn verify_intern_fingerprint_fast_path_parity(&self) -> bool {
        (0..self.node_count()).all(|i| {
            let id = i as NodeId;
            let key = self.node_columns.packed_key(id);
            let fp = self.node_columns.fingerprint(id);
            self.intern.get(&key) == self.intern.get_with_fingerprint(&key, fp)
        })
    }
}
