use super::*;
use crate::RequiredExt;

#[cfg(test)]
impl HashLifeEngine {
    pub(crate) fn d4_semantic_prefix_cost_probe(
        &mut self,
        grid: &BitGrid,
    ) -> (u32, usize, usize, usize, usize, usize) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);
        let before = self.stats.transform;
        self.scan_canonical_transform_winner(packed, crate::symmetry::D4Symmetry::Identity, false);
        let cold_attempts =
            self.stats.transform.d4_semantic_prefix_attempts - before.d4_semantic_prefix_attempts;
        let cold_visits = self.stats.transform.d4_semantic_prefix_leaf_visits
            - before.d4_semantic_prefix_leaf_visits;
        let cold_cost_bypasses = self.stats.transform.d4_semantic_prefix_cost_bypasses
            - before.d4_semantic_prefix_cost_bypasses;

        for symmetry in crate::symmetry::D4Symmetry::ALL {
            self.transform_packed_node_key(packed, symmetry);
        }
        let before_warm = self.stats.transform;
        self.scan_canonical_transform_winner(packed, crate::symmetry::D4Symmetry::Identity, false);
        (
            packed.level,
            cold_attempts,
            cold_visits,
            cold_cost_bypasses,
            self.stats.transform.d4_semantic_prefix_attempts
                - before_warm.d4_semantic_prefix_attempts,
            self.stats.transform.d4_semantic_prefix_leaf_visits
                - before_warm.d4_semantic_prefix_leaf_visits,
        )
    }

    pub(crate) fn d4_semantic_winner_ignores_intern_order(&mut self) -> (bool, bool, bool, bool) {
        let full = self.join(
            self.live_leaf,
            self.live_leaf,
            self.live_leaf,
            self.live_leaf,
        );
        let sparse = self.join(
            self.dead_leaf,
            self.dead_leaf,
            self.dead_leaf,
            self.live_leaf,
        );
        let ids_oppose_structure = full.raw() < sparse.raw();
        let root = self.join(full, sparse, full, full);
        let packed = self.node_columns.packed_key(root);
        let (winner, _, _) = self.scan_canonical_transform_winner(
            packed,
            crate::symmetry::D4Symmetry::Identity,
            false,
        );
        let winner_id = self.transform_packed_node_key(packed, winner);
        let exact_minimum = crate::symmetry::D4Symmetry::ALL
            .into_iter()
            .all(|candidate| {
                let candidate_id = self.transform_packed_node_key(packed, candidate);
                let ordering = self.compare_packed_transform_ids(candidate_id, winner_id);
                ordering != std::cmp::Ordering::Less
                    && (ordering != std::cmp::Ordering::Equal || candidate >= winner)
            });
        let materialized = self.materialize_packed_transform_root(winner_id);
        let inverse_roundtrip = self.transform_node(materialized, winner.inverse()) == root;
        (
            ids_oppose_structure,
            winner != crate::symmetry::D4Symmetry::Identity,
            exact_minimum,
            inverse_roundtrip,
        )
    }

    pub(crate) fn blocked_canonicalization_shape_growth(&mut self, grid: &BitGrid) -> usize {
        let (root, _, _) = self.embed_grid_state(grid);
        let before = self.canonical_caches.shape_intern.len();
        self.canonicalize_blocked_jump_node_for_tests(root);
        self.canonical_caches.shape_intern.len() - before
    }

    pub(crate) fn duplicate_cold_canonical_batch_fallbacks(&mut self, grid: &BitGrid) -> usize {
        let (root, _, _) = self.embed_grid_state(grid);
        let before = self.stats.canonical_fallback.symmetry_scan_fallbacks;
        let canonical = self.canonicalize_packed_nodes_batch(&[root; 8], 8);
        assert!(
            canonical
                .windows(2)
                .all(|pair| pair[0].node == pair[1].node),
            "duplicate cold lanes must receive the same canonical identity"
        );
        self.stats.canonical_fallback.symmetry_scan_fallbacks - before
    }

    pub(crate) fn repeated_canonical_packed_cache_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        let first = self.canonicalize_packed_key_for_snapshot(packed);
        let first_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        let second = self.canonicalize_packed_key_for_snapshot(packed);
        let second_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
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
            self.stats.canonical_cache.canonical_oriented_cache_hits,
            self.stats.canonical_cache.canonical_oriented_cache_misses,
        );
        let first = self.canonicalize_packed_under_symmetry(packed, Symmetry::Rotate90);
        let first_delta = (
            self.stats.canonical_cache.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_cache.canonical_oriented_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_cache.canonical_oriented_cache_hits,
            self.stats.canonical_cache.canonical_oriented_cache_misses,
        );
        let second = self.canonicalize_packed_under_symmetry(packed, Symmetry::Rotate90);
        let second_delta = (
            self.stats.canonical_cache.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_cache.canonical_oriented_cache_misses - before.1,
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
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        let first = self.canonicalize_packed_key_for_snapshot(packed);
        let populate_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        let _warm = self.canonicalize_packed_key_for_snapshot(packed);
        let warm_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
        );

        self.maybe_garbage_collect_with_budget("skip", u128::MAX);

        let protected_entries = (
            self.canonical_caches.hot_packed.len(),
            self.canonical_caches.hot_direct_parent.len(),
        );

        let before = (
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        let second = self.canonicalize_packed_key_for_snapshot(packed);
        let retained_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
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
        self.canonical_caches.oriented.clear();

        let before = (
            self.stats.canonical_cache.canonical_oriented_cache_hits,
            self.stats.canonical_cache.canonical_oriented_cache_misses,
        );
        self.insert_jump_result((source_root, 1), result);
        let first_delta = (
            self.stats.canonical_cache.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_cache.canonical_oriented_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_cache.canonical_oriented_cache_hits,
            self.stats.canonical_cache.canonical_oriented_cache_misses,
        );
        self.insert_jump_result((source_root, 1), result);
        let second_delta = (
            self.stats.canonical_cache.canonical_oriented_cache_hits - before.0,
            self.stats.canonical_cache.canonical_oriented_cache_misses - before.1,
        );

        (first_delta, second_delta)
    }

    pub(crate) fn identity_packed_canonicalization_avoids_oriented_cache(
        &mut self,
        grid: &BitGrid,
    ) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);
        let before = self.stats.canonical_cache.canonical_oriented_cache_lookups;
        let _ = self.canonicalize_packed_key_for_snapshot(packed);
        self.stats.canonical_cache.canonical_oriented_cache_lookups == before
    }

    pub(crate) fn repeated_gate_blocked_probe_stats(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);

        let before = (
            self.stats.result_cache.structural_fast_path_hits,
            self.stats.result_cache.structural_fast_path_misses,
        );
        let first = self.canonicalize_blocked_jump_node_for_tests(root);
        let first_delta = (
            self.stats.result_cache.structural_fast_path_hits - before.0,
            self.stats.result_cache.structural_fast_path_misses - before.1,
        );

        let before = (
            self.stats.result_cache.structural_fast_path_hits,
            self.stats.result_cache.structural_fast_path_misses,
        );
        let second = self.canonicalize_blocked_jump_node_for_tests(root);
        let second_delta = (
            self.stats.result_cache.structural_fast_path_hits - before.0,
            self.stats.result_cache.structural_fast_path_misses - before.1,
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
            self.stats.canonical_cache.direct_parent_winner_hits,
            self.stats.canonical_fallback.symmetry_scan_fallbacks,
        );
        let first = self
            .direct_parent_winner_for_tests(packed, Symmetry::Identity)
            .or_invariant("direct parent winner should exist for non-leaf packed node");
        let first_delta = (
            self.stats.canonical_cache.direct_parent_winner_hits - before.0,
            self.stats.canonical_fallback.symmetry_scan_fallbacks - before.1,
        );

        let before = (
            self.stats.canonical_cache.direct_parent_winner_hits,
            self.stats.canonical_fallback.symmetry_scan_fallbacks,
        );
        let second = self
            .direct_parent_winner_for_tests(packed, Symmetry::Identity)
            .or_invariant("direct parent winner should be cached after warmup");
        let second_delta = (
            self.stats.canonical_cache.direct_parent_winner_hits - before.0,
            self.stats.canonical_fallback.symmetry_scan_fallbacks - before.1,
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
            self.stats.canonical_cache.direct_parent_cached_result_hits,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions,
            self.stats.canonical_cache.direct_parent_winner_fallbacks,
        );
        let first = self.canonicalize_packed_direct_for_tests(packed, Symmetry::Identity);
        let first_delta = (
            self.stats.canonical_cache.direct_parent_cached_result_hits - before.0,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions
                - before.1,
            self.stats.canonical_cache.direct_parent_winner_fallbacks - before.2,
        );

        let before = (
            self.stats.canonical_cache.direct_parent_cached_result_hits,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions,
            self.stats.canonical_cache.direct_parent_winner_fallbacks,
        );
        let second = self.canonicalize_packed_direct_for_tests(packed, Symmetry::Identity);
        let second_delta = (
            self.stats.canonical_cache.direct_parent_cached_result_hits - before.0,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions
                - before.1,
            self.stats.canonical_cache.direct_parent_winner_fallbacks - before.2,
        );

        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn direct_parent_cache_survives_skip_gc(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize, usize), usize, (usize, usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.canonical_cache.direct_parent_cached_result_hits,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions,
            self.stats.canonical_cache.direct_parent_winner_fallbacks,
        );
        let first = self
            .canonicalize_packed_direct(packed, Symmetry::Identity, false)
            .node;
        let populate_delta = (
            self.stats.canonical_cache.direct_parent_cached_result_hits - before.0,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions
                - before.1,
            self.stats.canonical_cache.direct_parent_winner_fallbacks - before.2,
        );

        let before = (
            self.stats.canonical_cache.direct_parent_cached_result_hits,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions,
            self.stats.canonical_cache.direct_parent_winner_fallbacks,
        );
        let warm = self
            .canonicalize_packed_direct(packed, Symmetry::Identity, false)
            .node;
        let warm_delta = (
            self.stats.canonical_cache.direct_parent_cached_result_hits - before.0,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions
                - before.1,
            self.stats.canonical_cache.direct_parent_winner_fallbacks - before.2,
        );

        self.maybe_garbage_collect_with_budget("skip", u128::MAX);
        let protected_entries = self.canonical_caches.direct_parent.len()
            + self.canonical_caches.hot_direct_parent.len();

        let before = (
            self.stats.canonical_cache.direct_parent_cached_result_hits,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions,
            self.stats.canonical_cache.direct_parent_winner_fallbacks,
        );
        let second = self
            .canonicalize_packed_direct(packed, Symmetry::Identity, false)
            .node;
        let retained_delta = (
            self.stats.canonical_cache.direct_parent_cached_result_hits - before.0,
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions
                - before.1,
            self.stats.canonical_cache.direct_parent_winner_fallbacks - before.2,
        );

        assert_eq!(first, warm);
        assert_eq!(first, second);
        assert!(warm_delta.0 > 0);
        assert_eq!(warm_delta.1, 0);
        assert_eq!(warm_delta.2, 0);

        (populate_delta, protected_entries, retained_delta)
    }

    pub(crate) fn direct_parent_cache_respects_symmetry_mode(&mut self, grid: &BitGrid) -> bool {
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

        fresh_identity != fresh_rotated && identity == fresh_identity && rotated == fresh_rotated
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
        let mut lanes = [Phase2CommitLane {
            key: canonical_key,
            fallback: result,
            result,
            unique_input_index: 0,
            packed_input,
            canonical_entry: PackedSymmetryKey {
                packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
                symmetry: Symmetry::Identity,
            },
        }; 4];

        let before = (
            self.stats.canonical_fallback.canonical_result_unique_inputs,
            self.stats
                .canonical_fallback
                .canonical_result_unique_parent_shapes,
            self.stats
                .canonical_fallback
                .canonical_result_batch_local_reuses,
        );
        self.canonicalize_phase2_commit_lanes(&mut lanes);
        assert!(
            lanes
                .windows(2)
                .all(|pair| pair[0].canonical_entry == pair[1].canonical_entry)
        );
        (
            self.stats.canonical_fallback.canonical_result_unique_inputs - before.0,
            self.stats
                .canonical_fallback
                .canonical_result_unique_parent_shapes
                - before.1,
            self.stats
                .canonical_fallback
                .canonical_result_batch_local_reuses
                - before.2,
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
            let canonical_node = self.materialize_packed_node_key(canonical.node.packed);
            for symmetry in crate::symmetry::D4Symmetry::ALL {
                let transformed = self.transform_node(node, symmetry);
                let transformed_canonical = self.canonicalize_packed_node(transformed);
                if transformed_canonical.node.structural != canonical.node.structural {
                    return false;
                }
                let transformed_to_canonical =
                    self.transform_node(transformed, transformed_canonical.node.symmetry);
                if transformed_to_canonical != canonical_node {
                    return false;
                }
                let reconstructed = self.transform_node(
                    transformed_to_canonical,
                    transformed_canonical.node.symmetry.inverse(),
                );
                if reconstructed != transformed {
                    return false;
                }

                let packed = self.node_columns.packed_key(transformed);
                let (winner, _, _) = self.scan_canonical_transform_winner(
                    packed,
                    crate::symmetry::D4Symmetry::Identity,
                    false,
                );
                let winner_id = self.transform_packed_node_key(packed, winner);
                if winner != transformed_canonical.node.symmetry {
                    return false;
                }
                for candidate in crate::symmetry::D4Symmetry::ALL {
                    let candidate_id = self.transform_packed_node_key(packed, candidate);
                    let ordering = self.compare_packed_transform_ids(candidate_id, winner_id);
                    if ordering == std::cmp::Ordering::Less
                        || (ordering == std::cmp::Ordering::Equal && candidate < winner)
                    {
                        return false;
                    }
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

    pub(crate) fn verify_packed_canonicalization_tie_breaking(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) == 0 {
                continue;
            }
            let packed = self.node_columns.packed_key(node);
            let (winner, _, _) = self.scan_canonical_transform_winner(
                packed,
                crate::symmetry::D4Symmetry::Identity,
                false,
            );
            let winner_id = self.transform_packed_node_key(packed, winner);
            let mut equal_winners = 0;
            for candidate in crate::symmetry::D4Symmetry::ALL {
                let candidate_id = self.transform_packed_node_key(packed, candidate);
                if self.compare_packed_transform_ids(candidate_id, winner_id)
                    == std::cmp::Ordering::Equal
                {
                    equal_winners += 1;
                    if candidate < winner {
                        return false;
                    }
                }
            }
            if equal_winners > 1 {
                return true;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.extend([nw, ne, sw, se]);
        }
        false
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
}
