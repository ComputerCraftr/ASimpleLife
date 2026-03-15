use super::*;
use crate::RequiredExt;
use crate::hashlife::GridExtractionPolicy;

#[cfg(test)]
impl HashLifeEngine {
    pub(crate) fn admitted_input_symmetry_survives_output_gate_growth(
        &mut self,
        grid: &BitGrid,
        mirrored: &BitGrid,
    ) -> bool {
        let expected = crate::life::step_grid(grid);
        let expected_mirrored = crate::life::step_grid(mirrored);
        let first = self.embed_for_jump(grid, 0);
        let first_result = self.advance_pow2(first.root, 0);
        let first_grid = self.extract_embedded_result(first, first_result);
        let hits_before_mirror = self.stats.result_cache.symmetric_jump_result_cache_hits;
        let second = self.embed_for_jump(mirrored, 0);
        let second_result = self.advance_pow2(second.root, 0);
        let second_grid = self.extract_embedded_result(second, second_result);

        first_grid == expected
            && second_grid == expected_mirrored
            && self.stats.result_cache.symmetric_jump_result_cache_hits > hits_before_mirror
    }

    pub(crate) fn empty_advance_fast_path_stats(
        &mut self,
        level: u32,
        step_exp: u32,
    ) -> (u32, u128, usize, usize) {
        let root = self.empty(level);
        let scheduler_tasks = self.stats.scheduler.scheduler_tasks;
        let canonical_lookups = self.stats.canonical_cache.canonical_packed_cache_lookups;
        let result = self.advance_pow2(root, step_exp);
        (
            self.node_columns.level(result),
            self.node_columns.population(result),
            self.stats.scheduler.scheduler_tasks - scheduler_tasks,
            self.stats.canonical_cache.canonical_packed_cache_lookups - canonical_lookups,
        )
    }

    pub(crate) fn full_node_extraction_matches_bounded(&mut self, grid: &BitGrid) -> bool {
        let (root, origin_x, origin_y) = self.embed_grid_state(grid);
        let direct = self.node_to_grid_all(root, origin_x, origin_y);
        let Some((min_x, min_y, max_x, max_y)) = self.node_bounds(root, origin_x, origin_y) else {
            return direct.is_empty();
        };
        let bounded = self
            .node_to_grid(
                root,
                origin_x,
                origin_y,
                GridExtractionPolicy::BoundedRegion {
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                },
            )
            .or_invariant("bounded extraction probe should succeed");
        direct == bounded
    }

    pub(crate) fn duplicate_jump_batch_query_stats(&mut self, grid: &BitGrid) -> (usize, usize) {
        let (root, _, _) = self.embed_grid_state(grid);
        self.insert_jump_result((root, 0), root);
        let before = (
            self.stats.canonical_fallback.jump_batch_unique_queries,
            self.stats.canonical_fallback.jump_batch_reused_queries,
        );
        let results = self.jump_result_batch([root, root, root, root], 0);
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
        assert_eq!(results[2], results[3]);
        (
            self.stats.canonical_fallback.jump_batch_unique_queries - before.0,
            self.stats.canonical_fallback.jump_batch_reused_queries - before.1,
        )
    }

    pub(crate) fn verify_packed_jump_cache_roundtrip(
        &mut self,
        grid: &BitGrid,
        step_exp: u32,
    ) -> bool {
        let embedded = self.embed_for_jump(grid, step_exp);
        let root = embedded.root;
        let before_hits = self.stats.result_cache.jump_result_cache_hits;
        let result = self.advance_pow2(root, step_exp);
        let cached = self.cached_jump_result((root, step_exp));
        cached == Some(result) && self.stats.result_cache.jump_result_cache_hits > before_hits
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
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        self.insert_canonical_jump_result(canonical_key, result);
        let first_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_cache.canonical_packed_cache_hits,
            self.stats.canonical_cache.canonical_packed_cache_misses,
        );
        self.insert_canonical_jump_result(canonical_key, result);
        let second_delta = (
            self.stats.canonical_cache.canonical_packed_cache_hits - before.0,
            self.stats.canonical_cache.canonical_packed_cache_misses - before.1,
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
            self.stats.canonical_cache.canonical_node_cache_hits,
            self.stats.canonical_cache.canonical_node_cache_misses,
        );
        self.insert_jump_result((root, 1), result);
        let first_delta = (
            self.stats.canonical_cache.canonical_node_cache_hits - before.0,
            self.stats.canonical_cache.canonical_node_cache_misses - before.1,
        );

        let before = (
            self.stats.canonical_cache.canonical_node_cache_hits,
            self.stats.canonical_cache.canonical_node_cache_misses,
        );
        self.insert_jump_result((root, 1), result);
        let second_delta = (
            self.stats.canonical_cache.canonical_node_cache_hits - before.0,
            self.stats.canonical_cache.canonical_node_cache_misses - before.1,
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
            self.stats.transform.packed_cache_result_materializations,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions,
        );
        let first =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let first_delta = (
            self.stats.transform.packed_cache_result_materializations - before.0,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions
                - before.1,
        );

        let before = (
            self.stats.transform.packed_cache_result_materializations,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions,
        );
        let second =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let second_delta = (
            self.stats.transform.packed_cache_result_materializations - before.0,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions
                - before.1,
        );

        assert_eq!(first, second);
        (first_delta, second_delta)
    }

    pub(crate) fn oriented_result_cache_is_retained_by_skip_gc(
        &mut self,
        grid: &BitGrid,
    ) -> ((usize, usize), usize, (usize, usize)) {
        let (root, _, _) = self.embed_grid_state(grid);
        let packed = self.node_columns.packed_key(root);

        let before = (
            self.stats.transform.packed_cache_result_materializations,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions,
        );
        let first =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let populate_delta = (
            self.stats.transform.packed_cache_result_materializations - before.0,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions
                - before.1,
        );

        let before = (
            self.stats.transform.packed_cache_result_materializations,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions,
        );
        let warm =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let warm_delta = (
            self.stats.transform.packed_cache_result_materializations - before.0,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions
                - before.1,
        );

        self.maybe_garbage_collect_with_budget("skip", u128::MAX);
        let protected_entries = self.result_caches.oriented.len();

        let before = (
            self.stats.transform.packed_cache_result_materializations,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions,
        );
        let second =
            self.materialize_oriented_packed_result(packed, Symmetry::Identity, Symmetry::Rotate90);
        let retained_delta = (
            self.stats.transform.packed_cache_result_materializations - before.0,
            self.stats
                .canonical_fallback
                .oriented_transform_root_reconstructions
                - before.1,
        );

        assert_eq!(first, warm);
        assert_eq!(first, second);
        assert_eq!(warm_delta.0, 0);
        assert_eq!(warm_delta.1, 0);

        (populate_delta, protected_entries, retained_delta)
    }
}
