use super::hashlife_support::{
    GEN_DEEP_DIAGNOSTIC, GEN_LARGE_PRIME, GEN_MIRROR_REUSE, GEN_POWER_OF_TWO,
    GEN_STRUCTURED_PROMOTION_REGRESSION, LARGE_SOUP_DIM, SEED_CACHE_BASELINE, SEED_CACHE_VARIANT,
    SEED_CANONICAL_PACKED_CACHE, SEED_CANONICAL_RESULT_INSERT, SEED_CANONICAL_SYMMETRY_PARITY,
    SEED_DIRECT_PARENT_WINNER, SEED_FINGERPRINT_FAST_PATH, SEED_GATE_BLOCKED_PROBE,
    SEED_GC_REBUILD, SEED_IDENTITY_CANONICALIZATION, SEED_JUMP_RESULT_INSERT,
    SEED_ORIENTED_RESULT_CACHE, SEED_PACKED_JUMP_ROUNDTRIP, SEED_PACKED_ROOT_PARITY,
    SEED_PACKED_TRANSFORM_PARITY, SMALL_SOUP_DIM, SMALL_SOUP_FILL, SYMMETRY_GATE_WIDE_LEVEL,
    SYMMETRY_GATE_WIDE_POPULATION, assert_grids_eq, assert_hashlife_probe_ok,
    assert_hashlife_runtime_stat, build_hashlife_structured_symmetry_grid, mirror_grid_x,
};
use crate::RequiredExt;
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeEngine;
use crate::life::GameOfLife;

#[test]
fn hashlife_empty_advance_bypasses_canonicalization_and_scheduler() {
    let mut oracle = HashLifeEngine::default();
    let (result_level, result_population, scheduler_tasks, canonical_lookups) =
        oracle.empty_advance_fast_path_stats(12, 8);

    assert_eq!(result_level, 11);
    assert_eq!(result_population, 0);
    assert_eq!(
        scheduler_tasks, 0,
        "empty advance should not schedule tasks"
    );
    assert_eq!(
        canonical_lookups, 0,
        "empty advance should not canonicalize its root"
    );
}

#[test]
fn hashlife_mark_only_gc_is_not_repeated_without_node_growth() {
    let mut engine = HashLifeEngine::default();
    assert!(
        engine.repeated_active_gc_without_growth_is_skipped(),
        "mark-only GC must record the current arena size so an unchanged graph is skipped: {:?}",
        engine.runtime_stats()
    );
}

#[test]
fn hashlife_full_node_extraction_matches_bounded_extraction() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CACHE_BASELINE,
    )
    .translated(-37, 19);
    let mut oracle = HashLifeEngine::default();

    assert!(
        oracle.full_node_extraction_matches_bounded(&grid),
        "single-pass full-node extraction must preserve bounded extraction contents"
    );
}

#[test]
fn hashlife_full_node_extraction_handles_sparse_large_roots() {
    let grid = crate::bitgrid::BitGrid::from_cells(&[
        (-536_870_912, -536_870_912),
        (536_870_912, 536_870_912),
    ]);
    let mut oracle = HashLifeEngine::default();

    assert!(
        oracle.full_node_extraction_matches_bounded(&grid),
        "population-pruned extraction must preserve a sparse billion-cell-span root"
    );
}

#[test]
fn hashlife_advance_preserves_absolute_stable_state() {
    let grid = pattern_by_name("block").or_invariant("block fixture should exist");
    let mut oracle = HashLifeEngine::default();

    let advanced = oracle.advance(&grid, 1);
    assert_eq!(advanced, grid, "block should remain stable after one step");
}

#[test]
fn hashlife_gc_preserves_reusable_retained_state_across_repeated_runs() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CACHE_BASELINE,
    );
    let mut oracle = HashLifeEngine::default();
    let mut first_reference = GameOfLife::new(grid.clone());
    for _ in 0..GEN_LARGE_PRIME {
        first_reference.step_with_changes();
    }
    let first_output = oracle.advance(&grid, GEN_LARGE_PRIME);
    let first = oracle.runtime_stats();
    assert!(first.engine.nodes > 2);
    assert!(first.engine.retained_roots >= 1);
    assert_eq!(first.engine.nodes, first.engine.intern);
    assert!(first.gc.jump_cache_before_clear >= first.engine.jump_cache);
    assert!(
        first.gc.nodes_before_mark >= first.gc.nodes_after_mark,
        "{first:?}"
    );
    assert!(
        first.gc.nodes_before_compact >= first.gc.nodes_after_compact,
        "{first:?}"
    );
    assert_grids_eq(
        "initial GC-backed advance should match the independent scalar engine",
        &first_output,
        first_reference.grid(),
    );

    let second_output = oracle.advance(&grid, GEN_LARGE_PRIME);
    let second = oracle.runtime_stats();
    assert!(second.engine.retained_roots >= 1);
    assert!(second.gc.jump_cache_before_clear > 0);
    assert_eq!(second.engine.nodes, second.engine.intern);
    assert!(
        second.gc.nodes_before_mark >= second.gc.nodes_after_mark,
        "{second:?}"
    );
    assert!(
        second.gc.nodes_before_compact >= second.gc.nodes_after_compact,
        "{second:?}"
    );
    assert_eq!(
        second_output, first_output,
        "repeated deep run on identical grid should preserve output\nfirst_stats={first:?}\nsecond_stats={second:?}"
    );

    let third_output = oracle.advance(&grid, GEN_LARGE_PRIME);
    let third = oracle.runtime_stats();
    assert!(third.engine.retained_roots >= 1);
    assert_eq!(third.engine.nodes, third.engine.intern);
    assert!(third.gc.jump_cache_before_clear >= third.engine.jump_cache);
    assert!(
        third.gc.nodes_before_mark >= third.gc.nodes_after_mark,
        "{third:?}"
    );
    assert!(
        third.gc.nodes_before_compact >= third.gc.nodes_after_compact,
        "{third:?}"
    );
    assert_eq!(
        third_output, first_output,
        "third identical deep run should preserve output\nfirst_stats={first:?}\nthird_stats={third:?}"
    );

    let other = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CACHE_VARIANT,
    );
    let mut other_reference = GameOfLife::new(other.clone());
    for _ in 0..GEN_LARGE_PRIME {
        other_reference.step_with_changes();
    }
    let fourth_output = oracle.advance(&other, GEN_LARGE_PRIME);
    let fourth = oracle.runtime_stats();
    assert!(fourth.engine.retained_roots >= 1);
    assert_eq!(fourth.engine.nodes, fourth.engine.intern);
    assert!(fourth.gc.jump_cache_before_clear >= fourth.engine.jump_cache);
    assert!(
        fourth.gc.nodes_before_mark >= fourth.gc.nodes_after_mark,
        "{fourth:?}"
    );
    assert!(
        fourth.gc.nodes_before_compact >= fourth.gc.nodes_after_compact,
        "{fourth:?}"
    );
    assert!(fourth.engine.nodes > 2, "{fourth:?}");

    let mut fresh = HashLifeEngine::default();
    let fresh_output = fresh.advance(&other, GEN_LARGE_PRIME);
    assert_eq!(
        fourth_output,
        fresh_output,
        "reused engine and fresh engine should agree on variant grid after GC reuse path\nreused_stats={fourth:?}\nfresh_stats={:?}",
        fresh.runtime_stats()
    );
    assert_grids_eq(
        "GC-reused variant advance should match the independent scalar engine",
        &fourth_output,
        other_reference.grid(),
    );
}

#[test]
fn hashlife_node_store_fingerprints_match_recomputed_keys() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_FINGERPRINT_FAST_PATH,
    );
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, GEN_POWER_OF_TWO);
    let fingerprint_ok = oracle.verify_node_fingerprint_invariants();
    let intern_ok = oracle.verify_intern_fingerprint_fast_path_parity();
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "node fingerprint invariants should match packed node keys",
        fingerprint_ok,
        stats,
    );
    assert_hashlife_probe_ok(
        "intern fingerprint fast-path should match recomputed lookup path",
        intern_ok,
        stats,
    );
    assert!(
        stats.canonical_fallback.cached_fingerprint_probes > 0,
        "{stats:?}"
    );
}

#[test]
fn hashlife_gc_rebuild_preserves_fingerprint_invariants() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_GC_REBUILD,
    );
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, GEN_LARGE_PRIME);
    oracle.advance(&grid, GEN_LARGE_PRIME);
    let fingerprint_ok = oracle.verify_node_fingerprint_invariants();
    let intern_ok = oracle.verify_intern_fingerprint_fast_path_parity();
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "GC rebuild should preserve node fingerprint invariants",
        fingerprint_ok,
        stats,
    );
    assert_hashlife_probe_ok(
        "GC rebuild should preserve intern fingerprint fast-path parity",
        intern_ok,
        stats,
    );
}

#[test]
fn hashlife_packed_jump_cache_roundtrip_matches_materialized_result() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PACKED_JUMP_ROUNDTRIP,
    );
    let mut oracle = HashLifeEngine::default();
    let roundtrip_ok = oracle.verify_packed_jump_cache_roundtrip(&grid, 2);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "packed jump cache roundtrip should match materialized result for step_exp=2",
        roundtrip_ok,
        stats,
    );
    assert!(stats.result_cache.jump_result_cache_hits > 0, "{stats:?}");
    assert!(
        stats.transform.packed_cache_result_materializations > 0,
        "{stats:?}"
    );
}

#[test]
fn hashlife_admitted_input_symmetry_is_preserved_when_result_exceeds_gate() {
    let grid = crate::bitgrid::BitGrid::from_cells(&[(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]);
    let mirrored = mirror_grid_x(&grid);
    assert_eq!(grid.population(), 5);
    assert!(
        crate::life::step_grid(&grid).population() > 5,
        "adversarial fixture must grow beyond the configured population gate"
    );

    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(u32::MAX, 5);
    let parity_ok = engine.admitted_input_symmetry_survives_output_gate_growth(&grid, &mirrored);
    assert_hashlife_probe_ok(
        "an input admitted to symmetry canonicalization must keep that decision when its result grows beyond the gate",
        parity_ok,
        engine.runtime_stats(),
    );
}

#[test]
fn hashlife_repeated_canonical_result_insertion_reuses_packed_canonical_cache() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_RESULT_INSERT,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_canonical_result_insertion_cache_stats(&grid);

    assert!(
        first_delta.0 + first_delta.1 > 0,
        "first canonical result insertion should touch packed canonical caching"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical canonical result insertion should not miss packed canonical cache"
    );
    assert!(
        second_delta.0 > 0,
        "second identical canonical result insertion should hit packed canonical cache"
    );
}

#[test]
fn hashlife_repeated_jump_result_insertion_reuses_canonical_node_cache() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_JUMP_RESULT_INSERT,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_jump_result_insertion_cache_stats(&grid);

    assert!(
        first_delta.0 + first_delta.1 > 0,
        "first jump-result insertion should touch canonical-node caching"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical jump-result insertion should not miss canonical-node cache"
    );
    assert!(
        second_delta.0 > 0,
        "second identical jump-result insertion should hit canonical-node cache"
    );
}

#[test]
fn hashlife_oriented_result_cache_reuses_materialized_output() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_ORIENTED_RESULT_CACHE,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.duplicate_oriented_result_cache_stats(&grid);
    assert!(
        first_delta.0 > 0,
        "first oriented result batch should materialize once"
    );
    assert!(
        first_delta.1 > 0,
        "first oriented result batch should reconstruct the oriented packed root once"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical oriented batch should not reconstruct the oriented packed root"
    );
    assert_eq!(
        second_delta.0, 0,
        "second identical oriented batch should reuse the materialized packed result"
    );
}

#[test]
fn hashlife_canonical_packed_cache_reuses_packed_canonicalization() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_PACKED_CACHE,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_canonical_packed_cache_stats(&grid);
    assert!(
        first_delta.1 > 0,
        "first packed canonicalization should miss canonical packed cache"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical packed canonicalization should not miss canonical packed cache"
    );
    assert!(
        second_delta.0 > 0,
        "second identical packed canonicalization should hit canonical packed cache"
    );
}

#[test]
fn hashlife_hot_canonical_cache_survives_skip_gc() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_PACKED_CACHE,
    );
    let mut oracle = HashLifeEngine::default();
    let (warm_delta, protected_entries, retained_delta) =
        oracle.canonical_hot_cache_survives_skip_gc(&grid);

    assert!(
        warm_delta.1 > 0,
        "first packed canonicalization should populate the canonical cache"
    );
    assert!(
        protected_entries.0 > 0 || protected_entries.1 > 0,
        "skip GC should preserve at least one protected canonical entry"
    );
    assert_eq!(
        retained_delta.1, 0,
        "protected canonical cache should avoid a post-skip-GC miss"
    );
    assert!(
        retained_delta.0 > 0,
        "protected canonical cache should hit after skip GC"
    );
}

#[test]
fn hashlife_oriented_result_cache_is_retained_by_skip_gc() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_ORIENTED_RESULT_CACHE,
    );
    let mut oracle = HashLifeEngine::default();
    let (populate_delta, protected_entries, retained_delta) =
        oracle.oriented_result_cache_is_retained_by_skip_gc(&grid);

    assert!(
        populate_delta.0 > 0,
        "first oriented result should materialize once"
    );
    assert!(
        populate_delta.1 > 0,
        "first oriented result should reconstruct the oriented packed root once"
    );
    assert!(
        protected_entries > 0,
        "skip GC should retain the useful oriented result cache"
    );
    assert_eq!(
        retained_delta.1, 0,
        "a retained oriented result should avoid transform-root reconstruction"
    );
    assert_eq!(
        retained_delta.0, 0,
        "a retained oriented result should avoid packed-result materialization"
    );
}

#[test]
fn hashlife_direct_parent_cache_survives_skip_gc() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_DIRECT_PARENT_WINNER,
    );
    let mut oracle = HashLifeEngine::default();
    let (populate_delta, protected_entries, retained_delta) =
        oracle.direct_parent_cache_survives_skip_gc(&grid);

    assert!(
        populate_delta.1 <= 1,
        "cold canonicalization should reconstruct at most once; an already canonical input needs none: {populate_delta:?}"
    );
    assert!(
        populate_delta.2 > 0,
        "first direct-parent canonicalization should pay winner fallback once"
    );
    assert!(
        protected_entries > 0,
        "skip GC should preserve at least one direct-parent cache entry"
    );
    assert!(
        retained_delta.0 > 0,
        "protected direct-parent cache should hit after skip GC"
    );
    assert_eq!(
        retained_delta.1, 0,
        "protected direct-parent cache should avoid post-skip-GC canonical root reconstruction"
    );
    assert_eq!(
        retained_delta.2, 0,
        "protected direct-parent cache should avoid post-skip-GC winner fallback"
    );
}

#[test]
fn hashlife_canonical_oriented_cache_reuses_rotated_canonicalization() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_oriented_canonical_cache_stats(&grid);
    assert!(
        first_delta.1 > 0,
        "first rotated packed canonicalization should miss oriented canonical cache"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical rotated packed canonicalization should not miss oriented canonical cache"
    );
    assert!(
        second_delta.0 > 0,
        "second identical rotated packed canonicalization should hit oriented canonical cache"
    );
}

#[test]
fn hashlife_nonidentity_jump_result_insertion_reuses_oriented_canonical_cache() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) =
        oracle.repeated_nonidentity_jump_result_insertion_oriented_cache_stats(&grid);
    assert!(
        first_delta.1 > 0,
        "first non-identity jump-result insertion should miss oriented canonical cache"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical non-identity jump-result insertion should not miss oriented canonical cache"
    );
    assert!(
        second_delta.0 > 0,
        "second identical non-identity jump-result insertion should hit oriented canonical cache"
    );
}

#[test]
fn hashlife_identity_packed_canonicalization_avoids_oriented_cache() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_IDENTITY_CANONICALIZATION,
    );
    let mut oracle = HashLifeEngine::default();
    let avoids_oriented_cache =
        oracle.identity_packed_canonicalization_avoids_oriented_cache(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "identity packed canonicalization should avoid oriented cache lookups",
        avoids_oriented_cache,
        stats,
    );
}

#[test]
fn hashlife_gate_blocked_probes_reuse_structural_fast_path_cache() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_GATE_BLOCKED_PROBE,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_gate_blocked_probe_stats(&grid);
    assert!(
        first_delta.1 > 0,
        "first blocked structural probe should miss structural fast-path cache"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical blocked structural probe should not miss structural fast-path cache"
    );
    assert!(
        second_delta.0 > 0,
        "second identical blocked structural probe should hit structural fast-path cache"
    );
}

#[test]
fn hashlife_direct_parent_winner_cache_reuses_parent_shape_after_warmup() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_DIRECT_PARENT_WINNER,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_direct_parent_winner_stats(&grid);
    assert!(
        first_delta.0 + first_delta.1 > 0,
        "first parent canonicalization should either fill direct-winner cache or pay one fallback"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical parent canonicalization should not fall back to full symmetry scan"
    );
    assert!(
        second_delta.0 > 0,
        "second identical parent canonicalization should hit direct parent winner cache"
    );
}

#[test]
fn hashlife_direct_parent_cached_result_hit_avoids_reconstruction_after_warmup() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_DIRECT_PARENT_WINNER,
    );
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.repeated_direct_parent_cached_result_stats(&grid);

    assert!(
        first_delta.1 > 0 || first_delta.2 > 0,
        "first direct parent canonicalization should reconstruct once or record one fallback"
    );
    assert_eq!(
        second_delta.1, 0,
        "second direct parent canonicalization should not reconstruct transform roots"
    );
    assert_eq!(
        second_delta.2, 0,
        "second direct parent canonicalization should not record a fallback"
    );
    assert!(
        second_delta.0 > 0,
        "second direct parent canonicalization should use the cached final result"
    );
}

#[test]
fn hashlife_direct_parent_cache_key_respects_symmetry_mode() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_DIRECT_PARENT_WINNER,
    );
    let mut oracle = HashLifeEngine::default();
    assert!(
        oracle.direct_parent_cache_respects_symmetry_mode(&grid),
        "direct-parent cached canonical results must not be reused across different base symmetries"
    );
}

#[test]
fn hashlife_packed_recursive_transform_matches_node_transform() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PACKED_TRANSFORM_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    let transform_ok = oracle.verify_packed_transform_parity(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "packed recursive transform should match node transform parity",
        transform_ok,
        stats,
    );
}

#[test]
fn hashlife_packed_transform_root_key_matches_materialized_root() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PACKED_ROOT_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    let root_key_ok = oracle.verify_packed_transform_root_key_parity(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "packed transform root key should match materialized root key parity",
        root_key_ok,
        stats,
    );
}

#[test]
fn hashlife_packed_canonicalization_matches_all_symmetry_variants() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_SYMMETRY_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    let symmetry_ok = oracle.verify_packed_canonicalization_symmetry_parity(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "packed canonicalization should preserve canonical identity, forward transform, inverse reconstruction, and deterministic lowest-symmetry ties across all variants",
        symmetry_ok,
        stats,
    );
}

#[test]
fn hashlife_symmetric_canonicalization_uses_deterministic_transform_ties() {
    let grid = pattern_by_name("block").or_invariant("block fixture must exist");
    let mut oracle = HashLifeEngine::default();
    let symmetry_ok = oracle.verify_packed_canonicalization_symmetry_parity(&grid)
        && oracle.verify_packed_canonicalization_tie_breaking(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "symmetric nodes must choose the lowest D4 transform and its inverse must reconstruct every oriented input",
        symmetry_ok,
        stats,
    );
}

#[test]
fn hashlife_d4_semantic_winner_ignores_opposing_intern_order() {
    let mut engine = HashLifeEngine::default();
    let (ids_oppose_structure, nonidentity_winner, exact_minimum, inverse_roundtrip) =
        engine.d4_semantic_winner_ignores_intern_order();
    assert!(
        ids_oppose_structure,
        "fixture must intern the structurally larger child before the smaller child"
    );
    assert!(
        nonidentity_winner,
        "fixture must require a nonidentity structural winner rather than insertion-order identity"
    );
    assert!(
        exact_minimum,
        "prefix selection plus recursive fallback must match the exact structural comparator"
    );
    assert!(
        inverse_roundtrip,
        "selected D4 transform and inverse must reconstruct the original node"
    );

    let stats = engine.runtime_stats();
    let native_d4_available = stats.simd.kernel.native_d4_prefix_compare_lanes != 0;
    assert_eq!(
        stats.simd.kernel.native_d4_candidate_lanes,
        if native_d4_available { 8 } else { 0 },
        "only an ISA candidate-construction kernel may claim native D4 candidate lanes: {stats:?}"
    );
    assert!(
        stats.simd.kernel.native_d4_exact_winner_lanes
            <= stats.simd.kernel.native_d4_prefix_compare_lanes,
        "exact native winners cannot exceed native D4 prefix work: {stats:?}"
    );
}

#[test]
fn hashlife_d4_prefix_metadata_avoids_recursive_leaf_rewalks() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_SYMMETRY_PARITY,
    );
    let mut engine = HashLifeEngine::default();
    let (level, cold_attempts, cold_visits, cold_cost_bypasses, warm_attempts, warm_visits) =
        engine.d4_semantic_prefix_cost_probe(&grid);
    eprintln!(
        "d4_prefix_probe level={level} attempts={cold_attempts} leaf_visits={cold_visits} cost_bypasses={cold_cost_bypasses} warm_attempts={warm_attempts} warm_leaf_visits={warm_visits}"
    );
    assert_eq!(
        cold_attempts + cold_cost_bypasses,
        1,
        "cold D4 scan must choose exactly one prefix or exact-recursive path: level={level} stats={:?}",
        engine.runtime_stats()
    );
    assert_eq!(
        cold_attempts, 1,
        "every D4 winner uses exact prefix ordering"
    );
    assert_eq!(
        cold_visits, 0,
        "canonical prefixes must be composed at intern time, not by a D4 leaf walk"
    );
    assert_eq!(
        warm_attempts, 1,
        "warm D4 scans must retain the same exact prefix-ordering contract"
    );
    assert_eq!(
        warm_visits, 0,
        "warm exact comparisons must not repeat the 8x128-leaf prefix walk"
    );
}

#[test]
fn hashlife_warmed_symmetric_workload_reduces_scheduler_work_against_fresh_run() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mirrored = mirror_grid_x(&grid);
    let generations = GEN_MIRROR_REUSE;
    let mut engine = HashLifeEngine::default();

    engine.advance(&grid, generations);
    let second = engine.advance(&mirrored, generations);
    let second_stats = engine.runtime_stats();

    let mut game = GameOfLife::new(mirrored.clone());
    for _ in 0..generations {
        game.step_with_changes();
    }
    let mut fresh_engine = HashLifeEngine::default();
    let fresh_mirrored = fresh_engine.advance(&mirrored, generations);
    let fresh_stats = fresh_engine.runtime_stats();

    assert_grids_eq(
        "mirrored second run should match scalar stepper result",
        &second,
        game.grid(),
    );
    assert_grids_eq(
        "fresh mirrored run should match scalar stepper result",
        &fresh_mirrored,
        game.grid(),
    );
    assert!(
        second_stats.scheduler.scheduler_tasks < fresh_stats.scheduler.scheduler_tasks,
        "warmed symmetry caches should reduce unresolved scheduler work, second={second_stats:?} fresh={fresh_stats:?}"
    );
}

#[test]
fn hashlife_summary_tracks_symmetry_reuse_invariants() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mirrored = mirror_grid_x(&grid);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, GEN_MIRROR_REUSE);
    oracle.advance(&mirrored, GEN_MIRROR_REUSE);
    let summary = oracle.diagnostic_summary();

    assert_hashlife_runtime_stat(
        "symmetry reuse summary should keep nodes and intern in sync",
        oracle.runtime_stats(),
        summary.overview.nodes_match_intern,
    );
    assert_eq!(summary.overview.dependency_stalls, 0, "{summary:?}");
    assert!(
        summary.fallbacks.symmetry_jump_result_hits > 0,
        "{summary:?}"
    );
    assert!(
        summary.cache_rates.jump_result_hit_rate >= 0.0
            && summary.cache_rates.jump_result_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.cache_rates.root_result_hit_rate >= 0.0
            && summary.cache_rates.root_result_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.cache_rates.jump_presence_hit_rate >= 0.0
            && summary.cache_rates.jump_presence_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.cache_rates.symmetry_gate_allow_rate >= 0.0
            && summary.cache_rates.symmetry_gate_allow_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.cache_rates.canonical_node_cache_hit_rate >= 0.0
            && summary.cache_rates.canonical_node_cache_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.cache_rates.direct_parent_winner_hit_rate >= 0.0
            && summary.cache_rates.direct_parent_winner_hit_rate <= 1.0,
        "{summary:?}"
    );
}

#[test]
fn hashlife_summary_reports_valid_gc_accounting_after_reclaim() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let mut oracle = HashLifeEngine::default();
    for _ in 0..3 {
        oracle.advance(&grid, GEN_DEEP_DIAGNOSTIC);
    }
    let stats = oracle.runtime_stats();
    let summary = oracle.diagnostic_summary();

    assert_hashlife_runtime_stat(
        "GC summary should keep nodes and intern in sync",
        stats,
        summary.overview.nodes_match_intern,
    );
    assert_eq!(summary.overview.dependency_stalls, 0, "{summary:?}");
    assert!(summary.gc.gc_runs > 0, "{summary:?}");
    assert_eq!(
        summary.overview.total_nodes, stats.engine.nodes,
        "{summary:?} vs {stats:?}"
    );
    assert_eq!(
        summary.overview.retained_roots, stats.engine.retained_roots,
        "{summary:?} vs {stats:?}"
    );
    assert_eq!(
        summary.gc.gc_runs, stats.gc.gc_runs,
        "{summary:?} vs {stats:?}"
    );
    assert!(
        stats.gc.nodes_before_mark >= stats.gc.nodes_after_mark,
        "{stats:?}"
    );
    assert!(
        stats.gc.nodes_before_compact >= stats.gc.nodes_after_compact,
        "{stats:?}"
    );
    assert!(
        (0.0..=1.0).contains(&summary.gc.gc_reclaim_ratio),
        "{summary:?}"
    );
    assert!(
        (0.0..=1.0).contains(&summary.gc.gc_compact_ratio),
        "{summary:?}"
    );

    let expected_reclaim_ratio = stats
        .gc
        .nodes_before_mark
        .saturating_sub(stats.gc.nodes_after_mark) as f64
        / stats.gc.nodes_before_mark.max(1) as f64;
    let expected_compact_ratio = stats
        .gc
        .nodes_before_compact
        .saturating_sub(stats.gc.nodes_after_compact) as f64
        / stats.gc.nodes_before_compact.max(1) as f64;
    assert!(
        (summary.gc.gc_reclaim_ratio - expected_reclaim_ratio).abs() < f64::EPSILON,
        "{summary:?} vs {stats:?}"
    );
    assert!(
        (summary.gc.gc_compact_ratio - expected_compact_ratio).abs() < f64::EPSILON,
        "{summary:?} vs {stats:?}"
    );
}

#[test]
fn hashlife_wider_structured_workload_triggers_gc_before_runaway_cache_growth() {
    let grid = build_hashlife_structured_symmetry_grid();
    let mut oracle = HashLifeEngine::with_symmetry_gate_for_tests(
        SYMMETRY_GATE_WIDE_LEVEL,
        SYMMETRY_GATE_WIDE_POPULATION,
    );
    oracle.advance(&grid, GEN_STRUCTURED_PROMOTION_REGRESSION);
    let summary = oracle.diagnostic_summary();

    let stats = oracle.runtime_stats();
    assert_hashlife_runtime_stat(
        "structured GC workload should keep nodes and intern in sync",
        stats,
        summary.overview.nodes_match_intern,
    );
    assert_eq!(summary.overview.dependency_stalls, 0, "{summary:?}");
    assert!(summary.gc.gc_runs > 0, "{summary:?}");
    assert!(
        summary.gc.gc_transient_pressure_entries_before > 0,
        "{summary:?}"
    );
    assert!(
        summary.gc.gc_canonical_cache_entries_before > 0,
        "{summary:?}"
    );
}
#[test]
fn hashlife_node_symmetry_metadata_stays_compact() {
    let bytes = HashLifeEngine::node_symmetry_metadata_size_for_tests();
    assert!(
        bytes <= 64,
        "per-node symmetry metadata must contain only compact canonical references, got {bytes} bytes"
    );
}

#[test]
fn hashlife_blocked_canonicalization_does_not_build_unused_orientations() {
    let grid = random_soup(64, 64, 20, 0xB10C_ED00);
    let mut oracle = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);

    assert_eq!(
        oracle.blocked_canonicalization_shape_growth(&grid),
        0,
        "gate-blocked nodes must not intern seven unused D4 orientations"
    );
}
