use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeEngine;
use crate::life::GameOfLife;
use crate::normalize::normalize;

use super::hashlife_support::{
    build_hashlife_structured_symmetry_grid, GEN_DEEP_DIAGNOSTIC, GEN_LARGE_PRIME,
    GEN_MIRROR_REUSE, GEN_POWER_OF_TWO, GEN_STRUCTURED_PROMOTION_REGRESSION,
    SEED_CACHE_BASELINE, SEED_CACHE_VARIANT, SEED_CANONICAL_PACKED_CACHE,
    SEED_CANONICAL_RESULT_INSERT, SEED_CANONICAL_SYMMETRY_PARITY, SEED_DIRECT_PARENT_WINNER,
    SEED_FINGERPRINT_FAST_PATH, SEED_GATE_BLOCKED_PROBE, SEED_GC_REBUILD,
    SEED_IDENTITY_CANONICALIZATION, SEED_JUMP_RESULT_INSERT, SEED_ORIENTED_RESULT_CACHE,
    SEED_PACKED_JUMP_ROUNDTRIP, SEED_PACKED_ROOT_PARITY, SEED_PACKED_TRANSFORM_PARITY,
    SMALL_SOUP_DIM, SMALL_SOUP_FILL, LARGE_SOUP_DIM, SYMMETRY_GATE_WIDE_LEVEL,
    SYMMETRY_GATE_WIDE_POPULATION, jump_presence_probe_misses, mirror_grid_x,
};

#[test]
fn hashlife_gc_keeps_state_bounded_and_reusable() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CACHE_BASELINE,
    );
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, GEN_LARGE_PRIME);
    let first = oracle.runtime_stats();
    assert!(first.nodes > 2);
    assert!(first.retained_roots >= 1);
    assert_eq!(first.nodes, first.intern);
    assert!(first.jump_cache_before_clear >= first.jump_cache);

    oracle.advance(&grid, GEN_LARGE_PRIME);
    let second = oracle.runtime_stats();
    assert!(second.retained_roots >= 1);
    assert!(second.jump_cache_before_clear > 0);
    assert_eq!(second.nodes, second.intern);
    assert!(
        second.nodes <= first.nodes.saturating_mul(2),
        "repeated identical deep runs should keep retained canonical state bounded, first={first:?} second={second:?}"
    );

    oracle.advance(&grid, GEN_LARGE_PRIME);
    let third = oracle.runtime_stats();
    assert!(third.retained_roots >= 1);
    assert_eq!(third.nodes, third.intern);
    assert!(third.jump_cache_before_clear >= third.jump_cache);
    assert!(
        third.nodes <= second.nodes,
        "repeated identical deep runs should stabilize retained canonical state, second={second:?} third={third:?}"
    );

    let other = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CACHE_VARIANT,
    );
    oracle.advance(&other, GEN_LARGE_PRIME);
    let fourth = oracle.runtime_stats();
    assert!(fourth.retained_roots >= 1);
    assert_eq!(fourth.nodes, fourth.intern);
    assert!(fourth.jump_cache_before_clear >= fourth.jump_cache);
    assert!(
        fourth.nodes > 2,
        "advancing a different grid should still leave a valid reusable retained state, fourth={fourth:?}"
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
    assert!(oracle.verify_node_fingerprint_invariants());
    assert!(oracle.verify_intern_fingerprint_fast_path_parity());
    let stats = oracle.runtime_stats();
    assert!(stats.cached_fingerprint_probes > 0, "{stats:?}");
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
    assert!(oracle.verify_node_fingerprint_invariants());
    assert!(oracle.verify_intern_fingerprint_fast_path_parity());
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
    assert!(oracle.verify_packed_jump_cache_roundtrip(&grid, 2));
    let stats = oracle.runtime_stats();
    assert!(stats.jump_result_cache_hits > 0, "{stats:?}");
    assert!(stats.packed_cache_result_materializations > 0, "{stats:?}");
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
fn hashlife_oriented_result_cache_survives_skip_gc() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_ORIENTED_RESULT_CACHE,
    );
    let mut oracle = HashLifeEngine::default();
    let (populate_delta, protected_entries, retained_delta) =
        oracle.oriented_result_cache_survives_skip_gc(&grid);

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
        "skip GC should preserve at least one oriented result cache entry"
    );
    assert_eq!(
        retained_delta.1, 0,
        "protected oriented result cache should avoid post-skip-GC transform-root reconstruction"
    );
    assert_eq!(
        retained_delta.0, 0,
        "protected oriented result cache should avoid post-skip-GC packed-result rematerialization"
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
        populate_delta.1 > 0,
        "first direct-parent canonicalization should reconstruct the canonical root once"
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
    let grid = pattern_by_name("glider").unwrap();
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
    let grid = pattern_by_name("glider").unwrap();
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
    assert!(oracle.identity_packed_canonicalization_avoids_oriented_cache(&grid));
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
    assert!(oracle.verify_packed_transform_parity(&grid));
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
    assert!(oracle.verify_packed_transform_root_key_parity(&grid));
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
    assert!(oracle.verify_packed_canonicalization_symmetry_parity(&grid));
}

#[test]
fn hashlife_symmetric_workloads_reuse_specific_cached_work_against_fresh_mirror_run() {
    let grid = pattern_by_name("glider").unwrap();
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

    assert_eq!(normalize(&second).0, normalize(game.grid()).0);
    assert_eq!(normalize(&fresh_mirrored).0, normalize(game.grid()).0);
    assert!(
        jump_presence_probe_misses(&second_stats) < jump_presence_probe_misses(&fresh_stats)
            || second_stats.packed_recursive_transform_misses
                < fresh_stats.packed_recursive_transform_misses,
        "expected mirrored workload to reuse at least one tracked cache boundary, second={second_stats:?} fresh={fresh_stats:?}"
    );
}

#[test]
fn hashlife_summary_tracks_symmetry_reuse_invariants() {
    let grid = pattern_by_name("glider").unwrap();
    let mirrored = mirror_grid_x(&grid);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, GEN_MIRROR_REUSE);
    oracle.advance(&mirrored, GEN_MIRROR_REUSE);
    let summary = oracle.diagnostic_summary();

    assert!(summary.nodes_match_intern, "{summary:?}");
    assert_eq!(summary.dependency_stalls, 0, "{summary:?}");
    assert!(summary.symmetry_jump_result_hits > 0, "{summary:?}");
    assert!(
        summary.jump_result_hit_rate >= 0.0 && summary.jump_result_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.root_result_hit_rate >= 0.0 && summary.root_result_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.jump_presence_hit_rate >= 0.0 && summary.jump_presence_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.symmetry_gate_allow_rate >= 0.0 && summary.symmetry_gate_allow_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.canonical_node_cache_hit_rate >= 0.0
            && summary.canonical_node_cache_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.direct_parent_winner_hit_rate >= 0.0
            && summary.direct_parent_winner_hit_rate <= 1.0,
        "{summary:?}"
    );
}

#[test]
fn hashlife_summary_tracks_gc_reclaim_invariants() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let mut oracle = HashLifeEngine::default();
    for _ in 0..3 {
        oracle.advance(&grid, GEN_DEEP_DIAGNOSTIC);
    }
    let summary = oracle.diagnostic_summary();

    assert!(summary.nodes_match_intern, "{summary:?}");
    assert_eq!(summary.dependency_stalls, 0, "{summary:?}");
    assert!(summary.gc_runs > 0, "{summary:?}");
    assert!(summary.gc_reclaim_ratio > 0.9, "{summary:?}");
    assert!(summary.gc_compact_ratio > 0.9, "{summary:?}");
}

#[test]
#[ignore = "promotion regression for wider symmetry gate"]
fn hashlife_wider_structured_workload_triggers_gc_before_runaway_cache_growth() {
    let grid = build_hashlife_structured_symmetry_grid();
    let mut oracle = HashLifeEngine::with_symmetry_gate_for_tests(
        SYMMETRY_GATE_WIDE_LEVEL,
        SYMMETRY_GATE_WIDE_POPULATION,
    );
    oracle.advance(&grid, GEN_STRUCTURED_PROMOTION_REGRESSION);
    let summary = oracle.diagnostic_summary();

    assert!(summary.nodes_match_intern, "{summary:?}");
    assert_eq!(summary.dependency_stalls, 0, "{summary:?}");
    assert!(summary.gc_runs > 0, "{summary:?}");
    assert!(summary.gc_transient_pressure_entries_before > 0, "{summary:?}");
    assert!(summary.gc_canonical_cache_entries_before > 0, "{summary:?}");
}
