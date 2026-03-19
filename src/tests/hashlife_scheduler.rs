use crate::generators::random_soup;
use crate::hashlife::HashLifeEngine;
use crate::life::{GameOfLife, step_grid};
use crate::normalize::normalize;

use super::hashlife_support::{
    build_hashlife_step0_stress_grid, GEN_POWER_OF_TWO, GEN_SINGLE_STEP, LARGE_SOUP_DIM,
    MEDIUM_SOUP_DIM, MEDIUM_SOUP_FILL, SEED_CANONICAL_OVERLAP_PARITY, SEED_CHILD_KEY_PARITY,
    SEED_JUMP_BATCH_DEDUPE, SEED_OVERLAP_BATCH_DEDUPE, SEED_OVERLAP_BATCH_PARITY,
    SEED_RECURSIVE_SIMD, SEED_STEP0_SMALL_BATCH, SMALL_SOUP_FILL, DEEP_SOUP_FILL,
};

#[test]
fn hashlife_overlap_batch_matches_scalar_overlap_builder() {
    let grid = random_soup(
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_FILL,
        SEED_OVERLAP_BATCH_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_overlap_batch_parity(&grid));
}

#[test]
fn hashlife_canonical_overlap_batch_matches_raw_overlap_builder() {
    let grid = random_soup(
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_FILL,
        SEED_CANONICAL_OVERLAP_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_canonical_overlap_batch_parity(&grid));
}

#[test]
fn hashlife_overlap_batch_dedupes_duplicate_miss_parents_locally() {
    let grid = random_soup(
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_FILL,
        SEED_OVERLAP_BATCH_DEDUPE,
    );
    let mut oracle = HashLifeEngine::default();
    let (misses, local_reuse) = oracle.duplicate_overlap_batch_dedupe_stats(&grid);

    assert_eq!(
        misses, 1,
        "duplicate parent overlap miss path should build once"
    );
    assert_eq!(
        local_reuse, 1,
        "second duplicate parent should reuse staged overlap work"
    );
}

#[test]
fn hashlife_jump_result_batch_reuses_duplicate_queries() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_JUMP_BATCH_DEDUPE,
    );
    let mut oracle = HashLifeEngine::default();
    let (unique, reused) = oracle.duplicate_jump_batch_query_stats(&grid);

    assert_eq!(
        unique, 1,
        "grouped jump-result batch should probe one unique key"
    );
    assert_eq!(
        reused, 3,
        "remaining duplicate lanes should reuse the grouped result"
    );
}

#[test]
fn hashlife_phase2_canonicalization_dedupes_duplicate_packed_results() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_JUMP_BATCH_DEDUPE,
    );
    let mut oracle = HashLifeEngine::default();
    let (unique_inputs, unique_parent_shapes, local_reuses) =
        oracle.duplicate_phase2_canonicalization_stats(&grid);

    assert_eq!(
        unique_inputs, 1,
        "duplicate phase2 lanes should canonicalize one unique packed input"
    );
    assert_eq!(
        unique_parent_shapes, 1,
        "duplicate phase2 lanes should canonicalize one unique parent shape"
    );
    assert!(
        local_reuses >= 3,
        "remaining duplicate phase2 lanes should reuse the staged canonical result"
    );
}

#[test]
fn hashlife_canonical_child_key_batch_matches_scalar_keys() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CHILD_KEY_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_canonical_child_key_batch_parity(&grid));
}

#[test]
fn hashlife_step0_simd_batches_are_exercised_on_large_single_step() {
    let grid = build_hashlife_step0_stress_grid();

    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, GEN_SINGLE_STEP);
    let stats = oracle.runtime_stats();
    let provisional_records_built = stats.step0_provisional_records
        + stats.phase1_provisional_records
        + stats.phase2_provisional_records;
    let simd_candidate_lanes = provisional_records_built;
    let simd_batches =
        stats.step0_simd_batches + stats.phase1_simd_batches + stats.phase2_simd_batches;

    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
    assert!(provisional_records_built > 0, "{stats:?}");
    assert!(simd_candidate_lanes > 0, "{stats:?}");
    assert!(simd_batches > 0, "{stats:?}");
    assert!(stats.simd_disabled_fast_exits > 0, "{stats:?}");
    assert!(stats.overlap_prep_batches > 0, "{stats:?}");
    assert!(stats.cache_probe_batches > 0, "{stats:?}");
}

#[test]
fn hashlife_step0_small_candidate_batches_still_run_through_simd() {
    let grid = random_soup(40, 40, SMALL_SOUP_FILL, SEED_STEP0_SMALL_BATCH);
    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, GEN_SINGLE_STEP);
    let stats = oracle.runtime_stats();
    let provisional_records_built = stats.step0_provisional_records
        + stats.phase1_provisional_records
        + stats.phase2_provisional_records;

    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
    assert!(provisional_records_built >= 1, "{stats:?}");
    assert!(stats.step0_simd_batches >= 1, "{stats:?}");
    assert!(stats.step0_simd_lanes >= 1, "{stats:?}");
}

#[test]
fn hashlife_recursive_phase_simd_batches_are_exercised_on_deep_jump() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        DEEP_SOUP_FILL,
        SEED_RECURSIVE_SIMD,
    );
    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, GEN_POWER_OF_TWO);
    let stats = oracle.runtime_stats();

    let mut game = GameOfLife::new(grid.clone());
    for _ in 0..GEN_POWER_OF_TWO {
        game.step_with_changes();
    }

    assert_eq!(normalize(&advanced).0, normalize(game.grid()).0);
    assert!(stats.phase1_provisional_records > 0, "{stats:?}");
    assert!(stats.phase2_provisional_records > 0, "{stats:?}");
    assert!(stats.phase1_simd_lanes > 0, "{stats:?}");
    assert!(stats.phase2_simd_lanes > 0, "{stats:?}");
    assert!(stats.phase1_simd_batches > 0, "{stats:?}");
    assert!(stats.phase2_simd_batches > 0, "{stats:?}");
    assert!(stats.scalar_commit_lanes > 0, "{stats:?}");
}
