use super::hashlife_support::{
    DEEP_SOUP_FILL, GEN_POWER_OF_TWO, GEN_SINGLE_STEP, LARGE_SOUP_DIM, MEDIUM_SOUP_DIM,
    MEDIUM_SOUP_FILL, SEED_CANONICAL_OVERLAP_PARITY, SEED_CHILD_KEY_PARITY, SEED_JUMP_BATCH_DEDUPE,
    SEED_OVERLAP_BATCH_DEDUPE, SEED_OVERLAP_BATCH_PARITY, SEED_RECURSIVE_SIMD,
    SEED_STEP0_SMALL_BATCH, SMALL_SOUP_FILL, assert_hashlife_probe_ok, assert_normalized_grids_eq,
    build_hashlife_step0_stress_grid,
};
use crate::RequiredExt;
use crate::generators::random_soup;
use crate::hashlife::HashLifeEngine;
use crate::life::{GameOfLife, step_grid};

#[test]
fn hashlife_overlap_batch_matches_scalar_overlap_builder() {
    let grid = random_soup(
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_FILL,
        SEED_OVERLAP_BATCH_PARITY,
    );
    let mut oracle = HashLifeEngine::default();
    let parity_ok = oracle.verify_overlap_batch_parity(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "overlap batch builder should match scalar overlap construction",
        parity_ok,
        stats,
    );
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
    let parity_ok = oracle.verify_canonical_overlap_batch_parity(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "canonical overlap batch should match raw overlap builder",
        parity_ok,
        stats,
    );
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
    assert!(
        unique_parent_shapes <= 1,
        "duplicate phase2 lanes may hit an existing parent cache but must never canonicalize more than one unique parent shape; unique_parent_shapes={unique_parent_shapes}"
    );
    assert!(
        local_reuses >= 3,
        "remaining duplicate phase2 lanes should reuse the staged canonical result"
    );
}

#[test]
fn hashlife_cold_canonical_batch_scans_duplicate_node_once() {
    let grid = random_soup(
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_DIM,
        MEDIUM_SOUP_FILL,
        SEED_JUMP_BATCH_DEDUPE,
    );
    let mut oracle = HashLifeEngine::default();

    assert_eq!(
        oracle.duplicate_cold_canonical_batch_fallbacks(&grid),
        1,
        "eight duplicate cold lanes should perform one D4 fallback scan"
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
    let parity_ok = oracle.verify_canonical_child_key_batch_parity(&grid);
    let stats = oracle.runtime_stats();
    assert_hashlife_probe_ok(
        "canonical child key batch should match scalar child key construction",
        parity_ok,
        stats,
    );
}

#[test]
fn hashlife_step0_kernel_candidates_are_exercised_on_large_single_step() {
    let grid = build_hashlife_step0_stress_grid();

    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, GEN_SINGLE_STEP);
    let stats = oracle.runtime_stats();
    let provisional_records_built = stats.simd.step0_provisional_records
        + stats.simd.phase1_provisional_records
        + stats.simd.phase2_provisional_records;
    let kernel_candidate_lanes = stats.simd.step0_kernel_candidate_lanes
        + stats.simd.phase1_kernel_candidate_lanes
        + stats.simd.phase2_kernel_candidate_lanes;
    let kernel_candidate_batches = stats.simd.step0_kernel_candidate_batches
        + stats.simd.phase1_kernel_candidate_batches
        + stats.simd.phase2_kernel_candidate_batches;

    assert_normalized_grids_eq(
        "single-step SIMD stress grid should match scalar stepper",
        &advanced,
        &step_grid(&grid),
    );
    assert!(provisional_records_built > 0, "{stats:?}");
    assert!(kernel_candidate_lanes > 0, "{stats:?}");
    assert!(kernel_candidate_batches > 0, "{stats:?}");
    assert!(stats.scheduler.simd_disabled_fast_exits > 0, "{stats:?}");
    assert!(stats.simd.overlap_prep_batches > 0, "{stats:?}");
    assert!(stats.scheduler.cache_probe_batches > 0, "{stats:?}");
}

#[test]
fn hashlife_centered_overlap_full_lanes_use_nine_staged_batches() {
    let grids = std::array::from_fn(|lane| {
        let lane = u64::try_from(lane).or_invariant("SIMD test lane exceeds u64");
        random_soup(
            MEDIUM_SOUP_DIM,
            MEDIUM_SOUP_DIM,
            MEDIUM_SOUP_FILL,
            SEED_STEP0_SMALL_BATCH.wrapping_add(lane),
        )
    });
    let mut oracle = HashLifeEngine::default();
    let (matches_scalar, distinct_inputs, probe_batches) =
        oracle.centered_overlap_full_batch_work(&grids);

    assert!(
        distinct_inputs,
        "full SIMD batch fixture must use distinct lanes"
    );
    assert!(
        matches_scalar,
        "all centered SIMD lanes must match scalar joins and populations"
    );
    assert_eq!(
        probe_batches, 9,
        "a full SIMD batch should use one staged cache probe per overlap column"
    );
}

#[test]
fn hashlife_step0_small_batches_reach_kernel_dispatch_without_claiming_vector_work() {
    let grid = random_soup(40, 40, SMALL_SOUP_FILL, SEED_STEP0_SMALL_BATCH);
    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, GEN_SINGLE_STEP);
    let stats = oracle.runtime_stats();
    let provisional_records_built = stats.simd.step0_provisional_records
        + stats.simd.phase1_provisional_records
        + stats.simd.phase2_provisional_records;

    assert_normalized_grids_eq(
        "small candidate single-step SIMD path should match scalar stepper",
        &advanced,
        &step_grid(&grid),
    );
    assert!(provisional_records_built >= 1, "{stats:?}");
    assert!(stats.simd.step0_kernel_candidate_batches >= 1, "{stats:?}");
    assert!(stats.simd.step0_kernel_candidate_lanes >= 1, "{stats:?}");
}

#[test]
fn hashlife_recursive_phase_kernel_candidates_are_exercised_on_deep_jump() {
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

    assert_normalized_grids_eq(
        "deep recursive SIMD path should match scalar stepper",
        &advanced,
        game.grid(),
    );
    assert!(stats.simd.phase1_provisional_records > 0, "{stats:?}");
    assert!(stats.simd.phase2_provisional_records > 0, "{stats:?}");
    assert!(stats.simd.phase1_kernel_candidate_lanes > 0, "{stats:?}");
    assert!(stats.simd.phase2_kernel_candidate_lanes > 0, "{stats:?}");
    assert!(stats.simd.phase1_kernel_candidate_batches > 0, "{stats:?}");
    assert!(stats.simd.phase2_kernel_candidate_batches > 0, "{stats:?}");
    assert!(stats.simd.scalar_commit_lanes > 0, "{stats:?}");
}
