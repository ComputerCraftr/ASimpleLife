use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::{HashLifeEngine, HashLifeSession};
use crate::life::{GameOfLife, step_grid};
use crate::normalize::normalize;

use super::hashlife_support::{
    assert_hashlife_matches_stepper, assert_stepper_matches_single_step, grid_from_mask,
    GEN_BLINKER_EVEN, GEN_BLINKER_ODD, GEN_DEEP_DIAGNOSTIC, GEN_GUN_PERIOD_MULTIPLE, GEN_LARGE_PRIME,
    GEN_MEDIUM_PRIME, GEN_POWER_OF_TWO, GEN_POWER_OF_TWO_SMALL, GEN_SEGMENTED_REMAINDER,
    GEN_SINGLE_STEP, GEN_RESUME_DELTA, GEN_SNAPSHOT, SEED_POWER_OF_TWO_SOUP,
    SEED_PRIME_JUMP_SOUP, SEED_SINGLE_STEP_SOUP, SMALL_SOUP_DIM, SMALL_SOUP_FILL,
};

#[test]
fn hashlife_matches_glider_after_single_step() {
    let grid = pattern_by_name("glider").unwrap();
    assert_stepper_matches_single_step(&grid);
}

#[test]
fn hashlife_matches_blinker_after_large_even_and_odd_jumps() {
    let grid = pattern_by_name("blinker").unwrap();
    let mut oracle = HashLifeEngine::default();
    let even = oracle.advance(&grid, GEN_BLINKER_EVEN);
    let odd = oracle.advance(&grid, GEN_BLINKER_ODD);

    assert_eq!(normalize(&even).0, normalize(&grid).0);
    assert_eq!(normalize(&odd).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_session_gosper_gun_core_survives_ten_million_scale_period_multiple() {
    let initial = pattern_by_name("gosper_glider_gun").unwrap();
    let (min_x, min_y, max_x, max_y) = initial
        .bounds()
        .expect("gosper glider gun should be non-empty");
    let target = GEN_GUN_PERIOD_MULTIPLE;
    assert_eq!(target % 30, 0, "test target must land on the gun period");

    let mut session = HashLifeSession::new();
    session.load_grid(&initial);
    session.advance_root(target);

    let sampled_core = session
        .sample_region(min_x, min_y, max_x, max_y)
        .expect("bounded gun core sample should remain available");

    assert_eq!(
        normalize(&sampled_core).0,
        normalize(&initial).0,
        "gosper gun core should still match generation 0 after a deep period-aligned jump"
    );
    assert!(
        session.sample_materializations() <= 1,
        "bounded gun-core regression should avoid full-grid materialization, got {} materializations",
        session.sample_materializations()
    );
}

#[test]
fn hashlife_snapshot_roundtrips_session_state() {
    let initial = pattern_by_name("glider").unwrap();
    let mut session = HashLifeSession::new();
    session.load_grid(&initial);
    session.advance_root(GEN_SNAPSHOT);

    let snapshot = session
        .export_snapshot_string()
        .expect("loaded session should export a snapshot");
    let expected_generation = session.generation();
    let expected_origin = session.origin();
    let expected_population = session.population();
    let expected_bounds = session.bounds();
    let expected_checkpoint = session.signature_checkpoint().cloned();
    let expected_grid = session
        .sample_grid()
        .expect("snapshot source should be materializable");

    let mut restored = HashLifeSession::new();
    restored
        .load_snapshot_string(&snapshot)
        .expect("snapshot should reload");

    assert_eq!(restored.generation(), expected_generation);
    assert_eq!(restored.origin(), expected_origin);
    assert_eq!(restored.population(), expected_population);
    assert_eq!(restored.bounds(), expected_bounds);
    assert_eq!(
        restored.signature_checkpoint().cloned(),
        expected_checkpoint
    );
    assert_eq!(
        normalize(
            &restored
                .sample_grid()
                .expect("restored snapshot should be materializable")
        )
        .0,
        normalize(&expected_grid).0
    );
}

#[test]
fn hashlife_snapshot_persists_deep_run_resume() {
    let initial = pattern_by_name("glider").unwrap();
    let mut uninterrupted = HashLifeSession::new();
    uninterrupted.load_grid(&initial);
    uninterrupted.advance_root(GEN_DEEP_DIAGNOSTIC / 3);
    let snapshot = uninterrupted
        .export_snapshot_string()
        .expect("deep run should export a snapshot");

    uninterrupted.advance_root(GEN_RESUME_DELTA);
    let expected = uninterrupted
        .signature_checkpoint()
        .cloned()
        .expect("continued deep run should have a checkpoint");

    let mut resumed = HashLifeSession::new();
    resumed
        .load_snapshot_string(&snapshot)
        .expect("snapshot should reload");
    resumed.advance_root(GEN_RESUME_DELTA);
    let actual = resumed
        .signature_checkpoint()
        .cloned()
        .expect("resumed deep run should have a checkpoint");

    assert_eq!(actual, expected);
}

#[test]
fn hashlife_snapshot_export_is_deterministic_for_same_session() {
    let initial = pattern_by_name("gosper_glider_gun").unwrap();
    let mut session = HashLifeSession::new();
    session.load_grid(&initial);
    session.advance_root(GEN_RESUME_DELTA);

    let first = session
        .export_snapshot_string()
        .expect("loaded session should export a snapshot");
    let second = session
        .export_snapshot_string()
        .expect("repeated export should succeed");
    assert_eq!(first, second);
}

#[test]
fn hashlife_matches_single_step_on_random_soup() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_SINGLE_STEP_SOUP,
    );
    assert_stepper_matches_single_step(&grid);
}

#[test]
fn hashlife_matches_glider_after_power_of_two_jump() {
    let grid = pattern_by_name("glider").unwrap();
    assert_hashlife_matches_stepper(grid, GEN_POWER_OF_TWO);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_power_of_two_jump() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_POWER_OF_TWO_SOUP,
    );
    assert_hashlife_matches_stepper(grid, GEN_POWER_OF_TWO_SMALL);
}

#[test]
fn hashlife_matches_stepper_on_glider_prime_jump() {
    let grid = pattern_by_name("glider").unwrap();
    assert_hashlife_matches_stepper(grid, GEN_MEDIUM_PRIME);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_prime_jump() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PRIME_JUMP_SOUP,
    );
    assert_hashlife_matches_stepper(grid, GEN_MEDIUM_PRIME);
}

#[test]
fn hashlife_matches_random_soup_after_large_power_of_two_jump() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PRIME_JUMP_SOUP,
    );
    assert_hashlife_matches_stepper(grid, GEN_POWER_OF_TWO);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_large_prime_jump() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PRIME_JUMP_SOUP,
    );
    assert_hashlife_matches_stepper(grid, GEN_LARGE_PRIME);
}

#[test]
fn hashlife_segmented_prime_equivalence_matches_single_advance() {
    let grid = random_soup(
        SMALL_SOUP_DIM,
        SMALL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_PRIME_JUMP_SOUP,
    );
    let mut oracle = HashLifeEngine::default();
    let combined = oracle.advance(&grid, GEN_LARGE_PRIME);
    let intermediate = oracle.advance(&grid, GEN_POWER_OF_TWO);
    let segmented = oracle.advance(&intermediate, GEN_SEGMENTED_REMAINDER);
    assert_eq!(normalize(&combined).0, normalize(&segmented).0);
}

#[test]
fn hashlife_matches_stepper_on_all_4x4_single_steps() {
    let mut oracle = HashLifeEngine::default();
    for mask in 0_u32..(1 << 16) {
        let grid = grid_from_mask(4, 4, mask);
        let advanced = oracle.advance(&grid, GEN_SINGLE_STEP);
        assert_eq!(
            normalize(&advanced).0,
            normalize(&step_grid(&grid)).0,
            "4x4 mask {mask:#06x}"
        );
    }
}

#[test]
fn hashlife_matches_stepper_on_sampled_5x5_small_jumps() {
    const MASKS: [u32; 8] = [
        0x0000000, 0x0000001, 0x001f000, 0x0108421, 0x1555555, 0x0f0f0f0, 0x1249249, 0x1ffffff,
    ];

    for mask in MASKS {
        let grid = grid_from_mask(5, 5, mask);
        for generations in [1_u64, 2, 4, 5] {
            let mut game = GameOfLife::new(grid.clone());
            for _ in 0..generations {
                game.step_with_changes();
            }
            let advanced = HashLifeEngine::default().advance(&grid, generations);
            assert_eq!(
                normalize(&advanced).0,
                normalize(game.grid()).0,
                "5x5 mask {mask:#08x} generations={generations}"
            );
        }
    }
}
