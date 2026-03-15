use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Coord};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::{HashLifeAdvanceError, HashLifeEngine, HashLifeLimits, HashLifeSession};
use crate::life::{GameOfLife, step_grid};

use super::hashlife_support::{
    GEN_BLINKER_EVEN, GEN_BLINKER_ODD, GEN_DEEP_DIAGNOSTIC, GEN_GUN_PERIOD_MULTIPLE,
    GEN_LARGE_PRIME, GEN_MEDIUM_PRIME, GEN_POWER_OF_TWO, GEN_POWER_OF_TWO_SMALL, GEN_RESUME_DELTA,
    GEN_SEGMENTED_REMAINDER, GEN_SINGLE_STEP, GEN_SNAPSHOT, SEED_POWER_OF_TWO_SOUP,
    SEED_PRIME_JUMP_SOUP, SEED_SINGLE_STEP_SOUP, SMALL_SOUP_DIM, SMALL_SOUP_FILL, assert_grids_eq,
    assert_hashlife_matches_stepper, assert_stepper_matches_single_step, grid_from_mask,
};

fn assert_checkpoint_state_eq(
    context: &str,
    actual: &crate::hashlife::HashLifeStateCheckpoint,
    expected: &crate::hashlife::HashLifeStateCheckpoint,
) {
    assert_eq!(
        actual.generation, expected.generation,
        "{context}: generation"
    );
    assert_eq!(actual.origin, expected.origin, "{context}: origin");
    assert_eq!(
        actual.population, expected.population,
        "{context}: population"
    );
    assert_eq!(actual.root_span, expected.root_span, "{context}: root span");
    assert!(
        !actual.identity.same_epoch(expected.identity),
        "{context}: independently loaded sessions must not share root-id identity"
    );
}

#[test]
fn hashlife_multi_step_advance_stops_cleanly_after_extinction() {
    let grid = crate::bitgrid::BitGrid::from_cells(&[(0, 0)]);
    let advanced = crate::hashlife::HashLifeEngine::default().advance(&grid, 3);

    assert!(
        advanced.is_empty(),
        "a pattern extinct after the first segment must remain empty for the rest of the advance"
    );
}

#[test]
fn hashlife_embedding_preserves_cells_at_both_coordinate_extremes() {
    for cell in [(Coord::MIN, Coord::MIN), (Coord::MAX, Coord::MAX)] {
        let expected = BitGrid::from_cells(&[cell]);
        let mut session = HashLifeSession::new();
        let loaded = session.try_load_grid(&expected);
        assert!(
            loaded.is_ok(),
            "extreme coordinate {cell:?} failed to embed: {loaded:?}"
        );
        let extracted = session.sample_grid();
        assert!(
            extracted.is_ok(),
            "extreme coordinate {cell:?} failed to extract: {extracted:?}"
        );
        let actual = extracted.or_invariant("successful extraction must contain a grid");
        assert_eq!(
            actual, expected,
            "extreme coordinate changed during roundtrip: {cell:?}"
        );
    }
}

#[test]
fn hashlife_origin_shift_failure_preserves_authoritative_geometry() {
    let grid = BitGrid::from_cells(&[(Coord::MAX, Coord::MAX)]);
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("maximum-coordinate fixture must embed");
    let origin_before = session.origin();
    let checkpoint_before = session
        .try_signature_checkpoint()
        .or_invariant("checkpoint geometry must be valid")
        .copied();

    let shifted = session.shift_origin(1, 1);

    assert!(
        shifted.is_err(),
        "shifting a maximum-endpoint root should return a typed geometry error"
    );
    assert_eq!(session.origin(), origin_before);
    assert_eq!(
        session
            .try_signature_checkpoint()
            .or_invariant("failed shift must preserve checkpoint geometry")
            .copied(),
        checkpoint_before
    );
}

#[test]
fn maximum_level_snapshot_checkpoint_is_typed_and_does_not_expand() {
    let level = crate::hashlife::MAX_COORD_ROOT_LEVEL;
    let mut snapshot = format!(
        "{}\ngeneration 7\norigin 0 0\nroot N{}@0\nnodes {}\n",
        crate::persistence::HASHLIFE_SNAPSHOT_MAGIC,
        level - 1,
        level
    );
    for node_level in 1..=level {
        let child = if node_level == 1 {
            "D".to_owned()
        } else {
            format!("N{}", node_level - 2)
        };
        snapshot.push_str(&format!(
            "node {node_level} {child}@0 {child}@0 {child}@0 {child}@0\n"
        ));
    }

    let mut session = HashLifeSession::new();
    let loaded = session.load_snapshot_string(&snapshot);
    assert!(
        loaded.is_ok(),
        "maximum-level snapshot failed to load: {loaded:?}"
    );
    let checkpoint = session.try_signature_checkpoint();
    assert!(
        checkpoint.is_ok(),
        "maximum-level checkpoint returned a geometry failure: {checkpoint:?}"
    );
    let checkpoint = checkpoint
        .or_invariant("successful checkpoint query must return a value")
        .or_invariant("loaded snapshot must have a checkpoint");
    assert_eq!(checkpoint.identity.level, level);
    assert_eq!(checkpoint.root_span, 1_i64 << level);
}

#[test]
fn hashlife_session_set_bit_decomposition_matches_scalar_without_materialization() {
    const GENERATIONS: u64 = 37;
    let grid = random_soup(
        SMALL_SOUP_DIM / 2,
        SMALL_SOUP_DIM / 2,
        SMALL_SOUP_FILL,
        SEED_PRIME_JUMP_SOUP,
    );
    let mut expected = GameOfLife::new(grid.clone());
    for _ in 0..GENERATIONS {
        expected.step_with_chunk_changes();
    }

    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    let advanced = session
        .advance_root(GENERATIONS)
        .or_invariant("bounded HashLife decomposition should complete");
    let stats_before_inspection = session.execution_stats();
    assert_eq!(advanced.completed_generations, GENERATIONS);
    assert_eq!(session.generation(), GENERATIONS);
    assert_eq!(
        stats_before_inspection.materializations, 0,
        "root advancement must not materialize cells: {stats_before_inspection:?}"
    );
    assert_grids_eq(
        "low-to-high HashLife set-bit decomposition should match scalar stepping",
        &session
            .sample_grid()
            .or_invariant("bounded result should be inspectable"),
        expected.grid(),
    );
}

#[test]
fn hashlife_memory_error_preserves_last_completed_generation() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    session.set_limits(HashLifeLimits {
        soft_memory_bytes: 0,
        hard_memory_bytes: 1,
    });
    let allocated_before = session.allocated_bytes();

    let error = match session.advance_root(10) {
        Err(error) => error,
        Ok(stats) => crate::invariant_failure!(
            "one-byte budget unexpectedly completed generation stats={stats:?}"
        ),
    };
    let HashLifeAdvanceError::MemoryBudgetExceeded {
        starting_generation,
        requested_delta,
        completed_generations,
        requested_generation,
        reached_generation,
        allocated_bytes,
        limit_bytes,
    } = error
    else {
        crate::invariant_failure!("expected memory budget error, got {error:?}");
    };
    assert_eq!(starting_generation, 0);
    assert_eq!(requested_delta, 10);
    assert_eq!(completed_generations, reached_generation);
    assert_eq!(requested_generation, 10);
    assert_eq!(reached_generation, 0);
    assert_eq!(session.generation(), reached_generation);
    assert_eq!(allocated_bytes, allocated_before);
    assert_eq!(session.allocated_bytes(), allocated_before);
    assert!(allocated_before > limit_bytes);
    assert_eq!(limit_bytes, 1);

    session.set_limits(HashLifeLimits::default());
    session
        .advance_root(10)
        .or_invariant("retry after raising the budget should complete");

    let mut uninterrupted = HashLifeSession::new();
    uninterrupted
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    uninterrupted
        .advance_root(10)
        .or_invariant("uninterrupted comparison should complete");
    let retried = session
        .signature_checkpoint()
        .cloned()
        .or_invariant("retried session should expose a checkpoint");
    let expected = uninterrupted
        .signature_checkpoint()
        .cloned()
        .or_invariant("uninterrupted session should expose a checkpoint");
    assert_checkpoint_state_eq(
        "retry after a failed transactional segment",
        &retried,
        &expected,
    );
}

#[test]
fn hashlife_matches_glider_after_single_step() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    assert_stepper_matches_single_step(&grid);
}

#[test]
fn hashlife_matches_blinker_after_large_even_and_odd_jumps() {
    let grid = pattern_by_name("blinker").or_invariant("required value");
    let mut oracle = HashLifeEngine::default();
    let even = oracle.advance(&grid, GEN_BLINKER_EVEN);
    let odd = oracle.advance(&grid, GEN_BLINKER_ODD);

    assert_grids_eq("blinker even jump should match generation 0", &even, &grid);
    assert_grids_eq(
        "blinker odd jump should match one stepped generation",
        &odd,
        &step_grid(&grid),
    );
}

#[test]
fn hashlife_session_gosper_gun_core_survives_ten_million_scale_period_multiple() {
    let initial = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let (min_x, min_y, max_x, max_y) = initial
        .bounds()
        .or_invariant("gosper glider gun should be non-empty");
    let target = GEN_GUN_PERIOD_MULTIPLE;
    assert_eq!(target % 30, 0, "test target must land on the gun period");

    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&initial)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(target)
        .or_invariant("large prime jump should complete");

    let sampled_core = session
        .sample_region(min_x, min_y, max_x, max_y)
        .or_invariant("bounded gun core sample should remain available");

    assert_grids_eq(
        "gosper gun core should still match generation 0 after a deep period-aligned jump",
        &sampled_core,
        &initial,
    );
    assert!(
        session.sample_materializations() <= 1,
        "bounded gun-core regression should avoid full-grid materialization, got {} materializations",
        session.sample_materializations()
    );
}

#[test]
fn hashlife_snapshot_roundtrips_session_state() {
    let initial = pattern_by_name("glider").or_invariant("required value");
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&initial)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(GEN_SNAPSHOT)
        .or_invariant("snapshot fixture should complete");

    let snapshot = session
        .export_snapshot_string()
        .or_invariant("snapshot export should succeed")
        .or_invariant("loaded session should export a snapshot");
    let expected_generation = session.generation();
    let expected_origin = session.origin();
    let expected_population = session.population_count();
    let expected_bounds = session.bounds();
    let expected_checkpoint = session.signature_checkpoint().cloned();
    let expected_grid = session
        .sample_grid()
        .or_invariant("snapshot source should be materializable");

    let mut restored = HashLifeSession::new();
    restored
        .load_snapshot_string(&snapshot)
        .or_invariant("snapshot should reload");

    assert_eq!(
        restored.generation(),
        expected_generation,
        "snapshot generation mismatch after roundtrip"
    );
    assert_eq!(
        restored.origin(),
        expected_origin,
        "snapshot origin mismatch after roundtrip"
    );
    assert_eq!(
        restored.population_count(),
        expected_population,
        "snapshot population mismatch after roundtrip"
    );
    assert_eq!(
        restored.bounds(),
        expected_bounds,
        "snapshot bounds mismatch after roundtrip"
    );
    let restored_checkpoint = restored
        .signature_checkpoint()
        .cloned()
        .or_invariant("restored snapshot should have a checkpoint");
    assert_checkpoint_state_eq(
        "snapshot checkpoint mismatch after roundtrip",
        &restored_checkpoint,
        &expected_checkpoint.or_invariant("snapshot source should have a checkpoint"),
    );
    assert_grids_eq(
        "snapshot roundtrip should preserve materialized grid",
        &restored
            .sample_grid()
            .or_invariant("restored snapshot should be materializable"),
        &expected_grid,
    );
}

#[test]
fn hashlife_snapshot_persists_deep_run_resume() {
    let initial = pattern_by_name("glider").or_invariant("required value");
    let mut uninterrupted = HashLifeSession::new();
    uninterrupted
        .try_load_grid(&initial)
        .or_invariant("test HashLife grid should load");
    uninterrupted
        .advance_root(GEN_DEEP_DIAGNOSTIC / 3)
        .or_invariant("checkpoint prefix should complete");
    let snapshot = uninterrupted
        .export_snapshot_string()
        .or_invariant("snapshot export should succeed")
        .or_invariant("deep run should export a snapshot");

    uninterrupted
        .advance_root(GEN_RESUME_DELTA)
        .or_invariant("uninterrupted suffix should complete");
    let expected = uninterrupted
        .signature_checkpoint()
        .cloned()
        .or_invariant("continued deep run should have a checkpoint");

    let mut resumed = HashLifeSession::new();
    resumed
        .load_snapshot_string(&snapshot)
        .or_invariant("snapshot should reload");
    resumed
        .advance_root(GEN_RESUME_DELTA)
        .or_invariant("resumed suffix should complete");
    let actual = resumed
        .signature_checkpoint()
        .cloned()
        .or_invariant("resumed deep run should have a checkpoint");

    assert_checkpoint_state_eq(
        "deep resume checkpoint mismatch after continuing snapshot session",
        &actual,
        &expected,
    );
    assert_grids_eq(
        "deep resume absolute state mismatch after continuing snapshot session",
        &resumed
            .sample_grid()
            .or_invariant("resumed deep run should be materializable"),
        &uninterrupted
            .sample_grid()
            .or_invariant("uninterrupted deep run should be materializable"),
    );
}

#[test]
fn hashlife_snapshot_export_is_deterministic_for_same_session() {
    let initial = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&initial)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(GEN_RESUME_DELTA)
        .or_invariant("imported snapshot should advance");

    let first = session
        .export_snapshot_string()
        .or_invariant("loaded session should export a snapshot");
    let second = session
        .export_snapshot_string()
        .or_invariant("repeated export should succeed");
    assert_eq!(first, second, "snapshot export should be deterministic");
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
    let grid = pattern_by_name("glider").or_invariant("required value");
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
    let grid = pattern_by_name("glider").or_invariant("required value");
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
    assert_grids_eq(
        "segmented prime advance should match one-shot advance",
        &combined,
        &segmented,
    );
}

#[test]
fn hashlife_matches_stepper_on_all_4x4_single_steps() {
    let mut oracle = HashLifeEngine::default();
    for mask in 0_u32..(1 << 16) {
        let grid = grid_from_mask(4, 4, mask);
        let advanced = oracle.advance(&grid, GEN_SINGLE_STEP);
        assert_grids_eq(
            &format!("4x4 single-step mismatch for mask={mask:#06x}"),
            &advanced,
            &step_grid(&grid),
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
            assert_grids_eq(
                &format!(
                    "5x5 sampled jump mismatch for mask={mask:#08x} generations={generations}"
                ),
                &advanced,
                game.grid(),
            );
        }
    }
}
