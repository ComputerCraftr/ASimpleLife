use crate::RequiredExt;
use crate::app::initial_grid;
use crate::bf::CodegenOpts;
use crate::cli::Config;
use crate::engine::{
    SimulationBackend, SimulationConversionError, SimulationSession, select_backend,
    should_use_exact_simd_repeat_skip,
};
use crate::hashlife::{
    GridExtractionPolicy, HashLifeAllocationClass, HashLifeAllocationFailure, HashLifeLimits,
};
use crate::life::step_grid;
use crate::normalize::normalize;

fn assert_normalized_grids_eq(
    label: &str,
    actual: &crate::bitgrid::BitGrid,
    expected: &crate::bitgrid::BitGrid,
) {
    let actual_normalized = normalize(actual);
    let expected_normalized = normalize(expected);
    assert_eq!(
        actual_normalized.0, expected_normalized.0,
        "{label}: normalized grids differ\nactual_origin={:?} expected_origin={:?}\nactual={:?}\nexpected={:?}",
        actual_normalized.1, expected_normalized.1, actual_normalized.0, expected_normalized.0
    );
}

#[test]
fn initial_grid_uses_named_pattern() {
    let config = Config {
        width: 80,
        height: 24,
        steps: 1,
        max_generations: None,
        target_generation: None,
        step_generations: 1,
        delay_ms: 0,
        seed: 123,
        pattern: "glider".to_string(),
        classify_only: false,
    };

    let grid = initial_grid(&config);
    let expected = crate::generators::pattern_by_name("glider").or_invariant("required value");
    assert_normalized_grids_eq(
        "named initial grid should match generator pattern",
        &grid,
        &expected,
    );
}

#[test]
fn initial_grid_random_soup_respects_config_dimensions() {
    let config = Config {
        width: 90,
        height: 30,
        steps: 1,
        max_generations: None,
        target_generation: None,
        step_generations: 1,
        delay_ms: 0,
        seed: 7,
        pattern: "random".to_string(),
        classify_only: false,
    };

    let grid = initial_grid(&config);
    assert!(
        grid.population() > 0,
        "random soup should not be empty for the test seed"
    );
    let (min_x, min_y, max_x, max_y) = grid.bounds().or_invariant("required value");
    assert_eq!((min_x, min_y), (0, 0));
    let width = i64::try_from(config.width).or_invariant("test width exceeded i64");
    let height = i64::try_from(config.height).or_invariant("test height exceeded i64");
    assert!(max_x < width.saturating_mul(2) / 3);
    assert!(max_y < height);
}

#[test]
fn engine_policy_uses_simd_for_small_fast_forward() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    assert_eq!(select_backend(&grid, 32), SimulationBackend::SimdChunk);
}

#[test]
fn engine_policy_uses_hashlife_for_large_sparse_fast_forward() {
    let grid = crate::generators::pattern_by_name("glider_producing_switch_engine")
        .or_invariant("required value");
    assert_eq!(select_backend(&grid, 2048), SimulationBackend::HashLife);
}

#[test]
fn engine_advance_handles_trillion_fast_forward_for_stable_pattern() {
    let grid = crate::generators::pattern_by_name("block").or_invariant("required value");
    let mut session = SimulationSession::new();
    let stats = {
        session
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");
        session.advance_hashlife_root(1_000_000_000_000)
    }
    .or_invariant("stable block should reach the requested generation");
    let advanced = session
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("hashlife state should be sampleable after deep run");

    assert_normalized_grids_eq(
        "trillion-generation stable block should remain unchanged",
        &advanced,
        &grid,
    );
    assert_eq!(
        stats.simd_generations + stats.hashlife_generations,
        1_000_000_000_000
    );
}

#[test]
fn deep_run_uses_hashlife_for_large_fast_forward() {
    let grid =
        crate::generators::pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    let stats = session
        .advance_hashlife_root(100_000)
        .or_invariant("gun should reach the requested generation");

    assert_eq!(stats.backend, SimulationBackend::HashLife);
    assert_eq!(stats.simd_generations, 0);
    assert_eq!(stats.hashlife_generations, 100_000);
}

#[test]
fn reloaded_hashlife_step_generations_matches_one_shot_target() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    let mut repeated = grid;
    let mut session = SimulationSession::new();
    for _ in 0..5 {
        session
            .try_load_hashlife_state(&repeated)
            .or_invariant("test HashLife state should load");
        session
            .advance_hashlife_root(5)
            .or_invariant("repeated HashLife segment should complete");
        repeated = session
            .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
                max_population: u128::MAX,
                max_chunks: usize::MAX,
                max_bounds_span: i64::MAX,
            })
            .or_invariant("hashlife state should be sampleable after repeated stepped reload");
    }

    let mut one_shot = SimulationSession::new();
    one_shot
        .try_load_hashlife_state(
            &crate::generators::pattern_by_name("glider").or_invariant("required value"),
        )
        .or_invariant("test HashLife state should load");
    one_shot
        .advance_hashlife_root(25)
        .or_invariant("one-shot HashLife segment should complete");
    let one_shot_grid = one_shot
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("hashlife state should be sampleable after one-shot stepping");

    assert_normalized_grids_eq(
        "reloaded repeated HashLife stepping should match one-shot stepping",
        &repeated,
        &one_shot_grid,
    );
}

#[test]
fn session_planner_prefers_hashlife_when_checkpointable_state_is_loaded() {
    let grid =
        crate::generators::pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(1_024)
        .or_invariant("planner fixture should complete");
    let shape = session
        .hashlife_bounds()
        .map(|(min_x, min_y, max_x, max_y)| {
            let span = (max_x - min_x + 1).max(max_y - min_y + 1);
            (
                usize::try_from(
                    session
                        .hashlife_population_count()
                        .or_invariant("required value")
                        .lower_bound(),
                )
                .or_invariant("required value"),
                span,
            )
        })
        .or_invariant("required value");

    assert_eq!(
        session.planned_backend_from_session_metrics(shape.0, shape.1, 32),
        SimulationBackend::HashLife
    );
}

#[test]
fn cell_advance_invalidates_loaded_hashlife_authority() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    assert!(session.hashlife_loaded());

    let (advanced, stats) = session.advance_simd_chunk_exact(&grid, 1);

    assert_eq!(stats.completed_generations, 1);
    assert!(!advanced.is_empty());
    assert!(
        !session.hashlife_loaded(),
        "cell-authoritative advancement must invalidate the stale HashLife root"
    );
    assert_ne!(
        session.planned_backend_from_metrics(advanced.population(), 3, 2),
        SimulationBackend::HashLife,
        "planner must not reuse an invalidated HashLife representation"
    );
}

#[test]
fn zero_cell_advance_preserves_loaded_hashlife_authority() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");

    let (unchanged, stats) = session.advance_simd_chunk_exact(&grid, 0);

    assert_eq!(stats.completed_generations, 0);
    assert_eq!(unchanged, grid);
    assert!(
        session.hashlife_loaded(),
        "a no-op cell request must not invalidate the authoritative HashLife root"
    );
}

#[test]
fn backend_planning_handles_full_coordinate_width_without_overflow() {
    let grid = crate::bitgrid::BitGrid::from_cells(&[(i64::MIN, 0), (i64::MAX, 0)]);

    let backend = select_backend(&grid, 2);

    assert_eq!(
        backend,
        SimulationBackend::SimdChunk,
        "wide sparse geometry should retain the low-work backend decision without overflowing"
    );
}

#[test]
fn short_multi_step_mode_prefers_exact_simd_repeat_skip() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    assert!(should_use_exact_simd_repeat_skip(&grid, 5));
}

#[test]
fn exact_simd_repeat_skip_matches_manual_glider_translation() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    let mut session = SimulationSession::new();
    let (advanced, _) = session.advance_simd_chunk_exact(&grid, 257);

    let mut expected = grid.clone();
    for _ in 0..257 {
        expected = step_grid(&expected);
    }

    assert_normalized_grids_eq(
        "exact SIMD repeat skip should match manual glider stepping",
        &advanced,
        &expected,
    );
}

#[test]
fn exact_simd_repeat_skip_matches_million_generation_blinker() {
    let grid = crate::generators::pattern_by_name("blinker").or_invariant("required value");
    let target = 1_000_003_u64;

    let mut session = SimulationSession::new();
    let (advanced, stats) = session.advance_simd_chunk_exact(&grid, target);
    let remainder = target % 2;
    let mut expected = grid.clone();
    for _ in 0..remainder {
        expected = step_grid(&expected);
    }

    assert_eq!(stats.backend, SimulationBackend::SimdChunk);
    assert_eq!(stats.simd_generations, target);
    assert!(stats.repeat_skip_events > 0);
    assert!(stats.repeat_skip_generations > 0);
    assert_normalized_grids_eq(
        "exact SIMD repeat skip should match million-generation blinker remainder",
        &advanced,
        &expected,
    );
}

#[test]
fn exact_simd_repeat_skip_matches_million_generation_glider() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("required value");
    let target = 1_000_003_u64;

    let mut session = SimulationSession::new();
    let (advanced, stats) = session.advance_simd_chunk_exact(&grid, target);
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, target);

    assert_eq!(stats.backend, SimulationBackend::SimdChunk);
    assert_eq!(stats.simd_generations, target);
    assert_normalized_grids_eq(
        "exact SIMD repeat skip should match HashLife glider target",
        &advanced,
        &expected,
    );
}

#[test]
fn simulation_session_grid_hashlife_snapshot_roundtrip_preserves_loaded_grid() {
    let grid = crate::generators::pattern_by_name("gosper_glider_gun")
        .or_invariant("required value")
        .translated(37, -19);

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    let snapshot = session
        .export_hashlife_snapshot()
        .or_invariant("snapshot export should succeed")
        .or_invariant("loaded grid should export a HashLife snapshot");

    let mut restored = SimulationSession::new();
    restored
        .load_hashlife_snapshot(&snapshot)
        .or_invariant("exported snapshot should reload");
    let restored_grid = restored
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("restored snapshot should materialize to a grid");

    assert_eq!(restored_grid, grid);
}

#[test]
fn bf_life_scaffold_grid_hashlife_snapshot_roundtrip_preserves_layout_grid() {
    let opts = CodegenOpts {
        io_mode: crate::bf::IoMode::Char,
        cell_bits: 32,
        input_bits: None,
        output_bits: None,
        cell_sign: crate::bf::CellSign::Unsigned,
    };
    let circuit = crate::bf::compile_life_scaffold(
        &crate::bf::optimize_with_opts(
            crate::bf::Parser::new("+.>++.>+++.")
                .parse()
                .or_invariant("required value"),
            opts,
        ),
        opts,
    )
    .or_invariant("BF circuit should compile");
    let grid = circuit.compiled_grid();

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    let snapshot = session
        .export_hashlife_snapshot()
        .or_invariant("snapshot export should succeed")
        .or_invariant("compiled BF circuit grid should export as a HashLife snapshot");

    let mut restored = SimulationSession::new();
    restored
        .load_hashlife_snapshot(&snapshot)
        .or_invariant("BF circuit snapshot should reload");
    let restored_grid = restored
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("restored BF circuit snapshot should materialize to a grid");

    assert_eq!(restored_grid, grid);
}

#[test]
fn failed_cell_to_hashlife_conversion_preserves_cell_authority() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("glider fixture");
    let mut session = SimulationSession::with_hashlife_limits(HashLifeLimits {
        soft_memory_bytes: 0,
        hard_memory_bytes: 1,
    });
    session.load_cell_state(grid.clone(), 37);

    let error = match session.try_convert_to_hashlife() {
        Err(error) => error,
        Ok(()) => crate::invariant_failure!("one-byte budget accepted HashLife conversion"),
    };

    assert!(
        matches!(error, SimulationConversionError::HashLife(_)),
        "unexpected conversion failure: {error:?}"
    );
    let (retained, generation) = session
        .cell_state()
        .or_invariant("failed conversion must retain cell authority");
    assert_eq!(
        retained, &grid,
        "failed conversion changed authoritative grid"
    );
    assert_eq!(generation, 37, "failed conversion changed generation");
}

#[test]
fn failed_hashlife_to_cell_conversion_preserves_root_for_retry() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("glider fixture");
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(4)
        .or_invariant("HashLife setup should advance");

    let error = match session.try_convert_to_cell(GridExtractionPolicy::FullGridIfUnder {
        max_population: 0,
        max_chunks: 0,
        max_bounds_span: 0,
    }) {
        Err(error) => error,
        Ok(()) => crate::invariant_failure!("zero extraction limits accepted a live root"),
    };
    assert!(
        matches!(error, SimulationConversionError::Extraction(_)),
        "unexpected reverse conversion failure: {error:?}"
    );
    assert!(
        session.hashlife_loaded(),
        "failed extraction lost HashLife authority"
    );
    let advanced = session
        .advance_hashlife_root(1)
        .or_invariant("retained HashLife root should remain usable");
    assert_eq!(advanced.starting_generation, 4);
    assert_eq!(advanced.reached_generation, 5);
}

#[test]
fn failed_snapshot_import_preserves_published_hashlife_state() {
    let grid = crate::generators::pattern_by_name("block").or_invariant("block fixture");
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(3)
        .or_invariant("stable fixture should advance");
    let before = session
        .hashlife_checkpoint()
        .copied()
        .or_invariant("loaded HashLife state should have a checkpoint");

    let result = session.load_hashlife_snapshot("not a HashLife snapshot");

    assert!(result.is_err(), "invalid snapshot unexpectedly loaded");
    let after = session
        .hashlife_checkpoint()
        .copied()
        .or_invariant("failed import lost the published checkpoint");
    assert_eq!(
        after, before,
        "failed import changed published HashLife state"
    );
    assert!(session.hashlife_loaded(), "failed import changed authority");
}

#[test]
fn production_allocation_gate_targets_conversion_class_and_ordinal() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("glider fixture");
    let mut session = SimulationSession::new();
    session.load_cell_state(grid.clone(), 9);
    session.configure_hashlife_allocation_failure(Some(HashLifeAllocationFailure {
        class: HashLifeAllocationClass::Embed,
        ordinal: 1,
    }));

    assert!(session.try_convert_to_hashlife().is_err());
    let (retained, generation) = session
        .cell_state()
        .or_invariant("injected embed failure changed authority");
    assert_eq!((retained, generation), (&grid, 9));

    session.configure_hashlife_allocation_failure(None);
    session
        .try_convert_to_hashlife()
        .or_invariant("conversion should retry after clearing allocation gate");
    session.configure_hashlife_allocation_failure(Some(HashLifeAllocationFailure {
        class: HashLifeAllocationClass::Materialize,
        ordinal: 1,
    }));
    assert!(
        session
            .try_convert_to_cell(GridExtractionPolicy::FullGridIfUnder {
                max_population: u128::MAX,
                max_chunks: usize::MAX,
                max_bounds_span: i64::MAX,
            })
            .is_err()
    );
    assert!(session.hashlife_loaded());
}

#[test]
fn failed_hashlife_segment_reports_only_committed_progress_and_retries_exactly() {
    let grid = crate::generators::pattern_by_name("glider").or_invariant("glider fixture");
    let mut interrupted = SimulationSession::new();
    interrupted
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    interrupted.configure_hashlife_allocation_failure(Some(HashLifeAllocationFailure {
        class: HashLifeAllocationClass::ArenaGrowth,
        ordinal: 2,
    }));

    let failure = match interrupted.advance_hashlife_root(3) {
        Err(error) => error,
        Ok(stats) => crate::invariant_failure!(
            "injected second-segment failure unexpectedly completed: {stats:?}"
        ),
    };
    assert_eq!(failure.completed_generations(), 1);
    assert_eq!(failure.reached_generation(), 1);

    interrupted.configure_hashlife_allocation_failure(None);
    interrupted
        .advance_hashlife_root(2)
        .or_invariant("retry should complete remaining committed delta");
    let interrupted_grid = interrupted
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("interrupted result should materialize");

    let mut uninterrupted = SimulationSession::new();
    uninterrupted
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    uninterrupted
        .advance_hashlife_root(3)
        .or_invariant("uninterrupted run should complete");
    let uninterrupted_grid = uninterrupted
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("uninterrupted result should materialize");
    assert_eq!(interrupted_grid, uninterrupted_grid);
}
