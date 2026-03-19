use crate::app::initial_grid;
use crate::bf::{CodegenOpts, compile_to_life_circuit};
use crate::cli::Config;
use crate::engine::{
    SimulationBackend, SimulationSession, select_backend, should_use_exact_simd_repeat_skip,
};
use crate::hashlife::GridExtractionPolicy;
use crate::life::step_grid;
use crate::normalize::normalize;

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
    let expected = crate::generators::pattern_by_name("glider").unwrap();
    assert_eq!(normalize(&grid).0, normalize(&expected).0);
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
    let (min_x, min_y, max_x, max_y) = grid.bounds().unwrap();
    assert_eq!((min_x, min_y), (0, 0));
    assert!(max_x < ((config.width as i64) * 2) / 3);
    assert!(max_y < config.height as i64);
}

#[test]
fn engine_policy_uses_simd_for_small_fast_forward() {
    let grid = crate::generators::pattern_by_name("glider").unwrap();
    assert_eq!(select_backend(&grid, 32), SimulationBackend::SimdChunk);
}

#[test]
fn engine_policy_uses_hashlife_for_large_sparse_fast_forward() {
    let grid = crate::generators::pattern_by_name("glider_producing_switch_engine").unwrap();
    assert_eq!(select_backend(&grid, 2048), SimulationBackend::HashLife);
}

#[test]
fn engine_advance_handles_trillion_fast_forward_for_stable_pattern() {
    let grid = crate::generators::pattern_by_name("block").unwrap();
    let mut session = SimulationSession::new();
    let stats = {
        session.load_hashlife_state(&grid);
        session.advance_hashlife_root(1_000_000_000_000)
    };
    let advanced = session
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u64::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .expect("hashlife state should be sampleable after deep run");

    assert_eq!(normalize(&advanced).0, normalize(&grid).0);
    assert_eq!(
        stats.simd_generations + stats.hashlife_generations,
        1_000_000_000_000
    );
}

#[test]
fn deep_run_uses_hashlife_for_large_fast_forward() {
    let grid = crate::generators::pattern_by_name("gosper_glider_gun").unwrap();
    let mut session = SimulationSession::new();
    session.load_hashlife_state(&grid);
    let stats = session.advance_hashlife_root(100_000);

    assert_eq!(stats.backend, SimulationBackend::HashLife);
    assert_eq!(stats.simd_generations, 0);
    assert_eq!(stats.hashlife_generations, 100_000);
}

#[test]
fn reloaded_hashlife_step_generations_matches_one_shot_target() {
    let grid = crate::generators::pattern_by_name("glider").unwrap();
    let mut repeated = grid;
    let mut session = SimulationSession::new();
    for _ in 0..5 {
        session.load_hashlife_state(&repeated);
        session.advance_hashlife_root(5);
        repeated = session
            .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
                max_population: u64::MAX,
                max_chunks: usize::MAX,
                max_bounds_span: i64::MAX,
            })
            .expect("hashlife state should be sampleable after repeated stepped reload");
    }

    let mut one_shot = SimulationSession::new();
    one_shot.load_hashlife_state(&crate::generators::pattern_by_name("glider").unwrap());
    one_shot.advance_hashlife_root(25);
    let one_shot_grid = one_shot
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u64::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .expect("hashlife state should be sampleable after one-shot stepping");

    assert_eq!(normalize(&repeated).0, normalize(&one_shot_grid).0);
}

#[test]
fn session_planner_prefers_hashlife_when_checkpointable_state_is_loaded() {
    let grid = crate::generators::pattern_by_name("gosper_glider_gun").unwrap();
    let mut session = SimulationSession::new();
    session.load_hashlife_state(&grid);
    session.advance_hashlife_root(1_024);
    let shape = session
        .hashlife_bounds()
        .map(|(min_x, min_y, max_x, max_y)| {
            let span = (max_x - min_x + 1).max(max_y - min_y + 1);
            (
                usize::try_from(session.hashlife_population().unwrap()).unwrap(),
                span,
            )
        })
        .unwrap();

    assert_eq!(
        session.planned_backend_from_session_metrics(shape.0, shape.1, 32),
        SimulationBackend::HashLife
    );
}

#[test]
fn short_multi_step_mode_prefers_exact_simd_repeat_skip() {
    let grid = crate::generators::pattern_by_name("glider").unwrap();
    assert!(should_use_exact_simd_repeat_skip(&grid, 5));
}

#[test]
fn exact_simd_repeat_skip_matches_manual_glider_translation() {
    let grid = crate::generators::pattern_by_name("glider").unwrap();
    let mut session = SimulationSession::new();
    let (advanced, _) = session.advance_simd_chunk_exact(&grid, 257);

    let mut expected = grid.clone();
    for _ in 0..257 {
        expected = step_grid(&expected);
    }

    assert_eq!(normalize(&advanced).0, normalize(&expected).0);
}

#[test]
fn exact_simd_repeat_skip_matches_million_generation_blinker() {
    let grid = crate::generators::pattern_by_name("blinker").unwrap();
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
    assert_eq!(normalize(&advanced).0, normalize(&expected).0);
}

#[test]
fn exact_simd_repeat_skip_matches_million_generation_glider() {
    let grid = crate::generators::pattern_by_name("glider").unwrap();
    let target = 1_000_003_u64;

    let mut session = SimulationSession::new();
    let (advanced, stats) = session.advance_simd_chunk_exact(&grid, target);
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, target);

    assert_eq!(stats.backend, SimulationBackend::SimdChunk);
    assert_eq!(stats.simd_generations, target);
    assert_eq!(normalize(&advanced).0, normalize(&expected).0);
}

#[test]
fn simulation_session_grid_hashlife_snapshot_roundtrip_preserves_loaded_grid() {
    let grid = crate::generators::pattern_by_name("gosper_glider_gun")
        .unwrap()
        .translated(37, -19);

    let mut session = SimulationSession::new();
    session.load_hashlife_state(&grid);
    let snapshot = session
        .export_hashlife_snapshot()
        .expect("loaded grid should export a HashLife snapshot");

    let mut restored = SimulationSession::new();
    restored
        .load_hashlife_snapshot(&snapshot)
        .expect("exported snapshot should reload");
    let restored_grid = restored
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u64::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .expect("restored snapshot should materialize to a grid");

    assert_eq!(restored_grid, grid);
}

#[test]
fn bf_life_circuit_grid_hashlife_snapshot_roundtrip_preserves_compiled_grid() {
    let opts = CodegenOpts {
        io_mode: crate::bf::IoMode::Char,
        cell_bits: 32,
        input_bits: None,
        output_bits: None,
        cell_sign: crate::bf::CellSign::Unsigned,
    };
    let mut circuit = compile_to_life_circuit(
        &crate::bf::optimize(crate::bf::Parser::new("+.>++.>+++.").parse().unwrap()),
        opts,
    )
    .expect("BF circuit should compile");
    circuit
        .run_to_completion()
        .expect("BF circuit should run to completion");
    let grid = circuit.to_initial_grid();

    let mut session = SimulationSession::new();
    session.load_hashlife_state(&grid);
    let snapshot = session
        .export_hashlife_snapshot()
        .expect("compiled BF circuit grid should export as a HashLife snapshot");

    let mut restored = SimulationSession::new();
    restored
        .load_hashlife_snapshot(&snapshot)
        .expect("BF circuit snapshot should reload");
    let restored_grid = restored
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u64::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .expect("restored BF circuit snapshot should materialize to a grid");

    assert_eq!(restored_grid, grid);
}
