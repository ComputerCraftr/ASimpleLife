use crate::app::initial_grid;
use crate::cli::Config;
use crate::engine::{SimulationBackend, advance_grid, select_backend};
use crate::life::GameOfLife;
use crate::normalize::normalize;

#[test]
fn initial_grid_uses_named_pattern() {
    let config = Config {
        width: 80,
        height: 24,
        steps: 1,
        max_generations: None,
        fast_forward: 0,
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
        fast_forward: 0,
        delay_ms: 0,
        seed: 7,
        pattern: "random".to_string(),
        classify_only: false,
    };

    let grid = initial_grid(&config);
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
fn engine_advance_matches_stepper_for_hybrid_candidate() {
    let grid = crate::generators::random_soup(80, 80, 20, 0x0123_4567_89AB_CDEF);
    let advanced = advance_grid(&grid, 128);

    let mut game = GameOfLife::new(grid);
    for _ in 0..128 {
        game.step_with_changes();
    }

    assert_eq!(normalize(&advanced.grid).0, normalize(game.grid()).0);
    assert_eq!(advanced.stats.backend, SimulationBackend::HybridSegmented);
}

#[test]
fn engine_advance_handles_trillion_fast_forward_for_stable_pattern() {
    let grid = crate::generators::pattern_by_name("block").unwrap();
    let advanced = advance_grid(&grid, 1_000_000_000_000);

    assert_eq!(normalize(&advanced.grid).0, normalize(&grid).0);
    assert_eq!(advanced.stats.simd_generations + advanced.stats.hashlife_generations, 1_000_000_000_000);
}
