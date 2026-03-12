use crate::bitgrid::{BitGrid, Coord};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeEngine;
use crate::life::{GameOfLife, step_grid};
use crate::normalize::normalize;
use std::time::Instant;

fn assert_hashlife_matches_stepper(grid: crate::bitgrid::BitGrid, generations: u64) {
    let mut game = GameOfLife::new(grid.clone());
    for _ in 0..generations {
        game.step_with_changes();
    }

    let advanced = HashLifeEngine::default().advance(&grid, generations);
    assert_eq!(normalize(&advanced).0, normalize(game.grid()).0);
}

fn grid_from_mask(width: Coord, height: Coord, mask: u32) -> BitGrid {
    let mut cells = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let bit = u32::try_from(y * width + x).expect("grid mask index exceeded u32");
            if (mask >> bit) & 1 == 1 {
                cells.push((x, y));
            }
        }
    }
    BitGrid::from_cells(&cells)
}

fn mirror_grid_x(grid: &BitGrid) -> BitGrid {
    let (min_x, _, max_x, _) = grid.bounds().expect("grid should be non-empty");
    let mirrored = grid
        .live_cells()
        .into_iter()
        .map(|(x, y)| (max_x - (x - min_x), y))
        .collect::<Vec<_>>();
    BitGrid::from_cells(&mirrored)
}

#[test]
fn hashlife_matches_glider_after_single_step() {
    let grid = pattern_by_name("glider").unwrap();
    let advanced = HashLifeEngine::default().advance(&grid, 1);
    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_matches_blinker_after_large_even_and_odd_jumps() {
    let grid = pattern_by_name("blinker").unwrap();
    let mut oracle = HashLifeEngine::default();
    let even = oracle.advance(&grid, 1_000);
    let odd = oracle.advance(&grid, 1_001);

    assert_eq!(normalize(&even).0, normalize(&grid).0);
    assert_eq!(normalize(&odd).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_matches_single_step_on_random_soup() {
    let grid = random_soup(24, 24, 20, 0xD1B54A32D192ED03);
    let advanced = HashLifeEngine::default().advance(&grid, 1);
    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_matches_glider_after_power_of_two_jump() {
    let grid = pattern_by_name("glider").unwrap();
    assert_hashlife_matches_stepper(grid, 256);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_power_of_two_jump() {
    let grid = random_soup(24, 24, 20, 0xDEADBEEFCAFEBABE);
    assert_hashlife_matches_stepper(grid, 64);
}

#[test]
fn hashlife_matches_stepper_on_glider_prime_jump() {
    let grid = pattern_by_name("glider").unwrap();
    assert_hashlife_matches_stepper(grid, 269);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_prime_jump() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    assert_hashlife_matches_stepper(grid, 269);
}

#[test]
fn hashlife_matches_random_soup_after_large_power_of_two_jump() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    assert_hashlife_matches_stepper(grid, 256);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_large_prime_jump() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    assert_hashlife_matches_stepper(grid, 509);
}

#[test]
fn hashlife_segmented_prime_equivalence_matches_single_advance() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    let mut oracle = HashLifeEngine::default();
    let combined = oracle.advance(&grid, 509);
    let intermediate = oracle.advance(&grid, 256);
    let segmented = oracle.advance(&intermediate, 253);
    assert_eq!(normalize(&combined).0, normalize(&segmented).0);
}

#[test]
fn hashlife_gc_keeps_state_bounded_and_reusable() {
    let grid = random_soup(24, 24, 20, 0xA5A5_5A5A_DEAD_BEEF);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, 509);
    let first = oracle.runtime_stats();
    assert!(first.nodes > 2);
    assert_eq!(first.jump_cache, 0);
    assert!(first.retained_roots >= 1);
    assert_eq!(first.nodes, first.intern);

    oracle.advance(&grid, 509);
    let second = oracle.runtime_stats();
    assert!(second.retained_roots >= 1);
    assert!(second.jump_cache_before_clear > 0);
    assert_eq!(second.nodes, second.intern);
    assert!(
        second.nodes <= first.nodes.saturating_mul(2),
        "repeated identical deep runs should keep retained canonical state bounded, first={first:?} second={second:?}"
    );

    oracle.advance(&grid, 509);
    let third = oracle.runtime_stats();
    assert_eq!(third.jump_cache, 0);
    assert!(third.retained_roots >= 1);
    assert_eq!(third.nodes, third.intern);
    assert!(
        third.nodes <= second.nodes,
        "repeated identical deep runs should stabilize retained canonical state, second={second:?} third={third:?}"
    );

    let other = random_soup(24, 24, 20, 0x0DDC_0FFE_EE11_2233);
    oracle.advance(&other, 509);
    let fourth = oracle.runtime_stats();
    assert_eq!(fourth.jump_cache, 0);
    assert!(fourth.retained_roots >= 1);
    assert_eq!(fourth.nodes, fourth.intern);
    assert!(
        fourth.nodes > 2,
        "advancing a different grid should still leave a valid reusable retained state, fourth={fourth:?}"
    );
}

#[test]
fn hashlife_matches_stepper_on_all_4x4_single_steps() {
    let mut oracle = HashLifeEngine::default();
    for mask in 0_u32..(1 << 16) {
        let grid = grid_from_mask(4, 4, mask);
        let advanced = oracle.advance(&grid, 1);
        assert_eq!(
            normalize(&advanced).0,
            normalize(&step_grid(&grid)).0,
            "4x4 mask {mask:#06x}"
        );
    }
}

#[test]
fn hashlife_symmetric_workloads_reduce_work_against_fresh_mirror_run() {
    let grid = pattern_by_name("glider").unwrap();
    let mirrored = mirror_grid_x(&grid);
    let generations = 16_u64;
    assert_ne!(
        grid.live_cells(),
        mirrored.live_cells(),
        "sanity check failed: mirrored workload should not be identical to the original grid"
    );
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
        second_stats.symmetric_jump_cache_hits > fresh_stats.symmetric_jump_cache_hits,
        "expected mirrored workload to benefit from prior symmetric jump-cache reuse, second={second_stats:?} fresh={fresh_stats:?}"
    );
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

#[test]
#[ignore = "diagnostic benchmark"]
fn hashlife_diagnostic_medium_prime_jump_benchmark() {
    let grid = random_soup(64, 64, 20, 0xDEADBEEFCAFEBABE);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, 509);
    let elapsed = start.elapsed();
    let stats = oracle.runtime_stats();
    eprintln!(
        "hashlife_bench total_us={} nodes={} marked={} compacted_from={} compacted_to={} builder_frames={} builder_splits={} builder_max_stack={} scheduler_tasks={} scheduler_ready_max={} jump_hits={} jump_misses={} root_hits={} root_misses={} overlap_hits={} overlap_misses={}",
        elapsed.as_micros(),
        stats.nodes,
        stats.nodes_after_mark,
        stats.nodes_before_compact,
        stats.nodes_after_compact,
        stats.builder_frames,
        stats.builder_partitions,
        stats.builder_max_stack,
        stats.scheduler_tasks,
        stats.scheduler_ready_max,
        stats.jump_cache_hits,
        stats.jump_cache_misses,
        stats.root_result_cache_hits,
        stats.root_result_cache_misses,
        stats.overlap_cache_hits,
        stats.overlap_cache_misses
    );
}

#[test]
#[ignore = "diagnostic GC/runtime benchmark"]
fn hashlife_gc_diagnostic_repeated_deep_runs() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    for _ in 0..3 {
        oracle.advance(&grid, 1_000_000);
    }
    let elapsed = start.elapsed();
    let stats = oracle.runtime_stats();
    eprintln!(
        "hashlife_gc_diag total_us={} nodes={} intern={} gc_runs={} gc_skips={} nodes_before_mark={} nodes_after_mark={} nodes_before_compact={} nodes_after_compact={} retained_roots={} jump_cache_before_clear={} gc_reason={}",
        elapsed.as_micros(),
        stats.nodes,
        stats.intern,
        stats.gc_runs,
        stats.gc_skips,
        stats.nodes_before_mark,
        stats.nodes_after_mark,
        stats.nodes_before_compact,
        stats.nodes_after_compact,
        stats.retained_roots,
        stats.jump_cache_before_clear,
        stats.gc_reason
    );
}
