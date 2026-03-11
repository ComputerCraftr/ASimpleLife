use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeEngine;
use crate::life::{GameOfLife, step_grid};
use crate::normalize::normalize;
use crate::bitgrid::{BitGrid, Coord};
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

#[test]
fn hashlife_matches_block_after_large_jump() {
    let grid = pattern_by_name("block").unwrap();
    let advanced = HashLifeEngine::default().advance(&grid, 1_000_000);
    assert_eq!(normalize(&advanced).0, normalize(&grid).0);
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
    let even = HashLifeEngine::default().advance(&grid, 1_000);
    let odd = HashLifeEngine::default().advance(&grid, 1_001);

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
    let _ = oracle.advance(&grid, 509);
    let first = oracle.runtime_stats();
    assert!(first.nodes > 2);
    assert_eq!(first.jump_cache, 0);
    assert_eq!(first.retained_roots, 1);
    assert_eq!(first.nodes, first.intern);

    let _ = oracle.advance(&grid, 509);
    let second = oracle.runtime_stats();
    assert_eq!(first.nodes, second.nodes);
    assert_eq!(first.intern, second.intern);
    assert_eq!(first.empty_levels, second.empty_levels);
    assert_eq!(first.jump_cache, second.jump_cache);
    assert_eq!(first.retained_roots, second.retained_roots);
    assert!(second.jump_cache_before_clear > 0);
    assert!(second.overlap_cache <= second.nodes);

    let other = random_soup(24, 24, 20, 0x0DDC_0FFE_EE11_2233);
    let _ = oracle.advance(&other, 509);
    let third = oracle.runtime_stats();
    assert_eq!(third.jump_cache, 0);
    assert_eq!(third.retained_roots, 1);
    assert_eq!(third.nodes, third.intern);
    assert!(third.nodes <= second.nodes * 2);
}

#[test]
fn hashlife_matches_stepper_on_all_4x4_single_steps() {
    for mask in 0_u32..(1 << 16) {
        let grid = grid_from_mask(4, 4, mask);
        let advanced = HashLifeEngine::default().advance(&grid, 1);
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
        0x0000000,
        0x0000001,
        0x001f000,
        0x0108421,
        0x1555555,
        0x0f0f0f0,
        0x1249249,
        0x1ffffff,
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
    let _ = oracle.advance(&grid, 509);
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
