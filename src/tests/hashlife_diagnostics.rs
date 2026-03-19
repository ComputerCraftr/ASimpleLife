use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeEngine;
use std::time::Instant;

use super::hashlife_support::{
    build_hashlife_step0_stress_grid, build_hashlife_structured_symmetry_grid, mirror_grid_x,
    print_hashlife_gate_comparison, print_hashlife_summary, GEN_DEEP_DIAGNOSTIC, GEN_LARGE_PRIME,
    GEN_MIRROR_REUSE, GEN_STRUCTURED_PROMOTION_DIAGNOSTIC, GEN_STRUCTURED_TUNING_DIAGNOSTIC,
    LARGE_SOUP_DIM,
    SEED_CANONICAL_RESULT_INSERT, SMALL_SOUP_FILL, SYMMETRY_GATE_DEFAULT_LEVEL,
    SYMMETRY_GATE_DEFAULT_POPULATION, SYMMETRY_GATE_UNGATED_LEVEL, SYMMETRY_GATE_UNGATED_POPULATION, SYMMETRY_GATE_WIDE_LEVEL,
    SYMMETRY_GATE_WIDE_POPULATION, XL_SOUP_DIM,
};

const SYMMETRY_GATE_CONFIGS: [(&str, u32, u64); 3] = [
    ("default", SYMMETRY_GATE_DEFAULT_LEVEL, SYMMETRY_GATE_DEFAULT_POPULATION),
    ("wider", SYMMETRY_GATE_WIDE_LEVEL, SYMMETRY_GATE_WIDE_POPULATION),
    ("ungated", SYMMETRY_GATE_UNGATED_LEVEL, SYMMETRY_GATE_UNGATED_POPULATION),
];

fn run_symmetry_gate_comparison(label: &str, grid: &crate::bitgrid::BitGrid, generations: u64) {
    for (mode, max_level, max_population) in SYMMETRY_GATE_CONFIGS {
        let mut oracle = HashLifeEngine::with_symmetry_gate_for_tests(max_level, max_population);
        let start = Instant::now();
        oracle.advance(grid, generations);
        let elapsed = start.elapsed();
        let summary = oracle.diagnostic_summary();
        print_hashlife_gate_comparison(label, mode, elapsed, summary);
    }
}

#[test]
#[ignore = "diagnostic symmetry-gate random benchmark"]
fn hashlife_diagnostic_symmetry_gate_random_soup() {
    let grid = random_soup(
        XL_SOUP_DIM,
        XL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_RESULT_INSERT,
    );
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, GEN_LARGE_PRIME);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_sym_gate_random", elapsed, summary);
}

#[test]
#[ignore = "diagnostic symmetry-gate structured benchmark"]
fn hashlife_diagnostic_symmetry_gate_structured_workload() {
    let grid = build_hashlife_structured_symmetry_grid();
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, GEN_DEEP_DIAGNOSTIC);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_sym_gate_structured", elapsed, summary);
}

#[test]
#[ignore = "diagnostic symmetry-gate comparison benchmark"]
fn hashlife_diagnostic_symmetry_gate_comparison_random_soup() {
    let grid = random_soup(
        XL_SOUP_DIM,
        XL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_RESULT_INSERT,
    );
    run_symmetry_gate_comparison("hashlife_sym_gate_compare_random", &grid, GEN_LARGE_PRIME);
}

#[test]
#[ignore = "diagnostic symmetry-gate comparison benchmark"]
fn hashlife_diagnostic_symmetry_gate_comparison_structured() {
    let grid = build_hashlife_structured_symmetry_grid();
    run_symmetry_gate_comparison("hashlife_sym_gate_compare_structured", &grid, GEN_DEEP_DIAGNOSTIC);
}

#[test]
#[ignore = "diagnostic symmetry-gate comparison benchmark"]
fn hashlife_diagnostic_symmetry_gate_comparison_structured_light() {
    let grid = build_hashlife_structured_symmetry_grid();
    run_symmetry_gate_comparison(
        "hashlife_sym_gate_compare_structured_light",
        &grid,
        GEN_STRUCTURED_TUNING_DIAGNOSTIC,
    );
}

#[test]
#[ignore = "diagnostic symmetry-gate promotion benchmark"]
fn hashlife_diagnostic_symmetry_gate_promotion_candidate() {
    let random = random_soup(
        XL_SOUP_DIM,
        XL_SOUP_DIM,
        SMALL_SOUP_FILL,
        SEED_CANONICAL_RESULT_INSERT,
    );
    let structured = build_hashlife_structured_symmetry_grid();
    let deep_gc = pattern_by_name("gosper_glider_gun").unwrap();

    run_symmetry_gate_comparison("hashlife_gate_promotion_random", &random, GEN_LARGE_PRIME);
    run_symmetry_gate_comparison(
        "hashlife_gate_promotion_structured_light",
        &structured,
        GEN_STRUCTURED_TUNING_DIAGNOSTIC,
    );
    run_symmetry_gate_comparison(
        "hashlife_gate_promotion_deep_gc",
        &deep_gc,
        GEN_STRUCTURED_PROMOTION_DIAGNOSTIC,
    );
}

#[test]
#[ignore = "diagnostic benchmark"]
fn hashlife_diagnostic_medium_prime_jump_benchmark() {
    let grid = random_soup(
        LARGE_SOUP_DIM,
        LARGE_SOUP_DIM,
        SMALL_SOUP_FILL,
        super::hashlife_support::SEED_POWER_OF_TWO_SOUP,
    );
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, GEN_LARGE_PRIME);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_bench", elapsed, summary);
}

#[test]
#[ignore = "diagnostic GC/runtime benchmark"]
fn hashlife_gc_diagnostic_repeated_deep_runs() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    for _ in 0..super::hashlife_support::DEEP_RUN_REPETITIONS {
        oracle.advance(&grid, GEN_DEEP_DIAGNOSTIC);
    }
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_gc_diag", elapsed, summary);
}

#[test]
#[ignore = "diagnostic symmetry benchmark"]
fn hashlife_diagnostic_symmetric_mirror_reuse() {
    let grid = pattern_by_name("glider").unwrap();
    let mirrored = mirror_grid_x(&grid);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, GEN_MIRROR_REUSE);
    oracle.advance(&mirrored, GEN_MIRROR_REUSE);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_symmetry_diag", elapsed, summary);
}

#[test]
#[ignore = "diagnostic step0-heavy benchmark"]
fn hashlife_diagnostic_step0_heavy_single_step() {
    let grid = build_hashlife_step0_stress_grid();
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, super::hashlife_support::GEN_STEP0_STRESS);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_step0_diag", elapsed, summary);
}
