use crate::classify::{ClassificationLimits, classify_seed};
use crate::memo::Memo;

use super::correctness::{
    assert_same_outcome, bitmask_pattern, reference_classify, reference_is_decisive,
    run_benchmark_report,
};

#[test]
#[ignore = "exhaustive 5x5 sweep is expensive"]
fn exhaustive_all_5x5_patterns_reference_check() {
    let limits = ClassificationLimits {
        max_generations: 128,
        max_population: 10_000,
        max_bounding_box: 256,
    };
    for mask in 0_u32..(1_u32 << 25) {
        if canonical_small_box_mask(mask, 5, 5) != mask {
            continue;
        }
        let grid = bitmask_pattern(mask, 5, 5);
        let expected = reference_classify(&grid, &limits);
        if !reference_is_decisive(&expected) {
            continue;
        }
        let actual = classify_seed(&grid, &limits, &mut Memo::default());
        assert_same_outcome(&format!("mask_{mask}"), &expected, &actual);
    }
}

#[test]
#[ignore = "benchmark/report harness"]
fn classification_benchmark_report() {
    run_benchmark_report(false);
}

#[test]
#[ignore = "benchmark/report harness"]
fn classification_benchmark_report_json() {
    run_benchmark_report(true);
}

fn canonical_small_box_mask(mask: u32, width: usize, height: usize) -> u32 {
    let mut best = transform_mask(mask, width, height, 0);
    for transform in 1..8 {
        best = best.min(transform_mask(mask, width, height, transform));
    }
    best
}

fn transform_mask(mask: u32, width: usize, height: usize, transform: usize) -> u32 {
    let mut transformed = 0_u32;
    for y in 0..height {
        for x in 0..width {
            let bit = y * width + x;
            if (mask & (1_u32 << bit)) == 0 {
                continue;
            }
            let (tx, ty) = transform_small_box_coord(x, y, width, height, transform);
            transformed |= 1_u32 << (ty * width + tx);
        }
    }
    transformed
}

fn transform_small_box_coord(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    transform: usize,
) -> (usize, usize) {
    let max_x = width - 1;
    let max_y = height - 1;
    match transform {
        0 => (x, y),
        1 => (max_x - x, y),
        2 => (x, max_y - y),
        3 => (max_x - x, max_y - y),
        4 => (y, x),
        5 => (max_y - y, x),
        6 => (y, max_x - x),
        7 => (max_y - y, max_x - x),
        _ => unreachable!(),
    }
}
