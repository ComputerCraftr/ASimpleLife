use std::collections::{HashMap, HashSet, VecDeque};

use crate::bitgrid::BitGrid;
use crate::classify::{Classification, ClassificationLimits, classify_seed};
use crate::generators::{mix_seed, pattern_by_name, random_soup};
use crate::life::{ChunkDiff, GameOfLife, step_grid, step_grid_with_changes_and_memo};
use crate::memo::Memo;
use crate::normalize::normalize;
use crate::render::{TerminalBackbuffer, compute_origin_for_cells};

#[test]
fn block_repeats_immediately() {
    let grid = pattern_by_name("block").unwrap();
    let result = classify_seed(
        &grid,
        &ClassificationLimits::default(),
        &mut Memo::default(),
    );
    assert_eq!(
        result,
        Classification::Repeats {
            period: 1,
            first_seen: 0
        }
    );
}

#[test]
fn blinker_has_period_two() {
    let grid = pattern_by_name("blinker").unwrap();
    let result = classify_seed(
        &grid,
        &ClassificationLimits::default(),
        &mut Memo::default(),
    );
    assert_eq!(
        result,
        Classification::Repeats {
            period: 2,
            first_seen: 0
        }
    );
}

#[test]
fn glider_is_detected_as_spaceship() {
    let grid = pattern_by_name("glider").unwrap();
    let result = classify_seed(
        &grid,
        &ClassificationLimits::default(),
        &mut Memo::default(),
    );
    assert_eq!(
        result,
        Classification::Spaceship {
            period: 4,
            first_seen: 0,
            delta: (1, 1),
            detected_at: 4,
        }
    );
}

#[test]
fn gosper_glider_gun_is_detected_as_likely_infinite() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let limits = ClassificationLimits {
        max_generations: 512,
        max_population: 5_000,
        max_bounding_box: 256,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert!(matches!(
        result,
        Classification::LikelyInfinite {
            reason: "population_growth" | "expanding_bounds" | "persistent_expansion",
            ..
        }
    ));
}

#[test]
fn gosper_glider_gun_preserves_gun_core_and_emits_glider() {
    let initial = pattern_by_name("gosper_glider_gun").unwrap();
    let core_cycle = run_steps(initial.clone(), 30);
    let emitted_field = run_steps(initial.clone(), 120);
    let initial_core = crop_grid(&initial, 0, 0, 36, 9);
    let evolved_core = crop_grid(&core_cycle, 0, 0, 36, 9);
    let glider_field = crop_grid(&emitted_field, 37, 0, 260, 260);

    assert_eq!(normalize(&initial_core).0, normalize(&evolved_core).0);
    assert!(contains_component_variant(
        &glider_field,
        &all_evolution_variants(&pattern_by_name("glider").unwrap(), 4)
    ));
}

#[test]
fn gosper_puffer_is_detected_as_likely_infinite() {
    let grid = pattern_by_name("glider_producing_switch_engine").unwrap();
    let limits = ClassificationLimits {
        max_generations: 512,
        max_population: 10_000,
        max_bounding_box: 512,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert!(matches!(
        result,
        Classification::LikelyInfinite {
            reason: "population_growth" | "expanding_bounds" | "persistent_expansion",
            ..
        }
    ));
}

#[test]
fn glider_puffer_seed_matches_known_population_and_bounds() {
    let grid = pattern_by_name("glider_producing_switch_engine").unwrap();
    let (min_x, min_y, max_x, max_y) = grid.bounds().unwrap();
    assert_eq!(grid.population(), 123);
    assert_eq!((min_x, min_y), (0, 0));
    assert_eq!((max_x - min_x + 1, max_y - min_y + 1), (67, 60));
}

#[test]
fn glider_puffer_emits_a_glider_after_simulation() {
    let grid = run_steps(
        pattern_by_name("glider_producing_switch_engine").unwrap(),
        256,
    );
    assert!(contains_component_variant(
        &grid,
        &all_evolution_variants(&pattern_by_name("glider").unwrap(), 4)
    ));
}

#[test]
fn blinker_puffer1_is_detected_as_likely_infinite() {
    let grid = pattern_by_name("blinker_puffer_1").unwrap();
    let limits = ClassificationLimits {
        max_generations: 512,
        max_population: 10_000,
        max_bounding_box: 512,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert!(matches!(
        result,
        Classification::LikelyInfinite {
            reason: "population_growth" | "expanding_bounds" | "persistent_expansion",
            ..
        }
    ));
}

#[test]
fn blinker_puffer1_leaves_a_blinker_after_simulation() {
    let grid = run_steps(pattern_by_name("blinker_puffer_1").unwrap(), 160);
    assert!(contains_component_variant(
        &grid,
        &all_normalized_variants(&pattern_by_name("blinker").unwrap())
    ));
}

#[test]
fn diehard_eventually_dies() {
    let grid = pattern_by_name("diehard").unwrap();
    let limits = ClassificationLimits {
        max_generations: 200,
        ..ClassificationLimits::default()
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_eq!(result, Classification::DiesOut { at_generation: 130 });
}

#[test]
fn diehard_stops_at_extinction_before_large_generation_limit() {
    let grid = pattern_by_name("diehard").unwrap();
    let limits = ClassificationLimits {
        max_generations: 10_000,
        ..ClassificationLimits::default()
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_eq!(result, Classification::DiesOut { at_generation: 130 });
}

#[test]
fn rpentomino_survives_short_horizon() {
    let grid = pattern_by_name("r_pentomino").unwrap();
    let limits = ClassificationLimits {
        max_generations: 100,
        ..ClassificationLimits::default()
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_eq!(result, Classification::Unknown { simulated: 100 });
}

#[test]
fn bounded_iid_soup_reaches_repeat_before_extended_limit() {
    let seed = mix_seed(
        ((16_u64) ^ ((30_u64) << 16) ^ (2_u64 << 32)).wrapping_add(0x9E3779B97F4A7C15),
    );
    let grid = random_soup(16, 16, 30, seed);
    let limits = ClassificationLimits {
        max_generations: 256,
        max_population: 20_000,
        max_bounding_box: i32::MAX,
    };

    let result = classify_seed(&grid, &limits, &mut Memo::default());

    assert_eq!(
        result,
        Classification::Repeats {
            period: 1,
            first_seen: 354,
        }
    );
}

#[test]
fn block_stops_at_repeat_before_large_generation_limit() {
    let grid = pattern_by_name("block").unwrap();
    let limits = ClassificationLimits {
        max_generations: 10_000,
        ..ClassificationLimits::default()
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_eq!(
        result,
        Classification::Repeats {
            period: 1,
            first_seen: 0,
        }
    );
}

#[test]
fn normalization_ignores_translation() {
    let base = BitGrid::from_cells(&[(2, 3), (3, 3), (4, 3)]);
    let shifted = BitGrid::from_cells(&[(20, -5), (21, -5), (22, -5)]);
    let (a, _) = normalize(&base);
    let (b, _) = normalize(&shifted);
    assert_eq!(a, b);
}

#[test]
fn normalization_anchors_cells_at_zero_zero() {
    let grid = BitGrid::from_cells(&[(5, 7), (6, 7), (5, 8)]);
    let (normalized, origin) = normalize(&grid);
    assert_eq!(origin, (5, 7));
    assert_eq!(normalized.cells.first().copied(), Some((0, 0)));
    assert!(normalized.cells.iter().all(|&(x, y)| x >= 0 && y >= 0));
}

#[test]
fn chunked_storage_handles_negative_coordinates() {
    let grid = BitGrid::from_cells(&[(-1, -1), (-8, -8), (-9, -9), (7, 7), (8, 8)]);
    for &(x, y) in &[(-1, -1), (-8, -8), (-9, -9), (7, 7), (8, 8)] {
        assert!(grid.get(x, y));
    }
}

#[test]
fn half_block_renderer_uses_vertical_pairing() {
    let grid = BitGrid::from_cells(&[(0, 0), (0, 1), (1, 0)]);
    let mut buffer = TerminalBackbuffer::new(2, 1);
    let frame = render_output(&mut buffer, &grid, None);
    assert!(frame.contains('█'));
    assert!(frame.contains('▀'));
}

#[test]
fn render_diff_only_emits_changed_cells() {
    let mut buffer = TerminalBackbuffer::new(4, 2);
    let initial = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 1)]);
    let updated = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 0)]);

    let full = render_output(&mut buffer, &initial, None);
    assert!(full.contains("\x1b[2;1H"));

    let diff = render_output(&mut buffer, &updated, Some(&[(1, 0), (1, 1)]));
    assert!(diff.contains("\x1b[2;2H▀"));
    assert!(!diff.contains("\x1b[2;1H"));
}

#[test]
fn render_chunk_diff_only_emits_changed_region() {
    let mut buffer = TerminalBackbuffer::new(4, 2);
    let initial = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 1)]);
    let updated = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 0)]);

    let mut out = Vec::new();
    buffer.render_chunk_into(&initial, None, &mut out).unwrap();

    let mut diff_out = Vec::new();
    buffer
        .render_chunk_into(
            &updated,
            Some(&[ChunkDiff {
                cx: 0,
                cy: 0,
                diff_bits: (1_u64 << 1) | (1_u64 << 9),
            }]),
            &mut diff_out,
        )
        .unwrap();
    let diff = String::from_utf8(diff_out).unwrap();
    assert!(diff.contains("\x1b[2;2H▀"));
    assert!(!diff.contains("\x1b[2;1H"));
}

#[test]
fn viewport_biases_toward_denser_cluster() {
    let cells = vec![(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (40, 10)];
    let origin = compute_origin_for_cells(10, 4, &cells);
    assert_eq!(origin.0, 0);
    assert_eq!(origin.1, 0);
}

#[test]
fn step_engine_matches_blinker_rotation() {
    let grid = pattern_by_name("blinker").unwrap();
    let next = step_grid(&grid);
    let expected = BitGrid::from_cells(&[(1, -1), (1, 0), (1, 1)]);
    assert_eq!(normalize(&next).0, normalize(&expected).0);
}

#[test]
fn chunk_transition_cache_reuses_local_neighborhoods() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let mut memo = Memo::default();

    let _ = step_grid_with_changes_and_memo(&grid, &mut memo);
    let after_first = memo.chunk_transition_cache_len();
    let _ = step_grid_with_changes_and_memo(&grid, &mut memo);
    let after_second = memo.chunk_transition_cache_len();

    assert!(after_first > 0);
    assert_eq!(after_first, after_second);
}

#[test]
fn stratified_reference_suite_matches_known_labels() {
    let limits = ClassificationLimits {
        max_generations: 256,
        max_population: 10_000,
        max_bounding_box: 512,
    };
    for case in curated_reference_suite() {
        assert_matches_reference_if_decisive(&case.name, &case.grid, &limits);
    }
}

#[test]
fn stratified_random_soups_match_reference_when_reference_is_decisive() {
    let limits = ClassificationLimits {
        max_generations: 192,
        max_population: 20_000,
        max_bounding_box: 512,
    };
    for size in [16, 32, 64] {
        for fill_percent in [5, 10, 20, 30, 50] {
            for seed in 1..=4_u64 {
                let grid = random_soup(
                    size,
                    size,
                    fill_percent,
                    hash_seed(size, fill_percent, seed),
                );
                assert_matches_reference_if_decisive(
                    &format!("random_{size}_{fill_percent}_{seed}"),
                    &grid,
                    &limits,
                );
            }
        }
    }
}

#[test]
fn stratified_clustered_soups_match_reference_when_reference_is_decisive() {
    let limits = ClassificationLimits {
        max_generations: 192,
        max_population: 20_000,
        max_bounding_box: 512,
    };
    for size in [16, 32, 64] {
        for fill_percent in [5, 10, 20, 30, 50] {
            for seed in 1..=4_u64 {
                let grid = clustered_noise_soup(
                    size,
                    size,
                    fill_percent,
                    hash_seed(size, fill_percent, seed),
                );
                assert_matches_reference_if_decisive(
                    &format!("clustered_{size}_{fill_percent}_{seed}"),
                    &grid,
                    &limits,
                );
            }
        }
    }
}

#[test]
fn structured_random_soups_match_reference_when_reference_is_decisive() {
    let limits = ClassificationLimits {
        max_generations: 192,
        max_population: 20_000,
        max_bounding_box: 512,
    };
    for size in [16, 32, 64] {
        for seed in 1..=4_u64 {
            let grid = structured_random_soup(size, size, hash_seed(size, 77, seed));
            assert_matches_reference_if_decisive(
                &format!("structured_{size}_{seed}"),
                &grid,
                &limits,
            );
        }
    }
}

fn run_steps(grid: BitGrid, steps: usize) -> BitGrid {
    let mut game = GameOfLife::new(grid);
    for _ in 0..steps {
        game.step_with_changes();
    }
    game.grid().clone()
}

fn crop_grid(grid: &BitGrid, min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> BitGrid {
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|&(x, y)| x >= min_x && x <= max_x && y >= min_y && y <= max_y)
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn contains_component_variant(grid: &BitGrid, variants: &[Vec<(i32, i32)>]) -> bool {
    connected_components(grid)
        .into_iter()
        .map(|component| normalize(&BitGrid::from_cells(&component)).0.cells)
        .any(|normalized| variants.iter().any(|variant| variant == &normalized))
}

fn all_normalized_variants(pattern: &BitGrid) -> Vec<Vec<(i32, i32)>> {
    all_evolution_variants(pattern, 1)
}

fn all_evolution_variants(pattern: &BitGrid, period: usize) -> Vec<Vec<(i32, i32)>> {
    let mut variants = Vec::new();
    let mut phase = pattern.clone();

    for _ in 0..period {
        append_symmetry_variants(&mut variants, &phase.live_cells());
        phase = step_grid(&phase);
    }

    variants
}

fn append_symmetry_variants(variants: &mut Vec<Vec<(i32, i32)>>, cells: &[(i32, i32)]) {
    let transforms: [fn(i32, i32) -> (i32, i32); 8] = [
        |x, y| (x, y),
        |x, y| (x, -y),
        |x, y| (-x, y),
        |x, y| (-x, -y),
        |x, y| (y, x),
        |x, y| (y, -x),
        |x, y| (-y, x),
        |x, y| (-y, -x),
    ];

    for transform in transforms {
        let transformed = cells
            .iter()
            .map(|&(x, y)| transform(x, y))
            .collect::<Vec<_>>();
        let normalized = normalize(&BitGrid::from_cells(&transformed)).0.cells;
        if !variants.iter().any(|existing| existing == &normalized) {
            variants.push(normalized);
        }
    }
}

fn connected_components(grid: &BitGrid) -> Vec<Vec<(i32, i32)>> {
    let mut remaining = grid.live_cells().into_iter().collect::<HashSet<_>>();
    let mut components = Vec::new();

    while let Some(&start) = remaining.iter().next() {
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        remaining.remove(&start);

        while let Some((x, y)) = queue.pop_front() {
            component.push((x, y));
            for ny in (y - 1)..=(y + 1) {
                for nx in (x - 1)..=(x + 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    if remaining.remove(&(nx, ny)) {
                        queue.push_back((nx, ny));
                    }
                }
            }
        }

        components.push(component);
    }

    components
}

fn render_output(
    buffer: &mut TerminalBackbuffer,
    grid: &BitGrid,
    changed_cells: Option<&[(i32, i32)]>,
) -> String {
    let mut out = Vec::new();
    buffer.render_into(grid, changed_cells, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

fn assert_matches_reference_if_decisive(name: &str, grid: &BitGrid, limits: &ClassificationLimits) {
    let expected = reference_classify(grid, limits);
    if !reference_is_decisive(&expected) {
        return;
    }
    let actual = classify_seed(grid, limits, &mut Memo::default());
    assert_same_outcome(name, &expected, &actual);
}

pub(super) fn assert_same_outcome(name: &str, expected: &Classification, actual: &Classification) {
    match (expected, actual) {
        (Classification::DiesOut { .. }, Classification::DiesOut { .. }) => {}
        (
            Classification::Repeats { period: ep, .. },
            Classification::Repeats { period: ap, .. },
        ) => {
            assert_eq!(
                ep, ap,
                "{name}: repeat period mismatch: expected {expected}, got {actual}"
            );
        }
        (
            Classification::Spaceship {
                period: ep,
                delta: ed,
                ..
            },
            Classification::Spaceship {
                period: ap,
                delta: ad,
                ..
            },
        ) => {
            assert_eq!(
                ep, ap,
                "{name}: spaceship period mismatch: expected {expected}, got {actual}"
            );
            assert_eq!(
                ed, ad,
                "{name}: spaceship displacement mismatch: expected {expected}, got {actual}"
            );
        }
        (Classification::Unknown { .. }, Classification::Unknown { .. }) => {}
        (Classification::LikelyInfinite { .. }, Classification::LikelyInfinite { .. }) => {}
        _ => panic!("{name}: expected {expected}, got {actual}"),
    }
}

pub(super) fn reference_is_decisive(classification: &Classification) -> bool {
    !matches!(classification, Classification::Unknown { .. })
}

fn curated_reference_suite() -> Vec<NamedCase> {
    let mut cases = Vec::new();
    for name in [
        "block",
        "blinker",
        "glider",
        "diehard",
        "r_pentomino",
        "gosper_glider_gun",
        "glider_producing_switch_engine",
        "blinker_puffer_1",
    ] {
        cases.push(NamedCase {
            name: name.to_string(),
            grid: pattern_by_name(name).unwrap(),
        });
    }

    let adversarial = [
        ("acorn_offset", pattern_by_name("acorn").unwrap().clone()),
        (
            "double_glider",
            BitGrid::from_cells(&[
                (1, 0),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (11, 10),
                (12, 11),
                (10, 12),
                (11, 12),
                (12, 12),
            ]),
        ),
        (
            "traffic_jam",
            BitGrid::from_cells(&[
                (0, 1),
                (1, 1),
                (2, 1),
                (1, 0),
                (1, 2),
                (6, 1),
                (7, 1),
                (8, 1),
                (7, 0),
                (7, 2),
            ]),
        ),
    ];

    for (name, grid) in adversarial {
        cases.push(NamedCase {
            name: name.to_string(),
            grid,
        });
    }

    cases
}

fn clustered_noise_soup(width: i32, height: i32, fill_percent: u32, seed: u64) -> BitGrid {
    let base = random_soup(width, height, fill_percent, seed);
    let mut cells = Vec::new();
    for (x, y) in base.live_cells() {
        cells.push((x, y));
        if ((x + y).unsigned_abs() + (seed as u32)) % 3 == 0 && x + 1 < width {
            cells.push((x + 1, y));
        }
        if ((x * 3 + y * 5).unsigned_abs() + (seed as u32)) % 5 == 0 && y + 1 < height {
            cells.push((x, y + 1));
        }
    }
    BitGrid::from_cells(&cells)
}

fn structured_random_soup(width: i32, height: i32, seed: u64) -> BitGrid {
    let left = random_soup(width / 2, height, 18, seed);
    let right = random_soup(width / 2, height, 12, seed ^ 0x9E3779B97F4A7C15);
    let mut cells = left.live_cells();
    cells.extend(
        right
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + (width / 2), y)),
    );
    cells.extend(
        pattern_by_name("blinker")
            .unwrap()
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + width / 3, y + height / 3)),
    );
    cells.extend(
        pattern_by_name("block")
            .unwrap()
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + width / 2, y + height / 2)),
    );
    BitGrid::from_cells(&cells)
}

fn hash_seed(a: i32, b: u32, c: u64) -> u64 {
    mix_seed(((a as u64) ^ ((b as u64) << 16) ^ (c << 32)).wrapping_add(0x9E3779B97F4A7C15))
}

pub(super) fn reference_classify(seed: &BitGrid, limits: &ClassificationLimits) -> Classification {
    let mut seen: HashMap<Vec<(i32, i32)>, (usize, (i32, i32))> = HashMap::new();
    let mut grid = seed.clone();

    for generation in 0..=limits.max_generations {
        let (signature, origin) = normalize(&grid);
        if grid.is_empty() {
            return Classification::DiesOut {
                at_generation: generation,
            };
        }

        if let Some(&(first_seen, first_origin)) = seen.get(&signature.cells) {
            let period = generation - first_seen;
            let dx = origin.0 - first_origin.0;
            let dy = origin.1 - first_origin.1;
            return if dx == 0 && dy == 0 {
                Classification::Repeats { period, first_seen }
            } else {
                Classification::Spaceship {
                    period,
                    first_seen,
                    delta: (dx, dy),
                    detected_at: generation,
                }
            };
        }

        if grid.population() > limits.max_population {
            return Classification::LikelyInfinite {
                reason: "population_growth",
                detected_at: generation,
            };
        }

        if let Some((min_x, min_y, max_x, max_y)) = grid.bounds() {
            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;
            if width > limits.max_bounding_box || height > limits.max_bounding_box {
                return Classification::LikelyInfinite {
                    reason: "expanding_bounds",
                    detected_at: generation,
                };
            }
        }

        seen.insert(signature.cells, (generation, origin));
        grid = reference_step_grid(&grid);
    }

    Classification::Unknown {
        simulated: limits.max_generations,
    }
}

fn reference_step_grid(grid: &BitGrid) -> BitGrid {
    let mut counts: HashMap<(i32, i32), u8> = HashMap::new();
    for (x, y) in grid.live_cells() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                *counts.entry((x + dx, y + dy)).or_insert(0) += 1;
            }
        }
    }

    let mut next = BitGrid::new();
    for ((x, y), count) in counts {
        if count == 3 || (count == 2 && grid.get(x, y)) {
            next.set(x, y, true);
        }
    }
    next
}

struct NamedCase {
    name: String,
    grid: BitGrid,
}
