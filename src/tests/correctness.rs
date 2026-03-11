use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::bitgrid::BitGrid;
use crate::classify::{Classification, ClassificationLimits, classify_seed};
use crate::generators::{mix_seed, pattern_by_name, random_soup};
use crate::life::{GameOfLife, step_grid, step_grid_with_changes_and_memo};
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

pub(super) fn run_benchmark_report(as_json: bool) {
    let limits = ClassificationLimits {
        max_generations: 256,
        max_population: 20_000,
        max_bounding_box: 512,
    };
    let filters = benchmark_family_filter();
    let suite = weighted_benchmark_suite()
        .into_iter()
        .filter(|case| {
            filters
                .as_ref()
                .map(|set| set.contains(&case.family))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let mut report = BenchmarkReport::default();

    for case in suite {
        let expected = reference_classify(&case.grid, &limits);
        let started = Instant::now();
        let actual = classify_seed(&case.grid, &limits, &mut Memo::default());
        report.record(&case, &expected, &actual, started.elapsed());
    }

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report.to_json()).unwrap()
        );
    } else {
        report.print();
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

pub(super) fn bitmask_pattern(mask: u32, width: i32, height: i32) -> BitGrid {
    let mut cells = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let bit = (y * width + x) as u32;
            if (mask & (1_u32 << bit)) != 0 {
                cells.push((x, y));
            }
        }
    }
    BitGrid::from_cells(&cells)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BenchmarkFamily {
    IidRandom,
    StructuredRandom,
    ExhaustiveSmallBox,
    LongLivedMethuselah,
    TranslatedPeriodicMover,
    GunPufferBreeder,
    DelayedInteraction,
    DeceptiveAsh,
    ComputationalGadget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum OutcomeBucket {
    DiesOut,
    Stabilizes,
    Spaceship,
    InfiniteGrowth,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ErrorBucket {
    ExactOrCompatible,
    FalseDiesOut,
    FalseRepeats,
    FalseLikelyInfinite,
    Unknown,
    OtherMismatch,
}

#[derive(Clone, Debug)]
struct BenchmarkCase {
    name: String,
    family: BenchmarkFamily,
    size: i32,
    density_percent: u32,
    grid: BitGrid,
}

#[derive(Default)]
struct BenchmarkReport {
    total: usize,
    compatible: usize,
    by_family: BTreeMap<BenchmarkFamily, SliceStats>,
    by_density: BTreeMap<u32, SliceStats>,
    by_size: BTreeMap<i32, SliceStats>,
    by_outcome: BTreeMap<OutcomeBucket, SliceStats>,
    by_error: BTreeMap<ErrorBucket, usize>,
    runtimes: Vec<Duration>,
    runtime_by_family: BTreeMap<BenchmarkFamily, Vec<Duration>>,
    runtime_by_size: BTreeMap<i32, Vec<Duration>>,
    unknown_count: usize,
}

#[derive(Default, Clone, Debug)]
struct SliceStats {
    total: usize,
    compatible: usize,
}

#[derive(Serialize)]
struct JsonReport {
    total: usize,
    compatible: usize,
    unknown_rate: f64,
    runtime_overall: JsonRuntimeSummary,
    by_family: BTreeMap<String, JsonSliceStats>,
    by_density: BTreeMap<String, JsonSliceStats>,
    by_size: BTreeMap<String, JsonSliceStats>,
    by_outcome: BTreeMap<String, JsonSliceStats>,
    runtime_by_family: BTreeMap<String, JsonRuntimeSummary>,
    runtime_by_size: BTreeMap<String, JsonRuntimeSummary>,
    by_error: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct JsonSliceStats {
    total: usize,
    compatible: usize,
    accuracy: f64,
}

#[derive(Serialize)]
struct JsonRuntimeSummary {
    median_us: u128,
    p90_us: u128,
    p95_us: u128,
    p99_us: u128,
    worst_us: u128,
}

impl BenchmarkReport {
    fn record(
        &mut self,
        case: &BenchmarkCase,
        expected: &Classification,
        actual: &Classification,
        runtime: Duration,
    ) {
        let outcome = outcome_bucket(expected);
        let error = error_bucket(expected, actual);
        let compatible = error == ErrorBucket::ExactOrCompatible;

        self.total += 1;
        if compatible {
            self.compatible += 1;
        }
        record_slice(&mut self.by_family, case.family, compatible);
        record_slice(&mut self.by_density, case.density_percent, compatible);
        record_slice(&mut self.by_size, case.size, compatible);
        record_slice(&mut self.by_outcome, outcome, compatible);
        *self.by_error.entry(error).or_insert(0) += 1;
        self.runtimes.push(runtime);
        self.runtime_by_family
            .entry(case.family)
            .or_default()
            .push(runtime);
        self.runtime_by_size
            .entry(case.size)
            .or_default()
            .push(runtime);
        if matches!(actual, Classification::Unknown { .. }) {
            self.unknown_count += 1;
        }

        if !compatible {
            println!(
                "mismatch case={} family={:?} size={} density={} expected={} actual={}",
                case.name, case.family, case.size, case.density_percent, expected, actual
            );
        }
    }

    fn print(&self) {
        println!(
            "benchmark total={} compatible={} accuracy={:.3}",
            self.total,
            self.compatible,
            self.compatible as f64 / self.total.max(1) as f64
        );
        println!(
            "unknown_rate={}/{} ({:.3})",
            self.unknown_count,
            self.total,
            self.unknown_count as f64 / self.total.max(1) as f64
        );
        print_runtime_summary("runtime_overall", &self.runtimes);
        print_slice_map("family", &self.by_family);
        print_slice_map("density", &self.by_density);
        print_slice_map("size", &self.by_size);
        print_slice_map("outcome", &self.by_outcome);
        print_runtime_map("runtime_by_family", &self.runtime_by_family);
        print_runtime_map("runtime_by_size", &self.runtime_by_size);
        println!("error_buckets:");
        for (bucket, count) in &self.by_error {
            println!("  {:?}: {}", bucket, count);
        }
    }

    fn to_json(&self) -> JsonReport {
        JsonReport {
            total: self.total,
            compatible: self.compatible,
            unknown_rate: self.unknown_count as f64 / self.total.max(1) as f64,
            runtime_overall: runtime_summary_json(&self.runtimes),
            by_family: slice_map_json(&self.by_family),
            by_density: slice_map_json(&self.by_density),
            by_size: slice_map_json(&self.by_size),
            by_outcome: slice_map_json(&self.by_outcome),
            runtime_by_family: runtime_map_json(&self.runtime_by_family),
            runtime_by_size: runtime_map_json(&self.runtime_by_size),
            by_error: self
                .by_error
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
        }
    }
}

fn record_slice<K: Ord + Copy>(map: &mut BTreeMap<K, SliceStats>, key: K, compatible: bool) {
    let stats = map.entry(key).or_default();
    stats.total += 1;
    if compatible {
        stats.compatible += 1;
    }
}

fn print_slice_map<K: std::fmt::Debug + Ord>(label: &str, map: &BTreeMap<K, SliceStats>) {
    println!("{label}:");
    for (key, stats) in map {
        println!(
            "  {:?}: {}/{} ({:.3})",
            key,
            stats.compatible,
            stats.total,
            stats.compatible as f64 / stats.total.max(1) as f64
        );
    }
}

fn print_runtime_map<K: std::fmt::Debug + Ord>(label: &str, map: &BTreeMap<K, Vec<Duration>>) {
    println!("{label}:");
    for (key, samples) in map {
        print!("  {:?}: ", key);
        print_runtime_summary_inline(samples);
    }
}

fn print_runtime_summary(label: &str, samples: &[Duration]) {
    print!("{label}: ");
    print_runtime_summary_inline(samples);
}

fn print_runtime_summary_inline(samples: &[Duration]) {
    if samples.is_empty() {
        println!("no samples");
        return;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    println!(
        "median={:?} p90={:?} p95={:?} p99={:?} worst={:?}",
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.90),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        sorted[sorted.len() - 1]
    );
}

fn percentile(samples: &[Duration], fraction: f64) -> Duration {
    let idx = ((samples.len() - 1) as f64 * fraction).round() as usize;
    samples[idx]
}

fn runtime_summary_json(samples: &[Duration]) -> JsonRuntimeSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    JsonRuntimeSummary {
        median_us: percentile(&sorted, 0.50).as_micros(),
        p90_us: percentile(&sorted, 0.90).as_micros(),
        p95_us: percentile(&sorted, 0.95).as_micros(),
        p99_us: percentile(&sorted, 0.99).as_micros(),
        worst_us: sorted.last().copied().unwrap_or_default().as_micros(),
    }
}

fn slice_map_json<K: std::fmt::Debug + Ord>(
    map: &BTreeMap<K, SliceStats>,
) -> BTreeMap<String, JsonSliceStats> {
    map.iter()
        .map(|(key, stats)| {
            (
                format!("{key:?}"),
                JsonSliceStats {
                    total: stats.total,
                    compatible: stats.compatible,
                    accuracy: stats.compatible as f64 / stats.total.max(1) as f64,
                },
            )
        })
        .collect()
}

fn runtime_map_json<K: std::fmt::Debug + Ord>(
    map: &BTreeMap<K, Vec<Duration>>,
) -> BTreeMap<String, JsonRuntimeSummary> {
    map.iter()
        .map(|(key, samples)| (format!("{key:?}"), runtime_summary_json(samples)))
        .collect()
}

fn outcome_bucket(classification: &Classification) -> OutcomeBucket {
    match classification {
        Classification::DiesOut { .. } => OutcomeBucket::DiesOut,
        Classification::Repeats { .. } => OutcomeBucket::Stabilizes,
        Classification::Spaceship { .. } => OutcomeBucket::Spaceship,
        Classification::LikelyInfinite { .. } => OutcomeBucket::InfiniteGrowth,
        Classification::Unknown { .. } => OutcomeBucket::Unknown,
    }
}

fn error_bucket(expected: &Classification, actual: &Classification) -> ErrorBucket {
    match (expected, actual) {
        (Classification::DiesOut { .. }, Classification::DiesOut { .. }) => {
            ErrorBucket::ExactOrCompatible
        }
        (
            Classification::Repeats { period: ep, .. },
            Classification::Repeats { period: ap, .. },
        ) if ep == ap => ErrorBucket::ExactOrCompatible,
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
        ) if ep == ap && ed == ad => ErrorBucket::ExactOrCompatible,
        (Classification::LikelyInfinite { .. }, Classification::LikelyInfinite { .. }) => {
            ErrorBucket::ExactOrCompatible
        }
        (Classification::Unknown { .. }, Classification::Unknown { .. }) => {
            ErrorBucket::ExactOrCompatible
        }
        (Classification::Spaceship { .. }, Classification::DiesOut { .. }) => {
            ErrorBucket::FalseDiesOut
        }
        (Classification::LikelyInfinite { .. }, Classification::DiesOut { .. }) => {
            ErrorBucket::FalseDiesOut
        }
        (Classification::Repeats { .. }, Classification::DiesOut { .. }) => {
            ErrorBucket::FalseDiesOut
        }
        (_, Classification::Spaceship { .. }) => ErrorBucket::FalseLikelyInfinite,
        (_, Classification::Repeats { .. }) => ErrorBucket::FalseRepeats,
        (_, Classification::LikelyInfinite { .. }) => ErrorBucket::FalseLikelyInfinite,
        (_, Classification::Unknown { .. }) => ErrorBucket::Unknown,
        _ => ErrorBucket::OtherMismatch,
    }
}

fn weighted_benchmark_suite() -> Vec<BenchmarkCase> {
    let mut suite = Vec::new();
    suite.extend(iid_random_cases());
    suite.extend(structured_random_cases());
    suite.extend(exhaustive_small_box_cases());
    suite.extend(long_lived_methuselah_cases());
    suite.extend(translated_periodic_mover_cases());
    suite.extend(gun_puffer_breeder_cases());
    suite.extend(delayed_interaction_cases());
    suite.extend(deceptive_ash_cases());
    suite.extend(computational_gadget_cases());
    suite
}

fn benchmark_family_filter() -> Option<HashSet<BenchmarkFamily>> {
    let value = env::var("BENCH_FAMILIES").ok()?;
    let mut families = HashSet::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if let Some(family) = parse_family(token) {
            families.insert(family);
        }
    }
    Some(families)
}

fn parse_family(token: &str) -> Option<BenchmarkFamily> {
    match token {
        "iid" | "IidRandom" => Some(BenchmarkFamily::IidRandom),
        "structured" | "StructuredRandom" => Some(BenchmarkFamily::StructuredRandom),
        "smallbox" | "ExhaustiveSmallBox" => Some(BenchmarkFamily::ExhaustiveSmallBox),
        "methuselah" | "LongLivedMethuselah" => Some(BenchmarkFamily::LongLivedMethuselah),
        "mover" | "TranslatedPeriodicMover" => Some(BenchmarkFamily::TranslatedPeriodicMover),
        "gun" | "GunPufferBreeder" => Some(BenchmarkFamily::GunPufferBreeder),
        "delayed" | "DelayedInteraction" => Some(BenchmarkFamily::DelayedInteraction),
        "ash" | "DeceptiveAsh" => Some(BenchmarkFamily::DeceptiveAsh),
        "gadget" | "ComputationalGadget" => Some(BenchmarkFamily::ComputationalGadget),
        _ => None,
    }
}

fn iid_random_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for size in [16, 32, 64, 128] {
        for density in [5, 10, 20, 30, 50] {
            for seed in 1..=4_u64 {
                let grid = random_soup(size, size, density, hash_seed(size, density, seed));
                cases.push(BenchmarkCase {
                    name: format!("iid_{size}_{density}_{seed}"),
                    family: BenchmarkFamily::IidRandom,
                    size,
                    density_percent: density,
                    grid,
                });
            }
        }
    }
    cases
}

fn structured_random_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for size in [16, 32, 64, 128] {
        for seed in 1..=10_u64 {
            let grid = structured_random_soup(size, size, hash_seed(size, 77, seed));
            cases.push(BenchmarkCase {
                name: format!("structured_{size}_{seed}"),
                family: BenchmarkFamily::StructuredRandom,
                size,
                density_percent: estimate_density_percent(&grid),
                grid,
            });
        }
    }
    cases
}

fn exhaustive_small_box_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for mask in 1_u32..(1_u32 << 12) {
        let grid = bitmask_pattern(mask, 4, 3);
        cases.push(BenchmarkCase {
            name: format!("smallbox_{mask}"),
            family: BenchmarkFamily::ExhaustiveSmallBox,
            size: 4,
            density_percent: estimate_density_percent(&grid),
            grid,
        });
    }
    cases
}

fn long_lived_methuselah_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for name in ["acorn", "diehard", "r_pentomino"] {
        let base = pattern_by_name(name).unwrap();
        for (idx, grid) in translated_variants(&base).into_iter().enumerate() {
            cases.push(BenchmarkCase {
                name: format!("{name}_{idx}"),
                family: BenchmarkFamily::LongLivedMethuselah,
                size: grid
                    .bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
                density_percent: estimate_density_percent(&grid),
                grid,
            });
        }
    }
    cases
}

fn translated_periodic_mover_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for (name, base) in [
        ("glider", pattern_by_name("glider").unwrap()),
        ("blinker", pattern_by_name("blinker").unwrap()),
    ] {
        for (idx, grid) in translated_variants(&base).into_iter().enumerate() {
            cases.push(BenchmarkCase {
                name: format!("{name}_{idx}"),
                family: BenchmarkFamily::TranslatedPeriodicMover,
                size: grid
                    .bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
                density_percent: estimate_density_percent(&grid),
                grid,
            });
        }
    }
    cases
}

fn gun_puffer_breeder_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for (name, base) in [
        (
            "gosper_glider_gun",
            pattern_by_name("gosper_glider_gun").unwrap(),
        ),
        (
            "glider_producing_switch_engine",
            pattern_by_name("glider_producing_switch_engine").unwrap(),
        ),
        (
            "blinker_puffer_1",
            pattern_by_name("blinker_puffer_1").unwrap(),
        ),
    ] {
        for (idx, grid) in translated_variants(&base).into_iter().enumerate() {
            cases.push(BenchmarkCase {
                name: format!("{name}_{idx}"),
                family: BenchmarkFamily::GunPufferBreeder,
                size: grid
                    .bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
                density_percent: estimate_density_percent(&grid),
                grid,
            });
        }
    }
    cases
}

fn delayed_interaction_cases() -> Vec<BenchmarkCase> {
    let bases = vec![
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
            "late_collision",
            BitGrid::from_cells(&[
                (1, 0),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (40, 30),
                (41, 30),
                (42, 30),
            ]),
        ),
    ];
    benchmark_cases_from_bases(BenchmarkFamily::DelayedInteraction, bases)
}

fn deceptive_ash_cases() -> Vec<BenchmarkCase> {
    let bases = vec![
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
        ("block", pattern_by_name("block").unwrap()),
    ];
    benchmark_cases_from_bases(BenchmarkFamily::DeceptiveAsh, bases)
}

fn computational_gadget_cases() -> Vec<BenchmarkCase> {
    let bases = vec![(
        "glider_logic_probe",
        BitGrid::from_cells(&[
            (1, 0),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (20, 5),
            (21, 5),
            (22, 5),
            (22, 4),
            (21, 3),
        ]),
    )];
    benchmark_cases_from_bases(BenchmarkFamily::ComputationalGadget, bases)
}

fn benchmark_cases_from_bases(
    family: BenchmarkFamily,
    bases: Vec<(&str, BitGrid)>,
) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for (name, base) in bases {
        for (idx, grid) in translated_variants(&base).into_iter().enumerate() {
            cases.push(BenchmarkCase {
                name: format!("{name}_{idx}"),
                family,
                size: grid
                    .bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
                density_percent: estimate_density_percent(&grid),
                grid,
            });
        }
    }
    cases
}

fn translated_variants(grid: &BitGrid) -> Vec<BitGrid> {
    let offsets = [(0, 0), (5, 7), (12, 3), (20, 15)];
    offsets
        .into_iter()
        .map(|(dx, dy)| {
            let cells = grid
                .live_cells()
                .into_iter()
                .map(|(x, y)| (x + dx, y + dy))
                .collect::<Vec<_>>();
            BitGrid::from_cells(&cells)
        })
        .collect()
}

fn estimate_density_percent(grid: &BitGrid) -> u32 {
    let Some((min_x, min_y, max_x, max_y)) = grid.bounds() else {
        return 0;
    };
    let area = ((max_x - min_x + 1) * (max_y - min_y + 1)).max(1) as usize;
    ((grid.population() * 100) / area) as u32
}
