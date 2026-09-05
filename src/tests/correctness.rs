use crate::RequiredExt;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::classify::{Classification, ClassificationLimits, classify_seed};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashing::{derive_seed, hash_words};
use crate::life::{ChunkDiff, GameOfLife, step_grid, step_grid_with_changes_and_memo};
use crate::memo::Memo;
use crate::normalize::normalize;
use crate::render::{
    TerminalBackbuffer, compute_origin_for_bounds, compute_origin_for_cells,
    resized_viewport_origin, stable_viewport_origin,
};

fn assert_grids_eq(label: &str, actual: &BitGrid, expected: &BitGrid) {
    assert_eq!(
        actual,
        expected,
        "{label}: absolute cell states differ\nactual_bounds={:?} expected_bounds={:?}\nactual={actual:?}\nexpected={expected:?}",
        actual.bounds(),
        expected.bounds(),
    );
}

#[test]
fn block_repeats_immediately() {
    let grid = pattern_by_name("block").or_invariant("required value");
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
    let grid = pattern_by_name("blinker").or_invariant("required value");
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
fn pulsar_has_period_three() {
    let grid = pattern_by_name("pulsar").or_invariant("required value");
    assert_eq!(
        grid.population(),
        48,
        "pulsar fixture should be the canonical 48-cell pattern"
    );
    let result = classify_seed(
        &grid,
        &ClassificationLimits::default(),
        &mut Memo::default(),
    );
    assert_eq!(
        result,
        Classification::Repeats {
            period: 3,
            first_seen: 0
        }
    );
    let evolved = run_steps(grid.clone(), 3);
    assert_grids_eq(
        "pulsar should return to its initial phase after 3 steps",
        &evolved,
        &grid,
    );
}

#[test]
fn glider_is_detected_as_spaceship() {
    let grid = pattern_by_name("glider").or_invariant("required value");
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
fn gosper_glider_gun_is_classified_as_persistently_expanding() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 512,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_persistent_expansion("gosper glider gun", &result, limits.max_generations);
}

#[test]
fn gosper_glider_gun_preserves_gun_core_and_emits_glider() {
    let initial = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let core_cycle = run_steps(initial.clone(), 30);
    let emitted_field = run_steps(initial.clone(), 120);
    let initial_core = crop_grid(&initial, 0, 0, 36, 9);
    let evolved_core = crop_grid(&core_cycle, 0, 0, 36, 9);
    let glider_field = crop_grid(&emitted_field, 37, 0, 260, 260);

    assert_grids_eq(
        "gosper gun core should remain stable across one gun period",
        &initial_core,
        &evolved_core,
    );
    assert!(contains_component_variant(
        &glider_field,
        &all_evolution_variants(&pattern_by_name("glider").or_invariant("required value"), 4)
    ));
}

#[test]
fn glider_producing_switch_engine_is_classified_as_persistently_expanding() {
    let grid = pattern_by_name("glider_producing_switch_engine").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 512,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_persistent_expansion(
        "glider-producing switch engine",
        &result,
        limits.max_generations,
    );
}

#[test]
fn glider_puffer_seed_matches_known_population_and_bounds() {
    let grid = pattern_by_name("glider_producing_switch_engine").or_invariant("required value");
    let (min_x, min_y, max_x, max_y) = grid.bounds().or_invariant("required value");
    assert_eq!(grid.population(), 123);
    assert_eq!((min_x, min_y), (0, 0));
    assert_eq!((max_x - min_x + 1, max_y - min_y + 1), (67, 60));
}

#[test]
fn glider_puffer_emits_a_glider_after_simulation() {
    let grid = run_steps(
        pattern_by_name("glider_producing_switch_engine").or_invariant("required value"),
        256,
    );
    assert!(contains_component_variant(
        &grid,
        &all_evolution_variants(&pattern_by_name("glider").or_invariant("required value"), 4)
    ));
}

#[test]
fn blinker_puffer1_is_classified_as_persistently_expanding() {
    let grid = pattern_by_name("blinker_puffer_1").or_invariant("required value");
    let (min_x, min_y, max_x, max_y) = grid.bounds().or_invariant("required value");
    assert_eq!(grid.population(), 37);
    assert_eq!((min_x, min_y), (0, 0));
    assert_eq!((max_x - min_x + 1, max_y - min_y + 1), (9, 18));
    let limits = ClassificationLimits {
        max_generations: 512,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_persistent_expansion("blinker puffer 1", &result, limits.max_generations);
}

#[test]
fn blinker_puffer1_leaves_a_blinker_after_simulation() {
    let grid = run_steps(
        pattern_by_name("blinker_puffer_1").or_invariant("required value"),
        160,
    );
    assert!(contains_component_variant(
        &grid,
        &all_normalized_variants(&pattern_by_name("blinker").or_invariant("required value"))
    ));
}

#[test]
fn diehard_eventually_dies() {
    let grid = pattern_by_name("diehard").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 200,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_eq!(result, Classification::DiesOut { at_generation: 130 });
}

#[test]
fn diehard_stops_at_extinction_before_large_generation_limit() {
    let grid = pattern_by_name("diehard").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 10_000,
    };
    let result = classify_seed(&grid, &limits, &mut Memo::default());
    assert_eq!(result, Classification::DiesOut { at_generation: 130 });
}

#[test]
fn rpentomino_survives_short_horizon() {
    let grid = pattern_by_name("r_pentomino").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 100,
    };
    assert_matches_reference_contract("r-pentomino short horizon", &grid, &limits);
}

#[test]
fn bounded_iid_soup_reaches_repeat_before_extended_limit() {
    const REGRESSION_SEED: u64 = 8_869_397_597_862_540_459;
    let seed = REGRESSION_SEED;
    let grid = random_soup(16, 16, 30, seed);
    let limits = ClassificationLimits {
        max_generations: 256,
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
    let grid = pattern_by_name("block").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 10_000,
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
    assert!(
        frame.contains('█'),
        "half-block renderer should emit a full block for vertically paired live cells\nframe={frame:?}"
    );
    assert!(
        frame.contains('▀'),
        "half-block renderer should emit an upper half block for mixed vertical occupancy\nframe={frame:?}"
    );
}

#[test]
fn render_diff_only_emits_changed_cells() {
    let mut buffer = TerminalBackbuffer::new(4, 2);
    let initial = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 1)]);
    let updated = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 0)]);

    let full = render_output(&mut buffer, &initial, None);
    assert!(
        full.contains("\x1b[2;1H"),
        "initial full render should reposition the cursor to the first visible cell\nfull={full:?}"
    );

    let diff = render_output(&mut buffer, &updated, Some(&[(1, 0), (1, 1)]));
    let origin = compute_origin_for_cells(4, 2, &updated.live_cells());
    let expected_row =
        usize::try_from((0 - origin.1).div_euclid(2)).or_invariant("required value") + 2;
    let expected_col = usize::try_from(1 - origin.0).or_invariant("required value") + 1;
    let expected_cursor = format!("\x1b[{expected_row};{expected_col}H▀");
    assert!(
        diff.contains(&expected_cursor),
        "diff render should emit only the changed cell cursor sequence\norigin={origin:?}\nexpected_cursor={expected_cursor:?}\ndiff={diff:?}"
    );
    assert!(
        !diff.contains("\x1b[2;1H"),
        "diff render should not emit a full-frame reset cursor sequence\ndiff={diff:?}"
    );
}

#[test]
fn render_chunk_diff_only_emits_changed_region() {
    let mut buffer = TerminalBackbuffer::new(4, 2);
    let initial = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 1)]);
    let updated = BitGrid::from_cells(&[(0, 0), (3, 3), (1, 0)]);

    let mut out = Vec::new();
    buffer
        .render_chunk_into(&initial, None, &mut out)
        .or_invariant("required value");

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
        .or_invariant("required value");
    let diff = String::from_utf8(diff_out).or_invariant("required value");
    let origin = compute_origin_for_cells(4, 2, &updated.live_cells());
    let expected_row =
        usize::try_from((0 - origin.1).div_euclid(2)).or_invariant("required value") + 2;
    let expected_col = usize::try_from(1 - origin.0).or_invariant("required value") + 1;
    let expected_cursor = format!("\x1b[{expected_row};{expected_col}H▀");
    assert!(
        diff.contains(&expected_cursor),
        "chunk diff render should emit only the changed chunk cursor sequence\norigin={origin:?}\nexpected_cursor={expected_cursor:?}\ndiff={diff:?}"
    );
    assert!(
        !diff.contains("\x1b[2;1H"),
        "chunk diff render should not emit a full-frame reset cursor sequence\ndiff={diff:?}"
    );
}

#[test]
fn viewport_biases_toward_denser_cluster() {
    let dense_cluster = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)];
    let sparse_cluster = [(40, 10)];
    let cells = dense_cluster
        .into_iter()
        .chain(sparse_cluster)
        .collect::<Vec<_>>();
    let origin = compute_origin_for_cells(10, 4, &cells);
    let expected = compute_origin_for_bounds(10, 4, (0, 0, 2, 1));
    assert_viewport_contains_live_cell(origin, 10, 4, &cells);
    assert_eq!(
        origin, expected,
        "viewport should focus the larger connected mass, not a distant sparse outlier"
    );
}

#[test]
fn viewport_avoids_empty_midpoint_for_split_equal_movers() {
    let left_glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let right_glider = [(40, 0), (41, 0), (42, 0), (40, 1), (41, 2)];
    let cells = left_glider
        .into_iter()
        .chain(right_glider)
        .collect::<Vec<_>>();

    let origin = compute_origin_for_cells(10, 4, &cells);
    assert_viewport_contains_live_cell(origin, 10, 4, &cells);
    let left_origin = compute_origin_for_bounds(10, 4, (0, 0, 2, 2));
    let right_origin = compute_origin_for_bounds(10, 4, (40, 0, 42, 2));
    let empty_midpoint_origin = compute_origin_for_bounds(10, 4, (0, 0, 42, 2));
    assert!(
        origin == left_origin || origin == right_origin,
        "equal-size split movers should lock onto one mover cluster, origin={origin:?} left={left_origin:?} right={right_origin:?}"
    );
    assert_ne!(
        origin, empty_midpoint_origin,
        "equal-size split movers should not center on the empty midpoint"
    );
}

#[test]
fn pulsar_viewport_stays_centered_across_phases_when_pattern_fits() {
    let initial = pattern_by_name("pulsar").or_invariant("required value");
    let phase_two = run_steps(initial.clone(), 1);

    let initial_origin = compute_origin_for_cells(20, 10, &initial.live_cells());
    let phase_two_origin = compute_origin_for_cells(20, 10, &phase_two.live_cells());
    let expected_origin = compute_origin_for_bounds(20, 10, (0, 0, 12, 12));

    assert_eq!(initial_origin, expected_origin);
    assert_eq!(phase_two_origin, expected_origin);
}

#[test]
fn distant_equal_viewport_candidate_does_not_replace_current_focus() {
    let current = (0, 0);
    let proposed = (1_000, 0);
    assert_eq!(
        stable_viewport_origin(Some(current), proposed, 100, 101, 80, 24),
        current,
        "a negligible population difference should not switch between distant life forms"
    );
}

#[test]
fn distant_population_lead_does_not_replace_an_occupied_focus() {
    assert_eq!(
        stable_viewport_origin(Some((0, 0)), (1_000, 0), 60, 100, 80, 24),
        (0, 0),
        "a population lead can be an oscillator phase, not loss of the current focus"
    );
}

#[test]
fn empty_viewport_reacquires_a_visible_candidate() {
    assert_eq!(
        stable_viewport_origin(Some((0, 0)), (1_000, 0), 0, 5, 80, 24),
        (1_000, 0)
    );
}

#[test]
fn resized_viewport_origin_preserves_world_space_center() {
    assert_eq!(
        resized_viewport_origin(Some((100, 200)), 80, 24, 120, 40),
        Some((80, 184))
    );
}

#[test]
fn resized_terminal_backbuffer_rebuilds_even_with_chunk_diffs() {
    let grid = BitGrid::from_cells(&[(0, 0), (1, 1)]);
    let mut renderer = TerminalBackbuffer::new(8, 4);
    let mut initial = Vec::new();
    renderer
        .render_chunk_into(&grid, None, &mut initial)
        .or_invariant("required value");

    renderer.resize(12, 6);
    let mut resized = Vec::new();
    renderer
        .render_chunk_into(&grid, Some(&[]), &mut resized)
        .or_invariant("required value");

    let frame = String::from_utf8(resized).or_invariant("required value");
    assert!(
        frame.contains('▀') || frame.contains('▄') || frame.contains('█'),
        "the first diff frame after resize must rebuild live cells\nframe={frame:?}"
    );
}

fn assert_viewport_contains_live_cell(origin: Cell, width: usize, height: usize, cells: &[Cell]) {
    assert!(
        !cells.is_empty(),
        "viewport test needs at least one live cell to prove visibility"
    );
    let (origin_x, origin_y) = origin;
    let width_coord = Coord::try_from(width).or_invariant("test viewport width exceeded Coord");
    let height_coord = Coord::try_from(height).or_invariant("test viewport height exceeded Coord");
    let max_x = origin_x + width_coord - 1;
    let max_y = origin_y + (height_coord * 2) - 1;
    let visible = cells
        .iter()
        .any(|&(x, y)| x >= origin_x && x <= max_x && y >= origin_y && y <= max_y);
    assert!(
        visible,
        "expected viewport origin={origin:?} width={width} height={height} to contain at least one live cell from {cells:?}"
    );
}

#[test]
fn step_engine_matches_blinker_rotation() {
    let grid = pattern_by_name("blinker").or_invariant("required value");
    let next = step_grid(&grid);
    let expected = BitGrid::from_cells(&[(1, -1), (1, 0), (1, 1)]);
    assert_grids_eq(
        "single-step engine should rotate blinker into its vertical phase",
        &next,
        &expected,
    );
}

#[test]
fn chunk_transition_cache_reuses_local_neighborhoods() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let mut memo = Memo::default();

    step_grid_with_changes_and_memo(&grid, &mut memo);
    let after_first = memo.chunk_transition_cache_len();
    step_grid_with_changes_and_memo(&grid, &mut memo);
    let after_second = memo.chunk_transition_cache_len();

    assert!(after_first > 0);
    assert_eq!(after_first, after_second);
}

#[test]
fn stratified_reference_suite_matches_exact_replay_contract() {
    let limits = ClassificationLimits {
        max_generations: 256,
    };
    for case in curated_reference_suite() {
        assert_matches_reference_contract(&case.name, &case.grid, &limits);
    }
}

#[test]
fn stratified_random_soups_match_exact_replay_contract() {
    let limits = ClassificationLimits {
        max_generations: 192,
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
                assert_matches_reference_contract(
                    &format!("random_{size}_{fill_percent}_{seed}"),
                    &grid,
                    &limits,
                );
            }
        }
    }
}

#[test]
fn stratified_clustered_soups_match_exact_replay_contract() {
    let limits = ClassificationLimits {
        max_generations: 192,
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
                assert_matches_reference_contract(
                    &format!("clustered_{size}_{fill_percent}_{seed}"),
                    &grid,
                    &limits,
                );
            }
        }
    }
}

#[test]
fn structured_random_soups_match_exact_replay_contract() {
    let limits = ClassificationLimits {
        max_generations: 192,
    };
    for size in [16, 32, 64] {
        for seed in 1..=4_u64 {
            let grid = structured_random_soup(size, size, hash_seed(size, 77, seed));
            assert_matches_reference_contract(&format!("structured_{size}_{seed}"), &grid, &limits);
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

fn crop_grid(grid: &BitGrid, min_x: Coord, min_y: Coord, max_x: Coord, max_y: Coord) -> BitGrid {
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|&(x, y)| x >= min_x && x <= max_x && y >= min_y && y <= max_y)
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn contains_component_variant(grid: &BitGrid, variants: &[Vec<Cell>]) -> bool {
    connected_components(grid)
        .into_iter()
        .map(|component| normalize(&BitGrid::from_cells(&component)).0.cells)
        .any(|normalized| variants.iter().any(|variant| variant == &normalized))
}

fn all_normalized_variants(pattern: &BitGrid) -> Vec<Vec<Cell>> {
    all_evolution_variants(pattern, 1)
}

fn all_evolution_variants(pattern: &BitGrid, period: usize) -> Vec<Vec<Cell>> {
    let mut variants = Vec::new();
    let mut phase = pattern.clone();

    for _ in 0..period {
        append_symmetry_variants(&mut variants, &phase.live_cells());
        phase = step_grid(&phase);
    }

    variants
}

fn append_symmetry_variants(variants: &mut Vec<Vec<Cell>>, cells: &[Cell]) {
    let transforms: [fn(Coord, Coord) -> Cell; 8] = [
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

fn connected_components(grid: &BitGrid) -> Vec<Vec<Cell>> {
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
    changed_cells: Option<&[Cell]>,
) -> String {
    let mut out = Vec::new();
    buffer
        .render_into(grid, changed_cells, &mut out)
        .or_invariant("required value");
    String::from_utf8(out).or_invariant("required value")
}

fn assert_matches_reference_contract(name: &str, grid: &BitGrid, limits: &ClassificationLimits) {
    let actual = classify_seed(grid, limits, &mut Memo::default());
    if let Classification::Unknown { simulated } = &actual {
        assert!(
            *simulated >= limits.max_generations,
            "{name}: Unknown shortened the requested classification horizon\nrequested={}\nactual={actual}",
            limits.max_generations,
        );
    }
    let verification_horizon = exact_outcome_horizon(&actual).or_invariant(
        "finite reference suites cannot accept an unverified likely-infinite classification",
    );
    let verified = reference_classify(
        grid,
        &ClassificationLimits {
            max_generations: verification_horizon,
        },
    );
    assert_same_outcome(name, &verified, &actual);
}

fn exact_outcome_horizon(classification: &Classification) -> Option<u64> {
    match classification {
        Classification::DiesOut { at_generation } => Some(*at_generation),
        Classification::Repeats { period, first_seen } => first_seen.checked_add(*period),
        Classification::Spaceship { detected_at, .. } => Some(*detected_at),
        Classification::Unknown { simulated } => Some(*simulated),
        Classification::LikelyInfinite { .. } => None,
    }
}

fn assert_same_outcome(name: &str, expected: &Classification, actual: &Classification) {
    assert_eq!(
        actual, expected,
        "{name}: complete classification contract mismatch\nexpected={expected}\nactual={actual}"
    );
}

fn assert_persistent_expansion(name: &str, actual: &Classification, horizon: u64) {
    match actual {
        Classification::LikelyInfinite {
            reason: "persistent_expansion",
            detected_at,
        } => assert!(
            *detected_at <= horizon,
            "{name}: persistent expansion was reported beyond the requested horizon\nhorizon={horizon}\nactual={actual}"
        ),
        _ => crate::invariant_failure!(
            "{name}: known emitter/puffer must be classified from its observed persistent expansion\nhorizon={horizon}\nactual={actual}"
        ),
    }
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
            grid: pattern_by_name(name).or_invariant("required value"),
        });
    }

    let adversarial = [
        (
            "acorn_offset",
            pattern_by_name("acorn")
                .or_invariant("required value")
                .clone(),
        ),
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

fn clustered_noise_soup(width: Coord, height: Coord, fill_percent: u32, seed: u64) -> BitGrid {
    let base = random_soup(width, height, fill_percent, seed);
    let mut cells = Vec::new();
    for (x, y) in base.live_cells() {
        cells.push((x, y));
        if ((x + y).unsigned_abs() + seed).is_multiple_of(3) && x + 1 < width {
            cells.push((x + 1, y));
        }
        if ((x * 3 + y * 5).unsigned_abs() + seed).is_multiple_of(5) && y + 1 < height {
            cells.push((x, y + 1));
        }
    }
    BitGrid::from_cells(&cells)
}

fn structured_random_soup(width: Coord, height: Coord, seed: u64) -> BitGrid {
    let left = random_soup(width / 2, height, 18, seed);
    let right = random_soup(width / 2, height, 12, derive_seed(seed, [1]));
    let mut cells = left.live_cells();
    cells.extend(
        right
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + (width / 2), y)),
    );
    cells.extend(
        pattern_by_name("blinker")
            .or_invariant("required value")
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + width / 3, y + height / 3)),
    );
    cells.extend(
        pattern_by_name("block")
            .or_invariant("required value")
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + width / 2, y + height / 2)),
    );
    BitGrid::from_cells(&cells)
}

fn hash_seed(a: Coord, b: u32, c: u64) -> u64 {
    hash_words(0x5445_5354_5345_4544, [a.cast_unsigned(), u64::from(b), c])
}

pub(super) fn reference_classify(seed: &BitGrid, limits: &ClassificationLimits) -> Classification {
    let mut seen: HashMap<Vec<Cell>, (u64, Cell)> = HashMap::new();
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

        seen.insert(signature.cells, (generation, origin));
        grid = reference_step_grid(&grid);
    }

    Classification::Unknown {
        simulated: limits.max_generations,
    }
}

fn reference_step_grid(grid: &BitGrid) -> BitGrid {
    let mut counts: HashMap<Cell, u8> = HashMap::new();
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
