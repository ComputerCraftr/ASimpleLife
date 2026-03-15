use crate::RequiredExt;
use crate::app::initial_grid;
use crate::classify::{Classification, ClassificationLimits};
use crate::cli::Config;
use crate::engine::{SimulationBackend, SimulationSession};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::{GridExtractionError, GridExtractionPolicy, HashLifeSession};
use crate::normalize::normalize;
use crate::oracle::{OracleRuntimeState, OracleSession, OracleStateMetrics, OracleStepPlan};
use crate::tests::hashlife_support::{
    GEN_SPARSE_SESSION_COVERAGE, GEN_SPARSE_SESSION_REGRESSION, assert_normalized_grids_eq,
};

#[test]
fn hashlife_session_matches_one_shot_advance() {
    let grid = random_soup(24, 24, 20, 0x0123_4567_89AB_CDEF);
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, 509);

    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(256)
        .or_invariant("oracle checkpoint prefix should complete");
    let segmented = session
        .sample_grid()
        .or_invariant("session should be sampleable after root advance");
    session
        .try_load_grid(&segmented)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(253)
        .or_invariant("oracle checkpoint suffix should complete");
    let segmented = session
        .sample_grid()
        .or_invariant("session should be sampleable after root advance");
    session.finish();

    assert_normalized_grids_eq(
        "segmented HashLife session should match one-shot advance",
        &segmented,
        &expected,
    );
}

#[test]
fn simulation_session_segmented_matches_single_call() {
    let grid = random_soup(32, 32, 20, 0xA5A5_5A5A_DEAD_BEEF);
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, 512);

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(256)
        .or_invariant("oracle session prefix should complete");
    let segmented = session
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("hashlife state should be sampleable after advance");
    session
        .try_load_hashlife_state(&segmented)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(256)
        .or_invariant("oracle session repeat should complete");
    let segmented = session
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("hashlife state should be sampleable after advance");
    session.finish();

    assert_normalized_grids_eq(
        "segmented SimulationSession HashLife state should match one-shot advance",
        &segmented,
        &expected,
    );
}

#[test]
fn oracle_session_uses_exact_repeat_to_reach_target() {
    let grid = pattern_by_name("block").or_invariant("required value");
    let target = 10_000_000_u64;
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid.clone(), 0, Default::default(), &mut simulation)
        .advance_to_target(target, None);

    assert!(matches!(
        outcome.classification,
        Classification::Repeats { period: 1, .. }
    ));
    assert_eq!(outcome.final_generation, target);
    assert_normalized_grids_eq(
        "exact repeat block target should preserve canonical grid",
        &outcome.grid,
        &grid,
    );
}

#[test]
fn oracle_session_uses_exact_blinker_cycle_to_reach_huge_target() {
    let grid = pattern_by_name("blinker").or_invariant("required value");
    let target = 1_000_001_u64;
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, target);
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_to_target(target, None);

    assert!(matches!(
        outcome.classification,
        Classification::Repeats { period: 2, .. }
    ));
    assert_eq!(outcome.final_generation, target);
    assert_normalized_grids_eq(
        "blinker huge-target oracle result should match one-shot HashLife",
        &outcome.grid,
        &expected,
    );
}

#[test]
fn oracle_session_uses_translated_cycle_to_reach_huge_target() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let target = 1_000_003_u64;
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, target);
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_to_target(target, None);

    assert!(matches!(
        outcome.classification,
        Classification::Spaceship { period: 4, .. }
    ));
    assert_eq!(outcome.final_generation, target);
    assert_normalized_grids_eq(
        "translated-cycle oracle result should match one-shot HashLife",
        &outcome.grid,
        &expected,
    );
}

#[test]
fn oracle_session_continuation_matches_expected_repeat() {
    let grid = pattern_by_name("blinker").or_invariant("required value");
    let limits = ClassificationLimits {
        max_generations: 1024,
    };
    let mut simulation = SimulationSession::new();
    let result = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .classify_continuation(limits.max_generations, 8);
    assert!(matches!(result, Classification::Repeats { period: 2, .. }));
}

#[test]
fn oracle_session_repeated_deep_runs_are_deterministic() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let target = 100_000_u64;

    let first = {
        let mut simulation = SimulationSession::new();
        OracleSession::new(grid.clone(), 0, Default::default(), &mut simulation)
            .advance_to_target(target, None)
    };
    let second = {
        let mut simulation = SimulationSession::new();
        OracleSession::new(grid, 0, Default::default(), &mut simulation)
            .advance_to_target(target, None)
    };

    assert_eq!(
        first.classification, second.classification,
        "repeated deep oracle runs should classify identically"
    );
    assert_eq!(
        first.final_generation, second.final_generation,
        "repeated deep oracle runs should land on the same generation"
    );
    assert_normalized_grids_eq(
        "repeated deep oracle runs should produce the same canonical grid",
        &first.grid,
        &second.grid,
    );
}

#[test]
fn hashlife_session_root_advance_matches_one_shot_advance() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, 256);

    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(256)
        .or_invariant("checkpoint identity prefix should complete");
    let advanced = session
        .sample_grid()
        .or_invariant("session should be sampleable after root advance");

    assert_normalized_grids_eq(
        "root-only HashLife session advance should match one-shot advance",
        &advanced,
        &expected,
    );
}

#[test]
fn hashlife_session_sampling_preserves_state() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(128)
        .or_invariant("checkpoint comparison prefix should complete");
    let advanced = session
        .sample_grid()
        .or_invariant("session should have a sampled grid");
    let sampled_signature = normalize(
        &session
            .sample_grid()
            .or_invariant("session should have a sampled grid"),
    )
    .0;
    assert_eq!(
        normalize(&advanced).0,
        sampled_signature,
        "sampling should preserve session state signature"
    );

    let sampled_grid = advanced.clone();
    session
        .try_load_grid(&sampled_grid)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(128)
        .or_invariant("checkpoint comparison repeat should complete");
    let continued = session
        .sample_grid()
        .or_invariant("session should have a sampled grid")
        .clone();
    let mut expected = crate::hashlife::HashLifeEngine::default();
    let expected = expected.advance(&grid, 256);
    assert_normalized_grids_eq(
        "reloading a sampled session grid should continue to the same canonical result",
        &continued,
        &expected,
    );
}

#[test]
fn oracle_session_keeps_large_emitter_target_on_hashlife_backend() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let target = 100_000_u64;
    let mut simulation = SimulationSession::new();
    let mut planned_backends = Vec::new();
    let mut callback = |plan: OracleStepPlan, _: OracleStateMetrics| {
        if plan.step_span > 0 && plan.generation >= 64 {
            planned_backends.push(plan.backend);
        }
    };

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_to_target(target, Some(&mut callback));

    assert_eq!(outcome.final_generation, target);
    assert_eq!(
        simulation.hashlife_generation(),
        target,
        "a materialized emitter outcome must leave the session at the claimed generation"
    );
    assert!(
        !planned_backends.is_empty(),
        "expected at least one planned backend after the exact probe prefix"
    );
    assert!(
        planned_backends
            .iter()
            .all(|backend| matches!(backend, SimulationBackend::HashLife)),
        "expected large emitter target run to remain on HashLife after the exact probe prefix, got {planned_backends:?}"
    );
    assert!(
        simulation.hashlife_sample_materializations() <= 2,
        "expected large HashLife emitter run to avoid repeated grid materialization"
    );
}

#[test]
fn oracle_runtime_target_matches_exact_gosper_metadata_at_moderate_target() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let target = 10_000_u64;
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, target);
    let expected_bounds_span = expected
        .bounds()
        .map(|bounds| {
            let (min_x, min_y, max_x, max_y) = bounds;
            (max_x - min_x + 1).max(max_y - min_y + 1)
        })
        .unwrap_or(0);
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(target, None);

    assert_eq!(outcome.final_generation, target);
    assert!(matches!(
        outcome.classification,
        Classification::LikelyInfinite {
            reason: "emitter_cycle",
            ..
        }
    ));
    assert_eq!(outcome.population, expected.population());
    assert_eq!(outcome.bounds_span, expected_bounds_span);
    assert_eq!(outcome.state, OracleRuntimeState::RetainedHashLife);
    assert_eq!(simulation.hashlife_generation(), target);
}

#[test]
fn oracle_runtime_target_uses_emitter_cycle_at_hundred_million() {
    let grid = pattern_by_name("gosper_glider_gun").or_invariant("required value");
    let target = 100_000_000_u64;
    let mut simulation = SimulationSession::new();
    let mut planned_steps = Vec::new();
    let mut callback = |plan: OracleStepPlan, _: OracleStateMetrics| {
        if plan.step_span > 0 {
            planned_steps.push((plan.generation, plan.step_span, plan.backend));
        }
    };

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(target, Some(&mut callback));

    assert_eq!(outcome.final_generation, target);
    assert!(matches!(
        outcome.classification,
        Classification::LikelyInfinite {
            reason: "emitter_cycle",
            ..
        }
    ));
    assert_eq!(outcome.state, OracleRuntimeState::Modeled);
    assert!(
        !simulation.hashlife_loaded(),
        "modeled emitter metrics must not masquerade as a materialized target session"
    );
    assert!(
        !planned_steps.is_empty(),
        "expected runtime target to plan at least one step"
    );
    let late_small_tail = planned_steps
        .iter()
        .filter(|(generation, step_span, _)| *generation > 0 && *step_span < 1_000_000)
        .collect::<Vec<_>>();
    assert!(
        late_small_tail.is_empty(),
        "expected 100,000,000-generation emitter-cycle landing to avoid diminishing late tail steps, got planned_steps={planned_steps:?}"
    );
}

#[test]
fn oracle_session_confirmed_cycle_does_not_schedule_diminishing_tail_jumps() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let target = 1_000_003_u64;
    let mut simulation = SimulationSession::new();
    let mut planned_steps = Vec::new();
    let mut callback = |plan: OracleStepPlan, _: OracleStateMetrics| {
        if plan.step_span > 0 {
            planned_steps.push((plan.generation, plan.step_span, plan.backend));
        }
    };

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_to_target(target, Some(&mut callback));

    assert_eq!(outcome.final_generation, target);
    let late_steps = planned_steps
        .into_iter()
        .filter(|(generation, _, _)| *generation >= 4)
        .collect::<Vec<_>>();
    assert!(
        late_steps.is_empty(),
        "expected exact cycle landing after confirmation without late planned tail steps, got {late_steps:?}"
    );
}

fn rotate_180(grid: &crate::bitgrid::BitGrid) -> crate::bitgrid::BitGrid {
    let (min_x, min_y, max_x, max_y) = grid.bounds().or_invariant("required value");
    let cells = grid
        .live_cells()
        .into_iter()
        .map(|(x, y)| (max_x - (x - min_x), max_y - (y - min_y)))
        .collect::<Vec<_>>();
    crate::bitgrid::BitGrid::from_cells(&cells)
}

fn huge_sparse_glider_pair() -> crate::bitgrid::BitGrid {
    let glider = pattern_by_name("glider").or_invariant("required value");
    let mirrored = rotate_180(&glider);
    let mut cells = mirrored.live_cells();
    cells.extend(
        glider
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + 256, y + 256)),
    );
    crate::bitgrid::BitGrid::from_cells(&cells)
}

fn huge_sparse_block_and_glider() -> crate::bitgrid::BitGrid {
    let block = pattern_by_name("block").or_invariant("required value");
    let glider = pattern_by_name("glider").or_invariant("required value");
    let mut cells = block.live_cells();
    cells.extend(
        glider
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + 512, y + 512)),
    );
    crate::bitgrid::BitGrid::from_cells(&cells)
}

fn narrow_live_region(
    session: &mut SimulationSession,
    bounds: (i64, i64, i64, i64),
) -> ((i64, i64, i64, i64), crate::bitgrid::BitGrid) {
    let mut region = bounds;
    let mut viewport = session
        .sample_hashlife_state_region(region.0, region.1, region.2, region.3)
        .or_invariant("full reported bounds should remain sampleable as a bounded region");
    assert!(viewport.population() > 0);

    while (region.2 - region.0) > 64 || (region.3 - region.1) > 64 {
        let mid_x = region.0 + (region.2 - region.0) / 2;
        let mid_y = region.1 + (region.3 - region.1) / 2;
        let quadrants = [
            (region.0, region.1, mid_x, mid_y),
            (mid_x + 1, region.1, region.2, mid_y),
            (region.0, mid_y + 1, mid_x, region.3),
            (mid_x + 1, mid_y + 1, region.2, region.3),
        ];
        if let Some((next_region, next_viewport)) = quadrants.into_iter().find_map(|candidate| {
            let sampled = session.sample_hashlife_state_region(
                candidate.0,
                candidate.1,
                candidate.2,
                candidate.3,
            )?;
            (sampled.population() > 0).then_some((candidate, sampled))
        }) {
            region = next_region;
            viewport = next_viewport;
        } else {
            break;
        }
    }

    (region, viewport)
}

#[test]
fn huge_sparse_hashlife_root_advance_preserves_population_and_bounds() {
    let grid = huge_sparse_glider_pair();
    let generations = GEN_SPARSE_SESSION_COVERAGE;
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, generations);

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(generations)
        .or_invariant("huge sparse root advance should complete");

    assert_eq!(
        session
            .hashlife_population_count()
            .map(crate::hashlife::PopulationCount::lower_bound),
        Some(expected.population() as u128)
    );
    assert_eq!(session.hashlife_bounds(), expected.bounds());
}

#[test]
fn huge_sparse_block_and_glider_root_advance_matches_one_shot_advance() {
    let grid = huge_sparse_block_and_glider();
    let generations = GEN_SPARSE_SESSION_REGRESSION;
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, generations);

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(generations)
        .or_invariant("sparse block and glider advance should complete");

    assert_eq!(
        session
            .hashlife_population_count()
            .map(crate::hashlife::PopulationCount::lower_bound),
        Some(expected.population() as u128)
    );
    assert_eq!(session.hashlife_bounds(), expected.bounds());
}

#[test]
fn huge_sparse_block_and_glider_root_advance_does_not_materialize_full_grid() {
    let grid = huge_sparse_block_and_glider();
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");

    assert_eq!(session.hashlife_sample_materializations(), 0);
    session
        .advance_hashlife_root(GEN_SPARSE_SESSION_REGRESSION)
        .or_invariant("materialization regression advance should complete");
    assert_eq!(
        session.hashlife_sample_materializations(),
        0,
        "root-native HashLife session advance should not materialize a full grid"
    );
}

#[test]
fn huge_sparse_block_and_glider_root_advance_matches_segment_prefixes() {
    let grid = huge_sparse_block_and_glider();
    let segments = [524_288_u64, 262_144, 131_072, 65_536, 16_384, 512, 64];
    let mut session = SimulationSession::new();
    let mut expected_engine = crate::hashlife::HashLifeEngine::default();
    let mut expected = grid.clone();
    let mut total = 0_u64;
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");

    for step in segments {
        session
            .advance_hashlife_root(step)
            .or_invariant("segmented sparse advance should complete");
        total += step;
        expected = expected_engine.advance(&expected, step);
        assert_eq!(
            session
                .hashlife_population_count()
                .map(crate::hashlife::PopulationCount::lower_bound),
            Some(expected.population() as u128),
            "population mismatch after cumulative generations={total}"
        );
        assert_eq!(
            session.hashlife_bounds(),
            expected.bounds(),
            "bounds mismatch after cumulative generations={total}"
        );
    }
}

#[test]
fn huge_sparse_hashlife_state_rejects_full_materialization_but_allows_viewport_sampling() {
    let grid = huge_sparse_block_and_glider();

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(GEN_SPARSE_SESSION_REGRESSION)
        .or_invariant("viewport fixture advance should complete");

    let full = session.sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
        max_population: 4,
        max_chunks: 1,
        max_bounds_span: i64::MAX,
    });
    assert!(
        matches!(
            full,
            Err(GridExtractionError::ChunkLimitExceeded { .. })
                | Err(GridExtractionError::BoundsSpanLimitExceeded { .. })
                | Err(GridExtractionError::PopulationLimitExceeded { .. })
        ),
        "expected bounded full-grid extraction to fail for a huge sparse state, got {full:?}"
    );
    let bounds = session
        .hashlife_bounds()
        .or_invariant("advanced sparse state should still report bounds");
    let (region, expected_viewport) = narrow_live_region(&mut session, bounds);
    let materializations_before_viewport = session.hashlife_sample_materializations();
    let (first_min_x, first_min_y, first_max_x, first_max_y) = region;
    let viewport = session
        .sample_hashlife_state_region(first_min_x, first_min_y, first_max_x, first_max_y)
        .or_invariant("expected live bounded region should remain sampleable");
    assert_normalized_grids_eq(
        "bounded sparse viewport should match the first narrowed live region",
        &viewport,
        &expected_viewport,
    );
    let materializations_after_first_viewport = session.hashlife_sample_materializations();
    let second_viewport = session
        .sample_hashlife_state_region(first_min_x, first_min_y, first_max_x, first_max_y)
        .or_invariant("repeated bounded live-region sampling should remain possible");
    assert_normalized_grids_eq(
        "repeated bounded sparse viewport sampling should remain stable",
        &second_viewport,
        &expected_viewport,
    );
    assert!(
        session.hashlife_sample_materializations() <= materializations_before_viewport + 2,
        "bounded viewport sampling should stay cheap and avoid sticky full-grid retention after a rejected extraction"
    );
    assert!(
        materializations_after_first_viewport <= materializations_before_viewport + 1,
        "first bounded viewport sample should require at most one additional materialization"
    );
}

#[test]
fn huge_sparse_bounded_region_sampling_never_counts_as_full_grid_materialization() {
    let grid = huge_sparse_block_and_glider();
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(GEN_SPARSE_SESSION_REGRESSION)
        .or_invariant("bounded extraction fixture should complete");

    let bounds = session
        .hashlife_bounds()
        .or_invariant("advanced sparse state should still report bounds");
    let (region, expected_viewport) = narrow_live_region(&mut session, bounds);
    let materializations_before = session.hashlife_sample_materializations();
    let viewport = session
        .sample_hashlife_state_region(region.0, region.1, region.2, region.3)
        .or_invariant("expected live bounded region should remain sampleable");
    assert_normalized_grids_eq(
        "bounded region sampling should match the narrowed expected viewport",
        &viewport,
        &expected_viewport,
    );
    assert_eq!(
        session.hashlife_sample_materializations(),
        materializations_before,
        "bounded region sampling must not be counted as full-grid materialization"
    );
}

#[test]
fn huge_sparse_failed_full_grid_extraction_counts_sampling_without_session_full_materialization() {
    let grid = huge_sparse_block_and_glider();
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(GEN_SPARSE_SESSION_REGRESSION)
        .or_invariant("clipped viewport fixture should complete");

    let sample_materializations_before = session.hashlife_sample_materializations();
    let full = session.sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
        max_population: 4,
        max_chunks: 1,
        max_bounds_span: i64::MAX,
    });

    assert!(
        matches!(
            full,
            Err(GridExtractionError::ChunkLimitExceeded { .. })
                | Err(GridExtractionError::BoundsSpanLimitExceeded { .. })
                | Err(GridExtractionError::PopulationLimitExceeded { .. })
        ),
        "expected bounded full-grid extraction to fail for a huge sparse state, got {full:?}"
    );
    assert_eq!(
        session.hashlife_sample_materializations(),
        sample_materializations_before + 1,
        "failed full-grid extraction should count as exactly one sampling attempt"
    );
}

#[test]
fn huge_sparse_hashlife_bounded_viewport_matches_unrestricted_region_contents() {
    let grid = huge_sparse_glider_pair();
    let generations = GEN_SPARSE_SESSION_COVERAGE;

    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");
    session
        .advance_hashlife_root(generations)
        .or_invariant("sparse viewport root advance should complete");

    let bounds = session
        .hashlife_bounds()
        .or_invariant("huge sparse state should still report bounds");
    let (region, viewport) = narrow_live_region(&mut session, bounds);
    let unrestricted = session
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("unrestricted sparse extraction should succeed");
    let expected_cells = unrestricted
        .live_cells()
        .into_iter()
        .filter(|(x, y)| *x >= region.0 && *x <= region.2 && *y >= region.1 && *y <= region.3)
        .collect::<Vec<_>>();
    let expected = crate::bitgrid::BitGrid::from_cells(&expected_cells);

    assert_normalized_grids_eq(
        "bounded sparse viewport should match unrestricted region contents",
        &viewport,
        &expected,
    );
}

#[test]
fn random_seed_420_billion_target_uses_bounded_constellation_projection() {
    let config = Config {
        pattern: "random".to_string(),
        steps: 1,
        max_generations: None,
        target_generation: Some(1_000_000_000),
        step_generations: 1,
        delay_ms: 0,
        width: 80,
        height: 24,
        classify_only: false,
        seed: 420,
    };
    let grid = initial_grid(&config);
    let mut simulation = SimulationSession::new();
    let mut planned_steps = Vec::new();
    let mut observed_metrics = Vec::new();
    let mut callback = |plan: OracleStepPlan, metrics: OracleStateMetrics| {
        if plan.step_span > 0 {
            planned_steps.push((plan.generation, plan.step_span, plan.backend));
            observed_metrics.push(metrics);
        }
    };
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(1_000_000_000, Some(&mut callback));
    let hashlife_steps = planned_steps
        .iter()
        .filter(|(_, _, backend)| matches!(backend, SimulationBackend::HashLife))
        .count();
    let hashlife_generations = planned_steps
        .iter()
        .filter(|(_, _, backend)| matches!(backend, SimulationBackend::HashLife))
        .map(|(_, step_span, _)| *step_span)
        .sum::<u64>();
    let simd_generations = planned_steps
        .iter()
        .filter(|(_, _, backend)| matches!(backend, SimulationBackend::SimdChunk))
        .map(|(_, step_span, _)| *step_span)
        .sum::<u64>();
    assert_eq!(outcome.final_generation, 1_000_000_000);
    assert_eq!(outcome.state, OracleRuntimeState::RetainedHashLife);
    assert!(
        matches!(outcome.classification, Classification::Unknown { .. }),
        "an unresolved target must remain unknown after reaching the requested generation: {:?}",
        outcome.classification
    );
    assert!(
        simulation.hashlife_loaded(),
        "an exact projected constellation should remain available as authoritative target state"
    );
    assert_eq!(
        simulation.hashlife_generation(),
        outcome.final_generation,
        "projected target state and reported generation must remain aligned"
    );
    assert!(
        simulation.hashlife_sample_materializations() <= 16,
        "constellation proof should use only bounded logarithmic samples, got {}",
        simulation.hashlife_sample_materializations()
    );
    let runtime_stats = simulation.hashlife_runtime_stats();
    assert_eq!(
        runtime_stats
            .materialization
            .checkpoint_cell_materializations,
        0,
        "root checkpoints must never materialize cell vectors: {runtime_stats:?}"
    );
    assert!(
        !planned_steps.is_empty(),
        "expected billion-generation runtime target to plan at least one step"
    );
    assert_eq!(
        hashlife_generations + simd_generations,
        outcome.final_generation,
        "planned SIMD and HashLife work should account for the requested generation"
    );
    assert_eq!(
        simd_generations, 8,
        "large inputs should receive only the bounded exact cycle probe, got {planned_steps:?}"
    );
    assert!(
        hashlife_steps >= 1,
        "expected HashLife to advance the unresolved target, got {planned_steps:?}"
    );
    assert!(
        observed_metrics
            .iter()
            .all(|metrics| metrics.population > 0 && metrics.bounds_span > 0),
        "expected runtime-target callback metrics to stay metadata-only but still report live nonzero state, got {observed_metrics:?}"
    );
}

#[test]
fn coarse_hashlife_extinction_does_not_claim_exact_die_out_generation() {
    let grid = crate::benchmark::oracle_extinction_seed_grid_for_tests();
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(1_000_000_000_000, None);

    assert_eq!(outcome.final_generation, 1_000_000_000_000);
    assert!(
        matches!(outcome.classification, Classification::Unknown { .. }),
        "coarse HashLife extinction should not claim an exact die-out generation, got {:?}",
        outcome.classification
    );
}

#[test]
fn large_stable_pattern_uses_cycle_probe_to_reach_trillion_without_hashlife() {
    let mut cells = Vec::new();
    for block_y in 0..20 {
        for block_x in 0..20 {
            let x = block_x * 4;
            let y = block_y * 4;
            cells.extend([(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]);
        }
    }
    let grid = crate::bitgrid::BitGrid::from_cells(&cells);
    let mut simulation = SimulationSession::new();
    let mut planned_steps = Vec::new();
    let mut callback = |plan: OracleStepPlan, _metrics: OracleStateMetrics| {
        if plan.step_span > 0 {
            planned_steps.push(plan);
        }
    };

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(1_000_000_000_000, Some(&mut callback));

    assert_eq!(outcome.final_generation, 1_000_000_000_000);
    assert_eq!(outcome.state, OracleRuntimeState::RetainedHashLife);
    assert_eq!(simulation.hashlife_generation(), 1_000_000_000_000);
    assert!(
        matches!(
            outcome.classification,
            Classification::Repeats {
                period: 1,
                first_seen: 0
            }
        ),
        "large stable pattern should be recognized by the bounded probe, got {:?}",
        outcome.classification
    );
    assert!(
        planned_steps
            .iter()
            .all(|plan| matches!(plan.backend, SimulationBackend::SimdChunk)),
        "a proven stable pattern should not build a trillion-generation HashLife tree, got {planned_steps:?}"
    );
}

#[test]
fn hashlife_first_bounded_probes_run_before_large_target_jump() {
    let grid = pattern_by_name("block").or_invariant("required value");
    let target = 1_000_000_000;
    let mut simulation = SimulationSession::new();

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target_hashlife_first(target, None);

    assert_eq!(outcome.final_generation, target);
    assert_eq!(simulation.hashlife_generation(), target);
    assert!(
        matches!(
            outcome.classification,
            Classification::Repeats {
                period: 1,
                first_seen: 0
            }
        ),
        "bounded checkpoint probes must run before the large target jump, got {:?}",
        outcome.classification
    );
}

#[test]
fn hashlife_backed_oracle_rejects_mismatched_start_generation() {
    let mut simulation = SimulationSession::new();
    simulation
        .try_load_hashlife_state_at_generation(
            &pattern_by_name("block").or_invariant("required value"),
            7,
        )
        .or_invariant("test generation-aware HashLife state should load");

    let result = OracleSession::from_hashlife_state(100, &mut simulation);
    assert!(
        result.is_err(),
        "constructing an oracle at generation 100 over a generation-7 HashLife session must be rejected"
    );
}

#[test]
fn runtime_oracle_reloads_exact_grid_over_stale_hashlife_session() {
    let mut simulation = SimulationSession::new();
    simulation
        .try_load_hashlife_state(&pattern_by_name("block").or_invariant("required value"))
        .or_invariant("test HashLife state should load");
    let blinker = pattern_by_name("blinker").or_invariant("required value");
    let expected = crate::life::step_grid(&blinker);

    let outcome = OracleSession::new(blinker, 0, Default::default(), &mut simulation)
        .advance_runtime_target(1, None);
    let actual = simulation
        .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: u128::MAX,
            max_chunks: usize::MAX,
            max_bounds_span: i64::MAX,
        })
        .or_invariant("runtime oracle should leave the exact target state in the session");

    assert_eq!(outcome.final_generation, 1);
    assert_eq!(simulation.hashlife_generation(), 1);
    assert_eq!(outcome.population, 3);
    assert_normalized_grids_eq(
        "stale HashLife state must not override the oracle's exact input grid",
        &actual,
        &expected,
    );
}
