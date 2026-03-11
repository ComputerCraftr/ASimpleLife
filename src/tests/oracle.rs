use crate::classify::{Classification, ClassificationLimits};
use crate::engine::{SimulationBackend, SimulationSession};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeSession;
use crate::normalize::normalize;
use crate::oracle::{OracleSession, OracleStateMetrics, OracleStepPlan};

#[test]
fn hashlife_session_matches_one_shot_advance() {
    let grid = random_soup(24, 24, 20, 0x0123_4567_89AB_CDEF);
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, 509);

    let mut session = HashLifeSession::new();
    let segmented = session.advance(&grid, 256);
    let segmented = session.advance(&segmented, 253);
    session.finish();

    assert_eq!(normalize(&expected).0, normalize(&segmented).0);
}

#[test]
fn simulation_session_segmented_matches_single_call() {
    let grid = random_soup(32, 32, 20, 0xA5A5_5A5A_DEAD_BEEF);
    let expected = crate::engine::advance_grid(&grid, 512);

    let mut session = SimulationSession::new();
    let segmented = session.advance(&grid, 256).grid;
    let segmented = session.advance(&segmented, 256).grid;
    session.finish();

    assert_eq!(normalize(&expected.grid).0, normalize(&segmented).0);
}

#[test]
fn oracle_session_uses_exact_repeat_to_reach_target() {
    let grid = pattern_by_name("block").unwrap();
    let target = 1_000_000_u64;
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid.clone(), 0, Default::default(), &mut simulation).advance_to_target(
        target,
        None,
    );

    assert!(matches!(
        outcome.classification,
        Classification::Repeats { period: 1, .. }
    ));
    assert_eq!(outcome.final_generation, target);
    assert_eq!(normalize(&outcome.grid).0, normalize(&grid).0);
}

#[test]
fn oracle_session_uses_translated_cycle_to_reach_target() {
    let grid = pattern_by_name("glider").unwrap();
    let target = 10_000_u64;
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, target);
    let mut simulation = SimulationSession::new();
    let outcome =
        OracleSession::new(grid, 0, Default::default(), &mut simulation).advance_to_target(
            target,
            None,
        );

    assert!(matches!(outcome.classification, Classification::Spaceship { .. }));
    assert_eq!(outcome.final_generation, target);
    assert_eq!(normalize(&outcome.grid).0, normalize(&expected).0);
}

#[test]
fn oracle_session_continuation_matches_expected_repeat() {
    let grid = pattern_by_name("blinker").unwrap();
    let limits = ClassificationLimits {
        max_generations: 1024,
        ..ClassificationLimits::default()
    };
    let mut simulation = SimulationSession::new();
    let result = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .classify_continuation(1024, 8, &limits);
    assert!(matches!(result, Classification::Repeats { period: 2, .. }));
}

#[test]
fn hashlife_session_root_advance_matches_one_shot_advance() {
    let grid = pattern_by_name("glider").unwrap();
    let expected = crate::hashlife::HashLifeEngine::default().advance(&grid, 256);

    let mut session = HashLifeSession::new();
    session.load_grid(&grid);
    session.advance_root(256);
    let advanced = session
        .sample_grid()
        .expect("session should be sampleable after root advance")
        .clone();

    assert_eq!(normalize(&advanced).0, normalize(&expected).0);
}

#[test]
fn hashlife_session_sampling_preserves_state() {
    let grid = pattern_by_name("glider").unwrap();
    let mut session = HashLifeSession::new();
    let advanced = session.advance(&grid, 128);
    let sampled_signature = normalize(
        session
            .sample_grid()
            .expect("session should have a sampled grid"),
    )
    .0;
    assert_eq!(normalize(&advanced).0, sampled_signature);

    let sampled_grid = advanced.clone();
    let continued = session.advance(&sampled_grid, 128);
    let mut expected = crate::hashlife::HashLifeEngine::default();
    let expected = expected.advance(&grid, 256);
    assert_eq!(normalize(&continued).0, normalize(&expected).0);
}

#[test]
fn oracle_session_keeps_large_emitter_target_on_hashlife_backend() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let target = 100_000_u64;
    let mut simulation = SimulationSession::new();
    let mut planned_backends = Vec::new();
    let mut callback = |plan: OracleStepPlan, _metrics: OracleStateMetrics| {
        if plan.step_span > 0 && plan.generation >= 64 {
            planned_backends.push(plan.backend);
        }
    };

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_to_target(target, Some(&mut callback));

    assert_eq!(outcome.final_generation, target);
    assert!(
        !planned_backends.is_empty(),
        "expected at least one planned backend after the exact probe prefix"
    );
    assert!(
        planned_backends
            .iter()
            .all(|backend| *backend == SimulationBackend::HashLife),
        "expected all post-prefix planned backends to stay on HashLife, got {planned_backends:?}"
    );
}
