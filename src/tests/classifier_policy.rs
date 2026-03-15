use crate::RequiredExt;
use crate::benchmark::effective_generation_limit;
use crate::bitgrid::BitGrid;
use crate::classify::{Classification, ClassificationLimits, classify_seed};
use crate::engine::SimulationSession;
use crate::generators::pattern_by_name;
use crate::memo::Memo;
use crate::normalize::normalize;
use crate::oracle::OracleSession;

fn separated_blocks(count: i64, spacing: i64) -> BitGrid {
    let count = usize::try_from(count).or_invariant("block count must fit usize");
    let mut cells = Vec::with_capacity(count.saturating_mul(4));
    for index in 0..count {
        let x = i64::try_from(index).or_invariant("block index exceeded i64") * spacing;
        cells.extend([(x, 0), (x + 1, 0), (x, 1), (x + 1, 1)]);
    }
    BitGrid::from_cells(&cells)
}

#[test]
fn large_stable_population_is_classified_by_repeat_not_size() {
    let grid = separated_blocks(300, 4);
    let result = classify_seed(
        &grid,
        &ClassificationLimits::default(),
        &mut Memo::default(),
    );

    assert_eq!(
        result,
        Classification::Repeats {
            period: 1,
            first_seen: 0,
        },
        "a 1,200-cell stable field must be classified from its observed period"
    );
}

#[test]
fn wide_sparse_hashlife_state_keeps_exact_cycle_checkpointing() {
    let grid = separated_blocks(2, 70_000);
    let mut simulation = SimulationSession::new();
    simulation
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");

    let result = OracleSession::from_hashlife_state(0, &mut simulation)
        .or_invariant("loaded HashLife state should initialize an aligned oracle")
        .classify_continuation(16, 16);

    assert_eq!(
        result,
        Classification::Repeats {
            period: 1,
            first_seen: 0,
        },
        "checkpointing must depend on live-cell work, not empty coordinate span"
    );
}

#[test]
fn hashlife_checkpoint_repeat_reports_fundamental_period() {
    let grid = pattern_by_name("pulsar").or_invariant("pulsar fixture should exist");
    let mut simulation = SimulationSession::new();
    simulation
        .try_load_hashlife_state(&grid)
        .or_invariant("test HashLife state should load");

    let outcome = OracleSession::from_hashlife_state(0, &mut simulation)
        .or_invariant("loaded HashLife state should initialize an aligned oracle")
        .advance_to_target(99, None);

    assert_eq!(
        outcome.classification,
        Classification::Repeats {
            period: 3,
            first_seen: 0,
        },
        "checkpoint cadence must not replace the oscillator's fundamental period"
    );
}

#[test]
fn classification_cache_rejects_limit_dependent_outcomes() {
    let grid = separated_blocks(1, 4);
    let signature = normalize(&grid).0;
    let mut memo = Memo::default();
    memo.insert_classification(
        signature.clone(),
        Classification::LikelyInfinite {
            reason: "persistent_expansion",
            detected_at: 512,
        },
    );

    assert_eq!(
        memo.get_classification(&signature),
        None,
        "heuristic classifications must not leak across classification horizons"
    );
}

#[test]
fn adaptive_generation_horizon_never_shortens_the_request() {
    let grid = separated_blocks(1, 4);
    let limits = ClassificationLimits {
        max_generations: 10_000,
    };

    assert!(
        effective_generation_limit(&limits, grid.population(), grid.bounds())
            >= limits.max_generations,
        "adaptive classification may extend, but never shorten, the requested horizon"
    );
}
