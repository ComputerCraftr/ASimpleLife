use crate::RequiredExt;
use crate::bitgrid::BitGrid;
use crate::classify::{
    Classification, ClassificationCache, ClassificationCertainty, ClassificationEvidence,
    ClassificationGrowthKind, ClassificationLimits, ClassificationOutcome, classify_seed,
    classify_seed_report_cached, effective_generation_limit,
};
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
    let mut cache = ClassificationCache::default();
    cache.insert(
        signature.clone(),
        crate::classify::ClassificationReport {
            outcome: ClassificationOutcome::Expanding,
            certainty: ClassificationCertainty::Heuristic,
            observed_through: 512,
            evidence: ClassificationEvidence::PersistentGrowth {
                kind: ClassificationGrowthKind::PersistentExpansion,
                detected_at: 512,
            },
        },
    );

    assert!(
        cache.is_empty(),
        "heuristic classifications must not leak across classification horizons"
    );
    assert_eq!(cache.get(&signature), None);
}

#[test]
fn classification_cache_is_independent_from_transition_cache_collection() {
    let grid = separated_blocks(1, 4);
    let signature = normalize(&grid).0;
    let mut classification_cache = ClassificationCache::default();
    let mut transition_memo = Memo::default();
    let report = classify_seed_report_cached(
        &grid,
        &ClassificationLimits::default(),
        &mut transition_memo,
        &mut classification_cache,
    );

    transition_memo.force_collect_transition_caches();

    assert_eq!(
        classification_cache.get(&signature),
        Some(report),
        "transition-cache collection must not own or invalidate classifier results"
    );
}

#[test]
fn typed_report_distinguishes_still_life_from_oscillator() {
    let mut cache = ClassificationCache::default();
    let still_life = classify_seed_report_cached(
        &separated_blocks(1, 4),
        &ClassificationLimits::default(),
        &mut Memo::default(),
        &mut cache,
    );
    let oscillator = classify_seed_report_cached(
        &pattern_by_name("blinker").or_invariant("blinker fixture should exist"),
        &ClassificationLimits::default(),
        &mut Memo::default(),
        &mut cache,
    );

    assert_eq!(still_life.outcome, ClassificationOutcome::StillLife);
    assert_eq!(still_life.certainty, ClassificationCertainty::Exact);
    assert_eq!(
        still_life.evidence,
        ClassificationEvidence::Recurrence {
            period: 1,
            first_seen: 0,
            displacement: (0, 0),
            detected_at: 1,
        }
    );
    assert_eq!(oscillator.outcome, ClassificationOutcome::Oscillator);
    assert_eq!(oscillator.certainty, ClassificationCertainty::Exact);
    assert!(
        matches!(
            oscillator.evidence,
            ClassificationEvidence::Recurrence {
                period: 2,
                displacement: (0, 0),
                ..
            }
        ),
        "blinker should carry exact period-two recurrence evidence: {oscillator:?}"
    );
    assert_eq!(cache.len(), 2, "both exact reports should be cached");
    assert_eq!(
        still_life.to_legacy(),
        Classification::Repeats {
            period: 1,
            first_seen: 0,
        }
    );
}

#[test]
fn classifier_and_oracle_share_exact_recurrence_semantics() {
    for name in ["block", "blinker", "glider"] {
        let grid = pattern_by_name(name).or_invariant("recurrence fixture should exist");
        let limits = ClassificationLimits {
            max_generations: 64,
        };
        let classified = classify_seed(&grid, &limits, &mut Memo::default());
        let mut simulation = SimulationSession::new();
        let oracle = OracleSession::new(grid, 0, &mut simulation)
            .classify_continuation(64, limits.max_generations);
        assert_eq!(classified, oracle, "repeat semantics diverged for {name}");
    }
}

#[test]
fn typed_reports_round_trip_legacy_classifier_results() {
    let cases = [
        Classification::DiesOut { at_generation: 7 },
        Classification::Repeats {
            period: 3,
            first_seen: 5,
        },
        Classification::Spaceship {
            period: 4,
            first_seen: 0,
            delta: (1, -1),
            detected_at: 4,
        },
        Classification::LikelyInfinite {
            reason: "persistent_expansion",
            detected_at: 512,
        },
        Classification::Unknown { simulated: 128 },
    ];

    for classification in cases {
        let report = crate::classify::ClassificationReport::from_legacy(&classification);
        assert_eq!(
            report.to_legacy(),
            classification,
            "typed compatibility conversion changed classifier semantics: {report:?}"
        );
    }
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
