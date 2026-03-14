use crate::benchmark::{
    BenchmarkOptions, assert_same_outcome, benchmark_case_names_for_options,
    benchmark_report_json_value, bitmask_pattern, build_benchmark_report, canonical_small_box_mask,
    special_oracle_case_for_tests,
    reference_classify_from_checkpoint, reference_is_decisive,
};
use crate::bitgrid::Coord;
use crate::classify::{ClassificationLimits, classify_seed, predict_seed_with_checkpoint};
use crate::memo::Memo;

#[test]
#[ignore = "exhaustive 5x5 sweep is expensive"]
fn exhaustive_all_5x5_patterns_reference_check() {
    let prediction_limits = ClassificationLimits {
        max_generations: 128,
        max_population: 10_000,
        max_bounding_box: 256,
    };
    let oracle_limits = ClassificationLimits {
        max_generations: 256,
        max_population: 20_000,
        max_bounding_box: Coord::MAX,
    };
    let mut oracle_simulation = crate::engine::SimulationSession::new();
    for mask in 0_u32..(1_u32 << 25) {
        if canonical_small_box_mask(mask, 5, 5) != mask {
            continue;
        }
        let grid = bitmask_pattern(mask, 5, 5);
        let (actual, checkpoint) =
            predict_seed_with_checkpoint(&grid, &prediction_limits, &mut Memo::default());
        let expected = reference_classify_from_checkpoint(
            checkpoint,
            &actual,
            &prediction_limits,
            &oracle_limits,
            &mut oracle_simulation,
        );
        if !reference_is_decisive(&expected) {
            continue;
        }
        let verification = classify_seed(&grid, &prediction_limits, &mut Memo::default());
        assert_same_outcome(&format!("mask_{mask}"), &expected, &verification);
    }
    oracle_simulation.finish();
}

#[test]
fn oracle_runtime_case_is_absent_by_default() {
    let json = benchmark_report_json_value(&BenchmarkOptions::default());
    assert!(json.get("oracle_runtime_case").is_some());
    assert!(json.get("oracle_runtime_case").unwrap().is_null());
}

#[test]
fn oracle_runtime_case_is_reported_when_requested() {
    let json = benchmark_report_json_value(&BenchmarkOptions {
        families: Some(vec!["iid".to_string()]),
        oracle_runtime_case: true,
        ..BenchmarkOptions::default()
    });
    let case = json
        .get("oracle_runtime_case")
        .and_then(|value| value.as_object())
        .expect("oracle runtime case should be present");
    assert_eq!(
        case.get("pattern").and_then(|value| value.as_str()),
        Some("gosper_glider_gun")
    );
    assert_eq!(
        case.get("target_generation")
            .and_then(|value| value.as_u64()),
        Some(4_096)
    );
}

#[test]
fn oracle_runtime_case_honors_generation_override() {
    let json = benchmark_report_json_value(&BenchmarkOptions {
        oracle_runtime_case: true,
        oracle_runtime_target_generation: Some(8_192),
        ..BenchmarkOptions::default()
    });
    let case = json
        .get("oracle_runtime_case")
        .and_then(|value| value.as_object())
        .expect("oracle runtime case should be present");
    assert_eq!(
        case.get("target_generation")
            .and_then(|value| value.as_u64()),
        Some(8_192)
    );
}

#[test]
fn oracle_runtime_case_reports_runtime_data() {
    let (pattern, target_generation, reached_target, population, bounds_span) =
        special_oracle_case_for_tests(true);
    assert_eq!(pattern, "gosper_glider_gun");
    assert_eq!(target_generation, 4_096);
    assert!(reached_target);
    assert!(population > 0);
    assert!(bounds_span > 0);
}

#[test]
fn oracle_runtime_case_mode_skips_main_suite_totals() {
    let json = benchmark_report_json_value(&BenchmarkOptions {
        families: Some(vec!["delayed".to_string()]),
        oracle_runtime_case: true,
        ..BenchmarkOptions::default()
    });
    assert_eq!(json.get("total").and_then(|value| value.as_u64()), Some(0));
    assert_eq!(
        json.get("compatible").and_then(|value| value.as_u64()),
        Some(0)
    );
}

#[test]
fn oracle_representative_case_is_reported_when_requested() {
    let json = benchmark_report_json_value(&BenchmarkOptions {
        oracle_representative_case: true,
        ..BenchmarkOptions::default()
    });
    let case = json
        .get("oracle_representative_case")
        .and_then(|value| value.as_object())
        .expect("oracle representative case should be present");
    assert_eq!(
        case.get("pattern").and_then(|value| value.as_str()),
        Some("random_seed420_53x24_37")
    );
}

#[test]
fn oracle_representative_case_reports_runtime_data() {
    let (pattern, target_generation, reached_target, population, bounds_span) =
        special_oracle_case_for_tests(false);
    assert_eq!(pattern, "random_seed420_53x24_37");
    assert_eq!(target_generation, 4_096);
    assert!(
        reached_target,
        "representative case should stay alive to target"
    );
    assert!(population > 0);
    assert!(bounds_span > 0);
}

#[test]
fn delayed_family_is_compatible_by_default() {
    let json = serde_json::to_value(build_benchmark_report(&BenchmarkOptions {
        families: Some(vec!["delayed".to_string()]),
        ..BenchmarkOptions::default()
    }).to_json()).unwrap();
    assert_eq!(json.get("total").and_then(|value| value.as_u64()), Some(4));
    assert_eq!(
        json.get("compatible").and_then(|value| value.as_u64()),
        Some(4)
    );
}

#[test]
fn default_seeded_report_uses_canonical_seed_and_no_threshold() {
    let json = serde_json::to_value(build_benchmark_report(&BenchmarkOptions {
        families: Some(vec!["gadget".to_string()]),
        prediction_max_generations: Some(64),
        oracle_max_generations: Some(256),
        cases_per_family: Some(1),
        ..BenchmarkOptions::default()
    }).to_json()).unwrap();
    assert_eq!(
        json.get("mode").and_then(|value| value.as_str()),
        Some("default_seeded")
    );
    assert_eq!(
        json.get("run_seed").and_then(|value| value.as_u64()),
        Some(0x5EED_2026_D15C_A11F)
    );
    assert!(json.get("accuracy_threshold").unwrap().is_null());
    assert!(json.get("accuracy_threshold_met").unwrap().is_null());
}

#[test]
fn seeded_report_includes_seed_mode_and_threshold() {
    let json = serde_json::to_value(build_benchmark_report(&BenchmarkOptions {
        families: Some(vec!["iid".to_string()]),
        prediction_max_generations: Some(128),
        oracle_max_generations: Some(512),
        randomized: true,
        seed: Some(42),
        cases_per_family: Some(2),
        ..BenchmarkOptions::default()
    }).to_json()).unwrap();
    assert_eq!(
        json.get("mode").and_then(|value| value.as_str()),
        Some("seeded")
    );
    assert_eq!(
        json.get("run_seed").and_then(|value| value.as_u64()),
        Some(42)
    );
    assert_eq!(
        json.get("overall_summary")
            .and_then(|value| value.get("total"))
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        json.get("accuracy_threshold")
            .and_then(|value| value.as_f64()),
        Some(0.90)
    );
    assert!(json.get("accuracy_threshold_met").is_some());
}

#[test]
fn seeded_case_names_are_replayable_and_seed_stable() {
    let options = BenchmarkOptions {
        families: Some(vec!["delayed".to_string(), "gadget".to_string()]),
        prediction_max_generations: Some(128),
        oracle_max_generations: Some(512),
        randomized: true,
        seed: Some(7),
        cases_per_family: Some(3),
        ..BenchmarkOptions::default()
    };
    let first = benchmark_case_names_for_options(&options);
    let second = benchmark_case_names_for_options(&options);
    assert_eq!(first, second);
    assert!(first.iter().all(|name| name.contains("seed")));
    assert!(first.iter().any(|name| name.contains("delayed_")));
}

#[test]
fn seeded_case_names_change_with_seed() {
    let first = benchmark_case_names_for_options(&BenchmarkOptions {
        families: Some(vec!["clustered".to_string()]),
        prediction_max_generations: Some(128),
        oracle_max_generations: Some(512),
        randomized: true,
        seed: Some(11),
        cases_per_family: Some(3),
        ..BenchmarkOptions::default()
    });
    let second = benchmark_case_names_for_options(&BenchmarkOptions {
        families: Some(vec!["clustered".to_string()]),
        prediction_max_generations: Some(128),
        oracle_max_generations: Some(512),
        randomized: true,
        seed: Some(12),
        cases_per_family: Some(3),
        ..BenchmarkOptions::default()
    });
    assert_ne!(first, second);
}

#[test]
fn time_seeded_report_has_threshold_and_seed() {
    let json = serde_json::to_value(build_benchmark_report(&BenchmarkOptions {
        families: Some(vec!["structured".to_string()]),
        prediction_max_generations: Some(128),
        oracle_max_generations: Some(512),
        randomized: true,
        cases_per_family: Some(2),
        ..BenchmarkOptions::default()
    }).to_json()).unwrap();
    assert_eq!(
        json.get("mode").and_then(|value| value.as_str()),
        Some("time_seeded")
    );
    assert!(
        json.get("run_seed")
            .and_then(|value| value.as_u64())
            .is_some()
    );
    assert_eq!(
        json.get("accuracy_threshold")
            .and_then(|value| value.as_f64()),
        Some(0.90)
    );
}
