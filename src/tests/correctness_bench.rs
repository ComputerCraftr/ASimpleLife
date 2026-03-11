use crate::benchmark::{
    BenchmarkOptions, assert_same_outcome, benchmark_report_json_value,
    bitmask_pattern, canonical_small_box_mask, oracle_runtime_case_for_tests,
    reference_classify_from_checkpoint, reference_is_decisive,
};
use crate::classify::{ClassificationLimits, classify_seed, predict_seed_with_checkpoint};
use crate::memo::Memo;
use crate::bitgrid::Coord;

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
        prediction_max_generations: None,
        oracle_max_generations: None,
        exhaustive_5x5: false,
        oracle_runtime_case: true,
        oracle_runtime_target_generation: None,
        progress: false,
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
        families: None,
        prediction_max_generations: None,
        oracle_max_generations: None,
        exhaustive_5x5: false,
        oracle_runtime_case: true,
        oracle_runtime_target_generation: Some(8_192),
        progress: false,
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
        oracle_runtime_case_for_tests();
    assert_eq!(pattern, "gosper_glider_gun");
    assert_eq!(target_generation, 4_096);
    assert!(population > 0);
    assert!(bounds_span > 0);
    let _ = reached_target;
}

#[test]
fn oracle_runtime_case_mode_skips_main_suite_totals() {
    let json = benchmark_report_json_value(&BenchmarkOptions {
        families: Some(vec!["delayed".to_string()]),
        prediction_max_generations: None,
        oracle_max_generations: None,
        exhaustive_5x5: false,
        oracle_runtime_case: true,
        oracle_runtime_target_generation: None,
        progress: false,
    });
    assert_eq!(json.get("total").and_then(|value| value.as_u64()), Some(0));
    assert_eq!(
        json.get("compatible").and_then(|value| value.as_u64()),
        Some(0)
    );
}
