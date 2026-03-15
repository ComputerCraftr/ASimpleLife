use crate::RequiredExt;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::classify::{
    Classification, ClassificationCheckpoint, ClassificationLimits, predict_seed_with_checkpoint,
};
use crate::engine::{SimulationBackend, select_backend};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashing::{SPLITMIX64_GAMMA, derive_seed, mix64};
use crate::memo::Memo;
use crate::oracle::{OracleSession, OracleStateMetrics, OracleStepPlan};

mod report;
mod runtime;
mod suite;

#[cfg(test)]
pub(crate) use report::classifications_are_compatible_for_tests;
use runtime::{run_exhaustive_5x5, run_oracle_representative_case, run_oracle_runtime_case};
pub(crate) use suite::effective_generation_limit;
use suite::{
    benchmark_family_filter, benchmark_run_mode_and_seed, exhaustive_small_box_cases,
    seeded_benchmark_suite,
};

fn bounds_dimensions(bounds: (Coord, Coord, Coord, Coord)) -> (Coord, Coord, Coord) {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;
    (width, height, width.max(height))
}

fn grid_bounds_span(grid: &BitGrid) -> Coord {
    grid.bounds()
        .map(|bounds| bounds_dimensions(bounds).2)
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum BenchmarkRunStatus {
    Clean,
    Diagnostic,
    Failed,
}

#[derive(Clone, Debug, Default)]
pub struct BenchmarkOptions {
    pub families: Option<Vec<String>>,
    pub prediction_max_generations: Option<u64>,
    pub oracle_max_generations: Option<u64>,
    pub randomized: bool,
    pub seed: Option<u64>,
    pub cases_per_family: Option<usize>,
    pub exhaustive_5x5: bool,
    pub oracle_runtime_case: bool,
    pub oracle_representative_case: bool,
    pub oracle_runtime_target_generation: Option<u64>,
    pub progress: bool,
    pub diagnostic: bool,
}

const BENCHMARK_PREDICTION_MAX_GENERATIONS: u64 = 1_024;
const DEFAULT_BENCHMARK_ORACLE_MAX_GENERATIONS: u64 = 1_000_000;
const DEFAULT_ORACLE_MAX_JUMP_TARGET: u64 = 10_000_000;
const DEFAULT_BENCHMARK_RUN_SEED: u64 = 0x5EED_2026_D15C_A11F;
const DEFAULT_CASES_PER_FAMILY: usize = 4;
const RANDOMIZED_COMPATIBILITY_THRESHOLD: f64 = 0.90;
#[cfg(test)]
const TEST_ORACLE_MAX_GENERATIONS: u64 = 4_096;

pub fn run_benchmark_report(
    format: BenchmarkFormat,
    options: &BenchmarkOptions,
) -> BenchmarkRunStatus {
    let report = build_benchmark_report(options);

    match format {
        BenchmarkFormat::Text => report.print(),
        BenchmarkFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report.to_json()).or_invariant("required value")
            );
        }
    }

    benchmark_run_status(options, &report)
}

fn benchmark_run_status(
    options: &BenchmarkOptions,
    report: &BenchmarkReport,
) -> BenchmarkRunStatus {
    if options.diagnostic || options.oracle_runtime_case || options.oracle_representative_case {
        BenchmarkRunStatus::Diagnostic
    } else if report.is_regression_clean() {
        BenchmarkRunStatus::Clean
    } else {
        BenchmarkRunStatus::Failed
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;

    #[test]
    fn normal_benchmark_status_fails_on_classifier_mismatch() {
        let report = BenchmarkReport {
            total: 1,
            conclusive: 1,
            mismatches: vec!["wrong classification".to_string()],
            ..BenchmarkReport::default()
        };
        assert_eq!(
            benchmark_run_status(&BenchmarkOptions::default(), &report),
            BenchmarkRunStatus::Failed
        );
    }

    #[test]
    fn normal_benchmark_status_fails_on_inconclusive_reference() {
        let report = BenchmarkReport {
            total: 1,
            inconclusive: 1,
            ..BenchmarkReport::default()
        };
        assert_eq!(
            benchmark_run_status(&BenchmarkOptions::default(), &report),
            BenchmarkRunStatus::Failed
        );
    }

    #[test]
    fn diagnostic_benchmark_status_does_not_fail_on_mismatch() {
        let report = BenchmarkReport {
            total: 1,
            conclusive: 1,
            mismatches: vec!["wrong classification".to_string()],
            ..BenchmarkReport::default()
        };
        assert_eq!(
            benchmark_run_status(
                &BenchmarkOptions {
                    diagnostic: true,
                    ..BenchmarkOptions::default()
                },
                &report
            ),
            BenchmarkRunStatus::Diagnostic
        );
    }
}

pub(crate) fn build_benchmark_report(options: &BenchmarkOptions) -> BenchmarkReport {
    let (run_mode, run_seed) = benchmark_run_mode_and_seed(options);
    let mut report = BenchmarkReport {
        mode: Some(run_mode),
        run_seed: Some(run_seed),
        accuracy_threshold: (run_mode != BenchmarkRunMode::DefaultSeeded)
            .then_some(RANDOMIZED_COMPATIBILITY_THRESHOLD),
        ..BenchmarkReport::default()
    };
    let prediction_max_generations = options
        .prediction_max_generations
        .unwrap_or(BENCHMARK_PREDICTION_MAX_GENERATIONS);
    let oracle_max_generations = options
        .oracle_max_generations
        .unwrap_or(DEFAULT_BENCHMARK_ORACLE_MAX_GENERATIONS);
    let prediction_limits = ClassificationLimits {
        max_generations: prediction_max_generations,
    };
    let oracle_limits = ClassificationLimits {
        max_generations: oracle_max_generations,
    };
    let filters = benchmark_family_filter(options);
    let include_smallbox = filters
        .as_ref()
        .map(|set| set.contains(&BenchmarkFamily::ExhaustiveSmallBox))
        .unwrap_or(false);
    let mut suite = seeded_benchmark_suite(
        run_seed,
        options.cases_per_family.unwrap_or(DEFAULT_CASES_PER_FAMILY),
    );
    if include_smallbox {
        suite.extend(exhaustive_small_box_cases());
    }
    let suite = suite
        .into_iter()
        .filter(|case| {
            filters
                .as_ref()
                .map(|set| {
                    set.contains(&case.family)
                        || (case.family == BenchmarkFamily::ExhaustiveSmallBox
                            && set.contains(&BenchmarkFamily::ExhaustiveSmallBox))
                })
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let mut oracle_simulation = crate::engine::SimulationSession::new();

    if options.oracle_runtime_case {
        report.oracle_runtime_case = Some(run_oracle_runtime_case(
            options
                .oracle_runtime_target_generation
                .unwrap_or(DEFAULT_ORACLE_MAX_JUMP_TARGET),
            options.progress,
            &mut oracle_simulation,
        ));
        oracle_simulation.finish();
        return report;
    }

    if options.oracle_representative_case {
        report.oracle_representative_case = Some(run_oracle_representative_case(
            options
                .oracle_runtime_target_generation
                .unwrap_or(DEFAULT_ORACLE_MAX_JUMP_TARGET),
            options.progress,
            &mut oracle_simulation,
        ));
        oracle_simulation.finish();
        return report;
    }

    if options.exhaustive_5x5 {
        run_exhaustive_5x5(&mut report, &mut oracle_simulation);
        oracle_simulation.finish();
        return report;
    }

    for case in suite {
        let started = Instant::now();
        let (actual, checkpoint) =
            predict_seed_with_checkpoint(&case.grid, &prediction_limits, &mut Memo::default());
        let expected = reference_classify_from_checkpoint(
            checkpoint,
            &prediction_limits,
            &oracle_limits,
            &mut oracle_simulation,
        );
        report.record(
            &case,
            &expected,
            &actual,
            started.elapsed(),
            prediction_limits.max_generations,
        );
    }

    oracle_simulation.finish();
    if let Some(threshold) = report.accuracy_threshold {
        report.accuracy_threshold_met = Some(
            report.conclusive > 0
                && report.compatible as f64 / report.conclusive as f64 >= threshold,
        );
    }

    report
}

#[cfg(test)]
pub(crate) fn benchmark_report_json_value(options: &BenchmarkOptions) -> serde_json::Value {
    let (run_mode, run_seed) = benchmark_run_mode_and_seed(options);
    let mut report = BenchmarkReport {
        mode: Some(run_mode),
        run_seed: Some(run_seed),
        accuracy_threshold: (run_mode != BenchmarkRunMode::DefaultSeeded)
            .then_some(RANDOMIZED_COMPATIBILITY_THRESHOLD),
        ..BenchmarkReport::default()
    };
    let mut oracle_simulation = crate::engine::SimulationSession::new();
    if options.oracle_runtime_case {
        report.oracle_runtime_case = Some(run_oracle_runtime_case(
            options
                .oracle_runtime_target_generation
                .unwrap_or(TEST_ORACLE_MAX_GENERATIONS),
            false,
            &mut oracle_simulation,
        ));
    }
    if options.oracle_representative_case {
        report.oracle_representative_case = Some(run_oracle_representative_case(
            options
                .oracle_runtime_target_generation
                .unwrap_or(TEST_ORACLE_MAX_GENERATIONS),
            false,
            &mut oracle_simulation,
        ));
    }
    oracle_simulation.finish();
    serde_json::to_value(report.to_json()).or_invariant("required value")
}

#[cfg(test)]
pub(crate) fn benchmark_case_names_for_options(options: &BenchmarkOptions) -> Vec<String> {
    let filters = benchmark_family_filter(options);
    let (_, run_seed) = benchmark_run_mode_and_seed(options);
    let include_smallbox = filters
        .as_ref()
        .map(|set| set.contains(&BenchmarkFamily::ExhaustiveSmallBox))
        .unwrap_or(false);
    let mut suite = seeded_benchmark_suite(
        run_seed,
        options.cases_per_family.unwrap_or(DEFAULT_CASES_PER_FAMILY),
    );
    if include_smallbox {
        suite.extend(exhaustive_small_box_cases());
    }
    suite
        .into_iter()
        .filter(|case| {
            filters
                .as_ref()
                .map(|set| set.contains(&case.family))
                .unwrap_or(true)
        })
        .map(|case| case.name)
        .collect()
}

pub(crate) fn oracle_representative_seed_grid_for_tests() -> BitGrid {
    random_soup(53, 24, 37, 420)
}

#[cfg(test)]
pub(crate) fn oracle_extinction_seed_grid_for_tests() -> BitGrid {
    suite::distant_glider_trigger(
        768,
        pattern_by_name("blinker").or_invariant("required value"),
        (0, 0),
    )
}

#[cfg(test)]
pub(crate) fn special_oracle_case_for_tests(
    runtime_case: bool,
) -> (String, u64, bool, usize, Coord) {
    let mut oracle_simulation = crate::engine::SimulationSession::new();
    let case = if runtime_case {
        run_oracle_runtime_case(TEST_ORACLE_MAX_GENERATIONS, false, &mut oracle_simulation)
    } else {
        run_oracle_representative_case(TEST_ORACLE_MAX_GENERATIONS, false, &mut oracle_simulation)
    };
    oracle_simulation.finish();
    (
        case.pattern.to_string(),
        case.target_generation,
        case.reached_target,
        case.population,
        case.bounds_span,
    )
}

#[cfg(test)]
pub(crate) fn canonical_small_box_mask(mask: u32, width: usize, height: usize) -> u32 {
    suite::canonical_small_box_mask(mask, width, height)
}

pub(crate) fn bitmask_pattern(mask: u32, width: Coord, height: Coord) -> BitGrid {
    let mut grid = BitGrid::new();
    for y in 0..height {
        for x in 0..width {
            let bit = u32::try_from(y * width + x)
                .or_invariant("bitmask pattern dimensions exceeded u32");
            if 1_u32
                .checked_shl(bit)
                .is_some_and(|value| mask & value != 0)
            {
                grid.set(x, y, true);
            }
        }
    }
    grid
}

pub(crate) fn reference_classify_from_checkpoint(
    checkpoint: ClassificationCheckpoint,
    prediction_limits: &ClassificationLimits,
    oracle_limits: &ClassificationLimits,
    oracle_simulation: &mut crate::engine::SimulationSession,
) -> Classification {
    let oracle_limit = effective_generation_limit(
        oracle_limits,
        checkpoint.grid.population(),
        checkpoint.grid.bounds(),
    );

    if checkpoint.generation > oracle_limit {
        return Classification::Unknown {
            simulated: checkpoint.generation,
        };
    }

    OracleSession::new(
        checkpoint.grid,
        checkpoint.generation,
        checkpoint.seen,
        oracle_simulation,
    )
    .classify_continuation(
        oracle_limit.max(prediction_limits.max_generations),
        prediction_limits.max_generations,
    )
}

#[cfg(test)]
pub(crate) fn assert_same_classification(
    name: &str,
    expected: &Classification,
    actual: &Classification,
) {
    assert_eq!(expected, actual, "{name}: classifier result mismatch");
}

#[cfg(test)]
pub(crate) fn assert_classifier_regression_clean(context: &str, report: &BenchmarkReport) {
    assert!(report.total > 0, "{context}: regression evaluated no cases");
    assert_eq!(
        report.inconclusive,
        0,
        "{context}: oracle could not decide {} case(s):\n{}",
        report.inconclusive,
        report.inconclusive_cases.join("\n")
    );
    assert_eq!(
        report.compatible,
        report.conclusive,
        "{context}: {} of {} conclusive classifier case(s) mismatched:\n{}",
        report.conclusive - report.compatible,
        report.conclusive,
        report.mismatches.join("\n")
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BenchmarkFamily {
    IidRandom,
    StructuredRandom,
    ClusteredNoise,
    ExhaustiveSmallBox,
    ExhaustiveFiveByFive,
    LongLivedMethuselah,
    TranslatedPeriodicMover,
    GunPufferBreeder,
    DelayedInteraction,
    DeceptiveAsh,
    ComputationalGadget,
    EmitterInteraction,
}

impl fmt::Display for BenchmarkFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::IidRandom => "iid",
            Self::StructuredRandom => "structured",
            Self::ClusteredNoise => "clustered",
            Self::ExhaustiveSmallBox => "smallbox",
            Self::ExhaustiveFiveByFive => "smallbox_5x5",
            Self::LongLivedMethuselah => "methuselah",
            Self::TranslatedPeriodicMover => "mover",
            Self::GunPufferBreeder => "gun",
            Self::DelayedInteraction => "delayed",
            Self::DeceptiveAsh => "ash",
            Self::ComputationalGadget => "gadget",
            Self::EmitterInteraction => "emitter",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchmarkRunMode {
    DefaultSeeded,
    Seeded,
    TimeSeeded,
}

impl fmt::Display for BenchmarkRunMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::DefaultSeeded => "default_seeded",
            Self::Seeded => "seeded",
            Self::TimeSeeded => "time_seeded",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum OutcomeBucket {
    DiesOut,
    Stabilizes,
    Spaceship,
    InfiniteGrowth,
    Unknown,
}

impl fmt::Display for OutcomeBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::DiesOut => "dies_out",
            Self::Stabilizes => "stabilizes",
            Self::Spaceship => "spaceship",
            Self::InfiniteGrowth => "infinite_growth",
            Self::Unknown => "unknown",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ErrorBucket {
    ExactOrCompatible,
    InconclusiveReference,
    FalseDiesOut,
    FalseRepeats,
    FalseLikelyInfinite,
    Unknown,
    OtherMismatch,
}

impl fmt::Display for ErrorBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::ExactOrCompatible => "exact_or_compatible",
            Self::InconclusiveReference => "inconclusive_reference",
            Self::FalseDiesOut => "false_dies_out",
            Self::FalseRepeats => "false_repeats",
            Self::FalseLikelyInfinite => "false_likely_infinite",
            Self::Unknown => "unknown",
            Self::OtherMismatch => "other_mismatch",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Debug)]
struct BenchmarkCase {
    name: String,
    family: BenchmarkFamily,
    size: i32,
    density_percent: u32,
    grid: BitGrid,
    replay: Option<String>,
}

#[derive(Default)]
pub(crate) struct BenchmarkReport {
    total: usize,
    conclusive: usize,
    inconclusive: usize,
    compatible: usize,
    mode: Option<BenchmarkRunMode>,
    by_family: BTreeMap<BenchmarkFamily, SliceStats>,
    by_error_by_family: BTreeMap<BenchmarkFamily, BTreeMap<ErrorBucket, usize>>,
    by_density: BTreeMap<u32, SliceStats>,
    by_size: BTreeMap<i32, SliceStats>,
    by_outcome: BTreeMap<OutcomeBucket, SliceStats>,
    by_error: BTreeMap<ErrorBucket, usize>,
    by_backend: BTreeMap<SimulationBackend, SliceStats>,
    runtimes: Vec<Duration>,
    runtime_by_family: BTreeMap<BenchmarkFamily, Vec<Duration>>,
    runtime_by_size: BTreeMap<i32, Vec<Duration>>,
    runtime_by_backend: BTreeMap<SimulationBackend, Vec<Duration>>,
    unknown_count: usize,
    mismatches: Vec<String>,
    inconclusive_cases: Vec<String>,
    run_seed: Option<u64>,
    accuracy_threshold: Option<f64>,
    accuracy_threshold_met: Option<bool>,
    oracle_runtime_case: Option<OracleRuntimeCaseReport>,
    oracle_representative_case: Option<OracleRuntimeCaseReport>,
}

impl BenchmarkReport {
    fn is_regression_clean(&self) -> bool {
        self.total > 0 && self.inconclusive == 0 && self.mismatches.is_empty()
    }
}

#[derive(Default, Clone, Debug)]
struct SliceStats {
    total: usize,
    compatible: usize,
}

#[derive(Serialize)]
pub(crate) struct JsonReport {
    total: usize,
    compatible: usize,
    classifier_assessment: JsonClassifierAssessment,
    mode: Option<String>,
    run_seed: Option<u64>,
    unknown_rate: f64,
    overall_summary: JsonSliceStats,
    accuracy_threshold: Option<f64>,
    accuracy_threshold_met: Option<bool>,
    runtime_overall: JsonRuntimeSummary,
    by_family: BTreeMap<String, JsonSliceStats>,
    by_error_by_family: BTreeMap<String, BTreeMap<String, usize>>,
    by_density: BTreeMap<String, JsonSliceStats>,
    by_size: BTreeMap<String, JsonSliceStats>,
    by_outcome: BTreeMap<String, JsonSliceStats>,
    by_backend: BTreeMap<String, JsonSliceStats>,
    runtime_by_family: BTreeMap<String, JsonRuntimeSummary>,
    runtime_by_size: BTreeMap<String, JsonRuntimeSummary>,
    runtime_by_backend: BTreeMap<String, JsonRuntimeSummary>,
    by_error: BTreeMap<String, usize>,
    oracle_runtime_case: Option<JsonOracleRuntimeCaseReport>,
    oracle_representative_case: Option<JsonOracleRuntimeCaseReport>,
}

#[derive(Serialize)]
struct JsonClassifierAssessment {
    conclusive: usize,
    inconclusive: usize,
    mismatch_count: usize,
    mismatches: Vec<String>,
    inconclusive_cases: Vec<String>,
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

#[derive(Clone, Debug)]
struct OracleRuntimeCaseReport {
    pattern: &'static str,
    target_generation: u64,
    final_generation: u64,
    reached_target: bool,
    runtime: Duration,
    classification: Classification,
    backend: SimulationBackend,
    population: usize,
    bounds_span: Coord,
}

#[derive(Serialize)]
struct JsonOracleRuntimeCaseReport {
    pattern: String,
    target_generation: u64,
    final_generation: u64,
    reached_target: bool,
    runtime_us: u128,
    classification: String,
    backend: String,
    population: usize,
    bounds_span: Coord,
}
