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
use crate::generators::{mix_seed, pattern_by_name, random_soup};
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};
use crate::oracle::{OracleSession, OracleStateMetrics, OracleStepPlan};

type SeenStates = HashMap<NormalizedGridSignature, (u64, Cell)>;

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
}

const BENCHMARK_PREDICTION_MAX_GENERATIONS: u64 = 1_024;
const DEFAULT_BENCHMARK_ORACLE_MAX_GENERATIONS: u64 = 1_000_000;
const DEFAULT_ORACLE_MAX_JUMP_TARGET: u64 = 10_000_000;
const DEFAULT_BENCHMARK_RUN_SEED: u64 = 0x5EED_2026_D15C_A11F;
const DEFAULT_CASES_PER_FAMILY: usize = 4;
const RANDOMIZED_COMPATIBILITY_THRESHOLD: f64 = 0.90;
#[cfg(test)]
const TEST_ORACLE_MAX_GENERATIONS: u64 = 4_096;

pub fn run_benchmark_report(format: BenchmarkFormat, options: &BenchmarkOptions) {
    let report = build_benchmark_report(options);

    match format {
        BenchmarkFormat::Text => report.print(),
        BenchmarkFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report.to_json()).unwrap()
            );
        }
    }
}

fn build_benchmark_report(options: &BenchmarkOptions) -> BenchmarkReport {
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
        max_population: 20_000,
        max_bounding_box: Coord::MAX,
    };
    let oracle_limits = ClassificationLimits {
        max_generations: oracle_max_generations,
        max_population: 5_000_000,
        max_bounding_box: Coord::MAX,
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
            &actual,
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
        report.accuracy_threshold_met =
            Some(report.compatible as f64 / report.total.max(1) as f64 >= threshold);
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
    serde_json::to_value(report.to_json()).unwrap()
}

#[cfg(test)]
pub(crate) fn benchmark_report_json_for_options(options: &BenchmarkOptions) -> serde_json::Value {
    serde_json::to_value(build_benchmark_report(options).to_json()).unwrap()
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

#[cfg(test)]
pub(crate) fn oracle_runtime_case_for_tests() -> (String, u64, bool, usize, Coord) {
    special_oracle_case_for_tests(true)
}

#[cfg(test)]
pub(crate) fn oracle_representative_case_for_tests() -> (String, u64, bool, usize, Coord) {
    special_oracle_case_for_tests(false)
}

pub(crate) fn oracle_representative_seed_grid_for_tests() -> BitGrid {
    random_soup(53, 24, 37, 420)
}

#[cfg(test)]
pub(crate) fn oracle_extinction_seed_grid_for_tests() -> BitGrid {
    distant_glider_trigger(768, pattern_by_name("blinker").unwrap(), (0, 0))
}

#[cfg(test)]
fn special_oracle_case_for_tests(runtime_case: bool) -> (String, u64, bool, usize, Coord) {
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

pub(crate) fn bitmask_pattern(mask: u32, width: Coord, height: Coord) -> BitGrid {
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

#[allow(dead_code)]
pub(crate) fn reference_classify(seed: &BitGrid, limits: &ClassificationLimits) -> Classification {
    let mut seen: SeenStates = HashMap::new();
    let mut grid = seed.clone();
    let mut generation = 0;
    let mut generation_limit = effective_generation_limit(limits, grid.population(), grid.bounds());

    while generation <= generation_limit {
        let (signature, origin) = normalize(&grid);
        if grid.is_empty() {
            return Classification::DiesOut {
                at_generation: generation,
            };
        }

        if let Some(&(first_seen, first_origin)) = seen.get(&signature) {
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

        if let Some(bounds) = grid.bounds() {
            let (width, height, _) = bounds_dimensions(bounds);
            if width > limits.max_bounding_box || height > limits.max_bounding_box {
                return Classification::LikelyInfinite {
                    reason: "expanding_bounds",
                    detected_at: generation,
                };
            }
        }

        seen.insert(signature, (generation, origin));
        grid = reference_step_grid(&grid);
        generation += 1;

        if generation > generation_limit {
            generation_limit = effective_generation_limit(limits, grid.population(), grid.bounds());
        }
    }

    Classification::Unknown {
        simulated: generation_limit,
    }
}

pub(crate) fn reference_classify_from_checkpoint(
    checkpoint: ClassificationCheckpoint,
    prediction: &Classification,
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
        return prediction.clone();
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
        oracle_limits,
    )
}

#[cfg(test)]
pub(crate) fn reference_is_decisive(classification: &Classification) -> bool {
    !matches!(classification, Classification::Unknown { .. })
}

#[cfg(test)]
pub(crate) fn assert_same_outcome(name: &str, expected: &Classification, actual: &Classification) {
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
struct BenchmarkReport {
    total: usize,
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
    run_seed: Option<u64>,
    accuracy_threshold: Option<f64>,
    accuracy_threshold_met: Option<bool>,
    oracle_runtime_case: Option<OracleRuntimeCaseReport>,
    oracle_representative_case: Option<OracleRuntimeCaseReport>,
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

impl BenchmarkReport {
    fn record(
        &mut self,
        case: &BenchmarkCase,
        expected: &Classification,
        actual: &Classification,
        runtime: Duration,
        planned_generations: u64,
    ) {
        let outcome = outcome_bucket(expected);
        let error = error_bucket(expected, actual);
        let compatible = error == ErrorBucket::ExactOrCompatible;
        let backend = select_backend(&case.grid, planned_generations);

        self.total += 1;
        if compatible {
            self.compatible += 1;
        }
        record_slice(&mut self.by_family, case.family, compatible);
        *self
            .by_error_by_family
            .entry(case.family)
            .or_default()
            .entry(error)
            .or_insert(0) += 1;
        record_slice(&mut self.by_density, case.density_percent, compatible);
        record_slice(&mut self.by_size, case.size, compatible);
        record_slice(&mut self.by_outcome, outcome, compatible);
        record_slice(&mut self.by_backend, backend, compatible);
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
        self.runtime_by_backend
            .entry(backend)
            .or_default()
            .push(runtime);
        if matches!(actual, Classification::Unknown { .. }) {
            self.unknown_count += 1;
        }

        if !compatible {
            println!(
                "mismatch case={} family={} size={} density={} expected={} actual={}{}",
                case.name,
                case.family,
                case.size,
                case.density_percent,
                expected,
                actual,
                case.replay
                    .as_ref()
                    .map(|replay| format!(" replay={replay}"))
                    .unwrap_or_default()
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
        if let Some(mode) = self.mode {
            println!("mode={mode}");
        }
        if let Some(run_seed) = self.run_seed {
            println!("run_seed={run_seed}");
        }
        if let Some(threshold_met) = self.accuracy_threshold_met {
            println!(
                "accuracy_threshold target={:.3} met={}",
                self.accuracy_threshold
                    .unwrap_or(RANDOMIZED_COMPATIBILITY_THRESHOLD),
                threshold_met
            );
        }
        print_runtime_summary("runtime_overall", &self.runtimes);
        print_slice_map("family", &self.by_family);
        print_nested_error_map("error_by_family", &self.by_error_by_family);
        print_slice_map("density", &self.by_density);
        print_slice_map("size", &self.by_size);
        print_slice_map("outcome", &self.by_outcome);
        print_slice_map("backend", &self.by_backend);
        print_runtime_map("runtime_by_family", &self.runtime_by_family);
        print_runtime_map("runtime_by_size", &self.runtime_by_size);
        print_runtime_map("runtime_by_backend", &self.runtime_by_backend);
        println!("error_buckets:");
        for (bucket, count) in &self.by_error {
            println!("  {}: {}", bucket, count);
        }
        if let Some(case) = &self.oracle_runtime_case {
            println!("oracle_runtime_case:");
            println!(
                "  pattern={} target_gen={} final_gen={} reached_target={} runtime={:?} classification={} backend={} population={} bounds_span={}",
                case.pattern,
                case.target_generation,
                case.final_generation,
                case.reached_target,
                case.runtime,
                case.classification,
                case.backend,
                case.population,
                case.bounds_span
            );
        }
        if let Some(case) = &self.oracle_representative_case {
            println!("oracle_representative_case:");
            println!(
                "  pattern={} target_gen={} final_gen={} reached_target={} runtime={:?} classification={} backend={} population={} bounds_span={}",
                case.pattern,
                case.target_generation,
                case.final_generation,
                case.reached_target,
                case.runtime,
                case.classification,
                case.backend,
                case.population,
                case.bounds_span
            );
        }
    }

    fn to_json(&self) -> JsonReport {
        JsonReport {
            total: self.total,
            compatible: self.compatible,
            mode: self.mode.map(|mode| mode.to_string()),
            run_seed: self.run_seed,
            unknown_rate: self.unknown_count as f64 / self.total.max(1) as f64,
            overall_summary: JsonSliceStats {
                total: self.total,
                compatible: self.compatible,
                accuracy: self.compatible as f64 / self.total.max(1) as f64,
            },
            accuracy_threshold: self.accuracy_threshold,
            accuracy_threshold_met: self.accuracy_threshold_met,
            runtime_overall: runtime_summary_json(&self.runtimes),
            by_family: slice_map_json(&self.by_family),
            by_error_by_family: nested_error_map_json(&self.by_error_by_family),
            by_density: slice_map_json(&self.by_density),
            by_size: slice_map_json(&self.by_size),
            by_outcome: slice_map_json(&self.by_outcome),
            by_backend: slice_map_json(&self.by_backend),
            runtime_by_family: runtime_map_json(&self.runtime_by_family),
            runtime_by_size: runtime_map_json(&self.runtime_by_size),
            runtime_by_backend: runtime_map_json(&self.runtime_by_backend),
            by_error: self
                .by_error
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            oracle_runtime_case: self.oracle_runtime_case.as_ref().map(|case| {
                JsonOracleRuntimeCaseReport {
                    pattern: case.pattern.to_string(),
                    target_generation: case.target_generation,
                    final_generation: case.final_generation,
                    reached_target: case.reached_target,
                    runtime_us: case.runtime.as_micros(),
                    classification: case.classification.to_string(),
                    backend: case.backend.to_string(),
                    population: case.population,
                    bounds_span: case.bounds_span,
                }
            }),
            oracle_representative_case: self.oracle_representative_case.as_ref().map(|case| {
                JsonOracleRuntimeCaseReport {
                    pattern: case.pattern.to_string(),
                    target_generation: case.target_generation,
                    final_generation: case.final_generation,
                    reached_target: case.reached_target,
                    runtime_us: case.runtime.as_micros(),
                    classification: case.classification.to_string(),
                    backend: case.backend.to_string(),
                    population: case.population,
                    bounds_span: case.bounds_span,
                }
            }),
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

fn print_slice_map<K: fmt::Display + Ord>(label: &str, map: &BTreeMap<K, SliceStats>) {
    println!("{label}:");
    for (key, stats) in map {
        println!(
            "  {}: {}/{} ({:.3})",
            key,
            stats.compatible,
            stats.total,
            stats.compatible as f64 / stats.total.max(1) as f64
        );
    }
}

fn print_runtime_map<K: fmt::Display + Ord>(label: &str, map: &BTreeMap<K, Vec<Duration>>) {
    println!("{label}:");
    for (key, samples) in map {
        print!("  {}: ", key);
        print_runtime_summary_inline(samples);
    }
}

fn print_nested_error_map<K: fmt::Display + Ord>(
    label: &str,
    map: &BTreeMap<K, BTreeMap<ErrorBucket, usize>>,
) {
    println!("{label}:");
    for (key, buckets) in map {
        println!("  {}:", key);
        for (bucket, count) in buckets {
            println!("    {}: {}", bucket, count);
        }
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
    if samples.is_empty() {
        return JsonRuntimeSummary {
            median_us: 0,
            p90_us: 0,
            p95_us: 0,
            p99_us: 0,
            worst_us: 0,
        };
    }
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

fn slice_map_json<K: fmt::Display + Ord>(
    map: &BTreeMap<K, SliceStats>,
) -> BTreeMap<String, JsonSliceStats> {
    map.iter()
        .map(|(key, stats)| {
            (
                key.to_string(),
                JsonSliceStats {
                    total: stats.total,
                    compatible: stats.compatible,
                    accuracy: stats.compatible as f64 / stats.total.max(1) as f64,
                },
            )
        })
        .collect()
}

fn runtime_map_json<K: fmt::Display + Ord>(
    map: &BTreeMap<K, Vec<Duration>>,
) -> BTreeMap<String, JsonRuntimeSummary> {
    map.iter()
        .map(|(key, samples)| (key.to_string(), runtime_summary_json(samples)))
        .collect()
}

fn nested_error_map_json<K: fmt::Display + Ord>(
    map: &BTreeMap<K, BTreeMap<ErrorBucket, usize>>,
) -> BTreeMap<String, BTreeMap<String, usize>> {
    map.iter()
        .map(|(key, buckets)| {
            (
                key.to_string(),
                buckets
                    .iter()
                    .map(|(bucket, count)| (bucket.to_string(), *count))
                    .collect(),
            )
        })
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

fn benchmark_family_filter(options: &BenchmarkOptions) -> Option<HashSet<BenchmarkFamily>> {
    let tokens = options.families.as_ref()?;
    let mut families = HashSet::new();
    for token in tokens {
        if let Some(family) = parse_family(token) {
            families.insert(family);
        }
    }
    Some(families)
}

fn benchmark_run_mode_and_seed(options: &BenchmarkOptions) -> (BenchmarkRunMode, u64) {
    match (options.randomized, options.seed) {
        (false, None) => (BenchmarkRunMode::DefaultSeeded, DEFAULT_BENCHMARK_RUN_SEED),
        (_, Some(seed)) => (BenchmarkRunMode::Seeded, seed),
        (true, None) => (BenchmarkRunMode::TimeSeeded, time_seed()),
    }
}

fn parse_family(token: &str) -> Option<BenchmarkFamily> {
    match token {
        "iid" => Some(BenchmarkFamily::IidRandom),
        "structured" => Some(BenchmarkFamily::StructuredRandom),
        "clustered" => Some(BenchmarkFamily::ClusteredNoise),
        "smallbox" => Some(BenchmarkFamily::ExhaustiveSmallBox),
        "smallbox_5x5" => Some(BenchmarkFamily::ExhaustiveFiveByFive),
        "methuselah" => Some(BenchmarkFamily::LongLivedMethuselah),
        "mover" => Some(BenchmarkFamily::TranslatedPeriodicMover),
        "gun" => Some(BenchmarkFamily::GunPufferBreeder),
        "delayed" => Some(BenchmarkFamily::DelayedInteraction),
        "ash" => Some(BenchmarkFamily::DeceptiveAsh),
        "gadget" => Some(BenchmarkFamily::ComputationalGadget),
        "emitter" => Some(BenchmarkFamily::EmitterInteraction),
        _ => None,
    }
}

fn seeded_benchmark_suite(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut suite = Vec::new();
    suite.extend(seeded_iid_cases(run_seed, cases_per_family));
    suite.extend(seeded_structured_cases(run_seed, cases_per_family));
    suite.extend(seeded_clustered_cases(run_seed, cases_per_family));
    suite.extend(seeded_methuselah_cases(run_seed, cases_per_family));
    suite.extend(seeded_mover_cases(run_seed, cases_per_family));
    suite.extend(seeded_gun_cases(run_seed, cases_per_family));
    suite.extend(seeded_delayed_cases(run_seed, cases_per_family));
    suite.extend(seeded_ash_cases(run_seed, cases_per_family));
    suite.extend(seeded_gadget_cases(run_seed, cases_per_family));
    suite.extend(seeded_emitter_cases(run_seed, cases_per_family));
    suite
}

fn seeded_case(
    name: String,
    family: BenchmarkFamily,
    size: i32,
    density_percent: u32,
    grid: BitGrid,
    replay: Option<String>,
) -> BenchmarkCase {
    BenchmarkCase {
        name,
        family,
        size,
        density_percent,
        grid,
        replay,
    }
}

fn seeded_iid_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ 0x1000_0000_0000_0000 ^ idx as u64);
        let size = pick_from(&[16_i64, 32, 64, 128, 256], case_seed, 0);
        let density = pick_from(&[5_u32, 10, 20, 30, 50], case_seed, 1);
        let grid_seed = mix_seed(case_seed ^ 0xA11D_D00D);
        let grid = random_soup(size, size, density, grid_seed);
        cases.push(seeded_case(
            format!("iid_s{size}_d{density}_seed{grid_seed}"),
            BenchmarkFamily::IidRandom,
            i32::try_from(size).expect("benchmark size exceeded i32"),
            density,
            grid,
            Some(format!(
                "family=iid,size={size},density={density},seed={grid_seed}"
            )),
        ));
    }
    cases
}

fn seeded_structured_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ 0x2000_0000_0000_0000 ^ idx as u64);
        let size = pick_from(&[16_i64, 32, 64, 128, 192], case_seed, 0);
        let grid_seed = mix_seed(case_seed ^ 0x51DE_BAAD);
        let grid = structured_random_soup(size, size, grid_seed);
        cases.push(seeded_case(
            format!("structured_s{size}_seed{grid_seed}"),
            BenchmarkFamily::StructuredRandom,
            i32::try_from(size).expect("benchmark size exceeded i32"),
            estimate_density_percent(&grid),
            grid,
            Some(format!("family=structured,size={size},seed={grid_seed}")),
        ));
    }
    cases
}

fn seeded_clustered_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ 0x3000_0000_0000_0000 ^ idx as u64);
        let size = pick_from(&[24_i64, 48, 96, 144], case_seed, 0);
        let density = pick_from(&[8_u32, 12, 18, 24, 30], case_seed, 1);
        let grid_seed = mix_seed(case_seed ^ 0xC1A5_7EED);
        let grid = clustered_noise_soup(size, size, density, grid_seed);
        cases.push(seeded_case(
            format!("clustered_s{size}_d{density}_seed{grid_seed}"),
            BenchmarkFamily::ClusteredNoise,
            i32::try_from(size).expect("benchmark size exceeded i32"),
            density,
            grid,
            Some(format!(
                "family=clustered,size={size},density={density},seed={grid_seed}"
            )),
        ));
    }
    cases
}

fn seeded_delayed_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ 0x4000_0000_0000_0000 ^ idx as u64);
        let distance = pick_from(&[24_i64, 48, 96, 192, 384, 768], case_seed, 0);
        let variant = pick_from(&[0_u32, 1, 2], case_seed, 1);
        let grid = match variant {
            0 => head_on_glider_collision(distance),
            1 => distant_glider_trigger(distance, pattern_by_name("block").unwrap(), (0, 0)),
            _ => distant_glider_trigger(distance, pattern_by_name("blinker").unwrap(), (0, 0)),
        };
        let variant_name = match variant {
            0 => "head_on_gliders",
            1 => "block_trigger",
            _ => "blinker_trigger",
        };
        cases.push(seeded_case(
            format!("delayed_{variant_name}_d{distance}_seed{case_seed}"),
            BenchmarkFamily::DelayedInteraction,
            i32::try_from(
                grid.bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
            )
            .expect("benchmark size exceeded i32"),
            estimate_density_percent(&grid),
            grid,
            Some(format!(
                "family=delayed,variant={variant_name},distance={distance},seed={case_seed}"
            )),
        ));
    }
    cases
}

fn seeded_gadget_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    let glider = pattern_by_name("glider").unwrap();
    let block = pattern_by_name("block").unwrap();
    let blinker = pattern_by_name("blinker").unwrap();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ 0x5000_0000_0000_0000 ^ idx as u64);
        let dx = (case_seed % 64) as Coord;
        let dy = ((case_seed >> 6) % 48) as Coord;
        let block_dx = 20 + ((case_seed >> 12) % 32) as Coord;
        let block_dy = 8 + ((case_seed >> 17) % 32) as Coord;
        let blink_dx = 40 + ((case_seed >> 22) % 32) as Coord;
        let blink_dy = 16 + ((case_seed >> 27) % 32) as Coord;
        let mut cells = glider
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + dx, y + dy))
            .collect::<Vec<_>>();
        cells.extend(
            block
                .live_cells()
                .into_iter()
                .map(|(x, y)| (x + block_dx, y + block_dy)),
        );
        cells.extend(
            blinker
                .live_cells()
                .into_iter()
                .map(|(x, y)| (x + blink_dx, y + blink_dy)),
        );
        let grid = BitGrid::from_cells(&cells);
        cases.push(seeded_case(
            format!("gadget_seed{case_seed}_g{dx}_{dy}_b{block_dx}_{block_dy}_l{blink_dx}_{blink_dy}"),
            BenchmarkFamily::ComputationalGadget,
            i32::try_from(grid.bounds().map(|(min_x, _, max_x, _)| max_x - min_x + 1).unwrap_or(0))
                .expect("benchmark size exceeded i32"),
            estimate_density_percent(&grid),
            grid,
            Some(format!(
                "family=gadget,glider=({dx},{dy}),block=({block_dx},{block_dy}),blinker=({blink_dx},{blink_dy}),seed={case_seed}"
            )),
        ));
    }
    cases
}

fn seeded_emitter_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    let gun = pattern_by_name("gosper_glider_gun").unwrap();
    let blocker = pattern_by_name("block").unwrap();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ 0x6000_0000_0000_0000 ^ idx as u64);
        let block_dx = 90 + ((case_seed >> 8) % 160) as Coord;
        let block_dy = 10 + ((case_seed >> 20) % 80) as Coord;
        let mut cells = gun.live_cells();
        cells.extend(
            blocker
                .live_cells()
                .into_iter()
                .map(|(x, y)| (x + block_dx, y + block_dy)),
        );
        let grid = BitGrid::from_cells(&cells);
        cases.push(seeded_case(
            format!("emitter_seed{case_seed}_b{block_dx}_{block_dy}"),
            BenchmarkFamily::EmitterInteraction,
            i32::try_from(
                grid.bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
            )
            .expect("benchmark size exceeded i32"),
            estimate_density_percent(&grid),
            grid,
            Some(format!(
                "family=emitter,blocker=({block_dx},{block_dy}),seed={case_seed}"
            )),
        ));
    }
    cases
}

fn time_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => mix_seed(duration.as_secs() ^ u64::from(duration.subsec_nanos())),
        Err(_) => mix_seed(0xD1CE_F00D_1234_5678),
    }
}

fn run_exhaustive_5x5(
    report: &mut BenchmarkReport,
    oracle_simulation: &mut crate::engine::SimulationSession,
) {
    let prediction_limits = ClassificationLimits {
        max_generations: 128,
        max_population: 10_000,
        max_bounding_box: Coord::MAX,
    };
    let oracle_limits = ClassificationLimits {
        max_generations: DEFAULT_BENCHMARK_ORACLE_MAX_GENERATIONS,
        max_population: 5_000_000,
        max_bounding_box: Coord::MAX,
    };

    for mask in 0_u32..(1_u32 << 25) {
        if canonical_small_box_mask(mask, 5, 5) != mask {
            continue;
        }
        let grid = bitmask_pattern(mask, 5, 5);
        let started = Instant::now();
        let (actual, checkpoint) =
            predict_seed_with_checkpoint(&grid, &prediction_limits, &mut Memo::default());
        let expected = reference_classify_from_checkpoint(
            checkpoint,
            &actual,
            &prediction_limits,
            &oracle_limits,
            oracle_simulation,
        );
        if !reference_is_decisive_runtime(&expected) {
            continue;
        }
        let case = seeded_case(
            format!("5x5_{mask}"),
            BenchmarkFamily::ExhaustiveFiveByFive,
            5,
            estimate_density_percent(&grid),
            grid,
            None,
        );
        report.record(
            &case,
            &expected,
            &actual,
            started.elapsed(),
            prediction_limits.max_generations,
        );
    }
}

fn run_oracle_runtime_case(
    target_generation: u64,
    progress: bool,
    oracle_simulation: &mut crate::engine::SimulationSession,
) -> OracleRuntimeCaseReport {
    let pattern = "gosper_glider_gun";
    let backend = select_backend(
        &pattern_by_name(pattern).expect("known pattern must exist"),
        target_generation,
    );
    let grid = pattern_by_name(pattern).expect("known pattern must exist");
    let started = Instant::now();
    let mut progress_logger = OracleProgressLogger::new(progress, pattern, target_generation);
    let outcome = reference_classify_to_generation_target(
        grid,
        target_generation,
        Some(&mut progress_logger),
        oracle_simulation,
    );
    let runtime = started.elapsed();
    progress_logger.finish(outcome.final_generation, runtime, &outcome.classification);

    OracleRuntimeCaseReport {
        pattern,
        target_generation,
        final_generation: outcome.final_generation,
        reached_target: outcome.final_generation == target_generation,
        runtime,
        classification: outcome.classification,
        backend,
        population: outcome.population,
        bounds_span: outcome.bounds_span,
    }
}

fn run_oracle_representative_case(
    target_generation: u64,
    progress: bool,
    oracle_simulation: &mut crate::engine::SimulationSession,
) -> OracleRuntimeCaseReport {
    let pattern = "random_seed420_53x24_37";
    let grid = oracle_representative_seed_grid_for_tests();
    let backend = select_backend(&grid, target_generation);
    let started = Instant::now();
    let mut progress_logger = OracleProgressLogger::new(progress, pattern, target_generation);
    let outcome = reference_classify_to_generation_target(
        grid,
        target_generation,
        Some(&mut progress_logger),
        oracle_simulation,
    );
    let runtime = started.elapsed();
    progress_logger.finish(outcome.final_generation, runtime, &outcome.classification);

    OracleRuntimeCaseReport {
        pattern,
        target_generation,
        final_generation: outcome.final_generation,
        reached_target: outcome.final_generation == target_generation,
        runtime,
        classification: outcome.classification,
        backend,
        population: outcome.population,
        bounds_span: outcome.bounds_span,
    }
}

fn reference_classify_to_generation_target(
    seed: BitGrid,
    target_generation: u64,
    mut progress: Option<&mut OracleProgressLogger>,
    oracle_simulation: &mut crate::engine::SimulationSession,
) -> crate::oracle::OracleRuntimeOutcome {
    if let Some(progress_logger) = progress.as_mut() {
        progress_logger.log(
            0,
            OracleStateMetrics {
                population: seed.population(),
                bounds_span: grid_bounds_span(&seed),
            },
            None,
        );
    }
    let session = OracleSession::new(seed, 0, HashMap::new(), oracle_simulation);
    if let Some(progress_logger) = progress {
        let mut step_logger = |plan: OracleStepPlan, metrics: OracleStateMetrics| {
            if plan.step_span == 0 {
                progress_logger.log(plan.generation, metrics, Some(plan.backend));
            } else {
                progress_logger.log_planned_step(
                    plan.generation,
                    plan.step_span,
                    plan.backend,
                    metrics,
                );
            }
        };
        session.advance_runtime_target(target_generation, Some(&mut step_logger))
    } else {
        session.advance_runtime_target(target_generation, None)
    }
}

struct OracleProgressLogger {
    enabled: bool,
    pattern: &'static str,
    target_generation: u64,
    started: Instant,
    next_log_at: Instant,
    pending_step: Option<PendingOracleStep>,
    backend_runtime: BTreeMap<SimulationBackend, Duration>,
    jump_runtime: BTreeMap<u64, JumpRuntimeStats>,
}

impl OracleProgressLogger {
    fn new(enabled: bool, pattern: &'static str, target_generation: u64) -> Self {
        let started = Instant::now();
        Self {
            enabled,
            pattern,
            target_generation,
            started,
            next_log_at: started,
            pending_step: None,
            backend_runtime: BTreeMap::new(),
            jump_runtime: BTreeMap::new(),
        }
    }

    fn log(
        &mut self,
        generation: u64,
        metrics: OracleStateMetrics,
        backend: Option<SimulationBackend>,
    ) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if now < self.next_log_at && generation < self.target_generation {
            return;
        }
        self.next_log_at = now + Duration::from_secs(2);
        let elapsed = now.duration_since(self.started);
        let progress = if self.target_generation == 0 {
            1.0
        } else {
            generation as f64 / self.target_generation as f64
        };
        let progress_percent = (progress * 100.0).min(100.0);
        let generations_per_second = if elapsed.is_zero() {
            0.0
        } else {
            generation as f64 / elapsed.as_secs_f64()
        };
        let remaining_generations = self.target_generation.saturating_sub(generation);
        let eta_seconds = if generations_per_second > 0.0 {
            remaining_generations as f64 / generations_per_second
        } else {
            f64::INFINITY
        };
        let backend_text = backend
            .map(|current_backend| format!(" backend={current_backend:?}"))
            .unwrap_or_default();

        eprintln!(
            "[oracle-max-jump] pattern={} gen={}/{} ({:.2}%) elapsed={:?} eta={} population={} bounds_span={}{}",
            self.pattern,
            generation,
            self.target_generation,
            progress_percent,
            elapsed,
            format_eta_seconds(eta_seconds),
            metrics.population,
            metrics.bounds_span,
            backend_text
        );
    }

    fn finish(&mut self, generation: u64, runtime: Duration, classification: &Classification) {
        if !self.enabled {
            return;
        }
        if let Some(pending) = self.pending_step.take() {
            self.record_step_runtime(
                pending.backend,
                pending.step_span,
                pending.started.elapsed(),
            );
        }
        eprintln!(
            "[oracle-max-jump] finished pattern={} gen={}/{} runtime={:?} classification={}",
            self.pattern, generation, self.target_generation, runtime, classification
        );
        self.print_runtime_profile();
    }

    fn log_planned_step(
        &mut self,
        generation: u64,
        step_span: u64,
        backend: SimulationBackend,
        metrics: OracleStateMetrics,
    ) {
        if !self.enabled || step_span <= 1 {
            return;
        }
        if let Some(pending) = self.pending_step.take() {
            self.record_step_runtime(
                pending.backend,
                pending.step_span,
                pending.started.elapsed(),
            );
        }
        self.pending_step = Some(PendingOracleStep {
            step_span,
            backend,
            started: Instant::now(),
        });
        eprintln!(
            "[oracle-max-jump] planning pattern={} gen={} next_step={} backend={} population={} bounds_span={}",
            self.pattern, generation, step_span, backend, metrics.population, metrics.bounds_span
        );
    }

    fn record_step_runtime(
        &mut self,
        backend: SimulationBackend,
        step_span: u64,
        elapsed: Duration,
    ) {
        *self.backend_runtime.entry(backend).or_default() += elapsed;
        let stats = self.jump_runtime.entry(step_span).or_default();
        stats.count += 1;
        stats.total += elapsed;
    }

    fn print_runtime_profile(&self) {
        if !self.enabled {
            return;
        }
        for (backend, runtime) in &self.backend_runtime {
            eprintln!("[oracle-max-jump] profile backend={backend} runtime={runtime:?}");
        }
        for (step_span, stats) in self.jump_runtime.iter().rev().take(12) {
            eprintln!(
                "[oracle-max-jump] profile step={} count={} runtime={:?}",
                step_span, stats.count, stats.total
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingOracleStep {
    step_span: u64,
    backend: SimulationBackend,
    started: Instant,
}

#[derive(Clone, Copy, Debug, Default)]
struct JumpRuntimeStats {
    count: usize,
    total: Duration,
}

fn format_eta_seconds(eta_seconds: f64) -> String {
    if !eta_seconds.is_finite() {
        return "unknown".to_string();
    }
    let total_seconds = eta_seconds.max(0.0).round() as u64;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn exhaustive_small_box_cases() -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for mask in 1_u32..(1_u32 << 12) {
        let grid = bitmask_pattern(mask, 4, 3);
        cases.push(seeded_case(
            format!("smallbox_{mask}"),
            BenchmarkFamily::ExhaustiveSmallBox,
            4,
            estimate_density_percent(&grid),
            grid,
            None,
        ));
    }
    cases
}

fn seeded_methuselah_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    seeded_translated_pattern_cases(
        run_seed,
        cases_per_family,
        BenchmarkFamily::LongLivedMethuselah,
        &["acorn", "diehard", "r_pentomino"],
        0x7000_0000_0000_0000,
    )
}

fn seeded_mover_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    seeded_translated_pattern_cases(
        run_seed,
        cases_per_family,
        BenchmarkFamily::TranslatedPeriodicMover,
        &["glider", "blinker"],
        0x7100_0000_0000_0000,
    )
}

fn seeded_gun_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    seeded_translated_pattern_cases(
        run_seed,
        cases_per_family,
        BenchmarkFamily::GunPufferBreeder,
        &[
            "gosper_glider_gun",
            "glider_producing_switch_engine",
            "blinker_puffer_1",
        ],
        0x7200_0000_0000_0000,
    )
}

fn seeded_ash_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let traffic_jam = BitGrid::from_cells(&[
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
    ]);
    seeded_translated_grid_cases(
        run_seed,
        cases_per_family,
        BenchmarkFamily::DeceptiveAsh,
        &[
            ("traffic_jam", traffic_jam),
            ("block", pattern_by_name("block").unwrap()),
        ],
        0x7300_0000_0000_0000,
    )
}

fn seeded_translated_pattern_cases(
    run_seed: u64,
    cases_per_family: usize,
    family: BenchmarkFamily,
    names: &[&str],
    salt: u64,
) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ salt ^ idx as u64);
        let name = pick_from(names, case_seed, 0);
        let base = pattern_by_name(name).unwrap();
        let (dx, dy) = seeded_translation(case_seed);
        let grid = translate_grid(&base, dx, dy);
        cases.push(seeded_case(
            format!("{family}_{name}_dx{dx}_dy{dy}_seed{case_seed}"),
            family,
            i32::try_from(
                grid.bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
            )
            .expect("benchmark size exceeded i32"),
            estimate_density_percent(&grid),
            grid,
            Some(format!(
                "family={family},pattern={name},dx={dx},dy={dy},seed={case_seed}"
            )),
        ));
    }
    cases
}

fn seeded_translated_grid_cases(
    run_seed: u64,
    cases_per_family: usize,
    family: BenchmarkFamily,
    bases: &[(&str, BitGrid)],
    salt: u64,
) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    for idx in 0..cases_per_family {
        let case_seed = mix_seed(run_seed ^ salt ^ idx as u64);
        let (name, base) = &bases[(mix_seed(case_seed) as usize) % bases.len()];
        let (dx, dy) = seeded_translation(case_seed);
        let grid = translate_grid(base, dx, dy);
        cases.push(seeded_case(
            format!("{family}_{name}_dx{dx}_dy{dy}_seed{case_seed}"),
            family,
            i32::try_from(
                grid.bounds()
                    .map(|(min_x, _, max_x, _)| max_x - min_x + 1)
                    .unwrap_or(0),
            )
            .expect("benchmark size exceeded i32"),
            estimate_density_percent(&grid),
            grid,
            Some(format!(
                "family={family},pattern={name},dx={dx},dy={dy},seed={case_seed}"
            )),
        ));
    }
    cases
}

fn head_on_glider_collision(distance: Coord) -> BitGrid {
    let southeast_glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let northwest_glider = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)];
    let mut cells = offset_cells(&southeast_glider, 0, 0);
    cells.extend(offset_cells(&northwest_glider, distance, distance));
    BitGrid::from_cells(&cells)
}

fn distant_glider_trigger(distance: Coord, target: BitGrid, target_origin: Cell) -> BitGrid {
    let glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let mut cells = target
        .live_cells()
        .into_iter()
        .map(|(x, y)| (x + target_origin.0, y + target_origin.1))
        .collect::<Vec<_>>();
    cells.extend(offset_cells(
        &glider,
        target_origin.0 - distance,
        target_origin.1 - distance,
    ));
    BitGrid::from_cells(&cells)
}

fn offset_cells(cells: &[Cell], dx: Coord, dy: Coord) -> Vec<Cell> {
    cells.iter().map(|&(x, y)| (x + dx, y + dy)).collect()
}

fn seeded_translation(seed: u64) -> (Coord, Coord) {
    let dx = ((seed >> 8) % 48) as Coord;
    let dy = ((seed >> 20) % 48) as Coord;
    (dx, dy)
}

fn translate_grid(grid: &BitGrid, dx: Coord, dy: Coord) -> BitGrid {
    let cells = grid
        .live_cells()
        .into_iter()
        .map(|(x, y)| (x + dx, y + dy))
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn estimate_density_percent(grid: &BitGrid) -> u32 {
    let Some((min_x, min_y, max_x, max_y)) = grid.bounds() else {
        return 0;
    };
    let area = ((max_x - min_x + 1) * (max_y - min_y + 1)).max(1) as usize;
    ((grid.population() * 100) / area) as u32
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

fn pick_from<T: Copy>(values: &[T], seed: u64, salt: u64) -> T {
    values[(mix_seed(seed ^ salt) as usize) % values.len()]
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

fn reference_is_decisive_runtime(classification: &Classification) -> bool {
    !matches!(classification, Classification::Unknown { .. })
}

pub(crate) fn effective_generation_limit(
    limits: &ClassificationLimits,
    population: usize,
    bounds: Option<(Coord, Coord, Coord, Coord)>,
) -> u64 {
    const SMALL_PATTERN_POPULATION: usize = 64;
    const SMALL_PATTERN_SPAN: Coord = 24;
    const MIN_EXTENDED_LIMIT: u64 = 1024;
    const MAX_EXTENDED_LIMIT: u64 = 2048;

    let Some((min_x, min_y, max_x, max_y)) = bounds else {
        return limits.max_generations;
    };
    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;

    if population <= SMALL_PATTERN_POPULATION
        && width <= SMALL_PATTERN_SPAN
        && height <= SMALL_PATTERN_SPAN
    {
        return limits
            .max_generations
            .clamp(MIN_EXTENDED_LIMIT, MAX_EXTENDED_LIMIT);
    }

    limits.max_generations
}

pub(crate) fn canonical_small_box_mask(mask: u32, width: usize, height: usize) -> u32 {
    let mut best = transform_mask(mask, width, height, 0);
    for transform in 1..8 {
        best = best.min(transform_mask(mask, width, height, transform));
    }
    best
}

fn transform_mask(mask: u32, width: usize, height: usize, transform: usize) -> u32 {
    let mut transformed = 0_u32;
    for y in 0..height {
        for x in 0..width {
            let bit = y * width + x;
            if (mask & (1_u32 << bit)) == 0 {
                continue;
            }
            let (tx, ty) = transform_small_box_coord(x, y, width, height, transform);
            transformed |= 1_u32 << (ty * width + tx);
        }
    }
    transformed
}

fn transform_small_box_coord(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    transform: usize,
) -> (usize, usize) {
    let max_x = width - 1;
    let max_y = height - 1;
    match transform {
        0 => (x, y),
        1 => (max_x - x, y),
        2 => (x, max_y - y),
        3 => (max_x - x, max_y - y),
        4 => (y, x),
        5 => (max_y - y, x),
        6 => (y, max_x - x),
        7 => (max_y - y, max_x - x),
        _ => unreachable!(),
    }
}
