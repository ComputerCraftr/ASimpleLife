use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::bitgrid::BitGrid;
use crate::classify::{Classification, ClassificationLimits, classify_seed};
use crate::generators::{mix_seed, pattern_by_name, random_soup};
use crate::memo::Memo;
use crate::normalize::normalize;

type SeenStates = HashMap<Vec<(i32, i32)>, (usize, (i32, i32))>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkFormat {
    Text,
    Json,
}

#[derive(Clone, Debug, Default)]
pub struct BenchmarkOptions {
    pub families: Option<Vec<String>>,
    pub exhaustive_5x5: bool,
}

pub fn run_benchmark_report(format: BenchmarkFormat, options: &BenchmarkOptions) {
    let limits = ClassificationLimits {
        max_generations: 256,
        max_population: 20_000,
        max_bounding_box: 512,
    };
    let filters = benchmark_family_filter(options);
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

    if options.exhaustive_5x5 {
        run_exhaustive_5x5(&mut report);
    }

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

pub(crate) fn bitmask_pattern(mask: u32, width: i32, height: i32) -> BitGrid {
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

pub(crate) fn reference_classify(seed: &BitGrid, limits: &ClassificationLimits) -> Classification {
    let mut seen: SeenStates = HashMap::new();
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
    ExhaustiveSmallBox,
    ExhaustiveFiveByFive,
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

fn parse_family(token: &str) -> Option<BenchmarkFamily> {
    match token {
        "iid" | "IidRandom" => Some(BenchmarkFamily::IidRandom),
        "structured" | "StructuredRandom" => Some(BenchmarkFamily::StructuredRandom),
        "smallbox" | "ExhaustiveSmallBox" => Some(BenchmarkFamily::ExhaustiveSmallBox),
        "5x5" | "ExhaustiveFiveByFive" => Some(BenchmarkFamily::ExhaustiveFiveByFive),
        "methuselah" | "LongLivedMethuselah" => Some(BenchmarkFamily::LongLivedMethuselah),
        "mover" | "TranslatedPeriodicMover" => Some(BenchmarkFamily::TranslatedPeriodicMover),
        "gun" | "GunPufferBreeder" => Some(BenchmarkFamily::GunPufferBreeder),
        "delayed" | "DelayedInteraction" => Some(BenchmarkFamily::DelayedInteraction),
        "ash" | "DeceptiveAsh" => Some(BenchmarkFamily::DeceptiveAsh),
        "gadget" | "ComputationalGadget" => Some(BenchmarkFamily::ComputationalGadget),
        _ => None,
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

fn run_exhaustive_5x5(report: &mut BenchmarkReport) {
    let limits = ClassificationLimits {
        max_generations: 128,
        max_population: 10_000,
        max_bounding_box: 256,
    };

    for mask in 0_u32..(1_u32 << 25) {
        if canonical_small_box_mask(mask, 5, 5) != mask {
            continue;
        }
        let grid = bitmask_pattern(mask, 5, 5);
        let expected = reference_classify(&grid, &limits);
        if !reference_is_decisive_runtime(&expected) {
            continue;
        }
        let started = Instant::now();
        let actual = classify_seed(&grid, &limits, &mut Memo::default());
        let case = BenchmarkCase {
            name: format!("5x5_{mask}"),
            family: BenchmarkFamily::ExhaustiveFiveByFive,
            size: 5,
            density_percent: estimate_density_percent(&grid),
            grid,
        };
        report.record(&case, &expected, &actual, started.elapsed());
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

fn reference_is_decisive_runtime(classification: &Classification) -> bool {
    !matches!(classification, Classification::Unknown { .. })
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
