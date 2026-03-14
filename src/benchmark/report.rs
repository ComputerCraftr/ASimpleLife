use super::*;

impl BenchmarkReport {
    pub(super) fn record(
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

    pub(super) fn print(&self) {
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

    pub(crate) fn to_json(&self) -> JsonReport {
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
