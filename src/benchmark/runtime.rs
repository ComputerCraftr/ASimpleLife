use super::*;
use super::suite::{
    canonical_small_box_mask, estimate_density_percent, reference_is_decisive_runtime,
    seeded_case,
};

pub(super) fn run_exhaustive_5x5(
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

pub(super) fn run_oracle_runtime_case(
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

pub(super) fn run_oracle_representative_case(
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
