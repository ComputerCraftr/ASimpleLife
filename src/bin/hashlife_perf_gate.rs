use std::collections::BTreeSet;
use std::error::Error;
use std::time::{Duration, Instant};

use a_simple_life::generators::pattern_by_name;
use a_simple_life::hashlife::{HashLifeLimits, HashLifeSession, PopulationCount};
use a_simple_life::life::GameOfLife;
use serde::Deserialize;

const CORPUS: &str = include_str!("../../benchmarks/hashlife_corpus_v1.json");
const REFERENCE_LIMIT: Duration = Duration::from_secs(5);
const REFERENCE_RSS_LIMIT: usize = 1024 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct Corpus {
    version: u32,
    targets: Vec<u64>,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
struct CorpusCase {
    pattern: String,
    period: Option<u64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let corpus: Corpus = serde_json::from_str(CORPUS)?;
    if corpus.version != 1 {
        return Err(std::io::Error::other("unsupported HashLife corpus version").into());
    }

    let runner = runner_model();
    let pattern_filter = std::env::var("HASHLIFE_PERF_PATTERN").ok();
    let target_filter = std::env::var("HASHLIFE_PERF_TARGET")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    if corpus
        .targets
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != corpus.targets.len()
    {
        return Err(std::io::Error::other("HashLife corpus targets must be unique").into());
    }
    let reference_runner =
        cfg!(all(target_os = "macos", target_arch = "aarch64")) && runner.contains("Apple M2");
    let started = Instant::now();
    let mut executed_cases = 0_usize;
    let mut peak_allocated = 0_u128;
    let mut peak_rss = current_rss_bytes();
    let mut native_kernel_lanes = 0_usize;
    let mut d4_candidate_lanes = 0_usize;
    let mut native_d4_candidate_lanes = 0_usize;
    let mut native_d4_prefix_lanes = 0_usize;
    let mut native_d4_exact_winners = 0_usize;
    let mut native_control_groups = 0_usize;
    for case in &corpus.cases {
        if pattern_filter
            .as_deref()
            .is_some_and(|pattern| pattern != case.pattern)
        {
            continue;
        }
        for (target_index, &manifest_target) in corpus.targets.iter().enumerate() {
            if target_filter.is_some() && target_index != 0 {
                continue;
            }
            let target = target_filter.unwrap_or(manifest_target);
            let grid = pattern_by_name(&case.pattern).ok_or_else(|| {
                std::io::Error::other(format!("unknown corpus pattern {}", case.pattern))
            })?;
            let expected_population = residual_population(&grid, target, case.period);
            let mut session = HashLifeSession::with_limits(HashLifeLimits::default());
            session.try_load_grid(&grid).map_err(|error| {
                std::io::Error::other(format!(
                    "failed to load corpus pattern {}: {error:?}",
                    case.pattern
                ))
            })?;
            advance_target(&mut session, &case.pattern, target)?;
            let population = session.population_count().ok_or_else(|| {
                std::io::Error::other(format!("corpus pattern {} was not loaded", case.pattern))
            })?;
            validate_case(
                case,
                target,
                session.generation(),
                population,
                expected_population,
            )?;
            if case.period.is_none() {
                validate_nonperiodic_decomposition(case, target, &grid, &mut session)?;
            }
            let stats = session.execution_stats();
            if stats.materializations != 0 || stats.dependency_stalls != 0 {
                return Err(std::io::Error::other(format!(
                    "hot-path invariant failed pattern={} target={target} stats={stats:?}",
                    case.pattern
                ))
                .into());
            }
            if stats.gc_runs > 2 {
                return Err(std::io::Error::other(format!(
                    "corpus collection budget exceeded pattern={} target={target} gc_runs={}",
                    case.pattern, stats.gc_runs
                ))
                .into());
            }
            peak_allocated = peak_allocated.max(stats.allocated_bytes);
            peak_rss = peak_rss.max(current_rss_bytes());
            native_kernel_lanes = native_kernel_lanes
                .saturating_add(stats.native_avx2_lanes)
                .saturating_add(stats.native_neon_lanes);
            d4_candidate_lanes = d4_candidate_lanes.saturating_add(stats.d4_candidate_lanes);
            native_d4_candidate_lanes =
                native_d4_candidate_lanes.saturating_add(stats.native_d4_candidate_lanes);
            native_d4_prefix_lanes =
                native_d4_prefix_lanes.saturating_add(stats.native_d4_prefix_compare_lanes);
            native_d4_exact_winners =
                native_d4_exact_winners.saturating_add(stats.native_d4_exact_winner_lanes);
            native_control_groups = native_control_groups
                .saturating_add(stats.native_avx2_control_groups)
                .saturating_add(stats.native_neon_control_groups);
            executed_cases += 1;
            println!(
                "pattern={} target={} population={population:?} allocated_bytes={} nodes={} gc_runs={}",
                case.pattern, target, stats.allocated_bytes, stats.nodes, stats.gc_runs
            );
        }
    }

    if executed_cases == 0 {
        return Err(std::io::Error::other(format!(
            "HashLife performance filters matched no cases: pattern={pattern_filter:?} target={target_filter:?}"
        ))
        .into());
    }
    if native_kernels_supported() && native_kernel_lanes == 0 {
        return Err(std::io::Error::other(
            "HashLife corpus used no native structural kernel lanes on supported hardware",
        )
        .into());
    }
    if native_kernels_supported()
        && (native_d4_candidate_lanes == 0
            || native_d4_prefix_lanes == 0
            || native_d4_exact_winners == 0)
    {
        return Err(std::io::Error::other(format!(
            "HashLife corpus did not execute complete native D4 support: candidate_lanes={native_d4_candidate_lanes} prefix_lanes={native_d4_prefix_lanes} exact_winners={native_d4_exact_winners}"
        ))
        .into());
    }

    let elapsed = started.elapsed();
    println!(
        "corpus_version={} executed_cases={executed_cases} runner={runner:?} elapsed_ms={} peak_rss_bytes={} peak_allocated_bytes={peak_allocated} native_kernel_lanes={native_kernel_lanes} native_control_groups={native_control_groups} d4_candidate_lanes={d4_candidate_lanes} native_d4_candidate_lanes={native_d4_candidate_lanes} native_d4_prefix_lanes={native_d4_prefix_lanes} native_d4_exact_winners={native_d4_exact_winners}",
        corpus.version,
        elapsed.as_millis(),
        peak_rss
    );
    if reference_runner && (elapsed > REFERENCE_LIMIT || peak_rss > REFERENCE_RSS_LIMIT) {
        return Err(std::io::Error::other(format!(
            "reference performance gate failed elapsed={elapsed:?} peak_rss={peak_rss}"
        ))
        .into());
    }
    Ok(())
}

fn validate_nonperiodic_decomposition(
    case: &CorpusCase,
    target: u64,
    grid: &a_simple_life::bitgrid::BitGrid,
    primary: &mut HashLifeSession,
) -> Result<(), Box<dyn Error>> {
    let first = 1_u64 << target.trailing_zeros();
    let second = target - first;
    let mut alternate = HashLifeSession::with_limits(HashLifeLimits::default());
    alternate.try_load_grid(grid).map_err(|error| {
        std::io::Error::other(format!(
            "failed to load alternate decomposition pattern={}: {error:?}",
            case.pattern
        ))
    })?;
    alternate.advance_root(first).map_err(|error| {
        std::io::Error::other(format!(
            "first alternate segment failed pattern={} step={first}: {error:?}",
            case.pattern
        ))
    })?;
    alternate.advance_root(second).map_err(|error| {
        std::io::Error::other(format!(
            "second alternate segment failed pattern={} step={second}: {error:?}",
            case.pattern
        ))
    })?;

    let primary_snapshot = primary
        .export_snapshot_string()?
        .ok_or_else(|| std::io::Error::other("primary corpus session was unexpectedly empty"))?;
    let alternate_snapshot = alternate
        .export_snapshot_string()?
        .ok_or_else(|| std::io::Error::other("alternate corpus session was unexpectedly empty"))?;
    if primary_snapshot != alternate_snapshot {
        return Err(std::io::Error::other(format!(
            "decomposition mismatch pattern={} target={target} split={first}+{second}",
            case.pattern
        ))
        .into());
    }
    Ok(())
}

fn native_kernels_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

fn advance_target(
    session: &mut HashLifeSession,
    pattern: &str,
    target: u64,
) -> Result<(), Box<dyn Error>> {
    if std::env::var_os("HASHLIFE_PERF_TRACE_SEGMENTS").is_none() {
        session.advance_root(target).map_err(|error| {
            std::io::Error::other(format!(
                "HashLife corpus case failed pattern={pattern} target={target}: {error:?}"
            ))
        })?;
        return Ok(());
    }

    let mut remaining = target;
    while remaining != 0 {
        let step = 1_u64 << remaining.trailing_zeros();
        let started = Instant::now();
        session.advance_root(step).map_err(|error| {
            std::io::Error::other(format!(
                "HashLife traced segment failed pattern={pattern} step={step}: {error:?}"
            ))
        })?;
        eprintln!(
            "segment pattern={pattern} step={step} elapsed_ms={} stats={:?}",
            started.elapsed().as_millis(),
            session.execution_stats()
        );
        remaining -= step;
    }
    Ok(())
}

fn validate_case(
    case: &CorpusCase,
    target: u64,
    generation: u64,
    population: PopulationCount,
    expected_population: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    if generation != target {
        return Err(std::io::Error::other(format!(
            "generation accounting mismatch pattern={} expected={target} actual={generation}",
            case.pattern
        ))
        .into());
    }
    if let Some(expected) = expected_population
        && population != PopulationCount::Exact(u128::from(expected))
    {
        return Err(std::io::Error::other(format!(
            "population mismatch pattern={} target={target} expected={expected} actual={population:?}",
            case.pattern
        ))
        .into());
    }
    if population.is_zero() {
        return Err(std::io::Error::other(format!(
            "corpus pattern unexpectedly extinct pattern={} target={target}",
            case.pattern
        ))
        .into());
    }
    Ok(())
}

fn residual_population(
    grid: &a_simple_life::bitgrid::BitGrid,
    target: u64,
    period: Option<u64>,
) -> Option<u64> {
    let period = period?;
    let mut game = GameOfLife::new(grid.clone());
    for _ in 0..target % period {
        game.step_with_chunk_changes();
    }
    u64::try_from(game.grid().population()).ok()
}

fn runner_model() -> String {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output();
        output
            .ok()
            .filter(|result| result.status.success())
            .map(|result| String::from_utf8_lossy(&result.stdout).trim().to_owned())
            .unwrap_or_else(|| "unknown-macos-runner".to_owned())
    }
    #[cfg(not(target_os = "macos"))]
    {
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
    }
}

fn current_rss_bytes() -> usize {
    memory_stats::memory_stats().map_or(0, |stats| stats.physical_mem)
}
