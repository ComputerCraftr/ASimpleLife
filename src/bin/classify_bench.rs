use std::process::ExitCode;

use a_simple_life::benchmark::{BenchmarkOptions, run_benchmark_report};
use a_simple_life::cli;

fn main() -> ExitCode {
    let parsed = match cli::parse_benchmark_args() {
        Ok(config) => config,
        Err(cli::CliAction::Help) => {
            cli::print_benchmark_help();
            return ExitCode::SUCCESS;
        }
        Err(cli::CliAction::Error(message)) => {
            eprintln!("{message}");
            eprintln!();
            cli::print_benchmark_help();
            return ExitCode::from(2);
        }
    };

    let options = BenchmarkOptions {
        families: (!parsed.families.is_empty()).then_some(parsed.families),
        prediction_max_generations: parsed.predictor_max_generations,
        oracle_max_generations: parsed.oracle_max_generations,
        exhaustive_5x5: parsed.exhaustive_5x5,
        oracle_runtime_case: parsed.oracle_runtime_case,
        oracle_runtime_target_generation: parsed.oracle_runtime_target_generation,
        progress: parsed.progress,
    };
    run_benchmark_report(parsed.format, &options);
    ExitCode::SUCCESS
}
