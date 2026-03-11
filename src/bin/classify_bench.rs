use std::process::ExitCode;

use a_simple_life::benchmark::{BenchmarkFormat, BenchmarkOptions, run_benchmark_report};

fn main() -> ExitCode {
    let mut format = BenchmarkFormat::Text;
    let mut families = Vec::new();
    let mut exhaustive_5x5 = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => format = BenchmarkFormat::Json,
            "--text" => format = BenchmarkFormat::Text,
            "--families" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --families");
                    eprintln!(
                        "usage: cargo run --release --bin classify_bench -- [--text|--json] [--families iid,structured]"
                    );
                    return ExitCode::from(2);
                };
                families.extend(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|token| !token.is_empty())
                        .filter(|token| *token != "5x5" && *token != "ExhaustiveFiveByFive")
                        .map(str::to_owned),
                );
            }
            "--exhaustive-5x5" => exhaustive_5x5 = true,
            "--help" | "-h" => {
                eprintln!(
                    "usage: cargo run --release --bin classify_bench -- [--text|--json] [--families iid,structured] [--exhaustive-5x5]"
                );
                eprintln!(
                    "families: iid,structured,smallbox,methuselah,mover,gun,delayed,ash,gadget"
                );
                return ExitCode::SUCCESS;
            }
            _ => {
                eprintln!("unknown argument: {arg}");
                eprintln!(
                    "usage: cargo run --release --bin classify_bench -- [--text|--json] [--families iid,structured] [--exhaustive-5x5]"
                );
                return ExitCode::from(2);
            }
        }
    }

    let options = BenchmarkOptions {
        families: (!families.is_empty()).then_some(families),
        exhaustive_5x5,
    };
    run_benchmark_report(format, &options);
    ExitCode::SUCCESS
}
