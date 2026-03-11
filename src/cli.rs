use std::env;

use crate::benchmark::BenchmarkFormat;
use crate::generators::pattern_by_name;

#[derive(Debug, Clone)]
pub struct Config {
    pub pattern: String,
    pub steps: usize,
    pub max_generations: Option<u64>,
    pub fast_forward: u64,
    pub delay_ms: u64,
    pub width: usize,
    pub height: usize,
    pub classify_only: bool,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pattern: "glider".to_string(),
            steps: 200,
            max_generations: None,
            fast_forward: 0,
            delay_ms: 80,
            width: 80,
            height: 24,
            classify_only: false,
            seed: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliAction {
    Help,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkCliConfig {
    pub format: BenchmarkFormat,
    pub families: Vec<String>,
    pub predictor_max_generations: Option<u64>,
    pub oracle_max_generations: Option<u64>,
    pub exhaustive_5x5: bool,
    pub oracle_runtime_case: bool,
    pub oracle_runtime_target_generation: Option<u64>,
    pub progress: bool,
}

pub fn parse_args() -> Result<Config, CliAction> {
    parse_from(env::args().skip(1))
}

pub fn parse_benchmark_args() -> Result<BenchmarkCliConfig, CliAction> {
    parse_benchmark_from(env::args().skip(1))
}

fn parse_from<I>(args: I) -> Result<Config, CliAction>
where
    I: IntoIterator<Item = String>,
{
    let mut config = Config::default();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pattern" => config.pattern = next_string(&mut args, "--pattern")?,
            "--steps" => config.steps = next_parsed(&mut args, "--steps")?,
            "--max-generations" => {
                config.max_generations = Some(next_parsed(&mut args, "--max-generations")?)
            }
            "--fast-forward" => config.fast_forward = next_parsed(&mut args, "--fast-forward")?,
            "--delay-ms" => config.delay_ms = next_parsed(&mut args, "--delay-ms")?,
            "--width" => config.width = next_parsed(&mut args, "--width")?,
            "--height" => config.height = next_parsed(&mut args, "--height")?,
            "--seed" => config.seed = next_parsed(&mut args, "--seed")?,
            "--classify" => config.classify_only = true,
            "--help" | "-h" => return Err(CliAction::Help),
            _ => return Err(CliAction::Error(format!("unknown argument: {arg}"))),
        }
    }

    validate(config)
}

fn parse_benchmark_from<I>(args: I) -> Result<BenchmarkCliConfig, CliAction>
where
    I: IntoIterator<Item = String>,
{
    let mut config = BenchmarkCliConfig {
        format: BenchmarkFormat::Text,
        families: Vec::new(),
        predictor_max_generations: None,
        oracle_max_generations: None,
        exhaustive_5x5: false,
        oracle_runtime_case: false,
        oracle_runtime_target_generation: None,
        progress: false,
    };
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => config.format = BenchmarkFormat::Json,
            "--text" => config.format = BenchmarkFormat::Text,
            "--families" => {
                let value = next_string(&mut args, "--families")?;
                config.families.extend(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|token| !token.is_empty())
                        .filter(|token| *token != "5x5" && *token != "ExhaustiveFiveByFive")
                        .map(str::to_owned),
                );
            }
            "--predictor-max-generations" => {
                config.predictor_max_generations =
                    Some(next_parsed(&mut args, "--predictor-max-generations")?)
                }
            "--oracle-max-generations" => {
                config.oracle_max_generations =
                    Some(next_parsed(&mut args, "--oracle-max-generations")?)
            }
            "--exhaustive-5x5" => config.exhaustive_5x5 = true,
            "--oracle-max-jump" => config.oracle_runtime_case = true,
            "--oracle-runtime-target-generation" => {
                config.oracle_runtime_target_generation =
                    Some(next_parsed(&mut args, "--oracle-runtime-target-generation")?)
                }
            "--progress" => config.progress = true,
            "--help" | "-h" => return Err(CliAction::Help),
            _ => return Err(CliAction::Error(format!("unknown argument: {arg}"))),
        }
    }

    validate_benchmark(config)
}

fn validate(config: Config) -> Result<Config, CliAction> {
    if config.width == 0 {
        return Err(CliAction::Error(
            "--width must be greater than zero".to_string(),
        ));
    }
    if config.height == 0 {
        return Err(CliAction::Error(
            "--height must be greater than zero".to_string(),
        ));
    }
    if config.steps == 0 {
        return Err(CliAction::Error(
            "--steps must be greater than zero".to_string(),
        ));
    }
    if config.pattern.trim().is_empty() {
        return Err(CliAction::Error("--pattern must not be empty".to_string()));
    }
    if config.pattern != "random" && pattern_by_name(&config.pattern).is_none() {
        return Err(CliAction::Error(format!(
            "unknown pattern: {}",
            config.pattern
        )));
    }

    Ok(config)
}

fn validate_benchmark(config: BenchmarkCliConfig) -> Result<BenchmarkCliConfig, CliAction> {
    if config.exhaustive_5x5 && config.oracle_runtime_case {
        return Err(CliAction::Error(
            "--exhaustive-5x5 and --oracle-max-jump cannot be combined".to_string(),
        ));
    }

    Ok(config)
}

fn next_string<I>(args: &mut I, flag: &str) -> Result<String, CliAction>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| CliAction::Error(format!("missing value for {flag}")))
}

fn next_parsed<T, I>(args: &mut I, flag: &str) -> Result<T, CliAction>
where
    T: std::str::FromStr,
    I: Iterator<Item = String>,
{
    let value = next_string(args, flag)?;
    value
        .parse()
        .map_err(|_| CliAction::Error(format!("invalid value for {flag}: {value}")))
}

pub fn print_help() {
    println!("ASimpleLife");
    println!(
        "  --pattern <name>   glider | blinker | block | diehard | acorn | r_pentomino | gosper_glider_gun | glider_producing_switch_engine | blinker_puffer_1 | random"
    );
    println!("  --steps <n>        number of generations to run");
    println!("  --max-generations <n>  classifier generation horizon");
    println!("  --fast-forward <n> jump ahead n generations before rendering");
    println!("  --delay-ms <n>     frame delay in milliseconds");
    println!("  --width <n>        terminal character width");
    println!("  --height <n>       terminal character height");
    println!("  --seed <n>         RNG seed for random soups");
    println!("  --classify         print classification and exit");
}

pub fn print_benchmark_help() {
    println!(
        "usage: cargo run --release --bin classify_bench -- [--text|--json] [--families iid,structured] [--predictor-max-generations <n>] [--oracle-max-generations <n>] [--exhaustive-5x5] [--oracle-max-jump] [--oracle-runtime-target-generation <n>] [--progress]"
    );
    println!("families: iid,structured,smallbox,methuselah,mover,gun,delayed,ash,gadget");
    println!("modes:");
    println!("  default             run the main accuracy report");
    println!("  --exhaustive-5x5    run only the exhaustive 5x5 sweep");
    println!("  --oracle-max-jump   run only the deep oracle runtime case");
}

#[cfg(test)]
mod tests {
    use super::{
        BenchmarkCliConfig, CliAction, Config, parse_benchmark_from, parse_from,
    };
    use crate::benchmark::BenchmarkFormat;

    #[test]
    fn parses_valid_args() {
        let config = parse_from(vec![
            "--pattern".to_string(),
            "gosper_glider_gun".to_string(),
            "--steps".to_string(),
            "10".to_string(),
            "--fast-forward".to_string(),
            "64".to_string(),
            "--width".to_string(),
            "100".to_string(),
            "--height".to_string(),
            "20".to_string(),
            "--classify".to_string(),
        ])
        .unwrap();

        assert_eq!(config.pattern, "gosper_glider_gun");
        assert_eq!(config.steps, 10);
        assert_eq!(config.max_generations, None);
        assert_eq!(config.fast_forward, 64);
        assert_eq!(config.width, 100);
        assert_eq!(config.height, 20);
        assert!(config.classify_only);
    }

    #[test]
    fn parses_trillion_fast_forward() {
        let config = parse_from(vec![
            "--pattern".to_string(),
            "block".to_string(),
            "--fast-forward".to_string(),
            "1000000000000".to_string(),
        ])
        .unwrap();

        assert_eq!(config.fast_forward, 1_000_000_000_000);
    }

    #[test]
    fn parses_max_generations_for_app() {
        let config = parse_from(vec![
            "--max-generations".to_string(),
            "2048".to_string(),
        ])
        .unwrap();

        assert_eq!(config.max_generations, Some(2_048));
    }

    #[test]
    fn rejects_unknown_argument() {
        let error = parse_from(vec!["--bogus".to_string()]).unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("unknown argument: --bogus".to_string())
        );
    }

    #[test]
    fn rejects_zero_dimensions() {
        let error = parse_from(vec!["--height".to_string(), "0".to_string()]).unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("--height must be greater than zero".to_string())
        );
    }

    #[test]
    fn rejects_unknown_pattern() {
        let error =
            parse_from(vec!["--pattern".to_string(), "not_a_pattern".to_string()]).unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("unknown pattern: not_a_pattern".to_string())
        );
    }

    #[test]
    fn defaults_remain_stable() {
        let config = Config::default();
        assert_eq!(config.width, 80);
        assert_eq!(config.height, 24);
    }

    #[test]
    fn parses_valid_benchmark_args() {
        let config = parse_benchmark_from(vec![
            "--json".to_string(),
            "--families".to_string(),
            "iid,structured".to_string(),
            "--exhaustive-5x5".to_string(),
        ])
        .unwrap();

        assert_eq!(
            config,
            BenchmarkCliConfig {
                format: BenchmarkFormat::Json,
                families: vec!["iid".to_string(), "structured".to_string()],
                predictor_max_generations: None,
                oracle_max_generations: None,
                exhaustive_5x5: true,
                oracle_runtime_case: false,
                oracle_runtime_target_generation: None,
                progress: false,
            }
        );
    }

    #[test]
    fn parses_oracle_max_jump_flag() {
        let config = parse_benchmark_from(vec!["--oracle-max-jump".to_string()]).unwrap();
        assert_eq!(
            config,
            BenchmarkCliConfig {
                format: BenchmarkFormat::Text,
                families: Vec::new(),
                predictor_max_generations: None,
                oracle_max_generations: None,
                exhaustive_5x5: false,
                oracle_runtime_case: true,
                oracle_runtime_target_generation: None,
                progress: false,
            }
        );
    }

    #[test]
    fn parses_oracle_runtime_target_generation() {
        let config = parse_benchmark_from(vec![
            "--oracle-runtime-target-generation".to_string(),
            "123456".to_string(),
        ])
        .unwrap();
        assert_eq!(
            config,
            BenchmarkCliConfig {
                format: BenchmarkFormat::Text,
                families: Vec::new(),
                predictor_max_generations: None,
                oracle_max_generations: None,
                exhaustive_5x5: false,
                oracle_runtime_case: false,
                oracle_runtime_target_generation: Some(123_456),
                progress: false,
            }
        );
    }

    #[test]
    fn parses_progress_flag() {
        let config = parse_benchmark_from(vec!["--progress".to_string()]).unwrap();
        assert_eq!(
            config,
            BenchmarkCliConfig {
                format: BenchmarkFormat::Text,
                families: Vec::new(),
                predictor_max_generations: None,
                oracle_max_generations: None,
                exhaustive_5x5: false,
                oracle_runtime_case: false,
                oracle_runtime_target_generation: None,
                progress: true,
            }
        );
    }

    #[test]
    fn parses_benchmark_generation_limits() {
        let config = parse_benchmark_from(vec![
            "--predictor-max-generations".to_string(),
            "1024".to_string(),
            "--oracle-max-generations".to_string(),
            "65536".to_string(),
        ])
        .unwrap();
        assert_eq!(
            config,
            BenchmarkCliConfig {
                format: BenchmarkFormat::Text,
                families: Vec::new(),
                predictor_max_generations: Some(1_024),
                oracle_max_generations: Some(65_536),
                exhaustive_5x5: false,
                oracle_runtime_case: false,
                oracle_runtime_target_generation: None,
                progress: false,
            }
        );
    }

    #[test]
    fn benchmark_rejects_legacy_predictor_generation_alias() {
        let error = parse_benchmark_from(vec![
            "--max-generations".to_string(),
            "2048".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("unknown argument: --max-generations".to_string())
        );
    }

    #[test]
    fn benchmark_rejects_legacy_oracle_runtime_target_alias() {
        let error = parse_benchmark_from(vec![
            "--oracle-max-generation".to_string(),
            "8192".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("unknown argument: --oracle-max-generation".to_string())
        );
    }

    #[test]
    fn benchmark_help_is_reported() {
        let error = parse_benchmark_from(vec!["--help".to_string()]).unwrap_err();
        assert_eq!(error, CliAction::Help);
    }

    #[test]
    fn benchmark_rejects_unknown_argument() {
        let error = parse_benchmark_from(vec!["--bogus".to_string()]).unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("unknown argument: --bogus".to_string())
        );
    }

    #[test]
    fn benchmark_rejects_missing_families_value() {
        let error = parse_benchmark_from(vec!["--families".to_string()]).unwrap_err();
        assert_eq!(
            error,
            CliAction::Error("missing value for --families".to_string())
        );
    }

    #[test]
    fn benchmark_rejects_combining_special_modes() {
        let error = parse_benchmark_from(vec![
            "--exhaustive-5x5".to_string(),
            "--oracle-max-jump".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            error,
            CliAction::Error(
                "--exhaustive-5x5 and --oracle-max-jump cannot be combined".to_string()
            )
        );
    }
}
