use std::env;

use crate::generators::pattern_by_name;

#[derive(Debug, Clone)]
pub struct Config {
    pub pattern: String,
    pub steps: usize,
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

pub fn parse_args() -> Result<Config, CliAction> {
    parse_from(env::args().skip(1))
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
    println!("  --fast-forward <n> jump ahead n generations before rendering");
    println!("  --delay-ms <n>     frame delay in milliseconds");
    println!("  --width <n>        terminal character width");
    println!("  --height <n>       terminal character height");
    println!("  --seed <n>         RNG seed for random soups");
    println!("  --classify         print classification and exit");
}

#[cfg(test)]
mod tests {
    use super::{CliAction, Config, parse_from};

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
        assert_eq!(config.fast_forward, 64);
        assert_eq!(config.width, 100);
        assert_eq!(config.height, 20);
        assert!(config.classify_only);
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
}
