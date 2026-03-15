use crate::RequiredExt;
use crate::bitgrid::BitGrid;
use crate::cli::Config;
use crate::generators::{pattern_by_name, pattern_from_file, random_soup};

pub fn initial_grid(config: &Config) -> BitGrid {
    if config.pattern == "random" {
        let width = i64::try_from(config.width).or_invariant("validated width exceeded Coord");
        let height = i64::try_from(config.height).or_invariant("validated height exceeded Coord");
        return random_soup(width.saturating_mul(2) / 3, height, 37, config.seed);
    }

    if std::path::Path::new(&config.pattern).exists() {
        return pattern_from_file(&config.pattern).unwrap_or_else(|| {
            crate::invariant_failure!("failed to load life grid from {}", config.pattern)
        });
    }

    pattern_by_name(&config.pattern).or_invariant("validated pattern name")
}
