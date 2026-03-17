use crate::bitgrid::BitGrid;
use crate::cli::Config;
use crate::generators::{pattern_by_name, pattern_from_file, random_soup};

pub fn initial_grid(config: &Config) -> BitGrid {
    if config.pattern == "random" {
        return random_soup(
            ((config.width as i64) * 2) / 3,
            config.height as i64,
            37,
            config.seed,
        );
    }

    if std::path::Path::new(&config.pattern).exists() {
        return pattern_from_file(&config.pattern)
            .unwrap_or_else(|| panic!("failed to load life grid from {}", config.pattern));
    }

    pattern_by_name(&config.pattern).expect("validated pattern name")
}
