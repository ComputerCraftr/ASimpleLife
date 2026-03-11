use crate::bitgrid::BitGrid;
use crate::cli::Config;
use crate::generators::{pattern_by_name, random_soup};

pub fn initial_grid(config: &Config) -> BitGrid {
    if config.pattern == "random" {
        return random_soup(
            (config.width as i32) * 2 / 3,
            config.height as i32,
            37,
            config.seed,
        );
    }

    pattern_by_name(&config.pattern).expect("validated pattern name")
}
