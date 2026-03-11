use super::super::cli::Config;
use super::super::initial_grid;
use crate::normalize::normalize;

#[test]
fn initial_grid_uses_named_pattern() {
    let config = Config {
        width: 80,
        height: 24,
        steps: 1,
        delay_ms: 0,
        seed: 123,
        pattern: "glider".to_string(),
        classify_only: false,
    };

    let grid = initial_grid(&config);
    let expected = crate::generators::pattern_by_name("glider").unwrap();
    assert_eq!(normalize(&grid).0, normalize(&expected).0);
}

#[test]
fn initial_grid_random_soup_respects_config_dimensions() {
    let config = Config {
        width: 90,
        height: 30,
        steps: 1,
        delay_ms: 0,
        seed: 7,
        pattern: "random".to_string(),
        classify_only: false,
    };

    let grid = initial_grid(&config);
    let (min_x, min_y, max_x, max_y) = grid.bounds().unwrap();
    assert_eq!((min_x, min_y), (0, 0));
    assert!(max_x < (config.width as i32) * 2 / 3);
    assert!(max_y < config.height as i32);
}
