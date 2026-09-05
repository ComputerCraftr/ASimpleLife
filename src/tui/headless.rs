use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use crate::bitgrid::Coord;
use crate::classify::{ClassificationLimits, analysis::classify_capture};
use crate::cli::Config;
use crate::hashlife::session::capture::CaptureLimits;
use crate::render::ViewportController;
use std::sync::atomic::AtomicBool;

use super::source::prepare_config_source;

pub fn run_headless_source(config: &Config) -> Result<(), String> {
    let mut prepared = prepare_config_source(config)?;
    if config.classify_only {
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        let capture = prepared
            .session
            .capture_analysis(None, CaptureLimits::default(), &cancelled)
            .map_err(|error| format!("classification capture unavailable: {error:?}"))?;
        let limits = ClassificationLimits {
            max_generations: config.max_generations.unwrap_or(512),
        };
        println!(
            "{:?}",
            classify_capture(capture, &limits, 128 * 1024 * 1024, &cancelled)
                .map_err(|error| format!("classification unavailable: {error:?}"))?
        );
        return Ok(());
    }

    let mut viewport =
        ViewportController::new(config.width, config.height).map_err(|error| error.to_string())?;
    let mut stdout = io::stdout().lock();
    for frame_index in 0..config.steps {
        if frame_index != 0 {
            prepared
                .session
                .advance_hashlife_root(config.step_generations)
                .map_err(|error| format!("HashLife advance failed: {error:?}"))?;
        }
        let sample = viewport
            .sample(&mut prepared.session)
            .map_err(|error| error.to_string())?;
        writeln!(
            stdout,
            "generation={} population={} backend=hashlife source={}",
            prepared.session.hashlife_generation(),
            prepared
                .session
                .hashlife_population_count()
                .map_or(0, |population| population.lower_bound()),
            prepared.label
        )
        .map_err(|error| error.to_string())?;
        write_grid(
            &mut stdout,
            &sample.grid,
            sample.origin,
            config.width,
            config.height,
        )?;
        stdout.flush().map_err(|error| error.to_string())?;
        if config.delay_ms != 0 && frame_index + 1 < config.steps {
            thread::sleep(Duration::from_millis(config.delay_ms));
        }
    }
    Ok(())
}

fn write_grid(
    writer: &mut impl Write,
    grid: &crate::bitgrid::BitGrid,
    origin: (Coord, Coord),
    width: usize,
    height: usize,
) -> Result<(), String> {
    for row in 0..height {
        let row = Coord::try_from(row).map_err(|_| "viewport row exceeds coordinate range")?;
        let y = origin
            .1
            .checked_add(row.checked_mul(2).ok_or("viewport row overflow")?)
            .ok_or("viewport row overflow")?;
        let lower_y = y.checked_add(1).ok_or("viewport row overflow")?;
        for column in 0..width {
            let column =
                Coord::try_from(column).map_err(|_| "viewport column exceeds coordinate range")?;
            let x = origin
                .0
                .checked_add(column)
                .ok_or("viewport column overflow")?;
            let glyph = match (grid.get(x, y), grid.get(x, lower_y)) {
                (false, false) => ' ',
                (true, false) => '▀',
                (false, true) => '▄',
                (true, true) => '█',
            };
            write!(writer, "{glyph}").map_err(|error| error.to_string())?;
        }
        writeln!(writer).map_err(|error| error.to_string())?;
    }
    Ok(())
}
