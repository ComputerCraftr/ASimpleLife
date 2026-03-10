mod bitgrid;
mod classify;
mod cli;
mod generators;
mod life;
mod memo;
mod normalize;
mod render;
mod term;
#[cfg(test)]
mod tests;

use std::thread;
use std::time::Duration;
use std::{io::Write, io::stdout};

use bitgrid::BitGrid;
use classify::{ClassificationLimits, classify_seed};
use cli::Config;
use generators::{pattern_by_name, random_soup};
use life::GameOfLife;
use memo::Memo;
use render::TerminalBackbuffer;
use term::terminal_view_size;

fn initial_grid(config: &Config) -> BitGrid {
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

fn main() {
    let config = match cli::parse_args() {
        Ok(config) => config,
        Err(cli::CliAction::Help) => {
            cli::print_help();
            return;
        }
        Err(cli::CliAction::Error(message)) => {
            eprintln!("{message}");
            eprintln!();
            cli::print_help();
            std::process::exit(2);
        }
    };
    let grid = initial_grid(&config);
    let mut memo = Memo::default();
    let limits = ClassificationLimits::default();
    let classification = classify_seed(&grid, &limits, &mut memo);

    if config.classify_only {
        println!("{classification}");
        return;
    }

    let mut game = GameOfLife::new(grid);
    let (mut view_width, mut view_height) = terminal_view_size(config.width, config.height);
    let mut backbuffer = TerminalBackbuffer::new(view_width, view_height);
    let mut previous_generation = usize::MAX;
    let mut previous_population = usize::MAX;
    let mut stdout = stdout();
    let mut status_buffer = Vec::with_capacity(256);
    let mut frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
    let mut changed_cells: Option<Vec<(i32, i32)>> = None;

    print!("\x1b[2J\x1b[?25l");

    for _ in 0..config.steps {
        let (next_width, next_height) = terminal_view_size(config.width, config.height);
        if next_width != view_width || next_height != view_height {
            view_width = next_width;
            view_height = next_height;
            backbuffer.resize(view_width, view_height);
            frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
            stdout.write_all(b"\x1b[2J").unwrap();
            changed_cells = None;
        }

        let generation = game.generation();
        let population = game.grid().population();
        if generation != previous_generation || population != previous_population {
            status_buffer.clear();
            write_status_line(&mut status_buffer, generation, population, &classification);
            stdout.write_all(&status_buffer).unwrap();
            previous_generation = generation;
            previous_population = population;
        }

        frame_buffer.clear();
        backbuffer
            .render_into(game.grid(), changed_cells.as_deref(), &mut frame_buffer)
            .unwrap();
        write!(&mut frame_buffer, "\x1b[{};1H", view_height + 2).unwrap();
        stdout.write_all(&frame_buffer).unwrap();
        stdout.flush().unwrap();

        thread::sleep(Duration::from_millis(config.delay_ms));
        changed_cells = Some(game.step_with_changes());
    }

    println!("\x1b[?25h");
}

fn write_status_line(
    out: &mut Vec<u8>,
    generation: usize,
    population: usize,
    classification: &impl std::fmt::Display,
) {
    write!(
        out,
        "\x1b[1;1Hgeneration={} population={} classification={}\x1b[K",
        generation, population, classification
    )
    .unwrap();
}
