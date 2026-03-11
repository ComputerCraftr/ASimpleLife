use std::thread;
use std::time::Duration;
use std::{io::Write, io::stdout};

use a_simple_life::app::initial_grid;
use a_simple_life::classify::{ClassificationLimits, classify_seed};
use a_simple_life::cli;
use a_simple_life::engine::advance_grid;
use a_simple_life::life::GameOfLife;
use a_simple_life::memo::Memo;
use a_simple_life::render::TerminalBackbuffer;
use a_simple_life::term::terminal_view_size;

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
    let initial = initial_grid(&config);
    let grid = if config.fast_forward == 0 {
        initial
    } else {
        advance_grid(&initial, config.fast_forward).grid
    };
    let mut memo = Memo::default();
    let mut limits = ClassificationLimits::default();
    if let Some(max_generations) = config.max_generations {
        limits.max_generations = max_generations;
    }
    let classification = classify_seed(&grid, &limits, &mut memo);

    if config.classify_only {
        println!("{classification}");
        return;
    }

    let mut game = GameOfLife::new_with_generation(grid, config.fast_forward);
    let (mut view_width, mut view_height) = terminal_view_size(config.width, config.height);
    let mut backbuffer = TerminalBackbuffer::new(view_width, view_height);
    let mut previous_generation = u64::MAX;
    let mut previous_population = usize::MAX;
    let mut stdout = stdout();
    let mut status_buffer = Vec::with_capacity(256);
    let mut frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
    let mut changed_chunks = None;

    print!("\x1b[2J\x1b[?25l");

    for _ in 0..config.steps {
        let (next_width, next_height) = terminal_view_size(config.width, config.height);
        if next_width != view_width || next_height != view_height {
            view_width = next_width;
            view_height = next_height;
            backbuffer.resize(view_width, view_height);
            frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
            stdout.write_all(b"\x1b[2J").unwrap();
            changed_chunks = None;
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
            .render_chunk_into(game.grid(), changed_chunks.as_deref(), &mut frame_buffer)
            .unwrap();
        write!(&mut frame_buffer, "\x1b[{};1H", view_height + 2).unwrap();
        stdout.write_all(&frame_buffer).unwrap();
        stdout.flush().unwrap();

        thread::sleep(Duration::from_millis(config.delay_ms));
        changed_chunks = Some(game.step_with_chunk_changes());
    }

    println!("\x1b[?25h");
}

fn write_status_line(
    out: &mut Vec<u8>,
    generation: u64,
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
