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
use a_simple_life::term::terminal_size;

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
    let (mut terminal_width, mut terminal_height) = terminal_size(config.width, config.height);
    let initial_status = status_text(
        game.generation(),
        game.grid().population(),
        &classification,
    );
    let mut status_lines = wrapped_line_count(&initial_status, terminal_width);
    let mut view_width = terminal_width;
    let mut view_height = compute_view_height(terminal_height, status_lines);
    let mut backbuffer = TerminalBackbuffer::new(view_width, view_height);
    backbuffer.set_row_offset(status_lines);
    let mut previous_status = String::new();
    let mut stdout = stdout();
    let mut status_buffer = Vec::with_capacity(256);
    let mut frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
    let mut changed_chunks = None;

    print!("\x1b[2J\x1b[?25l");

    for _ in 0..config.steps {
        let current_status = status_text(
            game.generation(),
            game.grid().population(),
            &classification,
        );
        let (next_terminal_width, next_terminal_height) = terminal_size(config.width, config.height);
        let next_view_width = next_terminal_width;
        let next_status_lines = wrapped_line_count(&current_status, next_terminal_width);
        let next_view_height = compute_view_height(next_terminal_height, next_status_lines);
        if next_terminal_width != terminal_width
            || next_terminal_height != terminal_height
            || next_view_width != view_width
            || next_view_height != view_height
            || next_status_lines != status_lines
        {
            terminal_width = next_terminal_width;
            terminal_height = next_terminal_height;
            view_width = next_view_width;
            view_height = next_view_height;
            status_lines = next_status_lines;
            backbuffer.resize(view_width, view_height);
            backbuffer.set_row_offset(status_lines);
            frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
            stdout.write_all(b"\x1b[2J").unwrap();
            changed_chunks = None;
        }

        if current_status != previous_status {
            status_buffer.clear();
            write_status_lines(&mut status_buffer, terminal_width, &current_status, status_lines);
            stdout.write_all(&status_buffer).unwrap();
            previous_status = current_status;
        }

        frame_buffer.clear();
        backbuffer
            .render_chunk_into(game.grid(), changed_chunks.as_deref(), &mut frame_buffer)
            .unwrap();
        write!(&mut frame_buffer, "\x1b[{};1H", view_height + status_lines + 1).unwrap();
        stdout.write_all(&frame_buffer).unwrap();
        stdout.flush().unwrap();

        thread::sleep(Duration::from_millis(config.delay_ms));
        changed_chunks = Some(game.step_with_chunk_changes());
    }

    println!("\x1b[?25h");
}

fn status_text(
    generation: u64,
    population: usize,
    classification: &impl std::fmt::Display,
) -> String {
    format!(
        "generation={} population={} classification={}",
        generation, population, classification
    )
}

fn wrapped_line_count(status: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    status
        .lines()
        .map(|line| line.chars().count().max(1).div_ceil(width))
        .sum::<usize>()
        .max(1)
}

fn compute_view_height(terminal_height: usize, status_lines: usize) -> usize {
    terminal_height.saturating_sub(status_lines).max(1)
}

fn write_status_lines(
    out: &mut Vec<u8>,
    width: usize,
    status: &str,
    status_lines: usize,
) {
    let width = width.max(1);
    for row in 1..=status_lines {
        write!(out, "\x1b[{};1H\x1b[K", row).unwrap();
    }

    let mut row = 1usize;
    let mut column_count = 0usize;
    write!(out, "\x1b[1;1H").unwrap();
    for ch in status.chars() {
        if ch == '\n' || column_count == width {
            row += 1;
            if row > status_lines {
                break;
            }
            write!(out, "\x1b[{};1H", row).unwrap();
            column_count = 0;
            if ch == '\n' {
                continue;
            }
        }
        let mut encoded = [0_u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
        column_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_view_height, wrapped_line_count};

    #[test]
    fn wrapped_status_reduces_view_height() {
        let status = "generation=123 population=456 classification=likely_infinite(oracle_generation_limit, gen=1000000)";
        let status_lines = wrapped_line_count(status, 20);
        assert!(status_lines > 1);
        assert_eq!(compute_view_height(25, status_lines), 25 - status_lines);
    }

    #[test]
    fn wrapped_line_count_respects_multiple_rows() {
        assert_eq!(wrapped_line_count("abcd", 10), 1);
        assert_eq!(wrapped_line_count("abcdefghij", 5), 2);
        assert_eq!(wrapped_line_count("abc\ndefghij", 5), 3);
    }
}
