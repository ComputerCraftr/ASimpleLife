use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use std::{io::Write, io::stdout};

use a_simple_life::app::initial_grid;
use a_simple_life::bitgrid::{BitGrid, Coord};
use a_simple_life::classify::{ClassificationLimits, classify_seed};
use a_simple_life::cli;
use a_simple_life::engine::{SimulationSession, should_use_exact_simd_repeat_skip};
use a_simple_life::hashlife::{
    GridExtractionPolicy, HASHLIFE_FULL_GRID_MAX_CHUNKS, HASHLIFE_FULL_GRID_MAX_POPULATION,
};
use a_simple_life::life::GameOfLife;
use a_simple_life::memo::Memo;
use a_simple_life::oracle::OracleSession;
use a_simple_life::render::{TerminalBackbuffer, compute_origin_for_bounds};
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
    let start_generation = config.target_generation.unwrap_or(0);
    let mut startup_simulation = (start_generation > 0).then(SimulationSession::new);
    let mut startup_population = None;
    let classification = if start_generation == 0 {
        let mut memo = Memo::default();
        let mut limits = ClassificationLimits::default();
        if let Some(max_generations) = config.max_generations {
            limits.max_generations = max_generations;
        }
        classify_seed(&initial, &limits, &mut memo).to_string()
    } else {
        let simulation = startup_simulation
            .as_mut()
            .expect("startup target generation should create a simulation session");
        let outcome = OracleSession::new(initial.clone(), 0, HashMap::new(), simulation)
            .advance_runtime_target(start_generation, None);
        startup_population = Some(outcome.population);
        outcome.classification.to_string()
    };
    if config.classify_only {
        println!("{classification}");
        return;
    }

    let grid = (start_generation == 0).then_some(initial);
    let mut game = if start_generation == 0 && config.step_generations == 1 {
        Some(GameOfLife::new_with_generation(
            grid.clone()
                .expect("initial grid should exist for exact stepping"),
            start_generation,
        ))
    } else {
        None
    };
    let mut simulation = if start_generation > 0 {
        startup_simulation
    } else if config.step_generations > 1 {
        Some(SimulationSession::new())
    } else {
        None
    };
    let mut sampled_generation = start_generation;
    let (mut terminal_width, mut terminal_height) = terminal_size(config.width, config.height);
    let initial_status = status_text(
        start_generation,
        startup_population
            .or_else(|| grid.as_ref().map(BitGrid::population))
            .unwrap_or(0),
        &classification,
    );
    let mut sampled_grid = (start_generation == 0 && config.step_generations > 1).then(|| {
        grid.clone()
            .expect("initial grid should exist for multi-generation stepping")
    });
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
        let session_render_grid;
        let mut session_origin = None;
        let (current_generation, current_population, current_grid) =
            if let Some(game) = game.as_ref() {
                (game.generation(), game.grid().population(), game.grid())
            } else if let Some(grid) = sampled_grid.as_ref() {
                (sampled_generation, grid.population(), grid)
            } else {
                let simulation = simulation
                    .as_mut()
                    .expect("session-based startup should keep a simulation session");
                let (origin, render_grid) =
                    sample_visible_session_grid(simulation, view_width, view_height);
                session_origin = Some(origin);
                session_render_grid = Some(render_grid);
                let grid = session_render_grid
                    .as_ref()
                    .expect("session render grid should exist");
                (
                    simulation.hashlife_generation(),
                    usize::try_from(simulation.hashlife_population().unwrap_or(0))
                        .expect("hashlife population exceeded usize"),
                    grid,
                )
            };
        let current_status = status_text(current_generation, current_population, &classification);
        let (next_terminal_width, next_terminal_height) =
            terminal_size(config.width, config.height);
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
            write_status_lines(
                &mut status_buffer,
                terminal_width,
                &current_status,
                status_lines,
            );
            stdout.write_all(&status_buffer).unwrap();
            previous_status = current_status;
        }

        frame_buffer.clear();
        if let Some(origin) = session_origin {
            backbuffer
                .render_at_origin_into(current_grid, origin, &mut frame_buffer)
                .unwrap();
        } else {
            backbuffer
                .render_chunk_into(current_grid, changed_chunks.as_deref(), &mut frame_buffer)
                .unwrap();
        }
        write!(
            &mut frame_buffer,
            "\x1b[{};1H",
            view_height + status_lines + 1
        )
        .unwrap();
        stdout.write_all(&frame_buffer).unwrap();
        stdout.flush().unwrap();

        thread::sleep(Duration::from_millis(config.delay_ms));
        if let Some(game) = game.as_mut() {
            changed_chunks = Some(game.step_with_chunk_changes());
        } else if sampled_grid.is_some() {
            let simulation = simulation
                .as_mut()
                .expect("multi-generation stepping should keep a simulation session");
            let current_grid = sampled_grid
                .as_ref()
                .expect("multi-generation stepping should keep a sampled grid");
            let next_grid =
                if should_use_exact_simd_repeat_skip(current_grid, config.step_generations) {
                    Some(
                        simulation
                            .advance_simd_chunk_exact(current_grid, config.step_generations)
                            .0,
                    )
                } else {
                    simulation.load_hashlife_state(current_grid);
                    simulation.advance_hashlife_root(config.step_generations);
                    simulation
                        .sample_hashlife_state_grid(interactive_full_grid_policy())
                        .ok()
                };
            sampled_grid = next_grid;
            sampled_generation = sampled_generation.saturating_add(config.step_generations);
            changed_chunks = None;
        } else {
            let simulation = simulation
                .as_mut()
                .expect("session-based startup should keep a simulation session");
            simulation.advance_hashlife_root(config.step_generations);
            changed_chunks = None;
        }
    }

    println!("\x1b[?25h");
}

fn interactive_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: HASHLIFE_FULL_GRID_MAX_POPULATION,
        max_chunks: HASHLIFE_FULL_GRID_MAX_CHUNKS,
        max_bounds_span: i64::MAX,
    }
}

fn status_text(generation: u64, population: usize, classification: &str) -> String {
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

fn write_status_lines(out: &mut Vec<u8>, width: usize, status: &str, status_lines: usize) {
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

fn sample_visible_session_grid(
    simulation: &mut SimulationSession,
    view_width: usize,
    view_height: usize,
) -> ((Coord, Coord), BitGrid) {
    if let Some(bounds) = simulation.hashlife_bounds() {
        let viewport_width = view_width as Coord;
        let viewport_height = (view_height as Coord) * 2;
        let (min_x, min_y, max_x, max_y) = bounds;
        let centered_origin = compute_origin_for_bounds(view_width, view_height, bounds);
        let candidate_origins = [
            centered_origin,
            (min_x, min_y),
            (max_x - viewport_width + 1, min_y),
            (min_x, max_y - viewport_height + 1),
            (max_x - viewport_width + 1, max_y - viewport_height + 1),
        ];

        let mut best: Option<((Coord, Coord), BitGrid)> = None;
        for origin in candidate_origins {
            let max_x = origin.0 + viewport_width - 1;
            let max_y = origin.1 + viewport_height - 1;
            let render_grid = simulation
                .sample_hashlife_state_region(origin.0, origin.1, max_x, max_y)
                .expect("hashlife state should be sampleable in the visible region");
            let render_population = render_grid.population();
            let best_population = best
                .as_ref()
                .map(|(_, grid)| grid.population())
                .unwrap_or(0);
            if render_population > best_population {
                best = Some((origin, render_grid));
            }
        }

        best.unwrap_or_else(|| (centered_origin, BitGrid::empty()))
    } else {
        ((0, 0), BitGrid::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_view_height, sample_visible_session_grid, status_text, wrapped_line_count,
    };
    use a_simple_life::classify::Classification;
    use a_simple_life::engine::SimulationSession;
    use a_simple_life::generators::{pattern_by_name, random_soup};
    use a_simple_life::oracle::OracleSession;

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

    #[test]
    fn startup_status_includes_classification_text() {
        let status = status_text(1_000_000_000, 42, "likely_infinite(emitter_cycle, gen=300)");
        assert!(status.contains("classification=likely_infinite(emitter_cycle, gen=300)"));
    }

    #[test]
    fn empty_hashlife_state_renders_empty_visible_viewport() {
        let mut simulation = SimulationSession::new();
        simulation.load_hashlife_state(&a_simple_life::bitgrid::BitGrid::empty());

        let (origin, grid) = sample_visible_session_grid(&mut simulation, 80, 24);

        assert_eq!(origin, (0, 0));
        assert_eq!(grid.population(), 0);
        assert!(grid.bounds().is_none());
    }

    #[test]
    fn startup_target_generation_lands_session_at_requested_generation() {
        let initial = random_soup(60, 40, 35, 420);
        let target_generation = 100_000;
        let mut simulation = SimulationSession::new();
        let outcome = OracleSession::new(initial, 0, Default::default(), &mut simulation)
            .advance_runtime_target(target_generation, None);

        assert_eq!(outcome.final_generation, target_generation);
        assert_eq!(simulation.hashlife_generation(), target_generation);
        assert!(simulation.hashlife_population().unwrap_or(0) > 0);
        assert!(simulation.hashlife_bounds().is_some());
    }

    #[test]
    fn startup_target_generation_keeps_small_oscillator_first_seen_exact() {
        let initial = pattern_by_name("pulsar").unwrap();
        let target_generation = 1_000_000;
        let mut simulation = SimulationSession::new();
        let outcome = OracleSession::new(initial, 0, Default::default(), &mut simulation)
            .advance_runtime_target(target_generation, None);

        assert_eq!(outcome.final_generation, target_generation);
        assert_eq!(
            outcome.classification,
            Classification::Repeats {
                period: 3,
                first_seen: 0,
            }
        );
    }

    #[test]
    fn split_sparse_hashlife_viewport_keeps_live_cells_visible() {
        let mut grid = a_simple_life::bitgrid::BitGrid::empty();
        grid.set(0, 0, true);
        grid.set(1, 0, true);
        grid.set(2, 0, true);
        grid.set(500, 0, true);
        grid.set(501, 0, true);
        grid.set(502, 0, true);

        let mut simulation = SimulationSession::new();
        simulation.load_hashlife_state(&grid);

        let (_origin, visible) = sample_visible_session_grid(&mut simulation, 80, 24);
        assert!(
            visible.population() > 0,
            "viewport sampling should keep at least one live cell visible for split sparse states"
        );
    }

    #[test]
    fn huge_sparse_late_generation_like_viewport_keeps_live_cells_visible() {
        let mut grid = a_simple_life::bitgrid::BitGrid::empty();
        for y in 0..8 {
            for x in 0..8 {
                grid.set(x, y, true);
            }
        }
        grid.set(1_000_000, 0, true);
        grid.set(1_000_001, 1, true);
        grid.set(999_999, 2, true);

        let mut simulation = SimulationSession::new();
        simulation.load_hashlife_state(&grid);

        assert!(simulation.hashlife_population().unwrap_or(0) > 0);
        assert!(simulation.hashlife_bounds().is_some());

        let (_origin, visible) = sample_visible_session_grid(&mut simulation, 80, 24);
        assert!(
            visible.population() > 0,
            "late-generation-like sparse startup should keep at least one live cell visible in the initial viewport"
        );
        let (min_x, min_y, max_x, max_y) = visible
            .bounds()
            .expect("visible viewport should contain live cells");
        assert!(
            min_x <= 7 && min_y <= 7 && max_x >= 0 && max_y >= 0,
            "viewport should lock onto the dominant nearby active mass instead of empty space or a tiny distant mover"
        );
    }
}
