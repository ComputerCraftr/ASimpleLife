use a_simple_life::RequiredExt;
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
use a_simple_life::oracle::{OracleRuntimeState, OracleSession};
use a_simple_life::render::{
    TerminalBackbuffer, compute_origin_for_bounds, resized_viewport_origin, stable_viewport_origin,
};
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
    let mut startup_state_materialized = true;
    let mut startup_generation = 0;
    let mut runtime_unknown_tracks_generation = false;
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
            .or_invariant("startup target generation should create a simulation session");
        let outcome = OracleSession::new(initial.clone(), 0, HashMap::new(), simulation)
            .advance_runtime_target(start_generation, None);
        if let Some(failure) = outcome.failure {
            eprintln!(
                "failed to reach target generation {start_generation}: stopped at generation {}: {failure:?}",
                outcome.final_generation
            );
            std::process::exit(1);
        }
        if outcome.final_generation != start_generation {
            eprintln!(
                "failed to reach target generation {start_generation}: stopped at generation {}",
                outcome.final_generation
            );
            std::process::exit(1);
        }
        startup_population = Some(outcome.population);
        startup_state_materialized = outcome.state == OracleRuntimeState::RetainedHashLife;
        startup_generation = outcome.final_generation;
        runtime_unknown_tracks_generation = matches!(
            outcome.classification,
            a_simple_life::classify::Classification::Unknown { .. }
        );
        let classification = outcome.classification.to_string();
        if outcome.state == OracleRuntimeState::RetainedHashLife {
            classification
        } else {
            format!("{classification} state=modeled view=unavailable")
        }
    };
    if config.classify_only {
        println!("{classification}");
        return;
    }

    let grid = (start_generation == 0).then_some(initial);
    let mut game = if start_generation == 0 && config.step_generations == 1 {
        Some(GameOfLife::new_with_generation(
            grid.clone()
                .or_invariant("initial grid should exist for exact stepping"),
            startup_generation,
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
    let mut sampled_generation = startup_generation;
    let modeled_population = (!startup_state_materialized).then(|| {
        startup_population.or_invariant("modeled runtime outcome should include population")
    });
    let (mut terminal_width, mut terminal_height) = terminal_size(config.width, config.height);
    let initial_status = runtime_status_text(
        startup_generation,
        startup_population
            .or_else(|| grid.as_ref().map(BitGrid::population))
            .unwrap_or(0),
        &classification,
        runtime_unknown_tracks_generation,
    );
    let mut sampled_grid = if modeled_population.is_some() {
        Some(BitGrid::empty())
    } else {
        (start_generation == 0 && config.step_generations > 1).then(|| {
            grid.clone()
                .or_invariant("initial grid should exist for multi-generation stepping")
        })
    };
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
    let mut session_viewport = SessionViewport::new(view_width, view_height);

    print!("\x1b[2J\x1b[?25l");

    for _ in 0..config.steps {
        let (current_generation, current_population) = if let Some(game) = game.as_ref() {
            (game.generation(), game.grid().population())
        } else if let Some(population) = modeled_population {
            (sampled_generation, population)
        } else if let Some(grid) = sampled_grid.as_ref() {
            (sampled_generation, grid.population())
        } else {
            let simulation = simulation
                .as_ref()
                .or_invariant("session-based startup should keep a simulation session");
            (
                simulation.hashlife_generation(),
                simulation
                    .hashlife_population_count()
                    .map(|population| {
                        usize::try_from(population.lower_bound()).unwrap_or(usize::MAX)
                    })
                    .unwrap_or(0),
            )
        };
        let current_status = runtime_status_text(
            current_generation,
            current_population,
            &classification,
            runtime_unknown_tracks_generation,
        );
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
            session_viewport.resize(view_width, view_height);
            backbuffer.set_row_offset(status_lines);
            frame_buffer = Vec::with_capacity((view_width * view_height) + 64);
            stdout.write_all(b"\x1b[2J").or_invariant("required value");
            changed_chunks = None;
        }

        let session_render;
        let (session_origin, current_grid) = if let Some(game) = game.as_ref() {
            (None, game.grid())
        } else if let Some(grid) = sampled_grid.as_ref() {
            (None, grid)
        } else {
            let simulation = simulation
                .as_mut()
                .or_invariant("session-based startup should keep a simulation session");
            session_render = session_viewport.sample(simulation);
            (Some(session_render.0), &session_render.1)
        };

        if current_status != previous_status {
            status_buffer.clear();
            write_status_lines(
                &mut status_buffer,
                terminal_width,
                &current_status,
                status_lines,
            );
            stdout
                .write_all(&status_buffer)
                .or_invariant("required value");
            previous_status = current_status;
        }

        frame_buffer.clear();
        if let Some(origin) = session_origin {
            backbuffer
                .render_at_origin_into(current_grid, origin, &mut frame_buffer)
                .or_invariant("required value");
        } else {
            backbuffer
                .render_chunk_into(current_grid, changed_chunks.as_deref(), &mut frame_buffer)
                .or_invariant("required value");
        }
        write!(
            &mut frame_buffer,
            "\x1b[{};1H",
            view_height + status_lines + 1
        )
        .or_invariant("required value");
        stdout
            .write_all(&frame_buffer)
            .or_invariant("required value");
        stdout.flush().or_invariant("required value");

        thread::sleep(Duration::from_millis(config.delay_ms));
        if modeled_population.is_some() {
            changed_chunks = None;
        } else if let Some(game) = game.as_mut() {
            changed_chunks = Some(game.step_with_chunk_changes());
        } else if sampled_grid.is_some() {
            let simulation = simulation
                .as_mut()
                .or_invariant("multi-generation stepping should keep a simulation session");
            let current_grid = sampled_grid
                .as_ref()
                .or_invariant("multi-generation stepping should keep a sampled grid");
            let next_grid =
                if should_use_exact_simd_repeat_skip(current_grid, config.step_generations) {
                    Some(
                        simulation
                            .advance_simd_chunk_exact(current_grid, config.step_generations)
                            .0,
                    )
                } else {
                    if let Err(error) = simulation.try_load_hashlife_state(current_grid) {
                        eprintln!("HashLife conversion failed: {error:?}");
                        break;
                    }
                    if let Err(error) = simulation.advance_hashlife_root(config.step_generations) {
                        eprintln!("HashLife advance stopped early: {error:?}");
                        break;
                    }
                    simulation
                        .sample_hashlife_state_grid(interactive_full_grid_policy())
                        .ok()
                };
            sampled_grid = next_grid;
            sampled_generation = sampled_generation
                .checked_add(config.step_generations)
                .or_invariant("displayed generation overflow");
            changed_chunks = None;
        } else {
            let simulation = simulation
                .as_mut()
                .or_invariant("session-based startup should keep a simulation session");
            if let Err(error) = simulation.advance_hashlife_root(config.step_generations) {
                eprintln!("HashLife advance stopped early: {error:?}");
                break;
            }
            changed_chunks = None;
        }
    }

    println!("\x1b[?25h");
}

fn interactive_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: u128::from(HASHLIFE_FULL_GRID_MAX_POPULATION),
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

fn runtime_status_text(
    generation: u64,
    population: usize,
    classification: &str,
    unknown_tracks_generation: bool,
) -> String {
    if unknown_tracks_generation {
        return status_text(
            generation,
            population,
            &format!("unknown(after={generation})"),
        );
    }
    status_text(generation, population, classification)
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
    // Keep the final terminal row clear so rendered cells and the cursor never
    // trigger an implicit scroll at the bottom edge.
    terminal_height
        .saturating_sub(status_lines.saturating_add(1))
        .max(1)
}

fn write_status_lines(out: &mut Vec<u8>, width: usize, status: &str, status_lines: usize) {
    let width = width.max(1);
    for row in 1..=status_lines {
        write!(out, "\x1b[{};1H\x1b[K", row).or_invariant("required value");
    }

    let mut row = 1usize;
    let mut column_count = 0usize;
    write!(out, "\x1b[1;1H").or_invariant("required value");
    for ch in status.chars() {
        if ch == '\n' || column_count == width {
            row += 1;
            if row > status_lines {
                break;
            }
            write!(out, "\x1b[{};1H", row).or_invariant("required value");
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
fn sample_visible_session_grid(
    simulation: &mut SimulationSession,
    view_width: usize,
    view_height: usize,
) -> ((Coord, Coord), BitGrid) {
    let mut viewport = SessionViewport::new(view_width, view_height);
    viewport.sample(simulation)
}

#[derive(Clone, Debug)]
struct SessionViewport {
    origin: Option<(Coord, Coord)>,
    width: usize,
    height: usize,
    preserve_origin_once: bool,
}

impl SessionViewport {
    fn new(width: usize, height: usize) -> Self {
        Self {
            origin: None,
            width,
            height,
            preserve_origin_once: false,
        }
    }

    fn resize(&mut self, width: usize, height: usize) {
        self.origin = resized_viewport_origin(self.origin, self.width, self.height, width, height);
        self.preserve_origin_once = self.origin.is_some();
        self.width = width;
        self.height = height;
    }

    fn sample(&mut self, simulation: &mut SimulationSession) -> ((Coord, Coord), BitGrid) {
        let Some(bounds) = simulation.hashlife_bounds() else {
            self.origin = None;
            self.preserve_origin_once = false;
            return ((0, 0), BitGrid::empty());
        };
        let viewport_width = Coord::try_from(self.width)
            .or_invariant("terminal viewport width should fit coordinate domain");
        let viewport_height = Coord::try_from(self.height)
            .or_invariant("terminal viewport height should fit coordinate domain")
            .checked_mul(2)
            .or_invariant("terminal viewport height should fit doubled coordinate domain");
        let (min_x, min_y, max_x, max_y) = bounds;
        let centered_origin = compute_origin_for_bounds(self.width, self.height, bounds);
        let mut candidate_origins = vec![
            centered_origin,
            (min_x, min_y),
            (max_x - viewport_width + 1, min_y),
            (min_x, max_y - viewport_height + 1),
            (max_x - viewport_width + 1, max_y - viewport_height + 1),
        ];
        if let Some(origin) = self.origin {
            candidate_origins.push(origin);
        }
        candidate_origins.sort_unstable();
        candidate_origins.dedup();

        let coarse_candidate_count = candidate_origins.len();
        let mut candidates = Vec::with_capacity(coarse_candidate_count * 2);
        for &origin in &candidate_origins {
            let render_grid =
                sample_session_region(simulation, origin, viewport_width, viewport_height);
            candidates.push((origin, render_grid));
        }
        let refinement_limit = coarse_candidate_count.saturating_mul(4);
        let mut index = 0;
        while index < candidates.len() && candidates.len() < refinement_limit {
            let Some(visible_bounds) = candidates[index].1.bounds() else {
                index += 1;
                continue;
            };
            let origin = compute_origin_for_bounds(self.width, self.height, visible_bounds);
            if candidates
                .iter()
                .any(|(candidate_origin, _)| *candidate_origin == origin)
            {
                index += 1;
                continue;
            }
            let render_grid =
                sample_session_region(simulation, origin, viewport_width, viewport_height);
            candidates.push((origin, render_grid));
            index += 1;
        }
        let proposed_index = candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, (origin, grid))| {
                viewport_candidate_score(*origin, grid, viewport_width, viewport_height)
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let proposed = &candidates[proposed_index];
        let current_population = self.origin.map_or(0, |origin| {
            candidates
                .iter()
                .find(|(candidate, _)| *candidate == origin)
                .map_or(0, |(_, grid)| grid.population())
        });
        let selected_origin = if self.preserve_origin_once && current_population != 0 {
            self.origin
                .or_invariant("a preserved session viewport population requires an origin")
        } else {
            stable_viewport_origin(
                self.origin,
                proposed.0,
                current_population,
                proposed.1.population(),
                self.width,
                self.height,
            )
        };
        self.preserve_origin_once = false;
        let selected_index = candidates
            .iter()
            .position(|(origin, _)| *origin == selected_origin)
            .unwrap_or(proposed_index);
        self.origin = Some(selected_origin);
        candidates.swap_remove(selected_index)
    }
}

fn viewport_candidate_score(
    origin: (Coord, Coord),
    grid: &BitGrid,
    viewport_width: Coord,
    viewport_height: Coord,
) -> (
    usize,
    Coord,
    std::cmp::Reverse<u64>,
    std::cmp::Reverse<(Coord, Coord)>,
) {
    let Some((min_x, min_y, max_x, max_y)) = grid.bounds() else {
        return (0, 0, std::cmp::Reverse(u64::MAX), std::cmp::Reverse(origin));
    };
    let right = origin.0 + viewport_width - 1;
    let bottom = origin.1 + viewport_height - 1;
    let left_margin = min_x - origin.0;
    let right_margin = right - max_x;
    let top_margin = min_y - origin.1;
    let bottom_margin = bottom - max_y;
    let minimum_margin = left_margin
        .min(right_margin)
        .min(top_margin)
        .min(bottom_margin);
    let imbalance = left_margin
        .abs_diff(right_margin)
        .saturating_add(top_margin.abs_diff(bottom_margin));
    (
        grid.population(),
        minimum_margin,
        std::cmp::Reverse(imbalance),
        std::cmp::Reverse(origin),
    )
}

fn sample_session_region(
    simulation: &mut SimulationSession,
    origin: (Coord, Coord),
    viewport_width: Coord,
    viewport_height: Coord,
) -> BitGrid {
    let max_x = origin.0 + viewport_width - 1;
    let max_y = origin.1 + viewport_height - 1;
    simulation
        .sample_hashlife_state_region(origin.0, origin.1, max_x, max_y)
        .or_invariant("hashlife state should be sampleable in the visible region")
}

#[cfg(test)]
mod tests {
    use super::{
        SessionViewport, compute_view_height, runtime_status_text, sample_visible_session_grid,
        status_text, wrapped_line_count,
    };
    use a_simple_life::RequiredExt;
    use a_simple_life::classify::Classification;
    use a_simple_life::engine::SimulationSession;
    use a_simple_life::generators::{pattern_by_name, random_soup};
    use a_simple_life::life::GameOfLife;
    use a_simple_life::oracle::{OracleRuntimeState, OracleSession};

    #[test]
    fn wrapped_status_reduces_view_height() {
        let status = "generation=123 population=456 classification=likely_infinite(oracle_generation_limit, gen=1000000)";
        let status_lines = wrapped_line_count(status, 20);
        assert!(status_lines > 1);
        assert_eq!(compute_view_height(25, status_lines), 24 - status_lines);
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
        assert!(
            status.contains("classification=likely_infinite(emitter_cycle, gen=300)"),
            "startup status should include the full classification text\nstatus={status:?}"
        );
    }

    #[test]
    fn runtime_unknown_status_uses_the_displayed_generation_horizon() {
        let status = runtime_status_text(8_388_835, 116, "unknown(after=8388736)", true);

        assert_eq!(
            status,
            "generation=8388835 population=116 classification=unknown(after=8388835)"
        );
    }

    #[test]
    fn empty_hashlife_state_renders_empty_visible_viewport() {
        let mut simulation = SimulationSession::new();
        simulation
            .try_load_hashlife_state(&a_simple_life::bitgrid::BitGrid::empty())
            .or_invariant("test HashLife state should load");

        let (origin, grid) = sample_visible_session_grid(&mut simulation, 80, 24);

        assert_eq!(origin, (0, 0));
        assert_eq!(grid.population(), 0);
        assert!(
            grid.bounds().is_none(),
            "empty HashLife viewport should have no bounds\norigin={origin:?}\npopulation={}\ngrid={grid:?}",
            grid.population()
        );
    }

    #[test]
    fn startup_target_generation_retains_exact_projected_state() {
        let initial = random_soup(60, 40, 35, 420);
        let target_generation = 100_000;
        let mut simulation = SimulationSession::new();
        let outcome = OracleSession::new(initial, 0, Default::default(), &mut simulation)
            .advance_runtime_target(target_generation, None);

        assert_eq!(outcome.final_generation, target_generation);
        assert_eq!(outcome.state, OracleRuntimeState::RetainedHashLife);
        assert!(
            outcome.population > 0 && outcome.bounds_span > 0,
            "projected target generation should retain non-empty state metrics\ngeneration={target_generation}\noutcome={outcome:?}"
        );
        assert!(
            simulation.hashlife_loaded(),
            "an exact projected target should remain available to the renderer"
        );
        assert_eq!(simulation.hashlife_generation(), target_generation);
    }

    #[test]
    fn startup_target_generation_keeps_small_oscillator_first_seen_exact() {
        let initial = pattern_by_name("pulsar").or_invariant("required value");
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
        simulation
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");

        let (_origin, visible) = sample_visible_session_grid(&mut simulation, 80, 24);
        assert!(
            visible.population() > 0,
            "viewport sampling should keep at least one live cell visible for split sparse states"
        );
    }

    #[test]
    fn session_viewport_retains_one_large_cluster_when_a_distant_peer_barely_grows() {
        let mut grid = a_simple_life::bitgrid::BitGrid::empty();
        for y in 0..10 {
            for x in 0..10 {
                grid.set(x, y, true);
                grid.set(1_000 + x, y, true);
            }
        }
        let mut simulation = SimulationSession::new();
        simulation
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");
        let mut viewport = SessionViewport::new(20, 10);

        let (first_origin, first_visible) = viewport.sample(&mut simulation);
        assert_eq!(first_visible.population(), 100);

        grid.set(1_010, 0, true);
        simulation
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");
        let (second_origin, second_visible) = viewport.sample(&mut simulation);
        assert_eq!(
            second_origin, first_origin,
            "a one-cell population lead in a distant cluster must not make the viewport jump"
        );
        assert_eq!(second_visible.population(), 100);
    }

    #[test]
    fn session_viewport_resize_preserves_world_space_center() {
        let mut grid = a_simple_life::bitgrid::BitGrid::empty();
        for y in 0..10 {
            for x in 0..10 {
                grid.set(x, y, true);
            }
        }
        let mut simulation = SimulationSession::new();
        simulation
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");
        let mut viewport = SessionViewport::new(20, 10);
        let (old_origin, _) = viewport.sample(&mut simulation);

        viewport.resize(30, 15);
        let expected_origin = (old_origin.0 + 10 - 15, old_origin.1 + 10 - 15);
        let (resized_origin, visible) = viewport.sample(&mut simulation);

        assert_eq!(
            resized_origin, expected_origin,
            "resize should preserve the old world-space center instead of re-anchoring the pattern"
        );
        assert_eq!(visible.population(), 100);
    }

    #[test]
    fn session_viewport_does_not_cycle_with_blinker_bounds() {
        let grid =
            a_simple_life::bitgrid::BitGrid::from_cells(&[(100, 100), (101, 100), (102, 100)]);
        let mut game = GameOfLife::new(grid);
        let mut simulation = SimulationSession::new();
        let mut viewport = SessionViewport::new(20, 10);
        let mut origins = Vec::new();

        for _ in 0..6 {
            simulation
                .try_load_hashlife_state(game.grid())
                .or_invariant("blinker phase should load");
            let (origin, visible) = viewport.sample(&mut simulation);
            origins.push(origin);
            assert_eq!(visible.population(), 3);
            game.step();
        }

        assert!(
            origins.iter().all(|origin| *origin == origins[0]),
            "a bounded oscillator that remains fully visible must not move the viewport between recurring phases\norigins={origins:?}"
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
        simulation
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");

        assert!(
            simulation
                .hashlife_population_count()
                .is_some_and(|population| !population.is_zero()),
            "huge sparse viewport regression should keep a non-empty HashLife state\npopulation={:?}\nbounds={:?}",
            simulation.hashlife_population_count(),
            simulation.hashlife_bounds()
        );
        assert!(
            simulation.hashlife_bounds().is_some(),
            "huge sparse viewport regression should keep bounds metadata\npopulation={:?}",
            simulation.hashlife_population_count()
        );

        let (_origin, visible) = sample_visible_session_grid(&mut simulation, 80, 24);
        assert!(
            visible.population() > 0,
            "late-generation-like sparse startup should keep at least one live cell visible in the initial viewport"
        );
        let (min_x, min_y, max_x, max_y) = visible
            .bounds()
            .or_invariant("visible viewport should contain live cells");
        assert!(
            min_x <= 7 && min_y <= 7 && max_x >= 0 && max_y >= 0,
            "viewport should lock onto the dominant nearby active mass instead of empty space or a tiny distant mover"
        );
    }

    #[test]
    fn late_r_pentomino_viewport_centers_the_selected_component_away_from_edges() {
        let initial = pattern_by_name("r_pentomino").or_invariant("required value");
        let target_generation = 100_000_000;
        let mut simulation = SimulationSession::new();
        let outcome = OracleSession::new(initial, 0, Default::default(), &mut simulation)
            .advance_runtime_target(target_generation, None);
        assert_eq!(outcome.final_generation, target_generation);

        let (origin, visible) = sample_visible_session_grid(&mut simulation, 80, 24);
        let (min_x, min_y, max_x, max_y) = visible
            .bounds()
            .or_invariant("late R-pentomino viewport should contain live cells");
        let margins = (
            min_x - origin.0,
            origin.0 + 79 - max_x,
            min_y - origin.1,
            origin.1 + 47 - max_y,
        );
        assert!(
            margins.0.abs_diff(margins.1) <= 1 && margins.2.abs_diff(margins.3) <= 1,
            "selected R-pentomino component should be centered in both axes\norigin={origin:?}\nbounds={:?}\nmargins={margins:?}",
            visible.bounds()
        );
        assert!(
            margins.0 > 0 && margins.1 > 0 && margins.2.saturating_add(margins.3) > 0,
            "selected R-pentomino component should retain centering space; the reserved terminal gutter protects a one-cell vertical remainder\norigin={origin:?}\nbounds={:?}\nmargins={margins:?}\nvisible={visible:?}",
            visible.bounds()
        );
    }
}
