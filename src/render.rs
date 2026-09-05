use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::life::ChunkDiff;
use std::collections::{HashSet, VecDeque};
use std::io::{self, Write};

pub(crate) mod activity;
mod viewport;

pub use viewport::{
    ViewportController, ViewportError, ViewportMode, ViewportSample, ViewportSource,
};

#[derive(Clone, Debug)]
pub struct TerminalBackbuffer {
    width: usize,
    height: usize,
    row_offset: usize,
    origin: Option<Cell>,
    preserve_origin_once: bool,
    needs_rebuild: bool,
    cells: Vec<u8>,
    dirty_rows: Vec<Option<(usize, usize)>>,
}

impl TerminalBackbuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            row_offset: 1,
            origin: None,
            preserve_origin_once: false,
            needs_rebuild: true,
            cells: vec![0; width * height],
            dirty_rows: vec![None; height],
        }
    }

    pub fn render_into<W: Write>(
        &mut self,
        grid: &BitGrid,
        changed_cells: Option<&[Cell]>,
        out: &mut W,
    ) -> io::Result<()> {
        let next_origin = stable_origin_for_cells(
            self.width,
            self.height,
            &grid.live_cells(),
            self.origin,
            self.preserve_origin_once,
        );
        self.preserve_origin_once = false;

        if self.needs_rebuild || self.origin != Some(next_origin) || changed_cells.is_none() {
            self.origin = Some(next_origin);
            self.rebuild_all(grid);
            self.needs_rebuild = false;
        } else if let Some(changed_cells) = changed_cells {
            for &(x, y) in changed_cells {
                self.update_terminal_cell_for_world(grid, x, y);
            }
        }

        self.flush_dirty(out)
    }

    pub fn render_chunk_into<W: Write>(
        &mut self,
        grid: &BitGrid,
        changed_chunks: Option<&[ChunkDiff]>,
        out: &mut W,
    ) -> io::Result<()> {
        let next_origin = stable_origin_for_cells(
            self.width,
            self.height,
            &grid.live_cells(),
            self.origin,
            self.preserve_origin_once,
        );
        self.preserve_origin_once = false;

        if self.needs_rebuild || self.origin != Some(next_origin) || changed_chunks.is_none() {
            self.origin = Some(next_origin);
            self.rebuild_all(grid);
            self.needs_rebuild = false;
        } else if let Some(changed_chunks) = changed_chunks {
            for &diff in changed_chunks {
                self.update_terminal_cells_for_chunk(grid, diff);
            }
        }

        self.flush_dirty(out)
    }

    pub fn render_at_origin_into<W: Write>(
        &mut self,
        grid: &BitGrid,
        origin: Cell,
        out: &mut W,
    ) -> io::Result<()> {
        if self.origin != Some(origin) {
            self.origin = Some(origin);
        }
        self.preserve_origin_once = false;
        self.rebuild_all(grid);
        self.needs_rebuild = false;
        self.flush_dirty(out)
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }

        self.origin = resized_viewport_origin(self.origin, self.width, self.height, width, height);
        self.preserve_origin_once = self.origin.is_some();
        self.needs_rebuild = true;
        self.width = width;
        self.height = height;
        self.cells = vec![0; width * height];
        self.dirty_rows = vec![None; height];
    }

    pub fn set_row_offset(&mut self, row_offset: usize) {
        self.row_offset = row_offset;
    }

    fn rebuild_all(&mut self, grid: &BitGrid) {
        for row in 0..self.height {
            for col in 0..self.width {
                self.write_cell(grid, row, col);
            }
            self.dirty_rows[row] = Some((0, self.width - 1));
        }
    }

    fn update_terminal_cell_for_world(&mut self, grid: &BitGrid, x: Coord, y: Coord) {
        let Some((origin_x, origin_y)) = self.origin else {
            return;
        };

        let col = x - origin_x;
        if !(0..viewport_dimension(self.width)).contains(&col) {
            return;
        }

        let relative_y = y - origin_y;
        let row = relative_y.div_euclid(2);
        if !(0..viewport_dimension(self.height)).contains(&row) {
            return;
        }

        self.write_cell(
            grid,
            usize::try_from(row).or_invariant("row exceeded usize"),
            usize::try_from(col).or_invariant("column exceeded usize"),
        );
    }

    fn update_terminal_cells_for_chunk(&mut self, grid: &BitGrid, diff: ChunkDiff) {
        let Some((origin_x, origin_y)) = self.origin else {
            return;
        };
        if diff.diff_bits == 0 {
            return;
        }

        let mut remaining = diff.diff_bits;
        let mut min_local_x = Coord::MAX;
        let mut max_local_x = Coord::MIN;
        let mut min_local_y = Coord::MAX;
        let mut max_local_y = Coord::MIN;

        while remaining != 0 {
            let bit = Coord::from(remaining.trailing_zeros());
            min_local_x = min_local_x.min(bit % 8);
            max_local_x = max_local_x.max(bit % 8);
            min_local_y = min_local_y.min(bit / 8);
            max_local_y = max_local_y.max(bit / 8);
            remaining &= remaining - 1;
        }

        let min_world_x = diff.cx * 8 + min_local_x;
        let max_world_x = diff.cx * 8 + max_local_x;
        let min_world_y = diff.cy * 8 + min_local_y;
        let max_world_y = diff.cy * 8 + max_local_y;

        let min_col = (min_world_x - origin_x).max(0);
        let max_col = (max_world_x - origin_x).min(viewport_dimension(self.width) - 1);
        if min_col > max_col {
            return;
        }

        let min_row = (min_world_y - origin_y).div_euclid(2).max(0);
        let max_row = (max_world_y - origin_y)
            .div_euclid(2)
            .min(viewport_dimension(self.height) - 1);
        if min_row > max_row {
            return;
        }

        for row in usize::try_from(min_row).or_invariant("row range start exceeded usize")
            ..=usize::try_from(max_row).or_invariant("row range end exceeded usize")
        {
            for col in usize::try_from(min_col).or_invariant("column range start exceeded usize")
                ..=usize::try_from(max_col).or_invariant("column range end exceeded usize")
            {
                self.write_cell(grid, row, col);
            }
        }
    }

    fn write_cell(&mut self, grid: &BitGrid, row: usize, col: usize) {
        let (origin_x, origin_y) = self.origin.unwrap_or((0, 0));
        let x = origin_x + viewport_dimension(col);
        let y = origin_y + (viewport_dimension(row) * 2);
        let encoded = encode_cell(grid.get(x, y), grid.get(x, y + 1));
        let idx = row * self.width + col;

        if self.cells[idx] != encoded {
            self.cells[idx] = encoded;
            self.mark_dirty(row, col);
        }
    }

    fn mark_dirty(&mut self, row: usize, col: usize) {
        match &mut self.dirty_rows[row] {
            Some((start, end)) => {
                *start = (*start).min(col);
                *end = (*end).max(col);
            }
            slot @ None => *slot = Some((col, col)),
        }
    }

    fn flush_dirty<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        for row in 0..self.height {
            let Some((start, end)) = self.dirty_rows[row].take() else {
                continue;
            };

            write_cursor_move(out, row + self.row_offset + 1, start + 1)?;
            for col in start..=end {
                let mut encoded = [0_u8; 4];
                let ch = decode_cell(self.cells[row * self.width + col]);
                let bytes = ch.encode_utf8(&mut encoded).as_bytes();
                out.write_all(bytes)?;
            }
        }

        Ok(())
    }
}

pub(crate) fn compute_origin_for_cells(width: usize, height: usize, cells: &[Cell]) -> Cell {
    if cells.is_empty() {
        return (0, 0);
    }

    let all_bounds = component_bounds(cells);
    if bounds_fit_viewport(width, height, all_bounds) {
        return compute_origin_for_bounds(width, height, all_bounds);
    }

    let focus_cells = dominant_component_cells(cells);

    compute_origin_for_bounds(width, height, component_bounds(&focus_cells))
}

pub fn stable_viewport_origin(
    current: Option<Cell>,
    proposed: Cell,
    current_population: usize,
    _proposed_population: usize,
    _width: usize,
    _height: usize,
) -> Cell {
    let Some(current) = current else {
        return proposed;
    };
    // Population changes do not imply motion: distant oscillators can swap
    // rankings every phase. Retain the occupied focus until it is lost.
    if current_population != 0 {
        current
    } else {
        proposed
    }
}

pub fn resized_viewport_origin(
    origin: Option<Cell>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
) -> Option<Cell> {
    origin.map(|(x, y)| {
        let old_width = i128::try_from(old_width).or_invariant("terminal width should fit i128");
        let old_height = i128::try_from(old_height).or_invariant("terminal height should fit i128");
        let new_width = i128::try_from(new_width).or_invariant("terminal width should fit i128");
        let new_height = i128::try_from(new_height).or_invariant("terminal height should fit i128");
        let center_x = i128::from(x) + old_width / 2;
        let center_y = i128::from(y) + old_height;
        (
            Coord::try_from(center_x - new_width / 2)
                .or_invariant("resized viewport x origin exceeded Coord"),
            Coord::try_from(center_y - new_height)
                .or_invariant("resized viewport y origin exceeded Coord"),
        )
    })
}

fn stable_origin_for_cells(
    width: usize,
    height: usize,
    cells: &[Cell],
    current: Option<Cell>,
    preserve_current: bool,
) -> Cell {
    let proposed = compute_origin_for_cells(width, height, cells);
    let current_population = current.map_or(0, |origin| {
        viewport_population(cells, origin, width, height)
    });
    let proposed_population = viewport_population(cells, proposed, width, height);
    if preserve_current && current_population != 0 {
        return current.or_invariant("a preserved viewport population requires an origin");
    }
    stable_viewport_origin(
        current,
        proposed,
        current_population,
        proposed_population,
        width,
        height,
    )
}

fn viewport_population(cells: &[Cell], origin: Cell, width: usize, height: usize) -> usize {
    let width = viewport_dimension(width);
    let height = viewport_dimension(height);
    let max_x = origin.0.saturating_add(width - 1);
    let max_y = origin
        .1
        .saturating_add(height.saturating_mul(2).saturating_sub(1));
    cells
        .iter()
        .filter(|&&(x, y)| x >= origin.0 && x <= max_x && y >= origin.1 && y <= max_y)
        .count()
}

fn viewport_dimension(value: usize) -> Coord {
    Coord::try_from(value).or_invariant("terminal dimension exceeded Coord")
}

fn dominant_component_cells(cells: &[Cell]) -> Vec<Cell> {
    let occupied: HashSet<Cell> = cells.iter().copied().collect();
    let mut remaining = occupied.clone();
    let mut best_component = Vec::new();
    let mut best_bounds = (Coord::MAX, Coord::MAX, Coord::MIN, Coord::MIN);
    let global_centroid = centroid(cells);

    while let Some(&start) = remaining.iter().next() {
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        let mut bounds = (start.0, start.1, start.0, start.1);
        remaining.remove(&start);

        while let Some((x, y)) = queue.pop_front() {
            component.push((x, y));
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.min(y);
            bounds.2 = bounds.2.max(x);
            bounds.3 = bounds.3.max(y);

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let neighbor = (x + dx, y + dy);
                    if remaining.remove(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if component_better_than(
            &component,
            bounds,
            &best_component,
            best_bounds,
            global_centroid,
        ) {
            best_bounds = bounds;
            best_component = component;
        }
    }

    best_component
}

fn component_better_than(
    candidate: &[Cell],
    candidate_bounds: (Coord, Coord, Coord, Coord),
    best: &[Cell],
    best_bounds: (Coord, Coord, Coord, Coord),
    global_centroid: (i64, i64),
) -> bool {
    if best.is_empty() {
        return true;
    }

    let candidate_len = candidate.len();
    let best_len = best.len();
    if candidate_len != best_len {
        return candidate_len > best_len;
    }

    let candidate_area = bounds_area(candidate_bounds);
    let best_area = bounds_area(best_bounds);
    if candidate_area != best_area {
        return candidate_area < best_area;
    }

    let candidate_distance = centroid_distance_sq(centroid(candidate), global_centroid);
    let best_distance = centroid_distance_sq(centroid(best), global_centroid);
    if candidate_distance != best_distance {
        return candidate_distance < best_distance;
    }

    let candidate_anchor = (candidate_bounds.0, candidate_bounds.1);
    let best_anchor = (best_bounds.0, best_bounds.1);
    candidate_anchor < best_anchor
}

fn component_bounds(cells: &[Cell]) -> (Coord, Coord, Coord, Coord) {
    let mut min_x = cells[0].0;
    let mut max_x = cells[0].0;
    let mut min_y = cells[0].1;
    let mut max_y = cells[0].1;

    for &(x, y) in cells {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

fn bounds_area(bounds: (Coord, Coord, Coord, Coord)) -> u128 {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = u128::try_from(i128::from(max_x) - i128::from(min_x) + 1)
        .or_invariant("component width should be positive");
    let height = u128::try_from(i128::from(max_y) - i128::from(min_y) + 1)
        .or_invariant("component height should be positive");
    width.saturating_mul(height)
}

fn bounds_fit_viewport(width: usize, height: usize, bounds: (Coord, Coord, Coord, Coord)) -> bool {
    let (min_x, min_y, max_x, max_y) = bounds;
    let viewport_width = u128::try_from(width).or_invariant("viewport width should fit u128");
    let viewport_height = u128::try_from(height)
        .or_invariant("viewport height should fit u128")
        .saturating_mul(2);
    let bounds_width = u128::try_from(i128::from(max_x) - i128::from(min_x) + 1)
        .or_invariant("component width should be positive");
    let bounds_height = u128::try_from(i128::from(max_y) - i128::from(min_y) + 1)
        .or_invariant("component height should be positive");

    bounds_width <= viewport_width && bounds_height <= viewport_height
}

fn centroid(cells: &[Cell]) -> (i64, i64) {
    let mut sum_x = 0_i128;
    let mut sum_y = 0_i128;

    for &(x, y) in cells {
        sum_x += i128::from(x);
        sum_y += i128::from(y);
    }

    let count = i128::try_from(cells.len()).or_invariant("cell count should fit i128");
    (
        i64::try_from(sum_x / count).or_invariant("centroid x should remain in coordinate range"),
        i64::try_from(sum_y / count).or_invariant("centroid y should remain in coordinate range"),
    )
}

fn centroid_distance_sq(center: (i64, i64), global_centroid: (i64, i64)) -> crate::wide_math::U129 {
    crate::wide_math::squared_i64_distance(center, global_centroid)
}

pub fn compute_origin_for_bounds(
    width: usize,
    height: usize,
    bounds: (Coord, Coord, Coord, Coord),
) -> Cell {
    let (min_x, min_y, max_x, max_y) = bounds;
    let viewport_width = i128::try_from(width).or_invariant("viewport width should fit i128");
    let viewport_height = i128::try_from(height)
        .or_invariant("viewport height should fit i128")
        .saturating_mul(2);
    let min_x = i128::from(min_x);
    let min_y = i128::from(min_y);
    let max_x = i128::from(max_x);
    let max_y = i128::from(max_y);
    // Center the inclusive pattern and viewport intervals. Computing their
    // midpoints separately biases even-sized intervals by one cell per side.
    let ideal_x = (min_x + max_x - viewport_width + 1).div_euclid(2);
    let ideal_y = (min_y + max_y - viewport_height + 1).div_euclid(2);
    (
        Coord::try_from(ideal_x).or_invariant("viewport x origin exceeded Coord"),
        Coord::try_from(ideal_y).or_invariant("viewport y origin exceeded Coord"),
    )
}

fn encode_cell(top: bool, bottom: bool) -> u8 {
    match (top, bottom) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

fn decode_cell(encoded: u8) -> char {
    match encoded {
        0 => ' ',
        1 => '▀',
        2 => '▄',
        3 => '█',
        _ => ' ',
    }
}

fn write_cursor_move<W: Write>(out: &mut W, row: usize, col: usize) -> io::Result<()> {
    write!(out, "\x1b[{row};{col}H")
}

#[cfg(test)]
mod wide_geometry_tests {
    use super::*;

    #[test]
    fn centroid_handles_opposite_coordinate_extremes_without_narrowing() {
        let cells = [(i64::MIN, i64::MIN), (i64::MAX, i64::MAX)];
        assert_eq!(centroid(&cells), (0, 0));
        assert!(centroid_distance_sq(cells[0], cells[1]) > crate::wide_math::U129::default());
    }
}
