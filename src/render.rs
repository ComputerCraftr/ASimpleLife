use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::life::ChunkDiff;
use std::collections::{HashSet, VecDeque};
use std::io::{self, Write};

#[derive(Clone, Debug)]
pub struct TerminalBackbuffer {
    width: usize,
    height: usize,
    row_offset: usize,
    origin: Option<Cell>,
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
        let next_origin = compute_origin_for_cells(self.width, self.height, &grid.live_cells());

        if self.origin != Some(next_origin) || changed_cells.is_none() {
            self.origin = Some(next_origin);
            self.rebuild_all(grid);
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
        let next_origin = compute_origin_for_cells(self.width, self.height, &grid.live_cells());

        if self.origin != Some(next_origin) || changed_chunks.is_none() {
            self.origin = Some(next_origin);
            self.rebuild_all(grid);
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
        self.rebuild_all(grid);
        self.flush_dirty(out)
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }

        self.width = width;
        self.height = height;
        self.origin = None;
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
        if !(0..self.width as Coord).contains(&col) {
            return;
        }

        let relative_y = y - origin_y;
        let row = relative_y.div_euclid(2);
        if !(0..self.height as Coord).contains(&row) {
            return;
        }

        self.write_cell(
            grid,
            usize::try_from(row).expect("row exceeded usize"),
            usize::try_from(col).expect("column exceeded usize"),
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
            let bit = remaining.trailing_zeros() as Coord;
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
        let max_col = (max_world_x - origin_x).min(self.width as Coord - 1);
        if min_col > max_col {
            return;
        }

        let min_row = (min_world_y - origin_y).div_euclid(2).max(0);
        let max_row = (max_world_y - origin_y)
            .div_euclid(2)
            .min(self.height as Coord - 1);
        if min_row > max_row {
            return;
        }

        for row in usize::try_from(min_row).expect("row range start exceeded usize")
            ..=usize::try_from(max_row).expect("row range end exceeded usize")
        {
            for col in usize::try_from(min_col).expect("column range start exceeded usize")
                ..=usize::try_from(max_col).expect("column range end exceeded usize")
            {
                self.write_cell(grid, row, col);
            }
        }
    }

    fn write_cell(&mut self, grid: &BitGrid, row: usize, col: usize) {
        let (origin_x, origin_y) = self.origin.unwrap_or((0, 0));
        let x = origin_x + col as Coord;
        let y = origin_y + (row as Coord * 2);
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

fn bounds_area(bounds: (Coord, Coord, Coord, Coord)) -> i128 {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = i128::from(max_x - min_x + 1);
    let height = i128::from(max_y - min_y + 1);
    width * height
}

fn bounds_fit_viewport(width: usize, height: usize, bounds: (Coord, Coord, Coord, Coord)) -> bool {
    let (min_x, min_y, max_x, max_y) = bounds;
    let viewport_width = width as Coord;
    let viewport_height = (height as Coord) * 2;
    let bounds_width = max_x - min_x + 1;
    let bounds_height = max_y - min_y + 1;

    bounds_width <= viewport_width && bounds_height <= viewport_height
}

fn centroid(cells: &[Cell]) -> (i64, i64) {
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;

    for &(x, y) in cells {
        sum_x += x;
        sum_y += y;
    }

    (sum_x / cells.len() as i64, sum_y / cells.len() as i64)
}

fn centroid_distance_sq(center: (i64, i64), global_centroid: (i64, i64)) -> i128 {
    let dx = i128::from(center.0 - global_centroid.0);
    let dy = i128::from(center.1 - global_centroid.1);
    dx * dx + dy * dy
}

pub fn compute_origin_for_bounds(
    width: usize,
    height: usize,
    bounds: (Coord, Coord, Coord, Coord),
) -> Cell {
    let (min_x, min_y, max_x, max_y) = bounds;
    let viewport_width = width as Coord;
    let viewport_height = (height as Coord) * 2;
    let center_x = (min_x + max_x) / 2;
    let center_y = (min_y + max_y) / 2;
    let ideal_x = center_x - viewport_width / 2;
    let ideal_y = center_y - viewport_height / 2;
    let min_origin_x = max_x - viewport_width + 1;
    let min_origin_y = max_y - viewport_height + 1;
    let max_origin_x = min_x;
    let max_origin_y = min_y;
    (
        clamp_coord(ideal_x, min_origin_x, max_origin_x),
        clamp_coord(ideal_y, min_origin_y, max_origin_y),
    )
}

fn clamp_coord(value: Coord, low: Coord, high: Coord) -> Coord {
    if low > high {
        return high;
    }
    value.clamp(low, high)
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
