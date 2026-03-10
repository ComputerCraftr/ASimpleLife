use crate::bitgrid::BitGrid;
use std::io::{self, Write};

#[derive(Clone, Debug)]
pub struct TerminalBackbuffer {
    width: usize,
    height: usize,
    origin: Option<(i32, i32)>,
    cells: Vec<u8>,
    dirty_rows: Vec<Option<(usize, usize)>>,
}

impl TerminalBackbuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            origin: None,
            cells: vec![0; width * height],
            dirty_rows: vec![None; height],
        }
    }

    pub fn render_into<W: Write>(
        &mut self,
        grid: &BitGrid,
        changed_cells: Option<&[(i32, i32)]>,
        out: &mut W,
    ) -> io::Result<()> {
        let next_origin = compute_origin(self.width, self.height, grid);

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

    fn rebuild_all(&mut self, grid: &BitGrid) {
        for row in 0..self.height {
            for col in 0..self.width {
                self.write_cell(grid, row, col);
            }
            self.dirty_rows[row] = Some((0, self.width - 1));
        }
    }

    fn update_terminal_cell_for_world(&mut self, grid: &BitGrid, x: i32, y: i32) {
        let Some((origin_x, origin_y)) = self.origin else {
            return;
        };

        let col = x - origin_x;
        if !(0..self.width as i32).contains(&col) {
            return;
        }

        let relative_y = y - origin_y;
        let row = relative_y.div_euclid(2);
        if !(0..self.height as i32).contains(&row) {
            return;
        }

        self.write_cell(grid, row as usize, col as usize);
    }

    fn write_cell(&mut self, grid: &BitGrid, row: usize, col: usize) {
        let (origin_x, origin_y) = self.origin.unwrap_or((0, 0));
        let x = origin_x + col as i32;
        let y = origin_y + (row as i32 * 2);
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

            write_cursor_move(out, row + 2, start + 1)?;
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

fn compute_origin(width: usize, height: usize, grid: &BitGrid) -> (i32, i32) {
    compute_origin_for_cells(width, height, &grid.live_cells())
}

pub(crate) fn compute_origin_for_cells(
    width: usize,
    height: usize,
    cells: &[(i32, i32)],
) -> (i32, i32) {
    if cells.is_empty() {
        return (0, 0);
    }

    let mut min_x = cells[0].0;
    let mut max_x = cells[0].0;
    let mut min_y = cells[0].1;
    let mut max_y = cells[0].1;
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;

    for &(x, y) in cells {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        sum_x += x as i64;
        sum_y += y as i64;
    }

    let viewport_width = width as i32;
    let viewport_height = (height as i32) * 2;
    let centroid_x = (sum_x / cells.len() as i64) as i32;
    let centroid_y = (sum_y / cells.len() as i64) as i32;
    let ideal_x = centroid_x - viewport_width / 2;
    let ideal_y = centroid_y - viewport_height / 2;
    let min_origin_x = max_x - viewport_width + 1;
    let min_origin_y = max_y - viewport_height + 1;
    let max_origin_x = min_x;
    let max_origin_y = min_y;

    (
        clamp_i32(ideal_x, min_origin_x, max_origin_x),
        clamp_i32(ideal_y, min_origin_y, max_origin_y),
    )
}

fn clamp_i32(value: i32, low: i32, high: i32) -> i32 {
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
