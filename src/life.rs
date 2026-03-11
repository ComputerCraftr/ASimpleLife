use std::collections::HashSet;

use bytemuck::{must_cast, must_cast_mut, must_cast_ref};
use wide::{i8x16, u8x16, u16x8, u16x16, u16x32, u64x8};

use crate::bitgrid::{BitGrid, CHUNK_SIZE};
use crate::memo::{ChunkNeighborhood, Memo};

const ROW_LOW_BYTE_MASK: u16x8 = u16x8::splat(0x00FF);
const ROW_BLOCK_LOW_BYTE_MASK: u16x32 = u16x32::splat(0x00FF);
const SHIFT_ROWS_DOWN_BYTES: i8x16 =
    i8x16::new([-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
const SHIFT_ROWS_UP_BYTES: i8x16 =
    i8x16::new([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1]);

type ChunkRowBatch = [u16x32; 2];
type ChunkRowView = [u16x8; 8];

struct DiagonalSpec {
    edge_row: usize,
    edge_col: usize,
    shift_cols: i32,
    shift_rows: i32,
}

impl DiagonalSpec {
    const fn new(edge_row: usize, edge_col: usize, shift_cols: i32, shift_rows: i32) -> Self {
        Self {
            edge_row,
            edge_col,
            shift_cols,
            shift_rows,
        }
    }
}

// Public API
#[derive(Clone, Debug)]
pub struct GameOfLife {
    grid: BitGrid,
    generation: usize,
    memo: Memo,
}

impl GameOfLife {
    pub fn new(grid: BitGrid) -> Self {
        Self {
            grid,
            generation: 0,
            memo: Memo::default(),
        }
    }

    pub fn grid(&self) -> &BitGrid {
        &self.grid
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn step_with_changes(&mut self) -> Vec<(i32, i32)> {
        let (next, changed) = step_grid_with_changes_and_memo(&self.grid, &mut self.memo);
        self.grid = next;
        self.generation += 1;
        changed
    }
}

#[cfg(test)]
pub fn step_grid(grid: &BitGrid) -> BitGrid {
    let mut memo = Memo::default();
    step_grid_with_changes_and_memo(grid, &mut memo).0
}

// Stepping pipeline
pub fn step_grid_with_changes_and_memo(
    grid: &BitGrid,
    memo: &mut Memo,
) -> (BitGrid, Vec<(i32, i32)>) {
    if grid.is_empty() {
        return (BitGrid::new(), Vec::new());
    }

    let target_chunks = collect_target_chunks(grid);
    let mut next = BitGrid::with_chunk_capacity(target_chunks.len());
    let mut changed = Vec::new();
    let mut pending = Vec::with_capacity(8);

    for (cx, cy) in target_chunks {
        let neighborhood = build_neighborhood(grid, cx, cy);
        let current_bits = grid.chunk_bits(cx, cy);
        if let Some(next_bits) = memo.get_chunk_transition(&neighborhood) {
            apply_chunk_step(&mut next, &mut changed, cx, cy, current_bits, next_bits);
        } else {
            pending.push((cx, cy, current_bits, neighborhood));
            if pending.len() == 8 {
                flush_pending_chunks(&mut pending, memo, &mut next, &mut changed);
            }
        }
    }
    flush_pending_chunks(&mut pending, memo, &mut next, &mut changed);

    (next, changed)
}

fn flush_pending_chunks(
    pending: &mut Vec<(i32, i32, u64, ChunkNeighborhood)>,
    memo: &mut Memo,
    next: &mut BitGrid,
    changed: &mut Vec<(i32, i32)>,
) {
    if pending.is_empty() {
        return;
    }

    let next_bits = evolve_center_chunks_bitwise_batch_from_pending(pending);

    for ((cx, cy, current_bits, neighborhood), next_bits) in
        pending.drain(..).zip(next_bits.into_iter())
    {
        memo.insert_chunk_transition(neighborhood, next_bits);
        apply_chunk_step(next, changed, cx, cy, current_bits, next_bits);
    }
}

fn apply_chunk_step(
    next: &mut BitGrid,
    changed: &mut Vec<(i32, i32)>,
    cx: i32,
    cy: i32,
    current_bits: u64,
    next_bits: u64,
) {
    if next_bits != 0 {
        next.set_chunk_bits(cx, cy, next_bits);
    }
    append_changed_cells(changed, cx, cy, current_bits ^ next_bits);
}

// Neighborhood collection
fn collect_target_chunks(grid: &BitGrid) -> HashSet<(i32, i32)> {
    let mut targets = HashSet::new();
    for (cx, cy) in grid.chunk_coords() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                targets.insert((cx + dx, cy + dy));
            }
        }
    }
    targets
}

fn build_neighborhood(grid: &BitGrid, cx: i32, cy: i32) -> ChunkNeighborhood {
    let mut chunks = [0_u64; 9];
    let mut index = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            chunks[index] = grid.chunk_bits(cx + dx, cy + dy);
            index += 1;
        }
    }
    ChunkNeighborhood(chunks)
}

// Evolution kernels
fn evolve_center_chunk_bitwise(neighborhood: &ChunkNeighborhood) -> u64 {
    let [nw, n, ne, w, center, e, sw, s, se] = neighborhood.0;
    let north = align_vertical_neighbor(center, n, 7, 1);
    let south = align_vertical_neighbor(center, s, 0, -1);
    let west = align_horizontal_neighbor(center, w, 7, 1);
    let east = align_horizontal_neighbor(center, e, 0, -1);
    let northwest = align_diagonal_neighbor(center, n, w, nw, DiagonalSpec::new(7, 7, 1, 1));
    let northeast = align_diagonal_neighbor(center, n, e, ne, DiagonalSpec::new(7, 0, -1, 1));
    let southwest = align_diagonal_neighbor(center, s, w, sw, DiagonalSpec::new(0, 7, 1, -1));
    let southeast = align_diagonal_neighbor(center, s, e, se, DiagonalSpec::new(0, 0, -1, -1));

    let mut bit0 = 0_u64;
    let mut bit1 = 0_u64;
    let mut bit2 = 0_u64;
    let mut bit3 = 0_u64;

    for neighbors in [
        north, south, west, east, northwest, northeast, southwest, southeast,
    ] {
        let carry0 = bit0 & neighbors;
        bit0 ^= neighbors;
        let carry1 = bit1 & carry0;
        bit1 ^= carry0;
        let carry2 = bit2 & carry1;
        bit2 ^= carry1;
        bit3 ^= carry2;
    }

    let exactly_three = bit0 & bit1 & !bit2 & !bit3;
    let exactly_two = !bit0 & bit1 & !bit2 & !bit3;
    exactly_three | (center & exactly_two)
}

#[cfg(test)]
fn evolve_center_chunks_bitwise_batch(neighborhoods: &[ChunkNeighborhood]) -> Vec<u64> {
    debug_assert!(!neighborhoods.is_empty());
    debug_assert!(neighborhoods.len() <= 8);
    if neighborhoods.len() == 1 {
        return vec![evolve_center_chunk_bitwise(&neighborhoods[0])];
    }

    let chunks = build_chunk_row_batches(neighborhoods);
    let next = evolve_packed_chunk_rows(&chunks);
    next[..neighborhoods.len()].to_vec()
}

fn evolve_center_chunks_bitwise_batch_from_pending(
    pending: &[(i32, i32, u64, ChunkNeighborhood)],
) -> Vec<u64> {
    debug_assert!(!pending.is_empty());
    debug_assert!(pending.len() <= 8);
    if pending.len() == 1 {
        return vec![evolve_center_chunk_bitwise(&pending[0].3)];
    }

    let chunks = build_chunk_row_batches_from_pending(pending);
    let next = evolve_packed_chunk_rows(&chunks);
    next[..pending.len()].to_vec()
}

fn evolve_packed_chunk_rows(chunks: &[ChunkRowBatch; 9]) -> [u64; 8] {
    let center = pack_row_batch(&chunks[4]);
    let neighbors = packed_neighbor_boards(chunks);
    let (bit0, bit1, bit2, bit3) = accumulate_neighbor_bitplanes(&neighbors);
    let exactly_three = bit0 & bit1 & !bit2 & !bit3;
    let exactly_two = !bit0 & bit1 & !bit2 & !bit3;
    must_cast(exactly_three | (center & exactly_two))
}

fn packed_neighbor_boards(chunks: &[ChunkRowBatch; 9]) -> [u64x8; 8] {
    [
        pack_row_batch(&align_vertical_rows(&chunks[4], &chunks[1], 7, 1)),
        pack_row_batch(&align_vertical_rows(&chunks[4], &chunks[7], 0, -1)),
        pack_row_batch(&align_horizontal_rows(&chunks[4], &chunks[3], 7, 1)),
        pack_row_batch(&align_horizontal_rows(&chunks[4], &chunks[5], 0, -1)),
        pack_row_batch(&align_diagonal_rows(
            &chunks[4],
            &chunks[1],
            &chunks[3],
            &chunks[0],
            DiagonalSpec::new(7, 7, 1, 1),
        )),
        pack_row_batch(&align_diagonal_rows(
            &chunks[4],
            &chunks[1],
            &chunks[5],
            &chunks[2],
            DiagonalSpec::new(7, 0, -1, 1),
        )),
        pack_row_batch(&align_diagonal_rows(
            &chunks[4],
            &chunks[7],
            &chunks[3],
            &chunks[6],
            DiagonalSpec::new(0, 7, 1, -1),
        )),
        pack_row_batch(&align_diagonal_rows(
            &chunks[4],
            &chunks[7],
            &chunks[5],
            &chunks[8],
            DiagonalSpec::new(0, 0, -1, -1),
        )),
    ]
}

fn accumulate_neighbor_bitplanes(neighbors: &[u64x8; 8]) -> (u64x8, u64x8, u64x8, u64x8) {
    let mut bit0 = u64x8::ZERO;
    let mut bit1 = u64x8::ZERO;
    let mut bit2 = u64x8::ZERO;
    let mut bit3 = u64x8::ZERO;

    for &lanes in neighbors {
        let carry0 = bit0 & lanes;
        bit0 ^= lanes;
        let carry1 = bit1 & carry0;
        bit1 ^= carry0;
        let carry2 = bit2 & carry1;
        bit2 ^= carry1;
        bit3 ^= carry2;
    }

    (bit0, bit1, bit2, bit3)
}

// Batched row layout
#[cfg(test)]
fn build_chunk_row_batches(neighborhoods: &[ChunkNeighborhood]) -> [ChunkRowBatch; 9] {
    let mut chunks = [[[0_u16; 8]; 8]; 9];

    for (lane, neighborhood) in neighborhoods.iter().enumerate() {
        let first_batch = chunk_rows_batch_4([
            neighborhood.0[0],
            neighborhood.0[1],
            neighborhood.0[2],
            neighborhood.0[3],
        ]);
        let second_batch = chunk_rows_batch_4([
            neighborhood.0[4],
            neighborhood.0[5],
            neighborhood.0[6],
            neighborhood.0[7],
        ]);
        store_rows_batch_4(&mut chunks[0..4], lane, first_batch);
        store_rows_batch_4(&mut chunks[4..8], lane, second_batch);
        store_rows_for_lane(&mut chunks[8], lane, chunk_rows(neighborhood.0[8]));
    }

    chunks.map(must_cast)
}

fn build_chunk_row_batches_from_pending(
    pending: &[(i32, i32, u64, ChunkNeighborhood)],
) -> [ChunkRowBatch; 9] {
    let mut chunks = [[[0_u16; 8]; 8]; 9];

    for (lane, (_, _, _, neighborhood)) in pending.iter().enumerate() {
        let first_batch = chunk_rows_batch_4([
            neighborhood.0[0],
            neighborhood.0[1],
            neighborhood.0[2],
            neighborhood.0[3],
        ]);
        let second_batch = chunk_rows_batch_4([
            neighborhood.0[4],
            neighborhood.0[5],
            neighborhood.0[6],
            neighborhood.0[7],
        ]);
        store_rows_batch_4(&mut chunks[0..4], lane, first_batch);
        store_rows_batch_4(&mut chunks[4..8], lane, second_batch);
        store_rows_for_lane(&mut chunks[8], lane, chunk_rows(neighborhood.0[8]));
    }

    chunks.map(must_cast)
}

fn store_rows_for_lane(chunk: &mut [[u16; 8]; 8], lane: usize, rows: [u16; 8]) {
    chunk[0][lane] = rows[0];
    chunk[1][lane] = rows[1];
    chunk[2][lane] = rows[2];
    chunk[3][lane] = rows[3];
    chunk[4][lane] = rows[4];
    chunk[5][lane] = rows[5];
    chunk[6][lane] = rows[6];
    chunk[7][lane] = rows[7];
}

fn store_rows_batch_4(chunks: &mut [[[u16; 8]; 8]], lane: usize, rows_batch: [[u16; 8]; 4]) {
    let [first_rows, second_rows, third_rows, fourth_rows] = rows_batch;
    store_rows_for_lane(&mut chunks[0], lane, first_rows);
    store_rows_for_lane(&mut chunks[1], lane, second_rows);
    store_rows_for_lane(&mut chunks[2], lane, third_rows);
    store_rows_for_lane(&mut chunks[3], lane, fourth_rows);
}

// Batched row transforms
fn pack_row_batch(rows: &ChunkRowBatch) -> u64x8 {
    pack_row_block(rows[0], 0) | pack_row_block(rows[1], 32)
}

fn pack_row_block(block: u16x32, base_shift: u64) -> u64x8 {
    let [row0, row1, row2, row3]: [u16x8; 4] = must_cast(block & ROW_BLOCK_LOW_BYTE_MASK);
    pack_row_lanes(row0, base_shift)
        | pack_row_lanes(row1, base_shift + 8)
        | pack_row_lanes(row2, base_shift + 16)
        | pack_row_lanes(row3, base_shift + 24)
}

fn pack_row_lanes(lanes: u16x8, shift: u64) -> u64x8 {
    let narrowed: [u16; 8] = must_cast(lanes);
    let as_u64: u64x8 = must_cast(narrowed.map(u64::from));
    as_u64 << shift
}

fn align_vertical_rows(
    center: &ChunkRowBatch,
    edge: &ChunkRowBatch,
    edge_row: usize,
    shift_rows: i32,
) -> ChunkRowBatch {
    let edge_view: &ChunkRowView = must_cast_ref(edge);
    let mut shifted = *center;
    let shifted_view: &mut ChunkRowView = must_cast_mut(&mut shifted);
    match shift_rows {
        1 => {
            shifted_view.copy_within(0..7, 1);
            shifted_view[0] = edge_view[edge_row];
        }
        -1 => {
            shifted_view.copy_within(1..8, 0);
            shifted_view[7] = edge_view[edge_row];
        }
        _ => panic!("unsupported row shift: {shift_rows}"),
    }
    shifted
}

fn align_horizontal_rows(
    center: &ChunkRowBatch,
    edge: &ChunkRowBatch,
    edge_col: usize,
    shift_cols: i32,
) -> ChunkRowBatch {
    let edge_mask = edge_column_mask_batch(edge, edge_col, if shift_cols > 0 { 0 } else { 7 });
    if shift_cols > 0 {
        [
            (center[0] << (shift_cols as u16)) | edge_mask[0],
            (center[1] << (shift_cols as u16)) | edge_mask[1],
        ]
    } else {
        let amount = (-shift_cols) as u16;
        [
            (center[0] >> amount) | edge_mask[0],
            (center[1] >> amount) | edge_mask[1],
        ]
    }
}

// Scalar neighborhood alignment
fn align_vertical_neighbor(center: u64, edge_chunk: u64, edge_row: usize, shift_rows: i32) -> u64 {
    let [center_rows, edge_rows] = chunk_rows_batch_2([center, edge_chunk]);
    pack_rows(shift_rows_with_edge(
        center_rows,
        edge_rows[edge_row],
        shift_rows,
    ))
}

fn align_horizontal_neighbor(
    center: u64,
    edge_chunk: u64,
    edge_col: usize,
    shift_cols: i32,
) -> u64 {
    let [center_rows, edge_rows] = chunk_rows_batch_2([center, edge_chunk]);
    let center_row_lanes: u16x8 = must_cast(center_rows);
    let shifted = if shift_cols > 0 {
        center_row_lanes << (shift_cols as u16)
    } else {
        center_row_lanes >> ((-shift_cols) as u16)
    };
    let edge_mask = edge_column_mask_rows(
        must_cast(edge_rows),
        edge_col,
        if shift_cols > 0 { 0 } else { 7 },
    );
    pack_rows(shifted | edge_mask)
}

fn align_diagonal_neighbor(
    center: u64,
    vertical_chunk: u64,
    horizontal_chunk: u64,
    corner_chunk: u64,
    spec: DiagonalSpec,
) -> u64 {
    let [center_rows, vertical_rows] = chunk_rows_batch_2([center, vertical_chunk]);
    let [horizontal_rows, corner_rows] = chunk_rows_batch_2([horizontal_chunk, corner_chunk]);
    let source_row_lanes =
        shift_rows_with_edge(center_rows, vertical_rows[spec.edge_row], spec.shift_rows);
    let edge_source_rows =
        shift_rows_with_edge(horizontal_rows, corner_rows[spec.edge_row], spec.shift_rows);
    let shifted_source_rows = if spec.shift_cols > 0 {
        source_row_lanes << (spec.shift_cols as u16)
    } else {
        source_row_lanes >> ((-spec.shift_cols) as u16)
    };
    let edge_target = if spec.shift_cols > 0 { 0 } else { 7 };
    let edge_row_lanes = ((edge_source_rows >> (spec.edge_col as u16)) & u16x8::ONE) << edge_target;
    pack_rows(shifted_source_rows | edge_row_lanes)
}

// Row packing and view helpers
fn align_diagonal_rows(
    center: &ChunkRowBatch,
    vertical: &ChunkRowBatch,
    horizontal: &ChunkRowBatch,
    corner: &ChunkRowBatch,
    spec: DiagonalSpec,
) -> ChunkRowBatch {
    let source_rows = align_vertical_rows(center, vertical, spec.edge_row, spec.shift_rows);
    let edge_source_rows = align_vertical_rows(horizontal, corner, spec.edge_row, spec.shift_rows);
    let edge_target = if spec.shift_cols > 0 { 0 } else { 7 };
    let edge_rows = edge_column_mask_batch(&edge_source_rows, spec.edge_col, edge_target);

    if spec.shift_cols > 0 {
        [
            (source_rows[0] << (spec.shift_cols as u16)) | edge_rows[0],
            (source_rows[1] << (spec.shift_cols as u16)) | edge_rows[1],
        ]
    } else {
        let amount = (-spec.shift_cols) as u16;
        [
            (source_rows[0] >> amount) | edge_rows[0],
            (source_rows[1] >> amount) | edge_rows[1],
        ]
    }
}

fn shift_rows_with_edge(rows: [u16; 8], edge_fill: u16, shift_rows: i32) -> u16x8 {
    let row_bytes: i8x16 = must_cast(rows);
    let (edge_bytes, shifted_bytes): (i8x16, i8x16) = match shift_rows {
        1 => (
            must_cast([edge_fill, 0, 0, 0, 0, 0, 0, 0]),
            row_bytes.swizzle(SHIFT_ROWS_DOWN_BYTES),
        ),
        -1 => (
            must_cast([0, 0, 0, 0, 0, 0, 0, edge_fill]),
            row_bytes.swizzle(SHIFT_ROWS_UP_BYTES),
        ),
        _ => panic!("unsupported row shift: {shift_rows}"),
    };
    must_cast(shifted_bytes | edge_bytes)
}

fn chunk_rows(chunk: u64) -> [u16; 8] {
    chunk_rows_batch_2([chunk, 0])[0]
}

fn chunk_rows_batch_2(chunks: [u64; 2]) -> [[u16; 8]; 2] {
    let byte_lanes: u8x16 = must_cast(chunks);
    must_cast(u16x16::from(byte_lanes))
}

fn chunk_rows_batch_4(chunks: [u64; 4]) -> [[u16; 8]; 4] {
    let byte_lane_halves: [u8x16; 2] = must_cast(chunks);
    let widened_halves = [
        u16x16::from(byte_lane_halves[0]),
        u16x16::from(byte_lane_halves[1]),
    ];
    must_cast(widened_halves)
}

fn pack_rows(rows: u16x8) -> u64 {
    let narrowed: [u16; 8] = must_cast(rows & ROW_LOW_BYTE_MASK);
    let packed_bytes = narrowed.map(|row| row as u8);
    must_cast(packed_bytes)
}

fn edge_column_mask_rows(rows: u16x8, edge_col: usize, target_col: u16) -> u16x8 {
    ((rows >> (edge_col as u16)) & u16x8::ONE) << target_col
}

fn edge_column_mask_batch(
    chunk: &ChunkRowBatch,
    edge_col: usize,
    target_col: u16,
) -> ChunkRowBatch {
    [
        ((chunk[0] >> (edge_col as u16)) & u16x32::ONE) << target_col,
        ((chunk[1] >> (edge_col as u16)) & u16x32::ONE) << target_col,
    ]
}

// Changed-cell extraction
fn append_changed_cells(changed: &mut Vec<(i32, i32)>, cx: i32, cy: i32, diff_bits: u64) {
    if diff_bits == 0 {
        return;
    }

    let mut remaining = diff_bits;
    while remaining != 0 {
        let bit = remaining.trailing_zeros();
        let local_x = (bit % 8) as i32;
        let local_y = (bit / 8) as i32;
        changed.push((cx * CHUNK_SIZE + local_x, cy * CHUNK_SIZE + local_y));
        remaining &= remaining - 1;
    }
}

// Test-only reference kernels
#[cfg(test)]
fn evolve_center_chunk_naive(neighborhood: &ChunkNeighborhood) -> u64 {
    let mut next_bits = 0_u64;
    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let mut neighbors = 0_u8;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if naive_neighborhood_cell(neighborhood, lx + dx, ly + dy) {
                        neighbors += 1;
                    }
                }
            }

            let alive = naive_neighborhood_cell(neighborhood, lx, ly);
            if neighbors == 3 || (neighbors == 2 && alive) {
                let bit = (ly * CHUNK_SIZE + lx) as u32;
                next_bits |= 1_u64 << bit;
            }
        }
    }
    next_bits
}

#[cfg(test)]
fn naive_neighborhood_cell(neighborhood: &ChunkNeighborhood, x: i32, y: i32) -> bool {
    let chunk_x = if x < 0 {
        0
    } else if x >= CHUNK_SIZE {
        2
    } else {
        1
    };
    let chunk_y = if y < 0 {
        0
    } else if y >= CHUNK_SIZE {
        2
    } else {
        1
    };
    let local_x = x.rem_euclid(CHUNK_SIZE) as u32;
    let local_y = y.rem_euclid(CHUNK_SIZE) as u32;
    let bit = local_y * CHUNK_SIZE as u32 + local_x;
    (neighborhood.0[(chunk_y * 3 + chunk_x) as usize] & (1_u64 << bit)) != 0
}

#[cfg(test)]
#[path = "tests/life.rs"]
mod tests;
