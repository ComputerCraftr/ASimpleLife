use bytemuck::{must_cast, must_cast_ref};
use wide::{u16x8, u64x8};

use crate::flat_table::{FlatKey, FlatTable};
use crate::hashing::hash_chunk_coord_key;
use crate::simd_layout::{
    AlignedU64Value, SIMD_BATCH_LANES, compact_nonzero_u8_lanes,
    widen_u64_pair_to_aligned_u16_rows,
};

pub type Coord = i64;
pub type Cell = (Coord, Coord);
pub type Bounds = (Coord, Coord, Coord, Coord);

pub const CHUNK_SIZE: Coord = 8;
const DEFAULT_CHUNK_CAPACITY: usize = 64;
const ROW_BYTE_MASK_VEC: u64x8 = u64x8::splat(0xFF);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Chunk {
    bits: AlignedU64Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChunkCoordKey {
    cx: Coord,
    cy: Coord,
}

impl ChunkCoordKey {
    const fn new(cx: Coord, cy: Coord) -> Self {
        Self { cx, cy }
    }
}

impl FlatKey for ChunkCoordKey {
    fn fingerprint(&self) -> u64 {
        hash_chunk_coord_key(self.cx, self.cy)
    }
}

#[derive(Clone, Debug)]
pub struct BitGrid {
    chunks: FlatTable<ChunkCoordKey, Chunk>,
    population: usize,
}

impl BitGrid {
    pub fn empty() -> Self {
        Self {
            chunks: FlatTable::new(),
            population: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_chunk_capacity(DEFAULT_CHUNK_CAPACITY)
    }

    pub fn with_chunk_capacity(chunk_capacity: usize) -> Self {
        Self {
            chunks: FlatTable::with_capacity(chunk_capacity),
            population: 0,
        }
    }

    pub fn from_cells(cells: &[Cell]) -> Self {
        let estimated_chunks = cells.len().div_ceil(64).max(DEFAULT_CHUNK_CAPACITY);
        let mut grid = Self::with_chunk_capacity(estimated_chunks);
        for &(x, y) in cells {
            grid.set(x, y, true);
        }
        grid
    }

    pub fn population(&self) -> usize {
        self.population
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.population == 0
    }

    pub fn get(&self, x: Coord, y: Coord) -> bool {
        let (chunk, bit) = chunk_and_bit(x, y);
        if let Some(chunk) = self.chunks.get(&ChunkCoordKey::new(chunk.0, chunk.1)) {
            (chunk.bits.0 & (1_u64 << bit)) != 0
        } else {
            false
        }
    }

    pub fn set(&mut self, x: Coord, y: Coord, alive: bool) {
        let (chunk, bit) = chunk_and_bit(x, y);
        let mask = 1_u64 << bit;
        let chunk_key = ChunkCoordKey::new(chunk.0, chunk.1);
        let current = if let Some(chunk) = self.chunks.get(&chunk_key) {
            (chunk.bits.0 & mask) != 0
        } else {
            false
        };

        if current == alive {
            return;
        }

        if alive {
            let bits = self.chunk_bits(chunk.0, chunk.1) | mask;
            self.chunks.insert(chunk_key, Chunk { bits: AlignedU64Value(bits) });
            self.population += 1;
        } else if let Some(chunk) = self.chunks.get(&chunk_key) {
            let next_bits = chunk.bits.0 & !mask;
            self.population -= 1;
            if next_bits == 0 {
                self.chunks.remove(&chunk_key);
            } else {
                self.chunks.insert(chunk_key, Chunk { bits: AlignedU64Value(next_bits) });
            }
        }
    }

    pub fn live_cells(&self) -> Vec<Cell> {
        let mut cells = Vec::with_capacity(self.population);
        let mut chunk_batch = [((0, 0), 0_u64); SIMD_BATCH_LANES];
        let mut batch_len = 0;
        for (coord, chunk) in self.chunks.iter() {
            chunk_batch[batch_len] = ((coord.cx, coord.cy), chunk.bits.0);
            batch_len += 1;
            if batch_len == SIMD_BATCH_LANES {
                append_chunk_batch_live_cells(&chunk_batch, batch_len, &mut cells);
                batch_len = 0;
            }
        }
        if batch_len != 0 {
            append_chunk_batch_live_cells(&chunk_batch, batch_len, &mut cells);
        }
        cells
    }

    pub fn translated(&self, dx: Coord, dy: Coord) -> Self {
        if self.is_empty() || (dx == 0 && dy == 0) {
            return self.clone();
        }

        let chunk_dx = dx.div_euclid(CHUNK_SIZE);
        let chunk_dy = dy.div_euclid(CHUNK_SIZE);
        let local_dx =
            u32::try_from(dx.rem_euclid(CHUNK_SIZE)).expect("grid translation x remainder");
        let local_dy =
            usize::try_from(dy.rem_euclid(CHUNK_SIZE)).expect("grid translation y remainder");
        let mut translated = Self::with_chunk_capacity(self.chunks.len().saturating_mul(4));

        for (coord, chunk) in self.chunks.iter() {
            let cx = coord.cx;
            let cy = coord.cy;
            let base_cx = cx
                .checked_add(chunk_dx)
                .expect("grid translation chunk x overflow");
            let base_cy = cy
                .checked_add(chunk_dy)
                .expect("grid translation chunk y overflow");
            let [left_top, right_top, left_bottom, right_bottom] =
                translated_chunk_parts(chunk.bits.0, local_dx, local_dy);
            if left_top != 0 {
                translated.accumulate_chunk_bits(base_cx, base_cy, left_top);
            }
            if right_top != 0 {
                let target_cx = base_cx
                    .checked_add(1)
                    .expect("grid translation chunk x overflow");
                translated.accumulate_chunk_bits(target_cx, base_cy, right_top);
            }
            if left_bottom != 0 {
                let target_cy = base_cy
                    .checked_add(1)
                    .expect("grid translation chunk y overflow");
                translated.accumulate_chunk_bits(base_cx, target_cy, left_bottom);
            }
            if right_bottom != 0 {
                let target_cx = base_cx
                    .checked_add(1)
                    .expect("grid translation chunk x overflow");
                let target_cy = base_cy
                    .checked_add(1)
                    .expect("grid translation chunk y overflow");
                translated.accumulate_chunk_bits(target_cx, target_cy, right_bottom);
            }
        }

        translated
    }

    pub fn bounds(&self) -> Option<Bounds> {
        let mut found = false;
        let mut min_x = 0;
        let mut min_y = 0;
        let mut max_x = 0;
        let mut max_y = 0;

        let mut chunk_batch = [((0, 0), 0_u64); SIMD_BATCH_LANES];
        let mut batch_len = 0;
        for (coord, chunk) in self.chunks.iter() {
            chunk_batch[batch_len] = ((coord.cx, coord.cy), chunk.bits.0);
            batch_len += 1;
            if batch_len == SIMD_BATCH_LANES {
                update_bounds_from_chunk_batch(
                    &chunk_batch,
                    batch_len,
                    &mut found,
                    &mut min_x,
                    &mut min_y,
                    &mut max_x,
                    &mut max_y,
                );
                batch_len = 0;
            }
        }
        if batch_len != 0 {
            update_bounds_from_chunk_batch(
                &chunk_batch,
                batch_len,
                &mut found,
                &mut min_x,
                &mut min_y,
                &mut max_x,
                &mut max_y,
            );
        }

        found.then_some((min_x, min_y, max_x, max_y))
    }

    pub(crate) fn chunk_bits(&self, cx: Coord, cy: Coord) -> u64 {
        if let Some(chunk) = self.chunks.get(&ChunkCoordKey::new(cx, cy)) {
            chunk.bits.0
        } else {
            0
        }
    }

    pub(crate) fn chunk_coords(&self) -> Vec<Cell> {
        self.chunks.iter().map(|(coord, _)| (coord.cx, coord.cy)).collect()
    }

    pub(crate) fn set_chunk_bits(&mut self, cx: Coord, cy: Coord, bits: u64) {
        let previous = self.chunk_bits(cx, cy);
        if previous == bits {
            return;
        }

        self.population -= previous.count_ones() as usize;
        if bits == 0 {
            self.chunks.remove(&ChunkCoordKey::new(cx, cy));
            return;
        }

        self.population += bits.count_ones() as usize;
        self.chunks.insert(ChunkCoordKey::new(cx, cy), Chunk { bits: AlignedU64Value(bits) });
    }

    pub(crate) fn from_chunk_bits_map(
        chunks: impl IntoIterator<Item = (Cell, u64)>,
    ) -> Self {
        let collected = chunks.into_iter().collect::<Vec<_>>();
        let mut grid = Self::with_chunk_capacity(collected.len());
        for ((cx, cy), bits) in collected {
            if bits != 0 {
                grid.population += bits.count_ones() as usize;
                grid.chunks
                    .insert(ChunkCoordKey::new(cx, cy), Chunk { bits: AlignedU64Value(bits) });
            }
        }
        grid
    }

    fn accumulate_chunk_bits(&mut self, cx: Coord, cy: Coord, bits: u64) {
        let merged = self.chunk_bits(cx, cy) | bits;
        self.set_chunk_bits(cx, cy, merged);
    }
}

impl Default for BitGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for BitGrid {
    fn eq(&self, other: &Self) -> bool {
        if self.population != other.population || self.chunk_count() != other.chunk_count() {
            return false;
        }
        self.chunks
            .iter()
            .all(|(coord, chunk)| other.chunk_bits(coord.cx, coord.cy) == chunk.bits.0)
    }
}

impl Eq for BitGrid {}

fn chunk_and_bit(x: Coord, y: Coord) -> (Cell, u32) {
    let cx = x.div_euclid(CHUNK_SIZE);
    let cy = y.div_euclid(CHUNK_SIZE);
    let lx = x.rem_euclid(CHUNK_SIZE);
    let ly = y.rem_euclid(CHUNK_SIZE);
    (
        (cx, cy),
        u32::try_from(ly * CHUNK_SIZE + lx).expect("chunk bit index exceeded u32"),
    )
}

pub(crate) fn append_live_bits_as_cells(
    cells: &mut Vec<Cell>,
    cx: Coord,
    cy: Coord,
    bits: u64,
) {
    let mut remaining = bits;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as Coord;
        let local_x = bit % CHUNK_SIZE;
        let local_y = bit / CHUNK_SIZE;
        cells.push((cx * CHUNK_SIZE + local_x, cy * CHUNK_SIZE + local_y));
        remaining &= remaining - 1;
    }
}

fn pack_chunk_bits_for_batch(
    chunks: &[((Coord, Coord), u64); SIMD_BATCH_LANES],
    active_lanes: usize,
) -> u64x8 {
    let mut bits = [0_u64; SIMD_BATCH_LANES];
    for lane in 0..active_lanes {
        bits[lane] = chunks[lane].1;
    }
    must_cast(bits)
}

fn append_chunk_batch_live_cells(
    chunks: &[((Coord, Coord), u64); SIMD_BATCH_LANES],
    active_lanes: usize,
    cells: &mut Vec<Cell>,
) {
    let packed_bits = pack_chunk_bits_for_batch(chunks, active_lanes);

    for row in 0..CHUNK_SIZE as u32 {
        let row_bytes: [u64; SIMD_BATCH_LANES] =
            must_cast((packed_bits >> (row * 8)) & ROW_BYTE_MASK_VEC);
        let (active_indices, active_rows, active_count) =
            compact_nonzero_u8_lanes(row_bytes, active_lanes);
        for index in 0..active_count {
            let lane = active_indices.0[index];
            let row_bits = active_rows.0[index];
            let (cx, cy) = chunks[lane].0;
            append_live_row_bits_as_cells(cells, cx, cy, row as Coord, row_bits);
        }
    }
}

fn update_bounds_from_chunk_batch(
    chunks: &[((Coord, Coord), u64); SIMD_BATCH_LANES],
    active_lanes: usize,
    found: &mut bool,
    min_x: &mut Coord,
    min_y: &mut Coord,
    max_x: &mut Coord,
    max_y: &mut Coord,
) {
    let packed_bits = pack_chunk_bits_for_batch(chunks, active_lanes);

    for row in 0..CHUNK_SIZE as u32 {
        let row_bytes: [u64; SIMD_BATCH_LANES] =
            must_cast((packed_bits >> (row * 8)) & ROW_BYTE_MASK_VEC);
        let (active_indices, active_rows, active_count) =
            compact_nonzero_u8_lanes(row_bytes, active_lanes);
        for index in 0..active_count {
            let lane = active_indices.0[index];
            let row_bits = active_rows.0[index];
            let (cx, cy) = chunks[lane].0;
            update_bounds_from_live_row_bits(
                found,
                min_x,
                min_y,
                max_x,
                max_y,
                cx,
                cy,
                row as Coord,
                row_bits,
            );
        }
    }
}

fn append_live_row_bits_as_cells(
    cells: &mut Vec<Cell>,
    cx: Coord,
    cy: Coord,
    row: Coord,
    row_bits: u8,
) {
    let mut remaining = row_bits;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as Coord;
        cells.push((cx * CHUNK_SIZE + bit, cy * CHUNK_SIZE + row));
        remaining &= remaining - 1;
    }
}

fn update_bounds_from_live_row_bits(
    found: &mut bool,
    min_x: &mut Coord,
    min_y: &mut Coord,
    max_x: &mut Coord,
    max_y: &mut Coord,
    cx: Coord,
    cy: Coord,
    row: Coord,
    row_bits: u8,
) {
    let y = cy * CHUNK_SIZE + row;
    let x0 = cx * CHUNK_SIZE + row_bits.trailing_zeros() as Coord;
    let x1 = cx * CHUNK_SIZE + (7 - row_bits.leading_zeros() as Coord);
    if !*found {
        *min_x = x0;
        *min_y = y;
        *max_x = x1;
        *max_y = y;
        *found = true;
        return;
    }
    *min_x = (*min_x).min(x0);
    *min_y = (*min_y).min(y);
    *max_x = (*max_x).max(x1);
    *max_y = (*max_y).max(y);
}

fn translated_chunk_parts(bits: u64, local_dx: u32, local_dy: usize) -> [u64; 4] {
    let widened_rows = widen_u64_pair_to_aligned_u16_rows([bits, 0]);
    let widened_view: &[u16x8; 2] = must_cast_ref(&widened_rows);
    let shifted_rows: [u16; 8] = must_cast(widened_view[0] << local_dx);

    let mut left_top = 0_u64;
    let mut right_top = 0_u64;
    let mut left_bottom = 0_u64;
    let mut right_bottom = 0_u64;

    macro_rules! store_row {
        ($row:expr) => {{
            let target_row = $row + local_dy;
            let low_bits = u64::from(shifted_rows[$row] & 0x00FF);
            let high_bits = u64::from(shifted_rows[$row] >> 8);
            if target_row < 8 {
                let shift = u32::try_from(target_row * 8).expect("grid row shift exceeded u32");
                left_top |= low_bits << shift;
                right_top |= high_bits << shift;
            } else {
                let shift = u32::try_from((target_row - 8) * 8)
                    .expect("grid row shift exceeded u32");
                left_bottom |= low_bits << shift;
                right_bottom |= high_bits << shift;
            }
        }};
    }

    store_row!(0);
    store_row!(1);
    store_row!(2);
    store_row!(3);
    store_row!(4);
    store_row!(5);
    store_row!(6);
    store_row!(7);

    [left_top, right_top, left_bottom, right_bottom]
}

#[cfg(test)]
mod tests {
    use super::{BitGrid, Cell};
    use std::collections::HashMap;

    #[test]
    fn from_chunk_bits_map_matches_cell_construction() {
        let cells: Vec<Cell> = vec![
            (-9, -1),
            (-8, -1),
            (0, 0),
            (1, 0),
            (7, 7),
            (8, 8),
            (15, 15),
            (16, 0),
        ];
        let expected = BitGrid::from_cells(&cells);

        let mut chunks = HashMap::new();
        for &(x, y) in &cells {
            let cx = x.div_euclid(super::CHUNK_SIZE);
            let cy = y.div_euclid(super::CHUNK_SIZE);
            let lx = x.rem_euclid(super::CHUNK_SIZE);
            let ly = y.rem_euclid(super::CHUNK_SIZE);
            let bit = u32::try_from(ly * super::CHUNK_SIZE + lx).unwrap();
            *chunks.entry((cx, cy)).or_insert(0_u64) |= 1_u64 << bit;
        }

        let actual = BitGrid::from_chunk_bits_map(chunks);
        assert_eq!(actual, expected);
    }

    #[test]
    fn from_chunk_bits_map_drops_zero_chunks() {
        let mut chunks = HashMap::new();
        chunks.insert((0, 0), 0_u64);
        chunks.insert((1, 1), 1_u64);

        let grid = BitGrid::from_chunk_bits_map(chunks);
        assert_eq!(grid.chunk_count(), 1);
        assert_eq!(grid.population(), 1);
        assert!(grid.get(8, 8));
    }

    #[test]
    fn translated_matches_cellwise_reconstruction_for_arbitrary_offsets() {
        let cells: Vec<Cell> = vec![
            (-9, -1),
            (-8, -1),
            (-1, 7),
            (0, 0),
            (1, 0),
            (7, 7),
            (8, 8),
            (15, 15),
            (16, 0),
        ];
        let grid = BitGrid::from_cells(&cells);

        for (dx, dy) in [(-9, -10), (-1, 0), (0, 0), (3, -5), (8, 8), (17, 2)] {
            let expected_cells = cells
                .iter()
                .map(|&(x, y)| (x + dx, y + dy))
                .collect::<Vec<_>>();
            assert_eq!(BitGrid::from_cells(&expected_cells), grid.translated(dx, dy));
        }
    }

    #[test]
    fn live_cells_and_bounds_match_sparse_input_across_multiple_chunks() {
        let cells: Vec<Cell> = vec![
            (-17, -9),
            (-8, -8),
            (-1, -1),
            (0, 0),
            (7, 7),
            (8, 8),
            (15, 2),
            (24, 31),
        ];
        let grid = BitGrid::from_cells(&cells);

        let mut expected = cells.clone();
        expected.sort_unstable();
        let mut actual = grid.live_cells();
        actual.sort_unstable();

        assert_eq!(actual, expected);
        assert_eq!(grid.bounds(), Some((-17, -9, 24, 31)));
    }
}
