use crate::RequiredExt;
use bytemuck::{must_cast, must_cast_ref};
use wide::{u16x8, u64x8};

use crate::hashing::hash_chunk_coord_key;
use crate::probe_table::{ProbeKey, ProbeMode, ProbeReserveError, ProbeTable};
use crate::simd_layout::{
    AlignedU64Value, SIMD_BATCH_LANES, compact_nonzero_u8_lanes, widen_u64_pair_to_aligned_u16_rows,
};

pub type Coord = i64;
pub type Cell = (Coord, Coord);
pub type Bounds = (Coord, Coord, Coord, Coord);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridTranslationError {
    CoordinateOverflow,
    Allocation,
}

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

impl ProbeKey for ChunkCoordKey {
    fn fingerprint(&self) -> u64 {
        hash_chunk_coord_key(self.cx, self.cy)
    }
}

#[derive(Clone, Debug)]
pub struct BitGrid {
    chunks: ProbeTable<ChunkCoordKey, Chunk>,
    population: usize,
}

impl BitGrid {
    pub fn empty() -> Self {
        Self {
            chunks: ProbeTable::new(ProbeMode::Mutable),
            population: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_chunk_capacity(DEFAULT_CHUNK_CAPACITY)
    }

    pub fn with_chunk_capacity(chunk_capacity: usize) -> Self {
        Self {
            chunks: ProbeTable::with_capacity(ProbeMode::Mutable, chunk_capacity),
            population: 0,
        }
    }

    pub(crate) fn try_with_chunk_capacity(
        chunk_capacity: usize,
    ) -> Result<Self, ProbeReserveError> {
        Ok(Self {
            chunks: ProbeTable::try_with_capacity(ProbeMode::Mutable, chunk_capacity)?,
            population: 0,
        })
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

    pub(crate) fn occupied_chunks(&self) -> impl Iterator<Item = ((Coord, Coord), u64)> + '_ {
        self.chunks
            .iter()
            .map(|(coord, chunk)| ((coord.cx, coord.cy), chunk.bits.0))
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.chunks.allocated_bytes()
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
            self.chunks.insert(
                chunk_key,
                Chunk {
                    bits: AlignedU64Value(bits),
                },
            );
            self.population += 1;
        } else if let Some(chunk) = self.chunks.get(&chunk_key) {
            let next_bits = chunk.bits.0 & !mask;
            self.population -= 1;
            if next_bits == 0 {
                self.chunks.remove(&chunk_key);
            } else {
                self.chunks.insert(
                    chunk_key,
                    Chunk {
                        bits: AlignedU64Value(next_bits),
                    },
                );
            }
        }
    }

    pub fn live_cells(&self) -> Vec<Cell> {
        let mut cells = Vec::with_capacity(self.population);
        let mut chunk_batch = [((0, 0), 0_u64); SIMD_BATCH_LANES];
        let mut batch_len = 0;
        for chunk in self.occupied_chunks() {
            chunk_batch[batch_len] = chunk;
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
            u32::try_from(dx.rem_euclid(CHUNK_SIZE)).or_invariant("grid translation x remainder");
        let local_dy =
            usize::try_from(dy.rem_euclid(CHUNK_SIZE)).or_invariant("grid translation y remainder");
        let mut translated = Self::with_chunk_capacity(self.chunks.len().saturating_mul(4));

        for (coord, chunk) in self.chunks.iter() {
            let cx = coord.cx;
            let cy = coord.cy;
            let base_cx = cx
                .checked_add(chunk_dx)
                .or_invariant("grid translation chunk x overflow");
            let base_cy = cy
                .checked_add(chunk_dy)
                .or_invariant("grid translation chunk y overflow");
            let [left_top, right_top, left_bottom, right_bottom] =
                translated_chunk_parts(chunk.bits.0, local_dx, local_dy);
            if left_top != 0 {
                translated.accumulate_chunk_bits(base_cx, base_cy, left_top);
            }
            if right_top != 0 {
                let target_cx = base_cx
                    .checked_add(1)
                    .or_invariant("grid translation chunk x overflow");
                translated.accumulate_chunk_bits(target_cx, base_cy, right_top);
            }
            if left_bottom != 0 {
                let target_cy = base_cy
                    .checked_add(1)
                    .or_invariant("grid translation chunk y overflow");
                translated.accumulate_chunk_bits(base_cx, target_cy, left_bottom);
            }
            if right_bottom != 0 {
                let target_cx = base_cx
                    .checked_add(1)
                    .or_invariant("grid translation chunk x overflow");
                let target_cy = base_cy
                    .checked_add(1)
                    .or_invariant("grid translation chunk y overflow");
                translated.accumulate_chunk_bits(target_cx, target_cy, right_bottom);
            }
        }

        translated
    }

    pub fn try_translated(&self, dx: i128, dy: i128) -> Result<Self, GridTranslationError> {
        if let Some((min_x, min_y, max_x, max_y)) = self.bounds() {
            for coordinate in [min_x, max_x] {
                let translated = i128::from(coordinate)
                    .checked_add(dx)
                    .ok_or(GridTranslationError::CoordinateOverflow)?;
                Coord::try_from(translated)
                    .map_err(|_| GridTranslationError::CoordinateOverflow)?;
            }
            for coordinate in [min_y, max_y] {
                let translated = i128::from(coordinate)
                    .checked_add(dy)
                    .ok_or(GridTranslationError::CoordinateOverflow)?;
                Coord::try_from(translated)
                    .map_err(|_| GridTranslationError::CoordinateOverflow)?;
            }
        }
        let chunk_size = i128::from(CHUNK_SIZE);
        let chunk_dx = dx.div_euclid(chunk_size);
        let chunk_dy = dy.div_euclid(chunk_size);
        let local_dx = u32::try_from(dx.rem_euclid(chunk_size))
            .map_err(|_| GridTranslationError::CoordinateOverflow)?;
        let local_dy = usize::try_from(dy.rem_euclid(chunk_size))
            .map_err(|_| GridTranslationError::CoordinateOverflow)?;
        let capacity = self
            .chunks
            .len()
            .checked_mul(4)
            .ok_or(GridTranslationError::Allocation)?
            .max(1);
        let mut translated = Self::try_with_chunk_capacity(capacity)
            .map_err(|_| GridTranslationError::Allocation)?;

        for (coord, chunk) in self.chunks.iter() {
            let base_x = i128::from(coord.cx)
                .checked_add(chunk_dx)
                .ok_or(GridTranslationError::CoordinateOverflow)?;
            let base_y = i128::from(coord.cy)
                .checked_add(chunk_dy)
                .ok_or(GridTranslationError::CoordinateOverflow)?;
            let [left_top, right_top, left_bottom, right_bottom] =
                translated_chunk_parts(chunk.bits.0, local_dx, local_dy);
            translated.try_accumulate_wide_chunk(base_x, base_y, left_top)?;
            translated.try_accumulate_wide_chunk(
                base_x
                    .checked_add(1)
                    .ok_or(GridTranslationError::CoordinateOverflow)?,
                base_y,
                right_top,
            )?;
            translated.try_accumulate_wide_chunk(
                base_x,
                base_y
                    .checked_add(1)
                    .ok_or(GridTranslationError::CoordinateOverflow)?,
                left_bottom,
            )?;
            translated.try_accumulate_wide_chunk(
                base_x
                    .checked_add(1)
                    .ok_or(GridTranslationError::CoordinateOverflow)?,
                base_y
                    .checked_add(1)
                    .ok_or(GridTranslationError::CoordinateOverflow)?,
                right_bottom,
            )?;
        }

        Ok(translated)
    }

    pub fn bounds(&self) -> Option<Bounds> {
        let mut bounds = None;

        let mut chunk_batch = [((0, 0), 0_u64); SIMD_BATCH_LANES];
        let mut batch_len = 0;
        for (coord, chunk) in self.chunks.iter() {
            chunk_batch[batch_len] = ((coord.cx, coord.cy), chunk.bits.0);
            batch_len += 1;
            if batch_len == SIMD_BATCH_LANES {
                update_bounds_from_chunk_batch(&chunk_batch, batch_len, &mut bounds);
                batch_len = 0;
            }
        }
        if batch_len != 0 {
            update_bounds_from_chunk_batch(&chunk_batch, batch_len, &mut bounds);
        }
        bounds
    }

    pub(crate) fn chunk_bits(&self, cx: Coord, cy: Coord) -> u64 {
        if let Some(chunk) = self.chunks.get(&ChunkCoordKey::new(cx, cy)) {
            chunk.bits.0
        } else {
            0
        }
    }

    pub(crate) fn for_each_chunk_coord(&self, mut visit: impl FnMut(Coord, Coord)) {
        for (coord, _) in self.chunks.iter() {
            visit(coord.cx, coord.cy);
        }
    }

    pub(crate) fn set_chunk_bits(&mut self, cx: Coord, cy: Coord, bits: u64) {
        let previous = self.chunk_bits(cx, cy);
        if previous == bits {
            return;
        }

        self.population -=
            usize::try_from(previous.count_ones()).or_invariant("chunk population exceeded usize");
        if bits == 0 {
            self.chunks.remove(&ChunkCoordKey::new(cx, cy));
            return;
        }

        self.population +=
            usize::try_from(bits.count_ones()).or_invariant("chunk population exceeded usize");
        self.chunks.insert(
            ChunkCoordKey::new(cx, cy),
            Chunk {
                bits: AlignedU64Value(bits),
            },
        );
    }

    pub(crate) fn try_set_chunk_bits(
        &mut self,
        cx: Coord,
        cy: Coord,
        bits: u64,
    ) -> Result<(), ProbeReserveError> {
        let previous = self.chunk_bits(cx, cy);
        if previous == bits {
            return Ok(());
        }
        self.population -=
            usize::try_from(previous.count_ones()).or_invariant("chunk population exceeded usize");
        if bits == 0 {
            self.chunks.remove(&ChunkCoordKey::new(cx, cy));
        } else {
            self.population +=
                usize::try_from(bits.count_ones()).or_invariant("chunk population exceeded usize");
            self.chunks.try_insert(
                ChunkCoordKey::new(cx, cy),
                Chunk {
                    bits: AlignedU64Value(bits),
                },
            )?;
        }
        Ok(())
    }

    fn accumulate_chunk_bits(&mut self, cx: Coord, cy: Coord, bits: u64) {
        let merged = self.chunk_bits(cx, cy) | bits;
        self.set_chunk_bits(cx, cy, merged);
    }

    fn try_accumulate_wide_chunk(
        &mut self,
        cx: i128,
        cy: i128,
        bits: u64,
    ) -> Result<(), GridTranslationError> {
        if bits == 0 {
            return Ok(());
        }
        let cx = Coord::try_from(cx).map_err(|_| GridTranslationError::CoordinateOverflow)?;
        let cy = Coord::try_from(cy).map_err(|_| GridTranslationError::CoordinateOverflow)?;
        let merged = self.chunk_bits(cx, cy) | bits;
        self.try_set_chunk_bits(cx, cy, merged)
            .map_err(|_| GridTranslationError::Allocation)
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
        u32::try_from(ly * CHUNK_SIZE + lx).or_invariant("chunk bit index exceeded u32"),
    )
}

pub(crate) fn append_live_bits_as_cells(cells: &mut Vec<Cell>, cx: Coord, cy: Coord, bits: u64) {
    if bits == 0 {
        return;
    }
    let base_x = cx * CHUNK_SIZE;
    let base_y = cy * CHUNK_SIZE;
    let mut remaining = bits;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as Coord;
        let local_x = bit % CHUNK_SIZE;
        let local_y = bit / CHUNK_SIZE;
        cells.push((base_x + local_x, base_y + local_y));
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

    for row in 0..u32::try_from(CHUNK_SIZE).or_invariant("chunk size exceeded u32") {
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
    bounds: &mut Option<Bounds>,
) {
    let packed_bits = pack_chunk_bits_for_batch(chunks, active_lanes);

    for row in 0..u32::try_from(CHUNK_SIZE).or_invariant("chunk size exceeded u32") {
        let row_bytes: [u64; SIMD_BATCH_LANES] =
            must_cast((packed_bits >> (row * 8)) & ROW_BYTE_MASK_VEC);
        let (active_indices, active_rows, active_count) =
            compact_nonzero_u8_lanes(row_bytes, active_lanes);
        for index in 0..active_count {
            let lane = active_indices.0[index];
            let row_bits = active_rows.0[index];
            let (cx, cy) = chunks[lane].0;
            update_bounds_from_live_row_bits(bounds, cx, cy, row as Coord, row_bits);
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
    if row_bits == 0 {
        return;
    }
    let base_x = cx * CHUNK_SIZE;
    let y = cy * CHUNK_SIZE + row;
    let mut remaining = row_bits;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as Coord;
        cells.push((base_x + bit, y));
        remaining &= remaining - 1;
    }
}

fn update_bounds_from_live_row_bits(
    bounds: &mut Option<Bounds>,
    cx: Coord,
    cy: Coord,
    row: Coord,
    row_bits: u8,
) {
    let base_x = cx * CHUNK_SIZE;
    let y = cy * CHUNK_SIZE + row;
    let x0 = base_x + row_bits.trailing_zeros() as Coord;
    let x1 = base_x + (7 - row_bits.leading_zeros() as Coord);
    if let Some((min_x, min_y, max_x, max_y)) = bounds {
        *min_x = (*min_x).min(x0);
        *min_y = (*min_y).min(y);
        *max_x = (*max_x).max(x1);
        *max_y = (*max_y).max(y);
    } else {
        *bounds = Some((x0, y, x1, y));
    }
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
                let shift =
                    u32::try_from(target_row * 8).or_invariant("grid row shift exceeded u32");
                left_top |= low_bits << shift;
                right_top |= high_bits << shift;
            } else {
                let shift =
                    u32::try_from((target_row - 8) * 8).or_invariant("grid row shift exceeded u32");
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
    use super::{BitGrid, Cell, Coord, append_live_bits_as_cells, append_live_row_bits_as_cells};
    use crate::RequiredExt;
    use std::collections::HashMap;

    #[test]
    fn fallible_chunk_construction_matches_cell_construction() {
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
            let bit = u32::try_from(ly * super::CHUNK_SIZE + lx).or_invariant("required value");
            *chunks.entry((cx, cy)).or_insert(0_u64) |= 1_u64 << bit;
        }

        let mut actual = BitGrid::try_with_chunk_capacity(chunks.len())
            .or_invariant("test chunk table should allocate");
        for ((cx, cy), bits) in chunks {
            actual
                .try_set_chunk_bits(cx, cy, bits)
                .or_invariant("test chunk insertion should allocate");
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn fallible_chunk_construction_drops_zero_chunks() {
        let mut grid =
            BitGrid::try_with_chunk_capacity(2).or_invariant("test chunk table should allocate");
        grid.try_set_chunk_bits(0, 0, 0)
            .or_invariant("zero chunk should not allocate");
        grid.try_set_chunk_bits(1, 1, 1)
            .or_invariant("test chunk insertion should allocate");
        assert_eq!(grid.chunk_count(), 1);
        assert_eq!(grid.population(), 1);
        assert!(grid.get(8, 8));
    }

    #[test]
    fn empty_bit_expansion_does_not_evaluate_unused_extreme_chunk_origins() {
        let mut cells = vec![(2, 3)];
        for coordinate in [Coord::MIN, Coord::MAX] {
            append_live_bits_as_cells(&mut cells, coordinate, coordinate, 0);
            append_live_row_bits_as_cells(&mut cells, coordinate, coordinate, coordinate, 0);
        }
        assert_eq!(
            cells,
            [(2, 3)],
            "empty chunks and rows must leave output unchanged without origin arithmetic"
        );
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
            assert_eq!(
                BitGrid::from_cells(&expected_cells),
                grid.translated(dx, dy)
            );
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
