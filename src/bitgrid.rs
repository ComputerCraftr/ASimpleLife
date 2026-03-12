use std::collections::HashMap;

pub type Coord = i64;
pub type Cell = (Coord, Coord);
pub type Bounds = (Coord, Coord, Coord, Coord);

pub const CHUNK_SIZE: Coord = 8;
const DEFAULT_CHUNK_CAPACITY: usize = 64;

#[repr(align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Chunk {
    bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitGrid {
    chunks: HashMap<Cell, Chunk>,
    population: usize,
}

impl BitGrid {
    pub fn empty() -> Self {
        Self {
            chunks: HashMap::new(),
            population: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_chunk_capacity(DEFAULT_CHUNK_CAPACITY)
    }

    pub fn with_chunk_capacity(chunk_capacity: usize) -> Self {
        Self {
            chunks: HashMap::with_capacity(chunk_capacity),
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
        if let Some(chunk) = self.chunks.get(&chunk) {
            (chunk.bits & (1_u64 << bit)) != 0
        } else {
            false
        }
    }

    pub fn set(&mut self, x: Coord, y: Coord, alive: bool) {
        let (chunk, bit) = chunk_and_bit(x, y);
        let mask = 1_u64 << bit;
        let current = if let Some(chunk) = self.chunks.get(&chunk) {
            (chunk.bits & mask) != 0
        } else {
            false
        };

        if current == alive {
            return;
        }

        if alive {
            let entry = self.chunks.entry(chunk).or_default();
            entry.bits |= mask;
            self.population += 1;
        } else if let Some(entry) = self.chunks.get_mut(&chunk) {
            entry.bits &= !mask;
            self.population -= 1;
            if entry.bits == 0 {
                self.chunks.remove(&chunk);
            }
        }
    }

    pub fn live_cells(&self) -> Vec<Cell> {
        let mut cells = Vec::with_capacity(self.population);
        for (&(cx, cy), chunk) in &self.chunks {
            for bit in 0..64_u32 {
                if (chunk.bits & (1_u64 << bit)) == 0 {
                    continue;
                }
                let local_x = (bit % 8) as Coord;
                let local_y = (bit / 8) as Coord;
                cells.push((cx * CHUNK_SIZE + local_x, cy * CHUNK_SIZE + local_y));
            }
        }
        cells
    }

    pub fn bounds(&self) -> Option<Bounds> {
        let mut found = false;
        let mut min_x = 0;
        let mut min_y = 0;
        let mut max_x = 0;
        let mut max_y = 0;

        for (&(cx, cy), chunk) in &self.chunks {
            let mut remaining = chunk.bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as Coord;
                let x = cx * CHUNK_SIZE + (bit % CHUNK_SIZE);
                let y = cy * CHUNK_SIZE + (bit / CHUNK_SIZE);
                if !found {
                    min_x = x;
                    min_y = y;
                    max_x = x;
                    max_y = y;
                    found = true;
                } else {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
                remaining &= remaining - 1;
            }
        }

        found.then_some((min_x, min_y, max_x, max_y))
    }

    pub(crate) fn chunk_bits(&self, cx: Coord, cy: Coord) -> u64 {
        if let Some(chunk) = self.chunks.get(&(cx, cy)) {
            chunk.bits
        } else {
            0
        }
    }

    pub(crate) fn chunk_coords(&self) -> Vec<Cell> {
        self.chunks.keys().copied().collect()
    }

    pub(crate) fn set_chunk_bits(&mut self, cx: Coord, cy: Coord, bits: u64) {
        let previous = self.chunk_bits(cx, cy);
        if previous == bits {
            return;
        }

        self.population -= previous.count_ones() as usize;
        if bits == 0 {
            self.chunks.remove(&(cx, cy));
            return;
        }

        self.population += bits.count_ones() as usize;
        self.chunks.insert((cx, cy), Chunk { bits });
    }

    pub(crate) fn from_chunk_bits_map(chunks: HashMap<Cell, u64>) -> Self {
        let population = chunks.values().map(|bits| bits.count_ones() as usize).sum();
        let chunks = chunks
            .into_iter()
            .filter_map(|(coord, bits)| (bits != 0).then_some((coord, Chunk { bits })))
            .collect();
        Self { chunks, population }
    }
}

impl Default for BitGrid {
    fn default() -> Self {
        Self::new()
    }
}

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
}
