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

    pub fn is_empty(&self) -> bool {
        self.population == 0
    }

    pub fn get(&self, x: Coord, y: Coord) -> bool {
        let (chunk, bit) = chunk_and_bit(x, y);
        self.chunks
            .get(&chunk)
            .map(|chunk| (chunk.bits & (1_u64 << bit)) != 0)
            .unwrap_or(false)
    }

    pub fn set(&mut self, x: Coord, y: Coord, alive: bool) {
        let (chunk, bit) = chunk_and_bit(x, y);
        let mask = 1_u64 << bit;
        let current = self
            .chunks
            .get(&chunk)
            .map(|chunk| (chunk.bits & mask) != 0)
            .unwrap_or(false);

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
        let mut iter = self.live_cells().into_iter();
        let (mut min_x, mut min_y) = iter.next()?;
        let mut max_x = min_x;
        let mut max_y = min_y;

        for (x, y) in iter {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        Some((min_x, min_y, max_x, max_y))
    }

    pub(crate) fn chunk_bits(&self, cx: Coord, cy: Coord) -> u64 {
        self.chunks
            .get(&(cx, cy))
            .map(|chunk| chunk.bits)
            .unwrap_or(0)
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
