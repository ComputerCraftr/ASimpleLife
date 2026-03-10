use std::collections::HashSet;

use crate::bitgrid::{BitGrid, CHUNK_SIZE};
use crate::memo::{ChunkNeighborhood, Memo};

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

pub fn step_grid_with_changes_and_memo(
    grid: &BitGrid,
    memo: &mut Memo,
) -> (BitGrid, Vec<(i32, i32)>) {
    if grid.is_empty() {
        return (BitGrid::new(), Vec::new());
    }

    let target_chunks = collect_target_chunks(grid);
    let mut next = BitGrid::new();
    let mut changed = Vec::new();

    for (cx, cy) in target_chunks {
        let neighborhood = build_neighborhood(grid, cx, cy);
        let next_bits = evolve_center_chunk(&neighborhood, memo);
        let current_bits = grid.chunk_bits(cx, cy);

        if next_bits != 0 {
            next.set_chunk_bits(cx, cy, next_bits);
        }

        append_changed_cells(&mut changed, cx, cy, current_bits ^ next_bits);
    }

    changed.sort_unstable();
    changed.dedup();
    (next, changed)
}

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

fn evolve_center_chunk(neighborhood: &ChunkNeighborhood, memo: &mut Memo) -> u64 {
    if let Some(cached) = memo.get_chunk_transition(neighborhood) {
        return cached;
    }

    let mut next_bits = 0_u64;
    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let mut neighbors = 0_u8;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if neighborhood_cell(neighborhood, lx + dx, ly + dy) {
                        neighbors += 1;
                    }
                }
            }

            let alive = neighborhood_cell(neighborhood, lx, ly);
            if neighbors == 3 || (neighbors == 2 && alive) {
                let bit = (ly * CHUNK_SIZE + lx) as u32;
                next_bits |= 1_u64 << bit;
            }
        }
    }

    memo.insert_chunk_transition(neighborhood.clone(), next_bits);
    next_bits
}

fn neighborhood_cell(neighborhood: &ChunkNeighborhood, x: i32, y: i32) -> bool {
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

fn append_changed_cells(changed: &mut Vec<(i32, i32)>, cx: i32, cy: i32, diff_bits: u64) {
    if diff_bits == 0 {
        return;
    }

    for bit in 0..64_u32 {
        if (diff_bits & (1_u64 << bit)) == 0 {
            continue;
        }
        let local_x = (bit % 8) as i32;
        let local_y = (bit / 8) as i32;
        changed.push((cx * CHUNK_SIZE + local_x, cy * CHUNK_SIZE + local_y));
    }
}
