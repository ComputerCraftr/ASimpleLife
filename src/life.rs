use std::collections::HashSet;

use crate::RequiredExt;
use bytemuck::{must_cast, must_cast_mut, must_cast_ref};
use wide::{i8x16, u16x8, u16x32, u64x8};

use crate::bitgrid::{BitGrid, Cell, Coord, append_live_bits_as_cells};
use crate::memo::{ChunkNeighborhood, ChunkTransitionMemoIntent, Memo};
use crate::simd_layout::{
    AlignedU16ChunkRowBatches9, SIMD_BATCH_LANES, widen_u64_pair_to_u16_rows,
    widen_u64_quad_to_u16_rows,
};

const ROW_LOW_BYTE_MASK: u16x8 = u16x8::splat(0x00FF);
const ROW_BLOCK_LOW_BYTE_MASK: u16x32 = u16x32::splat(0x00FF);
const SHIFT_ROWS_DOWN_BYTES: i8x16 =
    i8x16::new([-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
const SHIFT_ROWS_UP_BYTES: i8x16 =
    i8x16::new([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1]);
const CHUNK_NEIGHBORHOOD_OFFSETS_3X3: [(Coord, Coord); 9] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const EMPTY_CHUNK_NEIGHBORHOOD: ChunkNeighborhood = ChunkNeighborhood([0; 9]);
const EMPTY_MEMO_INTENT: ChunkTransitionMemoIntent = ChunkTransitionMemoIntent {
    canonical: EMPTY_CHUNK_NEIGHBORHOOD,
    symmetry: crate::symmetry::D4Symmetry::Identity,
};

struct DiagonalSpec {
    edge_row: usize,
    edge_col: usize,
    shift_cols: i32,
    shift_rows: i32,
}

#[derive(Clone, Copy, Debug)]
struct PendingChunkBatch {
    len: usize,
    cx: [Coord; SIMD_BATCH_LANES],
    cy: [Coord; SIMD_BATCH_LANES],
    current_bits: [u64; SIMD_BATCH_LANES],
    neighborhoods: [ChunkNeighborhood; SIMD_BATCH_LANES],
    memo_intents: [ChunkTransitionMemoIntent; SIMD_BATCH_LANES],
}

impl PendingChunkBatch {
    fn new() -> Self {
        Self {
            len: 0,
            cx: [0; SIMD_BATCH_LANES],
            cy: [0; SIMD_BATCH_LANES],
            current_bits: [0; SIMD_BATCH_LANES],
            neighborhoods: [EMPTY_CHUNK_NEIGHBORHOOD; SIMD_BATCH_LANES],
            memo_intents: [EMPTY_MEMO_INTENT; SIMD_BATCH_LANES],
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn is_full(&self) -> bool {
        self.len == SIMD_BATCH_LANES
    }

    fn push(
        &mut self,
        cx: Coord,
        cy: Coord,
        current_bits: u64,
        neighborhood: ChunkNeighborhood,
        memo_intent: ChunkTransitionMemoIntent,
    ) {
        let lane = self.len;
        self.cx[lane] = cx;
        self.cy[lane] = cy;
        self.current_bits[lane] = current_bits;
        self.neighborhoods[lane] = neighborhood;
        self.memo_intents[lane] = memo_intent;
        self.len += 1;
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

#[derive(Clone, Debug, Default)]
struct CoordinateFrontier {
    seen: HashSet<Cell>,
    targets: Vec<Cell>,
}

impl CoordinateFrontier {
    fn rebuild(&mut self, grid: &BitGrid) {
        self.seen.clear();
        self.targets.clear();
        let additional = grid.chunk_count().saturating_mul(9);
        self.seen.reserve(additional);
        self.targets.reserve(additional);

        grid.for_each_chunk_coord(|cx, cy| {
            for (dx, dy) in CHUNK_NEIGHBORHOOD_OFFSETS_3X3 {
                let target = (cx + dx, cy + dy);
                if self.seen.insert(target) {
                    self.targets.push(target);
                }
            }
        });
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CellStepWorkspace {
    frontier: CoordinateFrontier,
    neighborhoods: [ChunkNeighborhood; SIMD_BATCH_LANES],
    pending: PendingChunkBatch,
    chunk_rows: AlignedU16ChunkRowBatches9,
}

impl Default for CellStepWorkspace {
    fn default() -> Self {
        Self {
            frontier: CoordinateFrontier::default(),
            neighborhoods: [EMPTY_CHUNK_NEIGHBORHOOD; SIMD_BATCH_LANES],
            pending: PendingChunkBatch::new(),
            chunk_rows: AlignedU16ChunkRowBatches9::default(),
        }
    }
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkDiff {
    pub cx: Coord,
    pub cy: Coord,
    pub diff_bits: u64,
}

#[derive(Clone, Debug)]
pub struct GameOfLife {
    grid: BitGrid,
    generation: u64,
    memo: Memo,
    workspace: CellStepWorkspace,
}

impl GameOfLife {
    pub fn new(grid: BitGrid) -> Self {
        Self::new_with_generation(grid, 0)
    }

    pub fn new_with_generation(grid: BitGrid, generation: u64) -> Self {
        Self {
            grid,
            generation,
            memo: Memo::default(),
            workspace: CellStepWorkspace::default(),
        }
    }

    pub fn grid(&self) -> &BitGrid {
        &self.grid
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn step(&mut self) {
        self.grid =
            step_grid_state_only_with_workspace(&self.grid, &mut self.memo, &mut self.workspace);
        self.memo.maybe_collect_transition_caches();
        self.generation += 1;
    }

    pub fn step_with_changes(&mut self) -> Vec<Cell> {
        let (next, chunk_changes) = step_grid_with_chunk_changes_and_workspace(
            &self.grid,
            &mut self.memo,
            &mut self.workspace,
        );
        self.memo.maybe_collect_transition_caches();
        self.grid = next;
        self.generation += 1;
        expand_chunk_diffs_to_cells(&chunk_changes)
    }

    pub fn step_with_chunk_changes(&mut self) -> Vec<ChunkDiff> {
        let (next, changed) = step_grid_with_chunk_changes_and_workspace(
            &self.grid,
            &mut self.memo,
            &mut self.workspace,
        );
        self.memo.maybe_collect_transition_caches();
        self.grid = next;
        self.generation += 1;
        changed
    }
}

#[cfg(test)]
pub fn step_grid(grid: &BitGrid) -> BitGrid {
    let mut memo = Memo::default();
    step_grid_with_chunk_changes_and_memo(grid, &mut memo).0
}

// Stepping pipeline
pub fn step_grid_with_changes_and_memo(grid: &BitGrid, memo: &mut Memo) -> (BitGrid, Vec<Cell>) {
    let (next, chunk_changes) = step_grid_with_chunk_changes_and_memo(grid, memo);
    (next, expand_chunk_diffs_to_cells(&chunk_changes))
}

pub fn step_grid_with_chunk_changes_and_memo(
    grid: &BitGrid,
    memo: &mut Memo,
) -> (BitGrid, Vec<ChunkDiff>) {
    step_grid_with_chunk_changes_and_workspace(grid, memo, &mut CellStepWorkspace::default())
}

pub(crate) fn step_grid_state_only_with_workspace(
    grid: &BitGrid,
    memo: &mut Memo,
    workspace: &mut CellStepWorkspace,
) -> BitGrid {
    step_grid_with_workspace(grid, memo, workspace, &mut ChunkChangeSink::Ignore)
}

fn step_grid_with_chunk_changes_and_workspace(
    grid: &BitGrid,
    memo: &mut Memo,
    workspace: &mut CellStepWorkspace,
) -> (BitGrid, Vec<ChunkDiff>) {
    let mut changed = Vec::new();
    let next = step_grid_with_workspace(
        grid,
        memo,
        workspace,
        &mut ChunkChangeSink::Collect(&mut changed),
    );
    changed.sort_unstable_by_key(|diff| (diff.cx, diff.cy));
    (next, changed)
}

enum ChunkChangeSink<'a> {
    Ignore,
    Collect(&'a mut Vec<ChunkDiff>),
}

impl ChunkChangeSink<'_> {
    fn record(&mut self, cx: Coord, cy: Coord, current_bits: u64, next_bits: u64) {
        let Self::Collect(changed) = self else {
            return;
        };
        let diff_bits = current_bits ^ next_bits;
        if diff_bits != 0 {
            changed.push(ChunkDiff { cx, cy, diff_bits });
        }
    }
}

fn step_grid_with_workspace(
    grid: &BitGrid,
    memo: &mut Memo,
    workspace: &mut CellStepWorkspace,
    changes: &mut ChunkChangeSink<'_>,
) -> BitGrid {
    if grid.is_empty() {
        workspace.frontier.seen.clear();
        workspace.frontier.targets.clear();
        workspace.pending.clear();
        return BitGrid::new();
    }

    workspace.frontier.rebuild(grid);
    let mut next = BitGrid::with_chunk_capacity(workspace.frontier.targets.len());
    workspace.pending.clear();
    let mut target_index = 0;

    while target_index < workspace.frontier.targets.len() {
        let batch_end = (target_index + SIMD_BATCH_LANES).min(workspace.frontier.targets.len());
        let targets = &workspace.frontier.targets[target_index..batch_end];
        gather_neighborhoods_into(grid, targets, &mut workspace.neighborhoods);
        let probe = memo.canonicalize_and_probe_chunk_transitions_staged(
            &workspace.neighborhoods,
            batch_end - target_index,
        );
        for (offset, &(cx, cy)) in targets.iter().enumerate() {
            let current_bits = grid.chunk_bits(cx, cy);
            if let Some(next_bits) = probe.hits[offset] {
                apply_chunk_step(&mut next, changes, cx, cy, current_bits, next_bits);
            } else {
                workspace.pending.push(
                    cx,
                    cy,
                    current_bits,
                    workspace.neighborhoods[offset],
                    probe.miss_intents[offset].or_invariant(
                        "memo probe miss lanes must carry canonicalized insert intent",
                    ),
                );
                if workspace.pending.is_full() {
                    flush_pending_chunks(
                        &mut workspace.pending,
                        &mut workspace.chunk_rows,
                        memo,
                        &mut next,
                        changes,
                    );
                }
            }
        }
        target_index = batch_end;
    }
    flush_pending_chunks(
        &mut workspace.pending,
        &mut workspace.chunk_rows,
        memo,
        &mut next,
        changes,
    );
    next
}

fn flush_pending_chunks(
    pending: &mut PendingChunkBatch,
    chunk_rows: &mut AlignedU16ChunkRowBatches9,
    memo: &mut Memo,
    next: &mut BitGrid,
    changes: &mut ChunkChangeSink<'_>,
) {
    if pending.is_empty() {
        return;
    }

    let next_bits = evolve_center_chunks_bitwise_batch_from_pending(pending, chunk_rows);

    for lane in 0..pending.len {
        let cx = pending.cx[lane];
        let cy = pending.cy[lane];
        let current_bits = pending.current_bits[lane];
        let next_bits = next_bits[lane];
        memo.insert_chunk_transition_from_intent(pending.memo_intents[lane], next_bits);
        apply_chunk_step(next, changes, cx, cy, current_bits, next_bits);
    }
    pending.clear();
}

// Neighborhood collection
fn gather_neighborhoods_into(
    grid: &BitGrid,
    targets: &[Cell],
    neighborhoods: &mut [ChunkNeighborhood; SIMD_BATCH_LANES],
) {
    for (lane, &(cx, cy)) in targets.iter().enumerate() {
        for (word, (dx, dy)) in CHUNK_NEIGHBORHOOD_OFFSETS_3X3.into_iter().enumerate() {
            neighborhoods[lane].0[word] = grid.chunk_bits(cx + dx, cy + dy);
        }
    }
}

#[cfg(test)]
fn build_neighborhood(grid: &BitGrid, cx: Coord, cy: Coord) -> ChunkNeighborhood {
    ChunkNeighborhood([
        grid.chunk_bits(cx - 1, cy - 1),
        grid.chunk_bits(cx, cy - 1),
        grid.chunk_bits(cx + 1, cy - 1),
        grid.chunk_bits(cx - 1, cy),
        grid.chunk_bits(cx, cy),
        grid.chunk_bits(cx + 1, cy),
        grid.chunk_bits(cx - 1, cy + 1),
        grid.chunk_bits(cx, cy + 1),
        grid.chunk_bits(cx + 1, cy + 1),
    ])
}

fn apply_chunk_step(
    next: &mut BitGrid,
    changes: &mut ChunkChangeSink<'_>,
    cx: Coord,
    cy: Coord,
    current_bits: u64,
    next_bits: u64,
) {
    if next_bits != 0 {
        next.set_chunk_bits(cx, cy, next_bits);
    }
    changes.record(cx, cy, current_bits, next_bits);
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

fn evolve_center_chunks_bitwise_batch_from_pending(
    pending: &PendingChunkBatch,
    chunks: &mut AlignedU16ChunkRowBatches9,
) -> [u64; SIMD_BATCH_LANES] {
    debug_assert!(!pending.is_empty());
    debug_assert!(pending.len <= SIMD_BATCH_LANES);
    if pending.len == 1 {
        let mut next = [0; SIMD_BATCH_LANES];
        next[0] = evolve_center_chunk_bitwise(&pending.neighborhoods[0]);
        return next;
    }

    build_chunk_row_batches_from_pending(pending, chunks);
    evolve_packed_chunk_rows(&chunks.0)
}

fn evolve_packed_chunk_rows(chunks: &[[u16x32; 2]; 9]) -> [u64; 8] {
    let center = pack_row_batch(&chunks[4]);
    let neighbors = packed_neighbor_boards(chunks);
    let (bit0, bit1, bit2, bit3) = accumulate_neighbor_bitplanes(&neighbors);
    let exactly_three = bit0 & bit1 & !bit2 & !bit3;
    let exactly_two = !bit0 & bit1 & !bit2 & !bit3;
    must_cast(exactly_three | (center & exactly_two))
}

fn packed_neighbor_boards(chunks: &[[u16x32; 2]; 9]) -> [u64x8; 8] {
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
fn build_chunk_row_batches_from_pending(
    pending: &PendingChunkBatch,
    chunks: &mut AlignedU16ChunkRowBatches9,
) {
    chunks.0 = [[u16x32::ZERO; 2]; 9];
    let row_lanes: &mut [[[u16; SIMD_BATCH_LANES]; 8]; 9] = must_cast_mut(&mut chunks.0);
    for lane in 0..pending.len {
        let neighborhood = pending.neighborhoods[lane];
        let first_batch = widen_u64_quad_to_u16_rows([
            neighborhood.0[0],
            neighborhood.0[1],
            neighborhood.0[2],
            neighborhood.0[3],
        ]);
        let second_batch = widen_u64_quad_to_u16_rows([
            neighborhood.0[4],
            neighborhood.0[5],
            neighborhood.0[6],
            neighborhood.0[7],
        ]);
        store_rows_batch_4(row_lanes, lane, 0, first_batch);
        store_rows_batch_4(row_lanes, lane, 4, second_batch);
        for (row, value) in chunk_rows(neighborhood.0[8]).into_iter().enumerate() {
            row_lanes[8][row][lane] = value;
        }
    }
}

fn store_rows_batch_4(
    row_lanes: &mut [[[u16; SIMD_BATCH_LANES]; 8]; 9],
    lane: usize,
    chunk_offset: usize,
    rows_batch: [[u16; 8]; 4],
) {
    for (chunk, rows) in rows_batch.into_iter().enumerate() {
        for (row, value) in rows.into_iter().enumerate() {
            row_lanes[chunk_offset + chunk][row][lane] = value;
        }
    }
}

// Batched row transforms
fn pack_row_batch(rows: &[u16x32; 2]) -> u64x8 {
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
    let as_u64: u64x8 = must_cast([
        u64::from(narrowed[0]),
        u64::from(narrowed[1]),
        u64::from(narrowed[2]),
        u64::from(narrowed[3]),
        u64::from(narrowed[4]),
        u64::from(narrowed[5]),
        u64::from(narrowed[6]),
        u64::from(narrowed[7]),
    ]);
    as_u64 << shift
}

fn align_vertical_rows(
    center: &[u16x32; 2],
    edge: &[u16x32; 2],
    edge_row: usize,
    shift_rows: i32,
) -> [u16x32; 2] {
    let edge_view: &[u16x8; 8] = must_cast_ref(edge);
    let mut shifted = *center;
    let shifted_view: &mut [u16x8; 8] = must_cast_mut(&mut shifted);
    match shift_rows {
        1 => {
            shifted_view.copy_within(0..7, 1);
            shifted_view[0] = edge_view[edge_row];
        }
        -1 => {
            shifted_view.copy_within(1..8, 0);
            shifted_view[7] = edge_view[edge_row];
        }
        _ => crate::invariant_failure!("unsupported row shift: {shift_rows}"),
    }
    shifted
}

fn align_horizontal_rows(
    center: &[u16x32; 2],
    edge: &[u16x32; 2],
    edge_col: usize,
    shift_cols: i32,
) -> [u16x32; 2] {
    let edge_mask = edge_column_mask_batch(edge, edge_col, edge_target_col(shift_cols));
    let amount = alignment_shift(shift_cols);
    if shift_cols > 0 {
        [
            (center[0] << amount) | edge_mask[0],
            (center[1] << amount) | edge_mask[1],
        ]
    } else {
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
    let amount = alignment_shift(shift_cols);
    let shifted = if shift_cols > 0 {
        center_row_lanes << amount
    } else {
        center_row_lanes >> amount
    };
    let edge_mask =
        edge_column_mask_rows(must_cast(edge_rows), edge_col, edge_target_col(shift_cols));
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
    let amount = alignment_shift(spec.shift_cols);
    let shifted_source_rows = if spec.shift_cols > 0 {
        source_row_lanes << amount
    } else {
        source_row_lanes >> amount
    };
    let edge_target = edge_target_col(spec.shift_cols);
    let edge_col = lane_index(spec.edge_col);
    let edge_row_lanes = ((edge_source_rows >> edge_col) & u16x8::ONE) << edge_target;
    pack_rows(shifted_source_rows | edge_row_lanes)
}

// Row packing and view helpers
fn align_diagonal_rows(
    center: &[u16x32; 2],
    vertical: &[u16x32; 2],
    horizontal: &[u16x32; 2],
    corner: &[u16x32; 2],
    spec: DiagonalSpec,
) -> [u16x32; 2] {
    let source_rows = align_vertical_rows(center, vertical, spec.edge_row, spec.shift_rows);
    let edge_source_rows = align_vertical_rows(horizontal, corner, spec.edge_row, spec.shift_rows);
    let edge_target = edge_target_col(spec.shift_cols);
    let edge_rows = edge_column_mask_batch(&edge_source_rows, spec.edge_col, edge_target);
    let amount = alignment_shift(spec.shift_cols);

    if spec.shift_cols > 0 {
        [
            (source_rows[0] << amount) | edge_rows[0],
            (source_rows[1] << amount) | edge_rows[1],
        ]
    } else {
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
            edge_fill_bytes(edge_fill, 0),
            row_bytes.swizzle(SHIFT_ROWS_DOWN_BYTES),
        ),
        -1 => (
            edge_fill_bytes(edge_fill, 7),
            row_bytes.swizzle(SHIFT_ROWS_UP_BYTES),
        ),
        _ => crate::invariant_failure!("unsupported row shift: {shift_rows}"),
    };
    must_cast(shifted_bytes | edge_bytes)
}

fn edge_fill_bytes(edge_fill: u16, target_row: usize) -> i8x16 {
    let mut fill_rows = [0_u16; 8];
    fill_rows[target_row] = edge_fill;
    must_cast(fill_rows)
}

fn chunk_rows(chunk: u64) -> [u16; 8] {
    chunk_rows_batch_2([chunk, 0])[0]
}

fn chunk_rows_batch_2(chunks: [u64; 2]) -> [[u16; 8]; 2] {
    widen_u64_pair_to_u16_rows(chunks)
}

fn pack_rows(rows: u16x8) -> u64 {
    let narrowed: [u16; 8] = must_cast(rows & ROW_LOW_BYTE_MASK);
    let packed_bytes = [
        u8::try_from(narrowed[0]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[1]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[2]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[3]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[4]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[5]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[6]).or_invariant("masked row exceeded u8"),
        u8::try_from(narrowed[7]).or_invariant("masked row exceeded u8"),
    ];
    must_cast(packed_bytes)
}

fn edge_target_col(shift_cols: i32) -> u16 {
    if shift_cols > 0 { 0 } else { 7 }
}

fn edge_column_mask_rows(rows: u16x8, edge_col: usize, target_col: u16) -> u16x8 {
    ((rows >> lane_index(edge_col)) & u16x8::ONE) << target_col
}

fn edge_column_mask_batch(chunk: &[u16x32; 2], edge_col: usize, target_col: u16) -> [u16x32; 2] {
    let edge_col = lane_index(edge_col);
    [
        ((chunk[0] >> edge_col) & u16x32::ONE) << target_col,
        ((chunk[1] >> edge_col) & u16x32::ONE) << target_col,
    ]
}

fn alignment_shift(shift: i32) -> u16 {
    u16::try_from(shift.unsigned_abs()).or_invariant("alignment shift exceeded u16")
}

fn lane_index(index: usize) -> u16 {
    u16::try_from(index).or_invariant("lane index exceeded u16")
}

// Changed-cell extraction
fn append_changed_cells(changed: &mut Vec<Cell>, cx: Coord, cy: Coord, diff_bits: u64) {
    if diff_bits == 0 {
        return;
    }
    append_live_bits_as_cells(changed, cx, cy, diff_bits);
}

fn expand_chunk_diffs_to_cells(chunk_diffs: &[ChunkDiff]) -> Vec<Cell> {
    let mut changed = Vec::new();
    for diff in chunk_diffs {
        append_changed_cells(&mut changed, diff.cx, diff.cy, diff.diff_bits);
    }
    changed
}

#[cfg(test)]
#[path = "tests/life.rs"]
mod tests;
