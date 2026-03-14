use super::{
    EmbedLayoutCacheKey, EmbeddedCell, EmbeddedJump, GridExtractionPolicy, HashLifeEngine, NodeId,
    hashlife_debug_enabled, quadrant_end,
};
use crate::bitgrid::{BitGrid, Coord};
use crate::hashing::morton_interleave_u64_batch;
use crate::simd_layout::{AlignedU64Batch, SIMD_BATCH_LANES};

const HASHLIFE_LINEAR_SPLIT_THRESHOLD: usize = 32;

fn translated_embedded_cells(grid: &BitGrid, shift_x: Coord, shift_y: Coord) -> Vec<EmbeddedCell> {
    let live_cells = grid.live_cells();
    let mut translated = Vec::with_capacity(live_cells.len());
    let mut index = 0;
    while index < live_cells.len() {
        let batch_len = (live_cells.len() - index).min(SIMD_BATCH_LANES);
        let mut xs = AlignedU64Batch::default();
        let mut ys = AlignedU64Batch::default();
        let mut lane = 0;
        while lane < batch_len {
            let (x, y) = live_cells[index + lane];
            xs.0[lane] =
                u64::try_from(x + shift_x).expect("hashlife translated x coordinate became invalid");
            ys.0[lane] =
                u64::try_from(y + shift_y).expect("hashlife translated y coordinate became invalid");
            lane += 1;
        }
        let keys = morton_interleave_u64_batch(xs.0, ys.0);
        let mut lane = 0;
        while lane < batch_len {
            translated.push(EmbeddedCell { key: keys[lane] });
            lane += 1;
        }
        index += batch_len;
    }
    translated.sort_unstable_by_key(|cell| cell.key);
    translated
}

impl HashLifeEngine {
    pub(super) fn embed_grid_at_level(&mut self, grid: &BitGrid, level: u32) -> NodeId {
        if grid.is_empty() {
            return self.empty(level);
        }
        let size = 1_i64 << level;
        if let Some((min_x, min_y, max_x, max_y)) = grid.bounds() {
            assert!(
                min_x >= 0 && min_y >= 0,
                "grid must be non-negative for exact level embed"
            );
            assert!(
                max_x < size && max_y < size,
                "grid exceeded exact level embed bounds"
            );
        }
        let translated = translated_embedded_cells(grid, 0, 0);
        self.build_node_from_cells_iterative(&translated, level)
    }

    pub(super) fn embed_grid_state(&mut self, grid: &BitGrid) -> (NodeId, Coord, Coord) {
        if grid.is_empty() {
            return (self.empty(2), 0, 0);
        }

        let (min_x, min_y, max_x, max_y) = grid.bounds().unwrap();
        let width = max_x - min_x + 1;
        let height = max_y - min_y + 1;
        let span = width.max(height).max(1);
        let root_size = u64::try_from(span)
            .expect("hashlife state span became negative")
            .next_power_of_two()
            .max(4);
        let root_size =
            Coord::try_from(root_size).expect("hashlife state root size exceeded Coord");
        let size_u64 = u64::try_from(root_size).expect("hashlife state root size became negative");
        let level = size_u64.trailing_zeros();
        let shift_x = (root_size - width) / 2 - min_x;
        let shift_y = (root_size - height) / 2 - min_y;

        let translated = translated_embedded_cells(grid, shift_x, shift_y);
        let root = self.build_node_from_cells_iterative(&translated, level);
        (root, -shift_x, -shift_y)
    }

    pub(super) fn required_root_size(span: Coord, jump: u64) -> Coord {
        let span = span.max(0) as u64;
        let needed = (2 * span + 4 * (jump + 2)).max((4 * jump) + 4).max(4);
        let size = needed.next_power_of_two();
        assert!(
            size <= Coord::MAX as u64,
            "hashlife root size overflow span={span} jump={jump} size={size}"
        );
        Coord::try_from(size).expect("hashlife root size exceeded Coord range")
    }

    pub(super) fn embed_for_jump(&mut self, grid: &BitGrid, step_exp: u32) -> EmbeddedJump {
        let jump = 1_u64 << step_exp;
        let (min_x, min_y, max_x, max_y) = grid.bounds().unwrap();
        let width = max_x - min_x + 1;
        let height = max_y - min_y + 1;
        let span = width.max(height);
        let size = *self
            .embed_layout_cache
            .entry(EmbedLayoutCacheKey {
                step_exp,
                width,
                height,
                span,
            })
            .or_insert_with(|| Self::required_root_size(span, jump));
        let size_u64 = u64::try_from(size).expect("hashlife root size became negative");
        let level = size_u64.trailing_zeros();
        if hashlife_debug_enabled() {
            eprintln!(
                "[hashlife] embed jump={jump} width={width} height={height} span={span} size={size} level={level}"
            );
        }
        let root_size = size;
        let shift_x = (root_size - width) / 2 - min_x;
        let shift_y = (root_size - height) / 2 - min_y;
        let translated = translated_embedded_cells(grid, shift_x, shift_y);
        let root = self.build_node_from_cells_iterative(&translated, level);
        EmbeddedJump {
            root,
            root_level: level,
            root_size,
            world_to_root_x: shift_x,
            world_to_root_y: shift_y,
            result_origin_x: root_size / 4 - shift_x,
            result_origin_y: root_size / 4 - shift_y,
        }
    }

    pub(super) fn extract_embedded_result(
        &self,
        embedded: EmbeddedJump,
        result: NodeId,
    ) -> BitGrid {
        debug_assert_eq!(self.node_columns.level(result) + 1, embedded.root_level);
        debug_assert_eq!(
            embedded.root_size / 2,
            1_i64 << self.node_columns.level(result)
        );
        debug_assert_eq!(
            embedded.result_origin_x,
            embedded.root_size / 4 - embedded.world_to_root_x
        );
        debug_assert_eq!(
            embedded.result_origin_y,
            embedded.root_size / 4 - embedded.world_to_root_y
        );
        self.node_to_grid(
            result,
            embedded.result_origin_x,
            embedded.result_origin_y,
            GridExtractionPolicy::FullGridIfUnder {
                max_population: u64::MAX,
                max_chunks: usize::MAX,
                max_bounds_span: Coord::MAX,
            },
        )
        .expect("embedded HashLife result extraction should be unrestricted")
    }

    pub(super) fn build_node_from_cells_iterative(
        &mut self,
        cells: &[EmbeddedCell],
        level: u32,
    ) -> NodeId {
        let estimated_internal = (cells.len().max(1).next_power_of_two() * 2).min(1 << 16);
        let mut ops = Vec::with_capacity((estimated_internal / 2).max(1));
        ops.push(BuildOp::Enter {
            start: 0,
            end: cells.len(),
            level,
            bit_shift: level.saturating_sub(1) * 2,
        });
        let mut results = Vec::with_capacity(estimated_internal);

        while let Some(op) = ops.pop() {
            if ops.len() > self.stats.builder_max_stack {
                self.stats.builder_max_stack = ops.len();
            }
            match op {
                BuildOp::Enter {
                    start,
                    end,
                    level,
                    bit_shift,
                } => {
                    process_build_enter_impl(
                        self,
                        cells,
                        BuildFrame {
                            start,
                            end,
                            level,
                            bit_shift,
                        },
                        &mut ops,
                        &mut results,
                    );
                    if matches!(ops.last(), Some(BuildOp::Enter { .. })) {
                        let BuildOp::Enter {
                            start,
                            end,
                            level,
                            bit_shift,
                        } = ops.pop().unwrap()
                        else {
                            unreachable!()
                        };
                        process_build_enter_impl(
                            self,
                            cells,
                            BuildFrame {
                                start,
                                end,
                                level,
                                bit_shift,
                            },
                            &mut ops,
                            &mut results,
                        );
                    }
                    if ops.len() > self.stats.builder_max_stack {
                        self.stats.builder_max_stack = ops.len();
                    }
                }
                BuildOp::Combine => {
                    let se = results.pop().unwrap();
                    let sw = results.pop().unwrap();
                    let ne = results.pop().unwrap();
                    let nw = results.pop().unwrap();
                    results.push(self.join(nw, ne, sw, se));
                }
            }
        }

        results.pop().unwrap()
    }
}

#[derive(Clone, Copy)]
enum BuildOp {
    Enter {
        start: usize,
        end: usize,
        level: u32,
        bit_shift: u32,
    },
    Combine,
}

#[derive(Clone, Copy)]
struct BuildFrame {
    start: usize,
    end: usize,
    level: u32,
    bit_shift: u32,
}

fn process_build_enter_impl(
    oracle: &mut HashLifeEngine,
    cells: &[EmbeddedCell],
    frame: BuildFrame,
    ops: &mut Vec<BuildOp>,
    results: &mut Vec<NodeId>,
) {
    oracle.stats.builder_frames += 1;
    let BuildFrame {
        start,
        end,
        level,
        bit_shift,
    } = frame;
    if start == end {
        results.push(oracle.empty(level));
        return;
    }

    if level == 0 {
        results.push(oracle.live_leaf);
        return;
    }

    oracle.stats.builder_partitions += 1;
    let (q0_end, q1_end, q2_end) = split_quadrants(cells, start, end, bit_shift);

    ops.push(BuildOp::Combine);
    ops.push(BuildOp::Enter {
        start: q2_end,
        end,
        level: level - 1,
        bit_shift: bit_shift.saturating_sub(2),
    });
    ops.push(BuildOp::Enter {
        start: q1_end,
        end: q2_end,
        level: level - 1,
        bit_shift: bit_shift.saturating_sub(2),
    });
    ops.push(BuildOp::Enter {
        start: q0_end,
        end: q1_end,
        level: level - 1,
        bit_shift: bit_shift.saturating_sub(2),
    });
    ops.push(BuildOp::Enter {
        start,
        end: q0_end,
        level: level - 1,
        bit_shift: bit_shift.saturating_sub(2),
    });
}

fn split_quadrants(
    cells: &[EmbeddedCell],
    start: usize,
    end: usize,
    bit_shift: u32,
) -> (usize, usize, usize) {
    let len = end - start;
    let mut q0_end;
    let mut q1_end;
    let mut q2_end;
    if len <= HASHLIFE_LINEAR_SPLIT_THRESHOLD {
        q0_end = start;
        while q0_end < end && ((cells[q0_end].key >> bit_shift) & 0b11) == 0 {
            q0_end += 1;
        }
        q1_end = q0_end;
        while q1_end < end && ((cells[q1_end].key >> bit_shift) & 0b11) == 1 {
            q1_end += 1;
        }
        q2_end = q1_end;
        while q2_end < end && ((cells[q2_end].key >> bit_shift) & 0b11) == 2 {
            q2_end += 1;
        }
    } else {
        q0_end = quadrant_end(cells, start, end, bit_shift, 0);
        q1_end = quadrant_end(cells, q0_end, end, bit_shift, 1);
        q2_end = quadrant_end(cells, q1_end, end, bit_shift, 2);
    }
    (q0_end, q1_end, q2_end)
}
