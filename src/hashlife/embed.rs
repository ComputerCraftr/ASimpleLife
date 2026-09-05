use super::advance::quadrant_end;
use super::geometry::{HashLifeGeometryError, RootGeometry, ValidatedLevel};
#[cfg(test)]
use super::{EmbedLayoutCacheKey, EmbeddedJump};
use super::{EmbeddedCell, HashLifeEngine, NodeId};
use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Coord};
use crate::hashing::morton_interleave_u64_batch;
use crate::simd_layout::{AlignedU64Batch, SIMD_BATCH_LANES};

const HASHLIFE_LINEAR_SPLIT_THRESHOLD: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HashLifeEmbedError {
    pub(super) axis: &'static str,
}

impl From<HashLifeGeometryError> for HashLifeEmbedError {
    fn from(error: HashLifeGeometryError) -> Self {
        let axis = match error {
            HashLifeGeometryError::LevelOutOfRange { .. } => "level",
            HashLifeGeometryError::CoordinateRangeExceeded { axis } => axis,
        };
        Self { axis }
    }
}

fn translated_embedded_cells(
    grid: &BitGrid,
    origin_x: Coord,
    origin_y: Coord,
    mut translated: Vec<EmbeddedCell>,
) -> Result<Vec<EmbeddedCell>, HashLifeEmbedError> {
    debug_assert!(translated.capacity() >= grid.population());
    let mut cell_batch = [(0_i64, 0_i64); SIMD_BATCH_LANES];
    let mut batch_len = 0;
    let flush_batch = |cell_batch: &[(i64, i64); SIMD_BATCH_LANES],
                       batch_len: usize,
                       translated: &mut Vec<EmbeddedCell>|
     -> Result<(), HashLifeEmbedError> {
        let mut xs = AlignedU64Batch::default();
        let mut ys = AlignedU64Batch::default();
        let mut lane = 0;
        while lane < batch_len {
            let (x, y) = cell_batch[lane];
            let translated_x = i128::from(x) - i128::from(origin_x);
            let translated_y = i128::from(y) - i128::from(origin_y);
            xs.0[lane] =
                u64::try_from(translated_x).map_err(|_| HashLifeEmbedError { axis: "x" })?;
            ys.0[lane] =
                u64::try_from(translated_y).map_err(|_| HashLifeEmbedError { axis: "y" })?;
            lane += 1;
        }
        let keys = morton_interleave_u64_batch(xs.0, ys.0);
        let mut lane = 0;
        while lane < batch_len {
            translated.push(EmbeddedCell { key: keys[lane] });
            lane += 1;
        }
        Ok(())
    };

    for ((chunk_x, chunk_y), mut bits) in grid.occupied_chunks() {
        if bits == 0 {
            continue;
        }
        let base_x = chunk_x * 8;
        let base_y = chunk_y * 8;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let local_x = i64::from(bit % 8);
            let local_y = i64::from(bit / 8);
            cell_batch[batch_len] = (base_x + local_x, base_y + local_y);
            batch_len += 1;
            bits &= bits - 1;
            if batch_len == SIMD_BATCH_LANES {
                flush_batch(&cell_batch, batch_len, &mut translated)?;
                batch_len = 0;
            }
        }
    }
    if batch_len != 0 {
        flush_batch(&cell_batch, batch_len, &mut translated)?;
    }
    translated.sort_unstable_by_key(|cell| cell.key);
    Ok(translated)
}

impl HashLifeEngine {
    pub(super) fn try_embed_grid_state(
        &mut self,
        grid: &BitGrid,
    ) -> Result<(NodeId, Coord, Coord), HashLifeEmbedError> {
        if grid.is_empty() {
            return Ok((self.empty(2), 0, 0));
        }

        let (min_x, min_y, max_x, max_y) = grid.bounds().or_invariant("required value");
        let width = i128::from(max_x) - i128::from(min_x) + 1;
        let height = i128::from(max_y) - i128::from(min_y) + 1;
        let span = width.max(height).max(1);
        let span = u128::try_from(span).map_err(|_| HashLifeEmbedError { axis: "span" })?;
        let root_size = span
            .checked_next_power_of_two()
            .ok_or(HashLifeEmbedError { axis: "span" })?
            .max(4);
        let level = root_size.trailing_zeros();
        let geometry = RootGeometry::containing_bounds(level, min_x, min_y, max_x, max_y)?;
        debug_assert_eq!(geometry.level, ValidatedLevel::new(level)?);
        let origin_x = geometry.origin_x;
        let origin_y = geometry.origin_y;

        let Some(translated_workspace) = self.try_transient_vec(grid.population()) else {
            return Err(HashLifeEmbedError { axis: "allocation" });
        };
        let translated = translated_embedded_cells(grid, origin_x, origin_y, translated_workspace)?;
        let root = self.build_node_from_cells_iterative(&translated, level);
        Ok((root, origin_x, origin_y))
    }

    pub(super) fn build_node_from_cells_iterative(
        &mut self,
        cells: &[EmbeddedCell],
        level: u32,
    ) -> NodeId {
        let depth_capacity = usize::try_from(level)
            .unwrap_or(usize::MAX)
            .saturating_add(1)
            .saturating_mul(5)
            .max(8);
        let Some(mut ops) = self.try_transient_vec(depth_capacity) else {
            return self.dead_leaf;
        };
        ops.push(BuildOp::Enter {
            start: 0,
            end: cells.len(),
            level,
            bit_shift: level.saturating_sub(1) * 2,
        });
        let Some(mut results) = self.try_transient_vec(depth_capacity) else {
            return self.dead_leaf;
        };

        while let Some(op) = ops.pop() {
            if ops.len() > self.stats.scheduler.builder_max_stack {
                self.stats.scheduler.builder_max_stack = ops.len();
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
                        } = ops.pop().or_invariant("required value")
                        else {
                            crate::invariant_failure!()
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
                    if ops.len() > self.stats.scheduler.builder_max_stack {
                        self.stats.scheduler.builder_max_stack = ops.len();
                    }
                }
                BuildOp::Combine => {
                    let se = results.pop().or_invariant("required value");
                    let sw = results.pop().or_invariant("required value");
                    let ne = results.pop().or_invariant("required value");
                    let nw = results.pop().or_invariant("required value");
                    results.push(self.join(nw, ne, sw, se));
                }
            }
        }

        results.pop().or_invariant("required value")
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(super) fn required_root_size(span: Coord, jump: u64) -> Coord {
        let span = u128::try_from(span.max(0)).or_invariant("nonnegative span should fit u128");
        let jump = u128::from(jump);
        let needed = span
            .saturating_mul(2)
            .saturating_add(jump.saturating_add(2).saturating_mul(4))
            .max(jump.saturating_mul(4).saturating_add(4))
            .max(4);
        let size = needed.next_power_of_two();
        assert!(
            size <= Coord::MAX as u128,
            "hashlife root size overflow span={span} jump={jump} size={size}"
        );
        Coord::try_from(size).or_invariant("hashlife root size exceeded Coord range")
    }

    pub(super) fn embed_for_jump_with_bounds(
        &mut self,
        grid: &BitGrid,
        (min_x, min_y, max_x, max_y): (Coord, Coord, Coord, Coord),
        step_exp: u32,
    ) -> EmbeddedJump {
        let jump = 1_u64 << step_exp;
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
        let level = u128::try_from(size)
            .or_invariant("hashlife root size became negative")
            .trailing_zeros();
        let root_size = size;
        let shift_x = (root_size - width) / 2 - min_x;
        let shift_y = (root_size - height) / 2 - min_y;
        let origin_x = shift_x
            .checked_neg()
            .or_invariant("test embedding origin x must be representable");
        let origin_y = shift_y
            .checked_neg()
            .or_invariant("test embedding origin y must be representable");
        let translated = translated_embedded_cells(
            grid,
            origin_x,
            origin_y,
            Vec::with_capacity(grid.population()),
        )
        .or_invariant("test embedding geometry must be representable");
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
        &mut self,
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
        if self.node_columns.population(result) == 0 {
            BitGrid::empty()
        } else {
            self.stats.materialization.embedded_result_full_extractions += 1;
            self.node_to_grid_all(result, embedded.result_origin_x, embedded.result_origin_y)
        }
    }

    pub(super) fn embed_for_jump(&mut self, grid: &BitGrid, step_exp: u32) -> EmbeddedJump {
        self.embed_for_jump_with_bounds(
            grid,
            grid.bounds()
                .or_invariant("cannot embed an empty HashLife jump"),
            step_exp,
        )
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
    oracle.stats.scheduler.builder_frames += 1;
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

    oracle.stats.scheduler.builder_partitions += 1;
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
