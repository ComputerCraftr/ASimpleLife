use super::{EmbeddedCell, EmbeddedJump, HashLifeOracle, NodeId, hashlife_debug_enabled, morton_key, quadrant_end};
use crate::bitgrid::BitGrid;

impl HashLifeOracle {
    pub(super) fn required_root_size(span: i32, jump: i32) -> u32 {
        let needed = (2 * span + 4 * (jump + 2)).max((4 * jump) + 4).max(4);
        (needed as u32).next_power_of_two()
    }

    pub(super) fn embed_for_jump(&mut self, grid: &BitGrid, step_exp: u32) -> EmbeddedJump {
        let jump = 1_i32 << step_exp;
        let (min_x, min_y, max_x, max_y) = grid.bounds().unwrap();
        let width = max_x - min_x + 1;
        let height = max_y - min_y + 1;
        let span = width.max(height);
        let size = *self
            .embed_layout_cache
            .entry((step_exp, width, height, span))
            .or_insert_with(|| Self::required_root_size(span, jump));
        let level = size.trailing_zeros();
        if hashlife_debug_enabled() {
            eprintln!(
                "[hashlife] embed jump={jump} width={width} height={height} span={span} size={size} level={level}"
            );
        }
        let shift_x = (size as i32 - width) / 2 - min_x;
        let shift_y = (size as i32 - height) / 2 - min_y;
        let live_cells = grid.live_cells();
        let mut translated = Vec::with_capacity(live_cells.len());
        translated.extend(
            live_cells
            .into_iter()
            .map(|(x, y)| {
                let tx = (x + shift_x) as u32;
                let ty = (y + shift_y) as u32;
                EmbeddedCell {
                    key: morton_key(tx, ty),
                }
            }),
        );
        translated.sort_unstable_by_key(|cell| cell.key);
        let root = self.build_node_from_cells_iterative(&translated, level);
        EmbeddedJump {
            root,
            root_level: level,
            root_size: size as i32,
            world_to_root_x: shift_x,
            world_to_root_y: shift_y,
            result_origin_x: (size as i32) / 4 - shift_x,
            result_origin_y: (size as i32) / 4 - shift_y,
        }
    }

    pub(super) fn extract_embedded_result(&self, embedded: EmbeddedJump, result: NodeId) -> BitGrid {
        debug_assert_eq!(self.nodes[result as usize].level + 1, embedded.root_level);
        debug_assert_eq!(embedded.root_size / 2, 1_i32 << self.nodes[result as usize].level);
        debug_assert_eq!(
            embedded.result_origin_x,
            embedded.root_size / 4 - embedded.world_to_root_x
        );
        debug_assert_eq!(
            embedded.result_origin_y,
            embedded.root_size / 4 - embedded.world_to_root_y
        );
        self.node_to_grid(result, embedded.result_origin_x, embedded.result_origin_y)
    }

    pub(super) fn build_node_from_cells_iterative(&mut self, cells: &[EmbeddedCell], level: u32) -> NodeId {
        enum BuildOp {
            Enter {
                start: usize,
                end: usize,
                level: u32,
                bit_shift: u32,
            },
            Combine,
        }

        let mut ops = vec![BuildOp::Enter {
            start: 0,
            end: cells.len(),
            level,
            bit_shift: level.saturating_sub(1) * 2,
        }];
        let estimated_internal = (cells.len().max(1).next_power_of_two() * 2).min(1 << 16);
        ops.reserve(estimated_internal / 2);
        let mut results = Vec::with_capacity(estimated_internal);

        while let Some(op) = ops.pop() {
            match op {
                BuildOp::Enter {
                    start,
                    end,
                    level,
                    bit_shift,
                } => {
                    if start == end {
                        results.push(self.empty(level));
                        continue;
                    }

                    if level == 0 {
                        results.push(self.live_leaf);
                        continue;
                    }

                    self.stats.builder_partitions += 1;
                    let q0_end = quadrant_end(cells, start, end, bit_shift, 0);
                    let q1_end = quadrant_end(cells, q0_end, end, bit_shift, 1);
                    let q2_end = quadrant_end(cells, q1_end, end, bit_shift, 2);

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
