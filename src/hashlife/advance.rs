use super::*;

impl HashLifeEngine {
    pub(super) fn begin_persistent_run(&mut self) -> Option<NodeId> {
        self.stats = HashLifeStats::default();
        self.retained_roots.last().copied()
    }

    pub(super) fn finish_persistent_run(
        &mut self,
        previous_root: Option<NodeId>,
        last_root: Option<NodeId>,
        hard_memory_bytes: u128,
    ) {
        if let Some(root) = last_root {
            self.record_retained_root(root);
        }
        self.stats.gc.jump_cache_before_clear = self.result_caches.jump.len();
        let gc_reason = self.gc_reason(previous_root, last_root);
        self.maybe_garbage_collect_with_budget(gc_reason, hard_memory_bytes);
    }

    pub(super) fn maybe_collect_active_run(
        &mut self,
        current_root: Option<NodeId>,
        soft_memory_bytes: u128,
        hard_memory_bytes: u128,
    ) -> Option<NodeId> {
        let allocated = super::memory::wide_allocated_bytes(self.allocated_bytes());
        let near_hard_limit = allocated >= hard_memory_bytes.saturating_sub(hard_memory_bytes / 5);
        let changed = self.node_count() != self.last_gc_nodes
            || self.retained_roots.last().copied() != current_root
            || self.canonical_caches.shapes.len() > self.node_count();
        if near_hard_limit && !changed && self.at_gc_safepoint() {
            // Cache-only pressure needs no repeated graph traversal or epoch change.
            self.release_optional_cache_storage();
            return current_root;
        }
        let urgent = near_hard_limit && (changed || allocated > hard_memory_bytes);
        let active_gc_needed = urgent
            || (allocated >= soft_memory_bytes
                && should_run_active_hashlife_gc(self.node_count(), self.last_gc_nodes));
        if !active_gc_needed {
            return current_root;
        }

        if let Some(root) = current_root {
            self.record_retained_root(root);
        } else {
            self.retained_roots.clear();
        }
        self.stats.gc.jump_cache_before_clear = self.result_caches.jump.len();
        self.maybe_garbage_collect_with_budget(
            if urgent {
                "budget_pressure"
            } else {
                "growth_threshold"
            },
            hard_memory_bytes,
        );
        self.retained_roots.last().copied()
    }

    pub(super) fn advance_pow2(&mut self, node: NodeId, step_exp: u32) -> NodeId {
        debug_assert!(self.node_columns.level(node) >= 2);
        if self.node_columns.population(node) == 0 {
            return self.empty(self.node_columns.level(node) - 1);
        }
        self.active_jump_results.reset();
        self.scheduler_active = true;
        let result = if step_exp == 0 {
            self.advance_one_generation_centered(node)
        } else {
            self.advance_power_of_two_recursive(node, step_exp)
        };
        self.scheduler_active = false;
        self.active_jump_results.reset();
        result
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(crate) fn advance(&mut self, grid: &BitGrid, generations: u64) -> BitGrid {
        let previous_root = self.begin_persistent_run();
        let (advanced, last_root) = self.advance_segment(grid, generations);
        self.finish_persistent_run(previous_root, last_root, u128::MAX);
        advanced
    }

    fn advance_segment(&mut self, grid: &BitGrid, generations: u64) -> (BitGrid, Option<NodeId>) {
        if grid.is_empty() {
            return (BitGrid::empty(), Some(self.empty(0)));
        }
        if generations == 0 {
            return (grid.clone(), None);
        }
        let mut current = None::<BitGrid>;
        let mut remaining = generations;
        let mut last_root = None;
        while remaining != 0 {
            let current_grid = current.as_ref().unwrap_or(grid);
            if current_grid.is_empty() {
                break;
            }
            let bounds = current_grid
                .bounds()
                .or_invariant("non-empty HashLife segment lost its bounds");
            let safe_jump = max_hashlife_safe_jump_from_bounds(bounds);
            let step_limit = remaining.min(safe_jump.max(1));
            let step_exp = 63 - step_limit.leading_zeros();
            let step = 1_u64 << step_exp;
            let Some((next, root)) = self.advance_power_of_two(current_grid, bounds, step_exp)
            else {
                debug_assert!(self.allocation_failed());
                return (current_grid.clone(), last_root);
            };
            current = Some(next);
            last_root = Some(root);
            remaining -= step;
        }
        (current.unwrap_or_else(BitGrid::empty), last_root)
    }

    fn advance_power_of_two(
        &mut self,
        grid: &BitGrid,
        bounds: (Coord, Coord, Coord, Coord),
        step_exp: u32,
    ) -> Option<(BitGrid, NodeId)> {
        if grid.is_empty() {
            let root = self.empty(0);
            return (!self.allocation_failed()).then_some((BitGrid::empty(), root));
        }
        let embedded = self.embed_for_jump_with_bounds(grid, bounds, step_exp);
        if self.allocation_failed() {
            return None;
        }
        let cache_key = (embedded.root, step_exp);
        let advanced = if let Some(cached) = self.cached_root_result(cache_key) {
            if self.allocation_failed() {
                return None;
            }
            cached
        } else {
            let result = self.advance_pow2(embedded.root, step_exp);
            if self.allocation_failed() {
                return None;
            }
            self.insert_root_result(cache_key, result);
            result
        };
        if self.allocation_failed() {
            return None;
        }
        Some((self.extract_embedded_result(embedded, advanced), advanced))
    }
}

#[cfg(test)]
fn max_hashlife_safe_jump_from_bounds(
    (min_x, min_y, max_x, max_y): (Coord, Coord, Coord, Coord),
) -> u64 {
    let width = (max_x - min_x + 1).max(1);
    let height = (max_y - min_y + 1).max(1);
    max_hashlife_safe_jump_from_span(width.max(height))
}

#[cfg(test)]
pub(super) fn max_hashlife_safe_jump_from_span(span: Coord) -> u64 {
    if span <= 0 {
        return 1;
    }
    let raw_max_jump =
        u64::try_from(((i128::from(Coord::MAX) - 2 * i128::from(span) - 8) / 4).max(1))
            .or_invariant("safe HashLife jump bound should fit u64");
    let mut jump = 1_u64 << (63 - raw_max_jump.leading_zeros());
    while jump > 1
        && required_root_size_for_jump(
            u128::try_from(span).or_invariant("positive span should fit u128"),
            u128::from(jump),
        ) > Coord::MAX as u128
    {
        jump >>= 1;
    }
    if required_root_size_for_jump(
        u128::try_from(span).or_invariant("positive span should fit u128"),
        u128::from(jump),
    ) <= Coord::MAX as u128
    {
        jump
    } else {
        0
    }
}

#[cfg(test)]
pub(super) fn required_root_size_for_jump(span: u128, jump: u128) -> u128 {
    span.saturating_mul(2)
        .saturating_add(jump.saturating_add(2).saturating_mul(4))
        .max(jump.saturating_mul(4).saturating_add(4))
        .max(4)
        .checked_next_power_of_two()
        .unwrap_or(u128::MAX)
}

pub(super) fn quadrant_end(
    cells: &[EmbeddedCell],
    start: usize,
    end: usize,
    bit_shift: u32,
    quadrant: u128,
) -> usize {
    let upper = quadrant + 1;
    start + cells[start..end].partition_point(|cell| ((cell.key >> bit_shift) & 0b11) < upper)
}

#[cfg(test)]
mod mandatory_failure_tests {
    use super::*;

    fn node_from_4x4_bits(engine: &mut HashLifeEngine, bits: u16) -> NodeId {
        let quadrants: [NodeId; 4] = std::array::from_fn(|quadrant| {
            let origin_x = (quadrant % 2) * 2;
            let origin_y = (quadrant / 2) * 2;
            let leaves: [NodeId; 4] = std::array::from_fn(|child| {
                let x = origin_x + child % 2;
                let y = origin_y + child / 2;
                if bits & (1_u16 << (y * 4 + x)) == 0 {
                    engine.dead_leaf
                } else {
                    engine.live_leaf
                }
            });
            engine.join(leaves[0], leaves[1], leaves[2], leaves[3])
        });
        engine.join(quadrants[0], quadrants[1], quadrants[2], quadrants[3])
    }

    fn node_from_8x8_bits(engine: &mut HashLifeEngine, bits: u64) -> NodeId {
        let quadrants: [NodeId; 4] = std::array::from_fn(|quadrant| {
            let origin_x = (quadrant % 2) * 4;
            let origin_y = (quadrant / 2) * 4;
            let mut quadrant_bits = 0_u16;
            for y in 0..4 {
                for x in 0..4 {
                    if bits & (1_u64 << ((origin_y + y) * 8 + origin_x + x)) != 0 {
                        quadrant_bits |= 1_u16 << (y * 4 + x);
                    }
                }
            }
            node_from_4x4_bits(engine, quadrant_bits)
        });
        engine.join(quadrants[0], quadrants[1], quadrants[2], quadrants[3])
    }

    #[test]
    fn embedding_failure_does_not_extract_or_publish_a_sentinel() {
        let mut engine = HashLifeEngine::default();
        let grid = BitGrid::from_cells(&[(0, 0)]);
        let bounds = grid.bounds().or_invariant("single-cell bounds");
        let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);

        assert_eq!(engine.advance_power_of_two(&grid, bounds, 0), None);
        assert!(engine.allocation_failed());
        assert_eq!(engine.result_caches.root.len(), 0);
        assert_eq!(engine.result_caches.materialized_packed.len(), 0);

        let mut segmented = HashLifeEngine::default();
        let retained = crate::hashlife::memory::wide_allocated_bytes(segmented.allocated_bytes());
        segmented.begin_allocation_transaction(retained);
        let (preserved, last_root) = segmented.advance_segment(&grid, 1);
        assert_eq!(preserved, grid);
        assert_eq!(last_root, None);
    }

    #[test]
    fn scheduler_scratch_failure_does_not_publish_a_sentinel() {
        let mut engine = HashLifeEngine::default();
        let source = node_from_8x8_bits(&mut engine, 1_u64 << (3 * 8 + 3));
        let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);

        assert_eq!(engine.advance_pow2(source, 0), engine.dead_leaf);
        assert!(engine.allocation_failed());
        assert_eq!(engine.active_jump_results.len(), 0);
        assert_eq!(engine.result_caches.jump.len(), 0);
    }
}
