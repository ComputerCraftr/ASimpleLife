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
        let root = current_root?;
        let active_gc_needed = super::memory::wide_allocated_bytes(self.allocated_bytes())
            >= soft_memory_bytes
            && should_run_active_hashlife_gc(self.node_count(), self.last_gc_nodes);
        if !active_gc_needed {
            return Some(root);
        }

        self.record_retained_root(root);
        self.stats.gc.jump_cache_before_clear = self.result_caches.jump.len();
        self.maybe_garbage_collect_with_budget("growth_threshold", hard_memory_bytes);
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
            let (next, root) = self.advance_power_of_two(current_grid, bounds, step_exp);
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
    ) -> (BitGrid, NodeId) {
        if grid.is_empty() {
            return (BitGrid::empty(), self.empty(0));
        }
        let embedded = self.embed_for_jump_with_bounds(grid, bounds, step_exp);
        let cache_key = (embedded.root, step_exp);
        let advanced = if let Some(cached) = self.cached_root_result(cache_key) {
            cached
        } else {
            let result = self.advance_pow2(embedded.root, step_exp);
            self.insert_root_result(cache_key, result);
            result
        };
        (self.extract_embedded_result(embedded, advanced), advanced)
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
