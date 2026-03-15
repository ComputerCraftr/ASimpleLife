use super::{
    GridExtractionError, GridExtractionPolicy, HashLifeEngine, NodeId, PackedNodeKey,
    RelativeBounds,
};
use crate::RequiredExt;
use crate::bitgrid::{BitGrid, CHUNK_SIZE, Cell, Coord};

#[derive(Clone, Copy)]
struct BoundsFrame {
    node: NodeId,
    next_child: usize,
    bounds: RelativeBounds,
}

impl HashLifeEngine {
    pub(super) fn node_cell_alive(
        &self,
        mut node: NodeId,
        origin_x: Coord,
        origin_y: Coord,
        x: Coord,
        y: Coord,
    ) -> bool {
        let mut level = self.node_columns.level(node);
        let mut relative_x = i128::from(x) - i128::from(origin_x);
        let mut relative_y = i128::from(y) - i128::from(origin_y);
        let size = 1_i128 << level;
        if relative_x < 0 || relative_y < 0 || relative_x >= size || relative_y >= size {
            return false;
        }

        while level != 0 {
            if self.node_columns.population(node) == 0 {
                return false;
            }
            let half = 1_i128 << (level - 1);
            let east = relative_x >= half;
            let south = relative_y >= half;
            if east {
                relative_x -= half;
            }
            if south {
                relative_y -= half;
            }
            let quadrants = self.node_columns.quadrants(node);
            node = quadrants[usize::from(south) * 2 + usize::from(east)];
            level -= 1;
        }
        self.node_columns.population(node) != 0
    }

    pub(super) fn centered_shell(&mut self, node: NodeId) -> NodeId {
        let level = self.node_columns.level(node);
        let key = super::ShellKey {
            node,
            target_level: u8::try_from(level + if level == 0 { 2 } else { 1 })
                .or_invariant("HashLife shell level exceeded u8 capacity"),
        };
        if let Some(shell) = self.result_caches.shells.get(&key) {
            return shell;
        }
        let shell = if level == 0 {
            let empty = self.dead_leaf;
            let nw = self.join(empty, empty, empty, node);
            let other = self.join(empty, empty, empty, empty);
            self.join(nw, other, other, other)
        } else {
            let empty = self.empty(level - 1);
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            let upper_left = self.join(empty, empty, empty, nw);
            let upper_right = self.join(empty, empty, ne, empty);
            let lower_left = self.join(empty, sw, empty, empty);
            let lower_right = self.join(se, empty, empty, empty);
            self.join(upper_left, upper_right, lower_left, lower_right)
        };
        self.publish_optional_cache(
            |engine| &engine.result_caches.shells,
            |engine| &mut engine.result_caches.shells,
            key,
            crate::flat_table::FlatKey::fingerprint(&key),
            shell,
        );
        shell
    }

    pub(super) fn base_transition_batch(
        &mut self,
        nodes: [NodeId; super::SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> [NodeId; super::SIMD_BATCH_LANES] {
        let mut masks = [0_u16; super::SIMD_BATCH_LANES];
        for lane in 0..active_lanes {
            masks[lane] = self.level2_to_4x4_mask(nodes[lane]);
        }
        let (centered, accounting) =
            super::kernels::KernelSet::selected().base_transition(&masks, active_lanes);
        self.record_kernel_accounting(accounting);

        let mut results = [self.dead_leaf; super::SIMD_BATCH_LANES];
        let leaves = [self.dead_leaf, self.live_leaf];
        for lane in 0..active_lanes {
            let output = centered[lane];
            results[lane] = self.join(
                leaves[(output & 0b0001 != 0) as usize],
                leaves[((output >> 1) & 0b0001 != 0) as usize],
                leaves[((output >> 2) & 0b0001 != 0) as usize],
                leaves[((output >> 3) & 0b0001 != 0) as usize],
            );
        }
        results
    }

    fn level2_to_4x4_mask(&self, node: NodeId) -> u16 {
        let [nw_node, ne_node, sw_node, se_node] = self.node_columns.quadrants(node);
        self.level1_to_4x4_mask(nw_node, 0, 0)
            | self.level1_to_4x4_mask(ne_node, 2, 0)
            | self.level1_to_4x4_mask(sw_node, 0, 2)
            | self.level1_to_4x4_mask(se_node, 2, 2)
    }

    fn level1_to_4x4_mask(&self, node: NodeId, base_x: u16, base_y: u16) -> u16 {
        let [nw, ne, sw, se] = self.node_columns.quadrants(node);
        debug_assert_eq!(self.node_columns.level(node), 1);
        (u16::from(self.node_columns.population(nw) != 0) << (base_y * 4 + base_x))
            | (u16::from(self.node_columns.population(ne) != 0) << (base_y * 4 + base_x + 1))
            | (u16::from(self.node_columns.population(sw) != 0) << ((base_y + 1) * 4 + base_x))
            | (u16::from(self.node_columns.population(se) != 0) << ((base_y + 1) * 4 + base_x + 1))
    }

    pub(super) fn node_to_grid(
        &mut self,
        node: NodeId,
        offset_x: Coord,
        offset_y: Coord,
        policy: GridExtractionPolicy,
    ) -> Result<BitGrid, GridExtractionError> {
        let size = 1_i64
            .checked_shl(self.node_columns.level(node))
            .or_invariant("validated HashLife level must fit coordinate geometry");
        let limits = extraction_limits(self, node, offset_x, offset_y, policy)?;
        let estimated_chunks = usize::try_from(
            self.node_columns
                .population(node)
                .div_ceil(64)
                .min(limits.max_chunks.unwrap_or(usize::MAX) as u128),
        )
        .unwrap_or(usize::MAX);
        let mut grid = BitGrid::try_with_chunk_capacity(estimated_chunks.max(1))
            .map_err(|_| GridExtractionError::AllocationFailed)?;
        self.collect_chunks_iterative(
            node,
            (offset_x, offset_y),
            size,
            limits.clip_bounds,
            limits.max_chunks,
            &mut grid,
        )?;
        Ok(grid)
    }

    pub(super) fn node_to_grid_clipped(
        &self,
        node: NodeId,
        offset_x: Coord,
        offset_y: Coord,
        clip_bounds: (Coord, Coord, Coord, Coord),
    ) -> BitGrid {
        let size = 1_i64
            .checked_shl(self.node_columns.level(node))
            .or_invariant("validated HashLife level must fit coordinate geometry");
        let estimated_chunks =
            usize::try_from(self.node_columns.population(node).div_ceil(64)).unwrap_or(usize::MAX);
        let mut grid = BitGrid::try_with_chunk_capacity(estimated_chunks.max(1))
            .or_invariant("bounded clipped extraction allocation failed");
        self.collect_chunks_iterative(
            node,
            (offset_x, offset_y),
            size,
            clip_bounds,
            None,
            &mut grid,
        )
        .or_invariant("clipped extraction should not enforce chunk limits");
        grid
    }

    #[cfg(test)]
    pub(super) fn node_to_grid_all(
        &self,
        node: NodeId,
        offset_x: Coord,
        offset_y: Coord,
    ) -> BitGrid {
        if self.node_columns.population(node) == 0 {
            return BitGrid::empty();
        }
        let size = 1_i64 << self.node_columns.level(node);
        let estimated_chunks =
            usize::try_from(self.node_columns.population(node).div_ceil(64)).unwrap_or(usize::MAX);
        let mut grid = BitGrid::try_with_chunk_capacity(estimated_chunks.max(1))
            .or_invariant("test extraction allocation failed");
        self.collect_all_chunks_iterative(node, offset_x, offset_y, size, &mut grid);
        grid
    }

    pub(super) fn node_bounds(
        &mut self,
        node: NodeId,
        origin_x: Coord,
        origin_y: Coord,
    ) -> Option<(Coord, Coord, Coord, Coord)> {
        let bounds = self.node_relative_bounds(node)?;
        Some((
            origin_x
                .checked_add(bounds.min_x)
                .or_invariant("HashLife bounds min x overflow"),
            origin_y
                .checked_add(bounds.min_y)
                .or_invariant("HashLife bounds min y overflow"),
            origin_x
                .checked_add(bounds.max_x)
                .or_invariant("HashLife bounds max x overflow"),
            origin_y
                .checked_add(bounds.max_y)
                .or_invariant("HashLife bounds max y overflow"),
        ))
    }

    fn node_relative_bounds(&mut self, node: NodeId) -> Option<RelativeBounds> {
        if self.node_columns.population(node) == 0 {
            return None;
        }
        if let Some(bounds) = self.result_caches.bounds.get(&node) {
            return Some(bounds);
        }

        let empty_bounds = RelativeBounds {
            min_x: Coord::MAX,
            min_y: Coord::MAX,
            max_x: Coord::MIN,
            max_y: Coord::MIN,
        };
        let mut stack = [BoundsFrame {
            node,
            next_child: 0,
            bounds: empty_bounds,
        }; 64];
        let mut stack_len = 1;
        let mut completed = None;
        while stack_len != 0 {
            if let Some(child_bounds) = completed.take() {
                let parent = &mut stack[stack_len - 1];
                merge_child_bounds(
                    &mut parent.bounds,
                    child_bounds,
                    parent.next_child - 1,
                    self.node_columns.level(parent.node),
                );
            }
            let frame = &mut stack[stack_len - 1];
            let level = self.node_columns.level(frame.node);
            if level == 0 || frame.next_child == 4 {
                let bounds = if level == 0 {
                    RelativeBounds {
                        min_x: 0,
                        min_y: 0,
                        max_x: 0,
                        max_y: 0,
                    }
                } else {
                    frame.bounds
                };
                let current = frame.node;
                self.publish_optional_cache(
                    |engine| &engine.result_caches.bounds,
                    |engine| &mut engine.result_caches.bounds,
                    current,
                    crate::flat_table::FlatKey::fingerprint(&current),
                    bounds,
                );
                stack_len -= 1;
                if stack_len == 0 {
                    return Some(bounds);
                }
                completed = Some(bounds);
                continue;
            }

            let child_index = frame.next_child;
            frame.next_child += 1;
            let child = self.node_columns.quadrants(frame.node)[child_index];
            if self.node_columns.population(child) == 0 {
                continue;
            }
            if let Some(bounds) = self.result_caches.bounds.get(&child) {
                merge_child_bounds(&mut frame.bounds, bounds, child_index, level);
                continue;
            }
            if stack_len == stack.len() {
                crate::invariant_failure!("validated bounds depth exceeded fixed workspace");
            }
            stack[stack_len] = BoundsFrame {
                node: child,
                next_child: 0,
                bounds: empty_bounds,
            };
            stack_len += 1;
        }
        crate::invariant_failure!("bounds traversal produced no result")
    }

    fn collect_chunks_iterative(
        &self,
        node: NodeId,
        origin: Cell,
        size: Coord,
        clip_bounds: (Coord, Coord, Coord, Coord),
        max_chunks: Option<usize>,
        out: &mut BitGrid,
    ) -> Result<(), GridExtractionError> {
        const MAX_EXTRACTION_STACK: usize = 256;
        let (clip_min_x, clip_min_y, clip_max_x, clip_max_y) = clip_bounds;
        let mut stack = [(0, 0, 0, 0); MAX_EXTRACTION_STACK];
        stack[0] = (node, origin.0, origin.1, size);
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (node, origin_x, origin_y, size) = stack[stack_len];
            let level = self.node_columns.level(node);
            if self.node_columns.population(node) == 0 {
                continue;
            }

            let node_max_x = i128::from(origin_x) + i128::from(size) - 1;
            let node_max_y = i128::from(origin_y) + i128::from(size) - 1;
            if node_max_x < i128::from(clip_min_x)
                || node_max_y < i128::from(clip_min_y)
                || i128::from(origin_x) > i128::from(clip_max_x)
                || i128::from(origin_y) > i128::from(clip_max_y)
            {
                continue;
            }

            if level == 0 {
                let cx = origin_x.div_euclid(CHUNK_SIZE);
                let cy = origin_y.div_euclid(CHUNK_SIZE);
                let lx = origin_x.rem_euclid(CHUNK_SIZE);
                let ly = origin_y.rem_euclid(CHUNK_SIZE);
                let bit = u32::try_from(ly * CHUNK_SIZE + lx)
                    .or_invariant("chunk bit index exceeded u32");
                let is_new_chunk = out.chunk_bits(cx, cy) == 0;
                if is_new_chunk
                    && let Some(limit) = max_chunks
                    && out.chunk_count() >= limit
                {
                    return Err(GridExtractionError::ChunkLimitExceeded {
                        chunks: out.chunk_count() + 1,
                        limit,
                    });
                }
                let bits = out.chunk_bits(cx, cy) | (1_u64 << bit);
                out.try_set_chunk_bits(cx, cy, bits)
                    .map_err(|_| GridExtractionError::AllocationFailed)?;
                continue;
            }

            let half = size / 2;
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            if stack_len + 4 > stack.len() {
                crate::invariant_failure!(
                    "validated extraction depth exceeded fixed traversal workspace"
                );
            }
            for child in [
                (se, origin_x + half, origin_y + half, half),
                (sw, origin_x, origin_y + half, half),
                (ne, origin_x + half, origin_y, half),
                (nw, origin_x, origin_y, half),
            ] {
                stack[stack_len] = child;
                stack_len += 1;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn collect_all_chunks_iterative(
        &self,
        node: NodeId,
        origin_x: Coord,
        origin_y: Coord,
        size: Coord,
        out: &mut BitGrid,
    ) {
        const MAX_EXTRACTION_STACK: usize = 256;
        let mut stack = [(0, 0, 0, 0); MAX_EXTRACTION_STACK];
        stack[0] = (node, origin_x, origin_y, size);
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (node, origin_x, origin_y, size) = stack[stack_len];
            let level = self.node_columns.level(node);
            if self.node_columns.population(node) == 0 {
                continue;
            }
            if level == 0 {
                let cx = origin_x.div_euclid(CHUNK_SIZE);
                let cy = origin_y.div_euclid(CHUNK_SIZE);
                let lx = origin_x.rem_euclid(CHUNK_SIZE);
                let ly = origin_y.rem_euclid(CHUNK_SIZE);
                let bit = u32::try_from(ly * CHUNK_SIZE + lx)
                    .or_invariant("chunk bit index exceeded u32");
                let bits = out.chunk_bits(cx, cy) | (1_u64 << bit);
                out.try_set_chunk_bits(cx, cy, bits)
                    .or_invariant("test extraction allocation failed");
                continue;
            }

            let half = size / 2;
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            if stack_len + 4 > stack.len() {
                crate::invariant_failure!(
                    "validated extraction depth exceeded fixed traversal workspace"
                );
            }
            for child in [
                (se, origin_x + half, origin_y + half, half),
                (sw, origin_x, origin_y + half, half),
                (ne, origin_x + half, origin_y, half),
                (nw, origin_x, origin_y, half),
            ] {
                stack[stack_len] = child;
                stack_len += 1;
            }
        }
    }

    pub(super) fn empty(&mut self, level: u32) -> NodeId {
        while self.empty_by_level.len() <= level as usize {
            let child = *self.empty_by_level.last().or_invariant("required value");
            let node = self.join(child, child, child, child);
            self.empty_by_level.push(node);
        }
        self.empty_by_level[level as usize]
    }

    pub(super) fn join(&mut self, nw: NodeId, ne: NodeId, sw: NodeId, se: NodeId) -> NodeId {
        let level = self.node_columns.level(nw) + 1;
        let key = PackedNodeKey::new(level, [nw, ne, sw, se]);
        if let Some(existing) = self.intern.get(&key) {
            return existing;
        }

        let population = super::PopulationStat::sum([
            self.node_columns.population_stat(nw),
            self.node_columns.population_stat(ne),
            self.node_columns.population_stat(sw),
            self.node_columns.population_stat(se),
        ]);

        if !self.prepare_mandatory_node_growth() {
            return self.dead_leaf;
        }

        let node_id = self.push_node(level, population, nw, ne, sw, se);
        self.intern
            .try_insert(key, node_id)
            .or_invariant("reserved HashLife structural index insertion failed");
        node_id
    }

    pub(super) fn intern_leaf(&mut self, alive: bool) -> NodeId {
        let key = HashLifeEngine::packed_leaf_key(alive);
        if let Some(existing) = self.intern.get(&key) {
            return existing;
        }

        if !self.prepare_mandatory_node_growth() {
            return self.dead_leaf;
        }
        let node_id = NodeId::try_from(self.node_count())
            .or_invariant("HashLife node arena exceeded u32 capacity");
        let node_id = self.push_node(
            0,
            super::PopulationStat::exact(u128::from(alive)),
            node_id,
            node_id,
            node_id,
            node_id,
        );
        self.intern
            .try_insert(key, node_id)
            .or_invariant("reserved HashLife leaf index insertion failed");
        node_id
    }
}

fn merge_child_bounds(
    bounds: &mut RelativeBounds,
    child: RelativeBounds,
    child_index: usize,
    parent_level: u32,
) {
    let half = 1_i64 << (parent_level - 1);
    let offset_x = if child_index.is_multiple_of(2) {
        0
    } else {
        half
    };
    let offset_y = if child_index < 2 { 0 } else { half };
    bounds.min_x = bounds.min_x.min(child.min_x + offset_x);
    bounds.min_y = bounds.min_y.min(child.min_y + offset_y);
    bounds.max_x = bounds.max_x.max(child.max_x + offset_x);
    bounds.max_y = bounds.max_y.max(child.max_y + offset_y);
}

struct ExtractionLimits {
    clip_bounds: (Coord, Coord, Coord, Coord),
    max_chunks: Option<usize>,
}

fn extraction_limits(
    engine: &mut HashLifeEngine,
    node: NodeId,
    offset_x: Coord,
    offset_y: Coord,
    policy: GridExtractionPolicy,
) -> Result<ExtractionLimits, GridExtractionError> {
    match policy {
        GridExtractionPolicy::ViewportOnly => Ok(ExtractionLimits {
            clip_bounds: (offset_x, offset_y, offset_x - 1, offset_y - 1),
            max_chunks: Some(0),
        }),
        GridExtractionPolicy::BoundedRegion {
            min_x,
            min_y,
            max_x,
            max_y,
        } => Ok(ExtractionLimits {
            clip_bounds: (min_x, min_y, max_x, max_y),
            max_chunks: None,
        }),
        GridExtractionPolicy::FullGridIfUnder {
            max_population,
            max_chunks,
            max_bounds_span,
        } => {
            let population = engine.node_columns.population(node);
            if population == 0 {
                return Ok(ExtractionLimits {
                    clip_bounds: (0, 0, -1, -1),
                    max_chunks: Some(max_chunks),
                });
            }
            if population > max_population {
                return Err(GridExtractionError::PopulationLimitExceeded {
                    population,
                    limit: max_population,
                });
            }
            let bounds = engine
                .node_bounds(node, offset_x, offset_y)
                .or_invariant("non-empty node should have bounds");
            let (min_x, min_y, max_x, max_y) = bounds;
            let bounds_span = (max_x - min_x + 1).max(max_y - min_y + 1);
            if bounds_span > max_bounds_span {
                return Err(GridExtractionError::BoundsSpanLimitExceeded {
                    bounds_span,
                    limit: max_bounds_span,
                });
            }
            Ok(ExtractionLimits {
                clip_bounds: bounds,
                max_chunks: Some(max_chunks),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressured_shell_and_bounds_publication_preserve_exact_results() {
        let mut engine = HashLifeEngine::default();
        engine.begin_allocation_transaction(u128::MAX);
        let source = engine.join(
            engine.live_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
        );
        let expected_shell = engine.centered_shell(source);
        assert_eq!(engine.take_allocation_failure(), None);

        engine.result_caches.shells.release_storage();
        engine.result_caches.bounds.release_storage();
        let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);
        let shell = engine.centered_shell(source);
        let bounds = engine.node_relative_bounds(source);

        assert_eq!(engine.take_allocation_failure(), None);
        assert_eq!(shell, expected_shell);
        assert_eq!(
            bounds,
            Some(RelativeBounds {
                min_x: 0,
                min_y: 0,
                max_x: 0,
                max_y: 0,
            })
        );
        assert_eq!(engine.result_caches.shells.len(), 0);
        assert_eq!(engine.result_caches.bounds.len(), 0);
    }
}
