use super::{HashLifeEngine, Node, NodeId, NodeKey, base_transitions};
use crate::bitgrid::{BitGrid, Cell, Coord};

impl HashLifeEngine {
    pub(super) fn base_transition(&mut self, node: NodeId) -> NodeId {
        let node_ref = &self.nodes[node as usize];
        let mask = self.level1_to_4x4_mask(node_ref.nw, 0, 0)
            | self.level1_to_4x4_mask(node_ref.ne, 2, 0)
            | self.level1_to_4x4_mask(node_ref.sw, 0, 2)
            | self.level1_to_4x4_mask(node_ref.se, 2, 2);
        let centered = base_transitions()[mask as usize];
        let leaves = [self.dead_leaf, self.live_leaf];
        let nw = leaves[(centered & 0b0001 != 0) as usize];
        let ne = leaves[((centered >> 1) & 0b0001) as usize];
        let sw = leaves[((centered >> 2) & 0b0001) as usize];
        let se = leaves[((centered >> 3) & 0b0001) as usize];
        self.join(nw, ne, sw, se)
    }

    fn level1_to_4x4_mask(&self, node: NodeId, base_x: u16, base_y: u16) -> u16 {
        let node_ref = &self.nodes[node as usize];
        debug_assert_eq!(node_ref.level, 1);
        (u16::from(self.nodes[node_ref.nw as usize].population != 0) << (base_y * 4 + base_x))
            | (u16::from(self.nodes[node_ref.ne as usize].population != 0)
                << (base_y * 4 + base_x + 1))
            | (u16::from(self.nodes[node_ref.sw as usize].population != 0)
                << ((base_y + 1) * 4 + base_x))
            | (u16::from(self.nodes[node_ref.se as usize].population != 0)
                << ((base_y + 1) * 4 + base_x + 1))
    }

    pub(super) fn node_to_grid(&self, node: NodeId, offset_x: Coord, offset_y: Coord) -> BitGrid {
        let size = 1_i64 << self.nodes[node as usize].level;
        let mut cells = Vec::with_capacity(self.nodes[node as usize].population as usize);
        self.collect_cells_iterative(node, offset_x, offset_y, size, &mut cells);
        BitGrid::from_cells(&cells)
    }

    pub(super) fn node_bounds(
        &self,
        node: NodeId,
        origin_x: Coord,
        origin_y: Coord,
    ) -> Option<(Coord, Coord, Coord, Coord)> {
        if self.nodes[node as usize].population == 0 {
            return None;
        }

        let mut stack = Vec::with_capacity(self.nodes[node as usize].level as usize + 1);
        let size = 1_i64 << self.nodes[node as usize].level;
        stack.push((node, origin_x, origin_y, size));
        let mut min_x = Coord::MAX;
        let mut min_y = Coord::MAX;
        let mut max_x = Coord::MIN;
        let mut max_y = Coord::MIN;

        while let Some((node, origin_x, origin_y, size)) = stack.pop() {
            let node_ref = &self.nodes[node as usize];
            if node_ref.population == 0 {
                continue;
            }
            if node_ref.level == 0 {
                min_x = min_x.min(origin_x);
                min_y = min_y.min(origin_y);
                max_x = max_x.max(origin_x);
                max_y = max_y.max(origin_y);
                continue;
            }

            let half = size / 2;
            stack.push((node_ref.se, origin_x + half, origin_y + half, half));
            stack.push((node_ref.sw, origin_x, origin_y + half, half));
            stack.push((node_ref.ne, origin_x + half, origin_y, half));
            stack.push((node_ref.nw, origin_x, origin_y, half));
        }

        Some((min_x, min_y, max_x, max_y))
    }

    pub(super) fn collect_cells_iterative(
        &self,
        node: NodeId,
        origin_x: Coord,
        origin_y: Coord,
        size: Coord,
        out: &mut Vec<Cell>,
    ) {
        let mut stack = Vec::with_capacity(self.nodes[node as usize].level as usize + 1);
        stack.push((node, origin_x, origin_y, size));
        while let Some((node, origin_x, origin_y, size)) = stack.pop() {
            let node_ref = &self.nodes[node as usize];
            if node_ref.population == 0 {
                continue;
            }
            if node_ref.level == 0 {
                out.push((origin_x, origin_y));
                continue;
            }

            let half = size / 2;
            stack.push((node_ref.se, origin_x + half, origin_y + half, half));
            stack.push((node_ref.sw, origin_x, origin_y + half, half));
            stack.push((node_ref.ne, origin_x + half, origin_y, half));
            stack.push((node_ref.nw, origin_x, origin_y, half));
        }
    }

    pub(super) fn empty(&mut self, level: u32) -> NodeId {
        while self.empty_by_level.len() <= level as usize {
            let child = *self.empty_by_level.last().unwrap();
            let node = self.join(child, child, child, child);
            self.empty_by_level.push(node);
        }
        self.empty_by_level[level as usize]
    }

    pub(super) fn join(&mut self, nw: NodeId, ne: NodeId, sw: NodeId, se: NodeId) -> NodeId {
        let level = self.nodes[nw as usize].level + 1;
        let key = NodeKey::Internal {
            level,
            nw,
            ne,
            sw,
            se,
        };
        if let Some(&existing) = self.intern.get(&key) {
            return existing;
        }

        let population = self.nodes[nw as usize].population
            + self.nodes[ne as usize].population
            + self.nodes[sw as usize].population
            + self.nodes[se as usize].population;

        let node_id = self.nodes.len() as NodeId;
        self.nodes.push(Node {
            level,
            population,
            nw,
            ne,
            sw,
            se,
        });
        self.intern.insert(key, node_id);
        node_id
    }

    pub(super) fn intern_leaf(&mut self, alive: bool) -> NodeId {
        let key = NodeKey::Leaf(alive);
        if let Some(&existing) = self.intern.get(&key) {
            return existing;
        }

        let node_id = self.nodes.len() as NodeId;
        self.nodes.push(Node {
            level: 0,
            population: u64::from(alive),
            nw: node_id,
            ne: node_id,
            sw: node_id,
            se: node_id,
        });
        self.intern.insert(key, node_id);
        node_id
    }
}
