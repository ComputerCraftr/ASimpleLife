use super::{HashLifeOracle, Node, NodeId, NodeKey, base_transitions};
use crate::bitgrid::BitGrid;

impl HashLifeOracle {
    pub(super) fn base_transition(&mut self, node: NodeId) -> NodeId {
        let mut mask = 0_u16;
        let mut live = Vec::with_capacity(self.nodes[node as usize].population as usize);
        self.collect_cells_iterative(node, 0, 0, 4, &mut live);
        for (x, y) in live {
            let bit = (y * 4 + x) as u16;
            mask |= 1 << bit;
        }

        let centered = base_transitions()[mask as usize];
        let nw = if centered & 0b0001 != 0 {
            self.live_leaf
        } else {
            self.dead_leaf
        };
        let ne = if centered & 0b0010 != 0 {
            self.live_leaf
        } else {
            self.dead_leaf
        };
        let sw = if centered & 0b0100 != 0 {
            self.live_leaf
        } else {
            self.dead_leaf
        };
        let se = if centered & 0b1000 != 0 {
            self.live_leaf
        } else {
            self.dead_leaf
        };
        self.join(nw, ne, sw, se)
    }

    pub(super) fn node_to_grid(&self, node: NodeId, offset_x: i32, offset_y: i32) -> BitGrid {
        let size = 1_i32 << self.nodes[node as usize].level;
        let mut cells = Vec::with_capacity(self.nodes[node as usize].population as usize);
        self.collect_cells_iterative(node, offset_x, offset_y, size, &mut cells);
        BitGrid::from_cells(&cells)
    }

    pub(super) fn collect_cells_iterative(
        &self,
        node: NodeId,
        origin_x: i32,
        origin_y: i32,
        size: i32,
        out: &mut Vec<(i32, i32)>,
    ) {
        let mut stack = vec![(node, origin_x, origin_y, size)];
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
