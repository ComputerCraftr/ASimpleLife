use super::{
    GridExtractionError, GridExtractionPolicy, HashLifeCheckpoint, HashLifeCheckpointSignature,
    HashLifeEngine, Node, NodeId, NodeKey, base_transitions,
};
use crate::bitgrid::{BitGrid, CHUNK_SIZE, Cell, Coord};
use crate::life::step_grid_with_changes_and_memo;
use crate::memo::Memo;
use std::collections::HashMap;

impl HashLifeEngine {
    pub(super) fn dense_advance_centered(&mut self, node: NodeId, step_exp: u32) -> NodeId {
        let level = self.nodes[node as usize].level;
        debug_assert!(level >= 2);
        let size = 1_i64 << level;
        let result_size = size / 2;
        let center_origin = size / 4;
        let generations = 1_u64 << step_exp;
        let mut grid = self
            .node_to_grid(
                node,
                0,
                0,
                GridExtractionPolicy::FullGridIfUnder {
                    max_population: u64::MAX,
                    max_chunks: usize::MAX,
                    max_bounds_span: Coord::MAX,
                },
            )
            .expect("dense HashLife shortcut extraction should be unrestricted");
        let mut memo = Memo::default();
        for _ in 0..generations {
            grid = step_grid_with_changes_and_memo(&grid, &mut memo).0;
        }

        let mut centered = BitGrid::empty();
        let max_x = center_origin + result_size;
        let max_y = center_origin + result_size;
        for (x, y) in grid.live_cells() {
            if x >= center_origin && x < max_x && y >= center_origin && y < max_y {
                centered.set(x - center_origin, y - center_origin, true);
            }
        }
        self.embed_grid_at_level(&centered, level - 1)
    }

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

    pub(super) fn node_to_grid(
        &self,
        node: NodeId,
        offset_x: Coord,
        offset_y: Coord,
        policy: GridExtractionPolicy,
    ) -> Result<BitGrid, GridExtractionError> {
        let size = 1_i64 << self.nodes[node as usize].level;
        let limits = extraction_limits(self, node, offset_x, offset_y, policy)?;
        let mut chunks = HashMap::new();
        self.collect_chunks_iterative(
            node,
            (offset_x, offset_y),
            size,
            limits.clip_bounds,
            limits.max_chunks,
            &mut chunks,
        )?;
        Ok(BitGrid::from_chunk_bits_map(chunks))
    }

    pub(super) fn node_to_grid_clipped(
        &self,
        node: NodeId,
        offset_x: Coord,
        offset_y: Coord,
        clip_bounds: (Coord, Coord, Coord, Coord),
    ) -> BitGrid {
        let size = 1_i64 << self.nodes[node as usize].level;
        let mut chunks = HashMap::new();
        self.collect_chunks_iterative(
            node,
            (offset_x, offset_y),
            size,
            clip_bounds,
            None,
            &mut chunks,
        )
        .expect("clipped extraction should not enforce chunk limits");
        BitGrid::from_chunk_bits_map(chunks)
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

        (min_x != Coord::MAX).then_some((min_x, min_y, max_x, max_y))
    }

    pub(super) fn node_checkpoint(
        &self,
        node: NodeId,
        origin_x: Coord,
        origin_y: Coord,
        generation: u64,
    ) -> Option<HashLifeCheckpoint> {
        let bounds = self.node_bounds(node, origin_x, origin_y)?;
        let (min_x, min_y, max_x, max_y) = bounds;
        let width = max_x - min_x + 1;
        let height = max_y - min_y + 1;
        let population = self.nodes[node as usize].population;
        if population > 250_000 || width.max(height) > 65_536 {
            return None;
        }
        let size = 1_i64 << self.nodes[node as usize].level;
        let mut cells = Vec::with_capacity(
            usize::try_from(population).expect("hashlife population exceeded usize"),
        );
        self.collect_cells_iterative(node, origin_x, origin_y, size, &mut cells);
        for cell in &mut cells {
            cell.0 -= min_x;
            cell.1 -= min_y;
        }
        cells.sort_unstable();

        Some(HashLifeCheckpoint {
            generation,
            origin: (min_x, min_y),
            signature: HashLifeCheckpointSignature {
                width,
                height,
                cells,
            },
            population,
            bounds,
            bounds_span: width.max(height),
        })
    }

    fn collect_chunks_iterative(
        &self,
        node: NodeId,
        origin: Cell,
        size: Coord,
        clip_bounds: (Coord, Coord, Coord, Coord),
        max_chunks: Option<usize>,
        out: &mut HashMap<Cell, u64>,
    ) -> Result<(), GridExtractionError> {
        let (clip_min_x, clip_min_y, clip_max_x, clip_max_y) = clip_bounds;
        let mut stack = Vec::with_capacity(self.nodes[node as usize].level as usize + 1);
        stack.push((node, origin.0, origin.1, size));
        while let Some((node, origin_x, origin_y, size)) = stack.pop() {
            let node_ref = &self.nodes[node as usize];
            if node_ref.population == 0 {
                continue;
            }

            let node_max_x = origin_x + size - 1;
            let node_max_y = origin_y + size - 1;
            if node_max_x < clip_min_x
                || node_max_y < clip_min_y
                || origin_x > clip_max_x
                || origin_y > clip_max_y
            {
                continue;
            }

            if node_ref.level == 0 {
                let cx = origin_x.div_euclid(CHUNK_SIZE);
                let cy = origin_y.div_euclid(CHUNK_SIZE);
                let lx = origin_x.rem_euclid(CHUNK_SIZE);
                let ly = origin_y.rem_euclid(CHUNK_SIZE);
                let bit =
                    u32::try_from(ly * CHUNK_SIZE + lx).expect("chunk bit index exceeded u32");
                let is_new_chunk = !out.contains_key(&(cx, cy));
                if is_new_chunk
                    && let Some(limit) = max_chunks
                    && out.len() >= limit
                {
                    return Err(GridExtractionError::ChunkLimitExceeded {
                        chunks: out.len() + 1,
                        limit,
                    });
                }
                let entry = out.entry((cx, cy)).or_insert(0);
                *entry |= 1_u64 << bit;
                continue;
            }

            let half = size / 2;
            stack.push((node_ref.se, origin_x + half, origin_y + half, half));
            stack.push((node_ref.sw, origin_x, origin_y + half, half));
            stack.push((node_ref.ne, origin_x + half, origin_y, half));
            stack.push((node_ref.nw, origin_x, origin_y, half));
        }
        Ok(())
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

struct ExtractionLimits {
    clip_bounds: (Coord, Coord, Coord, Coord),
    max_chunks: Option<usize>,
}

fn extraction_limits(
    engine: &HashLifeEngine,
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
            let population = engine.nodes[node as usize].population;
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
                .expect("non-empty node should have bounds");
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
