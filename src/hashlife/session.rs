#[cfg(test)]
use super::HashLifeRuntimeStats;
use super::{
    GridExtractionError, GridExtractionPolicy, HashLifeCheckpoint, HashLifeEngine, NodeId,
};
use crate::bitgrid::{BitGrid, Coord};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionAdvanceStats {
    pub generations: u64,
}

#[derive(Debug, Default)]
pub struct HashLifeSession {
    engine: HashLifeEngine,
    active_run: bool,
    previous_root: Option<NodeId>,
    current_root: Option<NodeId>,
    current_origin_x: Coord,
    current_origin_y: Coord,
    current_generation: u64,
    root_is_centered: bool,
    sampled_bounds: Option<Option<(Coord, Coord, Coord, Coord)>>,
    sampled_checkpoint: Option<Option<HashLifeCheckpoint>>,
    #[cfg(test)]
    sample_materializations: usize,
}

impl HashLifeSession {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_active_run(&mut self) {
        if self.active_run {
            return;
        }
        self.previous_root = self.engine.begin_persistent_run();
        self.active_run = true;
    }

    fn finish_active_run(&mut self) {
        if !self.active_run {
            return;
        }
        self.engine
            .finish_persistent_run(self.previous_root, self.current_root);
        self.active_run = false;
        self.previous_root = None;
    }

    pub fn load_grid(&mut self, grid: &BitGrid) {
        if self.current_root.is_some() || self.active_run {
            self.finish_active_run();
        }
        self.ensure_active_run();
        let (root, origin_x, origin_y) = self.engine.embed_grid_state(grid);
        self.current_root = Some(root);
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = 0;
        self.root_is_centered = false;
        self.sampled_bounds = Some(grid.bounds());
        self.sampled_checkpoint = None;
    }

    pub fn generation(&self) -> u64 {
        self.current_generation
    }

    pub fn is_loaded(&self) -> bool {
        self.current_root.is_some()
    }

    pub fn population(&self) -> Option<u64> {
        self.current_root
            .map(|root| self.engine.nodes[root as usize].population)
    }

    pub fn origin(&self) -> Option<(Coord, Coord)> {
        self.current_root
            .map(|_| (self.current_origin_x, self.current_origin_y))
    }

    pub fn bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)> {
        if let Some(bounds) = self.sampled_bounds {
            return bounds;
        }
        let root = self.current_root?;
        let bounds = self
            .engine
            .node_bounds(root, self.current_origin_x, self.current_origin_y);
        self.sampled_bounds = Some(bounds);
        bounds
    }

    pub fn shift_origin(&mut self, dx: Coord, dy: Coord) {
        if self.current_root.is_none() {
            return;
        }
        self.current_origin_x = self
            .current_origin_x
            .checked_add(dx)
            .expect("hashlife origin x overflow");
        self.current_origin_y = self
            .current_origin_y
            .checked_add(dy)
            .expect("hashlife origin y overflow");
        self.clear_cached_samples();
    }

    pub fn advance_root(&mut self, generations: u64) -> SessionAdvanceStats {
        self.ensure_active_run();
        let Some(_) = self.current_root else {
            return SessionAdvanceStats::default();
        };
        let mut remaining = generations;
        while remaining != 0 {
            let desired_step_exp = 63 - remaining.leading_zeros();
            self.ensure_centered_capacity(desired_step_exp);
            let root = self
                .current_root
                .expect("hashlife session root disappeared");
            let level = self.engine.nodes[root as usize].level;
            let step_exp = desired_step_exp.min(level.saturating_sub(2));
            let step = 1_u64 << step_exp;
            let root_size = 1_i64 << level;
            let advanced = self.engine.advance_pow2(root, step_exp);
            self.current_origin_x = self
                .current_origin_x
                .checked_add(root_size / 4)
                .expect("hashlife origin x overflow");
            self.current_origin_y = self
                .current_origin_y
                .checked_add(root_size / 4)
                .expect("hashlife origin y overflow");
            self.current_root = Some(advanced);
            self.current_generation = self.current_generation.saturating_add(step);
            self.root_is_centered = false;
            self.clear_cached_samples();
            self.current_root = self.engine.maybe_collect_active_run(self.current_root);
            remaining -= step;
        }
        SessionAdvanceStats { generations }
    }

    fn ensure_centered_capacity(&mut self, desired_step_exp: u32) {
        let Some(mut root) = self.current_root else {
            return;
        };

        loop {
            let level = self.engine.nodes[root as usize].level;
            let needs_expansion = !self.root_is_centered || level < desired_step_exp + 2;
            if !needs_expansion {
                break;
            }
            root = self.center_expand_root(root);
            self.current_root = Some(root);
            self.root_is_centered = true;
            self.clear_cached_samples();
        }
    }

    fn center_expand_root(&mut self, root: NodeId) -> NodeId {
        let node = self.engine.nodes[root as usize];
        if node.level == 0 {
            let empty = self.engine.dead_leaf;
            let nw = self.engine.join(empty, empty, empty, root);
            let ne = self.engine.join(empty, empty, empty, empty);
            let sw = self.engine.join(empty, empty, empty, empty);
            let se = self.engine.join(empty, empty, empty, empty);
            self.current_origin_x = self
                .current_origin_x
                .checked_sub(1)
                .expect("hashlife centered expansion x overflow");
            self.current_origin_y = self
                .current_origin_y
                .checked_sub(1)
                .expect("hashlife centered expansion y overflow");
            return self.engine.join(nw, ne, sw, se);
        }

        let empty = self.engine.empty(node.level - 1);
        let upper_left = self.engine.join(empty, empty, empty, node.nw);
        let upper_right = self.engine.join(empty, empty, node.ne, empty);
        let lower_left = self.engine.join(empty, node.sw, empty, empty);
        let lower_right = self.engine.join(node.se, empty, empty, empty);
        let child_size = 1_i64 << (node.level - 1);
        self.current_origin_x = self
            .current_origin_x
            .checked_sub(child_size)
            .expect("hashlife centered expansion x overflow");
        self.current_origin_y = self
            .current_origin_y
            .checked_sub(child_size)
            .expect("hashlife centered expansion y overflow");
        self.engine
            .join(upper_left, upper_right, lower_left, lower_right)
    }

    pub fn signature_checkpoint(&mut self) -> Option<&HashLifeCheckpoint> {
        if self.sampled_checkpoint.is_none() {
            let root = self.current_root?;
            let checkpoint = self.engine.node_checkpoint(
                root,
                self.current_origin_x,
                self.current_origin_y,
                self.current_generation,
            );
            if let Some(checkpoint) = checkpoint.as_ref() {
                self.sampled_bounds = Some(Some(checkpoint.bounds));
            } else {
                self.sampled_bounds = Some(None);
            }
            self.sampled_checkpoint = Some(checkpoint);
        }
        self.sampled_checkpoint.as_ref().and_then(Option::as_ref)
    }

    pub fn sample_grid(&mut self) -> Option<BitGrid> {
        self.extract_grid(unrestricted_debug_full_grid_policy())
            .ok()
    }

    pub fn extract_grid(
        &mut self,
        policy: GridExtractionPolicy,
    ) -> Result<BitGrid, GridExtractionError> {
        let root = self.current_root.ok_or(GridExtractionError::NotLoaded)?;
        #[cfg(test)]
        {
            self.sample_materializations += 1;
        }
        self.engine
            .node_to_grid(root, self.current_origin_x, self.current_origin_y, policy)
    }

    pub fn sample_region(
        &mut self,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Option<BitGrid> {
        let root = self.current_root?;
        Some(self.engine.node_to_grid_clipped(
            root,
            self.current_origin_x,
            self.current_origin_y,
            (min_x, min_y, max_x, max_y),
        ))
    }

    pub fn finish(&mut self) {
        self.finish_active_run();
        self.current_root = None;
        self.current_generation = 0;
        self.root_is_centered = false;
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn runtime_stats(&self) -> HashLifeRuntimeStats {
        self.engine.runtime_stats()
    }

    #[cfg(test)]
    pub(crate) fn sample_materializations(&self) -> usize {
        self.sample_materializations
    }

    fn clear_cached_samples(&mut self) {
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
    }
}

fn unrestricted_debug_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: u64::MAX,
        max_chunks: usize::MAX,
        max_bounds_span: Coord::MAX,
    }
}

impl Drop for HashLifeSession {
    fn drop(&mut self) {
        self.finish();
    }
}
