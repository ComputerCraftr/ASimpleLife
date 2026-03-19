use super::{
    GridExtractionError, GridExtractionPolicy, HASHLIFE_FULL_GRID_MAX_CHUNKS,
    HASHLIFE_FULL_GRID_MAX_POPULATION, HashLifeCheckpoint, HashLifeEngine,
    HashLifeSnapshotError, NodeId,
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
        self.sampled_bounds = Some(grid.bounds());
        self.sampled_checkpoint = None;
    }

    pub fn load_snapshot_string(&mut self, snapshot: &str) -> Result<(), HashLifeSnapshotError> {
        if self.current_root.is_some() || self.active_run {
            self.finish_active_run();
        }
        self.ensure_active_run();
        let (root, origin_x, origin_y, generation) =
            self.engine.import_snapshot_string(snapshot)?;
        self.current_root = Some(root);
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = generation;
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
        Ok(())
    }

    pub fn generation(&self) -> u64 {
        self.current_generation
    }

    pub fn is_loaded(&self) -> bool {
        self.current_root.is_some()
    }

    pub fn population(&self) -> Option<u64> {
        self.current_root
            .map(|root| self.engine.node_columns.population(root))
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
        let Some(root) = self.current_root else {
            return SessionAdvanceStats::default();
        };
        if generations == 0 {
            return SessionAdvanceStats::default();
        }

        let current = match self
            .engine
            .node_bounds(root, self.current_origin_x, self.current_origin_y)
        {
            Some((min_x, min_y, max_x, max_y)) => {
                self.engine.stats.session_bounded_grid_extractions += 1;
                self.engine
                    .node_to_grid(
                        root,
                        self.current_origin_x,
                        self.current_origin_y,
                        GridExtractionPolicy::BoundedRegion {
                            min_x,
                            min_y,
                            max_x,
                            max_y,
                        },
                    )
                    .expect("hashlife bounded session advance extraction should succeed")
            }
            None => BitGrid::empty(),
        };
        let (advanced, _) = self.engine.advance_segment(&current, generations);
        let (next_root, origin_x, origin_y) = self.engine.embed_grid_state(&advanced);
        self.current_root = Some(next_root);
        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = self.current_generation.saturating_add(generations);
        self.clear_cached_samples();
        self.current_root = self.engine.maybe_collect_active_run(self.current_root);
        SessionAdvanceStats { generations }
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
            }
            self.sampled_checkpoint = Some(checkpoint);
        }
        self.sampled_checkpoint.as_ref().and_then(Option::as_ref)
    }

    pub fn sample_grid(&mut self) -> Option<BitGrid> {
        self.extract_grid(default_full_grid_policy()).ok()
    }

    pub fn export_snapshot_string(&mut self) -> Option<String> {
        let root = self.current_root?;
        Some(self.engine.export_snapshot_string(
            root,
            self.current_origin_x,
            self.current_origin_y,
            self.current_generation,
        ))
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
        let grid = self
            .engine
            .node_to_grid(root, self.current_origin_x, self.current_origin_y, policy)?;
        if matches!(policy, GridExtractionPolicy::FullGridIfUnder { .. }) {
            self.engine.stats.session_full_grid_materializations += 1;
        }
        Ok(grid)
    }

    pub fn sample_region(
        &mut self,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Option<BitGrid> {
        let root = self.current_root?;
        self.engine.stats.clipped_viewport_extractions += 1;
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
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
    }

    #[cfg(test)]
    pub(crate) fn sample_materializations(&self) -> usize {
        self.sample_materializations
    }

    pub(crate) fn record_oracle_confirmation_materialization(&mut self) {
        self.engine.stats.oracle_confirmation_materializations += 1;
    }

    fn clear_cached_samples(&mut self) {
        self.sampled_bounds = None;
        self.sampled_checkpoint = None;
    }
}

fn default_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: HASHLIFE_FULL_GRID_MAX_POPULATION,
        max_chunks: HASHLIFE_FULL_GRID_MAX_CHUNKS,
        max_bounds_span: Coord::MAX,
    }
}

impl Drop for HashLifeSession {
    fn drop(&mut self) {
        self.finish();
    }
}
