use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Coord};
use crate::flat_table::FlatKey;
use crate::symmetry::D4Symmetry as Symmetry;

use super::{
    CanonicalJumpKey, CanonicalNodeIdentity, CanonicalNodeRef, CanonicalStructKey,
    DiagnosticCacheRates, DiagnosticCacheSizes, DiagnosticFallbacks, DiagnosticGc,
    DiagnosticMaterializations, DiagnosticOverview, DiagnosticTransforms, DiagnosticWorkRates,
    HashLifeDiagnosticSummary, HashLifeEngine, HashLifeRuntimeStats, NodeId, PackedNodeKey,
    PackedSymmetryKey, Phase2CommitLane, RuntimeEngineStats,
};

mod cache_stats;
mod canonical;
mod jump;
mod overlap;

trait EmbedGridProbeExt {
    fn embed_grid_state(&mut self, grid: &BitGrid) -> (NodeId, Coord, Coord);
}

impl EmbedGridProbeExt for HashLifeEngine {
    fn embed_grid_state(&mut self, grid: &BitGrid) -> (NodeId, Coord, Coord) {
        self.try_embed_grid_state(grid)
            .or_invariant("test grid geometry must be representable")
    }
}
