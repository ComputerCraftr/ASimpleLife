use crate::RequiredExt;
#[cfg(test)]
use crate::bitgrid::BitGrid;
use crate::bitgrid::Coord;
use crate::cache_policy::should_run_active_hashlife_gc;
use crate::hashing::{
    StructuralFingerprint, hash_packed_jump_fingerprint, hash_packed_node_fingerprint,
    hash_u64_words_with_level, hash_u64_words_with_level_batch, hash_words, mix64,
    structural_leaf_fingerprint, structural_node_fingerprint,
};
use crate::probe_table::{ProbeKey, ProbeMode, ProbeTable};
use crate::simd_layout::{
    AlignedU32Batch, AlignedU64WordBatch4, AlignedU64WordBatch9, SIMD_BATCH_LANES,
};
use crate::symmetry::D4Symmetry as Symmetry;
use bytemuck::must_cast;
use std::collections::HashMap;
use wide::u64x8;

mod advance;
mod arena;
mod cache_lifecycle;
mod canonical;
mod embed;
mod future;
mod gc;
mod geometry;
mod handles;
mod kernels;
mod memory;
mod node;
mod population;
mod scheduler;
pub(crate) mod session;
mod session_types;
mod signature;
mod simd;
mod snapshot;
mod stats;
#[cfg(test)]
mod test_probes;

pub use geometry::{HashLifeGeometryError, MAX_COORD_ROOT_LEVEL};
pub use memory::{HashLifeAllocationClass, HashLifeAllocationFailure};
pub use population::PopulationCount;
pub use session::HashLifeSession;
pub(crate) use session_types::HashLifeMaterializationError;
pub use session_types::{
    HashLifeAdvanceError, HashLifeConversionError, HashLifeExecutionStats, HashLifeLimits,
    SessionAdvanceStats,
};
pub use signature::{HashLifeStateCheckpoint, HashLifeStateIdentity};
pub use snapshot::{
    HashLifeSnapshotError, OwnedHashLifeSnapshot,
    deserialize_from_reader as deserialize_snapshot_from_reader,
    deserialize_to_grid as deserialize_snapshot_to_grid, serialize_grid as serialize_grid_snapshot,
    serialize_grid_to_writer as serialize_grid_snapshot_to_writer,
};
use stats::*;
#[cfg(test)]
pub(crate) use stats::{
    DiagnosticCacheRates, DiagnosticCacheSizes, DiagnosticFallbacks, DiagnosticGc,
    DiagnosticMaterializations, DiagnosticOverview, DiagnosticTransforms, DiagnosticWorkRates,
    HashLifeDiagnosticSummary, HashLifeRuntimeStats, RuntimeEngineStats,
};
pub use stats::{
    HASHLIFE_CHECKPOINT_MAX_POPULATION, HASHLIFE_FULL_GRID_MAX_CHUNKS,
    HASHLIFE_FULL_GRID_MAX_POPULATION,
};

use handles::{CanonicalShapeId, NodeId, PackedTransformId};
type CanonicalNodeRef = CanonicalShapeId;
use arena::NodeColumns;
use memory::{EngineAllocationFailure, EngineIdCapacity};
use population::PopulationStat;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridExtractionPolicy {
    ViewportOnly,
    BoundedRegion {
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    },
    FullGridIfUnder {
        max_population: u128,
        max_chunks: usize,
        max_bounds_span: Coord,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridExtractionError {
    NotLoaded,
    AllocationFailed,
    PopulationLimitExceeded { population: u128, limit: u128 },
    ChunkLimitExceeded { chunks: usize, limit: usize },
    BoundsSpanLimitExceeded { bounds_span: Coord, limit: Coord },
}

#[derive(Clone, Copy, Debug)]
struct CanonicalJumpKey {
    structural: CanonicalStructKey,
    step_exp: u32,
    symmetry_admitted: bool,
}

impl PartialEq for CanonicalJumpKey {
    fn eq(&self, other: &Self) -> bool {
        self.step_exp == other.step_exp
            && self.symmetry_admitted == other.symmetry_admitted
            && self.structural == other.structural
    }
}

impl Eq for CanonicalJumpKey {}

impl CanonicalJumpKey {
    fn empty() -> Self {
        Self {
            structural: CanonicalStructKey::leaf(false),
            step_exp: 0,
            symmetry_admitted: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalStructKey {
    level: u32,
    children: [CanonicalNodeRef; 4],
    fingerprint: StructuralFingerprint,
}

impl CanonicalStructKey {
    fn leaf(alive: bool) -> Self {
        Self {
            level: 0,
            children: [
                if alive {
                    CanonicalShapeId::LIVE
                } else {
                    CanonicalShapeId::DEAD
                },
                CanonicalShapeId::DEAD,
                CanonicalShapeId::DEAD,
                CanonicalShapeId::DEAD,
            ],
            fingerprint: structural_leaf_fingerprint(alive),
        }
    }

    #[cfg(test)]
    fn synthetic(level: u32, children: [CanonicalNodeRef; 4]) -> Self {
        let child_fingerprints = children.map(|child| {
            structural_node_fingerprint(
                0,
                [
                    structural_leaf_fingerprint(child.raw() & 1 != 0),
                    structural_leaf_fingerprint(child.raw() & 2 != 0),
                    structural_leaf_fingerprint(child.raw() & 4 != 0),
                    structural_leaf_fingerprint(child.raw() & 8 != 0),
                ],
            )
        });
        Self {
            level,
            children,
            fingerprint: structural_node_fingerprint(level, child_fingerprints),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SemanticPrefix {
    words: [u64; 2],
    bit_len: u16,
    complete: bool,
}

impl SemanticPrefix {
    const LIMIT: usize = 128;

    fn leaf(alive: bool) -> Self {
        Self {
            words: [u64::from(alive) << 63, 0],
            bit_len: 1,
            complete: true,
        }
    }

    fn parent(children: [Self; 4]) -> Self {
        let mut result = Self::default();
        let mut output_bit = 0_usize;
        for child in children {
            let child_len = usize::from(child.bit_len);
            for child_bit in 0..child_len {
                if output_bit == Self::LIMIT {
                    break;
                }
                let bit = (child.words[child_bit / 64] >> (63 - child_bit % 64)) & 1;
                result.words[output_bit / 64] |= bit << (63 - output_bit % 64);
                output_bit += 1;
            }
            if output_bit == Self::LIMIT {
                break;
            }
        }
        result.bit_len = u16::try_from(output_bit).or_invariant("semantic prefix exceeds 128 bits");
        result.complete = children.iter().all(|child| child.complete)
            && children
                .iter()
                .map(|child| usize::from(child.bit_len))
                .sum::<usize>()
                <= Self::LIMIT;
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalShapeMeta {
    key: CanonicalStructKey,
    prefix: SemanticPrefix,
    stabilizer: u8,
}

#[derive(Clone, Copy, Debug)]
struct PackedNodeKey {
    level: u32,
    children: [NodeId; 4],
}

impl PackedNodeKey {
    pub(super) fn new(level: u32, children: [NodeId; 4]) -> Self {
        Self { level, children }
    }
}

impl PartialEq for PackedNodeKey {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level && self.children == other.children
    }
}

impl Eq for PackedNodeKey {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
struct TransformCacheKey {
    node: NodeId,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedSymmetryKey {
    packed: PackedNodeKey,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformShapeKey {
    level: u32,
    children: [PackedTransformId; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformNode {
    level: u32,
    leaf_population: u64,
    children: [PackedTransformId; 4],
    canonical_ref: CanonicalNodeRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformOrderEntry {
    structural: CanonicalStructKey,
    aliases: u8,
    stabilizer: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JumpQuery {
    node: NodeId,
    step_exp: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ShellKey {
    node: NodeId,
    target_level: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RelativeBounds {
    min_x: Coord,
    min_y: Coord,
    max_x: Coord,
    max_y: Coord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalNodeIdentity {
    packed: PackedNodeKey,
    structural: CanonicalStructKey,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug)]
struct CanonicalNodeProbe {
    node: CanonicalNodeIdentity,
    fingerprint: u64,
    used_cached_fingerprint: bool,
}

#[derive(Clone, Copy, Debug)]
struct CanonicalJumpProbe {
    key: CanonicalJumpKey,
    node: CanonicalNodeIdentity,
    fingerprint: u64,
    used_cached_fingerprint: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OverlapMissRecord {
    representative_lane: usize,
    identity: CanonicalNodeIdentity,
    fingerprint: u64,
    join_level: u32,
    join_children: [[NodeId; 4]; 5],
    overlaps: [NodeId; 9],
}

#[derive(Clone, Copy, Debug)]
struct CompactedDiscoveredTask {
    task: DiscoveredJumpTask,
    duplicate_count: u8,
}

#[derive(Clone, Copy, Debug)]
struct ChunkChildState {
    compacted: CompactedDiscoveredTask,
    present: bool,
    blocked: bool,
    enqueued: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct EmbedLayoutCacheKey {
    step_exp: u32,
    width: Coord,
    height: Coord,
    span: Coord,
}

impl ProbeKey for PackedNodeKey {
    fn fingerprint(&self) -> u64 {
        hash_packed_node_fingerprint(self.level, self.children.map(u64::from))
    }
}

impl ProbeKey for CanonicalStructKey {
    fn fingerprint(&self) -> u64 {
        self.fingerprint.probe_hash()
    }
}

#[cfg(test)]
impl ProbeKey for TransformCacheKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5452_414E_5346_4F52,
            [u64::from(self.node), self.symmetry.fingerprint_code()],
        )
    }
}

impl ProbeKey for PackedSymmetryKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5041_434B_5359_4D4D,
            [self.packed.fingerprint(), self.symmetry.fingerprint_code()],
        )
    }
}

impl ProbeKey for PackedTransformShapeKey {
    fn fingerprint(&self) -> u64 {
        hash_u64_words_with_level(self.level, self.children.map(u64::from))
    }
}

impl ProbeKey for JumpQuery {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x4A55_4D50_5155_4552,
            [u64::from(self.node), u64::from(self.step_exp)],
        )
    }
}

impl ProbeKey for NodeId {
    fn fingerprint(&self) -> u64 {
        mix64(u64::from(*self))
    }
}

impl ProbeKey for CanonicalShapeId {
    fn fingerprint(&self) -> u64 {
        hash_words(0x5359_4D4D_4554_5259, [u64::from(*self)])
    }
}

impl ProbeKey for ShellKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5348_454C_4C5F_4B45,
            [u64::from(self.node), u64::from(self.target_level)],
        )
    }
}

impl ProbeKey for CanonicalJumpKey {
    fn fingerprint(&self) -> u64 {
        let fingerprint =
            hash_packed_jump_fingerprint(self.structural.fingerprint(), self.step_exp);
        if self.symmetry_admitted {
            hash_words(0xA5A5_5A5A_D4D4_C3C3, [fingerprint])
        } else {
            fingerprint
        }
    }
}

impl Symmetry {
    const fn fingerprint_code(self) -> u64 {
        match self {
            Self::Identity => 0,
            Self::Rotate90 => 1,
            Self::Rotate180 => 2,
            Self::Rotate270 => 3,
            Self::MirrorX => 4,
            Self::MirrorXRotate90 => 5,
            Self::MirrorXRotate180 => 6,
            Self::MirrorXRotate270 => 7,
        }
    }

    #[cfg(test)]
    fn transform_quadrants(self, quadrants: [NodeId; 4]) -> [NodeId; 4] {
        let permutation = self.quadrant_perm();
        [
            quadrants[permutation[0]],
            quadrants[permutation[1]],
            quadrants[permutation[2]],
            quadrants[permutation[3]],
        ]
    }

    #[cfg(test)]
    fn transform_overlap_nodes(
        self,
        engine: &mut HashLifeEngine,
        overlaps: [NodeId; 9],
    ) -> [NodeId; 9] {
        let permutation = self.grid3_perm();
        engine.transform_node_batch(
            [
                overlaps[permutation[0]],
                overlaps[permutation[1]],
                overlaps[permutation[2]],
                overlaps[permutation[3]],
                overlaps[permutation[4]],
                overlaps[permutation[5]],
                overlaps[permutation[6]],
                overlaps[permutation[7]],
                overlaps[permutation[8]],
            ],
            self,
        )
    }
}

#[derive(Debug)]
struct ResultCaches {
    jump: ProbeTable<CanonicalJumpKey, PackedSymmetryKey>,
    root: ProbeTable<CanonicalJumpKey, PackedSymmetryKey>,
    overlap: ProbeTable<CanonicalStructKey, [NodeId; 9]>,
    oriented: ProbeTable<PackedSymmetryKey, PackedNodeKey>,
    materialized_packed: ProbeTable<PackedNodeKey, NodeId>,
    structural_fast_path: ProbeTable<NodeId, CanonicalNodeIdentity>,
    packed_structural_fast_path: ProbeTable<PackedNodeKey, CanonicalNodeIdentity>,
    shells: ProbeTable<ShellKey, NodeId>,
    bounds: ProbeTable<NodeId, RelativeBounds>,
}

#[derive(Debug)]
struct TransformState {
    #[cfg(test)]
    cache: ProbeTable<TransformCacheKey, NodeId>,
    canonical_cache: ProbeTable<PackedSymmetryKey, PackedTransformId>,
    intern: ProbeTable<PackedTransformShapeKey, PackedTransformId>,
    nodes: Vec<PackedTransformNode>,
    materialized: Vec<Option<NodeId>>,
    packed_roots: Vec<Option<PackedNodeKey>>,
}

#[derive(Debug)]
struct CanonicalCaches {
    shape_epoch: u64,
    shape_intern: ProbeTable<CanonicalStructKey, CanonicalNodeRef>,
    shapes: Vec<CanonicalShapeMeta>,
    node: ProbeTable<NodeId, CanonicalNodeIdentity>,
    packed: ProbeTable<PackedNodeKey, CanonicalNodeIdentity>,
    hot_packed: ProbeTable<PackedNodeKey, CanonicalNodeIdentity>,
    hot_packed_budget: usize,
    oriented: ProbeTable<PackedSymmetryKey, CanonicalNodeIdentity>,
    hot_oriented: ProbeTable<PackedSymmetryKey, CanonicalNodeIdentity>,
    hot_oriented_budget: usize,
    direct_parent: ProbeTable<CanonicalStructKey, CanonicalNodeIdentity>,
    hot_direct_parent: ProbeTable<CanonicalStructKey, CanonicalNodeIdentity>,
    hot_direct_parent_budget: usize,
    symmetry_refs: ProbeTable<CanonicalShapeId, canonical::orientations::OrientationRecord>,
}

#[derive(Debug)]
pub struct HashLifeEngine {
    node_columns: NodeColumns,
    intern: ProbeTable<PackedNodeKey, NodeId>,
    empty_by_level: Vec<NodeId>,
    result_caches: ResultCaches,
    active_jump_results: ProbeTable<CanonicalJumpKey, PackedSymmetryKey>,
    transform_state: TransformState,
    canonical_caches: CanonicalCaches,
    future_state: future::FutureState,
    embed_layout_cache: HashMap<EmbedLayoutCacheKey, Coord>,
    retained_roots: Vec<NodeId>,
    dead_leaf: NodeId,
    live_leaf: NodeId,
    arena_epoch: u64,
    last_gc_nodes: usize,
    scheduler_active: bool,
    allocation_transaction_active: bool,
    allocation_hard_limit: u128,
    allocation_transient_reserved: u128,
    allocation_failure: Option<EngineAllocationFailure>,
    advance_cancellation: Option<[std::sync::Arc<std::sync::atomic::AtomicBool>; 2]>,
    id_capacity: EngineIdCapacity,
    #[cfg(test)]
    symmetry_gate_override: Option<(u32, u64)>,
    stats: HashLifeStats,
}

#[derive(Clone, Copy, Debug)]
#[cfg(test)]
struct EmbeddedJump {
    root: NodeId,
    root_level: u32,
    root_size: Coord,
    world_to_root_x: Coord,
    world_to_root_y: Coord,
    result_origin_x: Coord,
    result_origin_y: Coord,
}

impl Default for HashLifeEngine {
    fn default() -> Self {
        let mut oracle = Self {
            node_columns: NodeColumns::default(),
            intern: ProbeTable::new(ProbeMode::AppendOnly),
            empty_by_level: Vec::new(),
            result_caches: ResultCaches {
                jump: ProbeTable::new(ProbeMode::RebuildOnGc),
                root: ProbeTable::new(ProbeMode::RebuildOnGc),
                overlap: ProbeTable::new(ProbeMode::RebuildOnGc),
                oriented: ProbeTable::new(ProbeMode::RebuildOnGc),
                materialized_packed: ProbeTable::new(ProbeMode::RebuildOnGc),
                structural_fast_path: ProbeTable::new(ProbeMode::RebuildOnGc),
                packed_structural_fast_path: ProbeTable::new(ProbeMode::RebuildOnGc),
                shells: ProbeTable::new(ProbeMode::RebuildOnGc),
                bounds: ProbeTable::new(ProbeMode::RebuildOnGc),
            },
            active_jump_results: ProbeTable::new(ProbeMode::Mutable),
            transform_state: TransformState {
                #[cfg(test)]
                cache: ProbeTable::with_capacity(ProbeMode::Scratch, 64),
                canonical_cache: ProbeTable::new(ProbeMode::RebuildOnGc),
                intern: ProbeTable::new(ProbeMode::AppendOnly),
                nodes: Vec::new(),
                materialized: Vec::new(),
                packed_roots: Vec::new(),
            },
            canonical_caches: CanonicalCaches {
                shape_epoch: 0,
                shape_intern: ProbeTable::new(ProbeMode::AppendOnly),
                shapes: Vec::new(),
                node: ProbeTable::new(ProbeMode::RebuildOnGc),
                packed: ProbeTable::new(ProbeMode::RebuildOnGc),
                hot_packed: ProbeTable::new(ProbeMode::Mutable),
                hot_packed_budget: 0,
                oriented: ProbeTable::new(ProbeMode::RebuildOnGc),
                hot_oriented: ProbeTable::new(ProbeMode::Mutable),
                hot_oriented_budget: 0,
                direct_parent: ProbeTable::new(ProbeMode::RebuildOnGc),
                hot_direct_parent: ProbeTable::new(ProbeMode::Mutable),
                hot_direct_parent_budget: 0,
                symmetry_refs: ProbeTable::new(ProbeMode::RebuildOnGc),
            },
            future_state: future::FutureState::new(0),
            embed_layout_cache: HashMap::new(),
            retained_roots: Vec::with_capacity(1),
            dead_leaf: NodeId::ZERO,
            live_leaf: NodeId::ZERO,
            arena_epoch: 0,
            last_gc_nodes: 0,
            scheduler_active: false,
            allocation_transaction_active: false,
            allocation_hard_limit: u128::MAX,
            allocation_transient_reserved: 0,
            allocation_failure: None,
            advance_cancellation: None,
            id_capacity: EngineIdCapacity::FULL,
            #[cfg(test)]
            symmetry_gate_override: None,
            stats: HashLifeStats::default(),
        };
        oracle.initialize_runtime_state();
        oracle.last_gc_nodes = oracle.node_count();
        oracle
    }
}

impl HashLifeEngine {
    fn record_fingerprint_probe(&mut self, used_cached_fingerprint: bool, count: usize) {
        if used_cached_fingerprint {
            self.stats.canonical_fallback.cached_fingerprint_probes += count;
        } else {
            self.stats.canonical_fallback.recomputed_fingerprint_probes += count;
        }
    }

    #[cfg(test)]
    pub(crate) fn with_symmetry_gate_for_tests(max_level: u32, max_population: u64) -> Self {
        let mut oracle = Self {
            symmetry_gate_override: Some((max_level, max_population)),
            ..Self::default()
        };
        oracle.clear_transient_state(false);
        oracle.stats = HashLifeStats::default();
        oracle
    }

    fn node_count(&self) -> usize {
        self.node_columns.len()
    }

    fn allocated_bytes(&self) -> usize {
        self.node_columns.allocated_bytes()
            + self.intern.allocated_bytes()
            + self.result_caches.allocated_bytes()
            + self.active_jump_results.allocated_bytes()
            + self.canonical_caches.allocated_bytes()
            + self.transform_state.allocated_bytes()
            + self.future_state.allocated_bytes()
            + self.empty_by_level.capacity() * std::mem::size_of::<NodeId>()
            + self.retained_roots.capacity() * std::mem::size_of::<NodeId>()
            + self.embed_layout_cache.capacity()
                * std::mem::size_of::<(EmbedLayoutCacheKey, Coord)>()
    }

    fn push_node(
        &mut self,
        level: u32,
        population: PopulationStat,
        nw: NodeId,
        ne: NodeId,
        sw: NodeId,
        se: NodeId,
    ) -> NodeId {
        let node_id = NodeId::try_from(self.node_count())
            .or_invariant("HashLife node arena exceeded u32 capacity");
        if level != 0 {
            debug_assert!(
                [nw, ne, sw, se]
                    .into_iter()
                    .all(|child| child.precedes(node_id)),
                "HashLife arena topology requires every child id to precede its parent"
            );
        }
        let identity_ref =
            self.build_node_identity_ref(level, [nw, ne, sw, se], population.value());
        self.node_columns
            .push(level, population, [nw, ne, sw, se], identity_ref);
        node_id
    }

    fn packed_leaf_key(alive: bool) -> PackedNodeKey {
        PackedNodeKey::new(
            0,
            [
                NodeId::from(alive),
                NodeId::ZERO,
                NodeId::ZERO,
                NodeId::ZERO,
            ],
        )
    }
}
