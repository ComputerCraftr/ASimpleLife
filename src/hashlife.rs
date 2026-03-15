use crate::RequiredExt;
#[cfg(test)]
use crate::bitgrid::BitGrid;
use crate::bitgrid::Coord;
use crate::cache_policy::should_run_active_hashlife_gc;
use crate::flat_table::{FlatKey, FlatTable};
use crate::hashing::{
    hash_packed_jump_fingerprint, hash_packed_node_fingerprint, hash_u64_words_with_level,
    hash_u64_words_with_level_batch, hash_words, mix64,
};
use crate::probe_table::{ProbeMode, ProbeTable};
use crate::simd_layout::{
    AlignedU32Batch, AlignedU64WordBatch4, AlignedU64WordBatch9, SIMD_BATCH_LANES,
};
use crate::symmetry::D4Symmetry as Symmetry;
use bytemuck::must_cast;
use std::collections::HashMap;
use wide::u64x8;

mod advance;
mod arena;
mod canonical;
mod embed;
mod gc;
mod geometry;
mod kernels;
mod memory;
mod node;
mod population;
mod scheduler;
mod session;
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
    HashLifeSnapshotError, deserialize_to_grid as deserialize_snapshot_to_grid,
    serialize_grid as serialize_grid_snapshot,
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

type NodeId = u32;
type PackedTransformId = u32;
type CanonicalNodeRef = u32;
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
            structural: CanonicalStructKey::new(0, [0; 4]),
            step_exp: 0,
            symmetry_admitted: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalStructKey {
    level: u32,
    children: [CanonicalNodeRef; 4],
}

impl CanonicalStructKey {
    fn new(level: u32, children: [CanonicalNodeRef; 4]) -> Self {
        Self { level, children }
    }

    fn leaf(alive: bool) -> Self {
        Self::new(0, [u32::from(alive), 0, 0, 0])
    }
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

impl PartialOrd for PackedNodeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PackedNodeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.level
            .cmp(&other.level)
            .then_with(|| self.children.cmp(&other.children))
    }
}

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
struct PackedTransformCompareKey {
    left: PackedTransformId,
    right: PackedTransformId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformShapeKey {
    level: u32,
    children: [CanonicalNodeRef; 4],
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SymmetryRefKey {
    node: NodeId,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectCanonicalParentKey {
    level: u32,
    symmetry: Symmetry,
    children: [CanonicalNodeRef; 4],
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

impl FlatKey for PackedNodeKey {
    fn fingerprint(&self) -> u64 {
        hash_packed_node_fingerprint(self.level, self.children.map(u64::from))
    }
}

impl FlatKey for CanonicalStructKey {
    fn fingerprint(&self) -> u64 {
        hash_u64_words_with_level(self.level, self.children.map(u64::from))
    }
}

#[cfg(test)]
impl FlatKey for TransformCacheKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5452_414E_5346_4F52,
            [u64::from(self.node), self.symmetry.fingerprint_code()],
        )
    }
}

impl FlatKey for PackedSymmetryKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5041_434B_5359_4D4D,
            [self.packed.fingerprint(), self.symmetry.fingerprint_code()],
        )
    }
}

impl FlatKey for DirectCanonicalParentKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x4449_5245_4354_5041,
            [
                hash_u64_words_with_level(self.level, self.children.map(u64::from)),
                self.symmetry.fingerprint_code(),
            ],
        )
    }
}

impl FlatKey for PackedTransformCompareKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x434F_4D50_4152_4521,
            [u64::from(self.left), u64::from(self.right)],
        )
    }
}

impl FlatKey for PackedTransformShapeKey {
    fn fingerprint(&self) -> u64 {
        hash_u64_words_with_level(self.level, self.children.map(u64::from))
    }
}

impl FlatKey for JumpQuery {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x4A55_4D50_5155_4552,
            [u64::from(self.node), u64::from(self.step_exp)],
        )
    }
}

impl FlatKey for NodeId {
    fn fingerprint(&self) -> u64 {
        mix64(u64::from(*self))
    }
}

impl FlatKey for SymmetryRefKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5359_4D4D_4554_5259,
            [u64::from(self.node), self.symmetry.fingerprint_code()],
        )
    }
}

impl FlatKey for ShellKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            0x5348_454C_4C5F_4B45,
            [u64::from(self.node), u64::from(self.target_level)],
        )
    }
}

impl FlatKey for CanonicalJumpKey {
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
    cache: FlatTable<TransformCacheKey, NodeId>,
    canonical_cache: ProbeTable<PackedSymmetryKey, PackedTransformId>,
    compare_cache: ProbeTable<PackedTransformCompareKey, i8>,
    intern: ProbeTable<PackedTransformShapeKey, PackedTransformId>,
    nodes: Vec<PackedTransformNode>,
    materialized: Vec<Option<NodeId>>,
    packed_roots: Vec<Option<PackedNodeKey>>,
}

#[derive(Debug)]
struct CanonicalCaches {
    shape_intern: ProbeTable<CanonicalStructKey, CanonicalNodeRef>,
    node: ProbeTable<NodeId, CanonicalNodeIdentity>,
    packed: ProbeTable<PackedNodeKey, CanonicalNodeIdentity>,
    hot_packed: ProbeTable<PackedNodeKey, CanonicalNodeIdentity>,
    hot_packed_budget: usize,
    oriented: ProbeTable<PackedSymmetryKey, CanonicalNodeIdentity>,
    hot_oriented: ProbeTable<PackedSymmetryKey, CanonicalNodeIdentity>,
    hot_oriented_budget: usize,
    direct_parent: ProbeTable<DirectCanonicalParentKey, CanonicalNodeIdentity>,
    hot_direct_parent: ProbeTable<DirectCanonicalParentKey, CanonicalNodeIdentity>,
    hot_direct_parent_budget: usize,
    symmetry_refs: ProbeTable<SymmetryRefKey, CanonicalNodeRef>,
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
                cache: FlatTable::with_capacity(64),
                canonical_cache: ProbeTable::new(ProbeMode::RebuildOnGc),
                compare_cache: ProbeTable::new(ProbeMode::RebuildOnGc),
                intern: ProbeTable::new(ProbeMode::AppendOnly),
                nodes: Vec::new(),
                materialized: Vec::new(),
                packed_roots: Vec::new(),
            },
            canonical_caches: CanonicalCaches {
                shape_intern: ProbeTable::new(ProbeMode::AppendOnly),
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
            embed_layout_cache: HashMap::new(),
            retained_roots: Vec::new(),
            dead_leaf: 0,
            live_leaf: 0,
            arena_epoch: 0,
            last_gc_nodes: 0,
            scheduler_active: false,
            allocation_transaction_active: false,
            allocation_hard_limit: u128::MAX,
            allocation_transient_reserved: 0,
            allocation_failure: None,
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
    fn intern_canonical_shape(&mut self, structural: CanonicalStructKey) -> CanonicalNodeRef {
        if let Some(existing) = self.canonical_caches.shape_intern.get(&structural) {
            return existing;
        }
        if !self.prepare_mandatory_shape_growth() {
            return 0;
        }
        let Ok(id) = CanonicalNodeRef::try_from(self.canonical_caches.shape_intern.len()) else {
            self.reject_canonical_reference_exhaustion();
            return 0;
        };
        if self
            .canonical_caches
            .shape_intern
            .try_insert(structural, id)
            .is_err()
        {
            self.reject_allocation(u128::MAX);
            return 0;
        }
        id
    }

    fn rebuild_canonical_shapes(&mut self) {
        self.canonical_caches.shape_intern.clear();
        self.canonical_caches.symmetry_refs.clear();
        let dead_shape = self.intern_canonical_shape(CanonicalStructKey::leaf(false));
        let live_shape = self.intern_canonical_shape(CanonicalStructKey::leaf(true));
        debug_assert_eq!((dead_shape, live_shape), (0, 1));
        for index in 0..self.node_columns.len() {
            let node =
                NodeId::try_from(index).or_invariant("HashLife node arena exceeded u32 capacity");
            let identity_ref = self.build_node_identity_ref(
                self.node_columns.level(node),
                self.node_columns.quadrants(node),
                self.node_columns.population(node),
            );
            self.node_columns.set_identity_ref(node, identity_ref);
        }
    }

    fn build_node_identity_ref(
        &mut self,
        level: u32,
        children: [NodeId; 4],
        population: u128,
    ) -> CanonicalNodeRef {
        if level == 0 {
            return u32::from(population != 0);
        }

        let order_children = children.map(|child| self.node_columns.identity_ref(child));
        self.intern_canonical_shape(CanonicalStructKey::new(level, order_children))
    }

    fn cached_symmetry_ref(&self, node: NodeId, symmetry: Symmetry) -> Option<CanonicalNodeRef> {
        if symmetry == Symmetry::Identity || self.node_columns.level(node) == 0 {
            return Some(self.node_columns.identity_ref(node));
        }
        self.canonical_caches
            .symmetry_refs
            .get(&SymmetryRefKey { node, symmetry })
    }

    fn symmetry_canonical_ref(&mut self, node: NodeId, symmetry: Symmetry) -> CanonicalNodeRef {
        if let Some(cached) = self.cached_symmetry_ref(node, symmetry) {
            return cached;
        }

        const MAX_SYMMETRY_STACK: usize = 256;
        let mut stack = [(0, false); MAX_SYMMETRY_STACK];
        stack[0] = (node, false);
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (current, ready) = stack[stack_len];
            if self.cached_symmetry_ref(current, symmetry).is_some() {
                continue;
            }
            if !ready {
                if stack_len + 5 > stack.len() {
                    crate::invariant_failure!(
                        "validated symmetry traversal depth exceeded fixed workspace"
                    );
                }
                stack[stack_len] = (current, true);
                stack_len += 1;
                let children = self.node_columns.quadrants(current);
                for child in children {
                    if self.cached_symmetry_ref(child, symmetry).is_none() {
                        stack[stack_len] = (child, false);
                        stack_len += 1;
                    }
                }
                continue;
            }

            let children = self.node_columns.quadrants(current);
            let permutation = symmetry.quadrant_perm();
            let order_children = permutation.map(|child_index| {
                self.cached_symmetry_ref(children[child_index], symmetry)
                    .or_invariant("child symmetry reference should be resolved")
            });
            let structural =
                CanonicalStructKey::new(self.node_columns.level(current), order_children);
            let canonical_ref = self.intern_canonical_shape(structural);
            if !self.record_mandatory_symmetry_ref(
                SymmetryRefKey {
                    node: current,
                    symmetry,
                },
                canonical_ref,
            ) {
                return 0;
            }
        }

        match self.cached_symmetry_ref(node, symmetry) {
            Some(reference) => reference,
            None if self.allocation_failed() => 0,
            None => crate::invariant_failure!("requested symmetry reference should be resolved"),
        }
    }

    fn symmetry_entry(&mut self, node: NodeId, symmetry: Symmetry) -> PackedTransformOrderEntry {
        let level = self.node_columns.level(node);
        if level == 0 {
            let structural = CanonicalStructKey::leaf(self.node_columns.population(node) != 0);
            return PackedTransformOrderEntry { structural };
        }
        let children = self.node_columns.quadrants(node);
        let permutation = symmetry.quadrant_perm();
        let order_children = permutation
            .map(|child_index| self.symmetry_canonical_ref(children[child_index], symmetry));
        let structural = CanonicalStructKey::new(level, order_children);
        PackedTransformOrderEntry { structural }
    }

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
        let result_cache_bytes = self.result_caches.jump.allocated_bytes()
            + self.result_caches.root.allocated_bytes()
            + self.result_caches.overlap.allocated_bytes()
            + self.result_caches.oriented.allocated_bytes()
            + self.result_caches.materialized_packed.allocated_bytes()
            + self.result_caches.structural_fast_path.allocated_bytes()
            + self
                .result_caches
                .packed_structural_fast_path
                .allocated_bytes();
        let result_cache_bytes = result_cache_bytes
            + self.result_caches.shells.allocated_bytes()
            + self.result_caches.bounds.allocated_bytes()
            + self.active_jump_results.allocated_bytes();
        let canonical_cache_bytes = self.canonical_caches.shape_intern.allocated_bytes()
            + self.canonical_caches.node.allocated_bytes()
            + self.canonical_caches.packed.allocated_bytes()
            + self.canonical_caches.hot_packed.allocated_bytes()
            + self.canonical_caches.oriented.allocated_bytes()
            + self.canonical_caches.hot_oriented.allocated_bytes()
            + self.canonical_caches.direct_parent.allocated_bytes()
            + self.canonical_caches.hot_direct_parent.allocated_bytes()
            + self.canonical_caches.symmetry_refs.allocated_bytes();
        let transform_bytes = self.transform_state.canonical_cache.allocated_bytes()
            + self.transform_state.compare_cache.allocated_bytes()
            + self.transform_state.intern.allocated_bytes()
            + self.transform_state.nodes.capacity() * std::mem::size_of::<PackedTransformNode>()
            + self.transform_state.materialized.capacity() * std::mem::size_of::<Option<NodeId>>()
            + self.transform_state.packed_roots.capacity()
                * std::mem::size_of::<Option<PackedNodeKey>>();
        self.node_columns.allocated_bytes()
            + self.intern.allocated_bytes()
            + result_cache_bytes
            + canonical_cache_bytes
            + transform_bytes
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
                [nw, ne, sw, se].into_iter().all(|child| child < node_id),
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
        PackedNodeKey::new(0, [u32::from(alive), 0, 0, 0])
    }
}
