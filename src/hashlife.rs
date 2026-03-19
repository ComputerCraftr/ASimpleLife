use std::collections::HashMap;
use std::sync::OnceLock;
use crate::bitgrid::{BitGrid, Coord};
use crate::cache_policy::{
    HASHLIFE_GC_MIN_NODES, HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER,
    should_run_active_hashlife_gc,
};
use crate::flat_table::{FlatKey, FlatTable};
use crate::hashing::{
    hash_leaf_population, hash_packed_jump_fingerprint, hash_packed_node_fingerprint,
    hash_u64_words_with_level, hash_u64_words_with_level_batch, mix_seed,
};
#[cfg(test)]
use crate::simd_layout::transpose_u64_words_9xn;
use crate::simd_layout::{
    AlignedU32Batch, AlignedU64WordBatch4, AlignedU64WordBatch9, SIMD_BATCH_LANES,
    transpose_u64_lanes_9xn,
};
use crate::symmetry::D4Symmetry as Symmetry;
use bytemuck::must_cast;
use wide::u64x8;

mod canonical;
mod embed;
mod gc;
mod node;
mod scheduler;
mod session;
mod signature;
mod simd;
mod snapshot;
mod stats;
#[cfg(test)]
mod test_probes;

pub use session::HashLifeSession;
#[cfg(test)]
pub(crate) use session::HashLifeSessionAdvanceProfile;
pub use signature::{HashLifeCheckpoint, HashLifeCheckpointKey, HashLifeCheckpointSignature};
pub use snapshot::{
    HashLifeSnapshotError, deserialize_to_grid as deserialize_snapshot_to_grid,
    serialize_grid as serialize_grid_snapshot,
};
pub use stats::{
    HASHLIFE_CHECKPOINT_MAX_BOUNDS_SPAN, HASHLIFE_CHECKPOINT_MAX_POPULATION,
    HASHLIFE_FULL_GRID_MAX_CHUNKS, HASHLIFE_FULL_GRID_MAX_POPULATION,
};
#[cfg(test)]
pub(crate) use stats::{HashLifeDiagnosticSummary, HashLifeRuntimeStats};
use stats::*;

type NodeId = u64;
type PackedTransformId = u32;
type CanonicalNodeRef = u128;

const DENSE_SHORTCUT_MAX_LEVEL: u32 = 6;
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
        max_population: u64,
        max_chunks: usize,
        max_bounds_span: Coord,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridExtractionError {
    NotLoaded,
    PopulationLimitExceeded { population: u64, limit: u64 },
    ChunkLimitExceeded { chunks: usize, limit: usize },
    BoundsSpanLimitExceeded { bounds_span: Coord, limit: Coord },
}

#[derive(Clone, Copy, Debug)]
struct CanonicalJumpKey {
    structural: CanonicalStructKey,
    step_exp: u32,
}

impl PartialEq for CanonicalJumpKey {
    fn eq(&self, other: &Self) -> bool {
        self.step_exp == other.step_exp && self.structural == other.structural
    }
}

impl Eq for CanonicalJumpKey {}

impl CanonicalJumpKey {
    fn empty() -> Self {
        Self {
            structural: CanonicalStructKey::new(0, [0; 4]),
            step_exp: 0,
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
        Self::new(0, [u128::from(alive), 0, 0, 0])
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
    children: [u64; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformNode {
    level: u32,
    leaf_population: u64,
    children: [PackedTransformId; 4],
    structural: CanonicalStructKey,
    canonical_ref: CanonicalNodeRef,
    order_fingerprint: u64,
    order_children: [CanonicalNodeRef; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformOrderEntry {
    structural: CanonicalStructKey,
    canonical_ref: CanonicalNodeRef,
    order_fingerprint: u64,
    order_children: [CanonicalNodeRef; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NodeSymmetryMetadata {
    entries: [PackedTransformOrderEntry; 8],
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
        hash_packed_node_fingerprint(self.level, self.children)
    }
}

impl FlatKey for CanonicalStructKey {
    fn fingerprint(&self) -> u64 {
        let low = hash_u64_words_with_level(self.level, self.children.map(|child| child as u64));
        let high = hash_u64_words_with_level(
            self.level ^ 0x9E37_79B9,
            self.children.map(|child| (child >> 64) as u64),
        );
        mix_seed(low ^ high.rotate_left(17))
    }
}

#[cfg(test)]
impl FlatKey for TransformCacheKey {
    fn fingerprint(&self) -> u64 {
        mix_seed(self.node ^ ((self.symmetry as u64) << 48))
    }
}

impl FlatKey for PackedSymmetryKey {
    fn fingerprint(&self) -> u64 {
        mix_seed(self.packed.fingerprint() ^ ((self.symmetry as u64) << 48))
    }
}

impl FlatKey for DirectCanonicalParentKey {
    fn fingerprint(&self) -> u64 {
        let low = hash_u64_words_with_level(self.level, self.children.map(|child| child as u64));
        let high = hash_u64_words_with_level(
            self.level ^ 0x6D2B_79F5 ^ (self.symmetry as u32),
            self.children.map(|child| (child >> 64) as u64),
        );
        mix_seed(low ^ high.rotate_left(11))
    }
}

impl FlatKey for PackedTransformCompareKey {
    fn fingerprint(&self) -> u64 {
        mix_seed((self.left as u64) ^ ((self.right as u64) << 32))
    }
}

impl FlatKey for PackedTransformShapeKey {
    fn fingerprint(&self) -> u64 {
        hash_u64_words_with_level(self.level, self.children)
    }
}

impl FlatKey for JumpQuery {
    fn fingerprint(&self) -> u64 {
        mix_seed(self.node ^ ((self.step_exp as u64) << 48))
    }
}

impl FlatKey for NodeId {
    fn fingerprint(&self) -> u64 {
        mix_seed(*self)
    }
}

impl FlatKey for CanonicalJumpKey {
    fn fingerprint(&self) -> u64 {
        hash_packed_jump_fingerprint(self.structural.fingerprint(), self.step_exp)
    }
}

fn canonical_node_ref_from_structural(key: CanonicalStructKey) -> CanonicalNodeRef {
    let low = hash_u64_words_with_level(key.level, key.children.map(|child| child as u64));
    let high = mix_seed(
        hash_u64_words_with_level(
            key.level ^ 0xA5A5_5A5A,
            key.children.map(|child| (child >> 64) as u64),
        ) ^ low.rotate_left(13),
    );
    ((low as u128) << 64) | (high as u128)
}

impl Symmetry {
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

fn centered_2x2_from_4x4(mask: u16) -> u8 {
    let mut result = 0_u8;
    let mut y = 1_u8;
    while y <= 2 {
        let mut x = 1_u8;
        while x <= 2 {
            let mut neighbors = 0_u8;
            let mut ny = y - 1;
            while ny <= y + 1 {
                let mut nx = x - 1;
                while nx <= x + 1 {
                    if nx != x || ny != y {
                        let bit = ny as u16 * 4 + nx as u16;
                        neighbors += ((mask >> bit) & 1) as u8;
                    }
                    nx += 1;
                }
                ny += 1;
            }
            let cell_bit = y as u16 * 4 + x as u16;
            let alive = ((mask >> cell_bit) & 1) != 0;
            if neighbors == 3 || (neighbors == 2 && alive) {
                let centered_bit = (y - 1) * 2 + (x - 1);
                result |= 1 << centered_bit;
            }
            x += 1;
        }
        y += 1;
    }
    result
}

fn build_base_transition_table() -> [u8; 1 << 16] {
    let mut table = [0_u8; 1 << 16];
    let mut mask = 0_u32;
    while mask < (1 << 16) {
        table[mask as usize] = centered_2x2_from_4x4(mask as u16);
        mask += 1;
    }
    table
}

fn base_transitions() -> &'static [u8; 1 << 16] {
    static BASE_TRANSITIONS: OnceLock<[u8; 1 << 16]> = OnceLock::new();
    BASE_TRANSITIONS.get_or_init(build_base_transition_table)
}

#[derive(Debug, Default)]
struct NodeColumns {
    levels: Vec<u32>,
    populations: Vec<u64>,
    nws: Vec<NodeId>,
    nes: Vec<NodeId>,
    sws: Vec<NodeId>,
    ses: Vec<NodeId>,
    fingerprints: Vec<u64>,
    symmetry_metadata: Vec<NodeSymmetryMetadata>,
}

impl NodeColumns {
    fn len(&self) -> usize {
        self.levels.len()
    }

    fn push(
        &mut self,
        level: u32,
        population: u64,
        nw: NodeId,
        ne: NodeId,
        sw: NodeId,
        se: NodeId,
        symmetry_metadata: NodeSymmetryMetadata,
    ) {
        self.levels.push(level);
        self.populations.push(population);
        self.nws.push(nw);
        self.nes.push(ne);
        self.sws.push(sw);
        self.ses.push(se);
        self.fingerprints.push(if level == 0 {
            hash_leaf_population(population)
        } else {
            hash_u64_words_with_level(level, [nw, ne, sw, se])
        });
        self.symmetry_metadata.push(symmetry_metadata);
    }

    fn reserve(&mut self, additional: usize) {
        self.levels.reserve(additional);
        self.populations.reserve(additional);
        self.nws.reserve(additional);
        self.nes.reserve(additional);
        self.sws.reserve(additional);
        self.ses.reserve(additional);
        self.fingerprints.reserve(additional);
        self.symmetry_metadata.reserve(additional);
    }

    fn level(&self, node: NodeId) -> u32 {
        self.levels[node as usize]
    }

    fn population(&self, node: NodeId) -> u64 {
        self.populations[node as usize]
    }

    fn quadrants(&self, node: NodeId) -> [NodeId; 4] {
        let index = node as usize;
        [
            self.nws[index],
            self.nes[index],
            self.sws[index],
            self.ses[index],
        ]
    }

    fn fingerprint(&self, node: NodeId) -> u64 {
        self.fingerprints[node as usize]
    }

    fn symmetry_entry(&self, node: NodeId, symmetry: Symmetry) -> PackedTransformOrderEntry {
        self.symmetry_metadata[node as usize].entries[symmetry as usize]
    }

    pub(super) fn packed_key(&self, node: NodeId) -> PackedNodeKey {
        let index = node as usize;
        if self.levels[index] == 0 {
            return PackedNodeKey::new(0, [self.populations[index], 0, 0, 0]);
        }
        PackedNodeKey::new(
            self.levels[index],
            [
                self.nws[index],
                self.nes[index],
                self.sws[index],
                self.ses[index],
            ],
        )
    }

    fn packed_key_and_fingerprint(&self, node: NodeId) -> (PackedNodeKey, u64) {
        (self.packed_key(node), self.fingerprint(node))
    }
}

#[derive(Debug)]
pub struct HashLifeEngine {
    node_columns: NodeColumns,
    intern: FlatTable<PackedNodeKey, NodeId>,
    empty_by_level: Vec<NodeId>,
    jump_cache: FlatTable<CanonicalJumpKey, PackedSymmetryKey>,
    root_result_cache: FlatTable<CanonicalJumpKey, PackedSymmetryKey>,
    overlap_cache: FlatTable<CanonicalStructKey, [NodeId; 9]>,
    #[cfg(test)]
    transform_cache: FlatTable<TransformCacheKey, NodeId>,
    canonical_transform_cache: FlatTable<PackedSymmetryKey, PackedTransformId>,
    oriented_result_cache: FlatTable<PackedSymmetryKey, PackedNodeKey>,
    materialized_packed_result_cache: FlatTable<PackedNodeKey, NodeId>,
    packed_transform_compare_cache: FlatTable<PackedTransformCompareKey, i8>,
    packed_transform_intern: FlatTable<PackedTransformShapeKey, PackedTransformId>,
    packed_transform_nodes: Vec<PackedTransformNode>,
    packed_transform_materialized: Vec<Option<NodeId>>,
    packed_transform_packed_roots: Vec<Option<PackedNodeKey>>,
    canonical_node_cache: FlatTable<NodeId, CanonicalNodeIdentity>,
    canonical_packed_cache: FlatTable<PackedNodeKey, CanonicalNodeIdentity>,
    hot_canonical_packed_cache: FlatTable<PackedNodeKey, CanonicalNodeIdentity>,
    hot_canonical_packed_budget: usize,
    canonical_oriented_cache: FlatTable<PackedSymmetryKey, CanonicalNodeIdentity>,
    hot_canonical_oriented_cache: FlatTable<PackedSymmetryKey, CanonicalNodeIdentity>,
    hot_canonical_oriented_budget: usize,
    direct_parent_canonical_cache: FlatTable<DirectCanonicalParentKey, CanonicalNodeIdentity>,
    hot_direct_parent_canonical_cache: FlatTable<DirectCanonicalParentKey, CanonicalNodeIdentity>,
    hot_direct_parent_canonical_budget: usize,
    structural_fast_path_cache: FlatTable<NodeId, CanonicalNodeIdentity>,
    packed_structural_fast_path_cache: FlatTable<PackedNodeKey, CanonicalNodeIdentity>,
    embed_layout_cache: HashMap<EmbedLayoutCacheKey, Coord>,
    retained_roots: Vec<NodeId>,
    dead_leaf: NodeId,
    live_leaf: NodeId,
    last_gc_nodes: usize,
    #[cfg(test)]
    symmetry_gate_override: Option<(u32, u64)>,
    stats: HashLifeStats,
}

#[derive(Clone, Copy, Debug)]
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
            intern: FlatTable::new(),
            empty_by_level: Vec::new(),
            jump_cache: FlatTable::new(),
            root_result_cache: FlatTable::new(),
            overlap_cache: FlatTable::new(),
            #[cfg(test)]
            transform_cache: FlatTable::new(),
            canonical_transform_cache: FlatTable::new(),
            oriented_result_cache: FlatTable::new(),
            materialized_packed_result_cache: FlatTable::new(),
            packed_transform_compare_cache: FlatTable::new(),
            packed_transform_intern: FlatTable::new(),
            packed_transform_nodes: Vec::new(),
            packed_transform_materialized: Vec::new(),
            packed_transform_packed_roots: Vec::new(),
            canonical_node_cache: FlatTable::new(),
            canonical_packed_cache: FlatTable::new(),
            hot_canonical_packed_cache: FlatTable::new(),
            hot_canonical_packed_budget: 0,
            canonical_oriented_cache: FlatTable::new(),
            hot_canonical_oriented_cache: FlatTable::new(),
            hot_canonical_oriented_budget: 0,
            direct_parent_canonical_cache: FlatTable::new(),
            hot_direct_parent_canonical_cache: FlatTable::new(),
            hot_direct_parent_canonical_budget: 0,
            structural_fast_path_cache: FlatTable::new(),
            packed_structural_fast_path_cache: FlatTable::new(),
            embed_layout_cache: HashMap::new(),
            retained_roots: Vec::new(),
            dead_leaf: 0,
            live_leaf: 0,
            last_gc_nodes: 0,
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
    fn build_leaf_symmetry_metadata(alive: bool) -> NodeSymmetryMetadata {
        let structural = CanonicalStructKey::leaf(alive);
        let entry = PackedTransformOrderEntry {
            structural,
            canonical_ref: canonical_node_ref_from_structural(structural),
            order_fingerprint: structural.fingerprint(),
            order_children: structural.children,
        };
        NodeSymmetryMetadata { entries: [entry; 8] }
    }

    fn build_node_symmetry_metadata(
        &self,
        level: u32,
        children: [NodeId; 4],
        population: u64,
    ) -> NodeSymmetryMetadata {
        if level == 0 {
            return Self::build_leaf_symmetry_metadata(population != 0);
        }

        let child_entries = children.map(|child| {
            Symmetry::ALL.map(|symmetry| self.node_columns.symmetry_entry(child, symmetry))
        });
        let mut entries = [PackedTransformOrderEntry {
            structural: CanonicalStructKey::new(0, [0; 4]),
            canonical_ref: 0,
            order_fingerprint: 0,
            order_children: [0; 4],
        }; 8];

        let mut symmetry_index = 0;
        while symmetry_index < Symmetry::ALL.len() {
            let symmetry = Symmetry::ALL[symmetry_index];
            let perm = symmetry.quadrant_perm();
            let transformed_children = [
                child_entries[perm[0]][symmetry_index],
                child_entries[perm[1]][symmetry_index],
                child_entries[perm[2]][symmetry_index],
                child_entries[perm[3]][symmetry_index],
            ];
            let order_children = transformed_children.map(|child| child.canonical_ref);
            let structural = CanonicalStructKey::new(level, order_children);
            entries[symmetry_index] = PackedTransformOrderEntry {
                structural,
                canonical_ref: canonical_node_ref_from_structural(structural),
                order_fingerprint: structural.fingerprint(),
                order_children,
            };
            symmetry_index += 1;
        }

        NodeSymmetryMetadata { entries }
    }

    fn record_fingerprint_probe(&mut self, used_cached_fingerprint: bool, count: usize) {
        if used_cached_fingerprint {
            self.stats.cached_fingerprint_probes += count;
        } else {
            self.stats.recomputed_fingerprint_probes += count;
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

    fn push_node(
        &mut self,
        level: u32,
        population: u64,
        nw: NodeId,
        ne: NodeId,
        sw: NodeId,
        se: NodeId,
    ) -> NodeId {
        let node_id = self.node_count() as NodeId;
        let symmetry_metadata = self.build_node_symmetry_metadata(level, [nw, ne, sw, se], population);
        self.node_columns
            .push(level, population, nw, ne, sw, se, symmetry_metadata);
        node_id
    }

    fn packed_leaf_key(alive: bool) -> PackedNodeKey {
        PackedNodeKey::new(0, [u64::from(alive), 0, 0, 0])
    }

    pub(super) fn begin_persistent_run(&mut self) -> Option<NodeId> {
        self.stats = HashLifeStats::default();
        self.retained_roots.last().copied()
    }

    pub(super) fn advance_segment(
        &mut self,
        grid: &BitGrid,
        generations: u64,
    ) -> (BitGrid, Option<NodeId>) {
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
            let safe_jump = max_hashlife_safe_jump(current.as_ref().unwrap_or(grid));
            let step_limit = remaining.min(safe_jump.max(1));
            let step_exp = 63 - step_limit.leading_zeros();
            let step = 1_u64 << step_exp;
            let (next, root) =
                self.advance_power_of_two(current.as_ref().unwrap_or(grid), step_exp);
            current = Some(next);
            last_root = Some(root);
            remaining -= step;
        }

        (current.unwrap_or_else(BitGrid::empty), last_root)
    }

    pub(super) fn finish_persistent_run(
        &mut self,
        previous_root: Option<NodeId>,
        last_root: Option<NodeId>,
    ) {
        if let Some(root) = last_root {
            self.record_retained_root(root);
        }
        self.stats.jump_cache_before_clear = self.jump_cache.len();
        let gc_reason = self.gc_reason(previous_root, last_root);
        self.maybe_garbage_collect(gc_reason);
    }

    pub(super) fn maybe_collect_active_run(
        &mut self,
        current_root: Option<NodeId>,
    ) -> Option<NodeId> {
        let root = current_root?;
        let active_gc_needed = should_run_active_hashlife_gc(self.node_count(), self.last_gc_nodes);
        let transient_pressure_high =
            self.transient_cache_pressure_entries() >= HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER;
        if !active_gc_needed && !transient_pressure_high {
            return Some(root);
        }

        self.record_retained_root(root);
        self.stats.jump_cache_before_clear = self.jump_cache.len();
        if transient_pressure_high && !active_gc_needed {
            self.stats.gc_reason = "transient_deferred";
            self.stats.gc_skips += 1;
            self.stats.gc_skipped_with_transient_growth += 1;
            return Some(root);
        }
        let gc_reason = if self.node_count() >= HASHLIFE_GC_MIN_NODES {
            "node_threshold"
        } else {
            "growth_threshold"
        };
        self.maybe_garbage_collect(gc_reason);
        self.retained_roots.last().copied()
    }

    pub fn advance(&mut self, grid: &BitGrid, generations: u64) -> BitGrid {
        let previous_root = self.begin_persistent_run();
        let (advanced, last_root) = self.advance_segment(grid, generations);
        self.finish_persistent_run(previous_root, last_root);
        advanced
    }

    fn advance_power_of_two(&mut self, grid: &BitGrid, step_exp: u32) -> (BitGrid, NodeId) {
        if grid.is_empty() {
            return (BitGrid::empty(), self.empty(0));
        }

        let embedded = self.embed_for_jump(grid, step_exp);
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

    fn advance_pow2(&mut self, node: NodeId, step_exp: u32) -> NodeId {
        if step_exp == 0 {
            self.advance_one_generation_centered(node)
        } else {
            self.advance_power_of_two_recursive(node, step_exp)
        }
    }
}

fn max_hashlife_safe_jump(current: &BitGrid) -> u64 {
    let Some((min_x, min_y, max_x, max_y)) = current.bounds() else {
        return 1;
    };
    let width = (max_x - min_x + 1).max(1);
    let height = (max_y - min_y + 1).max(1);
    max_hashlife_safe_jump_from_span(width.max(height))
}

pub(super) fn max_hashlife_safe_jump_from_span(span: Coord) -> u64 {
    if span <= 0 {
        return 1;
    }
    let raw_max_jump = (((Coord::MAX as i128) - (2 * span as i128) - 8) / 4).max(1) as u64;
    let mut jump = 1_u64 << (63 - raw_max_jump.leading_zeros());
    while jump > 1 && required_root_size_for_jump(span as u64, jump) > Coord::MAX as u64 {
        jump >>= 1;
    }
    jump
}

pub(super) fn required_root_size_for_jump(span: u64, jump: u64) -> u64 {
    (2 * span + 4 * (jump + 2))
        .max((4 * jump) + 4)
        .max(4)
        .next_power_of_two()
}

fn quadrant_end(
    cells: &[EmbeddedCell],
    start: usize,
    end: usize,
    bit_shift: u32,
    quadrant: u128,
) -> usize {
    let upper = quadrant + 1;
    start + cells[start..end].partition_point(|cell| ((cell.key >> bit_shift) & 0b11) < upper)
}
