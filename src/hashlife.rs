use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;

use crate::bitgrid::{BitGrid, Coord};
use crate::cache_policy::{HASHLIFE_GC_MIN_NODES, should_run_active_hashlife_gc};
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
#[cfg(test)]
mod test_probes;

pub use session::HashLifeSession;
pub use signature::{HashLifeCheckpoint, HashLifeCheckpointKey, HashLifeCheckpointSignature};

type NodeId = u64;
type PackedTransformId = u32;

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
    packed: PackedNodeKey,
    step_exp: u32,
}

impl PartialEq for CanonicalJumpKey {
    fn eq(&self, other: &Self) -> bool {
        self.step_exp == other.step_exp && self.packed == other.packed
    }
}

impl Eq for CanonicalJumpKey {}

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
struct PackedJumpCacheKey {
    packed: PackedNodeKey,
    step_exp: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
struct TransformCacheKey {
    node: NodeId,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformCacheKey {
    packed: PackedNodeKey,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedTransformCompareKey {
    left: PackedTransformId,
    right: PackedTransformId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedSymmetryChildrenCacheEntry {
    children: [[PackedTransformId; 4]; 8],
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedResultCacheEntry {
    packed: PackedNodeKey,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalNodeCacheEntry {
    symmetry: Symmetry,
    packed: PackedNodeKey,
}

#[derive(Clone, Copy, Debug)]
struct CanonicalPackedNode {
    symmetry: Symmetry,
    packed: PackedNodeKey,
    fingerprint: u64,
    used_cached_fingerprint: bool,
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

impl FlatKey for PackedJumpCacheKey {
    fn fingerprint(&self) -> u64 {
        hash_packed_jump_fingerprint(self.packed.fingerprint(), self.step_exp)
    }
}

#[cfg(test)]
impl FlatKey for TransformCacheKey {
    fn fingerprint(&self) -> u64 {
        mix_seed(self.node ^ ((self.symmetry as u64) << 48))
    }
}

impl FlatKey for PackedTransformCacheKey {
    fn fingerprint(&self) -> u64 {
        mix_seed(self.packed.fingerprint() ^ ((self.symmetry as u64) << 48))
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

impl FlatKey for NodeId {
    fn fingerprint(&self) -> u64 {
        mix_seed(*self)
    }
}

impl FlatKey for CanonicalJumpKey {
    fn fingerprint(&self) -> u64 {
        hash_packed_jump_fingerprint(self.packed.fingerprint(), self.step_exp)
    }
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
    }

    fn reserve(&mut self, additional: usize) {
        self.levels.reserve(additional);
        self.populations.reserve(additional);
        self.nws.reserve(additional);
        self.nes.reserve(additional);
        self.sws.reserve(additional);
        self.ses.reserve(additional);
        self.fingerprints.reserve(additional);
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
    jump_cache: FlatTable<PackedJumpCacheKey, PackedResultCacheEntry>,
    root_result_cache: FlatTable<PackedJumpCacheKey, PackedResultCacheEntry>,
    overlap_cache: FlatTable<PackedNodeKey, [NodeId; 9]>,
    #[cfg(test)]
    transform_cache: FlatTable<TransformCacheKey, NodeId>,
    canonical_transform_cache: FlatTable<PackedTransformCacheKey, PackedTransformId>,
    oriented_result_cache: FlatTable<PackedTransformCacheKey, NodeId>,
    packed_transform_compare_cache: FlatTable<PackedTransformCompareKey, i8>,
    packed_symmetry_children_cache: FlatTable<PackedNodeKey, PackedSymmetryChildrenCacheEntry>,
    packed_transform_intern: FlatTable<PackedTransformShapeKey, PackedTransformId>,
    packed_transform_nodes: Vec<PackedTransformNode>,
    packed_transform_materialized: Vec<Option<NodeId>>,
    packed_transform_packed_keys: Vec<Option<PackedNodeKey>>,
    canonical_node_cache: FlatTable<NodeId, CanonicalNodeCacheEntry>,
    embed_layout_cache: HashMap<EmbedLayoutCacheKey, Coord>,
    retained_roots: Vec<NodeId>,
    dead_leaf: NodeId,
    live_leaf: NodeId,
    last_gc_nodes: usize,
    jump_symmetry_max_level: u32,
    jump_symmetry_max_population: u64,
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

#[derive(Clone, Copy, Debug, Default)]
struct HashLifeStats {
    jump_cache_hits: usize,
    symmetric_jump_cache_hits: usize,
    jump_cache_misses: usize,
    root_result_cache_hits: usize,
    root_result_cache_misses: usize,
    overlap_cache_hits: usize,
    overlap_cache_misses: usize,
    nodes_before_mark: usize,
    nodes_after_mark: usize,
    nodes_before_compact: usize,
    nodes_after_compact: usize,
    jump_cache_before_clear: usize,
    gc_runs: usize,
    gc_skips: usize,
    gc_reason: &'static str,
    builder_frames: usize,
    builder_partitions: usize,
    builder_max_stack: usize,
    scheduler_tasks: usize,
    scheduler_ready_max: usize,
    simd_disabled_fast_exits: usize,
    step0_simd_lanes: usize,
    phase1_simd_lanes: usize,
    phase2_simd_lanes: usize,
    step0_simd_batches: usize,
    phase1_simd_batches: usize,
    phase2_simd_batches: usize,
    step0_provisional_records: usize,
    phase1_provisional_records: usize,
    phase2_provisional_records: usize,
    scalar_commit_lanes: usize,
    join_shortcut_avoided: usize,
    dependency_stalls: usize,
    step0_ready_max: usize,
    phase1_ready_max: usize,
    phase2_ready_max: usize,
    canonical_batch_lanes: usize,
    canonical_batch_batches: usize,
    overlap_prep_lanes: usize,
    overlap_prep_batches: usize,
    recursive_overlap_batch_lanes: usize,
    recursive_overlap_batch_batches: usize,
    overlap_local_reuse_lanes: usize,
    cache_probe_batches: usize,
    scheduler_probe_batches: usize,
    symmetry_gate_allowed: usize,
    symmetry_gate_blocked: usize,
    canonical_node_cache_hits: usize,
    canonical_node_cache_misses: usize,
    jump_presence_probe_batches: usize,
    jump_presence_probe_lanes: usize,
    jump_presence_probe_hits: usize,
    jump_batch_unique_queries: usize,
    jump_batch_reused_queries: usize,
    cached_fingerprint_probes: usize,
    recomputed_fingerprint_probes: usize,
    gc_mark_batches: usize,
    gc_remap_batches: usize,
    packed_d4_canonicalization_misses: usize,
    packed_inverse_transform_hits: usize,
    packed_recursive_transform_hits: usize,
    packed_recursive_transform_misses: usize,
    packed_overlap_outputs_produced: usize,
    packed_cache_result_lookups: usize,
    packed_cache_result_hits: usize,
    packed_cache_result_materializations: usize,
    #[cfg(test)]
    transformed_node_materializations: usize,
}

#[derive(Clone, Copy)]
enum PendingTask {
    PhaseOne {
        next_exp: u32,
        a: NodeId,
        b: NodeId,
        c: NodeId,
        d: NodeId,
        e: NodeId,
        f: NodeId,
        g: NodeId,
        h: NodeId,
        i: NodeId,
    },
    PhaseTwo {
        next_exp: u32,
        nw: NodeId,
        ne: NodeId,
        sw: NodeId,
        se: NodeId,
    },
}

#[derive(Clone, Copy)]
struct TaskRecord {
    remaining: u8,
    task: PendingTask,
}

#[derive(Clone, Copy)]
struct Step0TaskRecord {
    remaining: u8,
    children: [NodeId; 4],
}

#[derive(Clone, Copy)]
struct RecursiveParentBatchRecord {
    cache_key: CanonicalJumpKey,
    packed_parent: PackedNodeKey,
    packed_fingerprint: u64,
    inverse_symmetry: Symmetry,
    level: u32,
    next_exp: u32,
    overlaps: [NodeId; 9],
    child_keys: [CanonicalJumpKey; 9],
    child_nodes: [NodeId; 9],
}

#[derive(Clone, Copy)]
struct DiscoveredJumpTask {
    key: CanonicalJumpKey,
    source_node: NodeId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step0LaneDispatch {
    SimdChild,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdTaskKind {
    Step0,
    PhaseOne,
    PhaseTwo,
}

#[derive(Clone, Copy)]
struct SimdProvisionalRecord {
    cache_key: CanonicalJumpKey,
    level: u32,
    kind: SimdTaskKind,
    input_nodes: [NodeId; 9],
    input_populations: [u64; 9],
    payload: SimdProvisionalPayload,
}

#[derive(Clone, Copy)]
enum SimdProvisionalPayload {
    Step0 {
        dispatch: Step0LaneDispatch,
    },
    Recursive {
        next_exp: u32,
        source_task_id: usize,
    },
}

#[derive(Clone, Copy)]
struct SimdPackedBatch {
    active_lanes: usize,
    active_mask: u8,
    populations: [u64x8; 9],
}

#[derive(Clone, Copy)]
struct SimdLaneResult {
    output_nonzero_mask: u8,
}

struct SimdBatchResult {
    lanes: [SimdLaneResult; SIMD_BATCH_LANES],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JoinIntent {
    level: u32,
    children: [NodeId; 4],
}

#[derive(Clone, Copy, Debug)]
struct EmbeddedCell {
    key: u128,
}

const NO_DEPENDENT: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
struct DependentEdge {
    task_id: usize,
    next: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HashLifeRuntimeStats {
    pub nodes: usize,
    pub intern: usize,
    pub empty_levels: usize,
    pub jump_cache: usize,
    pub retained_roots: usize,
    pub overlap_cache: usize,
    pub jump_cache_hits: usize,
    pub symmetric_jump_cache_hits: usize,
    pub jump_cache_misses: usize,
    pub root_result_cache_hits: usize,
    pub root_result_cache_misses: usize,
    pub overlap_cache_hits: usize,
    pub overlap_cache_misses: usize,
    pub gc_runs: usize,
    pub gc_skips: usize,
    pub nodes_before_mark: usize,
    pub nodes_after_mark: usize,
    pub nodes_before_compact: usize,
    pub nodes_after_compact: usize,
    pub jump_cache_before_clear: usize,
    pub gc_reason: &'static str,
    pub builder_frames: usize,
    pub builder_partitions: usize,
    pub builder_max_stack: usize,
    pub scheduler_tasks: usize,
    pub scheduler_ready_max: usize,
    pub simd_disabled_fast_exits: usize,
    pub step0_simd_lanes: usize,
    pub phase1_simd_lanes: usize,
    pub phase2_simd_lanes: usize,
    pub step0_simd_batches: usize,
    pub phase1_simd_batches: usize,
    pub phase2_simd_batches: usize,
    pub step0_provisional_records: usize,
    pub phase1_provisional_records: usize,
    pub phase2_provisional_records: usize,
    pub scalar_commit_lanes: usize,
    pub join_shortcut_avoided: usize,
    pub dependency_stalls: usize,
    pub step0_ready_max: usize,
    pub phase1_ready_max: usize,
    pub phase2_ready_max: usize,
    pub canonical_batch_lanes: usize,
    pub canonical_batch_batches: usize,
    pub overlap_prep_lanes: usize,
    pub overlap_prep_batches: usize,
    pub recursive_overlap_batch_lanes: usize,
    pub recursive_overlap_batch_batches: usize,
    pub overlap_local_reuse_lanes: usize,
    pub cache_probe_batches: usize,
    pub scheduler_probe_batches: usize,
    pub symmetry_gate_allowed: usize,
    pub symmetry_gate_blocked: usize,
    pub canonical_node_cache_hits: usize,
    pub canonical_node_cache_misses: usize,
    pub jump_presence_probe_batches: usize,
    pub jump_presence_probe_lanes: usize,
    pub jump_presence_probe_hits: usize,
    pub jump_batch_unique_queries: usize,
    pub jump_batch_reused_queries: usize,
    pub cached_fingerprint_probes: usize,
    pub recomputed_fingerprint_probes: usize,
    pub gc_mark_batches: usize,
    pub gc_remap_batches: usize,
    pub packed_d4_canonicalization_misses: usize,
    pub packed_inverse_transform_hits: usize,
    pub packed_recursive_transform_hits: usize,
    pub packed_recursive_transform_misses: usize,
    pub packed_overlap_outputs_produced: usize,
    pub packed_cache_result_lookups: usize,
    pub packed_cache_result_hits: usize,
    pub packed_cache_result_materializations: usize,
    #[cfg(test)]
    pub transformed_node_materializations: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HashLifeDiagnosticSummary {
    pub total_nodes: usize,
    pub retained_roots: usize,
    pub nodes_match_intern: bool,
    pub dependency_stalls: usize,
    pub jump_full_hit_rate: f64,
    pub jump_presence_hit_rate: f64,
    pub overlap_hit_rate: f64,
    pub overlap_local_reuse_rate: f64,
    pub symmetry_gate_allow_rate: f64,
    pub canonical_cache_hit_rate: f64,
    pub packed_cache_hit_rate: f64,
    pub symmetry_jump_hits: usize,
    pub simd_lane_coverage: f64,
    pub scalar_commit_ratio: f64,
    pub probes_per_scheduler_task: f64,
    pub recursive_overlap_batch_rate: f64,
    pub gc_reclaim_ratio: f64,
    pub gc_compact_ratio: f64,
    pub gc_reason: &'static str,
    pub gc_runs: usize,
    pub gc_skips: usize,
    pub packed_d4_canonicalization_misses: usize,
    pub packed_inverse_transform_hits: usize,
    pub packed_recursive_transform_hits: usize,
    pub packed_recursive_transform_misses: usize,
    pub packed_overlap_outputs_produced: usize,
    pub packed_cache_result_materializations: usize,
    #[cfg(test)]
    pub transformed_node_materializations: usize,
}

const DISCOVER_BATCH: usize = 4;
const JUMP_SYMMETRY_MAX_LEVEL: u32 = 8;
const JUMP_SYMMETRY_MAX_POPULATION: u64 = 4_096;
const HASHLIFE_DEBUG_ITERATION_LOG_INTERVAL: usize = 100_000;
const HASHLIFE_DEBUG_INITIAL_STACK_LOG_THRESHOLD: usize = 10_000;
pub const HASHLIFE_FULL_GRID_MAX_POPULATION: u64 = 250_000;
pub const HASHLIFE_FULL_GRID_MAX_CHUNKS: usize = 100_000;
pub const HASHLIFE_CHECKPOINT_MAX_POPULATION: u64 = 250_000;
pub const HASHLIFE_CHECKPOINT_MAX_BOUNDS_SPAN: Coord = 65_536;

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
            packed_transform_compare_cache: FlatTable::new(),
            packed_symmetry_children_cache: FlatTable::new(),
            packed_transform_intern: FlatTable::new(),
            packed_transform_nodes: Vec::new(),
            packed_transform_materialized: Vec::new(),
            packed_transform_packed_keys: Vec::new(),
            canonical_node_cache: FlatTable::new(),
            embed_layout_cache: HashMap::new(),
            retained_roots: Vec::new(),
            dead_leaf: 0,
            live_leaf: 0,
            last_gc_nodes: 0,
            jump_symmetry_max_level: JUMP_SYMMETRY_MAX_LEVEL,
            jump_symmetry_max_population: JUMP_SYMMETRY_MAX_POPULATION,
            stats: HashLifeStats::default(),
        };
        oracle.initialize_runtime_state();
        oracle.last_gc_nodes = oracle.node_count();
        oracle
    }
}

impl HashLifeEngine {
    #[cfg(test)]
    pub(crate) fn with_symmetry_gate_for_tests(max_level: u32, max_population: u64) -> Self {
        let mut oracle = Self {
            jump_symmetry_max_level: max_level,
            jump_symmetry_max_population: max_population,
            ..Self::default()
        };
        oracle.clear_transient_state();
        oracle.stats = HashLifeStats::default();
        oracle
    }

    fn record_fingerprint_probe(&mut self, used_cached_fingerprint: bool, count: usize) {
        if used_cached_fingerprint {
            self.stats.cached_fingerprint_probes += count;
        } else {
            self.stats.recomputed_fingerprint_probes += count;
        }
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
        self.node_columns.push(level, population, nw, ne, sw, se);
        node_id
    }

    fn packed_leaf_key(alive: bool) -> PackedNodeKey {
        PackedNodeKey::new(0, [u64::from(alive), 0, 0, 0])
    }

    pub(super) fn begin_persistent_run(&mut self) -> Option<NodeId> {
        self.stats = HashLifeStats::default();
        self.clear_transient_state();
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

        let debug = hashlife_debug_enabled();
        let mut current = None::<BitGrid>;
        let mut remaining = generations;
        let mut last_root = None;

        if debug {
            eprintln!(
                "[hashlife] advance start gens={generations} pop={} bounds={:?}",
                current.as_ref().unwrap_or(grid).population(),
                current.as_ref().unwrap_or(grid).bounds()
            );
        }

        while remaining != 0 {
            let safe_jump = max_hashlife_safe_jump(current.as_ref().unwrap_or(grid));
            let step_limit = remaining.min(safe_jump.max(1));
            let step_exp = 63 - step_limit.leading_zeros();
            let step = 1_u64 << step_exp;
            if debug {
                eprintln!(
                    "[hashlife] segment step={step} step_exp={step_exp} safe_jump={} pop={} bounds={:?}",
                    safe_jump,
                    current.as_ref().unwrap_or(grid).population(),
                    current.as_ref().unwrap_or(grid).bounds()
                );
            }
            let (next, root) =
                self.advance_power_of_two(current.as_ref().unwrap_or(grid), step_exp);
            current = Some(next);
            last_root = Some(root);
            if debug {
                eprintln!(
                    "[hashlife] segment done step={step} pop={} bounds={:?}",
                    current.as_ref().unwrap_or(grid).population(),
                    current.as_ref().unwrap_or(grid).bounds()
                );
            }
            remaining -= step;
        }

        (current.unwrap_or_else(BitGrid::empty), last_root)
    }

    pub(super) fn finish_persistent_run(
        &mut self,
        previous_root: Option<NodeId>,
        last_root: Option<NodeId>,
    ) {
        let debug = hashlife_debug_enabled();

        if let Some(root) = last_root {
            self.record_retained_root(root);
        }
        self.stats.jump_cache_before_clear = self.jump_cache.len();
        let gc_reason = self.gc_reason(previous_root, last_root);
        if debug {
            eprintln!(
                "[hashlife] gc before nodes={} intern={} retained={} jump_cache={} reason={gc_reason}",
                self.node_count(),
                self.intern.len(),
                self.retained_roots.len(),
                self.jump_cache.len()
            );
        }
        self.maybe_garbage_collect(gc_reason);
        if debug {
            eprintln!(
                "[hashlife] gc after nodes={} intern={} retained={} jump_cache={} jump_hits={} jump_misses={} root_hits={} root_misses={} overlap_hits={} overlap_misses={} marked={} compacted={}->{}",
                self.node_count(),
                self.intern.len(),
                self.retained_roots.len(),
                self.jump_cache.len(),
                self.stats.jump_cache_hits,
                self.stats.jump_cache_misses,
                self.stats.root_result_cache_hits,
                self.stats.root_result_cache_misses,
                self.stats.overlap_cache_hits,
                self.stats.overlap_cache_misses,
                self.stats.nodes_after_mark,
                self.stats.nodes_before_compact,
                self.stats.nodes_after_compact,
            );
        }
    }

    pub(super) fn maybe_collect_active_run(
        &mut self,
        current_root: Option<NodeId>,
    ) -> Option<NodeId> {
        let root = current_root?;
        if !should_run_active_hashlife_gc(self.node_count(), self.last_gc_nodes) {
            return Some(root);
        }

        self.record_retained_root(root);
        self.stats.jump_cache_before_clear = self.jump_cache.len();
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
            self.stats.root_result_cache_hits += 1;
            cached
        } else {
            self.stats.root_result_cache_misses += 1;
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
    let span = width.max(height);
    let raw_max_jump = (((Coord::MAX as i128) - (2 * span as i128) - 8) / 4).max(1) as u64;
    let mut jump = 1_u64 << (63 - raw_max_jump.leading_zeros());
    while jump > 1 && required_root_size_for_jump(span as u64, jump) > Coord::MAX as u64 {
        jump >>= 1;
    }
    jump
}

fn required_root_size_for_jump(span: u64, jump: u64) -> u64 {
    (2 * span + 4 * (jump + 2))
        .max((4 * jump) + 4)
        .max(4)
        .next_power_of_two()
}

fn hashlife_debug_enabled() -> bool {
    matches!(
        env::var("HASHLIFE_DEBUG").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
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
