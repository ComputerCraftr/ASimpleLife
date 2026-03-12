use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;

use crate::bitgrid::{BitGrid, Coord};
use crate::cache_policy::{HASHLIFE_GC_MIN_NODES, should_run_active_hashlife_gc};
use crate::symmetry::D4Symmetry as Symmetry;

mod embed;
mod gc;
mod node;
mod session;
mod signature;

pub use session::HashLifeSession;
pub use signature::{HashLifeCheckpoint, HashLifeCheckpointKey, HashLifeCheckpointSignature};

type NodeId = u64;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CanonicalJumpKey {
    node: NodeId,
    step_exp: u32,
}

impl Symmetry {
    const fn quadrant_permutation(self) -> [usize; 4] {
        match self {
            Self::Identity => [0, 1, 2, 3],
            Self::Rotate90 => [1, 3, 0, 2],
            Self::Rotate180 => [3, 2, 1, 0],
            Self::Rotate270 => [2, 0, 3, 1],
            Self::MirrorX => [1, 0, 3, 2],
            Self::MirrorXRotate90 => [3, 1, 2, 0],
            Self::MirrorXRotate180 => [2, 3, 0, 1],
            Self::MirrorXRotate270 => [0, 2, 1, 3],
        }
    }

    const fn overlap_permutation(self) -> [usize; 9] {
        match self {
            Self::Identity => [0, 1, 2, 3, 4, 5, 6, 7, 8],
            Self::Rotate90 => [2, 5, 8, 1, 4, 7, 0, 3, 6],
            Self::Rotate180 => [8, 7, 6, 5, 4, 3, 2, 1, 0],
            Self::Rotate270 => [6, 3, 0, 7, 4, 1, 8, 5, 2],
            Self::MirrorX => [2, 1, 0, 5, 4, 3, 8, 7, 6],
            Self::MirrorXRotate90 => [8, 5, 2, 7, 4, 1, 6, 3, 0],
            Self::MirrorXRotate180 => [6, 7, 8, 3, 4, 5, 0, 1, 2],
            Self::MirrorXRotate270 => [0, 3, 6, 1, 4, 7, 2, 5, 8],
        }
    }

    fn transform_quadrants(self, quadrants: [NodeId; 4]) -> [NodeId; 4] {
        let permutation = self.quadrant_permutation();
        [
            quadrants[permutation[0]],
            quadrants[permutation[1]],
            quadrants[permutation[2]],
            quadrants[permutation[3]],
        ]
    }

    fn transform_overlap_nodes(
        self,
        engine: &mut HashLifeEngine,
        overlaps: [NodeId; 9],
    ) -> [NodeId; 9] {
        let permutation = self.overlap_permutation();
        let mut transformed = [0; 9];
        let mut index = 0;
        while index < overlaps.len() {
            transformed[index] = engine.transform_node(overlaps[permutation[index]], self);
            index += 1;
        }
        transformed
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

#[derive(Clone, Copy, Debug)]
struct Node {
    level: u32,
    population: u64,
    nw: NodeId,
    ne: NodeId,
    sw: NodeId,
    se: NodeId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NodeKey {
    Leaf(bool),
    Internal {
        level: u32,
        nw: NodeId,
        ne: NodeId,
        sw: NodeId,
        se: NodeId,
    },
}

#[derive(Debug)]
pub struct HashLifeEngine {
    nodes: Vec<Node>,
    intern: HashMap<NodeKey, NodeId>,
    empty_by_level: Vec<NodeId>,
    jump_cache: HashMap<CanonicalJumpKey, NodeId>,
    root_result_cache: HashMap<(NodeId, u32), NodeId>,
    overlap_cache: HashMap<NodeId, [NodeId; 9]>,
    transform_cache: HashMap<(NodeId, Symmetry), NodeId>,
    canonical_node_cache: HashMap<NodeId, (NodeId, Symmetry)>,
    embed_layout_cache: HashMap<(u32, Coord, Coord, Coord), Coord>,
    retained_roots: Vec<NodeId>,
    dead_leaf: NodeId,
    live_leaf: NodeId,
    last_gc_nodes: usize,
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

#[derive(Clone, Copy, Debug)]
struct EmbeddedCell {
    key: u128,
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
}

const DISCOVER_BATCH: usize = 4;
const JUMP_SYMMETRY_MAX_LEVEL: u32 = 8;
const JUMP_SYMMETRY_MAX_POPULATION: u64 = 4_096;

impl Default for HashLifeEngine {
    fn default() -> Self {
        let mut oracle = Self {
            nodes: Vec::new(),
            intern: HashMap::new(),
            empty_by_level: Vec::new(),
            jump_cache: HashMap::new(),
            root_result_cache: HashMap::new(),
            overlap_cache: HashMap::new(),
            transform_cache: HashMap::new(),
            canonical_node_cache: HashMap::new(),
            embed_layout_cache: HashMap::new(),
            retained_roots: Vec::new(),
            dead_leaf: 0,
            live_leaf: 0,
            last_gc_nodes: 0,
            stats: HashLifeStats::default(),
        };
        oracle.initialize_runtime_state();
        oracle.last_gc_nodes = oracle.nodes.len();
        oracle
    }
}

impl HashLifeEngine {
    fn node_tuple_key(&self, node: NodeId) -> (u32, NodeId, NodeId, NodeId, NodeId) {
        let node_ref = &self.nodes[node as usize];
        (
            node_ref.level,
            node_ref.nw,
            node_ref.ne,
            node_ref.sw,
            node_ref.se,
        )
    }

    fn transform_node(&mut self, node: NodeId, symmetry: Symmetry) -> NodeId {
        if symmetry == Symmetry::Identity || self.nodes[node as usize].level == 0 {
            return node;
        }
        if let Some(&transformed) = self.transform_cache.get(&(node, symmetry)) {
            return transformed;
        }

        let (nw, ne, sw, se) = {
            let node_ref = self.nodes[node as usize];
            (node_ref.nw, node_ref.ne, node_ref.sw, node_ref.se)
        };
        let transformed_children = [
            self.transform_node(nw, symmetry),
            self.transform_node(ne, symmetry),
            self.transform_node(sw, symmetry),
            self.transform_node(se, symmetry),
        ];
        let [next_nw, next_ne, next_sw, next_se] =
            symmetry.transform_quadrants(transformed_children);
        let transformed = self.join(next_nw, next_ne, next_sw, next_se);
        self.transform_cache.insert((node, symmetry), transformed);
        transformed
    }

    fn canonicalize_node(&mut self, node: NodeId) -> (NodeId, Symmetry) {
        if self.nodes[node as usize].level == 0 {
            return (node, Symmetry::Identity);
        }
        if let Some(&canonical) = self.canonical_node_cache.get(&node) {
            return canonical;
        }

        let mut best_node = node;
        let mut best_symmetry = Symmetry::Identity;
        let mut best_key = self.node_tuple_key(node);
        for symmetry in Symmetry::ALL {
            let transformed = self.transform_node(node, symmetry);
            let transformed_key = self.node_tuple_key(transformed);
            if transformed_key < best_key {
                best_key = transformed_key;
                best_node = transformed;
                best_symmetry = symmetry;
            }
        }

        let canonical = (best_node, best_symmetry);
        self.canonical_node_cache.insert(node, canonical);
        canonical
    }

    fn canonical_jump_key(&mut self, key: (NodeId, u32)) -> (CanonicalJumpKey, Symmetry) {
        let (canonical_node, symmetry) = if self.should_symmetry_canonicalize_jump_node(key.0) {
            self.canonicalize_node(key.0)
        } else {
            (key.0, Symmetry::Identity)
        };
        (
            CanonicalJumpKey {
                node: canonical_node,
                step_exp: key.1,
            },
            symmetry,
        )
    }

    fn should_symmetry_canonicalize_jump_node(&self, node: NodeId) -> bool {
        let node_ref = &self.nodes[node as usize];
        node_ref.level <= JUMP_SYMMETRY_MAX_LEVEL
            && node_ref.population <= JUMP_SYMMETRY_MAX_POPULATION
    }

    fn canonicalized_cache_node(&mut self, node: NodeId) -> (NodeId, Symmetry) {
        if self.should_symmetry_canonicalize_jump_node(node) {
            self.canonicalize_node(node)
        } else {
            (node, Symmetry::Identity)
        }
    }

    fn cached_jump_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let (canonical_key, symmetry) = self.canonical_jump_key(key);
        let inverse = symmetry.inverse();
        let result = self.jump_cache.get(&canonical_key).copied()?;
        if symmetry != Symmetry::Identity {
            self.stats.symmetric_jump_cache_hits += 1;
        }
        Some(self.transform_node(result, inverse))
    }

    fn insert_jump_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let (canonical_key, symmetry) = self.canonical_jump_key(key);
        let canonical_result = self.transform_node(result, symmetry);
        self.jump_cache.insert(canonical_key, canonical_result);
    }

    fn jump_result(&mut self, key: (NodeId, u32)) -> NodeId {
        self.cached_jump_result(key)
            .expect("missing HashLife jump result")
    }

    fn cached_root_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let (canonical_node, symmetry) = self.canonicalized_cache_node(key.0);
        let inverse = symmetry.inverse();
        let result = self
            .root_result_cache
            .get(&(canonical_node, key.1))
            .copied()?;
        Some(self.transform_node(result, inverse))
    }

    fn insert_root_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let (canonical_node, symmetry) = self.canonicalized_cache_node(key.0);
        let canonical_result = self.transform_node(result, symmetry);
        self.root_result_cache
            .insert((canonical_node, key.1), canonical_result);
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
                self.nodes.len(),
                self.intern.len(),
                self.retained_roots.len(),
                self.jump_cache.len()
            );
        }
        self.maybe_garbage_collect(gc_reason);
        if debug {
            eprintln!(
                "[hashlife] gc after nodes={} intern={} retained={} jump_cache={} jump_hits={} jump_misses={} root_hits={} root_misses={} overlap_hits={} overlap_misses={} marked={} compacted={}->{}",
                self.nodes.len(),
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
        if !should_run_active_hashlife_gc(self.nodes.len(), self.last_gc_nodes) {
            return Some(root);
        }

        self.record_retained_root(root);
        self.stats.jump_cache_before_clear = self.jump_cache.len();
        let gc_reason = if self.nodes.len() >= HASHLIFE_GC_MIN_NODES {
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

    fn advance_power_of_two_recursive(&mut self, root_node: NodeId, root_step_exp: u32) -> NodeId {
        let debug = hashlife_debug_enabled();
        let level = self.nodes[root_node as usize].level as usize;
        let task_capacity = 1usize << level.saturating_sub(root_step_exp as usize + 1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        discover.push(self.canonical_jump_key((root_node, root_step_exp)).0);
        let mut task_index: HashMap<CanonicalJumpKey, usize> =
            HashMap::with_capacity(task_capacity);
        let mut tasks = Vec::<Option<TaskRecord>>::with_capacity(task_capacity);
        let mut task_keys = Vec::<Option<CanonicalJumpKey>>::with_capacity(task_capacity);
        let mut dependents: HashMap<CanonicalJumpKey, Vec<usize>> =
            HashMap::with_capacity(task_capacity);
        let mut ready = Vec::<usize>::with_capacity(task_capacity);
        let mut batch = [CanonicalJumpKey {
            node: 0,
            step_exp: 0,
        }; DISCOVER_BATCH];
        let mut iterations = 0_usize;
        let mut max_stack = discover.len();
        let mut next_iteration_log = 100_000_usize;
        let mut next_stack_log = 10_000_usize;

        while self
            .cached_jump_result((root_node, root_step_exp))
            .is_none()
        {
            while !discover.is_empty() {
                let mut batch_len = 0;
                while batch_len < DISCOVER_BATCH {
                    let Some(entry) = discover.pop() else {
                        break;
                    };
                    batch[batch_len] = entry;
                    batch_len += 1;
                }
                for &canonical_task in &batch[..batch_len] {
                    let discovered_node = canonical_task.node;
                    let discovered_step_exp = canonical_task.step_exp;
                    iterations += 1;
                    if discover.len() > max_stack {
                        max_stack = discover.len();
                    }
                    if debug && iterations >= next_iteration_log {
                        eprintln!(
                            "[hashlife] advance_pow2 progress node={discovered_node} step_exp={discovered_step_exp} iterations={iterations} stack={} max_stack={} cache={} pending={} ready={}",
                            discover.len(),
                            max_stack,
                            self.jump_cache.len(),
                            task_index.len(),
                            ready.len(),
                        );
                        next_iteration_log += 100_000;
                    }
                    if debug && max_stack >= next_stack_log {
                        eprintln!(
                            "[hashlife] advance_pow2 stack node={discovered_node} step_exp={discovered_step_exp} iterations={iterations} stack={} max_stack={} cache={} pending={} ready={}",
                            discover.len(),
                            max_stack,
                            self.jump_cache.len(),
                            task_index.len(),
                            ready.len(),
                        );
                        next_stack_log = next_stack_log.saturating_mul(2);
                    }
                    let cache_key = canonical_task;
                    if self
                        .cached_jump_result((cache_key.node, cache_key.step_exp))
                        .is_some()
                    {
                        self.stats.jump_cache_hits += 1;
                        continue;
                    }
                    self.stats.jump_cache_misses += 1;
                    if task_index.contains_key(&cache_key) {
                        continue;
                    }

                    let discovered_level = self.nodes[discovered_node as usize].level;
                    assert!(discovered_level >= 2);
                    assert!(discovered_step_exp <= discovered_level - 2);

                    if discovered_step_exp == 0 {
                        let result = self.advance_one_generation_centered(discovered_node);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_dependents(&cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if self.nodes[discovered_node as usize].population == 0 {
                        let result = self.empty(discovered_level - 1);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_dependents(&cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level == 2 {
                        let result = self.base_transition(discovered_node);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_dependents(&cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level <= DENSE_SHORTCUT_MAX_LEVEL {
                        let result =
                            self.dense_advance_centered(discovered_node, discovered_step_exp);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_dependents(&cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    let [
                        upper_left,
                        upper_center,
                        upper_right,
                        middle_left,
                        center,
                        middle_right,
                        lower_left,
                        lower_center,
                        lower_right,
                    ] = self.overlapping_subnodes(discovered_node);
                    let child_step_exp = discovered_step_exp - 1;
                    let task_id = tasks.len();
                    task_index.insert(cache_key, task_id);
                    tasks.push(Some(TaskRecord {
                        remaining: 0,
                        task: PendingTask::PhaseOne {
                            next_exp: child_step_exp,
                            a: upper_left,
                            b: upper_center,
                            c: upper_right,
                            d: middle_left,
                            e: center,
                            f: middle_right,
                            g: lower_left,
                            h: lower_center,
                            i: lower_right,
                        },
                    }));
                    task_keys.push(Some(cache_key));
                    self.stats.scheduler_tasks += 1;
                    for child_node in [
                        lower_right,
                        lower_center,
                        lower_left,
                        middle_right,
                        center,
                        middle_left,
                        upper_right,
                        upper_center,
                        upper_left,
                    ] {
                        let child_key = self.canonical_jump_key((child_node, child_step_exp)).0;
                        if self
                            .cached_jump_result((child_key.node, child_key.step_exp))
                            .is_none()
                        {
                            dependents.entry(child_key).or_default().push(task_id);
                            tasks[task_id].as_mut().unwrap().remaining += 1;
                            if !task_index.contains_key(&child_key) {
                                discover.push(child_key);
                            }
                        }
                    }
                    if tasks[task_id].as_ref().unwrap().remaining == 0 {
                        ready.push(task_id);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                    }
                }
            }

            if self
                .cached_jump_result((root_node, root_step_exp))
                .is_some()
            {
                break;
            }

            let Some(task_id) = ready.pop() else {
                let sample = task_index.iter().next().map(|(&pending_key, &task_id)| {
                    let task = tasks[task_id].unwrap();
                    let (recurse_exp, missing) = match task.task {
                        PendingTask::PhaseOne {
                            next_exp,
                            a,
                            b,
                            c,
                            d,
                            e,
                            f,
                            g,
                            h,
                            i,
                        } => (
                            next_exp,
                            [a, b, c, d, e, f, g, h, i]
                                .into_iter()
                                .filter(|&child| {
                                    self.cached_jump_result((child, next_exp)).is_none()
                                })
                                .collect::<Vec<_>>(),
                        ),
                        PendingTask::PhaseTwo {
                            next_exp,
                            nw,
                            ne,
                            sw,
                            se,
                        } => (
                            next_exp,
                            [nw, ne, sw, se]
                                .into_iter()
                                .filter(|&child| {
                                    self.cached_jump_result((child, next_exp)).is_none()
                                })
                                .collect::<Vec<_>>(),
                        ),
                    };
                    (pending_key.node, pending_key.step_exp, recurse_exp, missing)
                });
                panic!(
                    "hashlife dependency resolution stalled root_node={root_node} root_step_exp={root_step_exp} pending={} ready={} cache={} sample={sample:?}",
                    task_index.len(),
                    ready.len(),
                    self.jump_cache.len(),
                );
            };
            let Some(task_key) = task_keys[task_id].take() else {
                continue;
            };
            task_index.remove(&task_key);
            let task = tasks[task_id].take().unwrap();
            debug_assert_eq!(task.remaining, 0);
            match task.task {
                PendingTask::PhaseOne { .. } => {
                    let pending_node = task_key.node;
                    let pending_exp = task_key.step_exp;
                    let PendingTask::PhaseOne {
                        next_exp,
                        a,
                        b,
                        c,
                        d,
                        e,
                        f,
                        g,
                        h,
                        i,
                    } = task.task
                    else {
                        unreachable!()
                    };
                    let upper_left_result = self.jump_result((a, next_exp));
                    let upper_center_result = self.jump_result((b, next_exp));
                    let upper_right_result = self.jump_result((c, next_exp));
                    let middle_left_result = self.jump_result((d, next_exp));
                    let center_result = self.jump_result((e, next_exp));
                    let middle_right_result = self.jump_result((f, next_exp));
                    let lower_left_result = self.jump_result((g, next_exp));
                    let lower_center_result = self.jump_result((h, next_exp));
                    let lower_right_result = self.jump_result((i, next_exp));

                    let next_upper_left = self.join(
                        upper_left_result,
                        upper_center_result,
                        middle_left_result,
                        center_result,
                    );
                    let next_upper_right = self.join(
                        upper_center_result,
                        upper_right_result,
                        center_result,
                        middle_right_result,
                    );
                    let next_lower_left = self.join(
                        middle_left_result,
                        center_result,
                        lower_left_result,
                        lower_center_result,
                    );
                    let next_lower_right = self.join(
                        center_result,
                        middle_right_result,
                        lower_center_result,
                        lower_right_result,
                    );
                    let parent_key = CanonicalJumpKey {
                        node: pending_node,
                        step_exp: pending_exp,
                    };
                    let parent_id = task_id;
                    task_index.insert(parent_key, parent_id);
                    task_keys[parent_id] = Some(parent_key);
                    tasks[parent_id] = Some(TaskRecord {
                        remaining: 0,
                        task: PendingTask::PhaseTwo {
                            next_exp,
                            nw: next_upper_left,
                            ne: next_upper_right,
                            sw: next_lower_left,
                            se: next_lower_right,
                        },
                    });
                    for child_node in [
                        next_lower_right,
                        next_lower_left,
                        next_upper_right,
                        next_upper_left,
                    ] {
                        let child_key = self.canonical_jump_key((child_node, next_exp)).0;
                        if self
                            .cached_jump_result((child_key.node, child_key.step_exp))
                            .is_none()
                        {
                            dependents.entry(child_key).or_default().push(parent_id);
                            tasks[parent_id].as_mut().unwrap().remaining += 1;
                            if !task_index.contains_key(&child_key) {
                                discover.push(child_key);
                            }
                        }
                    }
                    if tasks[parent_id].as_ref().unwrap().remaining == 0 {
                        ready.push(parent_id);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                    }
                }
                PendingTask::PhaseTwo { .. } => {
                    let pending_node = task_key.node;
                    let pending_exp = task_key.step_exp;
                    let PendingTask::PhaseTwo {
                        next_exp,
                        nw,
                        ne,
                        sw,
                        se,
                    } = task.task
                    else {
                        unreachable!()
                    };
                    let q00 = self.jump_result((nw, next_exp));
                    let q01 = self.jump_result((ne, next_exp));
                    let q10 = self.jump_result((sw, next_exp));
                    let q11 = self.jump_result((se, next_exp));
                    let result = self.join(q00, q01, q10, q11);
                    let key = (pending_node, pending_exp);
                    self.insert_jump_result(key, result);
                    notify_dependents(
                        &self.canonical_jump_key(key).0,
                        &mut tasks,
                        &mut dependents,
                        &mut ready,
                    );
                    if ready.len() > self.stats.scheduler_ready_max {
                        self.stats.scheduler_ready_max = ready.len();
                    }
                }
            }
        }

        if debug {
            eprintln!(
                "[hashlife] advance_pow2 done root={root_node} step_exp={root_step_exp} iterations={iterations} max_stack={} cache={} pending={} ready={}",
                max_stack,
                self.jump_cache.len(),
                task_index.len(),
                ready.len(),
            );
        }

        self.jump_result((root_node, root_step_exp))
    }

    fn advance_one_generation_centered(&mut self, root_node: NodeId) -> NodeId {
        let root_key = (root_node, 0);
        if self.cached_jump_result(root_key).is_some() {
            self.stats.jump_cache_hits += 1;
            return self.jump_result(root_key);
        }

        let debug = hashlife_debug_enabled();
        let level = self.nodes[root_node as usize].level as usize;
        let task_capacity = 1usize << level.saturating_sub(1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        discover.push(self.canonical_jump_key((root_node, 0)).0);
        let mut task_index: HashMap<CanonicalJumpKey, usize> =
            HashMap::with_capacity(task_capacity);
        let mut tasks = Vec::<Option<Step0TaskRecord>>::with_capacity(task_capacity);
        let mut task_keys = Vec::<Option<CanonicalJumpKey>>::with_capacity(task_capacity);
        let mut dependents: HashMap<CanonicalJumpKey, Vec<usize>> =
            HashMap::with_capacity(task_capacity);
        let mut ready = Vec::<usize>::with_capacity(task_capacity);
        let mut batch = [CanonicalJumpKey {
            node: 0,
            step_exp: 0,
        }; DISCOVER_BATCH];
        let mut iterations = 0_usize;

        while self.cached_jump_result(root_key).is_none() {
            while !discover.is_empty() {
                let mut batch_len = 0;
                while batch_len < DISCOVER_BATCH {
                    let Some(entry) = discover.pop() else {
                        break;
                    };
                    batch[batch_len] = entry;
                    batch_len += 1;
                }
                for &canonical_task in &batch[..batch_len] {
                    let discovered_node = canonical_task.node;
                    iterations += 1;
                    let cache_key = canonical_task;
                    if self
                        .cached_jump_result((cache_key.node, cache_key.step_exp))
                        .is_some()
                    {
                        self.stats.jump_cache_hits += 1;
                        continue;
                    }
                    self.stats.jump_cache_misses += 1;
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.nodes[discovered_node as usize].level;
                    assert!(discovered_level >= 2);

                    if self.nodes[discovered_node as usize].population == 0 {
                        let result = self.empty(discovered_level - 1);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_step0_dependents(cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level == 2 {
                        let result = self.base_transition(discovered_node);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_step0_dependents(cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level <= DENSE_SHORTCUT_MAX_LEVEL {
                        let result = self.dense_advance_centered(discovered_node, 0);
                        self.insert_jump_result((cache_key.node, cache_key.step_exp), result);
                        notify_step0_dependents(cache_key, &mut tasks, &mut dependents, &mut ready);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    let [
                        upper_left,
                        upper_center,
                        upper_right,
                        middle_left,
                        center,
                        middle_right,
                        lower_left,
                        lower_center,
                        lower_right,
                    ] = self.overlapping_subnodes(discovered_node);
                    let next_upper_left = self.centered_subnode(upper_left);
                    let next_upper_center = self.centered_subnode(upper_center);
                    let next_upper_right = self.centered_subnode(upper_right);
                    let next_middle_left = self.centered_subnode(middle_left);
                    let next_center = self.centered_subnode(center);
                    let next_middle_right = self.centered_subnode(middle_right);
                    let next_lower_left = self.centered_subnode(lower_left);
                    let next_lower_center = self.centered_subnode(lower_center);
                    let next_lower_right = self.centered_subnode(lower_right);
                    let combined_upper_left = self.join(
                        next_upper_left,
                        next_upper_center,
                        next_middle_left,
                        next_center,
                    );
                    let combined_upper_right = self.join(
                        next_upper_center,
                        next_upper_right,
                        next_center,
                        next_middle_right,
                    );
                    let combined_lower_left = self.join(
                        next_middle_left,
                        next_center,
                        next_lower_left,
                        next_lower_center,
                    );
                    let combined_lower_right = self.join(
                        next_center,
                        next_middle_right,
                        next_lower_center,
                        next_lower_right,
                    );
                    let task_id = tasks.len();
                    task_index.insert(canonical_task, task_id);
                    tasks.push(Some(Step0TaskRecord {
                        remaining: 0,
                        children: [
                            combined_upper_left,
                            combined_upper_right,
                            combined_lower_left,
                            combined_lower_right,
                        ],
                    }));
                    task_keys.push(Some(canonical_task));
                    self.stats.scheduler_tasks += 1;

                    for child_node in [
                        combined_lower_right,
                        combined_lower_left,
                        combined_upper_right,
                        combined_upper_left,
                    ] {
                        let child_key = self.canonical_jump_key((child_node, 0)).0;
                        if self.cached_jump_result((child_node, 0)).is_none() {
                            dependents.entry(child_key).or_default().push(task_id);
                            tasks[task_id].as_mut().unwrap().remaining += 1;
                            if !task_index.contains_key(&child_key) {
                                discover.push(child_key);
                            }
                        }
                    }
                    if tasks[task_id].as_ref().unwrap().remaining == 0 {
                        ready.push(task_id);
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                    }
                }
            }

            if self.cached_jump_result(root_key).is_some() {
                break;
            }

            let Some(task_id) = ready.pop() else {
                let sample = task_index.iter().next().map(|(&pending_key, &task_id)| {
                    let task = tasks[task_id].unwrap();
                    (pending_key.node, task.remaining, task.children)
                });
                panic!(
                    "hashlife step-0 dependency resolution stalled root_node={root_node} pending={} ready={} cache={} sample={sample:?}",
                    task_index.len(),
                    ready.len(),
                    self.jump_cache.len(),
                );
            };
            let Some(task_key) = task_keys[task_id].take() else {
                continue;
            };
            task_index.remove(&task_key);
            let task = tasks[task_id].take().unwrap();
            debug_assert_eq!(task.remaining, 0);
            let [nw, ne, sw, se] = task.children;
            let q00 = self.jump_result((nw, 0));
            let q01 = self.jump_result((ne, 0));
            let q10 = self.jump_result((sw, 0));
            let q11 = self.jump_result((se, 0));
            let result = self.join(q00, q01, q10, q11);
            self.insert_jump_result((task_key.node, 0), result);
            notify_step0_dependents(task_key, &mut tasks, &mut dependents, &mut ready);
            if ready.len() > self.stats.scheduler_ready_max {
                self.stats.scheduler_ready_max = ready.len();
            }
        }

        if debug {
            eprintln!(
                "[hashlife] advance_step0 done root={root_node} iterations={iterations} cache={} pending={} ready={}",
                self.jump_cache.len(),
                task_index.len(),
                ready.len(),
            );
        }

        self.jump_result(root_key)
    }

    fn overlapping_subnodes(&mut self, node: NodeId) -> [NodeId; 9] {
        let (canonical_node, symmetry) = self.canonicalized_cache_node(node);
        if let Some(&overlaps) = self.overlap_cache.get(&canonical_node) {
            self.stats.overlap_cache_hits += 1;
            return symmetry.inverse().transform_overlap_nodes(self, overlaps);
        }
        self.stats.overlap_cache_misses += 1;
        let node_ref = &self.nodes[canonical_node as usize];
        let nw = node_ref.nw;
        let ne = node_ref.ne;
        let sw = node_ref.sw;
        let se = node_ref.se;
        let nw_node = &self.nodes[nw as usize];
        let nw_ne = nw_node.ne;
        let nw_se = nw_node.se;
        let nw_sw = nw_node.sw;
        let ne_node = &self.nodes[ne as usize];
        let ne_nw = ne_node.nw;
        let ne_sw = ne_node.sw;
        let ne_se = ne_node.se;
        let sw_node = &self.nodes[sw as usize];
        let sw_nw = sw_node.nw;
        let sw_ne = sw_node.ne;
        let sw_se = sw_node.se;
        let se_node = &self.nodes[se as usize];
        let se_nw = se_node.nw;
        let se_ne = se_node.ne;
        let se_sw = se_node.sw;

        let overlaps = [
            nw,
            self.join(nw_ne, ne_nw, nw_se, ne_sw),
            ne,
            self.join(nw_sw, nw_se, sw_nw, sw_ne),
            self.join(nw_se, ne_sw, sw_ne, se_nw),
            self.join(ne_sw, ne_se, se_nw, se_ne),
            sw,
            self.join(sw_ne, se_nw, sw_se, se_sw),
            se,
        ];
        self.overlap_cache.insert(canonical_node, overlaps);
        symmetry.inverse().transform_overlap_nodes(self, overlaps)
    }

    fn centered_subnode(&mut self, node: NodeId) -> NodeId {
        let node_ref = &self.nodes[node as usize];
        debug_assert!(node_ref.level >= 1);
        if node_ref.level == 1 {
            return node;
        }

        let nw_se = self.nodes[node_ref.nw as usize].se;
        let ne_sw = self.nodes[node_ref.ne as usize].sw;
        let sw_ne = self.nodes[node_ref.sw as usize].ne;
        let se_nw = self.nodes[node_ref.se as usize].nw;
        self.join(nw_se, ne_sw, sw_ne, se_nw)
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

fn notify_dependents(
    key: &CanonicalJumpKey,
    tasks: &mut [Option<TaskRecord>],
    dependents: &mut HashMap<CanonicalJumpKey, Vec<usize>>,
    ready: &mut Vec<usize>,
) {
    if let Some(waiters) = dependents.remove(key) {
        for waiter_id in waiters {
            if let Some(task) = tasks[waiter_id].as_mut() {
                task.remaining -= 1;
                if task.remaining == 0 {
                    ready.push(waiter_id);
                }
            }
        }
    }
}

fn notify_step0_dependents(
    key: CanonicalJumpKey,
    tasks: &mut [Option<Step0TaskRecord>],
    dependents: &mut HashMap<CanonicalJumpKey, Vec<usize>>,
    ready: &mut Vec<usize>,
) {
    if let Some(waiters) = dependents.remove(&key) {
        for waiter_id in waiters {
            if let Some(task) = tasks[waiter_id].as_mut() {
                task.remaining -= 1;
                if task.remaining == 0 {
                    ready.push(waiter_id);
                }
            }
        }
    }
}

fn morton_key(x: u64, y: u64) -> u128 {
    let mut key = 0_u128;
    let mut bit = 0_u32;
    while bit < 64 {
        key |= ((x as u128 >> bit) & 1) << (bit * 2);
        key |= ((y as u128 >> bit) & 1) << (bit * 2 + 1);
        bit += 1;
    }
    key
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
