use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;

use crate::bitgrid::BitGrid;

mod embed;
mod gc;
mod node;

type NodeId = u32;

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
pub struct HashLifeOracle {
    nodes: Vec<Node>,
    intern: HashMap<NodeKey, NodeId>,
    empty_by_level: Vec<NodeId>,
    jump_cache: HashMap<(NodeId, u32), NodeId>,
    root_result_cache: HashMap<(NodeId, u32), NodeId>,
    overlap_cache: HashMap<NodeId, [NodeId; 9]>,
    embed_layout_cache: HashMap<(u32, i32, i32, i32), u32>,
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
    root_size: i32,
    world_to_root_x: i32,
    world_to_root_y: i32,
    result_origin_x: i32,
    result_origin_y: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct HashLifeStats {
    jump_cache_hits: usize,
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
    builder_partitions: usize,
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
    key: u64,
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
    pub builder_partitions: usize,
}

const GC_MIN_NODES: usize = 4_096;
const GC_GROWTH_TRIGGER: usize = 1_024;
const GC_MIN_RECLAIM: usize = 256;

impl Default for HashLifeOracle {
    fn default() -> Self {
        let mut oracle = Self {
            nodes: Vec::new(),
            intern: HashMap::new(),
            empty_by_level: Vec::new(),
            jump_cache: HashMap::new(),
            root_result_cache: HashMap::new(),
            overlap_cache: HashMap::new(),
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

impl HashLifeOracle {
    fn cached_jump_result(&self, key: (NodeId, u32)) -> Option<NodeId> {
        self.jump_cache.get(&key).copied()
    }

    pub fn advance(&mut self, grid: &BitGrid, generations: u64) -> BitGrid {
        if generations == 0 || grid.is_empty() {
            return grid.clone();
        }

        let debug = hashlife_debug_enabled();
        let mut current = grid.clone();
        let mut remaining = generations;
        let mut last_root = None;
        let previous_root = self.retained_roots.last().copied();
        self.stats = HashLifeStats::default();
        self.clear_transient_state();

        if debug {
            eprintln!(
                "[hashlife] advance start gens={generations} pop={} bounds={:?}",
                current.population(),
                current.bounds()
            );
        }

        while remaining != 0 {
            let step_exp = 63 - remaining.leading_zeros();
            let step = 1_u64 << step_exp;
            if debug {
                eprintln!(
                    "[hashlife] segment step={step} step_exp={step_exp} pop={} bounds={:?}",
                    current.population(),
                    current.bounds()
                );
            }
            let (next, root) = self.advance_power_of_two(&current, step_exp);
            current = next;
            last_root = Some(root);
            if debug {
                eprintln!(
                    "[hashlife] segment done step={step} pop={} bounds={:?}",
                    current.population(),
                    current.bounds()
                );
            }
            remaining -= step;
        }

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

        current
    }

    fn advance_power_of_two(&mut self, grid: &BitGrid, step_exp: u32) -> (BitGrid, NodeId) {
        if grid.is_empty() {
            return (BitGrid::new(), self.empty(0));
        }

        let embedded = self.embed_for_jump(grid, step_exp);
        let cache_key = (embedded.root, step_exp);
        let advanced = if let Some(&cached) = self.root_result_cache.get(&cache_key) {
            self.stats.root_result_cache_hits += 1;
            cached
        } else {
            self.stats.root_result_cache_misses += 1;
            let result = self.advance_pow2(embedded.root, step_exp);
            self.root_result_cache.insert(cache_key, result);
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

    fn advance_power_of_two_recursive(&mut self, node: NodeId, step_exp: u32) -> NodeId {
        let debug = hashlife_debug_enabled();
        let level = self.nodes[node as usize].level as usize;
        let task_capacity = 1usize << level.saturating_sub(step_exp as usize + 1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        discover.push((node, step_exp));
        let mut task_index: HashMap<(NodeId, u32), usize> = HashMap::with_capacity(task_capacity);
        let mut tasks = Vec::<Option<TaskRecord>>::with_capacity(task_capacity);
        let mut task_keys = Vec::<Option<(NodeId, u32)>>::with_capacity(task_capacity);
        let mut dependents: HashMap<(NodeId, u32), Vec<usize>> = HashMap::with_capacity(task_capacity);
        let mut ready = Vec::<usize>::with_capacity(task_capacity);
        let mut iterations = 0_usize;
        let mut max_stack = discover.len();
        let mut next_iteration_log = 100_000_usize;
        let mut next_stack_log = 10_000_usize;

        while !self.jump_cache.contains_key(&(node, step_exp)) {
            while let Some((node, step_exp)) = discover.pop() {
                iterations += 1;
                if discover.len() > max_stack {
                    max_stack = discover.len();
                }
                if debug && iterations >= next_iteration_log {
                    eprintln!(
                        "[hashlife] advance_pow2 progress node={node} step_exp={step_exp} iterations={iterations} stack={} max_stack={} cache={} pending={} ready={}",
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
                        "[hashlife] advance_pow2 stack node={node} step_exp={step_exp} iterations={iterations} stack={} max_stack={} cache={} pending={} ready={}",
                        discover.len(),
                        max_stack,
                        self.jump_cache.len(),
                        task_index.len(),
                        ready.len(),
                    );
                    next_stack_log = next_stack_log.saturating_mul(2);
                }
                let key = (node, step_exp);
                if self.cached_jump_result(key).is_some() {
                    self.stats.jump_cache_hits += 1;
                    continue;
                }
                self.stats.jump_cache_misses += 1;
                if task_index.contains_key(&key) {
                    continue;
                }

                let level = self.nodes[node as usize].level;
                assert!(level >= 2);
                assert!(step_exp <= level - 2);

                if step_exp == 0 {
                    let result = self.advance_one_generation_centered(node);
                    self.jump_cache.insert(key, result);
                    notify_dependents(&key, &mut tasks, &mut dependents, &mut ready);
                    continue;
                }

                if self.nodes[node as usize].population == 0 {
                    let result = self.empty(level - 1);
                    self.jump_cache.insert(key, result);
                    notify_dependents(&key, &mut tasks, &mut dependents, &mut ready);
                    continue;
                }

                if level == 2 {
                    let result = self.base_transition(node);
                    self.jump_cache.insert(key, result);
                    notify_dependents(&key, &mut tasks, &mut dependents, &mut ready);
                    continue;
                }

                let [a, b, c, d, e, f, g, h, i] = self.overlapping_subnodes(node);
                let next_exp = step_exp - 1;
                let task_id = tasks.len();
                task_index.insert(key, task_id);
                tasks.push(Some(TaskRecord {
                    remaining: 0,
                    task: PendingTask::PhaseOne {
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
                    },
                }));
                task_keys.push(Some(key));
                for child in [i, h, g, f, e, d, c, b, a] {
                    let child_key = (child, next_exp);
                    if self.cached_jump_result(child_key).is_none() {
                        dependents.entry(child_key).or_default().push(task_id);
                        tasks[task_id].as_mut().unwrap().remaining += 1;
                        if !task_index.contains_key(&child_key) {
                            discover.push(child_key);
                        }
                    }
                }
                if tasks[task_id].as_ref().unwrap().remaining == 0 {
                    ready.push(task_id);
                }
                continue;
            }

            if self.jump_cache.contains_key(&(node, step_exp)) {
                break;
            }

            let Some(task_id) = ready.pop() else {
                let sample = task_index.iter().next().map(|(&(pending_node, pending_exp), &task_id)| {
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
                                .filter(|&child| self.cached_jump_result((child, next_exp)).is_none())
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
                                .filter(|&child| self.cached_jump_result((child, next_exp)).is_none())
                                .collect::<Vec<_>>(),
                        ),
                    };
                    (pending_node, pending_exp, recurse_exp, missing)
                });
                panic!(
                    "hashlife dependency resolution stalled for node={node} step_exp={step_exp} pending={} sample={sample:?}",
                    task_index.len()
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
                    let (pending_node, pending_exp) = task_key;
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
                    } = task.task else { unreachable!() };
                    let aa = self.jump_cache[&(a, next_exp)];
                    let bb = self.jump_cache[&(b, next_exp)];
                    let cc = self.jump_cache[&(c, next_exp)];
                    let dd = self.jump_cache[&(d, next_exp)];
                    let ee = self.jump_cache[&(e, next_exp)];
                    let ff = self.jump_cache[&(f, next_exp)];
                    let gg = self.jump_cache[&(g, next_exp)];
                    let hh = self.jump_cache[&(h, next_exp)];
                    let ii = self.jump_cache[&(i, next_exp)];

                    let nw = self.join(aa, bb, dd, ee);
                    let ne = self.join(bb, cc, ee, ff);
                    let sw = self.join(dd, ee, gg, hh);
                    let se = self.join(ee, ff, hh, ii);
                    let parent_key = (pending_node, pending_exp);
                    let parent_id = task_id;
                    task_index.insert(parent_key, parent_id);
                    task_keys[parent_id] = Some(parent_key);
                    tasks[parent_id] = Some(TaskRecord {
                        remaining: 0,
                        task: PendingTask::PhaseTwo {
                            next_exp,
                            nw,
                            ne,
                            sw,
                            se,
                        },
                    });
                    for child in [se, sw, ne, nw] {
                        let child_key = (child, next_exp);
                        if self.cached_jump_result(child_key).is_none() {
                            dependents.entry(child_key).or_default().push(parent_id);
                            tasks[parent_id].as_mut().unwrap().remaining += 1;
                            if !task_index.contains_key(&child_key) {
                                discover.push(child_key);
                            }
                        }
                    }
                    if tasks[parent_id].as_ref().unwrap().remaining == 0 {
                        ready.push(parent_id);
                    }
                }
                PendingTask::PhaseTwo { .. } => {
                    let (pending_node, pending_exp) = task_key;
                    let PendingTask::PhaseTwo {
                        next_exp,
                        nw,
                        ne,
                        sw,
                        se,
                    } = task.task else { unreachable!() };
                    let q00 = self.jump_cache[&(nw, next_exp)];
                    let q01 = self.jump_cache[&(ne, next_exp)];
                    let q10 = self.jump_cache[&(sw, next_exp)];
                    let q11 = self.jump_cache[&(se, next_exp)];
                    let result = self.join(q00, q01, q10, q11);
                    let key = (pending_node, pending_exp);
                    self.jump_cache.insert(key, result);
                    notify_dependents(&key, &mut tasks, &mut dependents, &mut ready);
                }
            }
        }

        if debug {
            eprintln!(
                "[hashlife] advance_pow2 done root={node} step_exp={step_exp} iterations={iterations} max_stack={} cache={} pending={} ready={}",
                max_stack,
                self.jump_cache.len(),
                task_index.len(),
                ready.len(),
            );
        }

        self.jump_cache[&(node, step_exp)]
    }

    fn advance_one_generation_centered(&mut self, node: NodeId) -> NodeId {
        let key = (node, 0);
        if self.jump_cache.contains_key(&key) {
            self.stats.jump_cache_hits += 1;
            return self.jump_cache[&key];
        }

        let debug = hashlife_debug_enabled();
        let level = self.nodes[node as usize].level as usize;
        let task_capacity = 1usize << level.saturating_sub(1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        discover.push(node);
        let mut task_index: HashMap<NodeId, usize> = HashMap::with_capacity(task_capacity);
        let mut tasks = Vec::<Option<Step0TaskRecord>>::with_capacity(task_capacity);
        let mut task_keys = Vec::<Option<NodeId>>::with_capacity(task_capacity);
        let mut dependents: HashMap<NodeId, Vec<usize>> = HashMap::with_capacity(task_capacity);
        let mut ready = Vec::<usize>::with_capacity(task_capacity);
        let mut iterations = 0_usize;

        while !self.jump_cache.contains_key(&key) {
            while let Some(node) = discover.pop() {
                iterations += 1;
                let key = (node, 0);
                if self.cached_jump_result(key).is_some() {
                    self.stats.jump_cache_hits += 1;
                    continue;
                }
                self.stats.jump_cache_misses += 1;
                if task_index.contains_key(&node) {
                    continue;
                }

                let level = self.nodes[node as usize].level;
                assert!(level >= 2);

                if self.nodes[node as usize].population == 0 {
                    let result = self.empty(level - 1);
                    self.jump_cache.insert(key, result);
                    notify_step0_dependents(node, &mut tasks, &mut dependents, &mut ready);
                    continue;
                }

                if level == 2 {
                    let result = self.base_transition(node);
                    self.jump_cache.insert(key, result);
                    notify_step0_dependents(node, &mut tasks, &mut dependents, &mut ready);
                    continue;
                }

                let [a, b, c, d, e, f, g, h, i] = self.overlapping_subnodes(node);
                let a = self.centered_subnode(a);
                let b = self.centered_subnode(b);
                let c = self.centered_subnode(c);
                let d = self.centered_subnode(d);
                let e = self.centered_subnode(e);
                let f = self.centered_subnode(f);
                let g = self.centered_subnode(g);
                let h = self.centered_subnode(h);
                let i = self.centered_subnode(i);
                let nw = self.join(a, b, d, e);
                let ne = self.join(b, c, e, f);
                let sw = self.join(d, e, g, h);
                let se = self.join(e, f, h, i);
                let task_id = tasks.len();
                task_index.insert(node, task_id);
                tasks.push(Some(Step0TaskRecord {
                    remaining: 0,
                    children: [nw, ne, sw, se],
                }));
                task_keys.push(Some(node));

                for child in [se, sw, ne, nw] {
                    if self.cached_jump_result((child, 0)).is_none() {
                        dependents.entry(child).or_default().push(task_id);
                        tasks[task_id].as_mut().unwrap().remaining += 1;
                        if !task_index.contains_key(&child) {
                            discover.push(child);
                        }
                    }
                }
                if tasks[task_id].as_ref().unwrap().remaining == 0 {
                    ready.push(task_id);
                }
            }

            if self.jump_cache.contains_key(&key) {
                break;
            }

            let Some(task_id) = ready.pop() else {
                let sample = task_index.iter().next().map(|(&pending_node, &task_id)| {
                    let task = tasks[task_id].unwrap();
                    (pending_node, task.remaining, task.children)
                });
                panic!(
                    "hashlife step-0 dependency resolution stalled for node={node} pending={} sample={sample:?}",
                    task_index.len()
                );
            };
            let Some(task_node) = task_keys[task_id].take() else {
                continue;
            };
            task_index.remove(&task_node);
            let task = tasks[task_id].take().unwrap();
            debug_assert_eq!(task.remaining, 0);
            let [nw, ne, sw, se] = task.children;
            let q00 = self.jump_cache[&(nw, 0)];
            let q01 = self.jump_cache[&(ne, 0)];
            let q10 = self.jump_cache[&(sw, 0)];
            let q11 = self.jump_cache[&(se, 0)];
            let result = self.join(q00, q01, q10, q11);
            self.jump_cache.insert((task_node, 0), result);
            notify_step0_dependents(task_node, &mut tasks, &mut dependents, &mut ready);
        }

        if debug {
            eprintln!(
                "[hashlife] advance_step0 done root={node} iterations={iterations} cache={} pending={} ready={}",
                self.jump_cache.len(),
                task_index.len(),
                ready.len(),
            );
        }

        self.jump_cache[&key]
    }

    fn overlapping_subnodes(&mut self, node: NodeId) -> [NodeId; 9] {
        if let Some(overlaps) = self.overlap_cache.get(&node).copied() {
            self.stats.overlap_cache_hits += 1;
            return overlaps;
        }
        self.stats.overlap_cache_misses += 1;
        let Node { nw, ne, sw, se, .. } = self.nodes[node as usize];
        let nw_node = self.nodes[nw as usize];
        let ne_node = self.nodes[ne as usize];
        let sw_node = self.nodes[sw as usize];
        let se_node = self.nodes[se as usize];

        let overlaps = [
            nw,
            self.join(nw_node.ne, ne_node.nw, nw_node.se, ne_node.sw),
            ne,
            self.join(nw_node.sw, nw_node.se, sw_node.nw, sw_node.ne),
            self.join(nw_node.se, ne_node.sw, sw_node.ne, se_node.nw),
            self.join(ne_node.sw, ne_node.se, se_node.nw, se_node.ne),
            sw,
            self.join(sw_node.ne, se_node.nw, sw_node.se, se_node.sw),
            se,
        ];
        self.overlap_cache.insert(node, overlaps);
        overlaps
    }

    fn centered_subnode(&mut self, node: NodeId) -> NodeId {
        let Node { level, nw, ne, sw, se, .. } = self.nodes[node as usize];
        debug_assert!(level >= 1);
        if level == 1 {
            return node;
        }

        let nw_node = self.nodes[nw as usize];
        let ne_node = self.nodes[ne as usize];
        let sw_node = self.nodes[sw as usize];
        let se_node = self.nodes[se as usize];
        self.join(nw_node.se, ne_node.sw, sw_node.ne, se_node.nw)
    }

}

fn hashlife_debug_enabled() -> bool {
    matches!(env::var("HASHLIFE_DEBUG").as_deref(), Ok("1") | Ok("true") | Ok("TRUE"))
}

fn notify_dependents(
    key: &(NodeId, u32),
    tasks: &mut [Option<TaskRecord>],
    dependents: &mut HashMap<(NodeId, u32), Vec<usize>>,
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
    node: NodeId,
    tasks: &mut [Option<Step0TaskRecord>],
    dependents: &mut HashMap<NodeId, Vec<usize>>,
    ready: &mut Vec<usize>,
) {
    if let Some(waiters) = dependents.remove(&node) {
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

fn morton_key(x: u32, y: u32) -> u64 {
    let mut key = 0_u64;
    let mut bit = 0_u32;
    while bit < 32 {
        key |= ((x as u64 >> bit) & 1) << (bit * 2);
        key |= ((y as u64 >> bit) & 1) << (bit * 2 + 1);
        bit += 1;
    }
    key
}

fn quadrant_end(
    cells: &[EmbeddedCell],
    start: usize,
    end: usize,
    bit_shift: u32,
    quadrant: u64,
) -> usize {
    let upper = quadrant + 1;
    start
        + cells[start..end]
            .partition_point(|cell| ((cell.key >> bit_shift) & 0b11) < upper)
}
