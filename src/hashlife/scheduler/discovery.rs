use super::*;
use super::deps::{notify_dependents, notify_step0_dependents, push_dependent};

impl HashLifeEngine {
    fn dedupe_discovered_jump_tasks<const N: usize>(
        child_tasks: [DiscoveredJumpTask; N],
    ) -> ([DiscoveredJumpTask; N], [u8; N], usize) {
        let mut unique_keys = [DiscoveredJumpTask {
            key: CanonicalJumpKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                step_exp: 0,
            },
            source_node: 0,
        }; N];
        let mut duplicate_counts = [0_u8; N];
        let mut unique_count = 0;

        for child_key in child_tasks {
            let mut existing = None;
            for index in 0..unique_count {
                if unique_keys[index].key == child_key.key {
                    existing = Some(index);
                    break;
                }
            }
            if let Some(index) = existing {
                duplicate_counts[index] += 1;
            } else {
                unique_keys[unique_count] = child_key;
                duplicate_counts[unique_count] = 1;
                unique_count += 1;
            }
        }

        (unique_keys, duplicate_counts, unique_count)
    }

    pub(in crate::hashlife) fn discovered_jump_tasks_from_nodes<const N: usize>(
        &mut self,
        child_nodes: [NodeId; N],
        step_exp: u32,
    ) -> [DiscoveredJumpTask; N] {
        let mut child_keys = [DiscoveredJumpTask {
            key: CanonicalJumpKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                step_exp: 0,
            },
            source_node: 0,
        }; N];
        for lane in 0..N {
            child_keys[lane] = DiscoveredJumpTask {
                key: self.canonical_jump_key_packed((child_nodes[lane], step_exp)).0,
                source_node: child_nodes[lane],
            };
        }
        child_keys
    }

    fn stage_pending_discovered_child_keys<const N: usize>(
        &mut self,
        unique_keys: &[DiscoveredJumpTask; N],
        duplicate_counts: &[u8; N],
        unique_count: usize,
    ) -> ([DiscoveredJumpTask; N], [u8; N], usize) {
        let mut present_keys = [CanonicalJumpKey {
            packed: PackedNodeKey::new(0, [0; 4]),
            step_exp: 0,
        }; N];
        for index in 0..unique_count {
            present_keys[index] = unique_keys[index].key;
        }
        let present = self.probe_jump_cache_presence_batch(&present_keys, unique_count);
        let mut pending_keys = [DiscoveredJumpTask {
            key: CanonicalJumpKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                step_exp: 0,
            },
            source_node: 0,
        }; N];
        let mut pending_counts = [0_u8; N];
        let mut pending_count = 0;
        for index in 0..unique_count {
            if present[index] {
                continue;
            }
            pending_keys[pending_count] = unique_keys[index];
            pending_counts[pending_count] = duplicate_counts[index];
            pending_count += 1;
        }
        (pending_keys, pending_counts, pending_count)
    }

    fn probe_jump_cache_presence_batch<const N: usize>(
        &mut self,
        keys: &[CanonicalJumpKey; N],
        active_lanes: usize,
    ) -> [bool; N] {
        let mut present = [false; N];
        if active_lanes == 0 {
            return present;
        }

        let mut packed_keys = [PackedJumpCacheKey {
            packed: PackedNodeKey::new(0, [0; 4]),
            step_exp: 0,
        }; N];
        let mut fingerprints = [0_u64; N];
        self.stats.jump_presence_probe_batches += 1;
        self.stats.jump_presence_probe_lanes += active_lanes;
        self.stats.scheduler_probe_batches += 1;
        for lane in 0..active_lanes {
            packed_keys[lane] = PackedJumpCacheKey {
                packed: keys[lane].packed,
                step_exp: keys[lane].step_exp,
            };
            fingerprints[lane] =
                hash_packed_jump_fingerprint(keys[lane].packed.fingerprint(), keys[lane].step_exp);
        }
        let cached =
            self.jump_cache
                .get_many_with_fingerprints(&packed_keys, &fingerprints, active_lanes);
        for lane in 0..active_lanes {
            present[lane] = cached[lane].is_some();
            self.stats.jump_presence_probe_hits += usize::from(present[lane]);
        }
        present
    }

    pub(in crate::hashlife::scheduler) fn schedule_recursive_children(
        &mut self,
        child_keys: [DiscoveredJumpTask; 4],
        task_id: usize,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut [Option<TaskRecord>],
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
    ) {
        let (unique_keys, duplicate_counts, unique_count) =
            Self::dedupe_discovered_jump_tasks(child_keys);
        let (pending_keys, pending_counts, pending_count) =
            self.stage_pending_discovered_child_keys(&unique_keys, &duplicate_counts, unique_count);
        for index in 0..pending_count {
            let child_key = pending_keys[index].key;
            for _ in 0..pending_counts[index] {
                push_dependent(dependents, dependent_edges, child_key, task_id);
                tasks[task_id].as_mut().unwrap().remaining += 1;
            }
            if !task_index.contains_key(&child_key) {
                discover.push(pending_keys[index]);
            }
        }
    }

    pub(in crate::hashlife::scheduler) fn schedule_step0_children(
        &mut self,
        child_nodes: [NodeId; 4],
        task_id: usize,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut [Option<Step0TaskRecord>],
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
    ) {
        let child_keys = self.discovered_jump_tasks_from_nodes(child_nodes, 0);
        let (unique_keys, duplicate_counts, unique_count) =
            Self::dedupe_discovered_jump_tasks(child_keys);
        let (pending_keys, pending_counts, pending_count) =
            self.stage_pending_discovered_child_keys(&unique_keys, &duplicate_counts, unique_count);
        for index in 0..pending_count {
            let child_key = pending_keys[index].key;
            for _ in 0..pending_counts[index] {
                push_dependent(dependents, dependent_edges, child_key, task_id);
                tasks[task_id].as_mut().unwrap().remaining += 1;
            }
            if !task_index.contains_key(&child_key) {
                discover.push(pending_keys[index]);
            }
        }
    }

    pub(in crate::hashlife::scheduler) fn advance_power_of_two_recursive_impl(
        &mut self,
        root_node: NodeId,
        root_step_exp: u32,
    ) -> NodeId {
        let debug = hashlife_debug_enabled();
        let level = self.node_columns.level(root_node) as usize;
        let task_capacity = 1usize << level.saturating_sub(root_step_exp as usize + 1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        discover.push(DiscoveredJumpTask {
            key: self.canonical_jump_key_packed((root_node, root_step_exp)).0,
            source_node: root_node,
        });
        let mut task_index: FlatTable<CanonicalJumpKey, usize> =
            FlatTable::with_capacity(task_capacity);
        let mut tasks = Vec::<Option<TaskRecord>>::with_capacity(task_capacity);
        let mut task_keys = Vec::<Option<CanonicalJumpKey>>::with_capacity(task_capacity);
        let mut dependents: FlatTable<CanonicalJumpKey, usize> =
            FlatTable::with_capacity(task_capacity);
        let mut dependent_edges =
            Vec::<DependentEdge>::with_capacity(task_capacity.saturating_mul(4));
        let mut ready = Vec::<usize>::with_capacity(task_capacity);
        let mut batch = [DiscoveredJumpTask {
            key: CanonicalJumpKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                step_exp: 0,
            },
            source_node: 0,
        }; DISCOVER_BATCH];
        let mut phase_one_candidates =
            Vec::<SimdProvisionalRecord>::with_capacity(SIMD_BATCH_LANES);
        let mut phase_two_candidates =
            Vec::<SimdProvisionalRecord>::with_capacity(SIMD_BATCH_LANES);
        let mut iterations = 0_usize;
        let mut max_stack = discover.len();
        let mut next_iteration_log = HASHLIFE_DEBUG_ITERATION_LOG_INTERVAL;
        let mut next_stack_log = HASHLIFE_DEBUG_INITIAL_STACK_LOG_THRESHOLD;

        while self.cached_jump_result((root_node, root_step_exp)).is_none() {
            while !discover.is_empty() {
                let mut batch_len = 0;
                let mut parent_count = 0;
                let mut parent_records = [RecursiveParentBatchRecord {
                    cache_key: CanonicalJumpKey {
                        packed: PackedNodeKey::new(0, [0; 4]),
                        step_exp: 0,
                    },
                    packed_parent: PackedNodeKey::new(0, [0; 4]),
                    packed_fingerprint: 0,
                    inverse_symmetry: Symmetry::Identity,
                    level: 0,
                    next_exp: 0,
                    overlaps: [0; 9],
                    child_keys: [CanonicalJumpKey {
                        packed: PackedNodeKey::new(0, [0; 4]),
                        step_exp: 0,
                    }; 9],
                    child_nodes: [0; 9],
                }; DISCOVER_BATCH];
                while batch_len < DISCOVER_BATCH {
                    let Some(entry) = discover.pop() else {
                        break;
                    };
                    batch[batch_len] = entry;
                    batch_len += 1;
                }
                let mut batch_keys = [CanonicalJumpKey {
                    packed: PackedNodeKey::new(0, [0; 4]),
                    step_exp: 0,
                }; DISCOVER_BATCH];
                for lane in 0..batch_len {
                    batch_keys[lane] = batch[lane].key;
                }
                let discovered_present =
                    self.probe_jump_cache_presence_batch(&batch_keys, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let canonical_task = discovered_task.key;
                    let discovered_node = discovered_task.source_node;
                    let discovered_step_exp = canonical_task.step_exp;
                    iterations += 1;
                    if discover.len() > max_stack {
                        max_stack = discover.len();
                    }
                    if debug && iterations == next_iteration_log {
                        eprintln!(
                            "[hashlife] rec iter={iterations} discover={} ready={} task_index={} jump_cache={}",
                            discover.len(),
                            ready.len(),
                            task_index.len(),
                            self.jump_cache.len(),
                        );
                        next_iteration_log += HASHLIFE_DEBUG_ITERATION_LOG_INTERVAL;
                    }
                    if debug && max_stack >= next_stack_log {
                        eprintln!(
                            "[hashlife] rec max_stack={max_stack} discover={} ready={} task_index={} jump_cache={}",
                            discover.len(),
                            ready.len(),
                            task_index.len(),
                            self.jump_cache.len(),
                        );
                        next_stack_log *= 2;
                    }

                    let cache_key = canonical_task;
                    if discovered_present[lane] {
                        self.stats.jump_cache_hits += 1;
                        self.stats.simd_disabled_fast_exits += 1;
                        continue;
                    }
                    self.stats.jump_cache_misses += 1;
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.node_columns.level(discovered_node);
                    assert!(discovered_level >= 3);

                    if self.node_columns.population(discovered_node) == 0 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.empty(discovered_level - 1);
                        self.insert_jump_result((discovered_node, cache_key.step_exp), result);
                        notify_dependents(
                            &cache_key,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level <= DENSE_SHORTCUT_MAX_LEVEL {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result =
                            self.dense_advance_centered(discovered_node, discovered_step_exp);
                        self.insert_jump_result((discovered_node, cache_key.step_exp), result);
                        notify_dependents(
                            &cache_key,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_step_exp == 0 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.advance_one_generation_centered(discovered_node);
                        self.insert_jump_result((discovered_node, cache_key.step_exp), result);
                        notify_dependents(
                            &cache_key,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    parent_records[parent_count] = RecursiveParentBatchRecord {
                        cache_key: canonical_task,
                        packed_parent: canonical_task.packed,
                        packed_fingerprint: canonical_task.packed.fingerprint(),
                        inverse_symmetry: Symmetry::Identity,
                        level: discovered_level,
                        next_exp: discovered_step_exp - 1,
                        overlaps: [0; 9],
                        child_keys: [CanonicalJumpKey {
                            packed: PackedNodeKey::new(0, [0; 4]),
                            step_exp: 0,
                        }; 9],
                        child_nodes: [0; 9],
                    };
                    parent_count += 1;
                }

                if parent_count != 0 {
                    self.stats.recursive_overlap_batch_batches += 1;
                    self.stats.recursive_overlap_batch_lanes += parent_count;
                    let mut parent_packed_keys = [PackedNodeKey::new(0, [0; 4]); DISCOVER_BATCH];
                    let mut parent_packed_fingerprints = [0_u64; DISCOVER_BATCH];
                    for lane in 0..parent_count {
                        parent_packed_keys[lane] = parent_records[lane].packed_parent;
                        parent_packed_fingerprints[lane] = parent_records[lane].packed_fingerprint;
                        debug_assert_eq!(parent_records[lane].inverse_symmetry, Symmetry::Identity);
                        debug_assert_eq!(parent_records[lane].level, parent_records[lane].packed_parent.level);
                        debug_assert_eq!(parent_records[lane].packed_parent, parent_records[lane].cache_key.packed);
                    }
                    self.stats.cache_probe_batches += 1;
                    self.stats.scheduler_probe_batches += 1;
                    self.stats.overlap_prep_batches += 1;
                    let overlaps = self.probe_and_build_canonical_overlaps_staged(
                        &parent_packed_keys,
                        &parent_packed_fingerprints,
                        parent_count,
                    );
                    for lane in 0..parent_count {
                        parent_records[lane].overlaps = overlaps[lane];
                        let child_nodes = [
                            overlaps[lane][8],
                            overlaps[lane][7],
                            overlaps[lane][6],
                            overlaps[lane][5],
                            overlaps[lane][4],
                            overlaps[lane][3],
                            overlaps[lane][2],
                            overlaps[lane][1],
                            overlaps[lane][0],
                        ];
                        let discovered_children = self
                            .discovered_jump_tasks_from_nodes(child_nodes, parent_records[lane].next_exp);
                        for index in 0..9 {
                            parent_records[lane].child_keys[index] = discovered_children[index].key;
                            parent_records[lane].child_nodes[index] = discovered_children[index].source_node;
                        }
                    }
                    for lane in 0..parent_count {
                        let record = parent_records[lane];
                        let canonical_task = record.cache_key;
                        let [q00, q01, q02, q10, q11, q12, q20, q21, q22] = record.overlaps;
                        let task_id = tasks.len();
                        self.stats.scheduler_tasks += 1;
                        task_index.insert(canonical_task, task_id);
                        tasks.push(Some(TaskRecord {
                            remaining: 0,
                            task: PendingTask::PhaseOne {
                                next_exp: record.next_exp,
                                a: q00,
                                b: q01,
                                c: q02,
                                d: q10,
                                e: q11,
                                f: q12,
                                g: q20,
                                h: q21,
                                i: q22,
                            },
                        }));
                        task_keys.push(Some(canonical_task));
                        let task = tasks[task_id].as_mut().unwrap();
                        let mut discovered_children = [DiscoveredJumpTask {
                            key: CanonicalJumpKey {
                                packed: PackedNodeKey::new(0, [0; 4]),
                                step_exp: 0,
                            },
                            source_node: 0,
                        }; 9];
                        for index in 0..9 {
                            discovered_children[index] = DiscoveredJumpTask {
                                key: record.child_keys[index],
                                source_node: record.child_nodes[index],
                            };
                        }
                        let (child_keys, child_duplicate_counts, child_unique_count) =
                            Self::dedupe_discovered_jump_tasks(discovered_children);
                        let (pending_child_keys, pending_child_counts, pending_child_count) =
                            self.stage_pending_discovered_child_keys(
                                &child_keys,
                                &child_duplicate_counts,
                                child_unique_count,
                            );
                        for index in 0..pending_child_count {
                            let child_key = pending_child_keys[index].key;
                            for _ in 0..pending_child_counts[index] {
                                push_dependent(
                                    &mut dependents,
                                    &mut dependent_edges,
                                    child_key,
                                    task_id,
                                );
                                task.remaining += 1;
                            }
                            if !task_index.contains_key(&child_key) {
                                discover.push(pending_child_keys[index]);
                            }
                        }
                        if task.remaining == 0 {
                            ready.push(task_id);
                            self.stats.phase1_ready_max =
                                self.stats.phase1_ready_max.max(ready.len());
                            self.stats.scheduler_ready_max =
                                self.stats.scheduler_ready_max.max(ready.len());
                        }
                    }
                }
            }

            if self.cached_jump_result((root_node, root_step_exp)).is_some() {
                break;
            }

            let Some(task_id) = ready.pop() else {
                self.stats.dependency_stalls += 1;
                panic!(
                    "hashlife recursive dependency resolution stalled root={root_node} step_exp={root_step_exp} pending={} ready={} cache={}",
                    task_index.len(),
                    ready.len(),
                    self.jump_cache.len(),
                );
            };
            {
                let Some(task) = tasks[task_id].take() else {
                    panic!("ready recursive task missing state for task_id={task_id}");
                };
                match task.task {
                    PendingTask::PhaseOne {
                        next_exp,
                        a: q00,
                        b: q01,
                        c: q02,
                        d: q10,
                        e: q11,
                        f: q12,
                        g: q20,
                        h: q21,
                        i: q22,
                    } => {
                        debug_assert_eq!(task.remaining, 0);
                        let parent_key = task_keys[task_id].take();
                        let task_key = parent_key.unwrap_or_else(|| {
                            panic!("phase1 task missing key for task_id={task_id}")
                        });
                        task_index.remove(&task_key);
                        phase_one_candidates.push(self.build_phase1_provisional_record(
                            task_id,
                            task_key,
                            next_exp,
                            [q00, q01, q02, q10, q11, q12, q20, q21, q22],
                        ));
                        if phase_one_candidates.len() == SIMD_BATCH_LANES {
                            self.flush_recursive_simd_candidates(
                                SimdTaskKind::PhaseOne,
                                &mut phase_one_candidates,
                                &mut discover,
                                &mut task_index,
                                &mut tasks,
                                &mut task_keys,
                                &mut dependents,
                                &mut dependent_edges,
                                &mut ready,
                            );
                        }
                    }
                    PendingTask::PhaseTwo {
                        next_exp,
                        nw,
                        ne,
                        sw,
                        se,
                    } => {
                        debug_assert_eq!(task.remaining, 0);
                        let task_key = task_keys[task_id]
                            .expect("phase2 task should always have a cached key");
                        task_keys[task_id] = None;
                        task_index.remove(&task_key);
                        phase_two_candidates.push(self.build_phase2_provisional_record(
                            task_id,
                            task_key,
                            next_exp,
                            [nw, ne, sw, se],
                        ));
                        if phase_two_candidates.len() == SIMD_BATCH_LANES {
                            self.flush_recursive_simd_candidates(
                                SimdTaskKind::PhaseTwo,
                                &mut phase_two_candidates,
                                &mut discover,
                                &mut task_index,
                                &mut tasks,
                                &mut task_keys,
                                &mut dependents,
                                &mut dependent_edges,
                                &mut ready,
                            );
                        }
                    }
                }
            }

            while let Some(task_id) = ready.pop() {
                let Some(task) = tasks[task_id].take() else {
                    panic!("drained recursive task missing state for task_id={task_id}");
                };
                match task.task {
                    PendingTask::PhaseOne {
                        next_exp,
                        a: q00,
                        b: q01,
                        c: q02,
                        d: q10,
                        e: q11,
                        f: q12,
                        g: q20,
                        h: q21,
                        i: q22,
                    } => {
                        debug_assert_eq!(task.remaining, 0);
                        let parent_key = task_keys[task_id].take();
                        let task_key = parent_key.unwrap_or_else(|| {
                            panic!("phase1 task missing key for task_id={task_id}")
                        });
                        task_index.remove(&task_key);
                        phase_one_candidates.push(self.build_phase1_provisional_record(
                            task_id,
                            task_key,
                            next_exp,
                            [q00, q01, q02, q10, q11, q12, q20, q21, q22],
                        ));
                        if phase_one_candidates.len() == SIMD_BATCH_LANES {
                            self.flush_recursive_simd_candidates(
                                SimdTaskKind::PhaseOne,
                                &mut phase_one_candidates,
                                &mut discover,
                                &mut task_index,
                                &mut tasks,
                                &mut task_keys,
                                &mut dependents,
                                &mut dependent_edges,
                                &mut ready,
                            );
                        }
                    }
                    PendingTask::PhaseTwo {
                        next_exp,
                        nw,
                        ne,
                        sw,
                        se,
                    } => {
                        debug_assert_eq!(task.remaining, 0);
                        let task_key = task_keys[task_id]
                            .expect("phase2 task should always have a cached key");
                        task_keys[task_id] = None;
                        task_index.remove(&task_key);
                        phase_two_candidates.push(self.build_phase2_provisional_record(
                            task_id,
                            task_key,
                            next_exp,
                            [nw, ne, sw, se],
                        ));
                        if phase_two_candidates.len() == SIMD_BATCH_LANES {
                            self.flush_recursive_simd_candidates(
                                SimdTaskKind::PhaseTwo,
                                &mut phase_two_candidates,
                                &mut discover,
                                &mut task_index,
                                &mut tasks,
                                &mut task_keys,
                                &mut dependents,
                                &mut dependent_edges,
                                &mut ready,
                            );
                        }
                    }
                }
            }

            self.flush_recursive_simd_candidates(
                SimdTaskKind::PhaseOne,
                &mut phase_one_candidates,
                &mut discover,
                &mut task_index,
                &mut tasks,
                &mut task_keys,
                &mut dependents,
                &mut dependent_edges,
                &mut ready,
            );
            self.flush_recursive_simd_candidates(
                SimdTaskKind::PhaseTwo,
                &mut phase_two_candidates,
                &mut discover,
                &mut task_index,
                &mut tasks,
                &mut task_keys,
                &mut dependents,
                &mut dependent_edges,
                &mut ready,
            );
        }

        if debug {
            eprintln!(
                "[hashlife] advance_pow2_recursive done root={root_node} step_exp={root_step_exp} iterations={iterations} max_stack={max_stack} cache={} pending={} ready={}",
                self.jump_cache.len(),
                task_index.len(),
                ready.len(),
            );
        }

        self.jump_result((root_node, root_step_exp))
    }

    pub(in crate::hashlife::scheduler) fn advance_one_generation_centered_impl(
        &mut self,
        root_node: NodeId,
    ) -> NodeId {
        let root_key = (root_node, 0);
        if self.cached_jump_result(root_key).is_some() {
            self.stats.jump_cache_hits += 1;
            return self.jump_result(root_key);
        }

        let debug = hashlife_debug_enabled();
        let level = self.node_columns.level(root_node) as usize;
        let task_capacity = 1usize << level.saturating_sub(1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        discover.push(DiscoveredJumpTask {
            key: self.canonical_jump_key_packed((root_node, 0)).0,
            source_node: root_node,
        });
        let mut task_index: FlatTable<CanonicalJumpKey, usize> =
            FlatTable::with_capacity(task_capacity);
        let mut tasks = Vec::<Option<Step0TaskRecord>>::with_capacity(task_capacity);
        let mut task_keys = Vec::<Option<CanonicalJumpKey>>::with_capacity(task_capacity);
        let mut dependents: FlatTable<CanonicalJumpKey, usize> =
            FlatTable::with_capacity(task_capacity);
        let mut dependent_edges =
            Vec::<DependentEdge>::with_capacity(task_capacity.saturating_mul(4));
        let mut ready = Vec::<usize>::with_capacity(task_capacity);
        let mut batch = [DiscoveredJumpTask {
            key: CanonicalJumpKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                step_exp: 0,
            },
            source_node: 0,
        }; DISCOVER_BATCH];
        let mut provisional_candidates = Vec::<CanonicalJumpKey>::with_capacity(SIMD_BATCH_LANES);
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
                let mut batch_keys = [CanonicalJumpKey {
                    packed: PackedNodeKey::new(0, [0; 4]),
                    step_exp: 0,
                }; DISCOVER_BATCH];
                for lane in 0..batch_len {
                    batch_keys[lane] = batch[lane].key;
                }
                let discovered_present =
                    self.probe_jump_cache_presence_batch(&batch_keys, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let canonical_task = discovered_task.key;
                    let discovered_node = discovered_task.source_node;
                    iterations += 1;
                    let cache_key = canonical_task;
                    if discovered_present[lane] {
                        self.stats.jump_cache_hits += 1;
                        self.stats.simd_disabled_fast_exits += 1;
                        continue;
                    }
                    self.stats.jump_cache_misses += 1;
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.node_columns.level(discovered_node);
                    assert!(discovered_level >= 2);

                    if self.node_columns.population(discovered_node) == 0 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.empty(discovered_level - 1);
                        self.insert_jump_result((discovered_node, cache_key.step_exp), result);
                        notify_step0_dependents(
                            cache_key,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level == 2 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.base_transition(discovered_node);
                        self.insert_jump_result((discovered_node, cache_key.step_exp), result);
                        notify_step0_dependents(
                            cache_key,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    if discovered_level <= DENSE_SHORTCUT_MAX_LEVEL {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.dense_advance_centered(discovered_node, 0);
                        self.insert_jump_result((discovered_node, cache_key.step_exp), result);
                        notify_step0_dependents(
                            cache_key,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        if ready.len() > self.stats.scheduler_ready_max {
                            self.stats.scheduler_ready_max = ready.len();
                        }
                        continue;
                    }

                    provisional_candidates.push(canonical_task);
                    if provisional_candidates.len() == SIMD_BATCH_LANES {
                        self.flush_step0_simd_candidates(
                            &mut provisional_candidates,
                            &mut discover,
                            &mut task_index,
                            &mut tasks,
                            &mut task_keys,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                    }
                }
                self.flush_step0_simd_candidates(
                    &mut provisional_candidates,
                    &mut discover,
                    &mut task_index,
                    &mut tasks,
                    &mut task_keys,
                    &mut dependents,
                    &mut dependent_edges,
                    &mut ready,
                );
            }

            if self.cached_jump_result(root_key).is_some() {
                break;
            }

            let Some(task_id) = ready.pop() else {
                let sample = task_index.iter().next().map(|(pending_key, task_id)| {
                    let task = tasks[task_id].unwrap();
                    (pending_key.packed, pending_key.step_exp, task.remaining, task.children)
                });
                self.stats.dependency_stalls += 1;
                panic!(
                    "hashlife step-0 dependency resolution stalled root_node={root_node} pending={} ready={} cache={} sample={sample:?}",
                    task_index.len(),
                    ready.len(),
                    self.jump_cache.len(),
                );
            };
            let Some(task_key) = task_keys[task_id].take() else {
                panic!("step0 task missing key for task_id={task_id}");
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
            self.insert_canonical_jump_result(task_key, result);
            notify_step0_dependents(
                task_key,
                &mut tasks,
                &mut dependents,
                &mut dependent_edges,
                &mut ready,
            );
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
}
