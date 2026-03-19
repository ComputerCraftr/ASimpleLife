use super::*;
use super::deps::{notify_dependents, notify_step0_dependents, push_dependent};

impl HashLifeEngine {
    fn probe_compacted_jump_cache_presence_batch<const N: usize>(
        &mut self,
        tasks: &[CompactedDiscoveredTask; N],
        active_lanes: usize,
    ) -> [bool; N] {
        let mut keys = [CanonicalJumpKey::empty(); N];
        for lane in 0..active_lanes {
            keys[lane] = tasks[lane].task.key;
        }
        self.probe_jump_cache_presence_batch(&keys, active_lanes)
    }

    fn probe_discovered_jump_cache_presence_batch<const N: usize>(
        &mut self,
        tasks: &[DiscoveredJumpTask; N],
        active_lanes: usize,
    ) -> [bool; N] {
        let mut keys = [CanonicalJumpKey::empty(); N];
        for lane in 0..active_lanes {
            keys[lane] = tasks[lane].key;
        }
        self.probe_jump_cache_presence_batch(&keys, active_lanes)
    }

    fn complete_recursive_fast_exit(
        &mut self,
        discovered: DiscoveredJumpTask,
        result: NodeId,
        tasks: &mut Vec<Option<TaskRecord>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        self.insert_jump_result((discovered.source_node, discovered.key.step_exp), result);
        notify_dependents(
            &discovered.key,
            tasks,
            dependents,
            dependent_edges,
            ready,
        );
        self.stats.scheduler_ready_max = self.stats.scheduler_ready_max.max(ready.len());
    }

    fn compact_discovered_jump_tasks<const N: usize>(
        &mut self,
        child_tasks: [DiscoveredJumpTask; N],
    ) -> ([CompactedDiscoveredTask; N], usize) {
        let mut compacted = [CompactedDiscoveredTask {
            task: DiscoveredJumpTask {
                key: CanonicalJumpKey::empty(),
                source_node: 0,
                canonical_packed: PackedNodeKey::new(0, [0; 4]),
            },
            duplicate_count: 0,
        }; N];
        let mut unique_count = 0;
        let mut unique_lookup = FlatTable::<CanonicalJumpKey, usize>::with_capacity(N.max(4));

        for child_key in child_tasks {
            if let Some(index) = unique_lookup.get(&child_key.key) {
                compacted[index].duplicate_count += 1;
            } else {
                compacted[unique_count] = CompactedDiscoveredTask {
                    task: child_key,
                    duplicate_count: 1,
                };
                unique_lookup.insert(child_key.key, unique_count);
                unique_count += 1;
            }
        }
        (compacted, unique_count)
    }

    fn probe_and_attach_recursive_parent_overlaps<const N: usize>(
        &mut self,
        parent_records: &mut [RecursiveParentBatchRecord; N],
        parent_count: usize,
    ) {
        let mut identities = [CanonicalNodeIdentity {
            packed: PackedNodeKey::new(0, [0; 4]),
            structural: CanonicalStructKey::new(0, [0; 4]),
            symmetry: Symmetry::Identity,
        }; N];
        let mut fingerprints = [0_u64; N];
        for lane in 0..parent_count {
            identities[lane] = CanonicalNodeIdentity {
                packed: parent_records[lane].discovered.canonical_packed,
                structural: parent_records[lane].canonical_structural,
                symmetry: Symmetry::Identity,
            };
            fingerprints[lane] = parent_records[lane].canonical_fingerprint;
        }
        let overlaps =
            self.probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, parent_count);
        for lane in 0..parent_count {
            parent_records[lane].overlaps = overlaps[lane];
        }
    }

    fn build_chunk_child_states<const N: usize>(
        &mut self,
        compacted: [CompactedDiscoveredTask; N],
        unique_count: usize,
        task_index: &FlatTable<CanonicalJumpKey, usize>,
    ) -> [ChunkChildState; N] {
        let present = self.probe_compacted_jump_cache_presence_batch(&compacted, unique_count);
        let mut states = [ChunkChildState {
            compacted: CompactedDiscoveredTask {
                task: DiscoveredJumpTask {
                    key: CanonicalJumpKey::empty(),
                    source_node: 0,
                    canonical_packed: PackedNodeKey::new(0, [0; 4]),
                },
                duplicate_count: 0,
            },
            present: false,
            blocked: false,
            enqueued: false,
        }; N];
        for index in 0..unique_count {
            let compacted_task = compacted[index];
            let blocked = present[index] || task_index.contains_key(&compacted_task.task.key);
            states[index] = ChunkChildState {
                compacted: compacted_task,
                present: present[index],
                blocked,
                enqueued: false,
            };
        }
        states
    }

    pub(in crate::hashlife) fn discovered_jump_tasks_from_nodes<const N: usize>(
        &mut self,
        child_nodes: [NodeId; N],
        step_exp: u32,
    ) -> [DiscoveredJumpTask; N] {
        let mut child_keys = [DiscoveredJumpTask {
            key: CanonicalJumpKey::empty(),
            source_node: 0,
            canonical_packed: PackedNodeKey::new(0, [0; 4]),
        }; N];
        let mut reuse = FlatTable::<NodeId, DiscoveredJumpTask>::with_capacity(N.max(4));
        for lane in 0..N {
            let child = child_nodes[lane];
            if let Some(reused) = reuse.get(&child) {
                child_keys[lane] = reused;
                continue;
            }
            let jump_probe = self.canonical_jump_probe((child, step_exp));
            let discovered = DiscoveredJumpTask {
                key: jump_probe.key,
                source_node: child,
                canonical_packed: jump_probe.node.packed,
            };
            child_keys[lane] = discovered;
            reuse.insert(child, discovered);
        }
        child_keys
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

        let mut fingerprints = [0_u64; N];
        self.stats.jump_presence_probe_batches += 1;
        self.stats.jump_presence_probe_lanes += active_lanes;
        self.stats.scheduler_probe_batches += 1;
        for lane in 0..active_lanes {
            fingerprints[lane] =
                hash_packed_jump_fingerprint(keys[lane].structural.fingerprint(), keys[lane].step_exp);
        }
        let cached =
            self.jump_cache
                .get_many_with_fingerprints(keys, &fingerprints, active_lanes);
        let mut hit_count = 0usize;
        for lane in 0..active_lanes {
            present[lane] = cached[lane].is_some();
            hit_count += usize::from(present[lane]);
        }
        self.stats.jump_presence_probe_hits += hit_count;
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
        let (compacted, unique_count) = self.compact_discovered_jump_tasks(child_keys);
        let chunk_child_states = self.build_chunk_child_states(compacted, unique_count, task_index);
        for index in 0..unique_count {
            if chunk_child_states[index].present {
                continue;
            }
            let compacted_child = chunk_child_states[index].compacted;
            let child_task = compacted_child.task;
            let child_key = child_task.key;
            for _ in 0..compacted_child.duplicate_count {
                push_dependent(dependents, dependent_edges, child_key, task_id);
                tasks[task_id].as_mut().unwrap().remaining += 1;
            }
            if !chunk_child_states[index].blocked {
                discover.push(child_task);
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
        let (compacted, unique_count) = self.compact_discovered_jump_tasks(child_keys);
        let chunk_child_states = self.build_chunk_child_states(compacted, unique_count, task_index);
        for index in 0..unique_count {
            if chunk_child_states[index].present {
                continue;
            }
            let compacted_child = chunk_child_states[index].compacted;
            let child_task = compacted_child.task;
            let child_key = child_task.key;
            for _ in 0..compacted_child.duplicate_count {
                push_dependent(dependents, dependent_edges, child_key, task_id);
                tasks[task_id].as_mut().unwrap().remaining += 1;
            }
            if !chunk_child_states[index].blocked {
                discover.push(child_task);
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
        let root_jump_probe = self.canonical_jump_probe((root_node, root_step_exp));
        discover.push(DiscoveredJumpTask {
            key: root_jump_probe.key,
            source_node: root_node,
            canonical_packed: root_jump_probe.node.packed,
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
            key: CanonicalJumpKey::empty(),
            source_node: 0,
            canonical_packed: PackedNodeKey::new(0, [0; 4]),
        }; DISCOVER_BATCH];
        let mut phase_one_candidates =
            Vec::<SimdProvisionalRecord>::with_capacity(SIMD_BATCH_LANES);
        let mut phase_two_candidates =
            Vec::<SimdProvisionalRecord>::with_capacity(SIMD_BATCH_LANES);
        let mut phase1_ready = [Phase1ReadyLane {
            task_id: 0,
            key: CanonicalJumpKey::empty(),
            next_exp: 0,
            inputs: [0; 9],
        }; SIMD_BATCH_LANES];
        let mut phase1_pending = 0usize;
        let mut phase2_ready = [Phase2ReadyLane {
            key: CanonicalJumpKey::empty(),
            next_exp: 0,
            inputs: [0; 4],
        }; SIMD_BATCH_LANES];
        let mut phase2_pending = 0usize;
        let mut iterations = 0_usize;
        let mut max_stack = discover.len();
        let mut next_iteration_log = HASHLIFE_DEBUG_ITERATION_LOG_INTERVAL;
        let mut next_stack_log = HASHLIFE_DEBUG_INITIAL_STACK_LOG_THRESHOLD;

        while self.cached_jump_result((root_node, root_step_exp)).is_none() {
            while !discover.is_empty() {
                let mut batch_len = 0;
                let mut parent_count = 0;
                let mut parent_records = [RecursiveParentBatchRecord {
                    discovered: DiscoveredJumpTask {
                        key: CanonicalJumpKey::empty(),
                        source_node: 0,
                        canonical_packed: PackedNodeKey::new(0, [0; 4]),
                    },
                    next_exp: 0,
                    canonical_structural: CanonicalStructKey::new(0, [0; 4]),
                    canonical_fingerprint: 0,
                    overlaps: [0; 9],
                    unique_child_tasks: [DiscoveredJumpTask {
                        key: CanonicalJumpKey::empty(),
                        source_node: 0,
                        canonical_packed: PackedNodeKey::new(0, [0; 4]),
                    }; 9],
                    unique_child_counts: [0; 9],
                    unique_child_count: 0,
                }; DISCOVER_BATCH];
                while batch_len < DISCOVER_BATCH {
                    let Some(entry) = discover.pop() else {
                        break;
                    };
                    batch[batch_len] = entry;
                    batch_len += 1;
                }
                let discovered_present =
                    self.probe_discovered_jump_cache_presence_batch(&batch, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let discovered = *discovered_task;
                    let canonical_task = discovered.key;
                    let discovered_node = discovered.source_node;
                    let discovered_step_exp = canonical_task.step_exp;
                    let discover_len = discover.len();
                    let ready_len = ready.len();
                    iterations += 1;
                    if discover_len > max_stack {
                        max_stack = discover_len;
                    }
                    if debug && iterations == next_iteration_log {
                        eprintln!(
                            "[hashlife] rec iter={iterations} discover={} ready={} task_index={} jump_cache={}",
                            discover_len,
                            ready_len,
                            task_index.len(),
                            self.jump_cache.len(),
                        );
                        next_iteration_log += HASHLIFE_DEBUG_ITERATION_LOG_INTERVAL;
                    }
                    if debug && max_stack >= next_stack_log {
                        eprintln!(
                            "[hashlife] rec max_stack={max_stack} discover={} ready={} task_index={} jump_cache={}",
                            discover_len,
                            ready_len,
                            task_index.len(),
                            self.jump_cache.len(),
                        );
                        next_stack_log *= 2;
                    }

                    if discovered_present[lane] {
                        self.stats.jump_presence_avoids += 1;
                        self.stats.simd_disabled_fast_exits += 1;
                        continue;
                    }
                    self.stats.jump_presence_misses += 1;
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.node_columns.level(discovered_node);
                    let discovered_population = self.node_columns.population(discovered_node);
                    assert!(discovered_level >= 3);

                    if discovered_population == 0 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.empty(discovered_level - 1);
                        self.complete_recursive_fast_exit(
                            discovered,
                            result,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        continue;
                    }

                    if discovered_level <= DENSE_SHORTCUT_MAX_LEVEL {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result =
                            self.dense_advance_centered(discovered_node, discovered_step_exp);
                        self.complete_recursive_fast_exit(
                            discovered,
                            result,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        continue;
                    }

                    if discovered_step_exp == 0 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.advance_one_generation_centered(discovered_node);
                        self.complete_recursive_fast_exit(
                            discovered,
                            result,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        continue;
                    }

                    parent_records[parent_count] = RecursiveParentBatchRecord {
                        discovered,
                        next_exp: discovered_step_exp - 1,
                        canonical_structural: canonical_task.structural,
                        canonical_fingerprint: canonical_task.structural.fingerprint(),
                        overlaps: [0; 9],
                        unique_child_tasks: [DiscoveredJumpTask {
                            key: CanonicalJumpKey::empty(),
                            source_node: 0,
                            canonical_packed: PackedNodeKey::new(0, [0; 4]),
                        }; 9],
                        unique_child_counts: [0; 9],
                        unique_child_count: 0,
                    };
                    parent_count += 1;
                }

                if parent_count != 0 {
                    self.stats.recursive_overlap_batch_batches += 1;
                    self.stats.recursive_overlap_batch_lanes += parent_count;
                    self.stats.cache_probe_batches += 1;
                    self.stats.scheduler_probe_batches += 1;
                    self.stats.overlap_prep_batches += 1;
                    self.probe_and_attach_recursive_parent_overlaps(&mut parent_records, parent_count);
                    const CHILD_CHUNK: usize = DISCOVER_BATCH * 9;
                    let mut chunk_child_lookup =
                        FlatTable::<CanonicalJumpKey, usize>::with_capacity(CHILD_CHUNK.max(8));
                    let mut chunk_unique_children = [CompactedDiscoveredTask {
                        task: DiscoveredJumpTask {
                            key: CanonicalJumpKey::empty(),
                            source_node: 0,
                            canonical_packed: PackedNodeKey::new(0, [0; 4]),
                        },
                        duplicate_count: 0,
                    }; CHILD_CHUNK];
                    let mut chunk_unique_count = 0usize;
                    for lane in 0..parent_count {
                        let child_nodes = [
                            parent_records[lane].overlaps[8],
                            parent_records[lane].overlaps[7],
                            parent_records[lane].overlaps[6],
                            parent_records[lane].overlaps[5],
                            parent_records[lane].overlaps[4],
                            parent_records[lane].overlaps[3],
                            parent_records[lane].overlaps[2],
                            parent_records[lane].overlaps[1],
                            parent_records[lane].overlaps[0],
                        ];
                        let discovered_children = self.discovered_jump_tasks_from_nodes(
                            child_nodes,
                            parent_records[lane].next_exp,
                        );
                        let mut parent_child_lookup =
                            FlatTable::<CanonicalJumpKey, usize>::with_capacity(16);
                        let mut unique_child_count = 0usize;
                        for child_task in discovered_children {
                            if let Some(existing) = parent_child_lookup.get(&child_task.key) {
                                parent_records[lane].unique_child_counts[existing] =
                                    parent_records[lane].unique_child_counts[existing]
                                        .saturating_add(1);
                            } else {
                                parent_records[lane].unique_child_tasks[unique_child_count] = child_task;
                                parent_records[lane].unique_child_counts[unique_child_count] = 1;
                                parent_child_lookup.insert(child_task.key, unique_child_count);
                                unique_child_count += 1;
                            }
                            if let Some(existing) = chunk_child_lookup.get(&child_task.key) {
                                chunk_unique_children[existing].duplicate_count = chunk_unique_children
                                    [existing]
                                    .duplicate_count
                                    .saturating_add(1);
                            } else {
                                chunk_unique_children[chunk_unique_count] = CompactedDiscoveredTask {
                                    task: child_task,
                                    duplicate_count: 1,
                                };
                                chunk_child_lookup.insert(child_task.key, chunk_unique_count);
                                chunk_unique_count += 1;
                            }
                        }
                        parent_records[lane].unique_child_count = unique_child_count as u8;
                    }
                    let mut chunk_child_states =
                        self.build_chunk_child_states(chunk_unique_children, chunk_unique_count, &task_index);
                    for lane in 0..parent_count {
                        let record = parent_records[lane];
                        let canonical_task = record.discovered.key;
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
                        for index in 0..usize::from(record.unique_child_count) {
                            let child_task = record.unique_child_tasks[index];
                            let child_key = child_task.key;
                            let chunk_index = chunk_child_lookup
                                .get(&child_key)
                                .expect("chunk child key must be present");
                            if chunk_child_states[chunk_index].present {
                                continue;
                            }
                            for _ in 0..record.unique_child_counts[index] {
                                push_dependent(
                                    &mut dependents,
                                    &mut dependent_edges,
                                    child_key,
                                    task_id,
                                );
                                task.remaining += 1;
                            }
                            if !chunk_child_states[chunk_index].blocked
                                && !chunk_child_states[chunk_index].enqueued
                            {
                                discover.push(child_task);
                                chunk_child_states[chunk_index].enqueued = true;
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
            ready.push(task_id);

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
                        phase1_ready[phase1_pending] = Phase1ReadyLane {
                            task_id,
                            key: task_key,
                            next_exp,
                            inputs: [q00, q01, q02, q10, q11, q12, q20, q21, q22],
                        };
                        phase1_pending += 1;
                        if phase1_pending == SIMD_BATCH_LANES {
                            self.build_phase1_provisional_records_batch(
                                &phase1_ready,
                                phase1_pending,
                                &mut phase_one_candidates,
                            );
                            phase1_pending = 0;
                            self.flush_recursive_simd_candidates(
                                false,
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
                        phase2_ready[phase2_pending] = Phase2ReadyLane {
                            key: task_key,
                            next_exp,
                            inputs: [nw, ne, sw, se],
                        };
                        phase2_pending += 1;
                        if phase2_pending == SIMD_BATCH_LANES {
                            self.build_phase2_provisional_records_batch(
                                &phase2_ready,
                                phase2_pending,
                                &mut phase_two_candidates,
                            );
                            phase2_pending = 0;
                            self.flush_recursive_simd_candidates(
                                true,
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

            if phase1_pending != 0 {
                self.build_phase1_provisional_records_batch(
                    &phase1_ready,
                    phase1_pending,
                    &mut phase_one_candidates,
                );
                phase1_pending = 0;
            }
            if phase2_pending != 0 {
                self.build_phase2_provisional_records_batch(
                    &phase2_ready,
                    phase2_pending,
                    &mut phase_two_candidates,
                );
                phase2_pending = 0;
            }

            self.flush_recursive_simd_candidates(
                false,
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
                true,
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
            return self.jump_result(root_key);
        }

        let debug = hashlife_debug_enabled();
        let level = self.node_columns.level(root_node) as usize;
        let task_capacity = 1usize << level.saturating_sub(1).min(10);
        let mut discover = Vec::with_capacity(task_capacity.max(8));
        let root_jump_probe = self.canonical_jump_probe((root_node, 0));
        discover.push(DiscoveredJumpTask {
            key: root_jump_probe.key,
            source_node: root_node,
            canonical_packed: root_jump_probe.node.packed,
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
            key: CanonicalJumpKey::empty(),
            source_node: 0,
            canonical_packed: PackedNodeKey::new(0, [0; 4]),
        }; DISCOVER_BATCH];
        let mut provisional_candidates = Vec::<DiscoveredJumpTask>::with_capacity(SIMD_BATCH_LANES);
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
                let discovered_present =
                    self.probe_discovered_jump_cache_presence_batch(&batch, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let canonical_task = discovered_task.key;
                    let discovered_node = discovered_task.source_node;
                    let discovered_population = self.node_columns.population(discovered_node);
                    iterations += 1;
                    if discovered_present[lane] {
                        self.stats.jump_presence_avoids += 1;
                        self.stats.simd_disabled_fast_exits += 1;
                        continue;
                    }
                    self.stats.jump_presence_misses += 1;
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.node_columns.level(discovered_node);
                    assert!(discovered_level >= 2);

                    if discovered_population == 0 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.empty(discovered_level - 1);
                        self.insert_jump_result((discovered_node, canonical_task.step_exp), result);
                        notify_step0_dependents(
                            canonical_task,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        self.stats.scheduler_ready_max =
                            self.stats.scheduler_ready_max.max(ready.len());
                        continue;
                    }

                    if discovered_level == 2 {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.base_transition(discovered_node);
                        self.insert_jump_result((discovered_node, canonical_task.step_exp), result);
                        notify_step0_dependents(
                            canonical_task,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        self.stats.scheduler_ready_max =
                            self.stats.scheduler_ready_max.max(ready.len());
                        continue;
                    }

                    if discovered_level <= DENSE_SHORTCUT_MAX_LEVEL {
                        self.stats.simd_disabled_fast_exits += 1;
                        let result = self.dense_advance_centered(discovered_node, 0);
                        self.insert_jump_result((discovered_node, canonical_task.step_exp), result);
                        notify_step0_dependents(
                            canonical_task,
                            &mut tasks,
                            &mut dependents,
                            &mut dependent_edges,
                            &mut ready,
                        );
                        self.stats.scheduler_ready_max =
                            self.stats.scheduler_ready_max.max(ready.len());
                        continue;
                    }

                    provisional_candidates.push(*discovered_task);
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
                    (
                        pending_key.structural,
                        pending_key.step_exp,
                        task.remaining,
                        task.children,
                    )
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
