use super::*;
use super::deps::{notify_dependents, notify_step0_dependents, push_dependent};

mod arena;

impl HashLifeEngine {
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
        let mut batch_keys = [CanonicalJumpKey::empty(); DISCOVER_BATCH];
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
        let mut parent_child_arena = Vec::<RecursiveParentChildRef>::with_capacity(DISCOVER_BATCH * 9);

        while self.cached_jump_result((root_node, root_step_exp)).is_none() {
            while !discover.is_empty() {
                let batch_len = Self::drain_discover_batch(&mut discover, &mut batch, &mut batch_keys);
                let mut parent_records =
                    Vec::<RecursiveParentBatchRecord>::with_capacity(batch_len);
                let discovered_present = self.probe_jump_cache_presence_batch(&batch_keys, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let discovered = *discovered_task;
                    let canonical_task = discovered.key;
                    let discovered_node = discovered.source_node;
                    let discovered_step_exp = canonical_task.step_exp;
                    if discovered_present[lane] {
                        self.stats.simd_disabled_fast_exits += 1;
                        continue;
                    }
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

                    parent_records.push(RecursiveParentBatchRecord {
                        discovered,
                        next_exp: discovered_step_exp - 1,
                        canonical_structural: canonical_task.structural,
                        canonical_fingerprint: canonical_task.structural.fingerprint(),
                        overlaps: [0; 9],
                        child_arena_start: 0,
                        child_arena_len: 0,
                    });
                }

                if !parent_records.is_empty() {
                    self.stats.recursive_overlap_batch_batches += 1;
                    self.stats.recursive_overlap_batch_lanes += parent_records.len();
                    self.stats.cache_probe_batches += 1;
                    self.stats.scheduler_probe_batches += 1;
                    self.stats.overlap_prep_batches += 1;
                    self.probe_and_attach_recursive_parent_overlaps(&mut parent_records);
                    const CHILD_CHUNK: usize = DISCOVER_BATCH * 9;
                    let (mut chunk_child_states, _) = self
                        .build_recursive_parent_chunk_child_states::<CHILD_CHUNK>(
                            &mut parent_records,
                            &task_index,
                            &mut parent_child_arena,
                        );
                    for record in &parent_records {
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
                        let child_range_start = usize::from(record.child_arena_start);
                        let child_range_end = child_range_start + usize::from(record.child_arena_len);
                        for child_ref in &parent_child_arena[child_range_start..child_range_end] {
                            let chunk_index = usize::from(child_ref.query_index);
                            let child_task = chunk_child_states[chunk_index].compacted.task;
                            let child_key = child_task.key;
                            if chunk_child_states[chunk_index].present {
                                continue;
                            }
                            for _ in 0..child_ref.duplicate_count {
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

            if ready.is_empty() {
                self.stats.dependency_stalls += 1;
                panic!(
                    "hashlife recursive dependency resolution stalled root={root_node} step_exp={root_step_exp} pending={} ready={} cache={}",
                    task_index.len(),
                    ready.len(),
                    self.jump_cache.len(),
                );
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
        let mut batch_keys = [CanonicalJumpKey::empty(); DISCOVER_BATCH];
        let mut provisional_candidates = Vec::<DiscoveredJumpTask>::with_capacity(SIMD_BATCH_LANES);
        while self.cached_jump_result(root_key).is_none() {
            while !discover.is_empty() {
                let batch_len = Self::drain_discover_batch(&mut discover, &mut batch, &mut batch_keys);
                let discovered_present = self.probe_jump_cache_presence_batch(&batch_keys, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let canonical_task = discovered_task.key;
                    let discovered_node = discovered_task.source_node;
                    let discovered_population = self.node_columns.population(discovered_node);
                    if discovered_present[lane] {
                        self.stats.simd_disabled_fast_exits += 1;
                        continue;
                    }
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

        self.jump_result(root_key)
    }
}
