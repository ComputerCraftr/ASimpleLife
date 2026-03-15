use super::deps::{notify_dependents, notify_step0_dependents, push_dependent};
use super::*;
use crate::RequiredExt;

mod arena;

impl HashLifeEngine {
    pub(in crate::hashlife::scheduler) fn schedule_recursive_children(
        &mut self,
        child_keys: [DiscoveredJumpTask; 4],
        task_id: usize,
        state: &mut RecursiveSchedulerState<'_>,
    ) {
        let (compacted, unique_count) = self.compact_discovered_jump_tasks(child_keys);
        let chunk_child_states =
            self.build_chunk_child_states(compacted, unique_count, state.task_index);
        for child_state in &chunk_child_states[..unique_count] {
            if child_state.present {
                continue;
            }
            let compacted_child = child_state.compacted;
            let child_task = compacted_child.task;
            let child_key = child_task.key;
            for _ in 0..compacted_child.duplicate_count {
                if !push_dependent(
                    self,
                    state.dependents,
                    state.dependent_edges,
                    child_key,
                    task_id,
                ) {
                    return;
                }
                state.tasks[task_id]
                    .as_mut()
                    .or_invariant("required value")
                    .remaining += 1;
            }
            if !child_state.blocked && !self.try_push_transient(state.discover, child_task) {
                return;
            }
        }
    }

    pub(in crate::hashlife::scheduler) fn schedule_step0_children(
        &mut self,
        child_nodes: [NodeId; 4],
        task_id: usize,
        state: &mut Step0SchedulerState<'_>,
    ) {
        let child_keys = self.discovered_jump_tasks_from_nodes(child_nodes, 0);
        let (compacted, unique_count) = self.compact_discovered_jump_tasks(child_keys);
        let chunk_child_states =
            self.build_chunk_child_states(compacted, unique_count, state.task_index);
        for child_state in &chunk_child_states[..unique_count] {
            if child_state.present {
                continue;
            }
            let compacted_child = child_state.compacted;
            let child_task = compacted_child.task;
            let child_key = child_task.key;
            for _ in 0..compacted_child.duplicate_count {
                if !push_dependent(
                    self,
                    state.dependents,
                    state.dependent_edges,
                    child_key,
                    task_id,
                ) {
                    return;
                }
                state.tasks[task_id]
                    .as_mut()
                    .or_invariant("required value")
                    .remaining += 1;
            }
            if !child_state.blocked && !self.try_push_transient(state.discover, child_task) {
                return;
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
        let Some(mut discover) = self.try_transient_vec(task_capacity.max(8)) else {
            return self.dead_leaf;
        };
        let root_jump_probe = self.canonical_jump_probe((root_node, root_step_exp));
        if self.allocation_failure.is_some() {
            return self.dead_leaf;
        }
        discover.push(DiscoveredJumpTask {
            key: root_jump_probe.key,
            source_node: root_node,
            canonical_packed: root_jump_probe.node.packed,
        });
        let Some(mut task_index) = self.try_transient_flat_table(task_capacity) else {
            return self.dead_leaf;
        };
        let Some(mut tasks) = self.try_transient_vec::<Option<TaskRecord>>(task_capacity) else {
            return self.dead_leaf;
        };
        let Some(mut task_keys) = self.try_transient_vec::<Option<CanonicalJumpKey>>(task_capacity)
        else {
            return self.dead_leaf;
        };
        let Some(mut dependents) = self.try_transient_flat_table(task_capacity) else {
            return self.dead_leaf;
        };
        let Some(mut dependent_edges) =
            self.try_transient_vec::<DependentEdge>(task_capacity.saturating_mul(4))
        else {
            return self.dead_leaf;
        };
        let Some(mut ready) = self.try_transient_vec::<usize>(task_capacity) else {
            return self.dead_leaf;
        };
        let mut batch = [DiscoveredJumpTask {
            key: CanonicalJumpKey::empty(),
            source_node: 0,
            canonical_packed: PackedNodeKey::new(0, [0; 4]),
        }; DISCOVER_BATCH];
        let mut batch_keys = [CanonicalJumpKey::empty(); DISCOVER_BATCH];
        let Some(mut phase_one_candidates) =
            self.try_transient_vec::<SimdProvisionalRecord>(SIMD_BATCH_LANES)
        else {
            return self.dead_leaf;
        };
        let Some(mut phase_two_candidates) =
            self.try_transient_vec::<SimdProvisionalRecord>(SIMD_BATCH_LANES)
        else {
            return self.dead_leaf;
        };
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
        let Some(mut parent_child_arena) =
            self.try_transient_vec::<RecursiveParentChildRef>(DISCOVER_BATCH * 9)
        else {
            return self.dead_leaf;
        };

        while self
            .cached_jump_result((root_node, root_step_exp))
            .is_none()
        {
            if self.allocation_failure.is_some() {
                return self.dead_leaf;
            }
            while !discover.is_empty() {
                if self.allocation_failure.is_some() {
                    return self.dead_leaf;
                }
                let batch_len =
                    Self::drain_discover_batch(&mut discover, &mut batch, &mut batch_keys);
                let Some(mut parent_records) =
                    self.try_transient_vec::<RecursiveParentBatchRecord>(batch_len)
                else {
                    return self.dead_leaf;
                };
                let mut base_tasks = [batch[0]; SIMD_BATCH_LANES];
                let mut base_nodes = [self.dead_leaf; SIMD_BATCH_LANES];
                let mut base_count = 0;
                let discovered_present =
                    self.probe_jump_cache_presence_batch(&batch_keys, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let discovered = *discovered_task;
                    let canonical_task = discovered.key;
                    let discovered_node = discovered.source_node;
                    let discovered_step_exp = canonical_task.step_exp;
                    if discovered_present[lane] {
                        self.stats.scheduler.simd_disabled_fast_exits += 1;
                        continue;
                    }
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.node_columns.level(discovered_node);
                    let discovered_population = self.node_columns.population(discovered_node);
                    assert!(discovered_level >= 2);

                    if discovered_population == 0 {
                        self.stats.scheduler.simd_disabled_fast_exits += 1;
                        let result = self.empty(discovered_level - 1);
                        self.complete_recursive_fast_exit(
                            discovered,
                            result,
                            &mut tasks,
                            &mut dependents,
                            &dependent_edges,
                            &mut ready,
                        );
                        continue;
                    }

                    if discovered_level == 2 {
                        debug_assert_eq!(discovered_step_exp, 0);
                        base_tasks[base_count] = discovered;
                        base_nodes[base_count] = discovered_node;
                        base_count += 1;
                        continue;
                    }

                    if discovered_step_exp == 0 {
                        self.stats.scheduler.simd_disabled_fast_exits += 1;
                        let result = self.advance_one_generation_centered(discovered_node);
                        self.complete_recursive_fast_exit(
                            discovered,
                            result,
                            &mut tasks,
                            &mut dependents,
                            &dependent_edges,
                            &mut ready,
                        );
                        continue;
                    }

                    if !self.try_push_transient(
                        &mut parent_records,
                        RecursiveParentBatchRecord {
                            discovered,
                            next_exp: discovered_step_exp - 1,
                            canonical_structural: canonical_task.structural,
                            canonical_fingerprint: canonical_task.structural.fingerprint(),
                            overlaps: [0; 9],
                            child_arena_start: 0,
                            child_arena_len: 0,
                        },
                    ) {
                        return self.dead_leaf;
                    }
                }

                if base_count != 0 {
                    let base_results = self.base_transition_batch(base_nodes, base_count);
                    for lane in 0..base_count {
                        self.complete_recursive_fast_exit(
                            base_tasks[lane],
                            base_results[lane],
                            &mut tasks,
                            &mut dependents,
                            &dependent_edges,
                            &mut ready,
                        );
                    }
                }

                if !parent_records.is_empty() {
                    self.stats.simd.recursive_overlap_batch_batches += 1;
                    self.stats.simd.recursive_overlap_batch_lanes += parent_records.len();
                    self.stats.scheduler.cache_probe_batches += 1;
                    self.stats.scheduler.scheduler_probe_batches += 1;
                    self.stats.simd.overlap_prep_batches += 1;
                    if !self.probe_and_attach_recursive_parent_overlaps(&mut parent_records) {
                        return self.dead_leaf;
                    }
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
                        self.stats.scheduler.scheduler_tasks += 1;
                        if !self.try_insert_transient_table(
                            &mut task_index,
                            canonical_task,
                            task_id,
                        ) || !self.try_push_transient(
                            &mut tasks,
                            Some(TaskRecord {
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
                            }),
                        ) || !self.try_push_transient(&mut task_keys, Some(canonical_task))
                        {
                            return self.dead_leaf;
                        }
                        let task = tasks[task_id].as_mut().or_invariant("required value");
                        let child_range_start = usize::from(record.child_arena_start);
                        let child_range_end =
                            child_range_start + usize::from(record.child_arena_len);
                        for child_ref in &parent_child_arena[child_range_start..child_range_end] {
                            let chunk_index = usize::from(child_ref.query_index);
                            let child_task = chunk_child_states[chunk_index].compacted.task;
                            let child_key = child_task.key;
                            if chunk_child_states[chunk_index].present {
                                continue;
                            }
                            for _ in 0..child_ref.duplicate_count {
                                if !push_dependent(
                                    self,
                                    &mut dependents,
                                    &mut dependent_edges,
                                    child_key,
                                    task_id,
                                ) {
                                    return self.dead_leaf;
                                }
                                task.remaining += 1;
                            }
                            if !chunk_child_states[chunk_index].blocked
                                && !chunk_child_states[chunk_index].enqueued
                            {
                                if !self.try_push_transient(&mut discover, child_task) {
                                    return self.dead_leaf;
                                }
                                chunk_child_states[chunk_index].enqueued = true;
                            }
                        }
                        if task.remaining == 0 {
                            if !self.try_push_transient(&mut ready, task_id) {
                                return self.dead_leaf;
                            }
                            self.stats.scheduler.phase1_ready_max =
                                self.stats.scheduler.phase1_ready_max.max(ready.len());
                            self.stats.scheduler.scheduler_ready_max =
                                self.stats.scheduler.scheduler_ready_max.max(ready.len());
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

            if ready.is_empty() {
                self.stats.scheduler.dependency_stalls += 1;
                crate::invariant_failure!(
                    "hashlife recursive dependency resolution stalled root={root_node} step_exp={root_step_exp} pending={} ready={} cache={}",
                    task_index.len(),
                    ready.len(),
                    self.result_caches.jump.len(),
                );
            }

            while let Some(task_id) = ready.pop() {
                let Some(task) = tasks[task_id].take() else {
                    crate::invariant_failure!(
                        "drained recursive task missing state for task_id={task_id}"
                    );
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
                            crate::invariant_failure!(
                                "phase1 task missing key for task_id={task_id}"
                            )
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
                            self.flush_recursive_kernel_candidates(
                                false,
                                &mut phase_one_candidates,
                                &mut RecursiveSchedulerState {
                                    discover: &mut discover,
                                    task_index: &mut task_index,
                                    tasks: &mut tasks,
                                    task_keys: &mut task_keys,
                                    dependents: &mut dependents,
                                    dependent_edges: &mut dependent_edges,
                                    ready: &mut ready,
                                },
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
                            .or_invariant("phase2 task should always have a cached key");
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
                            self.flush_recursive_kernel_candidates(
                                true,
                                &mut phase_two_candidates,
                                &mut RecursiveSchedulerState {
                                    discover: &mut discover,
                                    task_index: &mut task_index,
                                    tasks: &mut tasks,
                                    task_keys: &mut task_keys,
                                    dependents: &mut dependents,
                                    dependent_edges: &mut dependent_edges,
                                    ready: &mut ready,
                                },
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

            self.flush_recursive_kernel_candidates(
                false,
                &mut phase_one_candidates,
                &mut RecursiveSchedulerState {
                    discover: &mut discover,
                    task_index: &mut task_index,
                    tasks: &mut tasks,
                    task_keys: &mut task_keys,
                    dependents: &mut dependents,
                    dependent_edges: &mut dependent_edges,
                    ready: &mut ready,
                },
            );
            self.flush_recursive_kernel_candidates(
                true,
                &mut phase_two_candidates,
                &mut RecursiveSchedulerState {
                    discover: &mut discover,
                    task_index: &mut task_index,
                    tasks: &mut tasks,
                    task_keys: &mut task_keys,
                    dependents: &mut dependents,
                    dependent_edges: &mut dependent_edges,
                    ready: &mut ready,
                },
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
        let Some(mut discover) = self.try_transient_vec(task_capacity.max(8)) else {
            return self.dead_leaf;
        };
        let root_jump_probe = self.canonical_jump_probe((root_node, 0));
        if self.allocation_failure.is_some() {
            return self.dead_leaf;
        }
        discover.push(DiscoveredJumpTask {
            key: root_jump_probe.key,
            source_node: root_node,
            canonical_packed: root_jump_probe.node.packed,
        });
        let Some(mut task_index) = self.try_transient_flat_table(task_capacity) else {
            return self.dead_leaf;
        };
        let Some(mut tasks) = self.try_transient_vec::<Option<Step0TaskRecord>>(task_capacity)
        else {
            return self.dead_leaf;
        };
        let Some(mut task_keys) = self.try_transient_vec::<Option<CanonicalJumpKey>>(task_capacity)
        else {
            return self.dead_leaf;
        };
        let Some(mut dependents) = self.try_transient_flat_table(task_capacity) else {
            return self.dead_leaf;
        };
        let Some(mut dependent_edges) =
            self.try_transient_vec::<DependentEdge>(task_capacity.saturating_mul(4))
        else {
            return self.dead_leaf;
        };
        let Some(mut ready) = self.try_transient_vec::<usize>(task_capacity) else {
            return self.dead_leaf;
        };
        let mut batch = [DiscoveredJumpTask {
            key: CanonicalJumpKey::empty(),
            source_node: 0,
            canonical_packed: PackedNodeKey::new(0, [0; 4]),
        }; DISCOVER_BATCH];
        let mut batch_keys = [CanonicalJumpKey::empty(); DISCOVER_BATCH];
        let Some(mut provisional_candidates) =
            self.try_transient_vec::<DiscoveredJumpTask>(SIMD_BATCH_LANES)
        else {
            return self.dead_leaf;
        };
        while self.cached_jump_result(root_key).is_none() {
            if self.allocation_failure.is_some() {
                return self.dead_leaf;
            }
            while !discover.is_empty() {
                if self.allocation_failure.is_some() {
                    return self.dead_leaf;
                }
                let batch_len =
                    Self::drain_discover_batch(&mut discover, &mut batch, &mut batch_keys);
                let mut base_tasks = [batch[0]; SIMD_BATCH_LANES];
                let mut base_nodes = [self.dead_leaf; SIMD_BATCH_LANES];
                let mut base_count = 0;
                let discovered_present =
                    self.probe_jump_cache_presence_batch(&batch_keys, batch_len);
                for (lane, discovered_task) in batch[..batch_len].iter().enumerate() {
                    let canonical_task = discovered_task.key;
                    let discovered_node = discovered_task.source_node;
                    let discovered_population = self.node_columns.population(discovered_node);
                    if discovered_present[lane] {
                        self.stats.scheduler.simd_disabled_fast_exits += 1;
                        continue;
                    }
                    if task_index.contains_key(&canonical_task) {
                        continue;
                    }

                    let discovered_level = self.node_columns.level(discovered_node);
                    assert!(discovered_level >= 2);

                    if discovered_population == 0 {
                        self.stats.scheduler.simd_disabled_fast_exits += 1;
                        let result = self.empty(discovered_level - 1);
                        self.insert_jump_result((discovered_node, canonical_task.step_exp), result);
                        notify_step0_dependents(
                            self,
                            canonical_task,
                            &mut tasks,
                            &mut dependents,
                            &dependent_edges,
                            &mut ready,
                        );
                        self.stats.scheduler.scheduler_ready_max =
                            self.stats.scheduler.scheduler_ready_max.max(ready.len());
                        continue;
                    }

                    if discovered_level == 2 {
                        base_tasks[base_count] = *discovered_task;
                        base_nodes[base_count] = discovered_node;
                        base_count += 1;
                        continue;
                    }

                    if !self.try_push_transient(&mut provisional_candidates, *discovered_task) {
                        return self.dead_leaf;
                    }
                    if provisional_candidates.len() == SIMD_BATCH_LANES {
                        self.flush_step0_kernel_candidates(
                            &mut provisional_candidates,
                            &mut Step0SchedulerState {
                                discover: &mut discover,
                                task_index: &mut task_index,
                                tasks: &mut tasks,
                                task_keys: &mut task_keys,
                                dependents: &mut dependents,
                                dependent_edges: &mut dependent_edges,
                                ready: &mut ready,
                            },
                        );
                    }
                }
                if base_count != 0 {
                    let base_results = self.base_transition_batch(base_nodes, base_count);
                    for lane in 0..base_count {
                        let discovered_task = base_tasks[lane];
                        self.insert_jump_result(
                            (discovered_task.source_node, discovered_task.key.step_exp),
                            base_results[lane],
                        );
                        notify_step0_dependents(
                            self,
                            discovered_task.key,
                            &mut tasks,
                            &mut dependents,
                            &dependent_edges,
                            &mut ready,
                        );
                        self.stats.scheduler.scheduler_ready_max =
                            self.stats.scheduler.scheduler_ready_max.max(ready.len());
                    }
                }
                self.flush_step0_kernel_candidates(
                    &mut provisional_candidates,
                    &mut Step0SchedulerState {
                        discover: &mut discover,
                        task_index: &mut task_index,
                        tasks: &mut tasks,
                        task_keys: &mut task_keys,
                        dependents: &mut dependents,
                        dependent_edges: &mut dependent_edges,
                        ready: &mut ready,
                    },
                );
            }

            if self.cached_jump_result(root_key).is_some() {
                break;
            }

            let Some(task_id) = ready.pop() else {
                let sample = task_index.iter().next().map(|(pending_key, task_id)| {
                    let task = tasks[task_id].or_invariant("required value");
                    (
                        pending_key.structural,
                        pending_key.step_exp,
                        task.remaining,
                        task.children,
                    )
                });
                self.stats.scheduler.dependency_stalls += 1;
                crate::invariant_failure!(
                    "hashlife step-0 dependency resolution stalled root_node={root_node} pending={} ready={} cache={} sample={sample:?}",
                    task_index.len(),
                    ready.len(),
                    self.result_caches.jump.len(),
                );
            };
            let Some(task_key) = task_keys[task_id].take() else {
                crate::invariant_failure!("step0 task missing key for task_id={task_id}");
            };
            task_index.remove(&task_key);
            let task = tasks[task_id].take().or_invariant("required value");
            debug_assert_eq!(task.remaining, 0);
            let [nw, ne, sw, se] = task.children;
            let q00 = self.jump_result((nw, 0));
            let q01 = self.jump_result((ne, 0));
            let q10 = self.jump_result((sw, 0));
            let q11 = self.jump_result((se, 0));
            let result = self.join(q00, q01, q10, q11);
            self.insert_canonical_jump_result(task_key, result);
            notify_step0_dependents(
                self,
                task_key,
                &mut tasks,
                &mut dependents,
                &dependent_edges,
                &mut ready,
            );
            if ready.len() > self.stats.scheduler.scheduler_ready_max {
                self.stats.scheduler.scheduler_ready_max = ready.len();
            }
        }

        self.jump_result(root_key)
    }
}
