use super::*;
use super::deps::notify_dependents;

impl HashLifeEngine {
    pub(in crate::hashlife::scheduler) fn flush_step0_simd_candidates(
        &mut self,
        provisional_candidate_keys: &mut Vec<CanonicalJumpKey>,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut Vec<Option<Step0TaskRecord>>,
        task_keys: &mut Vec<Option<CanonicalJumpKey>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        if provisional_candidate_keys.is_empty() {
            return;
        }

        let mut provisional_candidates =
            Vec::<SimdProvisionalRecord>::with_capacity(provisional_candidate_keys.len());
        self.build_step0_provisional_records_staged(
            provisional_candidate_keys,
            &mut provisional_candidates,
        );
        provisional_candidate_keys.clear();

        self.stats.step0_simd_batches += 1;
        self.stats.step0_simd_lanes += provisional_candidates.len();
        let packed = Self::pack_simd_batch(&provisional_candidates);
        let batch_result = Self::evaluate_simd_batch(&packed);
        for (lane, provisional) in provisional_candidates.drain(..).enumerate() {
            self.schedule_step0_provisional_task(
                provisional,
                batch_result.lanes[lane],
                discover,
                task_index,
                tasks,
                task_keys,
                dependents,
                dependent_edges,
                ready,
            );
        }
    }

    fn schedule_step0_provisional_task(
        &mut self,
        provisional: SimdProvisionalRecord,
        lane_result: SimdLaneResult,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut Vec<Option<Step0TaskRecord>>,
        task_keys: &mut Vec<Option<CanonicalJumpKey>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        debug_assert_eq!(provisional.kind, SimdTaskKind::Step0);
        debug_assert!(matches!(
            provisional.payload,
            SimdProvisionalPayload::Step0 {
                dispatch: Step0LaneDispatch::SimdChild,
            }
        ));
        let children = self.build_step0_combined_children(&provisional, lane_result);
        let task_id = tasks.len();
        task_index.insert(provisional.cache_key, task_id);
        tasks.push(Some(Step0TaskRecord {
            remaining: 0,
            children,
        }));
        task_keys.push(Some(provisional.cache_key));
        self.stats.scheduler_tasks += 1;

        self.schedule_step0_children(
            [children[3], children[2], children[1], children[0]],
            task_id,
            discover,
            task_index,
            tasks,
            dependents,
            dependent_edges,
        );

        if tasks[task_id].as_ref().unwrap().remaining == 0 {
            ready.push(task_id);
            self.stats.step0_ready_max = self.stats.step0_ready_max.max(ready.len());
            if ready.len() > self.stats.scheduler_ready_max {
                self.stats.scheduler_ready_max = ready.len();
            }
        }
    }

    pub(in crate::hashlife::scheduler) fn flush_recursive_simd_candidates(
        &mut self,
        kind: SimdTaskKind,
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut Vec<Option<TaskRecord>>,
        task_keys: &mut Vec<Option<CanonicalJumpKey>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        if provisional_candidates.is_empty() {
            return;
        }

        match kind {
            SimdTaskKind::Step0 => {}
            SimdTaskKind::PhaseOne => {
                self.stats.phase1_simd_batches += 1;
                self.stats.phase1_simd_lanes += provisional_candidates.len();
            }
            SimdTaskKind::PhaseTwo => {
                self.stats.phase2_simd_batches += 1;
                self.stats.phase2_simd_lanes += provisional_candidates.len();
            }
        }
        let batch_result = Self::evaluate_simd_batch(&Self::pack_simd_batch(provisional_candidates));

        for (lane, provisional) in provisional_candidates.drain(..).enumerate() {
            let lane_result = batch_result.lanes[lane];
            match kind {
                SimdTaskKind::PhaseOne => self.commit_phase1_provisional_task(
                    provisional,
                    lane_result,
                    discover,
                    task_index,
                    tasks,
                    task_keys,
                    dependents,
                    dependent_edges,
                    ready,
                ),
                SimdTaskKind::PhaseTwo => self.commit_phase2_provisional_task(
                    provisional,
                    lane_result,
                    tasks,
                    dependents,
                    dependent_edges,
                    ready,
                ),
                SimdTaskKind::Step0 => unreachable!(),
            }
        }
    }

    fn commit_phase1_provisional_task(
        &mut self,
        provisional: SimdProvisionalRecord,
        lane_result: SimdLaneResult,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut Vec<Option<TaskRecord>>,
        task_keys: &mut Vec<Option<CanonicalJumpKey>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        let SimdProvisionalPayload::Recursive {
            next_exp,
            source_task_id: task_id,
        } = provisional.payload
        else {
            unreachable!("phase1 provisional records must use recursive payload");
        };
        debug_assert_eq!(provisional.kind, SimdTaskKind::PhaseOne);
        let level = provisional.level;
        let empty_child = self.empty(level - 1);
        let centered = provisional.input_nodes;
        let join_level = level - 1;
        let next_children = self.resolve_join_intents_staged([
            ((lane_result.output_nonzero_mask & 1) != 0).then_some(JoinIntent {
                level: join_level,
                children: [centered[0], centered[1], centered[3], centered[4]],
            }),
            ((lane_result.output_nonzero_mask & 2) != 0).then_some(JoinIntent {
                level: join_level,
                children: [centered[1], centered[2], centered[4], centered[5]],
            }),
            ((lane_result.output_nonzero_mask & 4) != 0).then_some(JoinIntent {
                level: join_level,
                children: [centered[3], centered[4], centered[6], centered[7]],
            }),
            ((lane_result.output_nonzero_mask & 8) != 0).then_some(JoinIntent {
                level: join_level,
                children: [centered[4], centered[5], centered[7], centered[8]],
            }),
        ]);
        let next_upper_left = next_children[0].unwrap_or_else(|| {
            self.stats.join_shortcut_avoided += 1;
            empty_child
        });
        let next_upper_right = next_children[1].unwrap_or_else(|| {
            self.stats.join_shortcut_avoided += 1;
            empty_child
        });
        let next_lower_left = next_children[2].unwrap_or_else(|| {
            self.stats.join_shortcut_avoided += 1;
            empty_child
        });
        let next_lower_right = next_children[3].unwrap_or_else(|| {
            self.stats.join_shortcut_avoided += 1;
            empty_child
        });
        self.stats.scalar_commit_lanes += 1;

        let parent_key = provisional.cache_key;
        task_index.insert(parent_key, task_id);
        task_keys[task_id] = Some(parent_key);
        tasks[task_id] = Some(TaskRecord {
            remaining: 0,
            task: PendingTask::PhaseTwo {
                next_exp,
                nw: next_upper_left,
                ne: next_upper_right,
                sw: next_lower_left,
                se: next_lower_right,
            },
        });
        let next_child_keys = self.discovered_jump_tasks_from_nodes(
            [
                next_lower_right,
                next_lower_left,
                next_upper_right,
                next_upper_left,
            ],
            next_exp,
        );
        self.schedule_recursive_children(
            next_child_keys,
            task_id,
            discover,
            task_index,
            tasks,
            dependents,
            dependent_edges,
        );
        if tasks[task_id].as_ref().unwrap().remaining == 0 {
            ready.push(task_id);
            self.stats.phase2_ready_max = self.stats.phase2_ready_max.max(ready.len());
            self.stats.scheduler_ready_max = self.stats.scheduler_ready_max.max(ready.len());
        }
    }

    fn commit_phase2_provisional_task(
        &mut self,
        provisional: SimdProvisionalRecord,
        lane_result: SimdLaneResult,
        tasks: &mut [Option<TaskRecord>],
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        let SimdProvisionalPayload::Recursive {
            next_exp: _next_exp,
            source_task_id: _task_id,
        } = provisional.payload
        else {
            unreachable!("phase2 provisional records must use recursive payload");
        };
        debug_assert_eq!(provisional.kind, SimdTaskKind::PhaseTwo);
        let key = provisional.cache_key;
        let level = provisional.level;
        let result = if lane_result.output_nonzero_mask == 0 {
            self.stats.join_shortcut_avoided += 1;
            self.empty(level - 1)
        } else {
            self.resolve_join_intents_staged([Some(JoinIntent {
                level: level - 1,
                children: [
                    provisional.input_nodes[0],
                    provisional.input_nodes[1],
                    provisional.input_nodes[2],
                    provisional.input_nodes[3],
                ],
            })])[0]
                .expect("phase2 join should resolve")
        };
        self.stats.scalar_commit_lanes += 1;
        self.insert_canonical_jump_result(key, result);
        notify_dependents(&key, tasks, dependents, dependent_edges, ready);
        self.stats.scheduler_ready_max = self.stats.scheduler_ready_max.max(ready.len());
    }
}
