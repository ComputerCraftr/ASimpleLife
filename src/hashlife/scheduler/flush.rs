use super::*;
use super::deps::notify_dependents;

impl HashLifeEngine {
    const PHASE1_JOIN_QUADS: [[usize; 4]; 4] = [
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 8],
    ];

    pub(in crate::hashlife::scheduler) fn flush_step0_simd_candidates(
        &mut self,
        provisional_candidate_keys: &mut Vec<DiscoveredJumpTask>,
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
        is_phase_two: bool,
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

        if is_phase_two {
            self.stats.phase2_simd_batches += 1;
            self.stats.phase2_simd_lanes += provisional_candidates.len();
        } else {
            self.stats.phase1_simd_batches += 1;
            self.stats.phase1_simd_lanes += provisional_candidates.len();
        }
        let batch_result = Self::evaluate_simd_batch(&Self::pack_simd_batch(provisional_candidates));

        if is_phase_two {
            self.commit_phase2_provisional_batch(
                provisional_candidates,
                &batch_result,
                tasks,
                dependents,
                dependent_edges,
                ready,
            )
        } else {
            self.commit_phase1_provisional_batch(
                provisional_candidates,
                &batch_result,
                discover,
                task_index,
                tasks,
                task_keys,
                dependents,
                dependent_edges,
                ready,
            )
        }
    }

    fn commit_phase1_provisional_batch(
        &mut self,
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
        batch_result: &SimdBatchResult,
        discover: &mut Vec<DiscoveredJumpTask>,
        task_index: &mut FlatTable<CanonicalJumpKey, usize>,
        tasks: &mut Vec<Option<TaskRecord>>,
        task_keys: &mut Vec<Option<CanonicalJumpKey>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        let active = provisional_candidates.len();
        if active == 0 {
            return;
        }
        let mut lanes = [Phase1CommitLane {
            provisional: SimdProvisionalRecord {
                cache_key: CanonicalJumpKey::empty(),
                level: 0,
                inputs: SimdProvisionalInputs::Nine {
                    nodes: [0; 9],
                    populations: [0; 9],
                },
                payload: SimdProvisionalPayload::PhaseTwo,
            },
            task_id: 0,
            next_exp: 0,
            next_children: [0; 4],
        }; SIMD_BATCH_LANES];
        let mut intents = [[None; SIMD_BATCH_LANES]; 4];
        for lane in 0..active {
            let provisional = provisional_candidates[lane];
            let lane_result = batch_result.lanes[lane];
            let SimdProvisionalPayload::PhaseOne {
                next_exp,
                source_task_id: task_id,
            } = provisional.payload
            else {
                unreachable!("phase1 provisional records must use recursive payload");
            };
            lanes[lane].provisional = provisional;
            lanes[lane].task_id = task_id;
            lanes[lane].next_exp = next_exp;
            let level = provisional.level;
            let centered = match provisional.inputs {
                SimdProvisionalInputs::Nine { nodes, .. } => nodes,
                SimdProvisionalInputs::Four { .. } => {
                    unreachable!("phase1 provisional records must carry 9-node inputs")
                }
            };
            let join_level = level - 1;
            for (join_index, quad) in Self::PHASE1_JOIN_QUADS.into_iter().enumerate() {
                intents[join_index][lane] =
                    ((lane_result.output_nonzero_mask & (1 << join_index)) != 0).then_some(
                        JoinIntent {
                            level: join_level,
                            children: [
                                centered[quad[0]],
                                centered[quad[1]],
                                centered[quad[2]],
                                centered[quad[3]],
                            ],
                        },
                    );
            }
        }
        let resolved_0 = self.resolve_join_intents_staged(intents[0]);
        let resolved_1 = self.resolve_join_intents_staged(intents[1]);
        let resolved_2 = self.resolve_join_intents_staged(intents[2]);
        let resolved_3 = self.resolve_join_intents_staged(intents[3]);
        for lane in 0..active {
            let provisional = lanes[lane].provisional;
            let empty_child = self.empty(provisional.level - 1);
            let mut shortcut_misses = 0usize;
            lanes[lane].next_children = [
                resolved_3[lane].unwrap_or_else(|| {
                    shortcut_misses += 1;
                    empty_child
                }),
                resolved_2[lane].unwrap_or_else(|| {
                    shortcut_misses += 1;
                    empty_child
                }),
                resolved_1[lane].unwrap_or_else(|| {
                    shortcut_misses += 1;
                    empty_child
                }),
                resolved_0[lane].unwrap_or_else(|| {
                    shortcut_misses += 1;
                    empty_child
                }),
            ];
            self.stats.join_shortcut_avoided += shortcut_misses;
        }
        for lane in 0..active {
            let provisional = lanes[lane].provisional;
            let task_id = lanes[lane].task_id;
            let next_exp = lanes[lane].next_exp;
            let next_children = lanes[lane].next_children;
            self.stats.scalar_commit_lanes += 1;
            task_index.insert(provisional.cache_key, task_id);
            task_keys[task_id] = Some(provisional.cache_key);
            tasks[task_id] = Some(TaskRecord {
                remaining: 0,
                task: PendingTask::PhaseTwo {
                    next_exp,
                    nw: next_children[3],
                    ne: next_children[2],
                    sw: next_children[1],
                    se: next_children[0],
                },
            });
            let next_child_keys = self.discovered_jump_tasks_from_nodes(next_children, next_exp);
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
        provisional_candidates.clear();
    }

    fn commit_phase2_provisional_batch(
        &mut self,
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
        batch_result: &SimdBatchResult,
        tasks: &mut [Option<TaskRecord>],
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        let active = provisional_candidates.len();
        if active == 0 {
            return;
        }
        let mut lanes = [Phase2CommitLane {
            key: CanonicalJumpKey::empty(),
            intent: None,
            fallback: 0,
            result: 0,
        }; SIMD_BATCH_LANES];
        for lane in 0..active {
            let provisional = provisional_candidates[lane];
            let lane_result = batch_result.lanes[lane];
            lanes[lane].key = provisional.cache_key;
            lanes[lane].fallback = self.empty(provisional.level - 1);
            let input_nodes = match provisional.inputs {
                SimdProvisionalInputs::Four { nodes, .. } => nodes,
                SimdProvisionalInputs::Nine { .. } => {
                    unreachable!("phase2 provisional records must carry 4-node inputs")
                }
            };
            if lane_result.output_nonzero_mask != 0 {
                lanes[lane].intent = Some(JoinIntent {
                    level: provisional.level - 1,
                    children: [
                        input_nodes[0],
                        input_nodes[1],
                        input_nodes[2],
                        input_nodes[3],
                    ],
                });
            }
        }
        let intents = lanes.map(|lane| lane.intent);
        let resolved = self.resolve_join_intents_staged(intents);
        let mut keys = [CanonicalJumpKey::empty(); SIMD_BATCH_LANES];
        let mut results = [0_u64; SIMD_BATCH_LANES];
        for (lane, _provisional) in provisional_candidates.drain(..).enumerate() {
            keys[lane] = lanes[lane].key;
            lanes[lane].result = if let Some(resolved) = resolved[lane] {
                resolved
            } else {
                self.stats.join_shortcut_avoided += 1;
                lanes[lane].fallback
            };
            results[lane] = lanes[lane].result;
            self.stats.scalar_commit_lanes += 1;
        }
        self.insert_canonical_jump_results_batch(&keys, &results, active);
        for lane in 0..active {
            notify_dependents(&keys[lane], tasks, dependents, dependent_edges, ready);
        }
        self.stats.scheduler_ready_max = self.stats.scheduler_ready_max.max(ready.len());
    }
}
