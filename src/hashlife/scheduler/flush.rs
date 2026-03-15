use super::deps::notify_dependents;
use super::*;
use crate::RequiredExt;

impl HashLifeEngine {
    const PHASE1_JOIN_QUADS: [[usize; 4]; 4] =
        [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]];

    pub(in crate::hashlife::scheduler) fn flush_step0_kernel_candidates(
        &mut self,
        provisional_candidate_keys: &mut Vec<DiscoveredJumpTask>,
        state: &mut Step0SchedulerState<'_>,
    ) {
        if provisional_candidate_keys.is_empty() {
            return;
        }

        let Some(mut provisional_candidates) =
            self.try_transient_vec::<SimdProvisionalRecord>(provisional_candidate_keys.len())
        else {
            return;
        };
        self.build_step0_provisional_records_staged(
            provisional_candidate_keys,
            &mut provisional_candidates,
        );
        provisional_candidate_keys.clear();

        self.stats.simd.step0_kernel_candidate_batches += 1;
        self.stats.simd.step0_kernel_candidate_lanes += provisional_candidates.len();
        let packed = Self::pack_simd_batch(&provisional_candidates);
        let batch_result = self.evaluate_simd_batch(&packed);
        for (lane, provisional) in provisional_candidates.drain(..).enumerate() {
            self.schedule_step0_provisional_task(provisional, batch_result.lanes[lane], state);
        }
    }

    fn schedule_step0_provisional_task(
        &mut self,
        provisional: SimdProvisionalRecord,
        lane_result: SimdLaneResult,
        state: &mut Step0SchedulerState<'_>,
    ) {
        debug_assert!(matches!(
            provisional.payload,
            SimdProvisionalPayload::Step0 {
                dispatch: Step0LaneDispatch::SimdChild,
            }
        ));
        let children = self.build_step0_combined_children(&provisional, lane_result);
        let task_id = state.tasks.len();
        if !self.try_insert_transient_table(state.task_index, provisional.cache_key, task_id)
            || !self.try_push_transient(
                state.tasks,
                Some(Step0TaskRecord {
                    remaining: 0,
                    children,
                }),
            )
            || !self.try_push_transient(state.task_keys, Some(provisional.cache_key))
        {
            return;
        }
        self.stats.scheduler.scheduler_tasks += 1;

        self.schedule_step0_children(
            [children[3], children[2], children[1], children[0]],
            task_id,
            state,
        );

        if state.tasks[task_id]
            .as_ref()
            .or_invariant("required value")
            .remaining
            == 0
        {
            if !self.try_push_transient(state.ready, task_id) {
                return;
            }
            self.stats.scheduler.step0_ready_max =
                self.stats.scheduler.step0_ready_max.max(state.ready.len());
            if state.ready.len() > self.stats.scheduler.scheduler_ready_max {
                self.stats.scheduler.scheduler_ready_max = state.ready.len();
            }
        }
    }

    pub(in crate::hashlife::scheduler) fn flush_recursive_kernel_candidates(
        &mut self,
        is_phase_two: bool,
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
        state: &mut RecursiveSchedulerState<'_>,
    ) {
        if provisional_candidates.is_empty() {
            return;
        }

        if is_phase_two {
            self.stats.simd.phase2_kernel_candidate_batches += 1;
            self.stats.simd.phase2_kernel_candidate_lanes += provisional_candidates.len();
        } else {
            self.stats.simd.phase1_kernel_candidate_batches += 1;
            self.stats.simd.phase1_kernel_candidate_lanes += provisional_candidates.len();
        }
        let batch_result = self.evaluate_simd_batch(&Self::pack_simd_batch(provisional_candidates));

        if is_phase_two {
            self.commit_phase2_provisional_batch(provisional_candidates, &batch_result, state)
        } else {
            self.commit_phase1_provisional_batch(provisional_candidates, &batch_result, state)
        }
    }

    fn commit_phase1_provisional_batch(
        &mut self,
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
        batch_result: &SimdBatchResult,
        state: &mut RecursiveSchedulerState<'_>,
    ) {
        let active = provisional_candidates.len();
        if active == 0 {
            return;
        }
        let mut lanes = [Phase1CommitLane {
            provisional: provisional_candidates[0],
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
                crate::invariant_failure!("phase1 provisional records must use recursive payload");
            };
            lanes[lane] = Phase1CommitLane {
                provisional,
                task_id,
                next_exp,
                next_children: [0; 4],
            };
            let level = provisional.level;
            let centered = match provisional.inputs {
                SimdProvisionalInputs::Nine { nodes, .. } => nodes,
                SimdProvisionalInputs::Four { .. } => {
                    crate::invariant_failure!("phase1 provisional records must carry 9-node inputs")
                }
            };
            let join_level = level - 1;
            for (join_index, quad) in Self::PHASE1_JOIN_QUADS.into_iter().enumerate() {
                intents[join_index][lane] = ((lane_result.output_nonzero_mask & (1 << join_index))
                    != 0)
                    .then_some(JoinIntent {
                        level: join_level,
                        children: [
                            centered[quad[0]],
                            centered[quad[1]],
                            centered[quad[2]],
                            centered[quad[3]],
                        ],
                    });
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
            self.stats.scheduler.join_shortcut_avoided += shortcut_misses;
        }
        for lane_state in &lanes[..active] {
            let provisional = lane_state.provisional;
            let task_id = lane_state.task_id;
            let next_exp = lane_state.next_exp;
            let next_children = lane_state.next_children;
            self.stats.simd.scalar_commit_lanes += 1;
            if !self.try_insert_transient_table(state.task_index, provisional.cache_key, task_id) {
                return;
            }
            state.task_keys[task_id] = Some(provisional.cache_key);
            state.tasks[task_id] = Some(TaskRecord {
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
            self.schedule_recursive_children(next_child_keys, task_id, state);
            if state.tasks[task_id]
                .as_ref()
                .or_invariant("required value")
                .remaining
                == 0
            {
                if !self.try_push_transient(state.ready, task_id) {
                    return;
                }
                self.stats.scheduler.phase2_ready_max =
                    self.stats.scheduler.phase2_ready_max.max(state.ready.len());
                self.stats.scheduler.scheduler_ready_max = self
                    .stats
                    .scheduler
                    .scheduler_ready_max
                    .max(state.ready.len());
            }
        }
        provisional_candidates.clear();
    }

    fn commit_phase2_provisional_batch(
        &mut self,
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
        batch_result: &SimdBatchResult,
        state: &mut RecursiveSchedulerState<'_>,
    ) {
        let active = provisional_candidates.len();
        if active == 0 {
            return;
        }
        let seed = provisional_candidates[0];
        let mut lanes = [Phase2CommitLane {
            key: seed.cache_key,
            fallback: self.dead_leaf,
            result: 0,
            unique_input_index: usize::MAX,
            packed_input: PackedSymmetryKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                symmetry: Symmetry::Identity,
            },
            canonical_entry: PackedSymmetryKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                symmetry: Symmetry::Identity,
            },
        }; SIMD_BATCH_LANES];
        let mut intents = [None; SIMD_BATCH_LANES];
        for (lane, provisional) in provisional_candidates.iter().enumerate() {
            let fallback = self.empty(provisional.level - 1);
            let input_nodes = match provisional.inputs {
                SimdProvisionalInputs::Four { nodes, .. } => nodes,
                SimdProvisionalInputs::Nine { .. } => {
                    crate::invariant_failure!("phase2 provisional records must carry 4-node inputs")
                }
            };
            let intent = if batch_result.lanes[lane].output_nonzero_mask != 0 {
                Some(JoinIntent {
                    level: provisional.level - 1,
                    children: [
                        input_nodes[0],
                        input_nodes[1],
                        input_nodes[2],
                        input_nodes[3],
                    ],
                })
            } else {
                None
            };
            intents[lane] = intent;
            lanes[lane] = Phase2CommitLane {
                key: provisional.cache_key,
                fallback,
                result: 0,
                unique_input_index: usize::MAX,
                packed_input: PackedSymmetryKey {
                    packed: PackedNodeKey::new(0, [0; 4]),
                    symmetry: Symmetry::Identity,
                },
                canonical_entry: PackedSymmetryKey {
                    packed: PackedNodeKey::new(0, [0; 4]),
                    symmetry: Symmetry::Identity,
                },
            };
        }
        let resolved = self.resolve_join_intents_staged(intents);
        for (lane, (lane_state, provisional)) in lanes
            .iter_mut()
            .zip(provisional_candidates.iter())
            .enumerate()
        {
            match provisional.payload {
                SimdProvisionalPayload::PhaseTwo => {}
                SimdProvisionalPayload::PhaseOne { .. } | SimdProvisionalPayload::Step0 { .. } => {
                    crate::invariant_failure!("phase2 flush must only receive phase2 provisionals")
                }
            };
            lane_state.result = if let Some(resolved) = resolved[lane] {
                resolved
            } else {
                self.stats.scheduler.join_shortcut_avoided += 1;
                lane_state.fallback
            };
            lane_state.packed_input = PackedSymmetryKey {
                packed: self.node_columns.packed_key(lane_state.result),
                symmetry: Symmetry::Identity,
            };
            self.stats.simd.scalar_commit_lanes += 1;
        }
        provisional_candidates.clear();
        self.canonicalize_phase2_commit_lanes(&mut lanes);
        for lane in &lanes[..active] {
            let fingerprint = lane.key.fingerprint();
            self.record_and_publish_jump_entry(lane.key, fingerprint, lane.canonical_entry);
            notify_dependents(
                self,
                &lane.key,
                state.tasks,
                state.dependents,
                state.dependent_edges,
                state.ready,
            );
        }
        self.stats.scheduler.scheduler_ready_max = self
            .stats
            .scheduler
            .scheduler_ready_max
            .max(state.ready.len());
    }
}
