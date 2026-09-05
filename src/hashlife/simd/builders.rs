use super::*;

impl HashLifeEngine {
    pub(super) fn write_population_lane<const N: usize>(
        populations: &mut AlignedU64WordBatch9,
        lane: usize,
        input_populations: [u64; N],
    ) {
        for (index, population) in input_populations.into_iter().enumerate() {
            populations.0[index][lane] = population;
        }
    }

    pub(in crate::hashlife) fn build_step0_provisional_records_staged(
        &mut self,
        discovered_tasks: &[DiscoveredJumpTask],
    ) -> Option<[SimdProvisionalRecord; SIMD_BATCH_LANES]> {
        if discovered_tasks.is_empty() {
            return None;
        }
        debug_assert!(discovered_tasks.len() <= SIMD_BATCH_LANES);
        let mut identities = [CanonicalNodeIdentity {
            packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
            structural: CanonicalStructKey::leaf(false),
            symmetry: Symmetry::Identity,
        }; SIMD_BATCH_LANES];
        let mut fingerprints = [0_u64; SIMD_BATCH_LANES];
        for (lane, discovered_task) in discovered_tasks.iter().enumerate() {
            identities[lane] = CanonicalNodeIdentity {
                packed: discovered_task.canonical_packed,
                structural: discovered_task.key.structural,
                symmetry: Symmetry::Identity,
            };
            fingerprints[lane] = discovered_task.key.structural.fingerprint();
        }
        self.stats.scheduler.cache_probe_batches += 1;
        self.stats.scheduler.scheduler_probe_batches += 1;
        self.stats.simd.overlap_prep_batches += 1;
        self.stats.transform.packed_overlap_outputs_produced += discovered_tasks.len();
        let overlaps = self.probe_and_build_canonical_overlaps_staged(
            &identities,
            &fingerprints,
            discovered_tasks.len(),
        )?;
        let (centered_lanes, population_lanes) =
            self.build_centered_population_lanes_9xn(&overlaps, discovered_tasks.len());
        if self.allocation_failed() {
            return None;
        }
        let mut provisional_candidates = [SimdProvisionalRecord {
            cache_key: CanonicalJumpKey::empty(),
            level: 0,
            inputs: SimdProvisionalInputs::Nine {
                nodes: [NodeId::ZERO; 9],
                populations: [0; 9],
            },
            payload: SimdProvisionalPayload::Step0 {
                dispatch: Step0LaneDispatch::SimdChild,
            },
        }; SIMD_BATCH_LANES];
        for (lane, discovered_task) in discovered_tasks.iter().enumerate() {
            let cache_key = discovered_task.key;
            let input_nodes = centered_lanes[lane];
            let input_populations = population_lanes[lane];
            self.stats.simd.step0_provisional_records += 1;
            provisional_candidates[lane] = SimdProvisionalRecord {
                cache_key,
                level: cache_key.structural.level,
                inputs: SimdProvisionalInputs::Nine {
                    nodes: input_nodes,
                    populations: input_populations,
                },
                payload: SimdProvisionalPayload::Step0 {
                    dispatch: Step0LaneDispatch::SimdChild,
                },
            };
        }
        Some(provisional_candidates)
    }

    pub(in crate::hashlife) fn build_centered_population_lanes_9xn(
        &mut self,
        overlap_lanes: &[[NodeId; 9]; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> (
        [[NodeId; 9]; SIMD_BATCH_LANES],
        [[u64; 9]; SIMD_BATCH_LANES],
    ) {
        debug_assert!(active_lanes <= SIMD_BATCH_LANES);
        let mut centered_lanes = [[NodeId::ZERO; 9]; SIMD_BATCH_LANES];
        let mut population_lanes = AlignedU64LaneWords9::default();
        for index in 0..9 {
            let mut overlap_word = [NodeId::ZERO; SIMD_BATCH_LANES];
            for lane in 0..active_lanes {
                overlap_word[lane] = overlap_lanes[lane][index];
            }
            let centered = self.centered_subnode_batch(overlap_word, active_lanes);
            for (lane, &node) in centered.iter().take(active_lanes).enumerate() {
                centered_lanes[lane][index] = node;
                population_lanes.0[lane][index] =
                    u64::from(self.node_columns.population(node) != 0);
            }
        }
        (centered_lanes, population_lanes.0)
    }

    pub(in crate::hashlife) fn build_step0_combined_children(
        &mut self,
        provisional: &SimdProvisionalRecord,
        lane_result: SimdLaneResult,
    ) -> [NodeId; 4] {
        let level = provisional.level;
        debug_assert!(level >= 3);
        let empty_child = self.empty(level - 1);
        let centered = match provisional.inputs {
            SimdProvisionalInputs::Nine { nodes, .. } => nodes,
            SimdProvisionalInputs::Four { .. } => {
                crate::invariant_failure!("step0 provisional records must carry 9-node inputs")
            }
        };
        let join_level = level - 1;
        let intents = [
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
        ];
        let resolved = self.resolve_join_intents_staged(intents);
        [
            resolved[0].unwrap_or(empty_child),
            resolved[1].unwrap_or(empty_child),
            resolved[2].unwrap_or(empty_child),
            resolved[3].unwrap_or(empty_child),
        ]
    }

    fn jump_result_query_batch<const N: usize>(
        &mut self,
        queries: [JumpQuery; N],
        active_lanes: usize,
    ) -> [NodeId; N] {
        self.with_transient_allocation_scope(|engine| {
            engine.resolve_jump_result_query_batch(queries, active_lanes)
        })
    }

    fn resolve_jump_result_query_batch<const N: usize>(
        &mut self,
        queries: [JumpQuery; N],
        active_lanes: usize,
    ) -> [NodeId; N] {
        debug_assert!(active_lanes <= N);
        let mut results = [NodeId::ZERO; N];
        if active_lanes == 0 {
            return results;
        }
        let mut unique_queries = [UniqueJumpQueryRecord {
            query: JumpQuery {
                node: NodeId::ZERO,
                step_exp: 0,
            },
            cache_key: CanonicalJumpKey::empty(),
            inverse_symmetry: Symmetry::Identity,
            fingerprint: 0,
        }; N];
        let mut unique_count = 0;
        let mut lane_to_unique = [usize::MAX; N];
        let Some(mut unique_lookup) = self.try_transient_probe_table::<JumpQuery, usize>(N.max(4))
        else {
            return results;
        };
        for lane in 0..active_lanes {
            let query = queries[lane];
            if let Some(index) = unique_lookup.get(&query) {
                self.stats.canonical_fallback.jump_batch_reused_queries += 1;
                lane_to_unique[lane] = index;
            } else {
                let jump_probe = self.canonical_jump_probe((query.node, query.step_exp));
                self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
                lane_to_unique[lane] = unique_count;
                unique_queries[unique_count] = UniqueJumpQueryRecord {
                    query,
                    cache_key: jump_probe.key,
                    inverse_symmetry: jump_probe.node.symmetry.inverse(),
                    fingerprint: jump_probe.fingerprint,
                };
                if !self.try_insert_transient_table(&mut unique_lookup, query, unique_count) {
                    return results;
                }
                unique_count += 1;
                self.stats.canonical_fallback.jump_batch_unique_queries += 1;
            }
        }

        self.stats.scheduler.cache_probe_batches += 1;
        let mut unique_cache_keys = [CanonicalJumpKey::empty(); N];
        let mut unique_fingerprints = [0_u64; N];
        for (index, record) in unique_queries[..unique_count].iter().enumerate() {
            unique_cache_keys[index] = record.cache_key;
            unique_fingerprints[index] = record.fingerprint;
        }
        let (mut cached, probe_accounting) = Self::probe_table_many(
            &self.result_caches.jump,
            &unique_cache_keys,
            &unique_fingerprints,
            unique_count,
        );
        self.record_kernel_accounting(probe_accounting);
        for index in 0..unique_count {
            if cached[index].is_none() {
                cached[index] = self
                    .active_jump_results
                    .get_with_fingerprint(&unique_cache_keys[index], unique_fingerprints[index]);
            }
        }
        self.stats.result_cache.jump_result_cache_lookups += unique_count;
        let mut unique_to_oriented = [usize::MAX; N];
        let mut oriented_results = [UniqueOrientedResultRecord {
            packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
            symmetry: Symmetry::Identity,
            node: NodeId::ZERO,
        }; N];
        let mut oriented_count = 0;
        let Some(mut oriented_lookup) =
            self.try_transient_probe_table::<PackedSymmetryKey, usize>(unique_count.max(4))
        else {
            return results;
        };
        for index in 0..unique_count {
            let Some(cached_entry) = cached[index] else {
                self.stats.result_cache.jump_result_cache_misses += 1;
                crate::invariant_failure!(
                    "missing HashLife jump result for grouped batch node={:?} step_exp={}",
                    unique_queries[index].query.node,
                    unique_queries[index].query.step_exp,
                );
            };
            self.stats.result_cache.jump_result_cache_hits += 1;
            let output_symmetry = unique_queries[index].inverse_symmetry;
            let combined = cached_entry.symmetry.inverse().then(output_symmetry);
            if combined != Symmetry::Identity {
                self.stats.result_cache.symmetric_jump_result_cache_hits += 1;
            }
            let oriented_key = PackedSymmetryKey {
                packed: cached_entry.packed,
                symmetry: combined,
            };
            unique_to_oriented[index] = if let Some(oriented) = oriented_lookup.get(&oriented_key) {
                oriented
            } else {
                let oriented = oriented_count;
                oriented_results[oriented] = UniqueOrientedResultRecord {
                    packed: cached_entry.packed,
                    symmetry: combined,
                    node: NodeId::ZERO,
                };
                if !self.try_insert_transient_table(&mut oriented_lookup, oriented_key, oriented) {
                    return results;
                }
                oriented_count += 1;
                oriented
            };
        }
        for oriented in &mut oriented_results[..oriented_count] {
            oriented.node = self.materialize_oriented_packed_result(
                oriented.packed,
                Symmetry::Identity,
                oriented.symmetry,
            );
        }
        for lane in 0..active_lanes {
            results[lane] = oriented_results[unique_to_oriented[lane_to_unique[lane]]].node;
        }
        results
    }

    pub(in crate::hashlife) fn build_phase1_provisional_records_batch(
        &mut self,
        ready_lanes: &[Phase1ReadyLane; SIMD_BATCH_LANES],
        active_lanes: usize,
        out: &mut Vec<SimdProvisionalRecord>,
    ) {
        if active_lanes == 0 {
            return;
        }
        let mut queries = [JumpQuery {
            node: NodeId::ZERO,
            step_exp: 0,
        }; SIMD_BATCH_LANES * 9];
        for (lane, ready) in ready_lanes[..active_lanes].iter().enumerate() {
            let lane_base = lane * 9;
            let inputs = ready.inputs;
            let next_exp = ready.next_exp;
            queries[lane_base] = JumpQuery {
                node: inputs[0],
                step_exp: next_exp,
            };
            queries[lane_base + 1] = JumpQuery {
                node: inputs[1],
                step_exp: next_exp,
            };
            queries[lane_base + 2] = JumpQuery {
                node: inputs[2],
                step_exp: next_exp,
            };
            queries[lane_base + 3] = JumpQuery {
                node: inputs[3],
                step_exp: next_exp,
            };
            queries[lane_base + 4] = JumpQuery {
                node: inputs[4],
                step_exp: next_exp,
            };
            queries[lane_base + 5] = JumpQuery {
                node: inputs[5],
                step_exp: next_exp,
            };
            queries[lane_base + 6] = JumpQuery {
                node: inputs[6],
                step_exp: next_exp,
            };
            queries[lane_base + 7] = JumpQuery {
                node: inputs[7],
                step_exp: next_exp,
            };
            queries[lane_base + 8] = JumpQuery {
                node: inputs[8],
                step_exp: next_exp,
            };
        }
        let query_results = self.jump_result_query_batch(queries, active_lanes * 9);
        for (lane, ready) in ready_lanes[..active_lanes].iter().enumerate() {
            let base = lane * 9;
            let input_nodes = [
                query_results[base],
                query_results[base + 1],
                query_results[base + 2],
                query_results[base + 3],
                query_results[base + 4],
                query_results[base + 5],
                query_results[base + 6],
                query_results[base + 7],
                query_results[base + 8],
            ];
            let input_populations =
                input_nodes.map(|node| u64::from(self.node_columns.population(node) != 0));
            self.stats.simd.phase1_provisional_records += 1;
            out.push(SimdProvisionalRecord {
                cache_key: ready.key,
                level: ready.key.structural.level,
                inputs: SimdProvisionalInputs::Nine {
                    nodes: input_nodes,
                    populations: input_populations,
                },
                payload: SimdProvisionalPayload::PhaseOne {
                    next_exp: ready.next_exp,
                    source_task_id: ready.task_id,
                },
            });
        }
    }

    pub(in crate::hashlife) fn build_phase2_provisional_records_batch(
        &mut self,
        ready_lanes: &[Phase2ReadyLane; SIMD_BATCH_LANES],
        active_lanes: usize,
        out: &mut Vec<SimdProvisionalRecord>,
    ) {
        if active_lanes == 0 {
            return;
        }
        let mut queries = [JumpQuery {
            node: NodeId::ZERO,
            step_exp: 0,
        }; SIMD_BATCH_LANES * 4];
        for (lane, ready) in ready_lanes[..active_lanes].iter().enumerate() {
            let lane_base = lane * 4;
            let inputs = ready.inputs;
            let next_exp = ready.next_exp;
            queries[lane_base] = JumpQuery {
                node: inputs[0],
                step_exp: next_exp,
            };
            queries[lane_base + 1] = JumpQuery {
                node: inputs[1],
                step_exp: next_exp,
            };
            queries[lane_base + 2] = JumpQuery {
                node: inputs[2],
                step_exp: next_exp,
            };
            queries[lane_base + 3] = JumpQuery {
                node: inputs[3],
                step_exp: next_exp,
            };
        }
        let query_results = self.jump_result_query_batch(queries, active_lanes * 4);
        for (lane, ready) in ready_lanes[..active_lanes].iter().enumerate() {
            let base = lane * 4;
            let input_nodes = [
                query_results[base],
                query_results[base + 1],
                query_results[base + 2],
                query_results[base + 3],
            ];
            let input_populations =
                input_nodes.map(|node| u64::from(self.node_columns.population(node) != 0));
            self.stats.simd.phase2_provisional_records += 1;
            out.push(SimdProvisionalRecord {
                cache_key: ready.key,
                level: ready.key.structural.level,
                inputs: SimdProvisionalInputs::Four {
                    nodes: input_nodes,
                    populations: input_populations,
                },
                payload: SimdProvisionalPayload::PhaseTwo,
            });
        }
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(in crate::hashlife) fn jump_result_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        step_exp: u32,
    ) -> [NodeId; N] {
        self.jump_result_query_batch(nodes.map(|node| JumpQuery { node, step_exp }), N)
    }
}
