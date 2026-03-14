use super::*;
#[cfg(test)]
use crate::simd_layout::AlignedLaneIndexBatch;
use crate::simd_layout::AlignedU64LaneWords9;

impl HashLifeEngine {
    pub(super) fn probe_and_build_overlaps_from_canonical_keys_staged<const N: usize>(
        &mut self,
        cache_keys: &[CanonicalJumpKey; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut canonical_keys = [PackedNodeKey::new(0, [0; 4]); N];
        let mut canonical_fingerprints = [0_u64; N];
        for lane in 0..active_lanes {
            canonical_keys[lane] = cache_keys[lane].packed;
            canonical_fingerprints[lane] = cache_keys[lane].packed.fingerprint();
        }
        self.stats.cache_probe_batches += 1;
        self.stats.scheduler_probe_batches += 1;
        self.stats.overlap_prep_batches += 1;
        self.stats.packed_overlap_outputs_produced += active_lanes;
        self.probe_and_build_canonical_overlaps_staged(
            &canonical_keys,
            &canonical_fingerprints,
            active_lanes,
        )
    }

    pub(super) fn probe_and_build_canonical_overlaps_staged<const N: usize>(
        &mut self,
        canonical_keys: &[PackedNodeKey; N],
        canonical_fingerprints: &[u64; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut canonical_overlap_lanes = [[0; 9]; N];
        let cached = self.overlap_cache.get_many_with_fingerprints(
            canonical_keys,
            canonical_fingerprints,
            active_lanes,
        );
        let mut miss_levels = [0_u32; N];
        let mut miss_children = [[[0_u64; 4]; N]; 5];
        let mut miss_unique_keys = [PackedNodeKey::new(0, [0; 4]); N];
        let mut miss_unique_fingerprints = [0_u64; N];
        let mut miss_unique_active = [false; N];
        let mut miss_lane_to_unique = [usize::MAX; N];
        let mut is_miss = [false; N];
        let mut miss_unique_count = 0;

        for lane in 0..active_lanes {
            if let Some(lane_overlaps) = cached[lane] {
                self.stats.overlap_cache_hits += 1;
                canonical_overlap_lanes[lane] = lane_overlaps;
                continue;
            }

            let canonical_key = canonical_keys[lane];
            let mut existing_unique = None;
            for unique in 0..miss_unique_count {
                if miss_unique_keys[unique] == canonical_key {
                    existing_unique = Some(unique);
                    break;
                }
            }
            if let Some(unique) = existing_unique {
                self.stats.overlap_local_reuse_lanes += 1;
                miss_lane_to_unique[lane] = unique;
                is_miss[lane] = true;
                continue;
            }

            self.stats.overlap_cache_misses += 1;
            let [nw, ne, sw, se] = canonical_key.children;
            let [_, nw_ne, nw_sw, nw_se] = self.node_columns.quadrants(nw);
            let [ne_nw, _, ne_sw, ne_se] = self.node_columns.quadrants(ne);
            let [sw_nw, sw_ne, _, sw_se] = self.node_columns.quadrants(sw);
            let [se_nw, se_ne, se_sw, _] = self.node_columns.quadrants(se);
            miss_levels[miss_unique_count] = canonical_key.level - 1;
            miss_children[0][miss_unique_count] = [nw_ne, ne_nw, nw_se, ne_sw];
            miss_children[1][miss_unique_count] = [nw_sw, nw_se, sw_nw, sw_ne];
            miss_children[2][miss_unique_count] = [nw_se, ne_sw, sw_ne, se_nw];
            miss_children[3][miss_unique_count] = [ne_sw, ne_se, se_nw, se_ne];
            miss_children[4][miss_unique_count] = [sw_ne, se_nw, sw_se, se_sw];
            miss_unique_keys[miss_unique_count] = canonical_key;
            miss_unique_fingerprints[miss_unique_count] = canonical_fingerprints[lane];
            miss_unique_active[miss_unique_count] = true;
            miss_lane_to_unique[lane] = miss_unique_count;
            is_miss[lane] = true;
            miss_unique_count += 1;
        }

        if miss_unique_count != 0 {
            let miss_join_intents = Self::join_intents_from_child_words_5x4xn(
                miss_unique_count,
                &miss_unique_active,
                &miss_levels,
                &miss_children,
            );
            let resolved_join_0 = self.resolve_join_intents_staged(miss_join_intents[0]);
            let resolved_join_1 = self.resolve_join_intents_staged(miss_join_intents[1]);
            let resolved_join_2 = self.resolve_join_intents_staged(miss_join_intents[2]);
            let resolved_join_3 = self.resolve_join_intents_staged(miss_join_intents[3]);
            let resolved_join_4 = self.resolve_join_intents_staged(miss_join_intents[4]);
            let mut miss_overlaps = [[0_u64; 9]; N];

            for unique in 0..miss_unique_count {
                let canonical_key = miss_unique_keys[unique];
                let [nw, ne, sw, se] = canonical_key.children;
                let lane_overlaps = [
                    nw,
                    resolved_join_0[unique].expect("overlap join should resolve"),
                    ne,
                    resolved_join_1[unique].expect("overlap join should resolve"),
                    resolved_join_2[unique].expect("overlap join should resolve"),
                    resolved_join_3[unique].expect("overlap join should resolve"),
                    sw,
                    resolved_join_4[unique].expect("overlap join should resolve"),
                    se,
                ];
                self.overlap_cache.insert_with_fingerprint(
                    canonical_key,
                    miss_unique_fingerprints[unique],
                    lane_overlaps,
                );
                miss_overlaps[unique] = lane_overlaps;
            }

            for lane in 0..active_lanes {
                if is_miss[lane] {
                    canonical_overlap_lanes[lane] = miss_overlaps[miss_lane_to_unique[lane]];
                }
            }
        }

        for _ in 0..active_lanes {
            self.stats.overlap_prep_lanes += 1;
        }
        canonical_overlap_lanes
    }

    #[inline]
    fn nonzero_lane_mask(vector: u64x8) -> u8 {
        let zero_compare: [u64; SIMD_BATCH_LANES] = must_cast(vector.simd_eq(u64x8::ZERO));
        let mut mask = 0_u8;
        for lane in 0..SIMD_BATCH_LANES {
            mask |= u8::from(zero_compare[lane] == 0) << lane;
        }
        mask
    }

    #[cfg(test)]
    pub(super) fn overlapping_subnodes(&mut self, node: NodeId) -> [NodeId; 9] {
        let (packed, symmetry, fingerprint, used_cached_fingerprint) =
            if self.record_symmetry_gate_decision(node) {
                let canonical = self.canonicalize_packed_node(node);
                (
                    canonical.packed,
                    canonical.symmetry,
                    canonical.fingerprint,
                    canonical.used_cached_fingerprint,
                )
            } else {
                let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
                (packed, Symmetry::Identity, fingerprint, true)
            };
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        if let Some(overlaps) = self
            .overlap_cache
            .get_with_fingerprint(&packed, fingerprint)
        {
            self.stats.overlap_cache_hits += 1;
            self.stats.overlap_prep_lanes += 1;
            return symmetry.inverse().transform_overlap_nodes(self, overlaps);
        }
        self.stats.overlap_cache_misses += 1;
        self.stats.overlap_prep_lanes += 1;
        let [nw, ne, sw, se] = packed.children;
        let [_, nw_ne, nw_sw, nw_se] = self.node_columns.quadrants(nw);
        let [ne_nw, _, ne_sw, ne_se] = self.node_columns.quadrants(ne);
        let [sw_nw, sw_ne, _, sw_se] = self.node_columns.quadrants(sw);
        let [se_nw, se_ne, se_sw, _] = self.node_columns.quadrants(se);

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
        self.overlap_cache
            .insert_with_fingerprint(packed, fingerprint, overlaps);
        symmetry.inverse().transform_overlap_nodes(self, overlaps)
    }

    #[cfg(test)]
    pub(super) fn probe_and_build_overlaps_staged<const N: usize>(
        &mut self,
        nodes: &[NodeId; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut inverse_symmetries = [Symmetry::Identity; N];
        let mut canonical_keys = [PackedNodeKey::new(0, [0; 4]); N];
        let mut canonical_fingerprints = [0_u64; N];
        let canonicalized = self.canonicalize_packed_nodes_batch(nodes, active_lanes);
        for lane in 0..active_lanes {
            let (packed, symmetry, fingerprint, used_cached_fingerprint) =
                if self.record_symmetry_gate_decision(nodes[lane]) {
                    let canonical = canonicalized[lane];
                    (
                        canonical.packed,
                        canonical.symmetry,
                        canonical.fingerprint,
                        canonical.used_cached_fingerprint,
                    )
                } else {
                    let (packed, fingerprint) =
                        self.node_columns.packed_key_and_fingerprint(nodes[lane]);
                    (packed, Symmetry::Identity, fingerprint, true)
                };
            self.record_fingerprint_probe(used_cached_fingerprint, 1);
            inverse_symmetries[lane] = symmetry.inverse();
            canonical_keys[lane] = packed;
            canonical_fingerprints[lane] = fingerprint;
        }

        self.stats.cache_probe_batches += 1;
        self.stats.scheduler_probe_batches += 1;
        self.stats.overlap_prep_batches += 1;
        let canonical_overlap_lanes = self.probe_and_build_canonical_overlaps_staged(
            &canonical_keys,
            &canonical_fingerprints,
            active_lanes,
        );
        self.transform_overlap_words_grouped(
            &canonical_overlap_lanes,
            &inverse_symmetries,
            active_lanes,
        )
    }

    pub(super) fn centered_subnode(&mut self, node: NodeId) -> NodeId {
        let level = self.node_columns.level(node);
        debug_assert!(level >= 1);
        if level == 1 {
            return node;
        }

        let [nw, ne, sw, se] = self.node_columns.quadrants(node);
        let nw_se = self.node_columns.quadrants(nw)[3];
        let ne_sw = self.node_columns.quadrants(ne)[2];
        let sw_ne = self.node_columns.quadrants(sw)[1];
        let se_nw = self.node_columns.quadrants(se)[0];
        self.join(nw_se, ne_sw, sw_ne, se_nw)
    }

    pub(super) fn centered_subnode_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        active_lanes: usize,
    ) -> [NodeId; N] {
        let mut centered = [0; N];
        for lane in 0..active_lanes {
            let node = nodes[lane];
            if self.node_columns.level(node) == 1 {
                centered[lane] = node;
                continue;
            }
            let mut reused = None;
            for prev in 0..lane {
                if nodes[prev] == node {
                    reused = Some(centered[prev]);
                    break;
                }
            }
            centered[lane] = reused.unwrap_or_else(|| self.centered_subnode(node));
        }
        centered
    }

    #[cfg(test)]
    pub(super) fn transform_overlap_words_grouped<const N: usize>(
        &mut self,
        canonical_overlap_lanes: &[[NodeId; 9]; N],
        inverse_symmetries: &[Symmetry; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut overlap_lanes = [[0; 9]; N];
        for symmetry in Symmetry::ALL {
            let perm = symmetry.grid3_perm();
            let mut grouped_indices = AlignedLaneIndexBatch::default();
            let mut source_lanes = [[0; 9]; N];
            let mut grouped_count = 0;
            for lane in 0..active_lanes {
                if inverse_symmetries[lane] == symmetry {
                    grouped_indices.0[grouped_count] = lane;
                    source_lanes[grouped_count] = canonical_overlap_lanes[lane];
                    grouped_count += 1;
                }
            }
            if grouped_count == 0 {
                continue;
            }
            let source_words = transpose_u64_lanes_9xn(&source_lanes, grouped_count);
            let transformed_words =
                source_words.map(|word| self.transform_node_batch(word, symmetry));
            let transformed_lanes = transpose_u64_words_9xn(grouped_count, &transformed_words);
            for index in 0..grouped_count {
                let lane = grouped_indices.0[index];
                let transformed = transformed_lanes[index];
                overlap_lanes[lane] = [
                    transformed[perm[0]],
                    transformed[perm[1]],
                    transformed[perm[2]],
                    transformed[perm[3]],
                    transformed[perm[4]],
                    transformed[perm[5]],
                    transformed[perm[6]],
                    transformed[perm[7]],
                    transformed[perm[8]],
                ];
            }
        }
        overlap_lanes
    }

    pub(super) fn join_intents_from_child_words_5x4xn<const N: usize>(
        active_lanes: usize,
        is_active: &[bool; N],
        levels: &[u32; N],
        child_words: &[[[u64; 4]; N]; 5],
    ) -> [[Option<JoinIntent>; N]; 5] {
        let mut intents = [[None; N]; 5];
        for join_index in 0..5 {
            for lane in 0..active_lanes {
                if is_active[lane] {
                    intents[join_index][lane] = Some(JoinIntent {
                        level: levels[lane],
                        children: child_words[join_index][lane],
                    });
                }
            }
        }
        intents
    }

    pub(super) fn build_step0_provisional_records_staged(
        &mut self,
        cache_keys: &[CanonicalJumpKey],
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
    ) {
        let mut canonical_jump_keys = [CanonicalJumpKey {
            packed: PackedNodeKey::new(0, [0; 4]),
            step_exp: 0,
        }; SIMD_BATCH_LANES];
        for (lane, cache_key) in cache_keys.iter().enumerate() {
            canonical_jump_keys[lane] = *cache_key;
        }
        let overlaps = self.probe_and_build_overlaps_from_canonical_keys_staged(
            &canonical_jump_keys,
            cache_keys.len(),
        );
        let (centered_lanes, population_lanes) =
            self.build_centered_population_lanes_9xn(&overlaps, cache_keys.len());
        for (lane, &cache_key) in cache_keys.iter().enumerate() {
            let input_nodes = centered_lanes[lane];
            let input_populations = population_lanes[lane];
            self.stats.step0_provisional_records += 1;
            provisional_candidates.push(SimdProvisionalRecord {
                cache_key,
                level: cache_key.packed.level,
                kind: SimdTaskKind::Step0,
                input_nodes,
                input_populations,
                payload: SimdProvisionalPayload::Step0 {
                    dispatch: Step0LaneDispatch::SimdChild,
                },
            });
        }
    }

    pub(super) fn build_centered_population_lanes_9xn(
        &mut self,
        overlap_lanes: &[[NodeId; 9]; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> ([[u64; 9]; SIMD_BATCH_LANES], [[u64; 9]; SIMD_BATCH_LANES]) {
        let overlap_words = transpose_u64_lanes_9xn(overlap_lanes, active_lanes);
        let mut centered_lanes = AlignedU64LaneWords9::default();
        let mut population_lanes = AlignedU64LaneWords9::default();
        for index in 0..9 {
            let centered = self.centered_subnode_batch(overlap_words[index], active_lanes);
            for lane in 0..active_lanes {
                centered_lanes.0[lane][index] = centered[lane];
                population_lanes.0[lane][index] = self.node_columns.population(centered[lane]);
            }
        }
        (centered_lanes.0, population_lanes.0)
    }

    pub(super) fn build_step0_combined_children(
        &mut self,
        provisional: &SimdProvisionalRecord,
        lane_result: SimdLaneResult,
    ) -> [NodeId; 4] {
        let level = provisional.level;
        debug_assert!(level >= 3);
        let empty_child = self.empty(level - 1);
        let centered = provisional.input_nodes;
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

    pub(super) fn resolve_join_intents_staged<const N: usize>(
        &mut self,
        intents: [Option<JoinIntent>; N],
    ) -> [Option<NodeId>; N] {
        let mut resolved = [None; N];
        let mut packed_keys = [PackedNodeKey::new(0, [0; 4]); N];
        let mut lane_map = [usize::MAX; N];
        let mut active = 0;

        for (lane, intent) in intents.iter().enumerate() {
            let Some(intent) = intent else {
                continue;
            };
            packed_keys[active] = PackedNodeKey::new(intent.level, intent.children);
            lane_map[active] = lane;
            active += 1;
        }
        if active == 0 {
            return resolved;
        }

        let mut active_levels = AlignedU32Batch::default();
        let mut active_words = AlignedU64WordBatch4::default();
        for (slot, key) in packed_keys[..active].iter().copied().enumerate() {
            active_levels.0[slot] = key.level;
            active_words.0[0][slot] = key.children[0];
            active_words.0[1][slot] = key.children[1];
            active_words.0[2][slot] = key.children[2];
            active_words.0[3][slot] = key.children[3];
        }
        let active_fingerprints = hash_u64_words_with_level_batch(active_levels.0, active_words.0);
        let mut packed_fingerprints = [0_u64; N];
        packed_fingerprints[..active].copy_from_slice(&active_fingerprints[..active]);

        self.stats.cache_probe_batches += 1;
        let cached =
            self.intern
                .get_many_with_fingerprints(&packed_keys, &packed_fingerprints, active);
        let mut unresolved_slots = [usize::MAX; N];
        let mut unresolved_count = 0;

        for slot in 0..active {
            let lane = lane_map[slot];
            if let Some(node_id) = cached[slot] {
                resolved[lane] = Some(node_id);
            } else {
                unresolved_slots[unresolved_count] = slot;
                unresolved_count += 1;
            }
        }

        if unresolved_count == 0 {
            return resolved;
        }

        let mut order = [usize::MAX; N];
        for (index, slot) in order[..unresolved_count].iter_mut().enumerate() {
            *slot = index;
        }
        order[..unresolved_count]
            .sort_unstable_by_key(|&index| packed_fingerprints[unresolved_slots[index]]);
        let mut committed_keys = [PackedNodeKey::new(0, [0; 4]); N];
        let mut committed_values = [0; N];
        let mut committed_count = 0;

        for &index in &order[..unresolved_count] {
            let slot = unresolved_slots[index];
            let key = packed_keys[slot];
            let mut reused = None;
            for committed_index in 0..committed_count {
                if committed_keys[committed_index] == key {
                    reused = Some(committed_values[committed_index]);
                    break;
                }
            }
            if let Some(node_id) = reused {
                let lane = lane_map[slot];
                resolved[lane] = Some(node_id);
                continue;
            }

            let [nw, ne, sw, se] = key.children;
            let population = self.node_columns.population(nw)
                + self.node_columns.population(ne)
                + self.node_columns.population(sw)
                + self.node_columns.population(se);
            let node_id = self.push_node(key.level, population, nw, ne, sw, se);
            self.intern.insert(key, node_id);
            committed_keys[committed_count] = key;
            committed_values[committed_count] = node_id;
            committed_count += 1;
            let lane = lane_map[slot];
            resolved[lane] = Some(node_id);
        }

        resolved
    }

    pub(super) fn pack_simd_batch(
        provisional_candidates: &[SimdProvisionalRecord],
    ) -> SimdPackedBatch {
        debug_assert!(!provisional_candidates.is_empty());
        debug_assert!(provisional_candidates.len() <= SIMD_BATCH_LANES);
        let mut populations = AlignedU64WordBatch9::default();
        let mut active_mask = 0_u8;
        for (lane, provisional) in provisional_candidates.iter().enumerate() {
            active_mask |= 1 << lane;
            populations.0[0][lane] = provisional.input_populations[0];
            populations.0[1][lane] = provisional.input_populations[1];
            populations.0[2][lane] = provisional.input_populations[2];
            populations.0[3][lane] = provisional.input_populations[3];
            populations.0[4][lane] = provisional.input_populations[4];
            populations.0[5][lane] = provisional.input_populations[5];
            populations.0[6][lane] = provisional.input_populations[6];
            populations.0[7][lane] = provisional.input_populations[7];
            populations.0[8][lane] = provisional.input_populations[8];
        }
        SimdPackedBatch {
            active_lanes: provisional_candidates.len(),
            active_mask,
            populations: populations.0.map(must_cast),
        }
    }

    pub(super) fn evaluate_simd_batch(batch: &SimdPackedBatch) -> SimdBatchResult {
        debug_assert_eq!(batch.active_lanes, batch.active_mask.count_ones() as usize);
        let output_population_vectors = [
            batch.populations[0]
                + batch.populations[1]
                + batch.populations[3]
                + batch.populations[4],
            batch.populations[1]
                + batch.populations[2]
                + batch.populations[4]
                + batch.populations[5],
            batch.populations[3]
                + batch.populations[4]
                + batch.populations[6]
                + batch.populations[7],
            batch.populations[4]
                + batch.populations[5]
                + batch.populations[7]
                + batch.populations[8],
        ];
        let output_nonzero_masks = output_population_vectors
            .map(|vector| Self::nonzero_lane_mask(vector) & batch.active_mask);
        let mut lanes = [SimdLaneResult {
            output_nonzero_mask: 0,
        }; SIMD_BATCH_LANES];
        for lane in 0..SIMD_BATCH_LANES {
            if (batch.active_mask & (1 << lane)) == 0 {
                continue;
            }
            let output_nonzero_mask = u8::from((output_nonzero_masks[0] & (1 << lane)) != 0)
                | (u8::from((output_nonzero_masks[1] & (1 << lane)) != 0) << 1)
                | (u8::from((output_nonzero_masks[2] & (1 << lane)) != 0) << 2)
                | (u8::from((output_nonzero_masks[3] & (1 << lane)) != 0) << 3);
            lanes[lane] = SimdLaneResult {
                output_nonzero_mask,
            };
        }
        SimdBatchResult { lanes }
    }

    pub(super) fn build_phase1_provisional_record(
        &mut self,
        task_id: usize,
        task_key: CanonicalJumpKey,
        next_exp: u32,
        inputs: [NodeId; 9],
    ) -> SimdProvisionalRecord {
        let input_nodes = self.jump_result_batch(inputs, next_exp);
        let input_populations = input_nodes.map(|node| self.node_columns.population(node));
        self.stats.phase1_provisional_records += 1;
        SimdProvisionalRecord {
            cache_key: task_key,
            level: task_key.packed.level,
            kind: SimdTaskKind::PhaseOne,
            input_nodes,
            input_populations,
            payload: SimdProvisionalPayload::Recursive {
                next_exp,
                source_task_id: task_id,
            },
        }
    }

    pub(super) fn build_phase2_provisional_record(
        &mut self,
        task_id: usize,
        task_key: CanonicalJumpKey,
        next_exp: u32,
        inputs: [NodeId; 4],
    ) -> SimdProvisionalRecord {
        let mut input_nodes = [self.empty(0); 9];
        let jump_inputs = self.jump_result_batch(inputs, next_exp);
        input_nodes[..4].copy_from_slice(&jump_inputs);
        let input_populations = input_nodes.map(|node| self.node_columns.population(node));
        self.stats.phase2_provisional_records += 1;
        SimdProvisionalRecord {
            cache_key: task_key,
            level: task_key.packed.level,
            kind: SimdTaskKind::PhaseTwo,
            input_nodes,
            input_populations,
            payload: SimdProvisionalPayload::Recursive {
                next_exp,
                source_task_id: task_id,
            },
        }
    }

    pub(super) fn jump_result_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        step_exp: u32,
    ) -> [NodeId; N] {
        let mut results = [0; N];
        let mut unique_nodes = [0; N];
        let mut lane_to_unique = [usize::MAX; N];
        let mut unique_cache_keys = [PackedJumpCacheKey {
            packed: PackedNodeKey::new(0, [0; 4]),
            step_exp: 0,
        }; N];
        let mut unique_inverse_symmetries = [Symmetry::Identity; N];
        let mut unique_fingerprints = [0_u64; N];
        let mut unique_count = 0;
        for lane in 0..N {
            let node = nodes[lane];
            let mut existing = None;
            for index in 0..unique_count {
                if unique_nodes[index] == node {
                    existing = Some(index);
                    break;
                }
            }
            if let Some(index) = existing {
                self.stats.jump_batch_reused_queries += 1;
                lane_to_unique[lane] = index;
            } else {
                let (cache_key, symmetry, fingerprint, used_cached_fingerprint) =
                    self.canonical_cache_key_packed_only((node, step_exp));
                self.record_fingerprint_probe(used_cached_fingerprint, 1);
                unique_nodes[unique_count] = node;
                unique_cache_keys[unique_count] = cache_key;
                unique_inverse_symmetries[unique_count] = symmetry.inverse();
                unique_fingerprints[unique_count] = fingerprint;
                lane_to_unique[lane] = unique_count;
                unique_count += 1;
                self.stats.jump_batch_unique_queries += 1;
            }
        }

        if unique_count == 0 {
            return results;
        }

        self.stats.cache_probe_batches += 1;
        let cached = self.jump_cache.get_many_with_fingerprints(
            &unique_cache_keys,
            &unique_fingerprints,
            unique_count,
        );
        self.stats.packed_cache_result_lookups += unique_count;
        let mut unique_oriented_packed = [PackedNodeKey::new(0, [0; 4]); N];
        let mut unique_oriented_symmetry = [Symmetry::Identity; N];
        let mut unique_to_oriented = [usize::MAX; N];
        let mut oriented_result_nodes = [0_u64; N];
        let mut oriented_count = 0;
        for index in 0..unique_count {
            self.stats.jump_cache_hits += 1;
            let cached_entry = cached[index].unwrap_or_else(|| {
                panic!(
                    "missing HashLife jump result for grouped batch node={} step_exp={step_exp}",
                    unique_nodes[index]
                )
            });
            self.stats.packed_cache_result_hits += 1;
            let output_symmetry = unique_inverse_symmetries[index];
            let combined = cached_entry.symmetry.inverse().then(output_symmetry);
            let mut existing = None;
            for oriented in 0..oriented_count {
                if unique_oriented_packed[oriented] == cached_entry.packed
                    && unique_oriented_symmetry[oriented] == combined
                {
                    existing = Some(oriented);
                    break;
                }
            }
            unique_to_oriented[index] = if let Some(oriented) = existing {
                oriented
            } else {
                unique_oriented_packed[oriented_count] = cached_entry.packed;
                unique_oriented_symmetry[oriented_count] = combined;
                oriented_count += 1;
                oriented_count - 1
            };
        }
        for oriented in 0..oriented_count {
            oriented_result_nodes[oriented] = self.materialize_oriented_packed_result(
                unique_oriented_packed[oriented],
                Symmetry::Identity,
                unique_oriented_symmetry[oriented],
            );
        }
        for lane in 0..N {
            results[lane] = oriented_result_nodes[unique_to_oriented[lane_to_unique[lane]]];
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero_lane_mask_matches_scalar_lane_order() {
        let lanes: u64x8 = must_cast([0_u64, 7, 0, 11, 13, 0, 0, 19]);
        assert_eq!(HashLifeEngine::nonzero_lane_mask(lanes), 0b1001_1010);
    }

    #[test]
    fn evaluate_simd_batch_masks_match_scalar_results() {
        let batch = SimdPackedBatch {
            active_lanes: 5,
            active_mask: 0b0001_1111,
            populations: [
                must_cast([1_u64, 0, 0, 4, 0, 0, 0, 0]),
                must_cast([0_u64, 0, 2, 0, 0, 0, 0, 0]),
                must_cast([0_u64, 3, 0, 0, 0, 0, 0, 0]),
                must_cast([0_u64, 0, 0, 0, 5, 0, 0, 0]),
                must_cast([0_u64, 1, 0, 0, 0, 0, 0, 0]),
                must_cast([0_u64, 0, 0, 6, 0, 0, 0, 0]),
                must_cast([0_u64, 0, 7, 0, 0, 0, 0, 0]),
                must_cast([0_u64, 0, 0, 0, 8, 0, 0, 0]),
                must_cast([9_u64, 0, 0, 0, 0, 0, 0, 0]),
            ],
        };

        let result = HashLifeEngine::evaluate_simd_batch(&batch);
        let arrays = batch
            .populations
            .map(bytemuck::must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>);
        for lane in 0..batch.active_lanes {
            let scalar_mask = u8::from(
                arrays[0][lane] + arrays[1][lane] + arrays[3][lane] + arrays[4][lane] != 0,
            ) | (u8::from(
                arrays[1][lane] + arrays[2][lane] + arrays[4][lane] + arrays[5][lane] != 0,
            ) << 1)
                | (u8::from(
                    arrays[3][lane] + arrays[4][lane] + arrays[6][lane] + arrays[7][lane] != 0,
                ) << 2)
                | (u8::from(
                    arrays[4][lane] + arrays[5][lane] + arrays[7][lane] + arrays[8][lane] != 0,
                ) << 3);
            assert_eq!(result.lanes[lane].output_nonzero_mask, scalar_mask);
        }
    }
}
