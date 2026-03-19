use super::*;
#[cfg(test)]
use crate::simd_layout::AlignedLaneIndexBatch;
use crate::simd_layout::AlignedU64LaneWords9;

impl HashLifeEngine {
    fn write_population_lane<const N: usize>(
        populations: &mut AlignedU64WordBatch9,
        lane: usize,
        input_populations: [u64; N],
    ) {
        for (index, population) in input_populations.into_iter().enumerate() {
            populations.0[index][lane] = population;
        }
    }

    fn populate_nodes_and_populations<const N: usize>(
        &mut self,
        source_nodes: &[NodeId],
    ) -> ([NodeId; N], [u64; N]) {
        let mut nodes = [0; N];
        let mut populations = [0; N];
        nodes.copy_from_slice(&source_nodes[..N]);
        for (index, node) in nodes.into_iter().enumerate() {
            populations[index] = self.node_columns.population(node);
            nodes[index] = node;
        }
        (nodes, populations)
    }

    fn overlap_miss_join_intents_from_records<const N: usize>(
        miss_records: &[OverlapMissRecord; N],
        miss_unique_count: usize,
    ) -> [[Option<JoinIntent>; N]; 5] {
        let mut intents = [[None; N]; 5];
        for unique in 0..miss_unique_count {
            let join_level = miss_records[unique].join_level;
            for join_index in 0..5 {
                intents[join_index][unique] = Some(JoinIntent {
                    level: join_level,
                    children: miss_records[unique].join_children[join_index],
                });
            }
        }
        intents
    }

    pub(super) fn probe_and_build_canonical_overlaps_staged<const N: usize>(
        &mut self,
        identities: &[CanonicalNodeIdentity; N],
        fingerprints: &[u64; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        #[derive(Clone, Copy)]
        struct DuplicateMissLane {
            lane: usize,
            unique: usize,
        }

        let mut canonical_overlap_lanes = [[0; 9]; N];
        let mut structural_keys = [CanonicalStructKey::new(0, [0; 4]); N];
        for lane in 0..active_lanes {
            structural_keys[lane] = identities[lane].structural;
        }
        let cached = self.overlap_cache.get_many_with_fingerprints(
            &structural_keys,
            fingerprints,
            active_lanes,
        );
        let mut miss_records = [OverlapMissRecord {
            representative_lane: 0,
            identity: CanonicalNodeIdentity {
                packed: PackedNodeKey::new(0, [0; 4]),
                structural: CanonicalStructKey::new(0, [0; 4]),
                symmetry: Symmetry::Identity,
            },
            fingerprint: 0,
            join_level: 0,
            join_children: [[0; 4]; 5],
            overlaps: [0; 9],
        }; N];
        let mut duplicate_miss_lanes = [DuplicateMissLane { lane: 0, unique: 0 }; N];
        let mut duplicate_miss_count = 0usize;
        let mut miss_unique_count = 0;
        let mut unique_lookup =
            FlatTable::<CanonicalStructKey, usize>::with_capacity(active_lanes.max(4));

        for lane in 0..active_lanes {
            if let Some(lane_overlaps) = cached[lane] {
                self.stats.overlap_cache_hits += 1;
                canonical_overlap_lanes[lane] = lane_overlaps;
                continue;
            }

            let canonical_identity = identities[lane];
            let canonical_key = canonical_identity.packed;
            let structural_key = canonical_identity.structural;
            if let Some(unique) = unique_lookup.get_with_fingerprint(&structural_key, fingerprints[lane])
            {
                self.stats.overlap_local_reuse_lanes += 1;
                duplicate_miss_lanes[duplicate_miss_count] = DuplicateMissLane { lane, unique };
                duplicate_miss_count += 1;
                continue;
            }

            self.stats.overlap_cache_misses += 1;
            let [nw, ne, sw, se] = canonical_key.children;
            let [_, nw_ne, nw_sw, nw_se] = self.node_columns.quadrants(nw);
            let [ne_nw, _, ne_sw, ne_se] = self.node_columns.quadrants(ne);
            let [sw_nw, sw_ne, _, sw_se] = self.node_columns.quadrants(sw);
            let [se_nw, se_ne, se_sw, _] = self.node_columns.quadrants(se);
            miss_records[miss_unique_count] = OverlapMissRecord {
                representative_lane: lane,
                identity: canonical_identity,
                fingerprint: fingerprints[lane],
                join_level: canonical_key.level - 1,
                join_children: [
                    [nw_ne, ne_nw, nw_se, ne_sw],
                    [nw_sw, nw_se, sw_nw, sw_ne],
                    [nw_se, ne_sw, sw_ne, se_nw],
                    [ne_sw, ne_se, se_nw, se_ne],
                    [sw_ne, se_nw, sw_se, se_sw],
                ],
                overlaps: [0; 9],
            };
            unique_lookup.insert_with_fingerprint(
                structural_key,
                fingerprints[lane],
                miss_unique_count,
            );
            miss_unique_count += 1;
        }

        if miss_unique_count != 0 {
            let miss_join_intents =
                Self::overlap_miss_join_intents_from_records(&miss_records, miss_unique_count);
            let resolved_join_0 = self.resolve_join_intents_staged(miss_join_intents[0]);
            let resolved_join_1 = self.resolve_join_intents_staged(miss_join_intents[1]);
            let resolved_join_2 = self.resolve_join_intents_staged(miss_join_intents[2]);
            let resolved_join_3 = self.resolve_join_intents_staged(miss_join_intents[3]);
            let resolved_join_4 = self.resolve_join_intents_staged(miss_join_intents[4]);

            for unique in 0..miss_unique_count {
                let miss_record = &mut miss_records[unique];
                let canonical_key = miss_record.identity.packed;
                let [nw, ne, sw, se] = canonical_key.children;
                miss_record.overlaps = [
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
                    miss_record.identity.structural,
                    miss_record.fingerprint,
                    miss_record.overlaps,
                );
                canonical_overlap_lanes[miss_record.representative_lane] = miss_record.overlaps;
            }

            for duplicate in &duplicate_miss_lanes[..duplicate_miss_count] {
                canonical_overlap_lanes[duplicate.lane] = miss_records[duplicate.unique].overlaps;
            }
        }

        self.stats.overlap_prep_lanes += active_lanes;
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
        let (packed, structural, symmetry, fingerprint, used_cached_fingerprint) =
            if self.record_symmetry_gate_decision(node) {
                let canonical = self.canonicalize_packed_node(node);
                (
                    canonical.node.packed,
                    canonical.node.structural,
                    canonical.node.symmetry,
                    canonical.fingerprint,
                    canonical.used_cached_fingerprint,
                )
            } else {
                let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
                let transform_id = self.transform_packed_node_key(packed, Symmetry::Identity);
                let structural = self.structural_key_from_transform_id(transform_id);
                (packed, structural, Symmetry::Identity, fingerprint, true)
            };
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        if let Some(overlaps) = self
            .overlap_cache
            .get_with_fingerprint(&structural, fingerprint)
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
            .insert_with_fingerprint(structural, fingerprint, overlaps);
        symmetry.inverse().transform_overlap_nodes(self, overlaps)
    }

    #[cfg(test)]
    pub(super) fn probe_and_build_overlaps_staged<const N: usize>(
        &mut self,
        nodes: &[NodeId; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut inverse_symmetries = [Symmetry::Identity; N];
        let mut identities = [CanonicalNodeIdentity {
            packed: PackedNodeKey::new(0, [0; 4]),
            structural: CanonicalStructKey::new(0, [0; 4]),
            symmetry: Symmetry::Identity,
        }; N];
        let mut fingerprints = [0_u64; N];
        let canonicalized = self.canonicalize_packed_nodes_batch(nodes, active_lanes);
        for lane in 0..active_lanes {
            let (packed, structural, symmetry, fingerprint, used_cached_fingerprint) =
                if self.record_symmetry_gate_decision(nodes[lane]) {
                    let canonical = canonicalized[lane];
                    (
                        canonical.node.packed,
                        canonical.node.structural,
                        canonical.node.symmetry,
                        canonical.fingerprint,
                        canonical.used_cached_fingerprint,
                    )
                } else {
                    let (packed, fingerprint) =
                        self.node_columns.packed_key_and_fingerprint(nodes[lane]);
                    let transform_id = self.transform_packed_node_key(packed, Symmetry::Identity);
                    let structural = self.structural_key_from_transform_id(transform_id);
                    (packed, structural, Symmetry::Identity, fingerprint, true)
                };
            self.record_fingerprint_probe(used_cached_fingerprint, 1);
            inverse_symmetries[lane] = symmetry.inverse();
            identities[lane] = CanonicalNodeIdentity {
                packed,
                structural,
                symmetry,
            };
            fingerprints[lane] = fingerprint;
        }

        self.stats.cache_probe_batches += 1;
        self.stats.scheduler_probe_batches += 1;
        self.stats.overlap_prep_batches += 1;
        let canonical_overlap_lanes =
            self.probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, active_lanes);
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
        let mut reuse = FlatTable::<NodeId, NodeId>::with_capacity(active_lanes.max(4));
        for lane in 0..active_lanes {
            let node = nodes[lane];
            if self.node_columns.level(node) == 1 {
                centered[lane] = node;
                continue;
            }
            if let Some(reused) = reuse.get(&node) {
                centered[lane] = reused;
                continue;
            }
            let computed = self.centered_subnode(node);
            centered[lane] = computed;
            reuse.insert(node, computed);
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

    pub(super) fn build_step0_provisional_records_staged(
        &mut self,
        discovered_tasks: &[DiscoveredJumpTask],
        provisional_candidates: &mut Vec<SimdProvisionalRecord>,
    ) {
        let mut identities = [CanonicalNodeIdentity {
            packed: PackedNodeKey::new(0, [0; 4]),
            structural: CanonicalStructKey::new(0, [0; 4]),
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
        self.stats.cache_probe_batches += 1;
        self.stats.scheduler_probe_batches += 1;
        self.stats.overlap_prep_batches += 1;
        self.stats.packed_overlap_outputs_produced += discovered_tasks.len();
        let overlaps = self
            .probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, discovered_tasks.len());
        let (centered_lanes, population_lanes) =
            self.build_centered_population_lanes_9xn(&overlaps, discovered_tasks.len());
        for (lane, discovered_task) in discovered_tasks.iter().enumerate() {
            let cache_key = discovered_task.key;
            let input_nodes = centered_lanes[lane];
            let input_populations = population_lanes[lane];
            self.stats.step0_provisional_records += 1;
            provisional_candidates.push(SimdProvisionalRecord {
                cache_key,
                level: cache_key.structural.level,
                inputs: SimdProvisionalInputs::Nine {
                    nodes: input_nodes,
                    populations: input_populations,
                },
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
        let centered = match provisional.inputs {
            SimdProvisionalInputs::Nine { nodes, .. } => nodes,
            SimdProvisionalInputs::Four { .. } => {
                unreachable!("step0 provisional records must carry 9-node inputs")
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

        let mut committed =
            FlatTable::<PackedNodeKey, NodeId>::with_capacity(unresolved_count.max(4));
        for &slot in &unresolved_slots[..unresolved_count] {
            let key = packed_keys[slot];
            if let Some(node_id) = committed.get_with_fingerprint(&key, packed_fingerprints[slot]) {
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
            committed.insert_with_fingerprint(key, packed_fingerprints[slot], node_id);
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
            match provisional.inputs {
                SimdProvisionalInputs::Nine {
                    populations: input_populations,
                    ..
                } => {
                    Self::write_population_lane(&mut populations, lane, input_populations);
                }
                SimdProvisionalInputs::Four {
                    populations: input_populations,
                    ..
                } => {
                    Self::write_population_lane(&mut populations, lane, input_populations);
                }
            }
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

    fn jump_result_query_batch<const N: usize>(
        &mut self,
        queries: [JumpQuery; N],
        active_lanes: usize,
    ) -> [NodeId; N] {
        let mut results = [0; N];
        let mut unique_queries = [UniqueJumpQueryRecord {
            query: JumpQuery {
                node: 0,
                step_exp: 0,
            },
            cache_key: CanonicalJumpKey::empty(),
            inverse_symmetry: Symmetry::Identity,
            fingerprint: 0,
        }; N];
        let mut lane_to_unique = [usize::MAX; N];
        let mut unique_count = 0;
        let mut unique_lookup = FlatTable::<JumpQuery, usize>::with_capacity(N.max(4));
        for lane in 0..active_lanes {
            let query = queries[lane];
            if let Some(index) = unique_lookup.get(&query) {
                self.stats.jump_batch_reused_queries += 1;
                lane_to_unique[lane] = index;
            } else {
                let jump_probe = self.canonical_jump_probe((query.node, query.step_exp));
                self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
                unique_queries[unique_count] = UniqueJumpQueryRecord {
                    query,
                    cache_key: jump_probe.key,
                    inverse_symmetry: jump_probe.node.symmetry.inverse(),
                    fingerprint: jump_probe.fingerprint,
                };
                lane_to_unique[lane] = unique_count;
                unique_lookup.insert(query, unique_count);
                unique_count += 1;
                self.stats.jump_batch_unique_queries += 1;
            }
        }

        if unique_count == 0 {
            return results;
        }

        self.stats.cache_probe_batches += 1;
        let unique_cache_keys = unique_queries.map(|record| record.cache_key);
        let unique_fingerprints = unique_queries.map(|record| record.fingerprint);
        let cached = self.jump_cache.get_many_with_fingerprints(
            &unique_cache_keys,
            &unique_fingerprints,
            unique_count,
        );
        self.stats.jump_result_cache_lookups += unique_count;
        let mut unique_to_oriented = [usize::MAX; N];
        let mut oriented_results = [UniqueOrientedResultRecord {
            packed: PackedNodeKey::new(0, [0; 4]),
            symmetry: Symmetry::Identity,
            node: 0,
        }; N];
        let mut oriented_count = 0;
        let mut oriented_lookup =
            FlatTable::<OrientedPackedResultKey, usize>::with_capacity(unique_count.max(4));
        for index in 0..unique_count {
            let cached_entry = cached[index].unwrap_or_else(|| {
                panic!(
                    "missing HashLife jump result for grouped batch node={} step_exp={}",
                    unique_queries[index].query.node, unique_queries[index].query.step_exp,
                )
            });
            self.stats.jump_result_cache_hits += 1;
            let output_symmetry = unique_queries[index].inverse_symmetry;
            let combined = cached_entry.symmetry.inverse().then(output_symmetry);
            if combined != Symmetry::Identity {
                self.stats.symmetric_jump_result_cache_hits += 1;
            }
            let oriented_key = OrientedPackedResultKey {
                packed: cached_entry.packed,
                symmetry: combined,
            };
            unique_to_oriented[index] = if let Some(oriented) = oriented_lookup.get(&oriented_key) {
                oriented
            } else {
                oriented_results[oriented_count] = UniqueOrientedResultRecord {
                    packed: cached_entry.packed,
                    symmetry: combined,
                    node: 0,
                };
                oriented_lookup.insert(oriented_key, oriented_count);
                oriented_count += 1;
                oriented_count - 1
            };
        }
        for oriented in 0..oriented_count {
            oriented_results[oriented].node = self.materialize_oriented_packed_result(
                oriented_results[oriented].packed,
                Symmetry::Identity,
                oriented_results[oriented].symmetry,
            );
        }
        for lane in 0..active_lanes {
            results[lane] = oriented_results[unique_to_oriented[lane_to_unique[lane]]].node;
        }
        results
    }

    #[cfg(test)]
    pub(super) fn jump_result_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        step_exp: u32,
    ) -> [NodeId; N] {
        self.jump_result_query_batch(nodes.map(|node| JumpQuery { node, step_exp }), N)
    }

    pub(super) fn build_phase1_provisional_records_batch(
        &mut self,
        ready_lanes: &[Phase1ReadyLane; SIMD_BATCH_LANES],
        active_lanes: usize,
        out: &mut Vec<SimdProvisionalRecord>,
    ) {
        if active_lanes == 0 {
            return;
        }
        let mut queries = [JumpQuery {
            node: 0,
            step_exp: 0,
        }; SIMD_BATCH_LANES * 9];
        let mut query_count = 0;
        for lane in 0..active_lanes {
            for index in 0..9 {
                queries[query_count] = JumpQuery {
                    node: ready_lanes[lane].inputs[index],
                    step_exp: ready_lanes[lane].next_exp,
                };
                query_count += 1;
            }
        }
        let query_results = self.jump_result_query_batch(queries, query_count);
        for lane in 0..active_lanes {
            let base = lane * 9;
            let (input_nodes, input_populations) =
                self.populate_nodes_and_populations::<9>(&query_results[base..base + 9]);
            self.stats.phase1_provisional_records += 1;
            out.push(SimdProvisionalRecord {
                cache_key: ready_lanes[lane].key,
                level: ready_lanes[lane].key.structural.level,
                inputs: SimdProvisionalInputs::Nine {
                    nodes: input_nodes,
                    populations: input_populations,
                },
                payload: SimdProvisionalPayload::PhaseOne {
                    next_exp: ready_lanes[lane].next_exp,
                    source_task_id: ready_lanes[lane].task_id,
                },
            });
        }
    }

    pub(super) fn build_phase2_provisional_records_batch(
        &mut self,
        ready_lanes: &[Phase2ReadyLane; SIMD_BATCH_LANES],
        active_lanes: usize,
        out: &mut Vec<SimdProvisionalRecord>,
    ) {
        if active_lanes == 0 {
            return;
        }
        let mut queries = [JumpQuery {
            node: 0,
            step_exp: 0,
        }; SIMD_BATCH_LANES * 4];
        let mut query_count = 0;
        for lane in 0..active_lanes {
            for index in 0..4 {
                queries[query_count] = JumpQuery {
                    node: ready_lanes[lane].inputs[index],
                    step_exp: ready_lanes[lane].next_exp,
                };
                query_count += 1;
            }
        }
        let query_results = self.jump_result_query_batch(queries, query_count);
        for lane in 0..active_lanes {
            let base = lane * 4;
            let (input_nodes, input_populations) =
                self.populate_nodes_and_populations::<4>(&query_results[base..base + 4]);
            self.stats.phase2_provisional_records += 1;
            out.push(SimdProvisionalRecord {
                cache_key: ready_lanes[lane].key,
                level: ready_lanes[lane].key.structural.level,
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
