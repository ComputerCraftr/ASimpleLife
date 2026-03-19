use super::*;
#[cfg(test)]
use crate::simd_layout::AlignedLaneIndexBatch;
use crate::simd_layout::AlignedU64LaneWords9;

mod builders;

impl HashLifeEngine {
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
