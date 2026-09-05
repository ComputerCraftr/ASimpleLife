use super::*;
use crate::RequiredExt;
use crate::probe_table::ProbeTable;
#[cfg(test)]
use crate::simd_layout::AlignedLaneIndexBatch;
use crate::simd_layout::AlignedU64LaneWords9;

mod builders;
#[cfg(test)]
mod hot_path_tests;

impl HashLifeEngine {
    pub(super) fn probe_table_many<K: crate::probe_table::ProbeKey, V: Copy, const N: usize>(
        table: &ProbeTable<K, V>,
        keys: &[K; N],
        fingerprints: &[u64; N],
        active_lanes: usize,
    ) -> ([Option<V>; N], crate::hashlife::kernels::KernelAccounting) {
        let mut total = crate::hashlife::kernels::KernelAccounting::default();
        let kernels = crate::hashlife::kernels::KernelSet::selected();
        let values = table.get_many_with_fingerprints_using(
            keys,
            fingerprints,
            active_lanes,
            |controls, tags, lanes| {
                let mut expanded = [0_u16; N];
                for start in (0..lanes).step_by(SIMD_BATCH_LANES) {
                    let chunk_lanes = (lanes - start).min(SIMD_BATCH_LANES);
                    let mut kernel_tags = [0_u8; SIMD_BATCH_LANES];
                    kernel_tags[..chunk_lanes].copy_from_slice(&tags[start..start + chunk_lanes]);
                    let (matches, accounting) =
                        kernels.control_matches(controls, &kernel_tags, chunk_lanes);
                    total.accumulate(accounting);
                    expanded[start..start + chunk_lanes].copy_from_slice(&matches[..chunk_lanes]);
                }
                expanded
            },
        );
        (values, total)
    }

    pub(super) fn record_kernel_accounting(
        &mut self,
        accounting: crate::hashlife::kernels::KernelAccounting,
    ) {
        use crate::hashlife::kernels::contracts::KernelOperation;

        let stats = &mut self.stats.simd.kernel;
        stats.candidate_lanes += accounting.candidate_lanes;
        stats.portable_vector_lanes += accounting.portable_vector_lanes;
        stats.scalar_fallback_lanes += accounting.scalar_lanes;
        stats.native_avx2_lanes += accounting.native_avx2_lanes;
        stats.native_neon_lanes += accounting.native_neon_lanes;
        stats.vectorized_structural_lanes += accounting.portable_vector_lanes
            + accounting.native_avx2_lanes
            + accounting.native_neon_lanes;
        match accounting.operation {
            Some(KernelOperation::OutputPresence) => {
                stats.output_presence_kernel_lanes += accounting.candidate_lanes;
            }
            Some(KernelOperation::Fingerprint) => {
                stats.fingerprint_kernel_lanes += accounting.candidate_lanes;
            }
            Some(KernelOperation::ControlMatch) => {
                stats.control_match_kernel_lanes += accounting.candidate_lanes;
                stats.swar_control_groups += accounting.swar_control_groups;
                stats.native_avx2_control_groups += accounting.native_avx2_control_groups;
                stats.native_neon_control_groups += accounting.native_neon_control_groups;
            }
            Some(KernelOperation::D4Candidate) => {
                stats.d4_candidate_lanes += accounting.candidate_lanes;
                stats.native_d4_candidate_lanes += accounting.native_d4_candidate_lanes;
            }
            Some(KernelOperation::D4SemanticPrefix) => {
                stats.native_d4_prefix_compare_lanes += accounting.native_d4_prefix_compare_lanes;
                stats.native_d4_exact_winner_lanes += accounting.native_d4_exact_winner_lanes;
            }
            Some(KernelOperation::Population) => {
                stats.population_kernel_lanes += accounting.candidate_lanes;
            }
            Some(KernelOperation::BaseTransition) => {
                stats.base_transition_kernel_lanes += accounting.candidate_lanes;
            }
            Some(KernelOperation::Dedup) => {
                stats.dedup_kernel_lanes += accounting.candidate_lanes;
            }
            None => {}
        }
        debug_assert_eq!(
            accounting.candidate_lanes,
            accounting.portable_vector_lanes
                + accounting.scalar_lanes
                + accounting.native_avx2_lanes
                + accounting.native_neon_lanes,
            "HashLife kernel lane accounting must conserve admitted work"
        );
    }

    #[inline]
    fn valid_simd_active_mask(active_lanes: usize, active_mask: u8) -> bool {
        if active_lanes > SIMD_BATCH_LANES {
            return false;
        }
        let expected = if active_lanes == SIMD_BATCH_LANES {
            u8::MAX
        } else {
            u8::try_from((1_u16 << active_lanes) - 1).or_invariant("SIMD active mask exceeds u8")
        };
        active_mask == expected
    }

    fn overlap_miss_join_intents_from_records<const N: usize>(
        miss_records: &[OverlapMissRecord; N],
        miss_unique_count: usize,
    ) -> [[Option<JoinIntent>; N]; 5] {
        let mut intents = [[None; N]; 5];
        for unique in 0..miss_unique_count {
            let join_level = miss_records[unique].join_level;
            for (join_index, intent_lanes) in intents.iter_mut().enumerate() {
                intent_lanes[unique] = Some(JoinIntent {
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
    ) -> Option<[[NodeId; 9]; N]> {
        #[derive(Clone, Copy)]
        struct DuplicateMissLane {
            lane: usize,
            unique: usize,
        }

        let mut canonical_overlap_lanes = [[NodeId::ZERO; 9]; N];
        let mut structural_keys = [CanonicalStructKey::leaf(false); N];
        for lane in 0..active_lanes {
            structural_keys[lane] = identities[lane].structural;
        }
        let (cached, probe_accounting) = Self::probe_table_many(
            &self.result_caches.overlap,
            &structural_keys,
            fingerprints,
            active_lanes,
        );
        self.record_kernel_accounting(probe_accounting);
        let mut miss_records = [OverlapMissRecord {
            representative_lane: 0,
            identity: CanonicalNodeIdentity {
                packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
                structural: CanonicalStructKey::leaf(false),
                symmetry: Symmetry::Identity,
            },
            fingerprint: 0,
            join_level: 0,
            join_children: [[NodeId::ZERO; 4]; 5],
            overlaps: [NodeId::ZERO; 9],
        }; N];
        let mut duplicate_miss_lanes = [DuplicateMissLane { lane: 0, unique: 0 }; N];
        let mut duplicate_miss_count = 0usize;
        let mut miss_unique_count = 0;

        for lane in 0..active_lanes {
            if let Some(lane_overlaps) = cached[lane] {
                self.stats.result_cache.overlap_cache_hits += 1;
                canonical_overlap_lanes[lane] = lane_overlaps;
                continue;
            }

            let canonical_identity = identities[lane];
            let canonical_key = canonical_identity.packed;
            let structural_key = canonical_identity.structural;
            if let Some(unique) = miss_records[..miss_unique_count].iter().position(|record| {
                record.fingerprint == fingerprints[lane]
                    && record.identity.structural == structural_key
            }) {
                self.stats.simd.overlap_local_reuse_lanes += 1;
                duplicate_miss_lanes[duplicate_miss_count] = DuplicateMissLane { lane, unique };
                duplicate_miss_count += 1;
                continue;
            }

            self.stats.result_cache.overlap_cache_misses += 1;
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
                overlaps: [NodeId::ZERO; 9],
            };
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
            if self.allocation_failed() {
                return None;
            }

            for unique in 0..miss_unique_count {
                let miss_record = &mut miss_records[unique];
                let canonical_key = miss_record.identity.packed;
                let [nw, ne, sw, se] = canonical_key.children;
                miss_record.overlaps = [
                    nw,
                    resolved_join_0[unique].or_invariant("overlap join should resolve"),
                    ne,
                    resolved_join_1[unique].or_invariant("overlap join should resolve"),
                    resolved_join_2[unique].or_invariant("overlap join should resolve"),
                    resolved_join_3[unique].or_invariant("overlap join should resolve"),
                    sw,
                    resolved_join_4[unique].or_invariant("overlap join should resolve"),
                    se,
                ];
                self.publish_optional_cache(
                    |engine| &engine.result_caches.overlap,
                    |engine| &mut engine.result_caches.overlap,
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

        self.stats.simd.overlap_prep_lanes += active_lanes;
        Some(canonical_overlap_lanes)
    }

    #[inline]
    #[cfg(test)]
    fn nonzero_lane_mask(vector: u64x8) -> u8 {
        let zero_compare: [u64; SIMD_BATCH_LANES] = must_cast(vector.simd_eq(u64x8::ZERO));
        let mut mask = 0_u8;
        for (lane, &zero) in zero_compare.iter().enumerate() {
            mask |= u8::from(zero == 0) << lane;
        }
        mask
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(super) fn overlapping_subnodes(&mut self, node: NodeId) -> [NodeId; 9] {
        let (packed, structural, symmetry, used_cached_fingerprint) =
            if self.record_symmetry_gate_decision(node) {
                let canonical = self.canonicalize_packed_node(node);
                (
                    canonical.node.packed,
                    canonical.node.structural,
                    canonical.node.symmetry,
                    canonical.used_cached_fingerprint,
                )
            } else {
                let (packed, _) = self.node_columns.packed_key_and_fingerprint(node);
                let structural = self.symmetry_entry(node, Symmetry::Identity).structural;
                (packed, structural, Symmetry::Identity, true)
            };
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        let fingerprint = structural.fingerprint();
        if let Some(overlaps) = self
            .result_caches
            .overlap
            .get_with_fingerprint(&structural, fingerprint)
        {
            self.stats.result_cache.overlap_cache_hits += 1;
            self.stats.simd.overlap_prep_lanes += 1;
            return symmetry.inverse().transform_overlap_nodes(self, overlaps);
        }
        self.stats.result_cache.overlap_cache_misses += 1;
        self.stats.simd.overlap_prep_lanes += 1;
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
        self.publish_optional_cache(
            |engine| &engine.result_caches.overlap,
            |engine| &mut engine.result_caches.overlap,
            structural,
            fingerprint,
            overlaps,
        );
        symmetry.inverse().transform_overlap_nodes(self, overlaps)
    }

    pub(super) fn probe_and_build_overlaps_staged<const N: usize>(
        &mut self,
        nodes: &[NodeId; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut inverse_symmetries = [Symmetry::Identity; N];
        let mut identities = [CanonicalNodeIdentity {
            packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
            structural: CanonicalStructKey::leaf(false),
            symmetry: Symmetry::Identity,
        }; N];
        let mut fingerprints = [0_u64; N];
        let canonicalized = self.canonicalize_packed_nodes_batch(nodes, active_lanes);
        for lane in 0..active_lanes {
            if let Some(source_lane) = nodes[..lane]
                .iter()
                .position(|&candidate| candidate == nodes[lane])
            {
                identities[lane] = identities[source_lane];
                inverse_symmetries[lane] = inverse_symmetries[source_lane];
                fingerprints[lane] = fingerprints[source_lane];
                self.record_fingerprint_probe(true, 1);
                continue;
            }
            let (packed, structural, symmetry, used_cached_fingerprint) =
                if self.record_symmetry_gate_decision(nodes[lane]) {
                    let canonical = canonicalized[lane];
                    (
                        canonical.node.packed,
                        canonical.node.structural,
                        canonical.node.symmetry,
                        canonical.used_cached_fingerprint,
                    )
                } else {
                    let (packed, _) = self.node_columns.packed_key_and_fingerprint(nodes[lane]);
                    let structural = self
                        .symmetry_entry(nodes[lane], Symmetry::Identity)
                        .structural;
                    (packed, structural, Symmetry::Identity, true)
                };
            self.record_fingerprint_probe(used_cached_fingerprint, 1);
            identities[lane] = CanonicalNodeIdentity {
                packed,
                structural,
                symmetry,
            };
            inverse_symmetries[lane] = identities[lane].symmetry.inverse();
            fingerprints[lane] = structural.fingerprint();
        }

        self.stats.scheduler.cache_probe_batches += 1;
        self.stats.scheduler.scheduler_probe_batches += 1;
        self.stats.simd.overlap_prep_batches += 1;
        let Some(canonical_overlap_lanes) = self.probe_and_build_canonical_overlaps_staged(
            &identities,
            &fingerprints,
            active_lanes,
        ) else {
            return [[NodeId::ZERO; 9]; N];
        };
        self.transform_overlap_words_grouped(
            &canonical_overlap_lanes,
            &inverse_symmetries,
            active_lanes,
        )
    }
}

#[cfg(test)]
impl HashLifeEngine {
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
}

impl HashLifeEngine {
    pub(super) fn centered_subnode_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        active_lanes: usize,
    ) -> [NodeId; N] {
        debug_assert!(active_lanes <= SIMD_BATCH_LANES);
        let mut centered = [NodeId::ZERO; N];
        let mut intents = [None; N];
        for lane in 0..active_lanes {
            let node = nodes[lane];
            let level = self.node_columns.level(node);
            if level == 1 {
                centered[lane] = node;
                continue;
            }

            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            intents[lane] = Some(JoinIntent {
                level: level - 1,
                children: [
                    self.node_columns.quadrants(nw)[3],
                    self.node_columns.quadrants(ne)[2],
                    self.node_columns.quadrants(sw)[1],
                    self.node_columns.quadrants(se)[0],
                ],
            });
        }
        let resolved = self.resolve_join_intents_staged(intents);
        for lane in 0..active_lanes {
            if let Some(node) = resolved[lane] {
                centered[lane] = node;
            }
        }
        centered
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(super) fn transform_overlap_words_grouped<const N: usize>(
        &mut self,
        canonical_overlap_lanes: &[[NodeId; 9]; N],
        inverse_symmetries: &[Symmetry; N],
        active_lanes: usize,
    ) -> [[NodeId; 9]; N] {
        let mut overlap_lanes = [[NodeId::ZERO; 9]; N];
        for symmetry in Symmetry::ALL {
            let perm = symmetry.grid3_perm();
            let mut grouped_indices = AlignedLaneIndexBatch::default();
            let mut source_lanes = [[NodeId::ZERO; 9]; N];
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
            let mut transformed_lanes = [[NodeId::ZERO; 9]; N];
            for word_index in 0..9 {
                let mut word = [NodeId::ZERO; N];
                for lane in 0..grouped_count {
                    word[lane] = source_lanes[lane][word_index];
                }
                let transformed = self.transform_node_batch(word, symmetry);
                for lane in 0..grouped_count {
                    transformed_lanes[lane][word_index] = transformed[lane];
                }
            }
            for (&lane, &transformed) in grouped_indices.0[..grouped_count]
                .iter()
                .zip(&transformed_lanes)
            {
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
}

impl HashLifeEngine {
    pub(super) fn resolve_join_intents_staged<const N: usize>(
        &mut self,
        intents: [Option<JoinIntent>; N],
    ) -> [Option<NodeId>; N] {
        let mut resolved = [None; N];
        let mut packed_keys = [PackedNodeKey::new(0, [NodeId::ZERO; 4]); N];
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
            active_words.0[0][slot] = u64::from(key.children[0]);
            active_words.0[1][slot] = u64::from(key.children[1]);
            active_words.0[2][slot] = u64::from(key.children[2]);
            active_words.0[3][slot] = u64::from(key.children[3]);
        }
        let fingerprint_batch = crate::hashlife::kernels::contracts::FingerprintBatch {
            levels: active_levels.0,
            words: active_words.0,
            active_lanes: active,
        };
        let (active_fingerprints, fingerprint_accounting) =
            crate::hashlife::kernels::KernelSet::selected().fingerprints(&fingerprint_batch);
        self.record_kernel_accounting(fingerprint_accounting);
        debug_assert_eq!(
            active_fingerprints,
            hash_u64_words_with_level_batch(active_levels.0, active_words.0),
            "native structural fingerprints must match the portable implementation"
        );
        let mut packed_fingerprints = [0_u64; N];
        packed_fingerprints[..active].copy_from_slice(&active_fingerprints[..active]);

        self.stats.scheduler.cache_probe_batches += 1;
        let (cached, probe_accounting) =
            Self::probe_table_many(&self.intern, &packed_keys, &packed_fingerprints, active);
        self.record_kernel_accounting(probe_accounting);
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

        let dedup_batch = crate::hashlife::kernels::contracts::DedupBatch {
            fingerprints: active_fingerprints,
            words: active_words.0,
            active_lanes: active,
        };
        let (duplicate_sources, dedup_accounting) =
            crate::hashlife::kernels::KernelSet::selected().dedup(&dedup_batch);
        self.record_kernel_accounting(dedup_accounting);

        let mut population_batch = crate::hashlife::kernels::contracts::PopulationBatch {
            active_lanes: active,
            ..crate::hashlife::kernels::contracts::PopulationBatch::default()
        };
        for (slot, key) in packed_keys[..active].iter().enumerate() {
            for (child, node) in key.children.into_iter().enumerate() {
                let population = self.node_columns.population_stat(node);
                population_batch.lo[child][slot] = population.lo;
                population_batch.hi[child][slot] = population.hi;
                population_batch.saturated[child][slot] = u64::from(population.saturated);
            }
        }
        let (populations, population_accounting) =
            crate::hashlife::kernels::KernelSet::selected().aggregate_population(&population_batch);
        self.record_kernel_accounting(population_accounting);

        let mut unique_slots = [usize::MAX; N];
        let mut unique_count = 0;
        for &slot in &unresolved_slots[..unresolved_count] {
            if duplicate_sources[slot] == u8::MAX {
                unique_slots[unique_count] = slot;
                unique_count += 1;
            }
        }
        for index in 1..unique_count {
            let slot = unique_slots[index];
            let slot_key = packed_keys[slot];
            let slot_structural = self.canonical_parent_key(
                slot_key.level,
                slot_key
                    .children
                    .map(|child| self.node_columns.identity_ref(child)),
            );
            let mut insertion = index;
            while insertion != 0 {
                let previous_slot = unique_slots[insertion - 1];
                let previous_key = packed_keys[previous_slot];
                let previous_structural = self.canonical_parent_key(
                    previous_key.level,
                    previous_key
                        .children
                        .map(|child| self.node_columns.identity_ref(child)),
                );
                if self.compare_canonical_keys(previous_structural, slot_structural)
                    != std::cmp::Ordering::Greater
                {
                    break;
                }
                unique_slots[insertion] = previous_slot;
                insertion -= 1;
            }
            unique_slots[insertion] = slot;
        }
        if !self.prepare_mandatory_node_batch_growth(unique_count) {
            return resolved;
        }

        for &slot in &unique_slots[..unique_count] {
            let key = packed_keys[slot];
            let [nw, ne, sw, se] = key.children;
            let population = PopulationStat::from_limbs(
                populations.lo[slot],
                populations.hi[slot],
                populations.saturated[slot],
            );
            let node_id = self.push_node(key.level, population, nw, ne, sw, se);
            if self.allocation_failed() {
                return resolved;
            }
            if self.intern.try_insert(key, node_id).is_err() {
                self.reject_allocation(u128::MAX);
                return resolved;
            }
            let lane = lane_map[slot];
            resolved[lane] = Some(node_id);
        }
        for &slot in &unresolved_slots[..unresolved_count] {
            let duplicate = duplicate_sources[slot];
            if duplicate == u8::MAX {
                continue;
            }
            let source_lane = lane_map[usize::from(duplicate)];
            resolved[lane_map[slot]] = resolved[source_lane];
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

    pub(super) fn evaluate_simd_batch(&mut self, batch: &SimdPackedBatch) -> SimdBatchResult {
        if !Self::valid_simd_active_mask(batch.active_lanes, batch.active_mask) {
            crate::invariant_failure!(
                "SIMD batch requires at most {SIMD_BATCH_LANES} contiguous active low lanes: active_lanes={} active_mask={:#010b}",
                batch.active_lanes,
                batch.active_mask
            );
        }
        let (result, accounting) = crate::hashlife::kernels::KernelSet::selected().evaluate(batch);
        self.stats.simd.ready_wave.samples += 1;
        self.stats.simd.ready_wave.total_lanes += batch.active_lanes;
        self.stats.simd.ready_wave.max = self.stats.simd.ready_wave.max.max(batch.active_lanes);
        self.stats.simd.ready_wave.histogram[batch.active_lanes] += 1;
        self.record_kernel_accounting(accounting);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_active_mask_accepts_only_contiguous_low_lanes() {
        for active_lanes in 0..=SIMD_BATCH_LANES {
            let expected = if active_lanes == SIMD_BATCH_LANES {
                u8::MAX
            } else {
                u8::try_from((1_u16 << active_lanes) - 1)
                    .or_invariant("SIMD active mask exceeds u8")
            };
            assert!(
                HashLifeEngine::valid_simd_active_mask(active_lanes, expected),
                "valid mask rejected active_lanes={active_lanes} mask={expected:#010b}"
            );
            for bit in 0..SIMD_BATCH_LANES {
                let malformed = expected ^ (1_u8 << bit);
                assert!(
                    !HashLifeEngine::valid_simd_active_mask(active_lanes, malformed),
                    "noncontiguous or count-mismatched mask accepted active_lanes={active_lanes} mask={malformed:#010b}"
                );
            }
        }
        assert!(
            !HashLifeEngine::valid_simd_active_mask(SIMD_BATCH_LANES + 1, u8::MAX),
            "lane counts above the physical batch width must be rejected"
        );
    }

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

        let result = HashLifeEngine::default().evaluate_simd_batch(&batch);
        let arrays = batch
            .populations
            .map(bytemuck::must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>);
        for (lane, result_lane) in result.lanes[..batch.active_lanes].iter().enumerate() {
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
            assert_eq!(result_lane.output_nonzero_mask, scalar_mask);
        }
    }

    #[test]
    fn overlap_batch_allocation_failure_returns_to_the_transaction_boundary() {
        let mut engine = HashLifeEngine::default();
        let empty = engine.empty(1);
        let single = engine.join(
            engine.live_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
        );
        let left = engine.join(single, empty, empty, empty);
        let right = engine.join(empty, single, empty, empty);
        let node = engine.join(left, right, right, left);
        let packed = engine.node_columns.packed_key(node);
        let identity = CanonicalNodeIdentity {
            packed,
            structural: engine.symmetry_entry(node, Symmetry::Identity).structural,
            symmetry: Symmetry::Identity,
        };
        let fingerprint = identity.structural.fingerprint();
        let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);

        let overlaps =
            engine.probe_and_build_canonical_overlaps_staged(&[identity], &[fingerprint], 1);

        assert!(
            overlaps.is_none(),
            "an overlap batch must not publish placeholder node IDs after allocation failure"
        );
        assert!(
            matches!(
                engine.take_allocation_failure(),
                Some(EngineAllocationFailure::Allocation { .. })
            ),
            "the transaction boundary must receive the typed overlap allocation failure"
        );
    }

    #[test]
    fn mandatory_join_batch_preflights_all_unique_nodes_before_publication() {
        let mut engine = HashLifeEngine::default();
        let empty = engine.empty(1);
        let sparse = engine.join(
            engine.live_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
        );
        let intents = [
            Some(JoinIntent {
                level: 2,
                children: [empty, sparse, empty, sparse],
            }),
            Some(JoinIntent {
                level: 2,
                children: [sparse, empty, sparse, empty],
            }),
        ];
        let nodes_before = engine.node_count();
        engine.id_capacity.node_count = nodes_before + 1;
        engine.begin_allocation_transaction(u128::MAX);

        let resolved = engine.resolve_join_intents_staged(intents);

        assert_eq!(resolved, [None, None]);
        assert_eq!(
            engine.node_count(),
            nodes_before,
            "a rejected mandatory batch must publish no prefix of its unique nodes"
        );
        assert_eq!(
            engine.take_allocation_failure(),
            Some(EngineAllocationFailure::NodeIdExhausted)
        );
    }

    #[test]
    fn mandatory_join_batch_commits_unique_nodes_in_structural_order() {
        fn fixture(engine: &mut HashLifeEngine) -> [Option<JoinIntent>; 2] {
            let empty = engine.empty(1);
            let sparse = engine.join(
                engine.live_leaf,
                engine.dead_leaf,
                engine.dead_leaf,
                engine.dead_leaf,
            );
            [
                Some(JoinIntent {
                    level: 2,
                    children: [empty, sparse, empty, sparse],
                }),
                Some(JoinIntent {
                    level: 2,
                    children: [sparse, empty, sparse, empty],
                }),
            ]
        }

        let mut forward = HashLifeEngine::default();
        let forward_intents = fixture(&mut forward);
        forward.begin_allocation_transaction(u128::MAX);
        let forward_nodes = forward.resolve_join_intents_staged(forward_intents);

        let mut reversed = HashLifeEngine::default();
        let mut reversed_intents = fixture(&mut reversed);
        reversed_intents.reverse();
        reversed.begin_allocation_transaction(u128::MAX);
        let reversed_nodes = reversed.resolve_join_intents_staged(reversed_intents);

        assert_eq!(forward.take_allocation_failure(), None);
        assert_eq!(reversed.take_allocation_failure(), None);
        assert_eq!(
            forward_nodes[0], reversed_nodes[1],
            "first structural node changed identity when input lanes were reversed"
        );
        assert_eq!(
            forward_nodes[1], reversed_nodes[0],
            "second structural node changed identity when input lanes were reversed"
        );
    }
}
