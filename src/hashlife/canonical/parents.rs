use super::*;

impl HashLifeEngine {
    pub(super) fn canonicalize_blocked_jump_node(&mut self, node: NodeId) -> CanonicalNodeProbe {
        self.stats
            .canonical_cache
            .symmetry_gate_canonical_cache_bypasses += 1;
        let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
        self.stats.result_cache.structural_fast_path_lookups += 1;
        if let Some(cached) = self.result_caches.structural_fast_path.get(&node) {
            self.stats.result_cache.structural_fast_path_hits += 1;
            return CanonicalNodeProbe {
                node: cached,
                fingerprint,
                used_cached_fingerprint: true,
            };
        }
        if let Some(cached) = self.result_caches.packed_structural_fast_path.get(&packed) {
            self.stats.result_cache.structural_fast_path_hits += 1;
            self.publish_optional_cache(
                |engine| &engine.result_caches.structural_fast_path,
                |engine| &mut engine.result_caches.structural_fast_path,
                node,
                ProbeKey::fingerprint(&node),
                cached,
            );
            return CanonicalNodeProbe {
                node: cached,
                fingerprint,
                used_cached_fingerprint: true,
            };
        }
        self.stats.result_cache.structural_fast_path_misses += 1;
        let structural = self.symmetry_entry(node, Symmetry::Identity).structural;
        let canonical = CanonicalNodeProbe {
            node: CanonicalNodeIdentity {
                packed,
                structural,
                symmetry: Symmetry::Identity,
            },
            fingerprint,
            used_cached_fingerprint: true,
        };
        self.publish_optional_cache(
            |engine| &engine.result_caches.structural_fast_path,
            |engine| &mut engine.result_caches.structural_fast_path,
            node,
            ProbeKey::fingerprint(&node),
            canonical.node,
        );
        self.publish_optional_cache(
            |engine| &engine.result_caches.packed_structural_fast_path,
            |engine| &mut engine.result_caches.packed_structural_fast_path,
            packed,
            packed.fingerprint(),
            canonical.node,
        );
        canonical
    }

    fn semantic_d4_candidates(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> Option<([CanonicalStructKey; 8], u8)> {
        let shapes = packed
            .children
            .map(|child| self.node_columns.identity_ref(child));
        let input_key = self.canonical_parent_key(packed.level, shapes);
        let stabilizer = self
            .canonical_caches
            .shape_intern
            .get(&input_key)
            .map(|shape| self.canonical_caches.shapes[shape.index()].stabilizer)
            .unwrap_or(1);
        let mut batch = crate::hashlife::kernels::contracts::D4CandidateBatch {
            active_lanes: 0,
            ..Default::default()
        };
        let mut normalized = [Symmetry::Identity; 8];
        let mut lanes = [0; 8];
        let mut requested = 0_u8;
        for (candidate, symmetry) in Symmetry::ALL.into_iter().enumerate() {
            // H fixes the unrotated input key. Normalize the composed action in
            // that same frame, not the relative candidate against a rotated H.
            let effective = base_symmetry.then(symmetry);
            let class = orientations::quotient_representative(effective, stabilizer);
            if let Some(previous) = normalized[..batch.active_lanes]
                .iter()
                .position(|&key| key == class)
            {
                lanes[candidate] = previous;
                continue;
            }
            let lane = batch.active_lanes;
            normalized[lane] = class;
            lanes[candidate] = lane;
            batch.transforms[lane] = symmetry;
            batch.permutations[lane] = effective
                .quadrant_perm()
                .map(|index| u8::try_from(index).or_invariant("D4 quadrant index exceeds u8"));
            requested |= 1 << (effective as u8);
            batch.active_lanes += 1;
        }
        let mut records = [orientations::OrientationRecord::default(); 4];
        // Child IDs denote exact local shapes, not orbit representatives with
        // hidden orientations. The effective parent action acts identically
        // inside each child; quadrant relocation is applied by the kernel.
        for source in 0..4 {
            if let Some(previous) = shapes[..source]
                .iter()
                .position(|&shape| shape == shapes[source])
            {
                records[source] = records[previous];
                self.stats.transform.orientation_requests += requested.count_ones() as usize;
                self.stats.transform.orientation_quotient_eliminations +=
                    requested.count_ones() as usize;
            } else {
                records[source] = self.resolve_shape_orientations(shapes[source], requested)?;
            }
        }
        for lane in 0..batch.active_lanes {
            let effective = base_symmetry.then(batch.transforms[lane]);
            batch.oriented_children[lane] = records.map(|record| record.reference(effective).raw());
        }
        // Unknown equivalences require constructing their signatures once.
        // Do not precompute scalar permutations and then repeat them in SIMD.
        let (constructed, accounting) =
            crate::hashlife::kernels::KernelSet::selected().construct_d4_candidates(&batch);
        self.record_kernel_accounting(accounting);
        debug_assert_eq!(
            constructed,
            crate::hashlife::kernels::contracts::scalar_d4_candidates(&batch),
            "native candidates must preserve exact recursive orientation semantics"
        );
        let mut keys = [CanonicalStructKey::leaf(false); 8];
        let mut class_lanes = [0; 8];
        for lane in 0..batch.active_lanes {
            if let Some(previous) = constructed.children[..lane]
                .iter()
                .position(|&key| key == constructed.children[lane])
            {
                class_lanes[lane] = class_lanes[previous];
            } else {
                class_lanes[lane] = lane;
                keys[lane] = self.canonical_parent_key(
                    packed.level,
                    constructed.children[lane].map(CanonicalShapeId::from_raw),
                );
            }
        }
        let classes = lanes.map(|lane| class_lanes[lane]);
        let representatives = classes
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (index, class)| {
                if classes[..index].contains(class) {
                    mask
                } else {
                    mask | (1 << index)
                }
            });
        let unique = representatives.count_ones() as usize;
        self.stats.transform.d4_candidate_requests += Symmetry::ALL.len();
        self.stats.transform.d4_duplicate_candidates += Symmetry::ALL.len() - unique;
        self.stats.transform.d4_unique_candidates += unique;
        Some((classes.map(|lane| keys[lane]), representatives))
    }

    fn exact_candidate_mask(
        &self,
        candidates: &[CanonicalStructKey; 8],
        expected: CanonicalStructKey,
    ) -> u8 {
        // Interned child classes prove exact equality within the live registry.
        // Fingerprints accelerate lookup but never define the quotient or order.
        candidates
            .iter()
            .enumerate()
            .fold(0, |mask, (index, candidate)| {
                if candidate.level == expected.level && candidate.children == expected.children {
                    mask | (1 << index)
                } else {
                    mask
                }
            })
    }

    pub(in crate::hashlife) fn scan_canonical_transform_winner(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (Symmetry, PackedTransformOrderEntry, [CanonicalStructKey; 8]) {
        if record_miss {
            self.stats.transform.packed_d4_canonicalization_misses += 1;
        }

        self.stats.transform.d4_semantic_prefix_attempts += 1;
        let Some((candidates, representatives)) =
            self.semantic_d4_candidates(packed, base_symmetry)
        else {
            let structural = self.canonical_parent_key(
                packed.level,
                packed
                    .children
                    .map(|child| self.node_columns.identity_ref(child)),
            );
            return (
                Symmetry::Identity,
                PackedTransformOrderEntry {
                    structural,
                    aliases: 1,
                    stabilizer: 1,
                },
                [structural; 8],
            );
        };
        let mut prefix_batch = crate::hashlife::kernels::contracts::D4PrefixBatch {
            active_lanes: 0,
            ..Default::default()
        };
        let mut all_complete = true;
        for (index, candidate) in candidates.iter().copied().enumerate() {
            if representatives & (1 << index) == 0 {
                continue;
            }
            let prefix = self.canonical_key_prefix(candidate);
            let lane = prefix_batch.active_lanes;
            prefix_batch.transforms[lane] = Symmetry::ALL[index];
            prefix_batch.words[0][lane] = prefix.words[0];
            prefix_batch.words[1][lane] = prefix.words[1];
            all_complete &= prefix.complete;
            prefix_batch.active_lanes += 1;
        }
        prefix_batch.complete = all_complete;
        let (prefix_decision, accounting) =
            crate::hashlife::kernels::KernelSet::selected().compare_d4_prefixes(&prefix_batch);
        self.record_kernel_accounting(accounting);

        let mut winner_index = Symmetry::ALL
            .into_iter()
            .position(|symmetry| symmetry == prefix_decision.transform)
            .unwrap_or_default();
        if !prefix_decision.exact {
            for candidate_index in 0..8 {
                let candidate_bit = 1_u8 << candidate_index;
                if prefix_decision.unresolved_mask & candidate_bit == 0
                    || candidate_index == winner_index
                {
                    continue;
                }
                let ordering = self
                    .compare_canonical_keys(candidates[candidate_index], candidates[winner_index]);
                self.stats.transform.d4_exact_comparator_calls += 1;
                if ordering == std::cmp::Ordering::Less
                    || (ordering == std::cmp::Ordering::Equal && candidate_index < winner_index)
                {
                    winner_index = candidate_index;
                }
            }
        }
        let winner = candidates[winner_index];
        let aliases = self.exact_candidate_mask(&candidates, winner);
        winner_index =
            usize::try_from(aliases.trailing_zeros()).or_invariant("D4 alias index exceeds usize");
        let canonical_symmetry = Symmetry::ALL[winner_index];
        let mut stabilizer = 0_u8;
        for (index, symmetry) in Symmetry::ALL.into_iter().enumerate() {
            // `then` is the tested left-to-right action: transform the original
            // by the winner first, then apply this canonical automorphism.
            let composed = canonical_symmetry.then(symmetry);
            if aliases & (1 << (composed as u8)) != 0 {
                stabilizer |= 1 << index;
            }
        }
        let canonical_ref = self.intern_canonical_shape(winner);
        if !self.allocation_failed() {
            self.canonical_caches.shapes[canonical_ref.index()].stabilizer = stabilizer;
        }
        let canonical_entry = PackedTransformOrderEntry {
            structural: winner,
            aliases,
            stabilizer,
        };
        (canonical_symmetry, canonical_entry, candidates)
    }

    pub(super) fn canonical_transform_winner_fallback(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (PackedNodeKey, Symmetry, CanonicalStructKey) {
        self.stats.canonical_cache.direct_parent_winner_fallbacks += 1;
        let (canonical_symmetry, canonical_entry, candidates) =
            self.scan_canonical_transform_winner(packed, base_symmetry, record_miss);
        if self.allocation_failed() {
            return (packed, Symmetry::Identity, canonical_entry.structural);
        }
        debug_assert_ne!(canonical_entry.aliases, 0);
        debug_assert_ne!(canonical_entry.stabilizer, 0);
        let original_alias = 1 << (base_symmetry.inverse() as u8);
        let canonical_packed = if canonical_entry.aliases & original_alias != 0 {
            packed
        } else {
            self.stats
                .canonical_fallback
                .canonical_transform_root_reconstructions += 1;
            let canonical_id =
                self.transform_packed_node_key(packed, base_symmetry.then(canonical_symmetry));
            self.packed_root_from_transform_id(canonical_id)
        };
        if !self.allocation_failed() {
            self.cache_canonical_orbit(&candidates, canonical_entry, canonical_packed);
        }
        (
            canonical_packed,
            canonical_symmetry,
            canonical_entry.structural,
        )
    }

    fn cache_canonical_orbit(
        &mut self,
        candidates: &[CanonicalStructKey; 8],
        winner: PackedTransformOrderEntry,
        packed: PackedNodeKey,
    ) {
        for (index, candidate) in candidates.iter().copied().enumerate() {
            // Exact interned child identities deduplicate automorphic orientations.
            if candidates[..index].contains(&candidate) {
                continue;
            }
            // Reuse the proven orbit to teach existing noncanonical shapes their
            // own automorphisms too. Do not intern otherwise unused parent shapes.
            if let Some(shape) = self.canonical_caches.shape_intern.get(&candidate) {
                let stabilizer = Symmetry::ALL.into_iter().fold(0, |mask, symmetry| {
                    let transformed = Symmetry::ALL[index].then(symmetry) as usize;
                    if candidates[transformed] == candidate {
                        mask | (1 << (symmetry as u8))
                    } else {
                        mask
                    }
                });
                self.canonical_caches.shapes[shape.index()].stabilizer = stabilizer;
            }
            // Find the lowest transform from this oriented input to the winner.
            // `then` applies the input orientation first, then the relative transform.
            let symmetry = Symmetry::ALL
                .into_iter()
                .find(|relative| {
                    winner.aliases & (1 << (Symmetry::ALL[index].then(*relative) as u8)) != 0
                })
                .or_invariant("every D4 orbit member must reach its canonical winner");
            let canonical = CanonicalNodeIdentity {
                packed,
                structural: winner.structural,
                symmetry,
            };
            if !self.publish_optional_cache(
                |engine| &engine.canonical_caches.direct_parent,
                |engine| &mut engine.canonical_caches.direct_parent,
                candidate,
                candidate.fingerprint(),
                canonical,
            ) {
                // Orbit backfilling is optional: pressure must not fail the result
                // or repeatedly attempt the same unavailable table growth.
                break;
            }
        }
    }

    #[inline]
    pub(super) fn direct_parent_cache_key(
        &self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
    ) -> CanonicalStructKey {
        self.canonical_parent_key(
            level,
            base_symmetry
                .quadrant_perm()
                .map(|index| input_children[index]),
        )
    }

    pub(super) fn direct_parent_input_children(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> [CanonicalNodeRef; 4] {
        let shapes = packed
            .children
            .map(|child| self.node_columns.identity_ref(child));
        if base_symmetry == Symmetry::Identity {
            return shapes;
        }
        let mut oriented = [CanonicalShapeId::DEAD; 4];
        for source in 0..4 {
            if let Some(previous) = shapes[..source]
                .iter()
                .position(|&shape| shape == shapes[source])
            {
                oriented[source] = oriented[previous];
            } else {
                let Some(record) =
                    self.resolve_shape_orientations(shapes[source], 1 << (base_symmetry as u8))
                else {
                    return oriented;
                };
                oriented[source] = record.reference(base_symmetry);
            }
        }
        oriented
    }

    pub(super) fn lookup_direct_parent_identity(
        &mut self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
    ) -> Option<CanonicalNodeIdentity> {
        if self.allocation_failed() {
            return None;
        }
        self.stats.canonical_cache.direct_parent_winner_lookups += 1;
        let cache_key = self.direct_parent_cache_key(level, base_symmetry, input_children);
        if let Some(cached) = self.canonical_caches.hot_direct_parent.get(&cache_key) {
            self.stats.canonical_cache.direct_parent_winner_hits += 1;
            return Some(cached);
        }
        if let Some(cached) = self.canonical_caches.direct_parent.get(&cache_key) {
            self.stats.canonical_cache.direct_parent_winner_hits += 1;
            self.maybe_promote_hot_direct_parent_identity(cache_key, cached);
            return Some(cached);
        }
        self.stats.canonical_cache.direct_parent_winner_misses += 1;
        None
    }

    #[inline]
    pub(super) fn backfill_direct_parent_identity(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
        canonical: CanonicalNodeIdentity,
    ) {
        if packed.level == 0 {
            return;
        }
        let cache_key = self.direct_parent_cache_key(packed.level, base_symmetry, input_children);
        if self
            .canonical_caches
            .hot_direct_parent
            .get(&cache_key)
            .is_some()
            || self
                .canonical_caches
                .direct_parent
                .get(&cache_key)
                .is_some()
        {
            return;
        }
        self.publish_optional_cache(
            |engine| &engine.canonical_caches.direct_parent,
            |engine| &mut engine.canonical_caches.direct_parent,
            cache_key,
            cache_key.fingerprint(),
            canonical,
        );
        self.maybe_promote_hot_direct_parent_identity(cache_key, canonical);
    }

    #[inline]
    pub(super) fn maybe_promote_hot_direct_parent_identity(
        &mut self,
        cache_key: CanonicalStructKey,
        canonical: CanonicalNodeIdentity,
    ) {
        if self.canonical_caches.hot_direct_parent_budget == 0 {
            self.rebalance_hot_canonical_budgets();
        }
        if self.canonical_caches.hot_direct_parent.len()
            < self.canonical_caches.hot_direct_parent_budget
        {
            self.publish_optional_cache(
                |engine| &engine.canonical_caches.hot_direct_parent,
                |engine| &mut engine.canonical_caches.hot_direct_parent,
                cache_key,
                cache_key.fingerprint(),
                canonical,
            );
        }
    }

    #[inline]
    pub(super) fn lookup_canonical_packed_identity(
        &mut self,
        packed: PackedNodeKey,
    ) -> Option<CanonicalNodeIdentity> {
        self.stats.canonical_cache.canonical_packed_cache_lookups += 1;
        if let Some(cached) = self.canonical_caches.hot_packed.get(&packed) {
            self.stats.canonical_cache.canonical_packed_cache_hits += 1;
            return Some(cached);
        }
        if let Some(cached) = self.canonical_caches.packed.get(&packed) {
            self.stats.canonical_cache.canonical_packed_cache_hits += 1;
            self.maybe_promote_hot_canonical_packed_identity(packed, cached);
            return Some(cached);
        }
        self.stats.canonical_cache.canonical_packed_cache_misses += 1;
        None
    }

    #[inline]
    pub(super) fn cache_canonical_packed_identity(
        &mut self,
        packed: PackedNodeKey,
        canonical: CanonicalNodeIdentity,
    ) {
        self.publish_optional_cache(
            |engine| &engine.canonical_caches.packed,
            |engine| &mut engine.canonical_caches.packed,
            packed,
            packed.fingerprint(),
            canonical,
        );
    }

    #[inline]
    pub(super) fn maybe_promote_hot_canonical_packed_identity(
        &mut self,
        packed: PackedNodeKey,
        canonical: CanonicalNodeIdentity,
    ) {
        if self.canonical_caches.hot_packed_budget == 0 {
            self.rebalance_hot_canonical_budgets();
        }
        if self.canonical_caches.hot_packed.len() < self.canonical_caches.hot_packed_budget {
            self.publish_optional_cache(
                |engine| &engine.canonical_caches.hot_packed,
                |engine| &mut engine.canonical_caches.hot_packed,
                packed,
                packed.fingerprint(),
                canonical,
            );
        }
    }

    #[inline]
    pub(super) fn lookup_canonical_oriented_identity(
        &mut self,
        cache_key: PackedSymmetryKey,
    ) -> Option<CanonicalNodeIdentity> {
        self.stats.canonical_cache.canonical_oriented_cache_lookups += 1;
        if let Some(cached) = self.canonical_caches.hot_oriented.get(&cache_key) {
            self.stats.canonical_cache.canonical_oriented_cache_hits += 1;
            return Some(cached);
        }
        if let Some(cached) = self.canonical_caches.oriented.get(&cache_key) {
            self.stats.canonical_cache.canonical_oriented_cache_hits += 1;
            self.maybe_promote_hot_canonical_oriented_identity(cache_key, cached);
            return Some(cached);
        }
        self.stats.canonical_cache.canonical_oriented_cache_misses += 1;
        None
    }

    #[inline]
    pub(super) fn cache_canonical_oriented_identity(
        &mut self,
        cache_key: PackedSymmetryKey,
        canonical: CanonicalNodeIdentity,
    ) {
        self.publish_optional_cache(
            |engine| &engine.canonical_caches.oriented,
            |engine| &mut engine.canonical_caches.oriented,
            cache_key,
            cache_key.fingerprint(),
            canonical,
        );
    }

    #[inline]
    pub(super) fn maybe_promote_hot_canonical_oriented_identity(
        &mut self,
        cache_key: PackedSymmetryKey,
        canonical: CanonicalNodeIdentity,
    ) {
        if self.canonical_caches.hot_oriented_budget == 0 {
            self.rebalance_hot_canonical_budgets();
        }
        if self.canonical_caches.hot_oriented.len() < self.canonical_caches.hot_oriented_budget {
            self.publish_optional_cache(
                |engine| &engine.canonical_caches.hot_oriented,
                |engine| &mut engine.canonical_caches.hot_oriented,
                cache_key,
                cache_key.fingerprint(),
                canonical,
            );
        }
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(in crate::hashlife) fn canonicalize_blocked_jump_node_for_tests(
        &mut self,
        node: NodeId,
    ) -> (u64, u64, u8) {
        let canonical = self.canonicalize_blocked_jump_node(node).node;
        (
            canonical.packed.fingerprint(),
            canonical.structural.fingerprint(),
            canonical.symmetry as u8,
        )
    }

    pub(in crate::hashlife) fn direct_parent_winner_for_tests(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> Option<CanonicalNodeIdentity> {
        if packed.level == 0 {
            return Some(CanonicalNodeIdentity {
                packed,
                structural: CanonicalStructKey::leaf(packed.children[0] != NodeId::ZERO),
                symmetry: Symmetry::Identity,
            });
        }
        let _ = self.canonicalize_packed_direct(packed, base_symmetry, false);
        let input_children = self.direct_parent_input_children(packed, base_symmetry);
        self.lookup_direct_parent_identity(packed.level, base_symmetry, input_children)
    }

    pub(in crate::hashlife) fn canonicalize_packed_direct_for_tests(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> (u64, u64, u8) {
        let canonical = self
            .canonicalize_packed_direct(packed, base_symmetry, false)
            .node;
        (
            canonical.packed.fingerprint(),
            canonical.structural.fingerprint(),
            canonical.symmetry as u8,
        )
    }
}

#[cfg(test)]
mod quotient_collision_tests {
    use super::*;

    #[test]
    fn colliding_fingerprints_do_not_merge_distinct_candidate_classes() {
        let engine = HashLifeEngine::default();
        let left = engine.canonical_parent_key(1, [CanonicalShapeId::DEAD; 4]);
        let mut right = engine.canonical_parent_key(1, [CanonicalShapeId::LIVE; 4]);
        // Inject a probe collision without changing either exact child tuple.
        right.fingerprint = left.fingerprint;
        let mut candidates = [right; 8];
        candidates[5] = left;
        assert_eq!(engine.exact_candidate_mask(&candidates, left), 1 << 5);
        assert_eq!(
            engine.exact_candidate_mask(&candidates, right),
            u8::MAX ^ (1 << 5)
        );
        assert_eq!(
            engine.compare_canonical_keys(left, right),
            std::cmp::Ordering::Less
        );
    }
}
