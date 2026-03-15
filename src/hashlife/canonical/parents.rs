use super::*;

impl HashLifeEngine {
    fn packed_semantic_prefix(
        &self,
        root: PackedNodeKey,
        symmetry: Symmetry,
        root_children: [NodeId; 4],
    ) -> ([u64; 2], bool, usize) {
        const PREFIX_BITS: usize = 128;
        const MAX_PREFIX_STACK: usize = 256;

        let mut words = [0_u64; 2];
        let mut stack = [root; MAX_PREFIX_STACK];
        let mut stack_len = 0;
        let mut leaves = 0;
        let permutation = symmetry.quadrant_perm();
        if root.level == 0 {
            stack[0] = root;
            stack_len = 1;
        } else {
            for child in root_children.into_iter().rev() {
                stack[stack_len] = self.node_columns.packed_key(child);
                stack_len += 1;
            }
        }
        while stack_len != 0 && leaves < PREFIX_BITS {
            stack_len -= 1;
            let current = stack[stack_len];
            if current.level == 0 {
                if current.children[0] != 0 {
                    words[leaves / 64] |= 1_u64 << (63 - leaves % 64);
                }
                leaves += 1;
                continue;
            }
            if stack_len + 4 > stack.len() {
                return (words, false, leaves);
            }
            for output_quadrant in (0..4).rev() {
                let child = current.children[permutation[output_quadrant]];
                stack[stack_len] = self.node_columns.packed_key(child);
                stack_len += 1;
            }
        }
        (words, stack_len == 0, leaves)
    }

    fn cached_d4_transform_ids(
        &self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> Option<[PackedTransformId; 8]> {
        let mut ids = [0; 8];
        for (index, symmetry) in Symmetry::ALL.into_iter().enumerate() {
            ids[index] = self
                .transform_state
                .canonical_cache
                .get(&PackedSymmetryKey {
                    packed,
                    symmetry: base_symmetry.then(symmetry),
                })?;
        }
        Some(ids)
    }

    fn exact_d4_winner_from_ids(&mut self, ids: [PackedTransformId; 8]) -> usize {
        let mut winner = 0;
        for candidate in 1..8 {
            if self.compare_packed_transform_ids(ids[candidate], ids[winner])
                == std::cmp::Ordering::Less
            {
                winner = candidate;
            }
        }
        winner
    }

    fn materialize_d4_transform_ids(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> [PackedTransformId; 8] {
        let mut ids = [0; 8];
        for (index, symmetry) in Symmetry::ALL.into_iter().enumerate() {
            ids[index] = self.transform_packed_node_key(packed, base_symmetry.then(symmetry));
        }
        ids
    }

    fn should_use_native_d4_prefix(&self, packed: PackedNodeKey) -> bool {
        if !crate::hashlife::kernels::KernelSet::selected().supports_native_d4_prefix() {
            return false;
        }
        if packed.level <= 3 {
            return true;
        }
        let population = PopulationStat::sum(
            packed
                .children
                .map(|child| self.node_columns.population_stat(child)),
        );
        population.value() >= 16
    }

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
                crate::flat_table::FlatKey::fingerprint(&node),
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
            crate::flat_table::FlatKey::fingerprint(&node),
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

    pub(in crate::hashlife) fn scan_canonical_transform_winner(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (Symmetry, PackedTransformOrderEntry) {
        if record_miss {
            self.stats.transform.packed_d4_canonicalization_misses += 1;
        }

        if let Some(cached_ids) = self.cached_d4_transform_ids(packed, base_symmetry) {
            self.stats.transform.d4_semantic_prefix_cache_bypasses += 1;
            let winner_index = self.exact_d4_winner_from_ids(cached_ids);
            let canonical_symmetry = Symmetry::ALL[winner_index];
            let canonical_entry =
                self.transformed_order_entry(packed, base_symmetry.then(canonical_symmetry));
            return (canonical_symmetry, canonical_entry);
        }

        if !self.should_use_native_d4_prefix(packed) {
            self.stats.transform.d4_semantic_prefix_cost_bypasses += 1;
            let ids = self.materialize_d4_transform_ids(packed, base_symmetry);
            let winner_index = self.exact_d4_winner_from_ids(ids);
            let canonical_symmetry = Symmetry::ALL[winner_index];
            let canonical_entry =
                self.transformed_order_entry(packed, base_symmetry.then(canonical_symmetry));
            return (canonical_symmetry, canonical_entry);
        }

        self.stats.transform.d4_semantic_prefix_attempts += 1;
        let base_children = base_symmetry
            .quadrant_perm()
            .map(|index| packed.children[index]);
        let (candidate_batch, candidate_accounting) =
            crate::hashlife::kernels::KernelSet::selected().construct_d4_candidates(
                &crate::hashlife::kernels::contracts::D4CandidateBatch {
                    children: base_children,
                },
            );
        self.record_kernel_accounting(candidate_accounting);
        let mut prefix_batch = crate::hashlife::kernels::contracts::D4PrefixBatch::default();
        let mut all_complete = true;
        for (index, symmetry) in Symmetry::ALL.into_iter().enumerate() {
            let candidate_symmetry = base_symmetry.then(symmetry);
            debug_assert_eq!(
                candidate_batch.children[index],
                candidate_symmetry
                    .quadrant_perm()
                    .map(|slot| packed.children[slot]),
                "native D4 candidate construction must match structural geometry"
            );
            let (prefix, complete, leaf_visits) = self.packed_semantic_prefix(
                packed,
                candidate_symmetry,
                candidate_batch.children[index],
            );
            prefix_batch.words[0][index] = prefix[0];
            prefix_batch.words[1][index] = prefix[1];
            all_complete &= complete;
            self.stats.transform.d4_semantic_prefix_leaf_visits += leaf_visits;
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
            let mut winner_id = self
                .transform_packed_node_key(packed, base_symmetry.then(Symmetry::ALL[winner_index]));
            let mut compared_mask = 1_u8 << winner_index;
            for candidate_index in 0..8 {
                let candidate_bit = 1_u8 << candidate_index;
                if prefix_decision.unresolved_mask & candidate_bit == 0
                    || compared_mask & candidate_bit != 0
                {
                    continue;
                }
                let candidate_id = self.transform_packed_node_key(
                    packed,
                    base_symmetry.then(Symmetry::ALL[candidate_index]),
                );
                let ordering = self.compare_packed_transform_ids(candidate_id, winner_id);
                if ordering == std::cmp::Ordering::Less
                    || (ordering == std::cmp::Ordering::Equal && candidate_index < winner_index)
                {
                    winner_index = candidate_index;
                    winner_id = candidate_id;
                }
                compared_mask |= candidate_bit;
            }
        }
        let canonical_symmetry = Symmetry::ALL[winner_index];
        let canonical_entry =
            self.transformed_order_entry(packed, base_symmetry.then(canonical_symmetry));
        (canonical_symmetry, canonical_entry)
    }

    pub(super) fn canonical_transform_winner_fallback(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (PackedTransformId, Symmetry, CanonicalStructKey) {
        self.stats.canonical_cache.direct_parent_winner_fallbacks += 1;
        let (canonical_symmetry, canonical_entry) =
            self.scan_canonical_transform_winner(packed, base_symmetry, record_miss);
        let canonical_id =
            self.transform_packed_node_key(packed, base_symmetry.then(canonical_symmetry));
        (canonical_id, canonical_symmetry, canonical_entry.structural)
    }

    #[inline]
    pub(super) fn direct_parent_cache_key(
        &self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
    ) -> DirectCanonicalParentKey {
        DirectCanonicalParentKey {
            level,
            symmetry: base_symmetry,
            children: input_children,
        }
    }

    pub(super) fn direct_parent_input_children(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> [CanonicalNodeRef; 4] {
        let child_keys = packed.children;
        child_keys.map(|child| self.symmetry_canonical_ref(child, base_symmetry))
    }

    pub(super) fn lookup_direct_parent_identity(
        &mut self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
    ) -> Option<CanonicalNodeIdentity> {
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

    pub(super) fn cache_direct_parent_identity(
        &mut self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
        canonical: CanonicalNodeIdentity,
    ) {
        let cache_key = self.direct_parent_cache_key(level, base_symmetry, input_children);
        self.publish_optional_cache(
            |engine| &engine.canonical_caches.direct_parent,
            |engine| &mut engine.canonical_caches.direct_parent,
            cache_key,
            cache_key.fingerprint(),
            canonical,
        );
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
        cache_key: DirectCanonicalParentKey,
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
    pub(crate) fn canonicalize_blocked_jump_node_for_tests(
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
                structural: CanonicalStructKey::leaf(packed.children[0] != 0),
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
