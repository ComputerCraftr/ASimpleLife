use super::*;

mod results;

impl HashLifeEngine {
    pub(super) fn reset_packed_transform_state(&mut self) {
        self.canonical_transform_cache.clear();
        self.oriented_result_cache.clear();
        self.packed_transform_compare_cache.clear();
        self.packed_transform_intern.clear();
        self.packed_transform_nodes.clear();
        self.packed_transform_materialized.clear();
        self.packed_transform_packed_roots.clear();

        for alive in [false, true] {
            let leaf_id = self.packed_transform_nodes.len() as PackedTransformId;
            let population = u64::from(alive);
            let structural = CanonicalStructKey::leaf(alive);
            self.packed_transform_nodes.push(PackedTransformNode {
                level: 0,
                leaf_population: population,
                children: [0; 4],
                structural,
                canonical_ref: canonical_node_ref_from_structural(structural),
                order_fingerprint: structural.fingerprint(),
                order_children: structural.children,
            });
            self.packed_transform_materialized.push(Some(if alive {
                self.live_leaf
            } else {
                self.dead_leaf
            }));
            self.packed_transform_packed_roots
                .push(Some(PackedNodeKey::new(0, [population, 0, 0, 0])));
            self.packed_transform_intern.insert(
                PackedTransformShapeKey {
                    level: 0,
                    children: [population, 0, 0, 0],
                },
                leaf_id,
            );
        }
    }

    #[inline]
    fn leaf_transform_id(&self, alive: bool) -> PackedTransformId {
        u32::from(alive)
    }

    fn intern_packed_transform_node(
        &mut self,
        level: u32,
        children: [PackedTransformId; 4],
    ) -> PackedTransformId {
        let shape = PackedTransformShapeKey {
            level,
            children: children.map(u64::from),
        };
        if let Some(existing) = self.packed_transform_intern.get(&shape) {
            return existing;
        }
        let id = self.packed_transform_nodes.len() as PackedTransformId;
        let structural = CanonicalStructKey::new(
            level,
            children.map(|child| self.packed_transform_nodes[child as usize].canonical_ref),
        );
        let order_children =
            children.map(|child| self.packed_transform_nodes[child as usize].canonical_ref);
        self.packed_transform_nodes.push(PackedTransformNode {
            level,
            leaf_population: 0,
            children,
            structural,
            canonical_ref: canonical_node_ref_from_structural(structural),
            order_fingerprint: structural.fingerprint(),
            order_children,
        });
        self.packed_transform_materialized.push(None);
        self.packed_transform_packed_roots.push(None);
        self.packed_transform_intern.insert(shape, id);
        id
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn structural_key_from_transform_id(
        &self,
        id: PackedTransformId,
    ) -> CanonicalStructKey {
        self.packed_transform_nodes[id as usize].structural
    }

    pub(super) fn transform_packed_node_key(
        &mut self,
        packed: PackedNodeKey,
        symmetry: Symmetry,
    ) -> PackedTransformId {
        if packed.level == 0 {
            return self.leaf_transform_id(packed.children[0] != 0);
        }
        let root_key = PackedSymmetryKey { packed, symmetry };
        if let Some(transformed) = self.canonical_transform_cache.get(&root_key) {
            self.stats.packed_recursive_transform_hits += 1;
            return transformed;
        }
        self.stats.packed_recursive_transform_misses += 1;
        let mut stack = Vec::with_capacity((packed.level as usize).saturating_mul(4).max(8));
        stack.push((packed, false));
        while let Some((current, ready)) = stack.pop() {
            if current.level == 0 {
                continue;
            }
            let current_key = PackedSymmetryKey {
                packed: current,
                symmetry,
            };
            if self.canonical_transform_cache.get(&current_key).is_some() {
                continue;
            }
            if !ready {
                stack.push((current, true));
                for child in current.children {
                    let child_key = self.node_columns.packed_key(child);
                    if child_key.level != 0
                        && self
                            .canonical_transform_cache
                            .get(&PackedSymmetryKey {
                                packed: child_key,
                                symmetry,
                            })
                            .is_none()
                    {
                        stack.push((child_key, false));
                    }
                }
                continue;
            }
            let child_keys = current
                .children
                .map(|child| self.node_columns.packed_key(child));
            let child_ids = child_keys.map(|child| {
                if child.level == 0 {
                    self.leaf_transform_id(child.children[0] != 0)
                } else {
                    self.canonical_transform_cache
                        .get(&PackedSymmetryKey {
                            packed: child,
                            symmetry,
                        })
                        .expect("iterative transform children must be cached before parent")
                }
            });
            let perm = symmetry.quadrant_perm();
            let transformed = self.intern_packed_transform_node(
                current.level,
                [
                    child_ids[perm[0]],
                    child_ids[perm[1]],
                    child_ids[perm[2]],
                    child_ids[perm[3]],
                ],
            );
            self.canonical_transform_cache
                .insert(current_key, transformed);
        }
        self.canonical_transform_cache
            .get(&root_key)
            .expect("iterative transform root must be cached")
    }

    fn compare_packed_transform_ids(
        &mut self,
        left: PackedTransformId,
        right: PackedTransformId,
    ) -> std::cmp::Ordering {
        if left == right {
            return std::cmp::Ordering::Equal;
        }
        let (key, invert) = if left < right {
            (PackedTransformCompareKey { left, right }, false)
        } else {
            (
                PackedTransformCompareKey {
                    left: right,
                    right: left,
                },
                true,
            )
        };
        if let Some(cached) = self.packed_transform_compare_cache.get(&key) {
            return match (cached, invert) {
                (0, _) => std::cmp::Ordering::Equal,
                (-1, false) | (1, true) => std::cmp::Ordering::Less,
                (1, false) | (-1, true) => std::cmp::Ordering::Greater,
                _ => unreachable!("invalid packed transform compare cache entry"),
            };
        }
        let mut ordering = std::cmp::Ordering::Equal;
        let mut stack = vec![(key.left, key.right, 0usize)];
        while let Some((current_left, current_right, child_index)) = stack.pop() {
            if current_left == current_right {
                continue;
            }
            let left_node = self.packed_transform_nodes[current_left as usize];
            let right_node = self.packed_transform_nodes[current_right as usize];
            let node_ordering = left_node.level.cmp(&right_node.level).then_with(|| {
                if left_node.level == 0 {
                    left_node.leaf_population.cmp(&right_node.leaf_population)
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            if node_ordering != std::cmp::Ordering::Equal {
                ordering = node_ordering;
                break;
            }
            if left_node.level == 0 {
                continue;
            }
            if child_index < 4 {
                stack.push((current_left, current_right, child_index + 1));
                let next_left = left_node.children[child_index];
                let next_right = right_node.children[child_index];
                if next_left != next_right {
                    stack.push((next_left, next_right, 0));
                }
            }
        }
        let cached_value = match ordering {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        };
        self.packed_transform_compare_cache
            .insert(key, cached_value);
        ordering
    }

    fn transformed_order_entry(
        &mut self,
        packed: PackedNodeKey,
        symmetry: Symmetry,
    ) -> PackedTransformOrderEntry {
        if let Some(existing) = self.intern.get(&packed) {
            return self.node_columns.symmetry_entry(existing, symmetry);
        }
        if packed.level == 0 {
            let structural = CanonicalStructKey::leaf(packed.children[0] != 0);
            return PackedTransformOrderEntry {
                structural,
                canonical_ref: canonical_node_ref_from_structural(structural),
                order_fingerprint: structural.fingerprint(),
                order_children: structural.children,
            };
        }
        let child_entries = packed
            .children
            .map(|child| self.node_columns.symmetry_entry(child, symmetry));
        let perm = symmetry.quadrant_perm();
        let transformed_children = [
            child_entries[perm[0]],
            child_entries[perm[1]],
            child_entries[perm[2]],
            child_entries[perm[3]],
        ];
        let order_children = transformed_children.map(|child| child.canonical_ref);
        let structural = CanonicalStructKey::new(packed.level, order_children);
        PackedTransformOrderEntry {
            structural,
            canonical_ref: canonical_node_ref_from_structural(structural),
            order_fingerprint: structural.fingerprint(),
            order_children,
        }
    }

    #[inline]
    fn compare_transform_order_entries(
        &mut self,
        left: PackedNodeKey,
        left_symmetry: Symmetry,
        left_entry: PackedTransformOrderEntry,
        right: PackedNodeKey,
        right_symmetry: Symmetry,
        right_entry: PackedTransformOrderEntry,
    ) -> std::cmp::Ordering {
        let ordering = left_entry
            .structural
            .level
            .cmp(&right_entry.structural.level)
            .then_with(|| {
                left_entry
                    .order_fingerprint
                    .cmp(&right_entry.order_fingerprint)
            })
            .then_with(|| left_entry.order_children.cmp(&right_entry.order_children));
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }

        let left_id = self.transform_packed_node_key(left, left_symmetry);
        let right_id = self.transform_packed_node_key(right, right_symmetry);
        self.compare_packed_transform_ids(left_id, right_id)
    }

    #[cfg(test)]
    fn materialize_packed_transform_node_internal(&mut self, id: PackedTransformId) -> NodeId {
        if let Some(node) = self.packed_transform_materialized[id as usize] {
            return node;
        }
        let mut stack = Vec::with_capacity(16);
        stack.push((id, false));
        while let Some((current, ready)) = stack.pop() {
            if self.packed_transform_materialized[current as usize].is_some() {
                continue;
            }
            let transform_node = self.packed_transform_nodes[current as usize];
            debug_assert!(transform_node.level != 0);
            if !ready {
                stack.push((current, true));
                for child in transform_node.children {
                    if self.packed_transform_materialized[child as usize].is_none() {
                        if self.packed_transform_nodes[child as usize].level == 0 {
                            let alive =
                                self.packed_transform_nodes[child as usize].leaf_population != 0;
                            self.packed_transform_materialized[child as usize] = Some(if alive {
                                self.live_leaf
                            } else {
                                self.dead_leaf
                            });
                        } else {
                            stack.push((child, false));
                        }
                    }
                }
                continue;
            }
            let children = transform_node.children.map(|child| {
                self.packed_transform_materialized[child as usize]
                    .expect("iterative transform materialization must resolve child before parent")
            });
            let node = self.join(children[0], children[1], children[2], children[3]);
            self.packed_transform_materialized[current as usize] = Some(node);
        }
        self.packed_transform_materialized[id as usize]
            .expect("iterative transform materialization must resolve target")
    }

    fn packed_root_from_transform_id(&mut self, id: PackedTransformId) -> PackedNodeKey {
        if let Some(packed) = self.packed_transform_packed_roots[id as usize] {
            return packed;
        }
        let mut stack = Vec::with_capacity(16);
        stack.push((id, false));
        while let Some((current, ready)) = stack.pop() {
            if self.packed_transform_packed_roots[current as usize].is_some() {
                continue;
            }
            let transform_node = self.packed_transform_nodes[current as usize];
            debug_assert!(transform_node.level != 0);
            if !ready {
                stack.push((current, true));
                for child in transform_node.children {
                    if self.packed_transform_packed_roots[child as usize].is_none() {
                        if self.packed_transform_nodes[child as usize].level == 0 {
                            let leaf_population =
                                self.packed_transform_nodes[child as usize].leaf_population;
                            self.packed_transform_packed_roots[child as usize] =
                                Some(PackedNodeKey::new(0, [leaf_population, 0, 0, 0]));
                        } else {
                            stack.push((child, false));
                        }
                    }
                }
                continue;
            }
            let child_roots = transform_node.children.map(|child| {
                self.packed_transform_packed_roots[child as usize]
                    .expect("iterative packed roots must resolve children before parent")
            });
            let packed = PackedNodeKey::new(
                transform_node.level,
                child_roots.map(|child| self.materialize_packed_node_key_internal(child)),
            );
            self.packed_transform_packed_roots[current as usize] = Some(packed);
        }
        self.packed_transform_packed_roots[id as usize]
            .expect("iterative packed root must resolve target")
    }

    #[cfg(test)]
    pub(super) fn materialize_winning_packed_transform_root(
        &mut self,
        id: PackedTransformId,
    ) -> PackedNodeKey {
        self.packed_root_from_transform_id(id)
    }

    fn canonicalize_blocked_jump_node(&mut self, node: NodeId) -> CanonicalNodeProbe {
        self.stats.symmetry_gate_canonical_cache_bypasses += 1;
        let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
        self.stats.structural_fast_path_lookups += 1;
        if let Some(cached) = self.structural_fast_path_cache.get(&node) {
            self.stats.structural_fast_path_hits += 1;
            return CanonicalNodeProbe {
                node: cached,
                fingerprint,
                used_cached_fingerprint: true,
            };
        }
        if let Some(cached) = self.packed_structural_fast_path_cache.get(&packed) {
            self.stats.structural_fast_path_hits += 1;
            self.structural_fast_path_cache.insert(node, cached);
            return CanonicalNodeProbe {
                node: cached,
                fingerprint,
                used_cached_fingerprint: true,
            };
        }
        self.stats.structural_fast_path_misses += 1;
        let structural = if packed.level == 0 {
            CanonicalStructKey::leaf(packed.children[0] != 0)
        } else {
            let input_children = self.direct_parent_input_children(packed, Symmetry::Identity);
            if let Some(canonical) =
                self.lookup_direct_parent_identity(packed.level, Symmetry::Identity, input_children)
            {
                canonical.structural
            } else {
                self.stats.canonical_blocked_structural_fallbacks += 1;
                self.stats.symmetry_scan_fallbacks += 1;
                self.transformed_order_entry(packed, Symmetry::Identity)
                    .structural
            }
        };
        let canonical = CanonicalNodeProbe {
            node: CanonicalNodeIdentity {
                packed,
                structural,
                symmetry: Symmetry::Identity,
            },
            fingerprint,
            used_cached_fingerprint: true,
        };
        self.structural_fast_path_cache.insert(node, canonical.node);
        self.packed_structural_fast_path_cache
            .insert(packed, canonical.node);
        canonical
    }

    fn scan_canonical_transform_winner(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (Symmetry, PackedTransformOrderEntry) {
        let mut canonical_symmetry = Symmetry::Identity;
        let mut canonical_entry =
            self.transformed_order_entry(packed, base_symmetry.then(Symmetry::Identity));
        if record_miss {
            self.stats.packed_d4_canonicalization_misses += 1;
        }
        for symmetry in Symmetry::ALL.into_iter().skip(1) {
            let candidate_symmetry = base_symmetry.then(symmetry);
            let candidate_entry = self.transformed_order_entry(packed, candidate_symmetry);
            if self.compare_transform_order_entries(
                packed,
                candidate_symmetry,
                candidate_entry,
                packed,
                base_symmetry.then(canonical_symmetry),
                canonical_entry,
            ) == std::cmp::Ordering::Less
            {
                canonical_symmetry = symmetry;
                canonical_entry = candidate_entry;
            }
        }
        (canonical_symmetry, canonical_entry)
    }

    fn canonical_transform_winner_fallback(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (PackedTransformId, Symmetry, CanonicalStructKey) {
        let (canonical_symmetry, canonical_entry) =
            self.scan_canonical_transform_winner(packed, base_symmetry, record_miss);
        let canonical_id =
            self.transform_packed_node_key(packed, base_symmetry.then(canonical_symmetry));
        (canonical_id, canonical_symmetry, canonical_entry.structural)
    }

    #[inline]
    fn direct_parent_cache_key(
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

    #[cfg(test)]
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

    fn direct_parent_input_children(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> [CanonicalNodeRef; 4] {
        let child_keys = packed
            .children;
        child_keys.map(|child| self.node_columns.symmetry_entry(child, base_symmetry).canonical_ref)
    }

    fn lookup_direct_parent_identity(
        &mut self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
    ) -> Option<CanonicalNodeIdentity> {
        self.stats.direct_parent_winner_lookups += 1;
        let cache_key = self.direct_parent_cache_key(level, base_symmetry, input_children);
        if let Some(cached) = self.hot_direct_parent_canonical_cache.get(&cache_key) {
            self.stats.direct_parent_winner_hits += 1;
            return Some(cached);
        }
        if let Some(cached) = self.direct_parent_canonical_cache.get(&cache_key) {
            self.stats.direct_parent_winner_hits += 1;
            self.maybe_promote_hot_direct_parent_identity(cache_key, cached);
            return Some(cached);
        }
        self.stats.direct_parent_winner_misses += 1;
        self.stats.direct_parent_winner_fallbacks += 1;
        None
    }

    fn cache_direct_parent_identity(
        &mut self,
        level: u32,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
        canonical: CanonicalNodeIdentity,
    ) {
        let cache_key = self.direct_parent_cache_key(level, base_symmetry, input_children);
        self.direct_parent_canonical_cache.insert(cache_key, canonical);
    }

    #[inline]
    fn maybe_promote_hot_direct_parent_identity(
        &mut self,
        cache_key: DirectCanonicalParentKey,
        canonical: CanonicalNodeIdentity,
    ) {
        if self.hot_direct_parent_canonical_budget == 0 {
            self.rebalance_hot_canonical_budgets();
        }
        if self.hot_direct_parent_canonical_cache.len() < self.hot_direct_parent_canonical_budget {
            self.hot_direct_parent_canonical_cache
                .insert(cache_key, canonical);
        }
    }

    #[inline]
    fn lookup_canonical_packed_identity(
        &mut self,
        packed: PackedNodeKey,
    ) -> Option<CanonicalNodeIdentity> {
        self.stats.canonical_packed_cache_lookups += 1;
        if let Some(cached) = self.hot_canonical_packed_cache.get(&packed) {
            self.stats.canonical_packed_cache_hits += 1;
            return Some(cached);
        }
        if let Some(cached) = self.canonical_packed_cache.get(&packed) {
            self.stats.canonical_packed_cache_hits += 1;
            self.maybe_promote_hot_canonical_packed_identity(packed, cached);
            return Some(cached);
        }
        self.stats.canonical_packed_cache_misses += 1;
        None
    }

    #[inline]
    fn cache_canonical_packed_identity(
        &mut self,
        packed: PackedNodeKey,
        canonical: CanonicalNodeIdentity,
    ) {
        self.canonical_packed_cache.insert(packed, canonical);
    }

    #[inline]
    fn maybe_promote_hot_canonical_packed_identity(
        &mut self,
        packed: PackedNodeKey,
        canonical: CanonicalNodeIdentity,
    ) {
        if self.hot_canonical_packed_budget == 0 {
            self.rebalance_hot_canonical_budgets();
        }
        if self.hot_canonical_packed_cache.len() < self.hot_canonical_packed_budget {
            self.hot_canonical_packed_cache.insert(packed, canonical);
        }
    }

    #[inline]
    fn lookup_canonical_oriented_identity(
        &mut self,
        cache_key: PackedSymmetryKey,
    ) -> Option<CanonicalNodeIdentity> {
        self.stats.canonical_oriented_cache_lookups += 1;
        if let Some(cached) = self.hot_canonical_oriented_cache.get(&cache_key) {
            self.stats.canonical_oriented_cache_hits += 1;
            return Some(cached);
        }
        if let Some(cached) = self.canonical_oriented_cache.get(&cache_key) {
            self.stats.canonical_oriented_cache_hits += 1;
            self.maybe_promote_hot_canonical_oriented_identity(cache_key, cached);
            return Some(cached);
        }
        self.stats.canonical_oriented_cache_misses += 1;
        None
    }

    #[inline]
    fn cache_canonical_oriented_identity(
        &mut self,
        cache_key: PackedSymmetryKey,
        canonical: CanonicalNodeIdentity,
    ) {
        self.canonical_oriented_cache.insert(cache_key, canonical);
    }

    #[inline]
    fn maybe_promote_hot_canonical_oriented_identity(
        &mut self,
        cache_key: PackedSymmetryKey,
        canonical: CanonicalNodeIdentity,
    ) {
        if self.hot_canonical_oriented_budget == 0 {
            self.rebalance_hot_canonical_budgets();
        }
        if self.hot_canonical_oriented_cache.len() < self.hot_canonical_oriented_budget {
            self.hot_canonical_oriented_cache.insert(cache_key, canonical);
        }
    }

    #[cfg(test)]
    pub(super) fn direct_parent_winner_for_tests(
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

    #[cfg(test)]
    pub(super) fn canonicalize_packed_direct_for_tests(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> (u64, u64, u8) {
        let canonical = self.canonicalize_packed_direct(packed, base_symmetry, false).node;
        (
            canonical.packed.fingerprint(),
            canonical.structural.fingerprint(),
            canonical.symmetry as u8,
        )
    }

    fn canonicalize_packed_direct(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        count_fallback: bool,
    ) -> CanonicalNodeProbe {
        let input_children = self.direct_parent_input_children(packed, base_symmetry);
        self.canonicalize_packed_direct_known_children(
            packed,
            base_symmetry,
            input_children,
            count_fallback,
        )
    }

    fn canonicalize_packed_direct_known_children(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
        count_fallback: bool,
    ) -> CanonicalNodeProbe {
        if let Some(canonical) =
            self.lookup_direct_parent_identity(packed.level, base_symmetry, input_children)
        {
            self.stats.direct_parent_cached_result_hits += 1;
            return CanonicalNodeProbe {
                node: canonical,
                fingerprint: if canonical.symmetry == Symmetry::Identity {
                    packed.fingerprint()
                } else {
                    canonical.packed.fingerprint()
                },
                used_cached_fingerprint: canonical.symmetry == Symmetry::Identity,
            };
        }

        self.stats.symmetry_scan_fallbacks += 1;
        let (canonical_id, canonical_symmetry, canonical_structural) =
            self.canonical_transform_winner_fallback(packed, base_symmetry, count_fallback);
        self.stats.canonical_transform_root_reconstructions += 1;
        let canonical = CanonicalNodeIdentity {
            packed: self.packed_root_from_transform_id(canonical_id),
            structural: canonical_structural,
            symmetry: canonical_symmetry,
        };
        self.cache_direct_parent_identity(packed.level, base_symmetry, input_children, canonical);
        CanonicalNodeProbe {
            node: canonical,
            fingerprint: if canonical_symmetry == Symmetry::Identity {
                packed.fingerprint()
            } else {
                canonical.packed.fingerprint()
            },
            used_cached_fingerprint: canonical_symmetry == Symmetry::Identity,
        }
    }

    fn canonicalize_packed_identity(&mut self, packed: PackedNodeKey) -> CanonicalNodeProbe {
        if packed.level == 0 {
            let structural = CanonicalStructKey::leaf(packed.children[0] != 0);
            return CanonicalNodeProbe {
                node: CanonicalNodeIdentity {
                    packed,
                    structural,
                    symmetry: Symmetry::Identity,
                },
                fingerprint: packed.fingerprint(),
                used_cached_fingerprint: true,
            };
        }
        if let Some(cached) = self.lookup_canonical_packed_identity(packed) {
            return CanonicalNodeProbe {
                node: cached,
                fingerprint: if cached.symmetry == Symmetry::Identity {
                    packed.fingerprint()
                } else {
                    cached.packed.fingerprint()
                },
                used_cached_fingerprint: cached.symmetry == Symmetry::Identity,
            };
        }
        let canonical = self.canonicalize_packed_direct(packed, Symmetry::Identity, true);
        self.cache_canonical_packed_identity(packed, canonical.node);
        canonical
    }

    pub(super) fn canonicalize_packed_under_symmetry(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> CanonicalNodeProbe {
        if base_symmetry == Symmetry::Identity {
            return self.canonicalize_packed_identity(packed);
        }
        if packed.level == 0 {
            let structural = CanonicalStructKey::leaf(packed.children[0] != 0);
            return CanonicalNodeProbe {
                node: CanonicalNodeIdentity {
                    packed,
                    structural,
                    symmetry: Symmetry::Identity,
                },
                fingerprint: packed.fingerprint(),
                used_cached_fingerprint: false,
            };
        }
        let cache_key = PackedSymmetryKey {
            packed,
            symmetry: base_symmetry,
        };
        if let Some(cached) = self.lookup_canonical_oriented_identity(cache_key) {
            return CanonicalNodeProbe {
                node: cached,
                fingerprint: cached.packed.fingerprint(),
                used_cached_fingerprint: false,
            };
        }
        let canonical = self.canonicalize_packed_direct(packed, base_symmetry, false);
        self.cache_canonical_oriented_identity(cache_key, canonical.node);
        canonical
    }

    #[cfg(test)]
    pub(super) fn materialize_packed_transform_root(&mut self, id: PackedTransformId) -> NodeId {
        self.stats.packed_cache_result_materializations += 1;
        self.materialize_packed_transform_node_internal(id)
    }

    #[inline]
    pub(super) fn canonical_jump_probe(&mut self, key: (NodeId, u32)) -> CanonicalJumpProbe {
        let (node, _packed_fingerprint, used_cached_fingerprint) =
            if self.record_symmetry_gate_decision(key.0) {
                let canonical = self.canonicalize_packed_node(key.0);
                (
                    canonical.node,
                    canonical.fingerprint,
                    canonical.used_cached_fingerprint,
                )
            } else {
                let canonical = self.canonicalize_blocked_jump_node(key.0);
                (
                    canonical.node,
                    canonical.fingerprint,
                    canonical.used_cached_fingerprint,
                )
            };
        CanonicalJumpProbe {
            key: CanonicalJumpKey {
                structural: node.structural,
                step_exp: key.1,
            },
            node,
            fingerprint: hash_packed_jump_fingerprint(node.structural.fingerprint(), key.1),
            used_cached_fingerprint,
        }
    }

    #[inline]
    pub(super) fn record_symmetry_gate_decision(&mut self, node: NodeId) -> bool {
        let allowed = self.should_symmetry_canonicalize_jump_node(node);
        if allowed {
            self.stats.symmetry_gate_allowed += 1;
        } else {
            self.stats.symmetry_gate_blocked += 1;
        }
        allowed
    }

    #[cfg(test)]
    pub(super) fn transform_node(&mut self, node: NodeId, symmetry: Symmetry) -> NodeId {
        if symmetry == Symmetry::Identity || self.node_columns.level(node) == 0 {
            return node;
        }
        if let Some(transformed) = self
            .transform_cache
            .get(&TransformCacheKey { node, symmetry })
        {
            return transformed;
        }
        let mut stack = Vec::with_capacity(
            (self.node_columns.level(node) as usize)
                .saturating_mul(4)
                .max(8),
        );
        stack.push((node, false));
        while let Some((current, ready)) = stack.pop() {
            if self.node_columns.level(current) == 0 {
                continue;
            }
            let key = TransformCacheKey {
                node: current,
                symmetry,
            };
            if self.transform_cache.get(&key).is_some() {
                continue;
            }
            if !ready {
                stack.push((current, true));
                for child in self.node_columns.quadrants(current) {
                    if self.node_columns.level(child) != 0
                        && self
                            .transform_cache
                            .get(&TransformCacheKey {
                                node: child,
                                symmetry,
                            })
                            .is_none()
                    {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            let transformed_children = self.node_columns.quadrants(current).map(|child| {
                if self.node_columns.level(child) == 0 {
                    child
                } else {
                    self.transform_cache
                        .get(&TransformCacheKey {
                            node: child,
                            symmetry,
                        })
                        .expect("iterative transform_node must resolve child before parent")
                }
            });
            let [next_nw, next_ne, next_sw, next_se] =
                symmetry.transform_quadrants(transformed_children);
            let transformed = self.join(next_nw, next_ne, next_sw, next_se);
            self.stats.transformed_node_materializations += 1;
            self.transform_cache.insert(key, transformed);
        }
        self.transform_cache
            .get(&TransformCacheKey { node, symmetry })
            .expect("iterative transform_node must resolve target")
    }

    #[cfg(test)]
    pub(super) fn transform_node_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        symmetry: Symmetry,
    ) -> [NodeId; N] {
        if symmetry == Symmetry::Identity {
            return nodes;
        }

        let mut transformed = [0; N];
        for lane in 0..N {
            let node = nodes[lane];
            if self.node_columns.level(node) == 0 {
                transformed[lane] = node;
                continue;
            }
            let mut reused = None;
            for prev in 0..lane {
                if nodes[prev] == node {
                    reused = Some(transformed[prev]);
                    break;
                }
            }
            transformed[lane] = reused.unwrap_or_else(|| self.transform_node(node, symmetry));
        }
        transformed
    }

    pub(super) fn canonicalize_packed_key_for_snapshot(
        &mut self,
        packed: PackedNodeKey,
    ) -> CanonicalNodeProbe {
        self.canonicalize_packed_identity(packed)
    }

    pub(super) fn canonicalize_packed_nodes_batch<const N: usize>(
        &mut self,
        nodes: &[NodeId; N],
        active_lanes: usize,
    ) -> [CanonicalNodeProbe; N] {
        let mut canonical = [CanonicalNodeProbe {
            node: CanonicalNodeIdentity {
                packed: PackedNodeKey::new(0, [0; 4]),
                structural: CanonicalStructKey::new(0, [0; 4]),
                symmetry: Symmetry::Identity,
            },
            fingerprint: 0,
            used_cached_fingerprint: true,
        }; N];
        let mut miss_indices = [usize::MAX; N];
        let mut miss_count = 0;

        self.stats.canonical_batch_lanes += active_lanes;
        self.stats.canonical_batch_batches += usize::from(active_lanes != 0);

        for lane in 0..active_lanes {
            let node = nodes[lane];
            if self.node_columns.level(node) == 0 {
                let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
                canonical[lane] = CanonicalNodeProbe {
                    node: CanonicalNodeIdentity {
                        packed,
                        structural: CanonicalStructKey::leaf(packed.children[0] != 0),
                        symmetry: Symmetry::Identity,
                    },
                    fingerprint,
                    used_cached_fingerprint: true,
                };
                continue;
            }
            if let Some(cached) = self.canonical_node_cache.get(&node) {
                self.stats.canonical_node_cache_hits += 1;
                let used_cached_fingerprint = cached.symmetry == Symmetry::Identity;
                canonical[lane] = CanonicalNodeProbe {
                    node: cached,
                    fingerprint: if used_cached_fingerprint {
                        self.node_columns.fingerprint(node)
                    } else {
                        cached.packed.fingerprint()
                    },
                    used_cached_fingerprint,
                };
                continue;
            }
            miss_indices[miss_count] = lane;
            miss_count += 1;
        }

        if miss_count != 0 {
            self.stats.canonical_node_cache_misses += miss_count;
            for compact_index in 0..miss_count {
                let lane = miss_indices[compact_index];
                canonical[lane] =
                    self.canonicalize_packed_identity(self.node_columns.packed_key(nodes[lane]));
                self.canonical_node_cache
                    .insert(nodes[lane], canonical[lane].node);
            }
        }

        canonical
    }

    fn materialize_packed_node_key_internal(&mut self, packed: PackedNodeKey) -> NodeId {
        if packed.level == 0 {
            return if packed.children[0] == 0 {
                self.dead_leaf
            } else {
                self.live_leaf
            };
        }
        self.join(
            packed.children[0],
            packed.children[1],
            packed.children[2],
            packed.children[3],
        )
    }

    pub(super) fn materialize_packed_node_key(&mut self, packed: PackedNodeKey) -> NodeId {
        self.stats.packed_cache_result_materializations += 1;
        self.materialize_packed_node_key_internal(packed)
    }

    pub(super) fn canonicalize_packed_node(&mut self, node: NodeId) -> CanonicalNodeProbe {
        self.canonicalize_packed_nodes_batch(&[node], 1)[0]
    }

    pub(super) fn should_symmetry_canonicalize_jump_node(&self, node: NodeId) -> bool {
        #[cfg(test)]
        if let Some((max_level, max_population)) = self.symmetry_gate_override {
            return self.node_columns.level(node) <= max_level
                && self.node_columns.population(node) <= max_population;
        }
        self.node_columns.level(node) <= JUMP_SYMMETRY_MAX_LEVEL
            && self.node_columns.population(node) <= JUMP_SYMMETRY_MAX_POPULATION
    }
}
