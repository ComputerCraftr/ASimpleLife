use super::*;

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
            self.packed_transform_materialized
                .push(Some(if alive { self.live_leaf } else { self.dead_leaf }));
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
        let order_children = children.map(|child| self.packed_transform_nodes[child as usize].canonical_ref);
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
    pub(super) fn structural_key_from_transform_id(&self, id: PackedTransformId) -> CanonicalStructKey {
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
        let cache_key = PackedTransformCacheKey { packed, symmetry };
        if let Some(transformed) = self.canonical_transform_cache.get(&cache_key) {
            self.stats.packed_recursive_transform_hits += 1;
            return transformed;
        }
        self.stats.packed_recursive_transform_misses += 1;
        let child_keys = packed.children.map(|child| self.node_columns.packed_key(child));
        let perm = symmetry.quadrant_perm();
        let child_ids = child_keys.map(|child| self.transform_packed_node_key(child, symmetry));
        let transformed = self.intern_packed_transform_node(
            packed.level,
            [
                child_ids[perm[0]],
                child_ids[perm[1]],
                child_ids[perm[2]],
                child_ids[perm[3]],
            ],
        );
        self.canonical_transform_cache.insert(cache_key, transformed);
        transformed
    }

    #[allow(dead_code)]
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
            (PackedTransformCompareKey { left: right, right: left }, true)
        };
        if let Some(cached) = self.packed_transform_compare_cache.get(&key) {
            return match (cached, invert) {
                (0, _) => std::cmp::Ordering::Equal,
                (-1, false) | (1, true) => std::cmp::Ordering::Less,
                (1, false) | (-1, true) => std::cmp::Ordering::Greater,
                _ => unreachable!("invalid packed transform compare cache entry"),
            };
        }
        let left_node = self.packed_transform_nodes[left as usize];
        let right_node = self.packed_transform_nodes[right as usize];
        let ordering = left_node.level.cmp(&right_node.level).then_with(|| {
            if left_node.level == 0 {
                left_node.leaf_population.cmp(&right_node.leaf_population)
            } else {
                for index in 0..4 {
                    let ordering =
                        self.compare_packed_transform_ids(left_node.children[index], right_node.children[index]);
                    if ordering != std::cmp::Ordering::Equal {
                        return ordering;
                    }
                }
                std::cmp::Ordering::Equal
            }
        });
        let cached_value = match ordering {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        };
        self.packed_transform_compare_cache.insert(key, cached_value);
        ordering
    }

    #[inline]
    fn compare_packed_transform_order(
        &mut self,
        left: PackedTransformId,
        right: PackedTransformId,
    ) -> std::cmp::Ordering {
        if left == right {
            return std::cmp::Ordering::Equal;
        }
        let left_node = self.packed_transform_nodes[left as usize];
        let right_node = self.packed_transform_nodes[right as usize];
        let ordering = left_node
            .level
            .cmp(&right_node.level)
            .then_with(|| left_node.order_fingerprint.cmp(&right_node.order_fingerprint))
            .then_with(|| left_node.order_children.cmp(&right_node.order_children));
        if ordering == std::cmp::Ordering::Equal {
            self.compare_packed_transform_ids(left, right)
        } else {
            ordering
        }
    }

    fn materialize_packed_transform_node_internal(&mut self, id: PackedTransformId) -> NodeId {
        if let Some(node) = self.packed_transform_materialized[id as usize] {
            return node;
        }
        let transform_node = self.packed_transform_nodes[id as usize];
        debug_assert!(transform_node.level != 0);
        let children = transform_node
            .children
            .map(|child| self.materialize_packed_transform_node_internal(child));
        let node = self.join(children[0], children[1], children[2], children[3]);
        self.packed_transform_materialized[id as usize] = Some(node);
        node
    }

    fn packed_root_from_transform_id(&mut self, id: PackedTransformId) -> PackedNodeKey {
        if let Some(packed) = self.packed_transform_packed_roots[id as usize] {
            return packed;
        }
        let transform_node = self.packed_transform_nodes[id as usize];
        debug_assert!(transform_node.level != 0);
        let child_roots = transform_node
            .children
            .map(|child| self.packed_root_from_transform_id(child));
        let packed = PackedNodeKey::new(
            transform_node.level,
            child_roots.map(|child| self.materialize_packed_node_key_internal(child)),
        );
        self.packed_transform_packed_roots[id as usize] = Some(packed);
        packed
    }

    pub(super) fn materialize_winning_packed_transform_root(
        &mut self,
        id: PackedTransformId,
    ) -> PackedNodeKey {
        self.packed_root_from_transform_id(id)
    }

    fn canonical_transform_winner(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        record_miss: bool,
    ) -> (PackedTransformId, Symmetry, CanonicalStructKey) {
        let mut canonical_symmetry = Symmetry::Identity;
        let mut canonical_id = self.transform_packed_node_key(packed, base_symmetry.then(Symmetry::Identity));
        let mut canonical_structural = self.structural_key_from_transform_id(canonical_id);
        if record_miss {
            self.stats.packed_d4_canonicalization_misses += 1;
        }
        for symmetry in Symmetry::ALL.into_iter().skip(1) {
            let candidate_id = self.transform_packed_node_key(packed, base_symmetry.then(symmetry));
            if self.compare_packed_transform_order(candidate_id, canonical_id) == std::cmp::Ordering::Less {
                canonical_symmetry = symmetry;
                canonical_id = candidate_id;
                canonical_structural = self.structural_key_from_transform_id(candidate_id);
            }
        }
        (canonical_id, canonical_symmetry, canonical_structural)
    }

    pub(super) fn materialize_packed_transform_root(&mut self, id: PackedTransformId) -> NodeId {
        self.stats.packed_cache_result_materializations += 1;
        self.materialize_packed_transform_node_internal(id)
    }

    #[inline]
    pub(super) fn canonical_jump_probe(
        &mut self,
        key: (NodeId, u32),
    ) -> CanonicalJumpProbe {
        let (node, _packed_fingerprint, used_cached_fingerprint) =
            if self.record_symmetry_gate_decision(key.0) {
                let canonical = self.canonicalize_packed_node(key.0);
                (
                    canonical.node,
                    canonical.fingerprint,
                    canonical.used_cached_fingerprint,
                )
            } else {
                let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(key.0);
                let transform_id = self.transform_packed_node_key(packed, Symmetry::Identity);
                let structural = self.structural_key_from_transform_id(transform_id);
                (
                    CanonicalNodeIdentity {
                        packed,
                        structural,
                        symmetry: Symmetry::Identity,
                    },
                    fingerprint,
                    true,
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
        if let Some(transformed) = self.transform_cache.get(&TransformCacheKey { node, symmetry }) {
            return transformed;
        }

        let [nw, ne, sw, se] = self.node_columns.quadrants(node);
        let transformed_children = [
            self.transform_node(nw, symmetry),
            self.transform_node(ne, symmetry),
            self.transform_node(sw, symmetry),
            self.transform_node(se, symmetry),
        ];
        let [next_nw, next_ne, next_sw, next_se] =
            symmetry.transform_quadrants(transformed_children);
        let transformed = self.join(next_nw, next_ne, next_sw, next_se);
        self.stats.transformed_node_materializations += 1;
        self.transform_cache
            .insert(TransformCacheKey { node, symmetry }, transformed);
        transformed
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

    fn canonicalize_packed_key_uncached(&mut self, packed: PackedNodeKey) -> CanonicalNodeProbe {
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

        let (canonical_id, canonical_symmetry, canonical_structural) =
            self.canonical_transform_winner(packed, Symmetry::Identity, true);
        let canonical_packed = self.materialize_winning_packed_transform_root(canonical_id);

        CanonicalNodeProbe {
            node: CanonicalNodeIdentity {
                packed: canonical_packed,
                structural: canonical_structural,
                symmetry: canonical_symmetry,
            },
            fingerprint: canonical_packed.fingerprint(),
            used_cached_fingerprint: canonical_symmetry == Symmetry::Identity,
        }
    }

    pub(super) fn canonicalize_packed_key_for_snapshot(
        &mut self,
        packed: PackedNodeKey,
    ) -> CanonicalNodeProbe {
        self.canonicalize_packed_key_uncached(packed)
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
        let mut miss_packed = [PackedNodeKey::new(0, [0; 4]); N];
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
            self.stats.canonical_node_cache_misses += 1;
            let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
            canonical[lane] = CanonicalNodeProbe {
                node: CanonicalNodeIdentity {
                    packed,
                    structural: CanonicalStructKey::new(packed.level, [0; 4]),
                    symmetry: Symmetry::Identity,
                },
                fingerprint,
                used_cached_fingerprint: true,
            };
            miss_indices[miss_count] = lane;
            miss_packed[miss_count] = packed;
            miss_count += 1;
        }

        if miss_count != 0 {
            for compact_index in 0..miss_count {
                let lane = miss_indices[compact_index];
                canonical[lane] = self.canonicalize_packed_key_uncached(miss_packed[compact_index]);
                self.canonical_node_cache.insert(nodes[lane], canonical[lane].node);
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

    #[inline]
    pub(super) fn materialize_oriented_packed_result(
        &mut self,
        packed: PackedNodeKey,
        cached_result_symmetry: Symmetry,
        output_symmetry: Symmetry,
    ) -> NodeId {
        let combined = cached_result_symmetry.inverse().then(output_symmetry);
        if combined == Symmetry::Identity {
            self.materialize_packed_node_key(packed)
        } else if let Some(oriented) = self.oriented_result_cache.get(&PackedTransformCacheKey {
            packed,
            symmetry: combined,
        }) {
            oriented
        } else {
            self.stats.packed_inverse_transform_hits += 1;
            let transformed = self.transform_packed_node_key(packed, combined);
            let oriented = self.materialize_packed_transform_root(transformed);
            self.oriented_result_cache.insert(
                PackedTransformCacheKey {
                    packed,
                    symmetry: combined,
                },
                oriented,
            );
            oriented
        }
    }

    pub(super) fn canonicalize_packed_node(&mut self, node: NodeId) -> CanonicalNodeProbe {
        self.canonicalize_packed_nodes_batch(&[node], 1)[0]
    }

    pub(super) fn should_symmetry_canonicalize_jump_node(&self, node: NodeId) -> bool {
        self.node_columns.level(node) <= self.jump_symmetry_max_level
            && self.node_columns.population(node) <= self.jump_symmetry_max_population
    }

    pub(super) fn cached_jump_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let inverse = jump_probe.node.symmetry.inverse();
        self.stats.jump_result_cache_lookups += 1;
        let result = self
            .jump_cache
            .get_with_fingerprint(&jump_probe.key, jump_probe.fingerprint)?;
        self.stats.jump_result_cache_hits += 1;
        if jump_probe.node.symmetry != Symmetry::Identity {
            self.stats.symmetric_jump_result_cache_hits += 1;
        }
        Some(self.materialize_oriented_packed_result(
            result.packed,
            result.symmetry,
            inverse,
        ))
    }

    pub(super) fn insert_jump_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let result_packed = self.node_columns.packed_key(result);
        let (canonical_id, canonical_symmetry, _) =
            self.canonical_transform_winner(result_packed, jump_probe.node.symmetry, false);
        let canonical_packed = self.materialize_winning_packed_transform_root(canonical_id);
        self.jump_cache.insert_with_fingerprint(
            jump_probe.key,
            jump_probe.fingerprint,
            PackedResultCacheEntry {
                packed: canonical_packed,
                symmetry: canonical_symmetry,
            },
        );
    }

    pub(super) fn insert_canonical_jump_result(
        &mut self,
        key: CanonicalJumpKey,
        result: NodeId,
    ) {
        self.insert_canonical_jump_results_batch(&[key], &[result], 1);
    }

    pub(super) fn insert_canonical_jump_results_batch<const N: usize>(
        &mut self,
        keys: &[CanonicalJumpKey; N],
        results: &[NodeId; N],
        active_lanes: usize,
    ) {
        if active_lanes == 0 {
            return;
        }
        let mut unique_lookup = FlatTable::<NodeId, usize>::with_capacity(active_lanes.max(4));
        let mut unique_nodes = [0_u64; N];
        let mut lane_to_unique = [usize::MAX; N];
        let mut unique_count = 0;
        for lane in 0..active_lanes {
            let result = results[lane];
            if let Some(index) = unique_lookup.get(&result) {
                lane_to_unique[lane] = index;
            } else {
                unique_nodes[unique_count] = result;
                lane_to_unique[lane] = unique_count;
                unique_lookup.insert(result, unique_count);
                unique_count += 1;
            }
        }
        let mut unique_entries = [PackedResultCacheEntry {
            packed: PackedNodeKey::new(0, [0; 4]),
            symmetry: Symmetry::Identity,
        }; N];
        for unique in 0..unique_count {
            let canonical_result =
                self.canonicalize_packed_key_uncached(self.node_columns.packed_key(unique_nodes[unique]));
            unique_entries[unique] = PackedResultCacheEntry {
                packed: canonical_result.node.packed,
                symmetry: canonical_result.node.symmetry,
            };
        }
        for lane in 0..active_lanes {
            let fingerprint =
                hash_packed_jump_fingerprint(keys[lane].structural.fingerprint(), keys[lane].step_exp);
            self.jump_cache.insert_with_fingerprint(
                keys[lane],
                fingerprint,
                unique_entries[lane_to_unique[lane]],
            );
        }
    }

    pub(super) fn jump_result(&mut self, key: (NodeId, u32)) -> NodeId {
        self.cached_jump_result(key)
            .expect("missing HashLife jump result")
    }

    pub(super) fn cached_root_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        self.stats.root_result_cache_lookups += 1;
        let result = self
            .root_result_cache
            .get_with_fingerprint(&jump_probe.key, jump_probe.fingerprint)?;
        self.stats.root_result_cache_hits += 1;
        Some(self.materialize_oriented_packed_result(
            result.packed,
            result.symmetry,
            jump_probe.node.symmetry.inverse(),
        ))
    }

    pub(super) fn insert_root_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let result_packed = self.node_columns.packed_key(result);
        let (canonical_id, canonical_symmetry, _) =
            self.canonical_transform_winner(result_packed, jump_probe.node.symmetry, false);
        let canonical_packed = self.materialize_winning_packed_transform_root(canonical_id);
        self.root_result_cache.insert_with_fingerprint(
            jump_probe.key,
            jump_probe.fingerprint,
            PackedResultCacheEntry {
                packed: canonical_packed,
                symmetry: canonical_symmetry,
            },
        );
    }
}
