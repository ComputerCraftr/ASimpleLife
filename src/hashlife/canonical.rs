use super::*;

impl HashLifeEngine {
    pub(super) fn reset_packed_transform_state(&mut self) {
        self.canonical_transform_cache.clear();
        self.oriented_result_cache.clear();
        self.packed_transform_compare_cache.clear();
        self.packed_symmetry_children_cache.clear();
        self.packed_transform_intern.clear();
        self.packed_transform_nodes.clear();
        self.packed_transform_materialized.clear();
        self.packed_transform_packed_keys.clear();

        for alive in [false, true] {
            let leaf_id = self.packed_transform_nodes.len() as PackedTransformId;
            let population = u64::from(alive);
            self.packed_transform_nodes.push(PackedTransformNode {
                level: 0,
                leaf_population: population,
                children: [0; 4],
            });
            self.packed_transform_materialized
                .push(Some(if alive { self.live_leaf } else { self.dead_leaf }));
            self.packed_transform_packed_keys
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
        self.packed_transform_nodes.push(PackedTransformNode {
            level,
            leaf_population: 0,
            children,
        });
        self.packed_transform_materialized.push(None);
        self.packed_transform_packed_keys.push(None);
        self.packed_transform_intern.insert(shape, id);
        id
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
        let transformed_children = if let Some(entry) = self.packed_symmetry_children_cache.get(&packed) {
            entry.children[symmetry as usize]
        } else {
            let child_keys = packed.children.map(|child| self.node_columns.packed_key(child));
            let mut children_by_symmetry = [[0_u32; 4]; 8];
            for candidate in Symmetry::ALL {
                let perm = candidate.quadrant_perm();
                let child_ids = child_keys.map(|child| self.transform_packed_node_key(child, candidate));
                children_by_symmetry[candidate as usize] = [
                    child_ids[perm[0]],
                    child_ids[perm[1]],
                    child_ids[perm[2]],
                    child_ids[perm[3]],
                ];
            }
            self.packed_symmetry_children_cache.insert(
                packed,
                PackedSymmetryChildrenCacheEntry {
                    children: children_by_symmetry,
                },
            );
            children_by_symmetry[symmetry as usize]
        };
        let transformed = self.intern_packed_transform_node(
            packed.level,
            transformed_children,
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
        self.packed_transform_packed_keys[id as usize] =
            Some(PackedNodeKey::new(transform_node.level, children));
        node
    }

    fn packed_key_from_transform_id(&mut self, id: PackedTransformId) -> PackedNodeKey {
        if let Some(packed) = self.packed_transform_packed_keys[id as usize] {
            return packed;
        }
        let transform_node = self.packed_transform_nodes[id as usize];
        if transform_node.level == 0 {
            return PackedNodeKey::new(0, [transform_node.leaf_population, 0, 0, 0]);
        }
        let children = transform_node
            .children
            .map(|child| self.materialize_packed_transform_node_internal(child));
        let packed = PackedNodeKey::new(transform_node.level, children);
        self.packed_transform_packed_keys[id as usize] = Some(packed);
        packed
    }

    pub(super) fn materialize_packed_transform_root(&mut self, id: PackedTransformId) -> NodeId {
        self.stats.packed_cache_result_materializations += 1;
        self.materialize_packed_transform_node_internal(id)
    }

    #[inline]
    pub(super) fn canonical_cache_key_packed_only(
        &mut self,
        key: (NodeId, u32),
    ) -> (PackedJumpCacheKey, Symmetry, u64, bool) {
        let (packed, symmetry, packed_fingerprint, used_cached_fingerprint) =
            if self.record_symmetry_gate_decision(key.0) {
                let canonical = self.canonicalize_packed_node(key.0);
                (
                    canonical.packed,
                    canonical.symmetry,
                    canonical.fingerprint,
                    canonical.used_cached_fingerprint,
                )
            } else {
                let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(key.0);
                (packed, Symmetry::Identity, fingerprint, true)
            };
        (
            PackedJumpCacheKey {
                packed,
                step_exp: key.1,
            },
            symmetry,
            hash_packed_jump_fingerprint(packed_fingerprint, key.1),
            used_cached_fingerprint,
        )
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

    fn canonicalize_packed_key_uncached(&mut self, packed: PackedNodeKey) -> CanonicalPackedNode {
        if packed.level == 0 {
            return CanonicalPackedNode {
                symmetry: Symmetry::Identity,
                packed,
                fingerprint: packed.fingerprint(),
                used_cached_fingerprint: true,
            };
        }

        let mut canonical_symmetry = Symmetry::Identity;
        let mut canonical_packed = packed;
        self.stats.packed_d4_canonicalization_misses += 1;
        for symmetry in Symmetry::ALL.into_iter().skip(1) {
            let candidate_id = self.transform_packed_node_key(packed, symmetry);
            let candidate_packed = self.packed_key_from_transform_id(candidate_id);
            if candidate_packed < canonical_packed {
                canonical_symmetry = symmetry;
                canonical_packed = candidate_packed;
            }
        }

        CanonicalPackedNode {
            symmetry: canonical_symmetry,
            packed: canonical_packed,
            fingerprint: canonical_packed.fingerprint(),
            used_cached_fingerprint: canonical_symmetry == Symmetry::Identity,
        }
    }

    pub(super) fn canonicalize_packed_nodes_batch<const N: usize>(
        &mut self,
        nodes: &[NodeId; N],
        active_lanes: usize,
    ) -> [CanonicalPackedNode; N] {
        let mut canonical = [CanonicalPackedNode {
            symmetry: Symmetry::Identity,
            packed: PackedNodeKey::new(0, [0; 4]),
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
                canonical[lane] = CanonicalPackedNode {
                    symmetry: Symmetry::Identity,
                    packed,
                    fingerprint,
                    used_cached_fingerprint: true,
                };
                continue;
            }
            if let Some(cached) = self.canonical_node_cache.get(&node) {
                self.stats.canonical_node_cache_hits += 1;
                let used_cached_fingerprint = cached.symmetry == Symmetry::Identity;
                canonical[lane] = CanonicalPackedNode {
                    symmetry: cached.symmetry,
                    packed: cached.packed,
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
            canonical[lane] = CanonicalPackedNode {
                symmetry: Symmetry::Identity,
                packed,
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
                self.canonical_node_cache.insert(
                    nodes[lane],
                    CanonicalNodeCacheEntry {
                        symmetry: canonical[lane].symmetry,
                        packed: canonical[lane].packed,
                    },
                );
            }
        }

        canonical
    }

    pub(super) fn materialize_packed_node_key(&mut self, packed: PackedNodeKey) -> NodeId {
        self.stats.packed_cache_result_materializations += 1;
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

    pub(super) fn canonicalize_packed_node(&mut self, node: NodeId) -> CanonicalPackedNode {
        self.canonicalize_packed_nodes_batch(&[node], 1)[0]
    }

    pub(super) fn canonical_jump_key_packed(
        &mut self,
        key: (NodeId, u32),
    ) -> (CanonicalJumpKey, Symmetry) {
        let (cache_key, symmetry, _fingerprint, _used_cached_fingerprint) =
            self.canonical_cache_key_packed_only(key);
        (
            CanonicalJumpKey {
                packed: cache_key.packed,
                step_exp: key.1,
            },
            symmetry,
        )
    }

    pub(super) fn should_symmetry_canonicalize_jump_node(&self, node: NodeId) -> bool {
        self.node_columns.level(node) <= self.jump_symmetry_max_level
            && self.node_columns.population(node) <= self.jump_symmetry_max_population
    }

    pub(super) fn cached_jump_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let (canonical_key, symmetry, fingerprint, used_cached_fingerprint) =
            self.canonical_cache_key_packed_only(key);
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        let inverse = symmetry.inverse();
        self.stats.packed_cache_result_lookups += 1;
        let result = self
            .jump_cache
            .get_with_fingerprint(&canonical_key, fingerprint)?;
        self.stats.packed_cache_result_hits += 1;
        if symmetry != Symmetry::Identity {
            self.stats.symmetric_jump_cache_hits += 1;
        }
        Some(self.materialize_oriented_packed_result(
            result.packed,
            result.symmetry,
            inverse,
        ))
    }

    pub(super) fn insert_jump_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let (canonical_key, symmetry, fingerprint, used_cached_fingerprint) =
            self.canonical_cache_key_packed_only(key);
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        let transformed_result =
            self.transform_packed_node_key(self.node_columns.packed_key(result), symmetry);
        let canonical_input_result = self.packed_key_from_transform_id(transformed_result);
        let canonical_result = self.canonicalize_packed_key_uncached(canonical_input_result);
        self.jump_cache.insert_with_fingerprint(
            canonical_key,
            fingerprint,
            PackedResultCacheEntry {
                packed: canonical_result.packed,
                symmetry: canonical_result.symmetry,
            },
        );
    }

    pub(super) fn insert_canonical_jump_result(
        &mut self,
        key: CanonicalJumpKey,
        result: NodeId,
    ) {
        let packed = key.packed;
        let fingerprint = hash_packed_jump_fingerprint(packed.fingerprint(), key.step_exp);
        let canonical_result = self.canonicalize_packed_key_uncached(self.node_columns.packed_key(result));
        self.jump_cache.insert_with_fingerprint(
            PackedJumpCacheKey {
                packed,
                step_exp: key.step_exp,
            },
            fingerprint,
            PackedResultCacheEntry {
                packed: canonical_result.packed,
                symmetry: canonical_result.symmetry,
            },
        );
    }

    pub(super) fn jump_result(&mut self, key: (NodeId, u32)) -> NodeId {
        self.cached_jump_result(key)
            .expect("missing HashLife jump result")
    }

    pub(super) fn cached_root_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let (cache_key, symmetry, fingerprint, used_cached_fingerprint) =
            self.canonical_cache_key_packed_only(key);
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        self.stats.packed_cache_result_lookups += 1;
        let result = self
            .root_result_cache
            .get_with_fingerprint(&cache_key, fingerprint)?;
        self.stats.packed_cache_result_hits += 1;
        Some(self.materialize_oriented_packed_result(
            result.packed,
            result.symmetry,
            symmetry.inverse(),
        ))
    }

    pub(super) fn insert_root_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let (cache_key, symmetry, fingerprint, used_cached_fingerprint) =
            self.canonical_cache_key_packed_only(key);
        self.record_fingerprint_probe(used_cached_fingerprint, 1);
        let transformed_result =
            self.transform_packed_node_key(self.node_columns.packed_key(result), symmetry);
        let canonical_input_result = self.packed_key_from_transform_id(transformed_result);
        let canonical_result = self.canonicalize_packed_key_uncached(canonical_input_result);
        self.root_result_cache.insert_with_fingerprint(
            cache_key,
            fingerprint,
            PackedResultCacheEntry {
                packed: canonical_result.packed,
                symmetry: canonical_result.symmetry,
            },
        );
    }
}
