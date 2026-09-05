use super::*;

impl HashLifeEngine {
    pub(in crate::hashlife) fn canonicalize_packed_direct(
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

    pub(super) fn canonicalize_packed_direct_known_children(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
        input_children: [CanonicalNodeRef; 4],
        count_fallback: bool,
    ) -> CanonicalNodeProbe {
        if let Some(canonical) =
            self.lookup_direct_parent_identity(packed.level, base_symmetry, input_children)
        {
            self.stats.canonical_cache.direct_parent_cached_result_hits += 1;
            let used_cached_fingerprint =
                base_symmetry == Symmetry::Identity && canonical.symmetry == Symmetry::Identity;
            return CanonicalNodeProbe {
                node: canonical,
                fingerprint: if used_cached_fingerprint {
                    packed.fingerprint()
                } else {
                    canonical.packed.fingerprint()
                },
                used_cached_fingerprint,
            };
        }

        self.stats.canonical_fallback.symmetry_scan_fallbacks += 1;
        let (canonical_packed, canonical_symmetry, canonical_structural) =
            self.canonical_transform_winner_fallback(packed, base_symmetry, count_fallback);
        let canonical = CanonicalNodeIdentity {
            packed: canonical_packed,
            structural: canonical_structural,
            symmetry: canonical_symmetry,
        };
        let used_cached_fingerprint =
            base_symmetry == Symmetry::Identity && canonical_symmetry == Symmetry::Identity;
        CanonicalNodeProbe {
            node: canonical,
            fingerprint: if used_cached_fingerprint {
                packed.fingerprint()
            } else {
                canonical.packed.fingerprint()
            },
            used_cached_fingerprint,
        }
    }

    pub(super) fn canonicalize_packed_identity(
        &mut self,
        packed: PackedNodeKey,
    ) -> CanonicalNodeProbe {
        if packed.level == 0 {
            let structural = CanonicalStructKey::leaf(packed.children[0] != NodeId::ZERO);
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

    pub(in crate::hashlife) fn canonicalize_packed_under_symmetry(
        &mut self,
        packed: PackedNodeKey,
        base_symmetry: Symmetry,
    ) -> CanonicalNodeProbe {
        if base_symmetry == Symmetry::Identity {
            return self.canonicalize_packed_identity(packed);
        }
        if packed.level == 0 {
            let structural = CanonicalStructKey::leaf(packed.children[0] != NodeId::ZERO);
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

    pub(in crate::hashlife) fn canonicalize_packed_key_for_snapshot(
        &mut self,
        packed: PackedNodeKey,
    ) -> CanonicalNodeProbe {
        self.canonicalize_packed_identity(packed)
    }

    pub(in crate::hashlife) fn canonicalize_packed_nodes_batch<const N: usize>(
        &mut self,
        nodes: &[NodeId; N],
        active_lanes: usize,
    ) -> [CanonicalNodeProbe; N] {
        let mut canonical = [CanonicalNodeProbe {
            node: CanonicalNodeIdentity {
                packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
                structural: CanonicalStructKey::leaf(false),
                symmetry: Symmetry::Identity,
            },
            fingerprint: 0,
            used_cached_fingerprint: true,
        }; N];
        let mut miss_indices = [usize::MAX; N];
        let mut duplicate_source = [usize::MAX; N];
        let mut miss_count = 0;

        self.stats.simd.canonical_batch_lanes += active_lanes;
        self.stats.simd.canonical_batch_batches += usize::from(active_lanes != 0);

        for lane in 0..active_lanes {
            let node = nodes[lane];
            if self.node_columns.level(node) == 0 {
                let (packed, fingerprint) = self.node_columns.packed_key_and_fingerprint(node);
                canonical[lane] = CanonicalNodeProbe {
                    node: CanonicalNodeIdentity {
                        packed,
                        structural: CanonicalStructKey::leaf(packed.children[0] != NodeId::ZERO),
                        symmetry: Symmetry::Identity,
                    },
                    fingerprint,
                    used_cached_fingerprint: true,
                };
                continue;
            }
            if let Some(cached) = self.canonical_caches.node.get(&node) {
                self.stats.canonical_cache.canonical_node_cache_hits += 1;
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
            self.stats.canonical_cache.canonical_node_cache_misses += 1;
            if let Some(&source_lane) = miss_indices[..miss_count]
                .iter()
                .find(|&&source_lane| nodes[source_lane] == node)
            {
                duplicate_source[lane] = source_lane;
                continue;
            }
            miss_indices[miss_count] = lane;
            miss_count += 1;
        }

        if miss_count != 0 {
            for &lane in &miss_indices[..miss_count] {
                canonical[lane] =
                    self.canonicalize_packed_identity(self.node_columns.packed_key(nodes[lane]));
                let node = nodes[lane];
                self.publish_optional_cache(
                    |engine| &engine.canonical_caches.node,
                    |engine| &mut engine.canonical_caches.node,
                    node,
                    ProbeKey::fingerprint(&node),
                    canonical[lane].node,
                );
            }
            for lane in 0..active_lanes {
                let source_lane = duplicate_source[lane];
                if source_lane != usize::MAX {
                    canonical[lane] = canonical[source_lane];
                }
            }
        }

        canonical
    }

    pub(super) fn materialize_packed_node_key_internal(&mut self, packed: PackedNodeKey) -> NodeId {
        if packed.level == 0 {
            return if packed.children[0] == NodeId::ZERO {
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

    pub(in crate::hashlife) fn materialize_packed_node_key(
        &mut self,
        packed: PackedNodeKey,
    ) -> NodeId {
        if let Some(node) = self.result_caches.materialized_packed.get(&packed) {
            return node;
        }
        self.stats.transform.packed_cache_result_materializations += 1;
        let node = self.materialize_packed_node_key_internal(packed);
        self.publish_optional_cache(
            |engine| &engine.result_caches.materialized_packed,
            |engine| &mut engine.result_caches.materialized_packed,
            packed,
            packed.fingerprint(),
            node,
        );
        node
    }

    pub(in crate::hashlife) fn canonicalize_packed_node(
        &mut self,
        node: NodeId,
    ) -> CanonicalNodeProbe {
        self.canonicalize_packed_nodes_batch(&[node], 1)[0]
    }

    pub(in crate::hashlife) fn should_symmetry_canonicalize_jump_node(&self, node: NodeId) -> bool {
        #[cfg(test)]
        if let Some((max_level, max_population)) = self.symmetry_gate_override {
            return self.node_columns.level(node) <= max_level
                && self.node_columns.population(node) <= u128::from(max_population);
        }
        self.node_columns.level(node) <= JUMP_SYMMETRY_MAX_LEVEL
            && self.node_columns.population(node) <= u128::from(JUMP_SYMMETRY_MAX_POPULATION)
    }
}
