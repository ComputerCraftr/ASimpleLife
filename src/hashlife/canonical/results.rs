use super::*;

#[derive(Clone, Copy)]
struct PackedCanonicalInputRecord {
    input: PackedSymmetryKey,
    input_children: [CanonicalNodeRef; 4],
}

impl HashLifeEngine {
    #[inline]
    pub(in crate::hashlife) fn materialize_oriented_packed_result(
        &mut self,
        packed: PackedNodeKey,
        cached_result_symmetry: Symmetry,
        output_symmetry: Symmetry,
    ) -> NodeId {
        let combined = cached_result_symmetry.inverse().then(output_symmetry);
        if combined == Symmetry::Identity {
            self.materialize_packed_node_key(packed)
        } else {
            let cache_key = PackedSymmetryKey {
                packed,
                symmetry: combined,
            };
            self.stats.oriented_result_cache_lookups += 1;
            if let Some(oriented) = self.oriented_result_cache.get(&cache_key) {
                self.stats.oriented_result_cache_hits += 1;
                oriented
            } else {
                self.stats.oriented_result_cache_misses += 1;
                self.stats.packed_inverse_transform_hits += 1;
                self.stats.oriented_transform_root_reconstructions += 1;
                let transformed = self.transform_packed_node_key(packed, combined);
                let oriented_packed = self.packed_root_from_transform_id(transformed);
                let oriented = self.materialize_packed_node_key(oriented_packed);
                self.oriented_result_cache.insert(cache_key, oriented);
                oriented
            }
        }
    }

    fn canonicalize_result_under_input_symmetry(
        &mut self,
        result: NodeId,
        input_symmetry: Symmetry,
    ) -> CanonicalNodeProbe {
        let result_packed = self.node_columns.packed_key(result);
        self.canonicalize_packed_result_input(PackedSymmetryKey {
            packed: result_packed,
            symmetry: input_symmetry,
        })
    }

    #[inline]
    fn canonicalize_packed_result_input(&mut self, input: PackedSymmetryKey) -> CanonicalNodeProbe {
        if input.symmetry == Symmetry::Identity {
            self.canonicalize_packed_identity(input.packed)
        } else {
            self.stats.symmetry_aware_result_canonicalization_lookups += 1;
            self.stats.canonical_result_insert_bypasses += 1;
            self.canonicalize_packed_under_symmetry(input.packed, input.symmetry)
        }
    }

    #[inline]
    fn canonical_packed_result_entry(&mut self, input: PackedSymmetryKey) -> PackedSymmetryKey {
        let canonical = self.canonicalize_packed_result_input(input);
        PackedSymmetryKey {
            packed: canonical.node.packed,
            symmetry: canonical.node.symmetry,
        }
    }

    fn canonical_packed_result_entries_for_unique_inputs(
        &mut self,
        unique_inputs: &[PackedCanonicalInputRecord],
        unique_entries: &mut [PackedSymmetryKey],
    ) {
        if unique_inputs.is_empty() {
            return;
        }

        let mut parent_shape_lookup =
            FlatTable::<PackedCanonicalBatchKey, usize>::with_capacity(unique_inputs.len().max(4));
        let mut parent_shape_entries = Vec::<PackedSymmetryKey>::with_capacity(unique_inputs.len());

        for (unique_index, record) in unique_inputs.iter().enumerate() {
            let input = record.input;
            if input.packed.level == 0 {
                unique_entries[unique_index] = self.canonical_packed_result_entry(input);
                continue;
            }

            if input.symmetry == Symmetry::Identity {
                if let Some(cached) = self.lookup_canonical_packed_identity(input.packed) {
                    unique_entries[unique_index] = PackedSymmetryKey {
                        packed: cached.packed,
                        symmetry: cached.symmetry,
                    };
                    continue;
                }
            } else if let Some(cached) = self.lookup_canonical_oriented_identity(input) {
                unique_entries[unique_index] = PackedSymmetryKey {
                    packed: cached.packed,
                    symmetry: cached.symmetry,
                };
                continue;
            }

            let batch_key = PackedCanonicalBatchKey {
                level: input.packed.level,
                symmetry: input.symmetry,
                children: record.input_children,
            };
            if let Some(parent_index) = parent_shape_lookup.get(&batch_key) {
                unique_entries[unique_index] = parent_shape_entries[parent_index];
                self.stats.canonical_result_batch_local_reuses += 1;
                continue;
            }

            self.stats.canonical_result_unique_parent_shapes += 1;
            self.stats.canonical_result_batch_fallbacks += 1;
            let canonical = self.canonicalize_packed_direct_known_children(
                input.packed,
                input.symmetry,
                record.input_children,
                input.symmetry == Symmetry::Identity,
            );
            let entry = PackedSymmetryKey {
                packed: canonical.node.packed,
                symmetry: canonical.node.symmetry,
            };
            parent_shape_lookup.insert(batch_key, parent_shape_entries.len());
            parent_shape_entries.push(entry);
            if input.symmetry == Symmetry::Identity {
                self.cache_canonical_packed_identity(input.packed, canonical.node);
            } else {
                self.cache_canonical_oriented_identity(input, canonical.node);
            }
            unique_entries[unique_index] = entry;
        }
    }

    pub(in crate::hashlife) fn canonicalize_phase2_commit_lanes(
        &mut self,
        lanes: &mut [Phase2CommitLane],
    ) {
        if lanes.is_empty() {
            return;
        }
        let mut unique_lookup =
            FlatTable::<PackedSymmetryKey, usize>::with_capacity(lanes.len().max(4));
        let mut unique_inputs = Vec::<PackedCanonicalInputRecord>::with_capacity(lanes.len());
        for lane in lanes.iter_mut() {
            let packed_input = lane.packed_input;
            if let Some(index) = unique_lookup.get(&packed_input) {
                lane.unique_input_index = index;
                self.stats.canonical_result_batch_local_reuses += 1;
            } else {
                lane.unique_input_index = unique_inputs.len();
                unique_lookup.insert(packed_input, unique_inputs.len());
                unique_inputs.push(PackedCanonicalInputRecord {
                    input: packed_input,
                    input_children: if packed_input.packed.level == 0 {
                        [0; 4]
                    } else {
                        self.direct_parent_input_children(
                            packed_input.packed,
                            packed_input.symmetry,
                        )
                    },
                });
            }
        }
        self.stats.canonical_result_unique_inputs += unique_inputs.len();

        let mut unique_entries = vec![
            PackedSymmetryKey {
                packed: PackedNodeKey::new(0, [0; 4]),
                symmetry: Symmetry::Identity,
            };
            unique_inputs.len()
        ];
        let phase2_fallbacks_before = self.stats.canonical_result_batch_fallbacks;
        self.canonical_packed_result_entries_for_unique_inputs(&unique_inputs, &mut unique_entries);
        self.stats.canonical_phase2_fallbacks +=
            self.stats.canonical_result_batch_fallbacks - phase2_fallbacks_before;
        for lane in lanes.iter_mut() {
            lane.canonical_entry = unique_entries[lane.unique_input_index];
        }
    }

    pub(in crate::hashlife) fn cached_jump_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
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
        Some(self.materialize_oriented_packed_result(result.packed, result.symmetry, inverse))
    }

    pub(in crate::hashlife) fn insert_jump_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let canonical_result =
            self.canonicalize_result_under_input_symmetry(result, jump_probe.node.symmetry);
        self.jump_cache.insert_with_fingerprint(
            jump_probe.key,
            jump_probe.fingerprint,
            PackedSymmetryKey {
                packed: canonical_result.node.packed,
                symmetry: canonical_result.node.symmetry,
            },
        );
    }

    pub(in crate::hashlife) fn insert_canonical_jump_result(
        &mut self,
        key: CanonicalJumpKey,
        result: NodeId,
    ) {
        let entry = self.canonical_packed_result_entry(PackedSymmetryKey {
            packed: self.node_columns.packed_key(result),
            symmetry: Symmetry::Identity,
        });
        let fingerprint = hash_packed_jump_fingerprint(key.structural.fingerprint(), key.step_exp);
        self.jump_cache
            .insert_with_fingerprint(key, fingerprint, entry);
    }

    pub(in crate::hashlife) fn jump_result(&mut self, key: (NodeId, u32)) -> NodeId {
        self.cached_jump_result(key)
            .expect("missing HashLife jump result")
    }

    pub(in crate::hashlife) fn cached_root_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        self.stats.root_result_cache_lookups += 1;
        let result = if let Some(result) = self
            .root_result_cache
            .get_with_fingerprint(&jump_probe.key, jump_probe.fingerprint)
        {
            result
        } else {
            self.stats.root_result_cache_misses += 1;
            return None;
        };
        self.stats.root_result_cache_hits += 1;
        Some(self.materialize_oriented_packed_result(
            result.packed,
            result.symmetry,
            jump_probe.node.symmetry.inverse(),
        ))
    }

    pub(in crate::hashlife) fn insert_root_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let canonical_result =
            self.canonicalize_result_under_input_symmetry(result, jump_probe.node.symmetry);
        self.root_result_cache.insert_with_fingerprint(
            jump_probe.key,
            jump_probe.fingerprint,
            PackedSymmetryKey {
                packed: canonical_result.node.packed,
                symmetry: canonical_result.node.symmetry,
            },
        );
    }
}
