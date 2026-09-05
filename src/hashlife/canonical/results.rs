use super::*;
use crate::RequiredExt;

#[derive(Clone, Copy)]
struct PackedCanonicalInputRecord {
    input: PackedSymmetryKey,
    input_children: [CanonicalNodeRef; 4],
    canonical_entry: PackedSymmetryKey,
}

impl HashLifeEngine {
    pub(in crate::hashlife) fn record_and_publish_jump_entry(
        &mut self,
        key: CanonicalJumpKey,
        fingerprint: u64,
        entry: PackedSymmetryKey,
    ) {
        if !self.record_active_jump_result(key, fingerprint, entry) {
            return;
        }
        self.publish_future_jump_entry(key, entry);
        self.publish_optional_cache(
            |engine| &engine.result_caches.jump,
            |engine| &mut engine.result_caches.jump,
            key,
            fingerprint,
            entry,
        );
    }

    #[inline]
    fn oriented_packed_result(
        &mut self,
        packed: PackedNodeKey,
        cached_result_symmetry: Symmetry,
        output_symmetry: Symmetry,
    ) -> PackedNodeKey {
        let combined = cached_result_symmetry.inverse().then(output_symmetry);
        if combined == Symmetry::Identity {
            packed
        } else {
            let cache_key = PackedSymmetryKey {
                packed,
                symmetry: combined,
            };
            self.stats.result_cache.oriented_result_cache_lookups += 1;
            if let Some(oriented_packed) = self.result_caches.oriented.get(&cache_key) {
                self.stats.result_cache.oriented_result_cache_hits += 1;
                oriented_packed
            } else {
                self.stats.result_cache.oriented_result_cache_misses += 1;
                self.stats.transform.packed_inverse_transform_hits += 1;
                self.stats
                    .canonical_fallback
                    .oriented_transform_root_reconstructions += 1;
                let transformed = self.transform_packed_node_key(packed, combined);
                let oriented_packed = self.packed_root_from_transform_id(transformed);
                self.publish_optional_cache(
                    |engine| &engine.result_caches.oriented,
                    |engine| &mut engine.result_caches.oriented,
                    cache_key,
                    cache_key.fingerprint(),
                    oriented_packed,
                );
                oriented_packed
            }
        }
    }

    #[inline]
    pub(in crate::hashlife) fn materialize_oriented_packed_result(
        &mut self,
        packed: PackedNodeKey,
        cached_result_symmetry: Symmetry,
        output_symmetry: Symmetry,
    ) -> NodeId {
        let oriented_packed =
            self.oriented_packed_result(packed, cached_result_symmetry, output_symmetry);
        self.materialize_packed_node_key(oriented_packed)
    }

    fn canonicalize_result_under_input_symmetry(
        &mut self,
        result: NodeId,
        input_symmetry: Symmetry,
        symmetry_admitted: bool,
    ) -> CanonicalNodeProbe {
        if !symmetry_admitted {
            return self.canonicalize_blocked_jump_node(result);
        }
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
            self.stats
                .canonical_cache
                .symmetry_aware_result_canonicalization_lookups += 1;
            self.stats.result_cache.canonical_result_insert_bypasses += 1;
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
        unique_inputs: &mut [PackedCanonicalInputRecord],
    ) {
        if unique_inputs.is_empty() {
            return;
        }

        for record in unique_inputs {
            let snapshot = *record;
            let input = snapshot.input;
            if input.packed.level == 0 {
                record.canonical_entry = self.canonical_packed_result_entry(input);
                continue;
            }

            if let Some(cached) = self.lookup_direct_parent_identity(
                input.packed.level,
                input.symmetry,
                snapshot.input_children,
            ) {
                self.stats.canonical_cache.direct_parent_cached_result_hits += 1;
                if input.symmetry == Symmetry::Identity {
                    self.cache_canonical_packed_identity(input.packed, cached);
                } else {
                    self.cache_canonical_oriented_identity(input, cached);
                }
                record.canonical_entry = PackedSymmetryKey {
                    packed: cached.packed,
                    symmetry: cached.symmetry,
                };
                continue;
            }

            if input.symmetry == Symmetry::Identity {
                if let Some(cached) = self.lookup_canonical_packed_identity(input.packed) {
                    self.backfill_direct_parent_identity(
                        input.packed,
                        input.symmetry,
                        snapshot.input_children,
                        cached,
                    );
                    record.canonical_entry = PackedSymmetryKey {
                        packed: cached.packed,
                        symmetry: cached.symmetry,
                    };
                    continue;
                }
            } else if let Some(cached) = self.lookup_canonical_oriented_identity(input) {
                self.backfill_direct_parent_identity(
                    input.packed,
                    input.symmetry,
                    snapshot.input_children,
                    cached,
                );
                record.canonical_entry = PackedSymmetryKey {
                    packed: cached.packed,
                    symmetry: cached.symmetry,
                };
                continue;
            }

            self.stats
                .canonical_fallback
                .canonical_result_unique_parent_shapes += 1;
            self.stats
                .canonical_fallback
                .canonical_result_batch_fallbacks += 1;
            let canonical = self.canonicalize_packed_direct_known_children(
                input.packed,
                input.symmetry,
                snapshot.input_children,
                input.symmetry == Symmetry::Identity,
            );
            let entry = PackedSymmetryKey {
                packed: canonical.node.packed,
                symmetry: canonical.node.symmetry,
            };
            if input.symmetry == Symmetry::Identity {
                self.cache_canonical_packed_identity(input.packed, canonical.node);
            } else {
                self.cache_canonical_oriented_identity(input, canonical.node);
            }
            record.canonical_entry = entry;
        }
    }

    pub(in crate::hashlife) fn canonicalize_phase2_commit_lanes(
        &mut self,
        lanes: &mut [Phase2CommitLane],
    ) {
        if lanes.is_empty() {
            return;
        }
        if lanes.len() > crate::simd_layout::SIMD_BATCH_LANES {
            crate::invariant_failure!("phase-2 canonicalization exceeded SIMD lane workspace");
        }
        let empty_entry = PackedSymmetryKey {
            packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
            symmetry: Symmetry::Identity,
        };
        let empty_record = PackedCanonicalInputRecord {
            input: empty_entry,
            input_children: [CanonicalShapeId::DEAD; 4],
            canonical_entry: empty_entry,
        };
        let mut unique_inputs = [empty_record; crate::simd_layout::SIMD_BATCH_LANES];
        let mut unique_count = 0;
        for lane in lanes.iter_mut() {
            let packed_input = lane.packed_input;
            if !lane.key.symmetry_admitted {
                lane.unique_input_index = usize::MAX;
                lane.canonical_entry = packed_input;
                continue;
            }
            if let Some(index) = unique_inputs[..unique_count]
                .iter()
                .position(|record| record.input == packed_input)
            {
                lane.unique_input_index = index;
                self.stats
                    .canonical_fallback
                    .canonical_result_batch_local_reuses += 1;
            } else {
                lane.unique_input_index = unique_count;
                unique_inputs[unique_count] = PackedCanonicalInputRecord {
                    input: packed_input,
                    input_children: if packed_input.packed.level == 0 {
                        [CanonicalShapeId::DEAD; 4]
                    } else {
                        self.direct_parent_input_children(
                            packed_input.packed,
                            packed_input.symmetry,
                        )
                    },
                    canonical_entry: empty_entry,
                };
                unique_count += 1;
            }
        }
        self.stats.canonical_fallback.canonical_result_unique_inputs += unique_count;

        let phase2_fallbacks_before = self
            .stats
            .canonical_fallback
            .canonical_result_batch_fallbacks;
        self.canonical_packed_result_entries_for_unique_inputs(&mut unique_inputs[..unique_count]);
        self.stats.canonical_fallback.canonical_phase2_fallbacks += self
            .stats
            .canonical_fallback
            .canonical_result_batch_fallbacks
            - phase2_fallbacks_before;
        for lane in lanes.iter_mut() {
            if lane.unique_input_index != usize::MAX {
                lane.canonical_entry = unique_inputs[lane.unique_input_index].canonical_entry;
            }
        }
    }

    pub(in crate::hashlife) fn cached_jump_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let inverse = jump_probe.node.symmetry.inverse();
        self.stats.result_cache.jump_result_cache_lookups += 1;
        let retained_result = self
            .result_caches
            .jump
            .get_with_fingerprint(&jump_probe.key, jump_probe.fingerprint);
        let exact_result = retained_result.or_else(|| {
            self.active_jump_results
                .get_with_fingerprint(&jump_probe.key, jump_probe.fingerprint)
        });
        let future_hit = exact_result.is_none();
        let result = if let Some(result) = exact_result {
            result
        } else {
            self.lookup_future_jump_result(jump_probe.key)?
        };
        if retained_result.is_some() {
            self.stats.result_cache.jump_result_cache_hits += 1;
        }
        if jump_probe.node.symmetry != Symmetry::Identity {
            self.stats.result_cache.symmetric_jump_result_cache_hits += 1;
        }
        let materialized =
            self.materialize_oriented_packed_result(result.packed, result.symmetry, inverse);
        if self.allocation_failed() {
            return None;
        }
        if future_hit {
            self.stats.future.saved_jump_lookups += 1;
        }
        Some(materialized)
    }

    pub(in crate::hashlife) fn insert_jump_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let canonical_result = self.canonicalize_result_under_input_symmetry(
            result,
            jump_probe.node.symmetry,
            jump_probe.key.symmetry_admitted,
        );
        self.record_and_publish_jump_entry(
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
        let packed = self.node_columns.packed_key(result);
        let entry = if key.symmetry_admitted {
            self.canonical_packed_result_entry(PackedSymmetryKey {
                packed,
                symmetry: Symmetry::Identity,
            })
        } else {
            PackedSymmetryKey {
                packed,
                symmetry: Symmetry::Identity,
            }
        };
        let fingerprint = key.fingerprint();
        self.record_and_publish_jump_entry(key, fingerprint, entry);
    }

    pub(in crate::hashlife) fn jump_result(&mut self, key: (NodeId, u32)) -> NodeId {
        self.cached_jump_result(key)
            .or_invariant("missing HashLife jump result")
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(in crate::hashlife) fn cached_root_result(&mut self, key: (NodeId, u32)) -> Option<NodeId> {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        self.stats.result_cache.root_result_cache_lookups += 1;
        let result = if let Some(result) = self
            .result_caches
            .root
            .get_with_fingerprint(&jump_probe.key, jump_probe.fingerprint)
        {
            result
        } else {
            self.stats.result_cache.root_result_cache_misses += 1;
            return None;
        };
        self.stats.result_cache.root_result_cache_hits += 1;
        Some(self.materialize_oriented_packed_result(
            result.packed,
            result.symmetry,
            jump_probe.node.symmetry.inverse(),
        ))
    }

    pub(in crate::hashlife) fn insert_root_result(&mut self, key: (NodeId, u32), result: NodeId) {
        let jump_probe = self.canonical_jump_probe(key);
        self.record_fingerprint_probe(jump_probe.used_cached_fingerprint, 1);
        let canonical_result = self.canonicalize_result_under_input_symmetry(
            result,
            jump_probe.node.symmetry,
            jump_probe.key.symmetry_admitted,
        );
        self.publish_optional_cache(
            |engine| &engine.result_caches.root,
            |engine| &mut engine.result_caches.root,
            jump_probe.key,
            jump_probe.fingerprint,
            PackedSymmetryKey {
                packed: canonical_result.node.packed,
                symmetry: canonical_result.node.symmetry,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase2_canonicalization_respects_partial_lane_slice() {
        let mut engine = HashLifeEngine::default();
        let source = engine.empty(2);
        let result = engine.empty(1);
        let packed_input = PackedSymmetryKey {
            packed: engine.node_columns.packed_key(result),
            symmetry: Symmetry::Identity,
        };
        let key = engine.canonical_jump_probe((source, 0)).key;
        let sentinel_entry = PackedSymmetryKey {
            packed: PackedNodeKey::new(
                0,
                [engine.live_leaf, NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            ),
            symmetry: Symmetry::Rotate180,
        };
        let sentinel = Phase2CommitLane {
            key,
            fallback: result,
            result,
            unique_input_index: 777,
            packed_input,
            canonical_entry: sentinel_entry,
        };
        let mut lanes = [sentinel; 8];
        for lane in &mut lanes[..2] {
            lane.unique_input_index = usize::MAX;
        }

        engine.canonicalize_phase2_commit_lanes(&mut lanes[..2]);

        assert_eq!(lanes[0].canonical_entry, lanes[1].canonical_entry);
        assert_eq!(lanes[0].unique_input_index, 0);
        assert_eq!(lanes[1].unique_input_index, 0);
        for lane in &lanes[2..] {
            assert_eq!(lane.unique_input_index, 777);
            assert_eq!(lane.canonical_entry, sentinel_entry);
        }
    }

    #[test]
    fn phase2_without_admitted_lanes_uses_no_transient_storage() {
        let mut engine = HashLifeEngine::default();
        let packed_input = PackedSymmetryKey {
            packed: PackedNodeKey::new(
                0,
                [engine.live_leaf, NodeId::ZERO, NodeId::ZERO, NodeId::ZERO],
            ),
            symmetry: Symmetry::Rotate90,
        };
        let lane = Phase2CommitLane {
            key: CanonicalJumpKey {
                structural: CanonicalStructKey::leaf(false),
                step_exp: 0,
                symmetry_admitted: false,
            },
            fallback: engine.live_leaf,
            result: engine.live_leaf,
            unique_input_index: 0,
            packed_input,
            canonical_entry: PackedSymmetryKey {
                packed: PackedNodeKey::new(0, [NodeId::ZERO; 4]),
                symmetry: Symmetry::Identity,
            },
        };
        let mut lanes = [lane; crate::simd_layout::SIMD_BATCH_LANES];
        let bytes_before = engine.allocated_bytes();
        let unique_before = engine
            .stats
            .canonical_fallback
            .canonical_result_unique_inputs;

        engine.canonicalize_phase2_commit_lanes(&mut lanes);

        assert_eq!(engine.allocated_bytes(), bytes_before);
        assert_eq!(
            engine
                .stats
                .canonical_fallback
                .canonical_result_unique_inputs,
            unique_before
        );
        assert!(lanes.iter().all(|lane| {
            lane.unique_input_index == usize::MAX && lane.canonical_entry == packed_input
        }));
    }

    #[test]
    fn pressured_optional_jump_publication_preserves_exact_active_result() {
        let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
        engine.begin_allocation_transaction(u128::MAX);
        let source = engine.empty(2);
        let result = engine.empty(1);
        assert_eq!(engine.take_allocation_failure(), None);

        engine.result_caches.jump.release_storage();
        engine.active_jump_results.reset();
        let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);
        engine.insert_jump_result((source, 0), result);

        assert_eq!(
            engine.take_allocation_failure(),
            None,
            "optional retained-cache pressure must not become a mandatory allocation failure"
        );
        assert_eq!(
            engine.result_caches.jump.len(),
            0,
            "the retained jump cache must bypass growth at its hard logical budget"
        );
        assert_eq!(
            engine.cached_jump_result((source, 0)),
            Some(result),
            "the exact scheduler result must remain usable after retained-cache publication is bypassed"
        );
    }
}
