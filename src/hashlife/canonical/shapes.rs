use super::*;

impl HashLifeEngine {
    pub(in crate::hashlife) fn canonical_parent_key(
        &self,
        level: u32,
        children: [CanonicalNodeRef; 4],
    ) -> CanonicalStructKey {
        let child_fingerprints =
            children.map(|child| self.canonical_caches.shapes[child.index()].key.fingerprint);
        CanonicalStructKey {
            level,
            children,
            fingerprint: structural_node_fingerprint(level, child_fingerprints),
        }
    }

    fn canonical_prefix(&self, shape: CanonicalNodeRef) -> SemanticPrefix {
        self.canonical_caches.shapes[shape.index()].prefix
    }

    pub(super) fn canonical_key_prefix(&self, key: CanonicalStructKey) -> SemanticPrefix {
        if key.level == 0 {
            SemanticPrefix::leaf(key.children[0] != CanonicalShapeId::DEAD)
        } else {
            SemanticPrefix::parent(key.children.map(|child| self.canonical_prefix(child)))
        }
    }

    fn compare_canonical_shapes(
        &self,
        left: CanonicalNodeRef,
        right: CanonicalNodeRef,
    ) -> std::cmp::Ordering {
        const MAX_COMPARE_STACK: usize = 128;
        let mut stack = [(left, right, 0_usize); MAX_COMPARE_STACK];
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (left, right, child_index) = stack[stack_len];
            if left == right {
                continue;
            }
            let left_key = self.canonical_caches.shapes[left.index()].key;
            let right_key = self.canonical_caches.shapes[right.index()].key;
            let ordering = left_key.level.cmp(&right_key.level);
            if ordering != std::cmp::Ordering::Equal {
                return ordering;
            }
            if left_key.level == 0 {
                return left_key.children[0].raw().cmp(&right_key.children[0].raw());
            }
            if child_index == 4 {
                continue;
            }
            if stack_len + 2 > stack.len() {
                crate::invariant_failure!(
                    "validated canonical comparison depth exceeded workspace"
                );
            }
            stack[stack_len] = (left, right, child_index + 1);
            stack_len += 1;
            let left_child = left_key.children[child_index];
            let right_child = right_key.children[child_index];
            if left_child != right_child {
                stack[stack_len] = (left_child, right_child, 0);
                stack_len += 1;
            }
        }
        std::cmp::Ordering::Equal
    }

    pub(in crate::hashlife) fn compare_canonical_keys(
        &self,
        left: CanonicalStructKey,
        right: CanonicalStructKey,
    ) -> std::cmp::Ordering {
        let level_ordering = left.level.cmp(&right.level);
        if level_ordering != std::cmp::Ordering::Equal {
            return level_ordering;
        }
        if left.level == 0 {
            return left.children[0].raw().cmp(&right.children[0].raw());
        }
        for index in 0..4 {
            let ordering =
                self.compare_canonical_shapes(left.children[index], right.children[index]);
            if ordering != std::cmp::Ordering::Equal {
                return ordering;
            }
        }
        std::cmp::Ordering::Equal
    }

    pub(in crate::hashlife) fn intern_canonical_shape(
        &mut self,
        structural: CanonicalStructKey,
    ) -> CanonicalNodeRef {
        if let Some(existing) = self.canonical_caches.shape_intern.get(&structural) {
            return existing;
        }
        if self.canonical_caches.shapes.len() >= self.id_capacity.canonical_count {
            self.reject_canonical_reference_exhaustion();
            return CanonicalShapeId::DEAD;
        }
        if !self.prepare_mandatory_shape_growth() {
            return CanonicalShapeId::DEAD;
        }
        let Ok(id) = CanonicalNodeRef::try_from(self.canonical_caches.shapes.len()) else {
            self.reject_canonical_reference_exhaustion();
            return CanonicalShapeId::DEAD;
        };
        let prefix = self.canonical_key_prefix(structural);
        self.canonical_caches.shapes.push(CanonicalShapeMeta {
            key: structural,
            prefix,
            stabilizer: if structural.level == 0 { u8::MAX } else { 1 },
        });
        if self
            .canonical_caches
            .shape_intern
            .try_insert(structural, id)
            .is_err()
        {
            self.canonical_caches.shapes.pop();
            self.reject_allocation(u128::MAX);
            return CanonicalShapeId::DEAD;
        }
        id
    }

    pub(in crate::hashlife) fn rebuild_canonical_shapes(&mut self) {
        self.prepare_future_for_shape_rebuild();
        self.canonical_caches.shape_epoch = self
            .canonical_caches
            .shape_epoch
            .checked_add(1)
            .or_invariant("canonical shape registry epoch overflow");
        self.canonical_caches.shape_intern.clear();
        self.canonical_caches.shapes.clear();
        self.canonical_caches.symmetry_refs.clear();
        let dead_shape = self.intern_canonical_shape(CanonicalStructKey::leaf(false));
        let live_shape = self.intern_canonical_shape(CanonicalStructKey::leaf(true));
        debug_assert_eq!(
            (dead_shape, live_shape),
            (CanonicalShapeId::DEAD, CanonicalShapeId::LIVE)
        );
        for index in 0..self.node_columns.len() {
            let node =
                NodeId::try_from(index).or_invariant("HashLife node arena exceeded u32 capacity");
            let identity_ref = self.build_node_identity_ref(
                self.node_columns.level(node),
                self.node_columns.quadrants(node),
                self.node_columns.population(node),
            );
            self.node_columns.set_identity_ref(node, identity_ref);
        }
    }

    pub(in crate::hashlife) fn build_node_identity_ref(
        &mut self,
        level: u32,
        children: [NodeId; 4],
        population: u128,
    ) -> CanonicalNodeRef {
        if level == 0 {
            return if population != 0 {
                CanonicalShapeId::LIVE
            } else {
                CanonicalShapeId::DEAD
            };
        }
        let order_children = children.map(|child| self.node_columns.identity_ref(child));
        let structural = self.canonical_parent_key(level, order_children);
        self.intern_canonical_shape(structural)
    }

    pub(super) fn symmetry_canonical_ref(
        &mut self,
        node: NodeId,
        symmetry: Symmetry,
    ) -> CanonicalNodeRef {
        let shape = self.node_columns.identity_ref(node);
        self.resolve_shape_orientations(shape, 1 << (symmetry as u8))
            .map(|record| record.reference(symmetry))
            .unwrap_or(CanonicalShapeId::DEAD)
    }

    pub(in crate::hashlife) fn symmetry_entry(
        &mut self,
        node: NodeId,
        symmetry: Symmetry,
    ) -> PackedTransformOrderEntry {
        let level = self.node_columns.level(node);
        if level == 0 {
            return PackedTransformOrderEntry {
                structural: CanonicalStructKey::leaf(self.node_columns.population(node) != 0),
                aliases: u8::MAX,
                stabilizer: u8::MAX,
            };
        }
        let children = self.node_columns.quadrants(node);
        let permutation = symmetry.quadrant_perm();
        let order_children = permutation
            .map(|child_index| self.symmetry_canonical_ref(children[child_index], symmetry));
        PackedTransformOrderEntry {
            structural: self.canonical_parent_key(level, order_children),
            aliases: 1 << (symmetry as u8),
            stabilizer: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_orientation_record_cannot_alias_reused_shape_id() {
        let mut engine = HashLifeEngine::default();
        let old_key = level_one_shapes(&mut engine)[8];
        let old_shape = engine.intern_canonical_shape(old_key);
        let stale = engine
            .resolve_shape_orientations(old_shape, u8::MAX)
            .or_invariant("old orientation orbit must resolve");
        let epoch = engine.canonical_caches.shape_epoch;

        engine.rebuild_canonical_shapes();
        assert_eq!(engine.canonical_caches.shape_epoch, epoch + 1);
        assert_eq!(engine.canonical_caches.symmetry_refs.len(), 0);
        assert_eq!(
            engine.canonical_caches.shapes.len(),
            2,
            "cached orientation results must not keep orphan shapes alive"
        );
        let new_key = level_one_shapes(&mut engine)[12];
        let new_shape = engine.intern_canonical_shape(new_key);
        assert_eq!(old_shape, new_shape, "fixture must reuse the exact old ID");

        // Deliberately simulate a missed cache invalidation. Neither the key
        // nor any result handle in this record belongs to the current registry.
        engine
            .canonical_caches
            .symmetry_refs
            .try_insert(new_shape, stale)
            .or_invariant("inject retained stale cache record");
        let hits = engine.stats.transform.orientation_cache_hits;
        let actual = engine
            .resolve_shape_orientations(new_shape, u8::MAX)
            .or_invariant("stale cache data must not prevent exact resolution");
        assert_eq!(engine.stats.transform.orientation_cache_hits, hits);

        let mut independent = HashLifeEngine::default();
        let key = level_one_shapes(&mut independent)[12];
        let shape = independent.intern_canonical_shape(key);
        let expected = independent
            .resolve_shape_orientations(shape, u8::MAX)
            .or_invariant("independent orientation orbit");
        for transform in Symmetry::ALL {
            assert_eq!(
                engine.canonical_prefix(actual.reference(transform)),
                independent.canonical_prefix(expected.reference(transform)),
                "stale epoch changed the literal 2x2 state for {transform:?}"
            );
        }
    }

    fn level_one_shapes(engine: &mut HashLifeEngine) -> Vec<CanonicalStructKey> {
        let mut shapes = Vec::with_capacity(16);
        for bits in 0..16_u8 {
            let children: [CanonicalNodeRef; 4] = std::array::from_fn(|index| {
                if bits & (1 << (3 - index)) == 0 {
                    CanonicalShapeId::DEAD
                } else {
                    CanonicalShapeId::LIVE
                }
            });
            shapes.push(engine.canonical_parent_key(1, children));
        }
        shapes
    }

    #[test]
    fn literal_prefix_order_agrees_with_exact_order_for_small_dags() {
        let mut engine = HashLifeEngine::default();
        let shapes = level_one_shapes(&mut engine);
        for &left in &shapes {
            for &right in &shapes {
                let left_prefix = engine.canonical_key_prefix(left);
                let right_prefix = engine.canonical_key_prefix(right);
                let exact = engine.compare_canonical_keys(left, right);
                let prefix = left_prefix.words.cmp(&right_prefix.words);
                assert_eq!(
                    prefix, exact,
                    "literal leaf-order prefix disagreed with exact structural order: left={left:?} right={right:?}"
                );
                assert!(left_prefix.complete && right_prefix.complete);
            }
        }
    }

    #[test]
    fn equal_incomplete_prefixes_require_exact_suffix_comparison() {
        let mut engine = HashLifeEngine::default();
        let dead = CanonicalShapeId::DEAD;
        let live = CanonicalShapeId::LIVE;
        let level1_dead = engine.intern_canonical_shape(engine.canonical_parent_key(1, [dead; 4]));
        let level1_live = engine.intern_canonical_shape(engine.canonical_parent_key(1, [live; 4]));
        let level2_dead =
            engine.intern_canonical_shape(engine.canonical_parent_key(2, [level1_dead; 4]));
        let level2_live =
            engine.intern_canonical_shape(engine.canonical_parent_key(2, [level1_live; 4]));
        let level3_dead =
            engine.intern_canonical_shape(engine.canonical_parent_key(3, [level2_dead; 4]));
        let level3_live =
            engine.intern_canonical_shape(engine.canonical_parent_key(3, [level2_live; 4]));

        let left =
            engine.canonical_parent_key(4, [level3_dead, level3_dead, level3_dead, level3_dead]);
        let right =
            engine.canonical_parent_key(4, [level3_dead, level3_dead, level3_live, level3_dead]);
        let left_prefix = engine.canonical_key_prefix(left);
        let right_prefix = engine.canonical_key_prefix(right);

        assert_eq!(left_prefix.words, right_prefix.words);
        assert!(!left_prefix.complete && !right_prefix.complete);
        assert_eq!(
            engine.compare_canonical_keys(left, right),
            std::cmp::Ordering::Less,
            "equal fixed prefixes must fall back to suffix-aware exact comparison"
        );
    }

    #[test]
    fn fingerprint_numeric_order_never_defines_structural_order() {
        let mut engine = HashLifeEngine::default();
        let shapes = level_one_shapes(&mut engine);
        let opposed = shapes.iter().copied().find_map(|left| {
            shapes
                .iter()
                .copied()
                .find(|&right| {
                    engine.compare_canonical_keys(left, right) == std::cmp::Ordering::Less
                        && left.fingerprint.words() > right.fingerprint.words()
                })
                .map(|right| (left, right))
        });
        let (left, right) = opposed.or_invariant(
            "adversarial small-DAG corpus should oppose fingerprint and structural order",
        );
        assert_ne!(left.fingerprint, right.fingerprint);
        assert_eq!(
            engine.compare_canonical_keys(left, right),
            std::cmp::Ordering::Less,
            "fingerprint magnitude must not influence canonical winner ordering"
        );
    }

    #[test]
    fn canonical_shape_ids_follow_metadata_length_after_intern_tombstones() {
        let mut engine = HashLifeEngine::default();
        let dead = CanonicalShapeId::DEAD;
        let live = CanonicalShapeId::LIVE;
        let removed_key = engine.canonical_parent_key(1, [live, dead, dead, dead]);
        let retained_key = engine.canonical_parent_key(1, [dead, live, dead, dead]);
        let removed = engine.intern_canonical_shape(removed_key);
        let retained = engine.intern_canonical_shape(retained_key);
        let metadata_len = engine.canonical_caches.shapes.len();

        engine
            .canonical_caches
            .shape_intern
            .retain_for_gc(|key, _| key != removed_key);
        assert!(
            engine
                .canonical_caches
                .shape_intern
                .get(&removed_key)
                .is_none()
        );
        assert_eq!(
            engine.canonical_caches.shapes[removed.index()].key,
            removed_key
        );

        let appended_key = engine.canonical_parent_key(1, [dead, dead, live, dead]);
        let appended = engine.intern_canonical_shape(appended_key);

        assert_eq!(appended.index(), metadata_len);
        assert_ne!(appended, retained);
        assert_eq!(
            engine.canonical_caches.shapes[retained.index()].key,
            retained_key
        );
        assert_eq!(
            engine.canonical_caches.shapes[appended.index()].key,
            appended_key
        );
    }

    #[test]
    fn fully_symmetric_parent_reports_all_aliases_and_automorphisms() {
        let mut engine = HashLifeEngine::default();
        let root = engine.join(
            engine.live_leaf,
            engine.live_leaf,
            engine.live_leaf,
            engine.live_leaf,
        );
        let packed = engine.node_columns.packed_key(root);
        let (winner, entry, _) =
            engine.scan_canonical_transform_winner(packed, Symmetry::Identity, false);
        assert_eq!(winner, Symmetry::Identity);
        assert_eq!(entry.aliases, u8::MAX);
        assert_eq!(entry.stabilizer, u8::MAX);
    }

    #[test]
    fn d4_winner_uses_exact_order_and_lowest_symmetry_for_every_level_one_shape() {
        let mut engine = HashLifeEngine::default();
        for bits in 0..16_u8 {
            let children: [NodeId; 4] = std::array::from_fn(|index| {
                if bits & (1 << (3 - index)) == 0 {
                    engine.dead_leaf
                } else {
                    engine.live_leaf
                }
            });
            let root = engine.join(children[0], children[1], children[2], children[3]);
            let packed = engine.node_columns.packed_key(root);
            for base in Symmetry::ALL {
                let entries = Symmetry::ALL
                    .map(|symmetry| engine.symmetry_entry(root, base.then(symmetry)).structural);
                let expected_index = (1..Symmetry::ALL.len()).fold(0, |winner, candidate| {
                    if engine.compare_canonical_keys(entries[candidate], entries[winner])
                        == std::cmp::Ordering::Less
                    {
                        candidate
                    } else {
                        winner
                    }
                });
                let expected_aliases =
                    entries.iter().enumerate().fold(0_u8, |mask, (index, key)| {
                        if engine.compare_canonical_keys(*key, entries[expected_index])
                            == std::cmp::Ordering::Equal
                        {
                            mask | (1 << index)
                        } else {
                            mask
                        }
                    });

                let (winner, entry, _) =
                    engine.scan_canonical_transform_winner(packed, base, false);
                let expected_stabilizer =
                    Symmetry::ALL
                        .iter()
                        .enumerate()
                        .fold(0_u8, |mask, (index, symmetry)| {
                            // Level-one children are leaves. Apply the group action
                            // directly to the winner, independently of alias composition.
                            let winner_children = entries[expected_index].children;
                            let transformed =
                                symmetry.quadrant_perm().map(|child| winner_children[child]);
                            if transformed == winner_children {
                                mask | (1 << index)
                            } else {
                                mask
                            }
                        });
                assert_eq!(
                    winner,
                    Symmetry::ALL[expected_index],
                    "bits={bits:04b} base={base:?}"
                );
                assert_eq!(
                    entry.aliases, expected_aliases,
                    "bits={bits:04b} base={base:?}"
                );
                assert_eq!(
                    entry.stabilizer, expected_stabilizer,
                    "bits={bits:04b} base={base:?}"
                );
                assert_eq!(
                    winner as usize,
                    expected_aliases.trailing_zeros() as usize,
                    "an exact tie must choose the lowest D4 symmetry"
                );
            }
        }
    }

    #[test]
    fn exact_automorphism_reuses_original_packed_shape_without_transform_growth() {
        let mut engine = HashLifeEngine::default();
        let root = engine.join(
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.live_leaf,
        );
        let packed = engine.node_columns.packed_key(root);
        let transform_nodes_before = engine.transform_state.nodes.len();

        let canonical =
            engine.canonicalize_packed_direct(packed, Symmetry::MirrorXRotate270, false);

        assert_eq!(canonical.node.packed, packed);
        assert_eq!(canonical.node.symmetry, Symmetry::Identity);
        assert_eq!(
            engine.transform_state.nodes.len(),
            transform_nodes_before,
            "an exact non-identity automorphism must not build a transform DAG"
        );
    }

    #[test]
    fn known_stabilizer_reuses_identity_ref_without_symmetry_cache_growth() {
        let mut engine = HashLifeEngine::default();
        let root = engine.join(
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.live_leaf,
        );
        let packed = engine.node_columns.packed_key(root);
        let identity_ref = engine.node_columns.identity_ref(root);
        let _ = engine.scan_canonical_transform_winner(packed, Symmetry::Identity, false);
        let entries_before = engine.canonical_caches.symmetry_refs.len();

        let transformed_ref = engine.symmetry_canonical_ref(root, Symmetry::MirrorXRotate270);

        assert_eq!(transformed_ref, identity_ref);
        assert_eq!(
            engine.canonical_caches.symmetry_refs.len(),
            entries_before,
            "an exact known automorphism must not allocate a symmetry-ref cache entry"
        );
    }

    #[test]
    fn non_alias_winner_builds_transform_only_once() {
        let mut engine = HashLifeEngine::default();
        let root = engine.join(
            engine.live_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
        );
        let packed = engine.node_columns.packed_key(root);
        let misses_before = engine.stats.transform.packed_recursive_transform_misses;
        let hits_before = engine.stats.transform.packed_recursive_transform_hits;

        let canonical = engine.canonicalize_packed_direct(packed, Symmetry::Identity, false);

        assert_eq!(canonical.node.symmetry, Symmetry::Rotate180);
        assert_eq!(
            canonical.node.packed.children,
            [
                engine.dead_leaf,
                engine.dead_leaf,
                engine.dead_leaf,
                engine.live_leaf,
            ]
        );
        assert_eq!(
            engine.stats.transform.packed_recursive_transform_misses - misses_before,
            1
        );
        assert_eq!(
            engine.stats.transform.packed_recursive_transform_hits - hits_before,
            0,
            "winner discovery must not rebuild the same transform through a cache hit"
        );
    }
}
