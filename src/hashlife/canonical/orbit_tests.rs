use super::*;

fn assert_orientation_accounting(engine: &HashLifeEngine) {
    let stats = engine.stats.transform;
    assert_eq!(
        stats.orientation_requests,
        stats.orientation_quotient_eliminations
            + stats.orientation_cache_hits
            + stats.orientation_uncached_requests,
        "orientation requests must have disjoint dispositions: {stats:?}"
    );
    assert_eq!(
        stats.d4_candidate_requests,
        stats.d4_duplicate_candidates + stats.d4_unique_candidates,
        "completed parent candidate sets must conserve all eight transforms: {stats:?}"
    );
}

#[test]
fn request_accounting_conserves_partial_fallback_and_cached_proofs() {
    for work_limit in [0, 3, usize::MAX] {
        let mut engine = HashLifeEngine::default();
        let root = node_from_bits(&mut engine, 0x0173);
        let shape = engine.node_columns.identity_ref(root);
        for mask in [0x02, 0x50, 0xff, 0xff] {
            let record = engine
                .resolve_shape_orientations_with_quotient_limit(shape, mask, work_limit)
                .or_invariant("partial orientation requests must resolve");
            assert_eq!(record.resolved & mask, mask);
            assert_orientation_accounting(&engine);
        }
        assert!(
            engine.stats.transform.orientation_cache_hits > 0,
            "fixture must reuse persistent proofs for quota={work_limit}"
        );
        if work_limit == 0 {
            assert!(engine.stats.transform.orientation_work_bypasses > 0);
        }
    }
}

#[test]
fn quotient_reduces_cold_symmetric_work_against_recorded_eight_lane_baseline() {
    // Before quotienting: 8 candidate + 8 prefix lanes per fixture, with
    // respectively 7, 28, 28, 28 mandatory (node, transform) cache entries.
    for (bits, old_entries, distinct) in
        [(0, 7, 1), (0x0660, 28, 1), (0x9009, 28, 1), (0x0173, 28, 8)]
    {
        let mut engine = HashLifeEngine::default();
        let root = node_from_bits(&mut engine, bits);
        let packed = engine.node_columns.packed_key(root);
        let shapes = engine.canonical_caches.shapes.len();
        engine.canonicalize_packed_direct(packed, Symmetry::Identity, true);
        let kernel = engine.stats.simd.kernel;
        assert_eq!(
            kernel.d4_candidate_lanes, 8,
            "cold signatures must be counted bits={bits:#06x}"
        );
        assert_eq!(engine.stats.transform.d4_unique_candidates, distinct);
        assert_eq!(
            kernel.candidate_lanes,
            8 + distinct,
            "candidate and prefix work bits={bits:#06x}"
        );
        assert!(
            engine.canonical_caches.symmetry_refs.len() < old_entries,
            "shape records must replace repeated transform entries bits={bits:#06x}"
        );
        assert_eq!(engine.stats.transform.d4_exact_comparator_calls, 0);
        assert_orientation_accounting(&engine);
        eprintln!(
            "D4 quotient bits={bits:#06x} old_lanes=16 lanes={} shapes_added={} orientation_records={} retained_bytes={} requests={} eliminated={}",
            kernel.candidate_lanes,
            engine.canonical_caches.shapes.len() - shapes,
            engine.canonical_caches.symmetry_refs.len(),
            engine.allocated_bytes(),
            engine.stats.transform.orientation_requests,
            engine.stats.transform.orientation_quotient_eliminations
        );
        // Losing the parent-result cache must not lose the proven automorphisms.
        engine.canonical_caches.direct_parent.release_storage();
        engine.canonical_caches.hot_direct_parent.release_storage();
        let signatures = engine.stats.transform.orientation_resolved_signatures;
        let scans = engine.stats.transform.d4_semantic_prefix_attempts;
        engine.canonicalize_packed_direct(packed, Symmetry::Identity, true);
        assert_eq!(
            engine.stats.transform.d4_semantic_prefix_attempts,
            scans + 1
        );
        assert_eq!(
            engine.stats.simd.kernel.d4_candidate_lanes - kernel.d4_candidate_lanes,
            distinct,
            "known classes must be pruned before construction bits={bits:#06x}"
        );
        assert_eq!(
            engine.stats.simd.kernel.candidate_lanes - kernel.candidate_lanes,
            distinct * 2
        );
        assert_eq!(
            engine.stats.transform.orientation_resolved_signatures,
            signatures
        );
        assert_orientation_accounting(&engine);
    }
}

fn node_from_bits(engine: &mut HashLifeEngine, bits: u16) -> NodeId {
    let children: [NodeId; 4] = std::array::from_fn(|quadrant| {
        let x = (quadrant % 2) * 2;
        let y = (quadrant / 2) * 2;
        let leaves: [NodeId; 4] = std::array::from_fn(|child| {
            if bits & (1 << ((y + child / 2) * 4 + x + child % 2)) == 0 {
                engine.dead_leaf
            } else {
                engine.live_leaf
            }
        });
        engine.join(leaves[0], leaves[1], leaves[2], leaves[3])
    });
    engine.join(children[0], children[1], children[2], children[3])
}

#[test]
fn failed_canonical_proof_cannot_publish_a_sentinel_and_poison_retry() {
    for base in [Symmetry::Identity, Symmetry::Rotate90] {
        let mut engine = HashLifeEngine::default();
        let root = node_from_bits(&mut engine, 0x0173);
        let packed = engine.node_columns.packed_key(root);
        engine.id_capacity.canonical_count = engine.canonical_caches.shapes.len();
        engine.begin_allocation_transaction(u128::MAX);
        let _ = engine.canonicalize_packed_under_symmetry(packed, base);
        assert_orientation_accounting(&engine);
        assert_eq!(
            engine.take_allocation_failure(),
            Some(EngineAllocationFailure::CanonicalReferenceExhausted),
            "base={base:?}"
        );
        assert_eq!(
            engine.canonical_caches.packed.len(),
            0,
            "failed identity proof was cached"
        );
        assert_eq!(
            engine.canonical_caches.oriented.len(),
            0,
            "failed oriented proof was cached"
        );
        assert_eq!(
            engine.canonical_caches.direct_parent.len(),
            0,
            "failed orbit proof was cached"
        );
        engine.id_capacity = EngineIdCapacity::FULL;
        let actual = engine.canonicalize_packed_under_symmetry(packed, base);
        assert_eq!(engine.take_allocation_failure(), None);
        let expected = scalar_winner(orient_bits(0x0173, base));
        let expected_root = node_from_bits(&mut engine, expected.1);
        assert_eq!(
            actual.node.packed,
            engine.node_columns.packed_key(expected_root)
        );
        assert_eq!(actual.node.symmetry, expected.0);
    }
}

fn orient_bits(bits: u16, symmetry: Symmetry) -> u16 {
    let mut transformed = 0;
    for y in 0..4 {
        for x in 0..4 {
            // HashLife orientations are pullbacks: output slots gather from
            // transformed source coordinates, as in `quadrant_perm`.
            let (source_x, source_y) = symmetry.transform_coords(x, y, 3);
            if bits & (1 << (source_y * 4 + source_x)) != 0 {
                transformed |= 1 << (y * 4 + x);
            }
        }
    }
    transformed
}

fn structural_order(bits: u16) -> u16 {
    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        .into_iter()
        .fold(0, |prefix, index| (prefix << 1) | ((bits >> index) & 1))
}

fn scalar_winner(bits: u16) -> (Symmetry, u16) {
    Symmetry::ALL
        .into_iter()
        .map(|symmetry| (symmetry, orient_bits(bits, symmetry)))
        .min_by_key(|(symmetry, transformed)| (structural_order(*transformed), *symmetry as u8))
        .or_invariant("the D4 orbit has eight candidates")
}

#[test]
fn one_scan_backfills_distinct_orientations_with_exact_lowest_transforms() {
    // Empty, full, partially symmetric, and recursively asymmetric children.
    for bits in [0, u16::MAX, 0x0660, 0x9009, 0x1248, 0x0173, 0x962d, 0x842f] {
        for first_base in Symmetry::ALL {
            let mut engine = HashLifeEngine::default();
            let nodes = Symmetry::ALL
                .map(|symmetry| node_from_bits(&mut engine, orient_bits(bits, symmetry)));
            let packed = engine.node_columns.packed_key(nodes[0]);
            let cold = engine.canonicalize_packed_direct(packed, first_base, true);
            let expected_bits = scalar_winner(bits).1;
            let expected_node = node_from_bits(&mut engine, expected_bits);
            let expected_packed = engine.node_columns.packed_key(expected_node);
            assert_eq!(
                cold.node.packed, expected_packed,
                "bits={bits:#06x} base={first_base:?}"
            );
            let scans = engine.stats.transform.d4_semantic_prefix_attempts;
            let transforms = engine.transform_state.nodes.len();
            let shapes = engine.canonical_caches.shapes.len();
            assert_eq!(scans, 1, "one cold search should prove the complete orbit");

            for (index, orientation) in Symmetry::ALL.into_iter().enumerate() {
                let oriented_bits = orient_bits(bits, orientation);
                let expected_stabilizer = Symmetry::ALL.into_iter().fold(0, |mask, symmetry| {
                    if orient_bits(oriented_bits, symmetry) == oriented_bits {
                        mask | (1 << (symmetry as u8))
                    } else {
                        mask
                    }
                });
                let shape = engine.node_columns.identity_ref(nodes[index]);
                assert_eq!(
                    engine.canonical_caches.shapes[shape.index()].stabilizer,
                    expected_stabilizer,
                    "every existing orientation should retain its own automorphisms"
                );
                let physical = engine.node_columns.packed_key(nodes[index]);
                for base in Symmetry::ALL {
                    let input = orient_bits(orient_bits(bits, orientation), base);
                    let expected_transform = scalar_winner(input).0;
                    let result = engine.canonicalize_packed_direct(physical, base, true);
                    assert_eq!(
                        result.node.packed, expected_packed,
                        "bits={bits:#06x} orientation={orientation:?} base={base:?}"
                    );
                    assert_eq!(
                        result.node.symmetry, expected_transform,
                        "lowest exact transform bits={bits:#06x} orientation={orientation:?} base={base:?}"
                    );
                    assert_eq!(
                        result.fingerprint,
                        expected_packed.fingerprint(),
                        "probe fingerprint must describe the returned canonical packed root"
                    );
                    let wrapped = engine.canonicalize_packed_under_symmetry(physical, base);
                    assert_eq!(wrapped.node, result.node);
                    assert_eq!(wrapped.fingerprint, result.fingerprint);
                }
            }
            assert_eq!(
                engine.stats.transform.d4_semantic_prefix_attempts, scans,
                "warm orbit must not search for a winner again"
            );
            assert_eq!(
                engine.transform_state.nodes.len(),
                transforms,
                "warm orbit must not build rotated transform trees"
            );
            assert_eq!(
                engine.canonical_caches.shapes.len(),
                shapes,
                "warm orbit must reuse exact shape identities"
            );
        }
    }
}

#[test]
fn orbit_cache_deduplicates_automorphisms_without_eager_rotated_roots() {
    for bits in [0, u16::MAX, 0x0660, 0x9009, 0x0173] {
        let mut engine = HashLifeEngine::default();
        let root = node_from_bits(&mut engine, bits);
        let packed = engine.node_columns.packed_key(root);
        let mut unique = [0_u16; 8];
        let mut count = 0;
        for symmetry in Symmetry::ALL {
            let oriented = orient_bits(bits, symmetry);
            if !unique[..count].contains(&oriented) {
                unique[count] = oriented;
                count += 1;
            }
        }
        engine.canonicalize_packed_direct(packed, Symmetry::Identity, true);
        assert_eq!(
            engine.canonical_caches.direct_parent.len(),
            count,
            "bits={bits:#06x}: cache only distinct semantic orientations"
        );
        assert!(
            engine
                .stats
                .canonical_fallback
                .canonical_transform_root_reconstructions
                <= 1,
            "backfilling must not materialize all rotated node trees"
        );
    }
}

#[test]
fn rejected_orbit_cache_growth_cannot_fail_an_exact_result() {
    let mut engine = HashLifeEngine::default();
    let root = node_from_bits(&mut engine, 0x0173);
    let packed = engine.node_columns.packed_key(root);
    let expected = engine
        .canonicalize_packed_direct(packed, Symmetry::Identity, true)
        .node;
    engine.canonical_caches.direct_parent.release_storage();
    engine.canonical_caches.hot_direct_parent.release_storage();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    engine.begin_allocation_transaction(retained);
    let actual = engine
        .canonicalize_packed_direct(packed, Symmetry::Identity, true)
        .node;
    assert_eq!(engine.take_allocation_failure(), None);
    assert_eq!(actual, expected);
    assert_eq!(
        engine.canonical_caches.direct_parent.len(),
        0,
        "optional orbit publication should be bypassed without budget headroom"
    );
    assert_eq!(engine.allocated_bytes() as u128, retained);
}

#[test]
fn orbit_results_are_discarded_and_relearned_after_arena_repacking() {
    let mut engine = HashLifeEngine::default();
    for bits in 1..48 {
        node_from_bits(&mut engine, bits);
    }
    let root = node_from_bits(&mut engine, 0x0173);
    let packed = engine.node_columns.packed_key(root);
    engine.canonicalize_packed_direct(packed, Symmetry::Identity, true);
    engine.record_retained_root(root);
    let epoch = engine.arena_epoch;
    assert!(engine.canonical_caches.direct_parent.len() > 0);

    engine.mark_live_nodes();
    engine.compact_marked_nodes();
    assert_eq!(engine.arena_epoch, epoch + 1);
    assert_eq!(engine.canonical_caches.direct_parent.len(), 0);
    assert_eq!(engine.canonical_caches.hot_direct_parent.len(), 0);
    let remapped = engine.retained_roots[0];
    assert!(
        remapped.index() < root.index(),
        "fixture must actually remap the root"
    );
    let packed = engine.node_columns.packed_key(remapped);
    let expected = node_from_bits(&mut engine, scalar_winner(0x0173).1);
    let expected_packed = engine.node_columns.packed_key(expected);
    let scans = engine.stats.transform.d4_semantic_prefix_attempts;
    for base in Symmetry::ALL {
        let result = engine.canonicalize_packed_direct(packed, base, true);
        assert_eq!(result.node.packed, expected_packed, "base={base:?}");
        assert_eq!(
            result.node.symmetry,
            scalar_winner(orient_bits(0x0173, base)).0
        );
    }
    assert_eq!(
        engine.stats.transform.d4_semantic_prefix_attempts,
        scans + 1,
        "a new epoch should need only one fresh orbit search"
    );
}

#[test]
fn retained_oriented_result_hit_does_not_rebuild_discarded_child_metadata() {
    let mut engine = HashLifeEngine::default();
    let root = node_from_bits(&mut engine, 0x0173);
    let packed = engine.node_columns.packed_key(root);
    let expected = engine.canonicalize_packed_under_symmetry(packed, Symmetry::Rotate90);
    engine.canonical_caches.direct_parent.release_storage();
    engine.canonical_caches.hot_direct_parent.release_storage();
    engine.canonical_caches.symmetry_refs.release_storage();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    let scans = engine.stats.transform.d4_semantic_prefix_attempts;
    engine.begin_allocation_transaction(retained);
    for _ in 0..32 {
        let actual = engine.canonicalize_packed_under_symmetry(packed, Symmetry::Rotate90);
        assert_eq!(actual.node, expected.node);
        assert_eq!(actual.fingerprint, expected.fingerprint);
    }
    assert_eq!(
        engine.take_allocation_failure(),
        None,
        "an exact cached answer cannot require fresh child symmetry allocations"
    );
    assert_eq!(engine.stats.transform.d4_semantic_prefix_attempts, scans);
    assert_eq!(engine.canonical_caches.symmetry_refs.len(), 0);
    assert_eq!(engine.canonical_caches.direct_parent.len(), 0);
}
