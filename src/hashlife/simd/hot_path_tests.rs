use super::*;

#[test]
fn cached_overlap_batches_need_no_temporary_heap_budget() {
    let mut engine = HashLifeEngine::default();
    let node = engine.empty(3);
    let identity = CanonicalNodeIdentity {
        packed: engine.node_columns.packed_key(node),
        structural: engine.symmetry_entry(node, Symmetry::Identity).structural,
        symmetry: Symmetry::Identity,
    };
    let identities = [identity; SIMD_BATCH_LANES];
    let fingerprints = [identity.structural.fingerprint(); SIMD_BATCH_LANES];
    let expected = engine
        .probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, SIMD_BATCH_LANES)
        .or_invariant("warm overlap batch should resolve");
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    engine.begin_allocation_transaction(retained);
    for batch in 0..64 {
        let actual = engine.probe_and_build_canonical_overlaps_staged(
            &identities,
            &fingerprints,
            SIMD_BATCH_LANES,
        );
        assert_eq!(
            actual,
            Some(expected),
            "cache-only batch {batch} required allocation"
        );
        assert_eq!(
            engine.allocation_transient_reserved, 0,
            "batch {batch} retained scratch charges"
        );
    }
    assert_eq!(engine.take_allocation_failure(), None);
}

#[test]
fn repeated_jump_result_batches_release_their_scratch_budget() {
    const LANES: usize = SIMD_BATCH_LANES * 9;
    let mut engine = HashLifeEngine::default();
    let source = engine.empty(3);
    let result = engine.empty(2);
    engine.insert_jump_result((source, 0), result);
    assert_eq!(
        engine.jump_result_batch([source; LANES], 0),
        [result; LANES]
    );
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    let scratch = ProbeTable::<JumpQuery, usize>::allocation_bytes_for_capacity(LANES)
        .or_invariant("bounded query table capacity")
        + ProbeTable::<PackedSymmetryKey, usize>::allocation_bytes_for_capacity(4)
            .or_invariant("bounded orientation table capacity");
    engine.begin_allocation_transaction(retained + scratch);
    for batch in 0..64 {
        assert_eq!(
            engine.jump_result_batch([source; LANES], 0),
            [result; LANES],
            "result lookup failed in batch {batch}"
        );
        assert_eq!(
            engine.allocation_transient_reserved, 0,
            "completed result batch {batch} retained scratch charges"
        );
    }
    assert_eq!(engine.take_allocation_failure(), None);
}

#[test]
fn control_match_groups_are_recorded_by_the_control_kernel() {
    let mut engine = HashLifeEngine::default();
    let (matches, accounting) = crate::hashlife::kernels::KernelSet::selected().control_matches(
        &[7; 16],
        &[7; SIMD_BATCH_LANES],
        SIMD_BATCH_LANES,
    );
    assert_eq!(matches, [u16::MAX; SIMD_BATCH_LANES]);
    engine.record_kernel_accounting(accounting);
    let stats = &engine.stats.simd.kernel;
    assert_eq!(stats.swar_control_groups, accounting.swar_control_groups);
    assert_eq!(
        stats.native_avx2_control_groups,
        accounting.native_avx2_control_groups
    );
    assert_eq!(
        stats.native_neon_control_groups,
        accounting.native_neon_control_groups
    );
    assert!(
        stats.swar_control_groups
            + stats.native_avx2_control_groups
            + stats.native_neon_control_groups
            > 0,
        "an executed control group must be credited to its actual kernel"
    );
    assert_eq!(stats.native_d4_exact_winner_lanes, 0);
}
