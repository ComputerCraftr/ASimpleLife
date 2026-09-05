use super::*;
use crate::hashlife::{EngineAllocationFailure, EngineIdCapacity};

fn symmetry_bit(symmetry: Symmetry) -> u8 {
    1 << (symmetry as u8)
}

fn corner_level_one_shape(engine: &mut HashLifeEngine) -> CanonicalShapeId {
    engine.intern_canonical_shape(engine.canonical_parent_key(
        1,
        [
            CanonicalShapeId::LIVE,
            CanonicalShapeId::DEAD,
            CanonicalShapeId::DEAD,
            CanonicalShapeId::DEAD,
        ],
    ))
}

fn nested_corner_shapes(engine: &mut HashLifeEngine) -> (CanonicalShapeId, CanonicalShapeId) {
    let child = corner_level_one_shape(engine);
    let empty =
        engine.intern_canonical_shape(engine.canonical_parent_key(1, [CanonicalShapeId::DEAD; 4]));
    let parent =
        engine.intern_canonical_shape(engine.canonical_parent_key(2, [child, empty, empty, empty]));
    (child, parent)
}

fn deep_uniform_shape(engine: &mut HashLifeEngine, depth: usize) -> CanonicalShapeId {
    let mut shape = CanonicalShapeId::LIVE;
    for level in 1..=depth {
        let level = u32::try_from(level).or_invariant("fixture depth exceeds u32");
        shape = engine.intern_canonical_shape(engine.canonical_parent_key(level, [shape; 4]));
    }
    shape
}

fn assert_same_record(
    left: orientations::OrientationRecord,
    right: orientations::OrientationRecord,
) {
    assert_eq!(left.resolved, right.resolved);
    for symmetry in Symmetry::ALL {
        if left.resolved & symmetry_bit(symmetry) != 0 {
            assert_eq!(left.reference(symmetry), right.reference(symmetry));
        }
    }
}

#[test]
fn optional_orientation_publication_can_be_denied_with_scratch_reserved() {
    let mut engine = HashLifeEngine::default();
    let shape = corner_level_one_shape(&mut engine);
    let expected = engine
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("cold orientation resolution should succeed");
    assert_eq!(engine.take_allocation_failure(), None);

    engine.canonical_caches.symmetry_refs.release_storage();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    engine.begin_allocation_transaction(retained + orientations::RESOLVER_BYTES);

    let actual = engine
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("prewarmed exact shapes must make the retry allocation-free");

    assert_same_record(actual, expected);
    assert_eq!(engine.take_allocation_failure(), None);
    assert_eq!(engine.canonical_caches.symmetry_refs.len(), 0);
}

#[test]
fn resolver_scratch_exhaustion_uses_exact_non_poisoning_scalar_fallback() {
    let mut engine = HashLifeEngine::default();
    let shape = corner_level_one_shape(&mut engine);
    let expected = engine
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("prewarming exact orientations should succeed");
    assert_eq!(engine.take_allocation_failure(), None);
    engine.canonical_caches.symmetry_refs.release_storage();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    let scratch_bypasses = engine.stats.transform.orientation_scratch_bypasses;
    let work_bypasses = engine.stats.transform.orientation_work_bypasses;
    engine.begin_allocation_transaction(retained);

    let actual = engine
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("scratch denial must fall back to exact scalar resolution");

    assert_same_record(actual, expected);
    assert_eq!(engine.take_allocation_failure(), None);
    assert_eq!(engine.canonical_caches.symmetry_refs.len(), 0);
    assert_eq!(
        engine.stats.transform.orientation_scratch_bypasses,
        scratch_bypasses + 1
    );
    assert_eq!(
        engine.stats.transform.orientation_work_bypasses,
        work_bypasses
    );
    assert!(
        (orientations::SCALAR_RESOLVER_STACK_BYTES as u128) < orientations::RESOLVER_BYTES,
        "the allocation-free fallback should use the smaller fixed stack"
    );
}

#[test]
fn warm_orientation_hit_does_not_resolve_more_signatures() {
    let mut engine = HashLifeEngine::default();
    let shape = corner_level_one_shape(&mut engine);
    let first = engine
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("cold orientation resolution should succeed");
    let signatures = engine.stats.transform.orientation_resolved_signatures;

    let second = engine
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("warm orientation resolution should succeed");

    assert_same_record(second, first);
    assert_eq!(
        engine.stats.transform.orientation_resolved_signatures,
        signatures
    );
    assert_eq!(engine.take_allocation_failure(), None);
}

#[test]
fn scalar_fallback_single_orientation_does_not_eagerly_resolve_the_full_orbit() {
    let mut engine = HashLifeEngine::default();
    let shape = corner_level_one_shape(&mut engine);
    let shapes_before = engine.canonical_caches.shapes.len();

    let record = engine
        .resolve_shape_orientations_with_quotient_limit(shape, symmetry_bit(Symmetry::Rotate90), 0)
        .or_invariant("single orientation resolution should succeed");

    assert_eq!(
        record.resolved,
        symmetry_bit(Symmetry::Identity) | symmetry_bit(Symmetry::Rotate90)
    );
    assert_eq!(
        engine.canonical_caches.shapes.len(),
        shapes_before + 1,
        "a single request should intern only its needed parent shape"
    );
    assert_eq!(engine.canonical_caches.symmetry_refs.len(), 1);
    assert_eq!(engine.take_allocation_failure(), None);
}

#[test]
fn quotient_work_exhaustion_falls_back_and_reuses_equal_children_per_frame() {
    const DEPTH: usize = 48;

    let mut engine = HashLifeEngine::default();
    let shape = deep_uniform_shape(&mut engine, DEPTH);
    engine.canonical_caches.symmetry_refs.release_storage();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    let requests_before = engine.stats.transform.orientation_requests;
    let scratch_bypasses = engine.stats.transform.orientation_scratch_bypasses;
    let work_bypasses = engine.stats.transform.orientation_work_bypasses;
    engine.begin_allocation_transaction(retained + orientations::RESOLVER_BYTES);

    let record = engine
        .resolve_shape_orientations_with_quotient_limit(shape, symmetry_bit(Symmetry::Rotate90), 3)
        .or_invariant("bounded quotient exhaustion must use the scalar fallback");

    assert_eq!(record.reference(Symmetry::Rotate90), shape);
    assert_eq!(engine.take_allocation_failure(), None);
    assert_eq!(
        engine.stats.transform.orientation_scratch_bypasses,
        scratch_bypasses
    );
    assert_eq!(
        engine.stats.transform.orientation_work_bypasses,
        work_bypasses + 1
    );
    assert!(
        engine.stats.transform.orientation_requests - requests_before <= DEPTH * 5 + 4,
        "equal-child reuse must keep denied-cache fallback work linear in depth"
    );
}

#[test]
fn scalar_scratch_bypass_reuses_equal_children_with_publication_denied() {
    const DEPTH: usize = 48;

    let mut engine = HashLifeEngine::default();
    let shape = deep_uniform_shape(&mut engine, DEPTH);
    engine.canonical_caches.symmetry_refs.release_storage();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    let requests_before = engine.stats.transform.orientation_requests;
    let scratch_bypasses = engine.stats.transform.orientation_scratch_bypasses;
    engine.begin_allocation_transaction(retained);

    let record = engine
        .resolve_shape_orientations(shape, symmetry_bit(Symmetry::Rotate90))
        .or_invariant("scratch denial must use exact scalar resolution");

    assert_eq!(record.reference(Symmetry::Rotate90), shape);
    assert_eq!(engine.take_allocation_failure(), None);
    assert_eq!(engine.canonical_caches.symmetry_refs.len(), 0);
    assert_eq!(
        engine.stats.transform.orientation_scratch_bypasses,
        scratch_bypasses + 1
    );
    assert!(
        engine.stats.transform.orientation_requests - requests_before <= DEPTH * 5 + 4,
        "equal-child reuse must keep denied-cache fallback work linear in depth"
    );
}

#[test]
fn scalar_fallback_preserves_mandatory_canonical_id_failure() {
    let mut engine = HashLifeEngine::default();
    let shape = corner_level_one_shape(&mut engine);
    engine.id_capacity.canonical_count = engine.canonical_caches.shapes.len();
    engine.begin_allocation_transaction(u128::MAX);

    assert!(
        engine
            .resolve_shape_orientations_with_quotient_limit(
                shape,
                symmetry_bit(Symmetry::Rotate90),
                0,
            )
            .is_none()
    );
    assert_eq!(
        engine.take_allocation_failure(),
        Some(EngineAllocationFailure::CanonicalReferenceExhausted)
    );
}

#[test]
fn canonical_id_exhaustion_does_not_publish_partial_parent_and_retry_reuses_children() {
    let mut engine = HashLifeEngine::default();
    let (child, parent) = nested_corner_shapes(&mut engine);
    let child_record = engine
        .resolve_shape_orientations(child, u8::MAX)
        .or_invariant("child orientations should prewarm");
    assert_eq!(child_record.resolved, u8::MAX);

    let shapes_before = engine.canonical_caches.shapes.len();
    engine.id_capacity.canonical_count = shapes_before;
    engine.begin_allocation_transaction(u128::MAX);

    assert!(engine.resolve_shape_orientations(parent, u8::MAX).is_none());
    assert_eq!(
        engine.take_allocation_failure(),
        Some(EngineAllocationFailure::CanonicalReferenceExhausted)
    );
    assert_eq!(engine.canonical_caches.shapes.len(), shapes_before);
    assert!(
        engine.canonical_caches.symmetry_refs.get(&child).is_some(),
        "resolved children must remain reusable after parent failure"
    );
    assert!(
        engine.canonical_caches.symmetry_refs.get(&parent).is_none(),
        "failed parent resolution must not publish a partial record"
    );

    engine.id_capacity = EngineIdCapacity::FULL;
    engine.begin_allocation_transaction(u128::MAX);
    let retry = engine
        .resolve_shape_orientations(parent, u8::MAX)
        .or_invariant("restored canonical ID capacity should permit retry");
    assert_eq!(retry.resolved, u8::MAX);
    assert_eq!(engine.take_allocation_failure(), None);
    assert!(engine.canonical_caches.symmetry_refs.get(&parent).is_some());
    let mut uninterrupted = HashLifeEngine::default();
    let (_, expected_parent) = nested_corner_shapes(&mut uninterrupted);
    let expected = uninterrupted
        .resolve_shape_orientations(expected_parent, u8::MAX)
        .or_invariant("uninterrupted orientation proof");
    for symmetry in Symmetry::ALL {
        let actual_meta = engine.canonical_caches.shapes[retry.reference(symmetry).index()];
        let expected_meta =
            uninterrupted.canonical_caches.shapes[expected.reference(symmetry).index()];
        assert!(
            actual_meta.prefix.complete && expected_meta.prefix.complete,
            "4x4 literal prefixes must contain the entire state"
        );
        assert_eq!(
            actual_meta.prefix, expected_meta.prefix,
            "retry differs for {symmetry:?}"
        );
        assert_eq!(actual_meta.key.level, expected_meta.key.level);
    }
}
