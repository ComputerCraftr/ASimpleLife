use super::*;

fn node_from_4x4_bits(engine: &mut HashLifeEngine, bits: u16) -> NodeId {
    let quadrants: [NodeId; 4] = std::array::from_fn(|quadrant| {
        let origin_x = (quadrant % 2) * 2;
        let origin_y = (quadrant / 2) * 2;
        let leaves: [NodeId; 4] = std::array::from_fn(|child| {
            let x = origin_x + child % 2;
            let y = origin_y + child / 2;
            if bits & (1_u16 << (y * 4 + x)) == 0 {
                engine.dead_leaf
            } else {
                engine.live_leaf
            }
        });
        engine.join(leaves[0], leaves[1], leaves[2], leaves[3])
    });
    engine.join(quadrants[0], quadrants[1], quadrants[2], quadrants[3])
}

fn quadrant_4x4(bits: u64, quadrant: usize) -> u16 {
    let origin_x = (quadrant % 2) * 4;
    let origin_y = (quadrant / 2) * 4;
    let mut result = 0_u16;
    for y in 0..4 {
        for x in 0..4 {
            if bits & (1_u64 << ((origin_y + y) * 8 + origin_x + x)) != 0 {
                result |= 1_u16 << (y * 4 + x);
            }
        }
    }
    result
}

fn node_from_8x8_bits(engine: &mut HashLifeEngine, bits: u64) -> NodeId {
    let quadrants: [NodeId; 4] =
        std::array::from_fn(|index| node_from_4x4_bits(engine, quadrant_4x4(bits, index)));
    engine.join(quadrants[0], quadrants[1], quadrants[2], quadrants[3])
}

fn shape_class(engine: &mut HashLifeEngine, node: NodeId) -> FutureClass {
    let structural = engine.canonical_jump_probe((node, 0)).key.structural;
    engine
        .future_class_for_structural(structural)
        .or_invariant("test FutureClass proof")
}

fn conway_center(pattern: u16) -> bool {
    let alive = pattern & (1 << 4) != 0;
    let neighbors = (pattern & !(1 << 4)).count_ones();
    neighbors == 3 || (alive && neighbors == 2)
}

fn place_neighborhood(pattern: u16, center_x: usize, center_y: usize) -> (u64, u64) {
    let mut bits = 0_u64;
    let mut mask = 0_u64;
    for local_y in 0..3 {
        for local_x in 0..3 {
            let bit = 1_u64 << ((center_y + local_y - 1) * 8 + center_x + local_x - 1);
            mask |= bit;
            if pattern & (1 << (local_y * 3 + local_x)) != 0 {
                bits |= bit;
            }
        }
    }
    (bits, mask)
}

#[test]
fn exact_8x8_signature_exhaustively_matches_every_local_boundary_truth_table() {
    const OUTSIDE_ASSIGNMENTS: [u64; 5] = [
        0,
        u64::MAX,
        0xAA55_AA55_AA55_AA55,
        0x55AA_55AA_55AA_55AA,
        0xC31E_69A5_5A96_78C3,
    ];
    for center_y in 1..7 {
        for center_x in 1..7 {
            let output_bit = 1_u64 << ((center_y - 1) * 6 + center_x - 1);
            for pattern in 0_u16..512 {
                let (local, local_mask) = place_neighborhood(pattern, center_x, center_y);
                for outside in OUTSIDE_ASSIGNMENTS {
                    let current = (outside & !local_mask) | local;
                    let (_, next) = exact_8x8_signature(current);
                    assert_eq!(
                        next & output_bit != 0,
                        conway_center(pattern),
                        "center=({center_x},{center_y}) pattern={pattern:#05x} outside={outside:#018x}"
                    );
                }
            }
        }
    }

    for bit in 0..64 {
        let (border, _) = exact_8x8_signature(1_u64 << bit);
        let x = bit % 8;
        let y = bit / 8;
        assert_eq!(border != 0, x.min(y).min(7 - x).min(7 - y) < 2);
    }
}

fn nonstructural_level_five_pair(engine: &mut HashLifeEngine) -> (NodeId, NodeId) {
    let block = (1_u64 << (2 * 8 + 2))
        | (1_u64 << (2 * 8 + 3))
        | (1_u64 << (3 * 8 + 2))
        | (1_u64 << (3 * 8 + 3));
    let base = node_from_8x8_bits(engine, block);
    let equivalent = node_from_8x8_bits(engine, block | (1_u64 << (5 * 8 + 5)));
    let empty_3 = engine.empty(3);
    let source_4 = engine.join(base, empty_3, empty_3, empty_3);
    let equivalent_4 = engine.join(equivalent, empty_3, empty_3, empty_3);
    let empty_4 = engine.empty(4);
    (
        engine.join(source_4, empty_4, empty_4, empty_4),
        engine.join(equivalent_4, empty_4, empty_4, empty_4),
    )
}

#[test]
fn parent_congruence_reuses_every_legal_exponent_under_all_d4_transforms() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(u32::MAX, u64::MAX);
    let (source, equivalent) = nonstructural_level_five_pair(&mut engine);
    assert_ne!(
        engine.node_columns.identity_ref(source),
        engine.node_columns.identity_ref(equivalent)
    );
    assert_eq!(
        shape_class(&mut engine, source),
        shape_class(&mut engine, equivalent)
    );

    for step_exp in 0..=3 {
        let result = engine.advance_pow2(source, step_exp);
        engine.result_caches.jump.reset();
        engine.active_jump_results.reset();
        for symmetry in Symmetry::ALL {
            let transformed = engine.transform_node(equivalent, symmetry);
            let expected = engine.transform_node(result, symmetry);
            assert_eq!(
                engine.cached_jump_result((transformed, step_exp)),
                Some(expected),
                "step_exp={step_exp} symmetry={symmetry:?}"
            );
        }
    }
}

#[test]
fn publication_preserves_a_nonidentity_stored_output_frame() {
    let mut engine = HashLifeEngine::default();
    let source = node_from_8x8_bits(
        &mut engine,
        (1_u64 << 18) | (1_u64 << 26) | (1_u64 << 27) | (1_u64 << 35),
    );
    let base_result = node_from_4x4_bits(&mut engine, 0x0136);
    let mut observed = None;
    for symmetry in Symmetry::ALL {
        engine.clear_future_results();
        engine.result_caches.jump.reset();
        engine.active_jump_results.reset();
        let result = engine.transform_node(base_result, symmetry);
        engine.insert_jump_result((source, 0), result);
        let probe = engine.canonical_jump_probe((source, 0));
        let class = engine
            .future_class_for_structural(probe.key.structural)
            .or_invariant("stored-output FutureClass");
        let result_key = FutureResultKey {
            class,
            jump: PositiveJump::power_of_two(0, 3).or_invariant("positive test jump"),
            source_level: 3,
            symmetry_admitted: probe.key.symmetry_admitted,
        };
        let stored = engine
            .future_state
            .results
            .get(&result_key)
            .or_invariant("published Future result");
        if stored.entry.symmetry != Symmetry::Identity {
            observed = Some(stored.entry.symmetry);
            break;
        }
    }
    assert!(
        observed.is_some(),
        "fixture never stored a nonidentity output frame"
    );
}

#[test]
fn future_hit_materialization_failure_is_flagged_without_a_second_probe() {
    let mut engine = HashLifeEngine::default();
    let source = node_from_8x8_bits(
        &mut engine,
        (1_u64 << 18) | (1_u64 << 26) | (1_u64 << 34) | (1_u64 << 35),
    );
    assert_ne!(
        engine.advance_one_generation_centered(source),
        engine.dead_leaf,
        "future-hit failure fixture must seed a completed result"
    );
    let probe = engine.canonical_jump_probe((source, 0));
    let class = engine
        .future_class_for_structural(probe.key.structural)
        .or_invariant("future-hit failure class");

    let live = engine.join(
        engine.live_leaf,
        engine.live_leaf,
        engine.live_leaf,
        engine.live_leaf,
    );
    let dead = engine.empty(1);
    let packed = PackedNodeKey::new(2, [live, dead, dead, live]);
    assert!(
        engine.intern.get(&packed).is_none(),
        "failure fixture must require a new result parent"
    );
    let result_key = FutureResultKey {
        class,
        jump: PositiveJump::power_of_two(0, probe.key.structural.level)
            .or_invariant("future-hit failure jump"),
        source_level: probe.key.structural.level,
        symmetry_admitted: probe.key.symmetry_admitted,
    };
    engine
        .future_state
        .results
        .try_insert_with_fingerprint(
            result_key,
            result_key.fingerprint(),
            FutureJumpResult {
                entry: PackedSymmetryKey {
                    packed,
                    symmetry: Symmetry::Identity,
                },
                result_level: 2,
                registry_epoch: engine.future_state.registry_epoch,
            },
        )
        .or_invariant("install future-hit failure fixture");

    engine.result_caches.jump.reset();
    engine.result_caches.materialized_packed.reset();
    engine.active_jump_results.reset();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    engine.begin_allocation_transaction(retained);
    let lookups_before = engine.stats.future.result_lookups;

    assert_eq!(
        engine.advance_one_generation_centered(source),
        engine.dead_leaf
    );
    assert!(engine.allocation_failed());
    assert_eq!(
        engine.stats.future.result_lookups,
        lookups_before + 1,
        "failed future materialization must not trigger a second memo probe"
    );
    assert_eq!(engine.result_caches.jump.len(), 0);
    assert_eq!(engine.active_jump_results.len(), 0);
    assert!(engine.take_allocation_failure().is_some());
}

#[test]
fn canonical_8x8_extraction_matches_all_64_independent_single_bit_grids() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    for bit in 0..64 {
        let expected = 1_u64 << bit;
        let node = node_from_8x8_bits(&mut engine, expected);
        let shape = engine.node_columns.identity_ref(node);
        let mut budget = FutureProofBudget::default();
        let actual = engine
            .canonical_shape_bits_8x8(shape, &mut budget)
            .or_invariant("level-3 shape extraction");
        assert_eq!(actual, expected, "single live cell at bit {bit}");
        assert_eq!(budget.visits, 84);

        let (actual_border, actual_next) = exact_8x8_signature(actual);
        let x = bit % 8;
        let y = bit / 8;
        let expected_border = if x.min(y).min(7 - x).min(7 - y) < 2 {
            expected
        } else {
            0
        };
        assert_eq!(actual_border, expected_border, "border bit {bit}");
        assert_eq!(actual_next, 0, "isolated cell must die at bit {bit}");
    }
}

#[test]
fn below_level_three_is_exact_at_generation_zero() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let empty = engine.empty(2);
    let live = node_from_4x4_bits(&mut engine, 1 << 5);

    assert_ne!(
        engine.node_columns.identity_ref(empty),
        engine.node_columns.identity_ref(live)
    );
    assert_ne!(
        shape_class(&mut engine, empty),
        shape_class(&mut engine, live)
    );
    assert_eq!(engine.future_state.results.len(), 0);
}

#[test]
fn below_level_three_successor_cache_attempts_are_ineligible_without_class_work() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let source = node_from_4x4_bits(&mut engine, 1 << 5);
    let result = engine.empty(1);
    let classes_before = engine.future_state.class_count;
    let class_lookups_before = engine.stats.future.class_lookups;

    engine.insert_jump_result((source, 0), result);
    engine.result_caches.jump.reset();
    engine.active_jump_results.reset();
    assert_eq!(engine.cached_jump_result((source, 0)), None);

    assert_eq!(engine.future_state.class_count, classes_before);
    assert_eq!(engine.stats.future.class_lookups, class_lookups_before);
    assert_eq!(engine.stats.future.ineligible_publications, 1);
    assert_eq!(engine.stats.future.ineligible_lookups, 1);
}

#[test]
fn nonstructural_result_reuse_skips_real_scheduler_work() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let block = (1_u64 << (2 * 8 + 2))
        | (1_u64 << (2 * 8 + 3))
        | (1_u64 << (3 * 8 + 2))
        | (1_u64 << (3 * 8 + 3));
    let source = node_from_8x8_bits(&mut engine, block);
    let equivalent = node_from_8x8_bits(&mut engine, block | (1_u64 << (5 * 8 + 5)));
    assert_ne!(
        engine.node_columns.identity_ref(source),
        engine.node_columns.identity_ref(equivalent)
    );
    assert_eq!(
        shape_class(&mut engine, source),
        shape_class(&mut engine, equivalent)
    );

    let expected = engine.advance_one_generation_centered(source);
    let tasks_before = engine.stats.scheduler.scheduler_tasks;
    let candidates_before = engine.stats.simd.kernel.candidate_lanes;
    let saves_before = engine.stats.future.saved_jump_lookups;
    let actual = engine.advance_one_generation_centered(equivalent);

    assert_eq!(actual, expected);
    assert_eq!(engine.node_columns.level(actual), 2);
    assert_eq!(engine.stats.scheduler.scheduler_tasks, tasks_before);
    assert_eq!(engine.stats.simd.kernel.candidate_lanes, candidates_before);
    assert!(engine.stats.future.saved_jump_lookups > saves_before);
}

#[test]
fn parent_exponent_and_crop_are_part_of_the_contract() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let empty = engine.empty(3);
    let single = node_from_8x8_bits(&mut engine, 1_u64 << (3 * 8 + 3));
    assert_eq!(
        shape_class(&mut engine, empty),
        shape_class(&mut engine, single)
    );

    let empty_parent = engine.join(empty, empty, empty, empty);
    let equivalent_parent = engine.join(single, empty, empty, empty);
    assert_eq!(
        shape_class(&mut engine, empty_parent),
        shape_class(&mut engine, equivalent_parent)
    );

    let result = engine.empty(3);
    engine.insert_jump_result((empty_parent, 0), result);
    assert_eq!(
        engine.cached_jump_result((equivalent_parent, 0)),
        Some(result)
    );
    assert_eq!(engine.cached_jump_result((equivalent_parent, 1)), None);
    assert_eq!(
        engine.stats.future.result_lookups,
        engine.stats.future.result_hits
            + engine.stats.future.result_misses
            + engine.stats.future.result_bypasses
            + engine.stats.future.ineligible_lookups
            + engine.stats.future.analysis_sampling_bypasses
    );
}

#[test]
fn future_hit_restores_original_orientation() {
    let mut engine = HashLifeEngine::default();
    let bits = (1_u64 << (2 * 8 + 2))
        | (1_u64 << (3 * 8 + 2))
        | (1_u64 << (4 * 8 + 2))
        | (1_u64 << (4 * 8 + 3));
    let source = node_from_8x8_bits(&mut engine, bits);
    let result = engine.advance_one_generation_centered(source);
    let rotated_source = engine.transform_node(source, Symmetry::Rotate90);
    let expected = engine.transform_node(result, Symmetry::Rotate90);
    engine.result_caches.jump.reset();
    engine.active_jump_results.reset();

    let saves_before = engine.stats.future.saved_jump_lookups;
    assert_eq!(
        engine.cached_jump_result((rotated_source, 0)),
        Some(expected)
    );
    assert_eq!(
        engine.node_columns.level(expected) + 1,
        engine.node_columns.level(source)
    );
    assert_eq!(engine.stats.future.saved_jump_lookups, saves_before + 1);
}

#[test]
fn colliding_fingerprints_do_not_alias_future_keys() {
    let left = FutureKey::Exact {
        level: 1,
        rule: FutureRule::Conway,
        shape: CanonicalShapeId::DEAD,
    };
    let right = FutureKey::Exact {
        level: 2,
        rule: FutureRule::Conway,
        shape: CanonicalShapeId::LIVE,
    };
    let mut table = ProbeTable::new(ProbeMode::AppendOnly);
    table
        .try_insert_with_fingerprint(left, 7, FutureClass(11))
        .or_invariant("left collision fixture");
    table
        .try_insert_with_fingerprint(right, 7, FutureClass(12))
        .or_invariant("right collision fixture");
    assert_eq!(table.get_with_fingerprint(&left, 7), Some(FutureClass(11)));
    assert_eq!(table.get_with_fingerprint(&right, 7), Some(FutureClass(12)));
}

#[test]
fn pressure_bypass_does_not_report_mandatory_failure() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let source = node_from_8x8_bits(&mut engine, 1_u64 << (3 * 8 + 3));
    let structural = engine.canonical_jump_probe((source, 0)).key.structural;
    engine.release_future_state();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    engine.begin_allocation_transaction(retained);

    assert_eq!(engine.future_class_for_structural(structural), None);
    assert_eq!(engine.take_allocation_failure(), None);
}

#[test]
fn failed_future_materialization_never_returns_a_sentinel_as_some() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let source = node_from_8x8_bits(&mut engine, 1_u64 << 27);
    let probe = engine.canonical_jump_probe((source, 0));
    let class = engine
        .future_class_for_structural(probe.key.structural)
        .or_invariant("materialization-failure FutureClass");
    let dead = engine.dead_leaf;
    let live = engine.live_leaf;
    let children = [
        engine.join(live, dead, dead, dead),
        engine.join(dead, live, dead, dead),
        engine.join(dead, dead, live, dead),
        engine.join(dead, dead, dead, live),
    ];
    let packed = PackedNodeKey::new(2, children);
    assert_eq!(
        engine.intern.get(&packed),
        None,
        "fixture result already materialized"
    );
    let result_key = FutureResultKey {
        class,
        jump: PositiveJump::power_of_two(0, 3).or_invariant("positive test jump"),
        source_level: 3,
        symmetry_admitted: false,
    };
    engine
        .future_state
        .results
        .try_insert(
            result_key,
            FutureJumpResult {
                entry: PackedSymmetryKey {
                    packed,
                    symmetry: Symmetry::Identity,
                },
                result_level: 2,
                registry_epoch: engine.future_state.registry_epoch,
            },
        )
        .or_invariant("Future result fixture publication");
    engine.id_capacity.node_count = engine.node_count();
    let retained = crate::hashlife::memory::wide_allocated_bytes(engine.allocated_bytes());
    engine.begin_allocation_transaction(retained);

    assert_eq!(engine.cached_jump_result((source, 0)), None);
    assert!(engine.take_allocation_failure().is_some());
    assert_eq!(engine.intern.get(&packed), None);
}

#[test]
fn registry_epoch_clears_every_dependent_cache() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let source = engine.empty(3);
    let result = engine.empty(2);
    engine.insert_jump_result((source, 0), result);
    assert_ne!(engine.future_state.results.len(), 0);
    assert_ne!(engine.future_state.class_count, 0);
    let epoch = engine.future_state.registry_epoch;

    engine.rebuild_canonical_shapes();
    let structural = engine.canonical_jump_probe((source, 0)).key.structural;
    let _ = engine.future_class_for_structural(structural);

    assert_eq!(engine.future_state.registry_epoch, epoch + 1);
    assert_eq!(engine.future_state.results.len(), 0);
}

#[test]
fn mark_only_gc_filters_weak_future_result_ids_without_rooting_them() {
    let mut engine = HashLifeEngine::with_symmetry_gate_for_tests(0, 0);
    let source = engine.empty(3);
    let weak_result = node_from_4x4_bits(&mut engine, 0x0362);
    engine.insert_jump_result((source, 0), weak_result);
    assert_eq!(engine.future_state.results.len(), 1);
    let class_count = engine.future_state.class_count;

    engine.record_retained_root(source);
    engine.mark_live_nodes();
    engine.filter_caches_to_live_nodes();

    assert_eq!(engine.future_state.results.len(), 0);
    assert_eq!(engine.future_state.class_count, class_count);
    assert_eq!(engine.stats.future.weak_results_dropped, 1);
}

#[test]
fn proof_visit_budget_has_an_exact_boundary() {
    let mut budget = FutureProofBudget::default();
    assert!(budget.visit_child_slots(FUTURE_PROOF_MAX_VISITS));
    assert!(!budget.visit_child_slots(1));
    assert_eq!(budget.visits, FUTURE_PROOF_MAX_VISITS);
    assert!(budget.exhausted);
}

#[test]
fn repeated_analysis_misses_enter_bounded_cooldown_then_resample() {
    let mut engine = HashLifeEngine::default();

    for _ in 0..FUTURE_MISS_SAMPLE {
        engine.record_future_analysis_miss();
    }

    assert_eq!(engine.future_state.analyzed_misses, 0);
    assert_eq!(engine.future_state.analysis_cooldown, FUTURE_MISS_COOLDOWN);
    assert_eq!(engine.stats.future.analysis_cooldowns, 1);

    for remaining in (0..FUTURE_MISS_COOLDOWN).rev() {
        assert!(!engine.future_analysis_admitted());
        assert_eq!(engine.future_state.analysis_cooldown, remaining);
    }

    assert!(engine.future_analysis_admitted());
    assert_eq!(
        engine.stats.future.analysis_sampling_bypasses,
        FUTURE_MISS_COOLDOWN
    );
}

#[test]
fn positive_jump_maps_exponent_zero_to_one_generation_and_rejects_illegal_levels() {
    assert_eq!(
        PositiveJump::power_of_two(0, 2),
        Some(PositiveJump { step_exp: 0 })
    );
    assert_eq!(PositiveJump::power_of_two(0, 1), None);
    assert_eq!(PositiveJump::power_of_two(2, 3), None);
    assert_eq!(
        PositiveJump::power_of_two(2, 4),
        Some(PositiveJump { step_exp: 2 })
    );
}
