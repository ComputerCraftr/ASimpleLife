use super::normalization::{RECURRENCE_REBLOCK_WORK_LIMIT, charge_work};
use super::*;
use crate::recurrence::{ExactRecurrenceTracker, ExactWitness, ObserveOutcome};

fn dag_witness(observation: &Observation) -> DagWitness {
    match observation.witness() {
        ExactWitness::Dag(witness) => *witness,
        ExactWitness::Cells(_) => {
            crate::invariant_failure!("HashLife recurrence observation must use a DAG witness")
        }
    }
}

fn translated(cells: &[(Coord, Coord)], dx: Coord, dy: Coord) -> BitGrid {
    BitGrid::from_cells(
        &cells
            .iter()
            .map(|&(x, y)| (x + dx, y + dy))
            .collect::<Vec<_>>(),
    )
}

fn add_unreachable_patterns(engine: &mut HashLifeEngine) {
    for bits in 1..8_192_u32 {
        let children = std::array::from_fn::<_, 4, _>(|quadrant| {
            let cells = std::array::from_fn::<_, 4, _>(|cell| {
                if bits & (1 << (quadrant * 4 + cell)) == 0 {
                    engine.dead_leaf
                } else {
                    engine.live_leaf
                }
            });
            engine.join(cells[0], cells[1], cells[2], cells[3])
        });
        engine.join(children[0], children[1], children[2], children[3]);
    }
}

#[test]
fn glider_normalization_ignores_unaligned_positive_and_negative_translation() {
    let glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let mut session = HashLifeSession::new();
    let lineage = Lineage::new(41, 1);
    let mut normalized_root = None;
    for offset in -7..=7 {
        session
            .try_load_grid(&translated(&glider, offset, -offset))
            .or_invariant("translated glider should load");
        let observation = session
            .try_recurrence_observation(lineage)
            .or_invariant("translated glider should normalize");
        let witness = dag_witness(&observation);
        assert_eq!(
            observation.anchor(),
            (i128::from(offset), i128::from(-offset))
        );
        normalized_root.get_or_insert(witness.root());
        assert_eq!(Some(witness.root()), normalized_root);
        assert_eq!(witness.level(), 2);
    }
    assert_eq!(session.sample_materializations(), 0);
}

#[test]
fn wide_sparse_normalization_depends_on_live_dag_not_empty_span() {
    let grid = BitGrid::from_cells(&[
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (70_000, 0),
        (70_001, 0),
        (70_000, 1),
        (70_001, 1),
    ]);
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("wide sparse fixture should load");
    let root = session
        .current_root
        .or_invariant("fixture root should exist");
    let mut memo = BoundsMemo::new();
    let mut bounds_work = 0;
    bounded_relative_bounds(&session.engine, root, &mut memo, &mut bounds_work)
        .or_invariant("wide sparse bounds should fit the DAG budget");
    assert!(
        bounds_work < RECURRENCE_REBLOCK_WORK_LIMIT,
        "bounds exhausted the complete evidence budget: {bounds_work}"
    );

    let observation = session
        .try_recurrence_observation(Lineage::new(50, 1))
        .or_invariant("wide sparse DAG should normalize within the evidence budget");

    assert_eq!(observation.anchor(), (0, 0));
    assert_eq!(dag_witness(&observation).level(), 17);
    assert_eq!(session.sample_materializations(), 0);
}

#[test]
fn normalization_is_independent_of_centered_shell_padding() {
    let grid = translated(&[(0, 0), (1, 0), (2, 0)], 17, -23);
    let mut session = HashLifeSession::new();
    let lineage = Lineage::new(42, 1);
    session
        .try_load_grid(&grid)
        .or_invariant("fixture should load");
    let before = session
        .try_recurrence_observation(lineage)
        .or_invariant("fixture should normalize");
    session.ensure_active_run();
    session
        .engine
        .begin_allocation_transaction(session.limits.hard_memory_bytes);
    session
        .ensure_centered_capacity(5)
        .or_invariant("fixture should admit padding");
    assert_eq!(session.engine.take_allocation_failure(), None);
    let after = session
        .try_recurrence_observation(lineage)
        .or_invariant("padded fixture should normalize");
    assert_eq!(dag_witness(&after).root(), dag_witness(&before).root());
    assert_eq!(dag_witness(&after).level(), dag_witness(&before).level());
    assert_eq!(dag_witness(&after).epoch(), dag_witness(&before).epoch());
    assert_eq!(after.anchor(), before.anchor());
    assert_eq!(session.sample_materializations(), 0);
    for &(x, y) in &[(17, -23), (18, -23), (19, -23)] {
        assert_eq!(session.sample_cell(x, y), Some(true));
    }
}

#[test]
fn gc_repack_retires_weak_witness_without_changing_authoritative_state() {
    let grid = translated(&[(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)], -11, 9);
    let mut session = HashLifeSession::new();
    let lineage = Lineage::new(46, 1);
    session
        .try_load_grid(&grid)
        .or_invariant("fixture should load");
    let before = session
        .try_recurrence_observation(lineage)
        .or_invariant("fixture should normalize");
    let before_epoch = dag_witness(&before).epoch();
    add_unreachable_patterns(&mut session.engine);
    let bytes = session.allocated_bytes();
    session.set_limits(HashLifeLimits {
        soft_memory_bytes: bytes / 2,
        hard_memory_bytes: bytes,
    });

    session.collect_before_allocation();

    let after = session
        .try_recurrence_observation(lineage)
        .or_invariant("repacked fixture should normalize");
    assert!(dag_witness(&after).epoch() > before_epoch);
    assert_eq!(after.anchor(), before.anchor());
    for &(x, y) in &[(-10, 9), (-9, 10), (-11, 11), (-10, 11), (-9, 11)] {
        assert_eq!(session.sample_cell(x, y), Some(true));
    }
    assert_eq!(session.population_count(), Some(PopulationCount::Exact(5)));
    assert_eq!(session.sample_materializations(), 0);
}

#[test]
fn repeated_dense_dag_is_charged_by_unique_nodes_not_paths() {
    let mut session = HashLifeSession::new();
    let mut root = session.engine.live_leaf;
    for _ in 0..7 {
        root = session.engine.join(root, root, root, root);
    }
    session.current_root = Some(root);
    session.current_origin_x = 0;
    session.current_origin_y = 0;
    let state = session.published_state();

    let observation = session
        .try_recurrence_observation(Lineage::new(47, 1))
        .or_invariant("shared dense DAG should fit the evidence budget");
    assert_eq!(dag_witness(&observation).root(), u64::from(root));
    assert_eq!(dag_witness(&observation).level(), 7);
    assert_eq!(session.published_state().current_root, state.current_root);
    assert_eq!(session.current_origin_x, state.current_origin_x);
    assert_eq!(session.current_origin_y, state.current_origin_y);
    assert_eq!(session.current_generation, state.current_generation);
    assert_eq!(session.engine.take_allocation_failure(), None);
    assert_eq!(session.sample_materializations(), 0);
}

#[test]
fn evidence_work_limit_rejects_the_first_excess_proof() {
    let mut work = 0;
    for _ in 0..RECURRENCE_REBLOCK_WORK_LIMIT {
        charge_work(&mut work).or_invariant("work through the exact limit should be admitted");
    }
    assert_eq!(
        charge_work(&mut work),
        Err(RecurrenceUnavailable::WitnessLimit)
    );
}

#[test]
fn extreme_coordinate_anchor_does_not_require_padded_coord_geometry() {
    for coordinate in [Coord::MIN, Coord::MAX] {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(coordinate, coordinate)]))
            .or_invariant("extreme coordinate fixture should load");

        let observation = session
            .try_recurrence_observation(Lineage::new(48, 1))
            .or_invariant("wide normalization should retain the extreme anchor");

        assert_eq!(
            observation.anchor(),
            (i128::from(coordinate), i128::from(coordinate))
        );
        assert_eq!(dag_witness(&observation).level(), 0);
        assert_eq!(session.sample_materializations(), 0);
    }
}

#[test]
fn unavailable_candidate_preserves_mandatory_failure_and_session_state() {
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&translated(&[(0, 0), (1, 1), (2, 2)], 3, 5))
        .or_invariant("fixture should load");
    let state = session.published_state();
    let lineage = Lineage::new(43, 1);
    let reserved = session.engine.allocation_transient_reserved;
    session.engine.reject_allocation(77);

    assert_eq!(
        session.try_recurrence_observation(lineage),
        Err(RecurrenceUnavailable::Allocation)
    );
    assert_eq!(session.published_state().current_root, state.current_root);
    assert_eq!(session.current_origin_x, state.current_origin_x);
    assert_eq!(session.current_origin_y, state.current_origin_y);
    assert_eq!(session.current_generation, state.current_generation);
    assert_eq!(session.engine.allocation_transient_reserved, reserved);
    assert_eq!(
        session.engine.take_allocation_failure(),
        Some(EngineAllocationFailure::Allocation {
            requested_bytes: 77
        })
    );
    assert_eq!(session.sample_materializations(), 0);
}

#[test]
fn optional_candidate_failure_is_unavailable_without_failure_poison() {
    let mut session = HashLifeSession::new();
    let dead = session.engine.dead_leaf;
    let live = session.engine.live_leaf;
    let empty = session.engine.join(dead, dead, dead, dead);
    let northwest = session.engine.join(dead, dead, dead, live);
    let southeast = session.engine.join(live, dead, dead, dead);
    let root = session.engine.join(northwest, empty, empty, southeast);
    session.current_root = Some(root);
    session.current_origin_x = -1;
    session.current_origin_y = -1;
    let state = session.published_state();
    let retained = session.allocated_bytes();
    let reserved = session.engine.allocation_transient_reserved;
    session.engine.id_capacity.node_count = session.engine.node_count();

    assert_eq!(
        session.try_recurrence_observation(Lineage::new(49, 1)),
        Err(RecurrenceUnavailable::Allocation)
    );
    assert_eq!(session.published_state().current_root, state.current_root);
    assert_eq!(session.current_origin_x, state.current_origin_x);
    assert_eq!(session.current_origin_y, state.current_origin_y);
    assert_eq!(session.current_generation, state.current_generation);
    assert_eq!(session.engine.allocation_transient_reserved, reserved);
    assert_eq!(session.engine.take_allocation_failure(), None);
    assert_eq!(session.allocated_bytes(), retained);
    assert_eq!(session.sample_materializations(), 0);
}

#[test]
fn recurrence_skip_commits_translation_and_generation_atomically() {
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&translated(&[(0, 0), (1, 0), (2, 0)], 10, -4))
        .or_invariant("fixture should load");
    let lineage = Lineage::new(44, 1);
    let first = session
        .try_recurrence_observation(lineage)
        .or_invariant("first observation should normalize");
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    assert_eq!(tracker.observe(first), ObserveOutcome::Recorded);
    session
        .shift_origin(1, -1)
        .or_invariant("fixture translation should fit");
    session
        .advance_root(2)
        .or_invariant("blinker should return to its phase after two generations");
    let repeated = session
        .try_recurrence_observation(lineage)
        .or_invariant("translated observation should normalize");
    let certificate = tracker
        .observe(repeated)
        .certificate()
        .or_invariant("translated DAG observation should recur");
    let skip = certificate
        .checked_power(session.generation(), 6)
        .or_invariant("two complete recurrence cycles should fit");
    let before_origin = session
        .origin()
        .or_invariant("session should remain loaded");

    let stats = session
        .try_apply_recurrence_skip(skip)
        .or_invariant("validated recurrence skip should commit");

    assert_eq!(stats.completed_generations, 4);
    assert_eq!(session.generation(), 6);
    assert_eq!(
        session.origin(),
        Some((before_origin.0 + 2, before_origin.1 - 2))
    );
}

#[test]
fn recurrence_skip_overflow_has_no_partial_commit() {
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&BitGrid::from_cells(&[(0, 0)]))
        .or_invariant("fixture should load");
    let lineage = Lineage::new(45, 1);
    let witness = DagWitness::new(lineage.session, lineage.epoch, 7, 0);
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    assert_eq!(
        tracker.observe(Observation::from_dag(lineage, 0, (0, 0), witness)),
        ObserveOutcome::Recorded
    );
    let certificate = tracker
        .observe(Observation::from_dag(lineage, 1, (i128::MAX, 0), witness))
        .certificate()
        .or_invariant("fixture recurrence should produce an extreme displacement");
    let skip = certificate
        .checked_power(1, 2)
        .or_invariant("one recurrence cycle should be representable");
    let origin = session.origin();
    let generation = session.generation();

    assert_eq!(
        session.try_apply_recurrence_skip(skip),
        Err(RecurrenceUnavailable::CoordinateOverflow)
    );
    assert_eq!(session.origin(), origin);
    assert_eq!(session.generation(), generation);
}
