use super::*;
use crate::RequiredExt;
use crate::bitgrid::BitGrid;

fn grid(cells: &[(i64, i64)]) -> BitGrid {
    BitGrid::from_cells(cells)
}

fn observation(lineage: Lineage, generation: u64, cells: &[(i64, i64)]) -> Observation {
    Observation::from_grid(lineage, generation, &grid(cells))
        .or_invariant("test witness should be available")
}

#[test]
fn packed_witness_normalizes_subchunk_carries_and_negative_translations() {
    let lineage = Lineage::new(1, 1);
    let base = [(-9, -1), (-8, 0), (-2, 7), (5, 8), (6, 15)];
    let shifted = base.map(|(x, y)| (x + 13, y - 11));
    let first = observation(lineage, 4, &base);
    let second = observation(lineage, 9, &shifted);
    assert_eq!(first.witness(), second.witness());

    let mut tracker = ExactRecurrenceTracker::new(lineage);
    assert_eq!(tracker.observe(first), ObserveOutcome::Recorded);
    let certificate = tracker
        .observe(second)
        .certificate()
        .or_invariant("translated packed witnesses must recur exactly");
    assert_eq!(certificate.period(), 5);
    assert_eq!(certificate.first_seen(), 4);
    assert_eq!(certificate.detected_at(), 9);
    assert_eq!(certificate.delta(), (13, -11));
    assert!(tracker.counters().partitions_hold());
}

#[test]
fn packed_witness_handles_extreme_coordinates_without_translation_overflow() {
    let lineage = Lineage::new(2, 3);
    let low = observation(
        lineage,
        0,
        &[(i64::MIN, i64::MIN), (i64::MIN + 9, i64::MIN + 8)],
    );
    let high = observation(
        lineage,
        1,
        &[(i64::MAX - 9, i64::MAX - 8), (i64::MAX, i64::MAX)],
    );
    assert_eq!(low.witness(), high.witness());
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    assert_eq!(tracker.observe(low), ObserveOutcome::Recorded);
    let certificate = tracker
        .observe(high)
        .certificate()
        .or_invariant("extreme translated states must recur");
    assert_eq!(
        certificate.delta(),
        (i128::from(i64::MAX) * 2 - 8, i128::from(i64::MAX) * 2 - 7)
    );
}

#[test]
fn fingerprint_collision_requires_exact_witness_confirmation() {
    let lineage = Lineage::new(4, 1);
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    let first = observation(lineage, 0, &[(0, 0), (1, 0)]);
    let collision = observation(lineage, 1, &[(0, 0), (0, 1)]);
    let repeated = observation(lineage, 2, &[(10, -3), (11, -3)]);
    assert_eq!(
        tracker.observe_forced_fingerprint(first, 7),
        ObserveOutcome::Recorded
    );
    assert_eq!(
        tracker.observe_forced_fingerprint(collision, 7),
        ObserveOutcome::Recorded
    );
    let certificate = tracker
        .observe_forced_fingerprint(repeated, 7)
        .certificate()
        .or_invariant("exact witness in collision bucket must be confirmed");
    assert_eq!(certificate.first_seen(), 0);
    assert_eq!(certificate.period(), 2);
    assert_eq!(tracker.counters().exact_witness_misses, 1);
    assert_eq!(tracker.counters().certificates_produced, 1);
}

#[test]
fn checked_power_commits_only_whole_safe_cycles() {
    let lineage = Lineage::new(5, 2);
    let certificate = PeriodicCertificate::new(lineage, 4, 8, 12, (3, -2));
    assert_eq!(certificate.lineage(), lineage);
    assert!(certificate.matches_lineage(lineage));
    assert!(!certificate.matches_lineage(Lineage::new(5, 3)));
    let powered = certificate
        .checked_power(12, 31)
        .or_invariant("four whole cycles should fit");
    assert_eq!(powered.committed_generations(), 16);
    assert_eq!(powered.displacement(), (12, -8));
    let translated = powered
        .try_translate_grid(&grid(&[(0, 0), (7, 8)]))
        .or_invariant("powered translation should fit");
    assert_eq!(translated, grid(&[(12, -8), (19, 0)]));
    let later_phase = certificate
        .checked_power(13, 31)
        .or_invariant("later phases retain the recurrence");
    assert_eq!(later_phase.committed_generations(), 16);
    assert_eq!(later_phase.displacement(), (12, -8));
    assert_eq!(certificate.checked_power(11, 31), None);
    assert_eq!(certificate.checked_power(12, 15), None);

    let overflow = PeriodicCertificate::new(lineage, 1, 0, 1, (i128::MAX, 0));
    assert_eq!(overflow.checked_power(1, 3), None);
}

#[test]
fn lineage_reset_duplicates_empty_state_and_limits_have_disjoint_outcomes() {
    let first_lineage = Lineage::new(7, 1);
    let second_lineage = Lineage::new(7, 2);
    let empty = observation(first_lineage, 0, &[]);
    let mut tracker = ExactRecurrenceTracker::limited(first_lineage, 1, MAX_RECURRENCE_BYTES);
    assert_eq!(tracker.observe(empty.clone()), ObserveOutcome::Recorded);
    assert_eq!(
        tracker.observe(empty.clone()),
        ObserveOutcome::DuplicateGeneration
    );
    let next_empty = observation(first_lineage, 1, &[]);
    let certificate = tracker
        .observe(next_empty)
        .certificate()
        .or_invariant("empty state must have an exact zero-displacement recurrence");
    assert_eq!(certificate.delta(), (0, 0));

    let mismatch = observation(second_lineage, 2, &[(0, 0)]);
    assert_eq!(
        tracker.observe(mismatch),
        ObserveOutcome::Unavailable(RecurrenceUnavailable::LineageMismatch)
    );
    tracker.reset(second_lineage);
    assert_eq!(tracker.entry_count(), 0);
    assert_eq!(tracker.counters(), TrackerCounters::default());
    assert!(
        tracker.allocated_bytes() > 0,
        "reset must charge retained table capacity"
    );

    let mut entry_limited = ExactRecurrenceTracker::limited(second_lineage, 1, usize::MAX);
    assert_eq!(
        entry_limited.observe(observation(second_lineage, 0, &[(0, 0)])),
        ObserveOutcome::Recorded
    );
    assert_eq!(
        entry_limited.observe(observation(second_lineage, 1, &[(0, 0), (1, 0)])),
        ObserveOutcome::Unavailable(RecurrenceUnavailable::EntryLimit)
    );

    let mut byte_limited = ExactRecurrenceTracker::limited(second_lineage, 2, 1);
    assert_eq!(
        byte_limited.observe(observation(second_lineage, 0, &[(0, 0)])),
        ObserveOutcome::Unavailable(RecurrenceUnavailable::ByteLimit)
    );
    assert_eq!(byte_limited.entry_count(), 0);
}

#[test]
fn dag_witness_is_weak_value_identity_and_obeys_lineage() {
    assert!(!std::mem::needs_drop::<DagWitness>());
    let lineage = Lineage::new(11, 5);
    let witness = DagWitness::new(11, 5, 99, 12);
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    assert_eq!(
        tracker.observe(Observation::from_dag(lineage, 10, (-4, 8), witness)),
        ObserveOutcome::Recorded
    );
    let certificate = tracker
        .observe(Observation::from_dag(lineage, 14, (2, -1), witness))
        .certificate()
        .or_invariant("equal weak DAG identities must recur");
    assert_eq!(certificate.delta(), (6, -9));

    let different_arena =
        Observation::from_dag(lineage, 15, (0, 0), DagWitness::new(11, 6, 99, 12));
    assert_eq!(
        tracker.observe(different_arena),
        ObserveOutcome::Recorded,
        "DAG arena epoch is witness identity, independent of observation lineage"
    );
}

#[test]
fn dag_epoch_change_retires_stale_weak_evidence_but_not_lineage() {
    let lineage = Lineage::new(12, 5);
    let old = DagWitness::new(88, 1, 7, 4);
    let current = DagWitness::new(88, 2, 9, 4);
    let mut tracker = ExactRecurrenceTracker::limited(lineage, 1, MAX_RECURRENCE_BYTES);

    assert_eq!(
        tracker.observe(Observation::from_dag(lineage, 0, (0, 0), old)),
        ObserveOutcome::Recorded
    );
    let old_certificate = tracker
        .observe(Observation::from_dag(lineage, 1, (1, 0), old))
        .certificate()
        .or_invariant("old arena evidence should certify before retirement");
    assert_eq!(
        tracker.observe(Observation::from_dag(lineage, 2, (5, -3), current)),
        ObserveOutcome::Recorded,
        "new arena epoch must reclaim the old weak entry slot"
    );
    assert_eq!(tracker.entry_count(), 1);
    let certificate = tracker
        .observe(Observation::from_dag(lineage, 4, (7, 1), current))
        .certificate()
        .or_invariant("current arena evidence should still certify");
    assert_eq!(certificate.first_seen(), 2);
    assert_eq!(certificate.delta(), (2, 4));
    assert!(certificate.matches_lineage(lineage));
    assert!(old_certificate.matches_lineage(lineage));
    assert!(old_certificate.checked_power(2, 4).is_some());
    assert_eq!(tracker.counters().certificates_produced, 2);
    assert!(tracker.counters().partitions_hold());
}

#[test]
fn powered_displacement_builds_an_atomic_fallible_translation_candidate() {
    let source = grid(&[(-8, -1), (-1, 0), (7, 8)]);
    let expected = source.translated(13, -11);
    assert_eq!(
        source
            .try_translated(13, -11)
            .or_invariant("checked translation should fit"),
        expected
    );
    assert_eq!(
        source.try_translated(i128::MAX, 0),
        Err(crate::bitgrid::GridTranslationError::CoordinateOverflow)
    );
    assert_eq!(
        source.try_translated(0, i128::MIN),
        Err(crate::bitgrid::GridTranslationError::CoordinateOverflow)
    );

    let max_corner = grid(&[(i64::MAX, i64::MAX)]);
    assert_eq!(
        max_corner.try_translated(1, 0),
        Err(crate::bitgrid::GridTranslationError::CoordinateOverflow)
    );
    assert_eq!(
        max_corner.try_translated(0, 1),
        Err(crate::bitgrid::GridTranslationError::CoordinateOverflow)
    );
    assert_eq!(
        max_corner
            .try_translated(0, 0)
            .or_invariant("maximum live-cell coordinate should remain representable"),
        max_corner
    );

    let min_corner = grid(&[(i64::MIN, i64::MIN)]);
    assert_eq!(
        min_corner.try_translated(-1, 0),
        Err(crate::bitgrid::GridTranslationError::CoordinateOverflow)
    );
    assert_eq!(
        min_corner.try_translated(0, -1),
        Err(crate::bitgrid::GridTranslationError::CoordinateOverflow)
    );
}

#[test]
fn witness_work_limits_are_checked_before_packing() {
    let lineage = Lineage::new(13, 1);
    let mut too_many_cells = BitGrid::empty();
    for x in 0_i64..=i64::try_from(MAX_WITNESS_CELLS).or_invariant("cell witness limit exceeds i64")
    {
        too_many_cells.set(x, 0, true);
    }
    assert_eq!(
        Observation::from_grid(lineage, 0, &too_many_cells),
        Err(RecurrenceUnavailable::WitnessLimit)
    );

    let mut too_many_chunks = BitGrid::empty();
    for chunk in
        0_i64..=i64::try_from(MAX_WITNESS_CHUNKS).or_invariant("chunk witness limit exceeds i64")
    {
        too_many_chunks.set(chunk * 8, chunk * 8, true);
    }
    assert_eq!(
        Observation::from_grid(lineage, 0, &too_many_chunks),
        Err(RecurrenceUnavailable::WitnessLimit)
    );

    let mut tracker = ExactRecurrenceTracker::new(lineage);
    assert_eq!(
        tracker.observe_result(Observation::from_grid(lineage, 0, &too_many_cells)),
        ObserveOutcome::Unavailable(RecurrenceUnavailable::WitnessLimit)
    );
    let counters = tracker.counters();
    assert_eq!(counters.observations, 1);
    assert_eq!(counters.unavailable, 1);
    assert_eq!(counters.repeat_candidates, 0);
}

#[test]
fn default_lineages_are_process_unique() {
    let first = Lineage::default();
    let second = Lineage::fresh();
    assert_ne!(first, second);
    assert_eq!(first.next_epoch().map(|lineage| lineage.epoch), Some(1));
}

#[test]
fn public_tracker_caps_entries_and_retained_capacity() {
    let lineage = Lineage::new(17, 1);
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    for generation in
        0_u64..u64::try_from(MAX_RECURRENCE_ENTRIES).or_invariant("entry limit exceeds u64")
    {
        let witness = DagWitness::new(41, 3, generation + 1, 12);
        assert_eq!(
            tracker.observe(Observation::from_dag(lineage, generation, (0, 0), witness,)),
            ObserveOutcome::Recorded
        );
    }
    assert_eq!(tracker.entry_count(), MAX_RECURRENCE_ENTRIES);
    assert!(tracker.allocated_bytes() <= MAX_RECURRENCE_BYTES);
    assert_eq!(
        tracker.observe(Observation::from_dag(
            lineage,
            u64::try_from(MAX_RECURRENCE_ENTRIES).or_invariant("entry limit exceeds u64"),
            (0, 0),
            DagWitness::new(41, 3, u64::MAX, 12),
        )),
        ObserveOutcome::Unavailable(RecurrenceUnavailable::EntryLimit)
    );
    assert!(tracker.counters().partitions_hold());
}

#[test]
fn failed_witness_insert_cannot_grow_retained_storage_past_the_cap() {
    let lineage = Lineage::new(19, 1);
    let first = observation(lineage, 0, &[(0, 0)]);
    let retained_candidate = first.clone();
    let larger = observation(lineage, 1, &[(0, 0), (16, 0)]);

    let mut sizing_tracker = ExactRecurrenceTracker::limited(lineage, 4, usize::MAX);
    assert_eq!(sizing_tracker.observe(first), ObserveOutcome::Recorded);
    let budget = sizing_tracker.allocated_bytes();
    let mut tracker = ExactRecurrenceTracker::limited(lineage, 4, budget);
    assert_eq!(
        tracker.observe(retained_candidate),
        ObserveOutcome::Recorded
    );
    let retained = tracker.allocated_bytes();
    assert!(retained <= budget);

    assert_eq!(
        tracker.observe(larger),
        ObserveOutcome::Unavailable(RecurrenceUnavailable::ByteLimit)
    );
    assert_eq!(tracker.allocated_bytes(), retained);
    assert!(tracker.allocated_bytes() <= budget);
    let retry = observation(lineage, 1, &[(9, -4)]);
    let certificate = tracker
        .observe(retry)
        .certificate()
        .or_invariant("failed evidence capture must leave its generation retryable");
    assert_eq!(certificate.period(), 1);
    assert_eq!(certificate.delta(), (9, -4));
    assert!(tracker.counters().partitions_hold());
}
