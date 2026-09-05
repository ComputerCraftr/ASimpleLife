use super::*;
use crate::generators::pattern_by_name;
use crate::life::step_grid;

fn glider_certificate(session: &mut SimulationSession) -> PeriodicCertificate {
    let seed = pattern_by_name("glider").or_invariant("glider fixture");
    session
        .try_load_hashlife_state(&seed)
        .or_invariant("load glider");
    let lineage = session.recurrence_lineage();
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    let first = Observation::from_grid(lineage, 0, &seed).or_invariant("first exact witness");
    assert_eq!(tracker.observe(first), ObserveOutcome::Recorded);
    let mut next = seed;
    for _ in 0..4 {
        next = step_grid(&next);
    }
    session
        .advance_hashlife_root(4)
        .or_invariant("advance glider one period");
    tracker
        .observe(Observation::from_grid(lineage, 4, &next).or_invariant("translated witness"))
        .certificate()
        .or_invariant("exact whole-state glider recurrence")
}

#[test]
fn shared_cell_certificate_applies_atomically_to_same_lineage_hashlife_state() {
    let mut session = SimulationSession::new();
    let certificate = glider_certificate(&mut session);
    let before = session.hashlife_sample_materializations();
    let stats = session
        .apply_hashlife_recurrence(certificate, 4_000)
        .or_invariant("checked recurrence geometry")
        .or_invariant("complete recurrence periods");
    assert_eq!(
        (
            stats.starting_generation,
            stats.completed_generations,
            stats.reached_generation
        ),
        (4, 3_996, 4_000)
    );
    let expected = pattern_by_name("glider")
        .or_invariant("glider fixture")
        .translated(1_000, 1_000);
    let actual = session
        .sample_hashlife_state_region(990, 990, 1_010, 1_010)
        .or_invariant("clipped recurrence result");
    assert_eq!(
        actual, expected,
        "certificate must move both origin axes by exact periods"
    );
    assert_eq!(
        session.hashlife_sample_materializations(),
        before,
        "certificate application and clipped observation must not materialize the full grid"
    );
}

#[test]
fn recurrence_certificate_powers_a_later_phase_then_advances_the_remainder() {
    let mut session = SimulationSession::new();
    let certificate = glider_certificate(&mut session);
    session
        .advance_hashlife_root(1)
        .or_invariant("advance into a different glider phase");
    let stats = session
        .apply_hashlife_recurrence(certificate, 4_000)
        .or_invariant("later phase shares the recurrence")
        .or_invariant("later phase has complete periods available");
    assert_eq!(
        (
            stats.starting_generation,
            stats.completed_generations,
            stats.reached_generation
        ),
        (5, 3_992, 3_997)
    );
    session
        .advance_hashlife_root(3)
        .or_invariant("advance the exact remainder");
    let expected = pattern_by_name("glider")
        .or_invariant("glider fixture")
        .translated(1_000, 1_000);
    assert_eq!(
        session.sample_hashlife_state_region(990, 990, 1_010, 1_010),
        Some(expected)
    );
    assert_eq!(session.hashlife_generation(), 4_000);
}

#[test]
fn source_replacement_rejects_old_certificate_without_committing_progress() {
    let mut session = SimulationSession::new();
    let certificate = glider_certificate(&mut session);
    let block = pattern_by_name("block").or_invariant("block fixture");
    session
        .try_load_hashlife_state(&block)
        .or_invariant("replace source");
    let revision = session.state_revision;
    assert_eq!(
        session.apply_hashlife_recurrence(certificate, 4_000),
        Err(RecurrenceUnavailable::LineageMismatch)
    );
    assert_eq!(session.hashlife_generation(), 0);
    assert_eq!(session.state_revision, revision);
    assert_eq!(
        session.sample_hashlife_state_region(-8, -8, 8, 8),
        Some(block)
    );
}

#[test]
fn generation_aware_load_cannot_reuse_a_previous_sources_certificate() {
    let mut session = SimulationSession::new();
    let certificate = glider_certificate(&mut session);
    let block = pattern_by_name("block").or_invariant("block fixture");
    session
        .try_load_hashlife_state_at_generation(&block, certificate.detected_at())
        .or_invariant("replace source at matching generation");
    let revision = session.state_revision;
    assert_eq!(
        session.apply_hashlife_recurrence(certificate, 4_000),
        Err(RecurrenceUnavailable::LineageMismatch)
    );
    assert_eq!(session.hashlife_generation(), certificate.detected_at());
    assert_eq!(session.state_revision, revision);
    assert_eq!(
        session.sample_hashlife_state_region(-8, -8, 8, 8),
        Some(block)
    );
}

#[test]
fn clipped_extraction_cannot_replace_the_authoritative_universe() {
    let mut session = SimulationSession::new();
    let certificate = glider_certificate(&mut session);
    let before = session
        .export_hashlife_snapshot()
        .or_invariant("snapshot before rejected conversion");
    let revision = session.state_revision;
    for policy in [
        GridExtractionPolicy::ViewportOnly,
        GridExtractionPolicy::BoundedRegion {
            min_x: -100,
            min_y: -100,
            max_x: -99,
            max_y: -99,
        },
    ] {
        assert_eq!(
            session.try_convert_to_cell(policy),
            Err(SimulationConversionError::PartialExtractionCannotBecomeAuthoritative),
            "partial policy {policy:?} must remain inspection-only"
        );
        assert!(session.hashlife_loaded());
        assert_eq!(session.state_revision, revision);
        assert_eq!(session.recurrence_lineage(), certificate.lineage());
        assert_eq!(
            session
                .export_hashlife_snapshot()
                .or_invariant("snapshot after rejected conversion"),
            before
        );
    }
    assert!(
        session
            .apply_hashlife_recurrence(certificate, 40)
            .or_invariant("unchanged source certificate remains usable")
            .is_some()
    );
}

#[test]
fn conversion_preserves_lineage_while_source_load_changes_it() {
    let mut session = SimulationSession::new();
    let seed = pattern_by_name("block").or_invariant("block fixture");
    session.load_cell_state(seed.clone(), 99);
    let lineage = session.recurrence_lineage();
    session
        .try_convert_to_hashlife()
        .or_invariant("convert cell authority");
    assert_eq!(session.recurrence_lineage(), lineage);
    session
        .try_convert_to_cell(GridExtractionPolicy::FullGridIfUnder {
            max_population: 4,
            max_chunks: 4,
            max_bounds_span: 8,
        })
        .or_invariant("convert root authority");
    assert_eq!(session.recurrence_lineage(), lineage);
    assert_eq!(session.cell_state(), Some((&seed, 99)));
    session.load_cell_state(seed, 99);
    assert_ne!(
        session.recurrence_lineage(),
        lineage,
        "same bytes loaded as a new source are still a new lineage"
    );
}

#[test]
fn overflowing_certificate_translation_commits_neither_origin_nor_generation() {
    let seed = pattern_by_name("glider")
        .or_invariant("glider fixture")
        .translated(Coord::MAX - 8_192, 0);
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&seed)
        .or_invariant("near-boundary seed loads");
    let lineage = session.recurrence_lineage();
    let mut tracker = ExactRecurrenceTracker::new(lineage);
    tracker.observe_result(Observation::from_grid(lineage, 0, &seed));
    let mut next = seed.clone();
    for _ in 0..4 {
        next = step_grid(&next);
    }
    session
        .advance_hashlife_root(4)
        .or_invariant("first complete near-boundary period");
    let certificate = tracker
        .observe_result(Observation::from_grid(lineage, 4, &next))
        .certificate()
        .or_invariant("near-boundary recurrence");
    let bounds = session.hashlife_bounds();
    let revision = session.state_revision;
    assert_eq!(
        session.apply_hashlife_recurrence(certificate, 131_072),
        Err(RecurrenceUnavailable::CoordinateOverflow)
    );
    assert_eq!(
        session.hashlife_generation(),
        4,
        "failed powered block advanced time"
    );
    assert_eq!(
        session.hashlife_bounds(),
        bounds,
        "failed powered block shifted space"
    );
    assert_eq!(
        session.state_revision, revision,
        "failed powered block published a revision"
    );
    session
        .advance_hashlife_root(4)
        .or_invariant("ordinary continuation after rejected skip");
    assert_eq!(session.hashlife_generation(), 8);
    assert_eq!(session.hashlife_bounds(), seed.translated(2, 2).bounds());
}
