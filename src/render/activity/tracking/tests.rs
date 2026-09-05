use super::*;
use crate::RequiredExt;
use crate::bitgrid::BitGrid;

fn group(name: &str, x: Coord, generation: u64) -> ActiveGroup {
    let mut cells = crate::generators::pattern_by_name(name)
        .or_invariant("fixture")
        .translated(x, 0)
        .live_cells();
    cells.sort_unstable();
    let bounds = cell_bounds(&cells).or_invariant("nonempty group");
    ActiveGroup {
        population: u32::try_from(cells.len()).or_invariant("fixture population"),
        cells,
        active: name != "block",
        bounds,
        active_tile: (x.div_euclid(8), 0),
        generation,
    }
}

#[test]
fn paired_history_requires_four_samples_and_three_full_window_wins() {
    let mut history = PairedHistory::new(7);
    for generation in 0..6 {
        let selected = group("blinker", 0, generation);
        let challenger = group("glider", 100, generation);
        assert_eq!(
            history.observe(&selected, &challenger),
            generation == 5,
            "generation={generation}: incomplete warm-up or early qualification"
        );
        assert!(
            !history.observe(&selected, &challenger),
            "same generation counted twice"
        );
    }
    assert!(
        !history.observe(&group("blinker", 0, 6), &group("glider", 100, 7)),
        "mixed-generation evidence entered paired history"
    );
}

#[test]
fn unknown_observations_never_release_selection_or_erase_negative_evidence() {
    let mut focus = ActiveFocus::default();
    focus.tracks.push(Track::new(1, group("blinker", 0, 0)));
    focus.selected = Some(1);
    focus.accept_selected(0, GroupObservation::Absent, 1);
    for generation in 2..20 {
        focus.accept_selected(0, GroupObservation::Incomplete, generation);
        assert_eq!(
            focus.missing, 1,
            "unknown observation changed absence count"
        );
        assert_eq!(focus.selected, Some(1));
    }
    focus.accept_selected(0, GroupObservation::Absent, 20);
    assert_eq!(focus.missing, 2);
    focus.accept_selected(0, GroupObservation::Absent, 20);
    assert_eq!(focus.missing, 2, "same-generation absence counted again");
}

#[test]
fn translation_envelope_does_not_accumulate_old_world_positions() {
    let mut track = Track::new(1, group("glider", 0, 0));
    for generation in 1..100 {
        let expected = group(
            "glider",
            Coord::try_from(generation * 100).or_invariant("position"),
            generation * 4,
        );
        let bounds = expected.bounds;
        track.observe(expected);
        assert_eq!(
            track.focus_bounds(),
            bounds,
            "generation={generation}: unbounded motion history"
        );
    }
}

#[test]
fn inactive_selected_group_without_replacement_keeps_camera() {
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(&[(0, 0), (0, 1), (1, 0)]))
        .or_invariant("settling seed");
    let mut viewport = ViewportController::new(40, 15).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for _ in 0..10 {
        focus
            .refresh(&mut session, &mut viewport, now, true)
            .or_invariant("initial focus");
    }
    let original = viewport.origin();
    for generation in 1..10 {
        session.advance_hashlife_root(1).or_invariant("advance");
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_secs(generation),
                true,
            )
            .or_invariant("settled focus");
        assert_eq!(
            viewport.origin(),
            original,
            "stable-only state moved camera"
        );
    }
    assert!(
        focus.status(ViewportMode::Auto).contains("stable"),
        "status={focus:?}"
    );
}

#[test]
fn stale_partial_catalog_does_not_evict_selected_identity() {
    let mut focus = ActiveFocus::default();
    focus.tracks.push(Track::new(1, group("glider", 0, 0)));
    focus.selected = Some(1);
    focus.catalog.groups.push(group("blinker", 1000, 99));
    focus.catalog.complete = false;
    focus.merge_discovery().or_invariant("partial merge");
    assert_eq!(focus.selected, Some(1));
    assert!(
        focus.selected_index().is_some(),
        "partial page discarded selected evidence"
    );
}

#[test]
fn complete_discovery_recovers_changed_membership_but_not_splits_or_merges() {
    for (name, complete, split, merge, observed_generation, recovers) in [
        ("complete", true, false, false, 2, true),
        ("partial", false, false, false, 2, false),
        ("stale", true, false, false, 1, false),
        ("split", true, true, false, 2, false),
        ("merge", true, false, true, 2, false),
    ] {
        let mut previous = group("blinker", 0, 0);
        previous.cells = (0..=6).map(|x| (x, 0)).collect();
        previous.bounds = (0, 0, 6, 0);
        previous.population = 7;
        let mut focus = ActiveFocus {
            selected: Some(1),
            tracks: vec![Track::new(1, previous)],
            evidence: FocusEvidence::Reacquiring,
            ..Default::default()
        };
        focus.catalog.generation = Some(observed_generation);
        focus.catalog.complete = complete;
        focus.catalog.groups = vec![group("glider", 0, observed_generation)];
        if split {
            focus
                .catalog
                .groups
                .push(group("blinker", 5, observed_generation));
        }
        if merge {
            focus.tracks.push(Track::new(2, group("glider", 0, 0)));
        }
        focus.recover_selected_from_discovery(2);
        assert_eq!(focus.selected, Some(1), "{name}: transferred identity");
        assert_eq!(
            focus.tracks[0].group.generation,
            if recovers { 2 } else { 0 },
            "{name}"
        );
        assert_eq!(focus.evidence == FocusEvidence::Active, recovers, "{name}");
    }
}

#[test]
fn paused_initial_discovery_waits_for_later_pages_before_selecting() {
    let mut focus = ActiveFocus {
        initial_generation: Some(0),
        ..ActiveFocus::default()
    };
    focus.tracks.push(Track::new(1, group("blinker", 0, 0)));
    let now = Instant::now();
    assert_eq!(
        focus.choose(Some(0), 0, now),
        None,
        "first page stole initial focus"
    );
    focus.tracks.push(Track::new(2, group("pulsar", 1000, 0)));
    focus.catalog.finished = true;
    focus.catalog.complete = true;
    let best = focus.best_candidate(0, true);
    assert_eq!(
        focus.choose(best, 0, now),
        Some(1),
        "paused scan failed to choose later larger group"
    );
}

#[test]
fn retained_evidence_checks_capacity_before_replacing_a_track() {
    let mut focus = ActiveFocus::default();
    focus.tracks.push(Track::new(1, group("glider", 0, 0)));
    let mut oversized = group("glider", 0, 1);
    oversized.cells.reserve_exact(MAX_RETAINED_CELLS);
    assert!(
        !focus.can_retain(0, &oversized),
        "spare allocated capacity escaped evidence accounting"
    );
    focus.selected = Some(1);
    focus.accept_selected(0, GroupObservation::Complete(oversized), 1);
    assert_eq!(
        focus.tracks[0].group.generation, 0,
        "over-budget observation replaced evidence"
    );
    assert_eq!(focus.evidence, FocusEvidence::Reacquiring);
}

#[test]
fn running_late_pentomino_does_not_switch_to_smaller_ash() {
    let seed = crate::generators::pattern_by_name("r_pentomino").or_invariant("seed");
    let mut session = SimulationSession::new();
    session.try_load_hashlife_state(&seed).or_invariant("load");
    session
        .advance_hashlife_root(100_000_000)
        .or_invariant("late ash");
    let mut viewport = ViewportController::new(78, 9).or_invariant("pane");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    let mut previous = None;
    for frame in 0..=520 {
        viewport
            .sample_focus(&mut session)
            .or_invariant("initial sample");
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(frame * 50),
                false,
            )
            .or_invariant("bounded running focus");
        assert_eq!(session.hashlife_generation(), 100_000_000 + frame);
        assert_eq!(
            session.hashlife_population_count(),
            Some(crate::hashlife::PopulationCount::Exact(116)),
            "late ash changed population at frame={frame}"
        );
        if frame >= 20 {
            assert!(
                focus.selected_index().is_some(),
                "frame={frame}: discovery never established focus"
            );
        }
        if let Some(index) = focus.selected_index() {
            let track = &focus.tracks[index];
            if let Some((id, population)) = previous
                && id != track.id
            {
                assert!(
                    track.group.population > population,
                    "frame={frame}: switched to equal/smaller ash; previous={previous:?} focus={focus:?}"
                );
            }
            previous = Some((track.id, track.group.population));
            let sample = viewport
                .sample_focus(&mut session)
                .or_invariant("selected frame");
            if track.group.generation == session.hashlife_generation() {
                assert!(
                    track
                        .group
                        .cells
                        .iter()
                        .any(|&(x, y)| sample.grid.get(x, y)),
                    "frame={frame}: frame contains cells, but not the selected organism"
                );
            }
            assert!(
                !sample.grid.is_empty(),
                "frame={frame}: selected live organism lost; origin={:?} focus={focus:?}",
                sample.origin
            );
        }
        session
            .advance_hashlife_root(1)
            .or_invariant("running generation");
    }
    assert!(previous.is_some(), "no complete group ever selected");
    for press in 0..12 {
        let before = focus.selected;
        assert!(
            focus
                .navigate(&mut session, &mut viewport, false)
                .or_invariant("single Tab"),
            "Tab press={press} failed to select the next late-pentomino group"
        );
        assert_ne!(
            focus.selected, before,
            "Tab press={press} reselected the same group"
        );
        assert!(
            focus
                .navigate(&mut session, &mut viewport, true)
                .or_invariant("single Shift-Tab"),
            "Shift-Tab press={press} failed to return"
        );
        assert_eq!(
            focus.selected, before,
            "opposite keys did not reverse selection at press={press}"
        );
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("next pair");
    }
    // Quantum 64 already moves a glider beyond this pane in one tick. Keep
    // this end-to-end regression small; the analytic fixture below exercises
    // larger camera deltas without repeating expensive universe evolution.
    for frame in 0..12_u64 {
        let quantum = 64;
        session
            .advance_hashlife_root(quantum)
            .or_invariant("fast tick");
        if frame % 3 == 0 {
            assert!(
                focus
                    .navigate(&mut session, &mut viewport, frame % 2 == 0)
                    .or_invariant("fast navigation")
            );
        }
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_secs(60) + Duration::from_millis(frame * 16),
                false,
            )
            .or_invariant("fast focus");
        let sample = viewport
            .sample_focus(&mut session)
            .or_invariant("fast sample");
        assert!(
            !sample.grid.is_empty(),
            "fast frame={frame} quantum={quantum}: empty selected view at {:?}, focus={focus:?}",
            sample.origin
        );
        assert!(
            focus.accepts_sample(viewport.mode(), session.hashlife_generation(), &sample),
            "fast frame={frame}: selected evidence did not follow the published generation"
        );
    }
    assert_eq!(
        session
            .hashlife_runtime_stats()
            .materialization
            .session_full_grid_materializations,
        0
    );
}

#[test]
fn fast_quantum_follows_selected_glider_between_discovery_intervals() {
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(&group("glider", 0, 0).cells))
        .or_invariant("glider");
    let mut viewport = ViewportController::new(40, 10).or_invariant("pane");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for _ in 0..20 {
        focus
            .refresh(&mut session, &mut viewport, now, true)
            .or_invariant("discovery");
    }
    let selected = focus.selected;
    assert!(selected.is_some());
    for frame in 1..=12 {
        session.advance_hashlife_root(4096).or_invariant("quantum");
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(frame * 16),
                false,
            )
            .or_invariant("follow");
        assert_eq!(focus.selected, selected);
        let sample = viewport.sample_focus(&mut session).or_invariant("sample");
        assert!(
            !sample.grid.is_empty(),
            "frame={frame}: time throttle lost moving selection"
        );
    }
}

#[test]
fn navigation_skips_historical_hints_for_the_same_current_glider() {
    let original = group("glider", 0, 0);
    let mut later = original.clone();
    later.generation = 4;
    later.cells = later.cells.iter().map(|&(x, y)| (x + 1, y + 1)).collect();
    later.bounds = cell_bounds(&later.cells).or_invariant("translated glider");
    let blinker = group("blinker", 1_000_000, 0);
    let mut cells = original.cells.clone();
    cells.extend(&blinker.cells);
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(&cells))
        .or_invariant("seed");
    session.advance_hashlife_root(4096).or_invariant("jump");
    let mut focus = ActiveFocus {
        tracks: vec![
            Track::new(1, original),
            Track::new(2, later),
            Track::new(3, blinker),
        ],
        selected: Some(1),
        ..ActiveFocus::default()
    };
    let mut viewport = ViewportController::new(40, 10).or_invariant("pane");
    assert!(
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("Tab")
    );
    assert_eq!(
        focus.selected,
        Some(3),
        "a duplicate hint consumed the key press"
    );
}

#[test]
fn analytic_large_quantum_frames_keep_pinned_gliders_and_blinkers_visible() {
    let glider = group("glider", 0, 0).cells;
    let blinker = group("blinker", 1000, 0).cells;
    let mut seed = glider.clone();
    seed.extend(&blinker);
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(&seed))
        .or_invariant("seed");
    let mut viewport = ViewportController::new(40, 10).or_invariant("pane");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for _ in 0..20 {
        focus
            .refresh(&mut session, &mut viewport, now, true)
            .or_invariant("discovery");
    }
    // The glider travels diagonally away from the stationary blinker. Every
    // quantum is a multiple of four, so these are independent exact target
    // states, not renderer predictions or a sampled output oracle.
    let mut generation = 0_u64;
    for (frame, quantum) in [64, 1024, 65_536, 1_048_576]
        .into_iter()
        .flat_map(|quantum| std::iter::repeat_n(quantum, 3))
        .enumerate()
    {
        generation = generation.checked_add(quantum).or_invariant("generation");
        let d = Coord::try_from(generation / 4).or_invariant("translation");
        let mut cells: Vec<_> = glider.iter().map(|&(x, y)| (x + d, y + d)).collect();
        cells.extend(&blinker);
        session
            .try_load_hashlife_state_at_generation(&BitGrid::from_cells(&cells), generation)
            .or_invariant("exact analytic state");
        if frame % 3 == 0 {
            assert!(
                focus
                    .navigate(&mut session, &mut viewport, frame % 2 == 0)
                    .or_invariant("navigation")
            );
        }
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(u64::try_from(frame + 1).or_invariant("frame") * 16),
                false,
            )
            .or_invariant("follow");
        let sample = viewport.sample_focus(&mut session).or_invariant("sample");
        assert!(
            focus.accepts_sample(viewport.mode(), generation, &sample),
            "frame={frame}, quantum={quantum}: selected evidence was not current and visible"
        );
    }
    assert_eq!(
        session
            .hashlife_runtime_stats()
            .materialization
            .session_full_grid_materializations,
        0
    );
}

#[test]
fn explicit_navigation_uses_fresh_budget_and_does_not_run_auto_selection() {
    let mut seed = group("glider", 0, 0).cells;
    seed.extend(group("blinker", 100, 0).cells);
    seed.extend(group("blinker", 200, 0).cells);
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(&seed))
        .or_invariant("load groups");
    let mut viewport = ViewportController::new(40, 15).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for _ in 0..30 {
        focus
            .refresh(&mut session, &mut viewport, now, true)
            .or_invariant("initial discovery");
    }
    let initial = focus.selected;
    session
        .advance_hashlife_root(1024)
        .or_invariant("stale discovery hints");
    focus.catalog.visits = MAX_VISITS;
    focus.catalog.evaluations = MAX_EVALUATIONS;
    assert!(
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("one Tab")
    );
    assert_ne!(focus.selected, initial, "Tab selected the old focus");
    assert!(
        focus
            .navigate(&mut session, &mut viewport, true)
            .or_invariant("one Shift-Tab")
    );
    assert_eq!(
        focus.selected, initial,
        "reverse navigation did not return to the original glider"
    );
    assert_eq!(session.hashlife_generation(), 1024);
}
