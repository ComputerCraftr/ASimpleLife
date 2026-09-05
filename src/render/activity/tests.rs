use super::*;
use crate::RequiredExt;
use crate::bitgrid::BitGrid;
use crate::generators::pattern_by_name;
use crate::render::{ViewportController, ViewportMode};

fn session(cells: &[Cell]) -> SimulationSession {
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(cells))
        .or_invariant("load fixture");
    session
}

fn cells(name: &str, dx: Coord, dy: Coord) -> Vec<Cell> {
    pattern_by_name(name)
        .or_invariant("named fixture")
        .live_cells()
        .into_iter()
        .map(|(x, y)| (x + dx, y + dy))
        .collect()
}

fn scan(catalog: &mut ActivityCatalog, session: &mut SimulationSession, now: Instant) {
    for slice in 0..100 {
        catalog.refresh(session, now + Duration::from_millis(slice * 250), true);
        assert!(
            catalog.visits <= MAX_VISITS,
            "slice={slice} visits={}",
            catalog.visits
        );
        assert!(
            catalog.evaluations <= MAX_EVALUATIONS,
            "slice={slice} evaluations={}",
            catalog.evaluations
        );
        if catalog.finished {
            return;
        }
    }
    assert!(
        catalog.finished,
        "bounded fixture discovery failed to finish"
    );
}

#[test]
fn discovery_excludes_still_lifes_and_finds_interior_oscillators_without_materialization() {
    let mut input = cells("block", -10_000, -10_000);
    input.extend(cells("block", 10_000, 10_000));
    input.extend(cells("pulsar", 173, 431));
    let mut session = session(&input);
    let before = session.hashlife_runtime_stats();
    let mut catalog = ActivityCatalog::default();
    scan(&mut catalog, &mut session, Instant::now());
    assert_eq!(catalog.groups.len(), 1, "groups={:?}", catalog.groups);
    assert!(
        catalog.groups[0].bounds.0 >= 160,
        "wrong active focus: {:?}",
        catalog.groups
    );
    assert_eq!(
        session.hashlife_generation(),
        0,
        "inspection advanced simulation"
    );
    assert_eq!(
        session
            .hashlife_runtime_stats()
            .materialization
            .session_full_grid_materializations,
        before.materialization.session_full_grid_materializations,
        "discovery materialized universe"
    );
    assert_eq!(
        session.hashlife_runtime_stats(),
        before,
        "read-only discovery changed engine state or allocated nodes"
    );
}

#[test]
fn one_generation_probe_detects_phase_aliased_blinker_and_boundary_births() {
    let mut session = session(&[(7, 7), (8, 7), (9, 7)]);
    let mut catalog = ActivityCatalog::default();
    let now = Instant::now();
    scan(&mut catalog, &mut session, now);
    assert_eq!(catalog.groups.len(), 1, "cross-tile blinker must be active");
    session
        .advance_hashlife_root(1024)
        .or_invariant("oscillator phase-aligned jump");
    scan(&mut catalog, &mut session, now + Duration::from_secs(1));
    assert_eq!(
        catalog.groups.len(),
        1,
        "identical rendered phase is not inactivity"
    );
    assert_eq!(catalog.groups[0].population, 3);
    assert!(
        !catalog.refresh(&session, now + Duration::from_secs(50), false),
        "unchanged paused state rescanned"
    );
}

#[test]
fn automatic_focus_leaves_large_still_life_for_active_group() {
    let mut input = Vec::new();
    for x in 0..20 {
        input.extend(cells("block", x * 8, 0));
    }
    input.extend(cells("blinker", 1000, 1000));
    let mut session = session(&input);
    let mut viewport = ViewportController::new(60, 20).or_invariant("viewport");
    viewport.set_origin((0, 0));
    viewport.set_mode(ViewportMode::Auto);
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for slice in 0..30 {
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(slice * 250),
                true,
            )
            .or_invariant("focus discovery");
    }
    let (x, y) = viewport.origin().or_invariant("active focus origin");
    assert!(x > 900 && y > 900, "still-life lock-on: origin=({x},{y})");
    assert_eq!(session.hashlife_generation(), 0);
}

#[test]
fn navigation_cycles_both_directions_and_preserves_manual_mode() {
    let mut input = cells("blinker", 0, 0);
    input.extend(cells("blinker", 1000, 1000));
    let mut session = session(&input);
    let mut viewport = ViewportController::new(40, 15).or_invariant("viewport");
    viewport.set_origin((-20, -15));
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for slice in 0..10 {
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(slice * 250),
                true,
            )
            .or_invariant("discovery");
    }
    assert!(
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("next")
    );
    let first = viewport.origin();
    assert!(
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("next")
    );
    let second = viewport.origin();
    assert_ne!(
        first, second,
        "next did not move to another group: {focus:?}"
    );
    assert!(
        focus
            .navigate(&mut session, &mut viewport, true)
            .or_invariant("previous")
    );
    assert_eq!(viewport.origin(), first, "reverse cycle mismatch");
    assert_eq!(viewport.mode(), ViewportMode::Manual);
    assert_eq!(session.hashlife_generation(), 0);
}

#[test]
fn exhausted_probe_is_unknown_and_extreme_regions_do_not_overflow() {
    let session = session(&[(0, 0)]);
    let mut budget = 0;
    assert_eq!(
        session.inspect_viewport_region((0, 0, 0, 0), &mut budget),
        None
    );
    let mut budget = MAX_VISITS;
    let tile = (Coord::MAX.div_euclid(8), Coord::MIN.div_euclid(8));
    assert_eq!(
        session.inspect_viewport_neighborhood(tile, &mut budget),
        Some([0; 9])
    );
}

#[test]
fn discovery_cursor_survives_replacement_of_the_underlying_arena() {
    let mut input = Vec::new();
    for x in 0..100 {
        input.extend(cells("block", x * 32, 0));
    }
    input.extend(cells("blinker", 5000, 0));
    let mut session = session(&input);
    let mut catalog = ActivityCatalog::default();
    catalog.refresh(&session, Instant::now(), true);
    assert!(
        !catalog.finished,
        "fixture must leave an unfinished spatial cursor"
    );
    let snapshot = session
        .export_hashlife_snapshot_owned()
        .or_invariant("snapshot export")
        .or_invariant("loaded state");
    // A fresh engine has unrelated epoch-local handles. Only the owned
    // semantic snapshot crosses the boundary; the cursor keeps world regions.
    let mut replacement = SimulationSession::new();
    replacement
        .load_hashlife_snapshot_owned(&snapshot)
        .or_invariant("replacement arena");
    scan(&mut catalog, &mut replacement, Instant::now());
    assert!(
        catalog.groups.iter().any(|group| group.bounds.0 > 4900),
        "resume lost distant activity"
    );
    assert_eq!(replacement.hashlife_generation(), 0);
}

#[test]
fn group_that_becomes_a_still_life_releases_automatic_focus() {
    let mut session = session(&[(0, 0), (0, 1), (1, 0), (1000, 0), (1001, 0), (1002, 0)]);
    let mut viewport = ViewportController::new(40, 15).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for _ in 0..10 {
        focus
            .refresh(&mut session, &mut viewport, now, true)
            .or_invariant("initial discovery");
    }
    assert!(viewport.origin().is_some_and(|(x, _)| x < 500));
    for generation in 1..=3 {
        session
            .advance_hashlife_root(1)
            .or_invariant("settling generation");
        for _ in 0..10 {
            focus
                .refresh(
                    &mut session,
                    &mut viewport,
                    now + Duration::from_secs(generation),
                    true,
                )
                .or_invariant("settled group detection");
        }
    }
    assert!(
        viewport.origin().is_some_and(|(x, _)| x > 900),
        "new still life retained focus over active blinker: {focus:?}"
    );
}

#[test]
fn pinned_spaceship_is_matched_after_large_generation_skips() {
    let mut session = session(&cells("glider", 0, 0));
    let mut viewport = ViewportController::new(40, 15).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    assert!(
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("pin glider")
    );
    for jump in 1..=3 {
        session
            .advance_hashlife_root(1024)
            .or_invariant("spaceship jump");
        for _ in 0..10 {
            focus
                .refresh(&mut session, &mut viewport, Instant::now(), true)
                .or_invariant("jump focus");
        }
        assert!(
            focus.status(ViewportMode::Auto).contains("auto pinned"),
            "lost identity after jump {jump}"
        );
        let sample = viewport
            .sample_focus(&mut session)
            .or_invariant("spaceship viewport");
        assert_eq!(
            sample.grid.population(),
            5,
            "skipped-generation motion left focus behind at jump {jump}"
        );
    }
}

#[test]
fn out_of_phase_pulsars_keep_focus_across_population_swings() {
    let mut input = cells("pulsar", 0, 0);
    let mut second = crate::life::GameOfLife::new(pattern_by_name("pulsar").or_invariant("pulsar"));
    second.step();
    input.extend(
        second
            .grid()
            .live_cells()
            .into_iter()
            .map(|(x, y)| (x + 1000, y + 1000)),
    );
    let mut session = session(&input);
    let mut viewport = ViewportController::new(60, 20).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for slice in 0..20 {
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(slice * 250),
                true,
            )
            .or_invariant("initial discovery");
    }
    let origin = viewport.origin();
    for generation in 1..=36 {
        session
            .advance_hashlife_root(1)
            .or_invariant("pulsar generation");
        for slice in 0..4 {
            focus
                .refresh(
                    &mut session,
                    &mut viewport,
                    now + Duration::from_secs(10 + generation) + Duration::from_millis(slice * 250),
                    true,
                )
                .or_invariant("phase discovery");
        }
        assert_eq!(
            viewport.origin(),
            origin,
            "camera changed at pulsar generation {generation}"
        );
    }
}

#[test]
fn auto_navigation_pins_smaller_group_until_auto_is_reenabled() {
    let mut input = cells("pulsar", 0, 0);
    input.extend(cells("blinker", 1000, 1000));
    let mut session = session(&input);
    let mut viewport = ViewportController::new(60, 20).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for slice in 0..20 {
        focus
            .refresh(
                &mut session,
                &mut viewport,
                now + Duration::from_millis(slice * 250),
                true,
            )
            .or_invariant("discovery");
    }
    assert!(
        viewport.origin().is_some_and(|(x, _)| x < 500),
        "largest group not selected"
    );
    assert!(
        focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("navigate")
    );
    let pinned = viewport.origin();
    assert!(
        pinned.is_some_and(|(x, _)| x > 900),
        "next did not reach smaller group: {focus:?}"
    );
    for generation in 1..=10 {
        session.advance_hashlife_root(1).or_invariant("advance");
        for slice in 0..4 {
            focus
                .refresh(
                    &mut session,
                    &mut viewport,
                    now + Duration::from_secs(10 + generation) + Duration::from_millis(slice * 250),
                    true,
                )
                .or_invariant("pinned refresh");
        }
        assert_eq!(
            viewport.origin(),
            pinned,
            "auto lost explicit selection at generation {generation}"
        );
    }
    viewport.set_mode(ViewportMode::Manual);
    focus
        .refresh(
            &mut session,
            &mut viewport,
            now + Duration::from_secs(30),
            true,
        )
        .or_invariant("manual");
    viewport.set_mode(ViewportMode::Auto);
    focus
        .refresh(
            &mut session,
            &mut viewport,
            now + Duration::from_secs(31),
            true,
        )
        .or_invariant("auto");
    assert!(
        viewport.origin().is_some_and(|(x, _)| x < 500),
        "auto reset did not restore largest-group selection"
    );
}

#[test]
fn navigation_without_activity_preserves_manual_origin_and_generation() {
    let mut session = session(&cells("block", 0, 0));
    let mut viewport = ViewportController::new(40, 15).or_invariant("viewport");
    viewport.set_origin((333, 444));
    let mut focus = ActiveFocus::default();
    assert!(
        !focus
            .navigate(&mut session, &mut viewport, false)
            .or_invariant("inactive navigation")
    );
    assert_eq!(viewport.origin(), Some((333, 444)));
    assert_eq!(viewport.mode(), ViewportMode::Manual);
    assert_eq!(session.hashlife_generation(), 0);
}

#[test]
fn sustained_larger_activity_replaces_focus_only_after_hold_interval() {
    let mut input = cells("glider", 0, 0);
    input.extend(cells("r_pentomino", 1000, 1000));
    let mut session = session(&input);
    let mut viewport = ViewportController::new(60, 20).or_invariant("viewport");
    let mut focus = ActiveFocus::default();
    let now = Instant::now();
    for _ in 0..20 {
        focus
            .refresh(&mut session, &mut viewport, now, true)
            .or_invariant("initial equal populations");
    }
    assert!(
        viewport.origin().is_some_and(|(x, _)| x < 500),
        "deterministic tie must choose first group"
    );
    for generation in 1..=24 {
        session
            .advance_hashlife_root(1)
            .or_invariant("growing group generation");
        for _ in 0..10 {
            focus
                .refresh(
                    &mut session,
                    &mut viewport,
                    now + Duration::from_millis(generation * 250),
                    true,
                )
                .or_invariant("growth discovery");
        }
        if generation < 8 {
            assert!(
                viewport.origin().is_some_and(|(x, _)| x < 500),
                "switched before two-second hold at generation {generation}"
            );
        }
    }
    assert!(
        viewport.origin().is_some_and(|(x, _)| x > 900),
        "sustained larger group never acquired focus"
    );
}

#[test]
fn capped_discovery_continues_past_first_page_without_whole_grid_extraction() {
    let mut input = Vec::new();
    for x in 0..500 {
        input.extend(cells("block", x * 32, 0));
    }
    input.extend(cells("blinker", 20_000, 0));
    let session = session(&input);
    let mut catalog = ActivityCatalog::default();
    for slice in 0..2000 {
        catalog.refresh(&session, Instant::now(), true);
        assert!(catalog.visits <= DISCOVERY_VISITS);
        if slice == 0 {
            assert!(
                matches!(
                    catalog.observe(
                        &session,
                        &ActiveGroup {
                            cells: cells("blinker", 20_000, 0),
                            active: true,
                            bounds: (20_000, 0, 20_007, 7),
                            population: 3,
                            active_tile: (2500, 0),
                            generation: 0,
                        }
                    ),
                    GroupObservation::Complete(_)
                ),
                "background discovery starved foreground validation"
            );
        }
        assert!(
            catalog.visits <= MAX_VISITS && catalog.evaluations <= MAX_EVALUATIONS,
            "slice={slice}"
        );
        assert!(catalog.tiles.len() <= MAX_TILES);
        assert!(
            catalog.tiles.allocated_bytes() < 4 * 1024 * 1024,
            "inspection table exceeded budget"
        );
        if catalog.groups.iter().any(|group| group.bounds.0 > 19_000) {
            assert!(
                !catalog.complete,
                "discarded pages cannot prove complete discovery"
            );
            assert_eq!(
                session
                    .hashlife_runtime_stats()
                    .materialization
                    .session_full_grid_materializations,
                0
            );
            return;
        }
        if catalog.finished {
            break;
        }
    }
    assert!(
        catalog.groups.iter().any(|group| group.bounds.0 > 19_000),
        "bounded traversal stuck: regions={:?} pending={:?} tiles={} finished={} truncated={} visits={} evaluations={} groups={:?}",
        catalog.regions,
        catalog.pending,
        catalog.tiles.len(),
        catalog.finished,
        catalog.truncated,
        catalog.visits,
        catalog.evaluations,
        catalog.groups
    );
}
