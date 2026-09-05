use super::*;
use crate::RequiredExt;
use crate::bitgrid::BitGrid;

fn session(cells: &[Cell]) -> SimulationSession {
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(cells))
        .or_invariant("load fixture");
    session
}

fn discover(cells: &[Cell]) -> (SimulationSession, ActivityCatalog) {
    let session = session(cells);
    let mut catalog = ActivityCatalog::default();
    let now = Instant::now();
    for slice in 0..128 {
        catalog.refresh(&session, now + Duration::from_millis(slice * 250), true);
        if catalog.finished {
            break;
        }
    }
    assert!(catalog.finished, "fixture discovery did not finish");
    (session, catalog)
}

#[test]
fn discovery_uses_exact_chebyshev_two_components() {
    let joined = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)];
    let (_, joined_catalog) = discover(&joined);
    assert_eq!(joined_catalog.groups.len(), 1);
    assert_eq!(joined_catalog.groups[0].cells, joined);

    let separate = [(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2)];
    let (_, separate_catalog) = discover(&separate);
    assert_eq!(separate_catalog.groups.len(), 2);
    assert_eq!(separate_catalog.groups[0].cells, separate[..3]);
    assert_eq!(separate_catalog.groups[1].cells, separate[3..]);
}

#[test]
fn component_membership_is_invariant_under_every_subchunk_translation() {
    let seed = [(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2)];
    for dx in -8..0 {
        for dy in -8..0 {
            let translated: Vec<_> = seed.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
            let (_, catalog) = discover(&translated);
            assert_eq!(
                catalog.groups.len(),
                2,
                "offset=({dx},{dy}) groups={:?}",
                catalog.groups
            );
            for (index, group) in catalog.groups.iter().enumerate() {
                assert_eq!(
                    group.cells,
                    translated[index * 3..index * 3 + 3],
                    "offset=({dx},{dy}) group={index}"
                );
            }
        }
    }
}

#[test]
fn unloaded_inspection_is_not_absence_evidence() {
    let (_, mut catalog) = discover(&[(0, 0), (0, 1), (0, 2)]);
    let previous = catalog.groups[0].clone();
    catalog.begin_observation();
    assert_eq!(
        catalog.observe(&SimulationSession::new(), &previous),
        GroupObservation::Failed
    );
}

#[test]
fn two_identical_spaceships_in_motion_region_require_reacquisition() {
    let glider = [(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)];
    let mut seed = glider.to_vec();
    seed.extend(glider.iter().map(|&(x, y)| (x + 100, y)));
    let (mut session, mut catalog) = discover(&seed);
    let previous = catalog.groups[0].clone();
    session
        .advance_hashlife_root(1024)
        .or_invariant("ambiguous spaceship jump");
    catalog.begin_observation();
    assert_eq!(
        catalog.observe(&session, &previous),
        GroupObservation::Incomplete,
        "one matching predicted location cannot prove unique identity across an unchecked motion region"
    );
    catalog.begin_observation();
    assert!(
        matches!(
            catalog.observe_navigation(&session, &previous),
            GroupObservation::Complete(_)
        ),
        "explicit selection should accept an exact current component without claiming historical identity"
    );
}

#[test]
fn activity_is_component_local_when_a_tile_contains_a_still_life_and_oscillator() {
    let cells = [(0, 0), (0, 1), (1, 0), (1, 1), (5, 0), (5, 1), (5, 2)];
    let (_, catalog) = discover(&cells);
    assert_eq!(catalog.groups.len(), 1);
    assert_eq!(catalog.groups[0].cells, cells[4..]);
    assert!(catalog.groups[0].active);
}

#[test]
fn component_is_not_published_until_two_cell_closure_is_known() {
    let session = session(&[(0, 0), (0, 1), (0, 2)]);
    let mut catalog = ActivityCatalog::default();
    let mut remaining = MAX_VISITS;
    let chunks = session
        .inspect_viewport_neighborhood((0, 0), &mut remaining)
        .or_invariant("center");
    catalog.tiles.insert(
        TileKey((0, 0)),
        Tile {
            bits: chunks[4],
            known: true,
        },
    );
    catalog.keys.push((0, 0));
    catalog.rebuild_groups(0);
    assert!(catalog.groups.is_empty(), "partial component was published");

    for ty in -1..=1 {
        for tx in -1..=1 {
            if (tx, ty) == (0, 0) {
                continue;
            }
            let chunks = session
                .inspect_viewport_neighborhood((tx, ty), &mut remaining)
                .or_invariant("halo");
            catalog.tiles.insert(
                TileKey((tx, ty)),
                Tile {
                    bits: chunks[4],
                    known: true,
                },
            );
            catalog.keys.push((tx, ty));
        }
    }
    catalog.rebuild_groups(0);
    assert_eq!(catalog.groups.len(), 1);
    assert_eq!(catalog.groups[0].cells, [(0, 0), (0, 1), (0, 2)]);
}

#[test]
fn observe_reports_exact_inactive_component() -> Result<(), &'static str> {
    let cells = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
    let session = session(&cells);
    let previous = ActiveGroup {
        cells: cells.clone(),
        active: false,
        bounds: (0, 0, 1, 1),
        population: 4,
        active_tile: (0, 0),
        generation: 0,
    };
    let mut catalog = ActivityCatalog::default();
    catalog.begin_observation();
    let GroupObservation::Inactive(observed) = catalog.observe(&session, &previous) else {
        return Err("still life was not observed as inactive");
    };
    assert_eq!(observed.cells, cells);
    assert!(!observed.active);
    Ok(())
}

#[test]
fn missing_predicted_membership_is_incomplete_not_absent() {
    let cells = vec![(0, 0), (0, 1), (0, 2)];
    let session = session(&[(100, 100), (100, 101), (101, 100), (101, 101)]);
    let previous = ActiveGroup {
        cells,
        active: true,
        bounds: (0, 0, 0, 2),
        population: 3,
        active_tile: (0, 0),
        generation: 0,
    };
    let mut catalog = ActivityCatalog::default();
    catalog.begin_observation();
    assert_eq!(
        catalog.observe(&session, &previous),
        GroupObservation::Incomplete
    );
}

#[test]
fn certified_glider_skip_relocates_without_engine_mutation_or_materialization()
-> Result<(), &'static str> {
    let cells = vec![(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let (mut session, mut catalog) = discover(&cells);
    let previous = catalog.groups[0].clone();
    session
        .advance_hashlife_root(1024)
        .or_invariant("glider skip");
    let before = session.hashlife_runtime_stats();
    catalog.begin_observation();
    let GroupObservation::Complete(observed) = catalog.observe(&session, &previous) else {
        return Err("certified glider relocation was unavailable");
    };
    assert_eq!(observed.generation, 1024);
    let mut expected = cells
        .iter()
        .map(|&(x, y)| (x + 256, y + 256))
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(observed.cells, expected);
    assert_eq!(session.hashlife_runtime_stats(), before);
    assert_eq!(before.materialization.session_full_grid_materializations, 0);
    assert!(catalog.visits <= MAX_VISITS && catalog.evaluations <= MAX_EVALUATIONS);
    Ok(())
}
