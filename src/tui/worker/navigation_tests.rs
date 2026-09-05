use super::*;
use crate::RequiredExt;
use crate::tui::protocol::PreparedSource;

fn worker() -> WorkerHandle {
    worker_with_cells(&[
        (0, 0),
        (1, 0),
        (2, 0),
        (1000, 1000),
        (1001, 1000),
        (1002, 1000),
    ])
}

fn worker_with_cells(cells: &[crate::bitgrid::Cell]) -> WorkerHandle {
    let mut simulation = SimulationSession::new();
    simulation
        .try_load_hashlife_state(&BitGrid::from_cells(cells))
        .or_invariant("two-group fixture");
    let worker = start_worker_with_analysis_limits(
        PreparedSource {
            session: simulation,
            label: "two groups".to_string(),
        },
        1,
        1000,
        super::super::analysis::AnalysisLimits {
            capture: crate::hashlife::session::capture::CaptureLimits {
                residency: Duration::from_secs(5),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    worker
        .commands
        .send(ControlCommand::Pause)
        .or_invariant("pause");
    frame(&worker, |frame| {
        !frame.running
            && frame.status.contains("active=1/2")
            && !frame.status.contains("incomplete")
    });
    worker
}

#[test]
fn early_pentomino_quantum_two_keeps_current_frames_and_population() {
    let seed = crate::generators::pattern_by_name("r_pentomino").or_invariant("seed");
    let mut reference = crate::life::GameOfLife::new(seed.clone());
    let mut simulation = SimulationSession::new();
    simulation
        .try_load_hashlife_state(&seed)
        .or_invariant("load");
    let worker = start_worker(
        PreparedSource {
            session: simulation,
            label: "pentomino".into(),
        },
        1,
        1000,
    );
    worker
        .submit(ControlCommand::Pause)
        .or_invariant("pause at start");
    frame(&worker, |frame| {
        !frame.running && frame.status.contains("active=1/1")
    });
    worker.set_tuning(WorkerTuning {
        quantum: 2,
        interval_ms: 16,
    });
    worker
        .submit(ControlCommand::Resume)
        .or_invariant("run with quantum two");
    let mut generation = 0;
    while generation < 128 {
        let current = frame(&worker, |frame| frame.generation > generation);
        assert_eq!(current.quantum, 2);
        assert!(current.running);
        while reference.generation() < current.generation {
            reference.step();
        }
        assert_eq!(
            current.population,
            u128::try_from(reference.grid().population()).or_invariant("population"),
            "population at generation {} is stale",
            current.generation
        );
        for y in current.origin.1..current.origin.1 + 40 {
            for x in current.origin.0..current.origin.0 + 80 {
                assert_eq!(
                    current.grid.get(x, y),
                    reference.grid().get(x, y),
                    "stale viewport at generation {}, cell=({x},{y})",
                    current.generation
                );
            }
        }
        generation = current.generation;
    }
    worker
        .submit(ControlCommand::Pause)
        .or_invariant("pause remains responsive");
    frame(&worker, |frame| !frame.running);
    worker.shutdown().or_invariant("shutdown");
}

#[test]
fn high_quantum_navigation_publishes_current_selected_cells() {
    let glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let mut cells = glider.to_vec();
    cells.extend([(16_384, 0), (16_385, 0), (16_386, 0)]);
    let worker = worker_with_cells(&cells);
    for cycle in 0..4_u64 {
        worker
            .submit(ControlCommand::AdvanceBy(4096))
            .or_invariant("fast advance");
        let generation = (cycle * 2 + 1) * 4096;
        let result = frame(&worker, |frame| frame.generation == generation);
        let displacement = crate::bitgrid::Coord::try_from(generation / 4).or_invariant("motion");
        assert!(
            glider
                .iter()
                .all(|&(x, y)| result.grid.get(x + displacement, y + displacement)),
            "selected glider not visible at generation={generation}: {result:?}"
        );
        worker
            .submit(ControlCommand::FocusNext(request(cycle * 2 + 1)))
            .or_invariant("Tab");
        frame(&worker, |frame| frame.viewport_revision == cycle * 2 + 1);
        worker
            .submit(ControlCommand::AdvanceBy(4096))
            .or_invariant("pinned blinker advance");
        let result = frame(&worker, |frame| frame.generation == generation + 4096);
        assert!(
            (16_384..=16_386).all(|x| result.grid.get(x, 0)),
            "lost pinned blinker"
        );
        worker
            .submit(ControlCommand::FocusPrevious(request(cycle * 2 + 2)))
            .or_invariant("Shift-Tab");
        let result = frame(&worker, |frame| frame.viewport_revision == cycle * 2 + 2);
        assert!(
            !result.grid.is_empty(),
            "reverse navigation published an empty frame"
        );
    }
    worker.shutdown().or_invariant("shutdown");
}

#[test]
fn uncertain_displaced_selection_keeps_last_frame_but_manual_empty_views_are_allowed() {
    let glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
    let mut cells = glider.to_vec();
    cells.extend(glider.iter().map(|&(x, y)| (x + 100, y)));
    let worker = worker_with_cells(&cells);
    worker
        .submit(ControlCommand::AdvanceBy(1024))
        .or_invariant("ambiguous advance");
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Ok(WorkerEvent::Error(error)) = worker.events.recv_timeout(Duration::from_millis(10))
            && error.contains("retaining previous view")
        {
            break;
        }
        assert!(Instant::now() < deadline, "missing reacquisition notice");
    }
    if let Some(update) = worker.next_frame() {
        assert_eq!(
            update.generation, 0,
            "unverified new frame replaced valid historical view"
        );
        assert!(!update.grid.is_empty());
    }
    let current = worker.next_status().or_invariant("authoritative progress");
    assert_eq!(
        current.generation, 1024,
        "holding the view changed simulation progress"
    );
    let mut manual = request(100);
    manual.auto = false;
    manual.origin = Some((-10_000, -10_000));
    worker.set_viewport(manual);
    let result = frame(&worker, |frame| frame.viewport_revision == 100);
    assert_eq!(result.generation, 1024);
    assert!(result.grid.is_empty(), "manual views must not auto-follow");
    worker.shutdown().or_invariant("shutdown");
}

fn frame(worker: &WorkerHandle, predicate: impl Fn(&RenderSnapshot) -> bool) -> RenderSnapshot {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut last = None;
    loop {
        if let Some(frame) = worker.frames.take() {
            if predicate(&frame) {
                return frame;
            }
            last = Some(frame);
        }
        assert!(
            Instant::now() < deadline,
            "worker frame deadline; last={last:?}; status={:?}; events={:?}",
            worker.next_status(),
            worker.events.try_iter().collect::<Vec<_>>()
        );
        thread::sleep(Duration::from_millis(2));
    }
}

fn request(revision: u64) -> ViewportRequest {
    ViewportRequest {
        revision,
        width: 60,
        height: 20,
        origin: None,
        auto: true,
        recenter: false,
    }
}

#[test]
fn classification_completion_cannot_move_a_paused_manual_camera_or_publish_a_new_frame() {
    let worker = worker();
    let mut manual = request(100);
    manual.auto = false;
    manual.origin = Some((123, 456));
    worker.set_viewport(manual);
    let before = frame(&worker, |frame| frame.viewport_revision == 100);
    worker
        .submit(ControlCommand::Classify {
            source_revision: 1,
            scope: super::super::analysis::AnalysisScope::WholeUniverse,
        })
        .or_invariant("analysis request");
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Ok(WorkerEvent::Analysis(update)) =
            worker.events.recv_timeout(Duration::from_millis(20))
        {
            if matches!(
                update.status,
                super::super::analysis::AnalysisStatus::Completed(_)
            ) {
                assert_eq!(
                    update
                        .descriptor
                        .or_invariant("capture descriptor")
                        .generation,
                    before.generation
                );
                break;
            }
            assert!(
                update.status.is_active(),
                "unexpected analysis termination: {update:?}"
            );
        }
        assert!(
            Instant::now() < deadline,
            "classification completion was not delivered"
        );
    }
    assert!(
        worker.next_frame().is_none(),
        "classification completion triggered viewport sampling or focus selection"
    );
    assert_eq!(before.origin, (123, 456));
    assert_eq!(before.generation, 0);
    assert!(!before.running);
    worker.shutdown().or_invariant("shutdown");
}

#[test]
fn ordered_navigation_keeps_every_press_and_does_not_advance_simulation() {
    let worker = worker();
    worker
        .commands
        .send(ControlCommand::FocusNext(request(1)))
        .or_invariant("first Tab");
    worker
        .commands
        .send(ControlCommand::FocusNext(request(2)))
        .or_invariant("second Tab");
    let result = frame(&worker, |frame| frame.viewport_revision == 2);
    assert!(
        result.origin.0 < 500,
        "two presses must wrap; one coalesced press would select distant group: {result:?}"
    );
    assert!(
        result.status.contains("auto pinned"),
        "explicit auto focus was not pinned: {}",
        result.status
    );
    assert_eq!(result.generation, 0);
    assert!(!result.running, "navigation changed paused state");
    worker
        .commands
        .send(ControlCommand::FocusPrevious(request(3)))
        .or_invariant("previous");
    let result = frame(&worker, |frame| frame.viewport_revision == 3);
    assert!(
        result.origin.0 > 900,
        "previous did not wrap to other group"
    );
    worker.shutdown().or_invariant("shutdown");
}

#[test]
fn later_manual_intent_wins_over_older_navigation_and_resize_mailboxes() {
    let worker = worker();
    let mut manual = request(3);
    manual.auto = false;
    manual.origin = Some((2222, 3333));
    worker.viewport.replace(manual);
    worker
        .commands
        .send(ControlCommand::FocusNext(request(1)))
        .or_invariant("first Tab");
    worker
        .commands
        .send(ControlCommand::FocusNext(request(2)))
        .or_invariant("second Tab");
    worker
        .commands
        .send(ControlCommand::AdvanceBy(0))
        .or_invariant("ordered barrier");
    let result = frame(&worker, |frame| {
        frame.viewport_revision == 3 && frame.status.starts_with("advanced 0")
    });
    assert_eq!(
        result.origin,
        manual.origin.or_invariant("manual origin"),
        "late navigation overwrote newer pan"
    );
    assert!(result.status.contains("viewport=manual"));
    worker.viewport.replace(request(1));
    worker
        .commands
        .send(ControlCommand::StepOne)
        .or_invariant("next frame barrier");
    let result = frame(&worker, |frame| frame.generation == 1);
    assert_eq!(
        result.viewport_revision, 3,
        "old mailbox reset accepted geometry revision"
    );
    assert_eq!(result.origin, (2222, 3333));
    worker.shutdown().or_invariant("shutdown");
}

#[test]
fn auto_reset_is_ordered_even_when_presentation_toggles_coalesce() {
    let worker = worker();
    worker
        .commands
        .send(ControlCommand::FocusNext(request(1)))
        .or_invariant("pin second group");
    frame(&worker, |frame| {
        frame.viewport_revision == 1 && frame.status.contains("auto pinned")
    });
    worker
        .commands
        .send(ControlCommand::ResetAutoFocus)
        .or_invariant("release pin");
    let result = frame(&worker, |frame| {
        frame.status.contains("auto largest verified active")
    });
    assert!(
        result.origin.0 < 500,
        "reset did not apply deterministic largest-group tie"
    );
    assert_eq!(result.generation, 0);
    worker.shutdown().or_invariant("shutdown");
}
