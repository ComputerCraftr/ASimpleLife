use super::*;
use crate::{RequiredExt, bitgrid::BitGrid};
use std::time::{Duration, Instant};

fn source() -> SimulationSession {
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state(&BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]))
        .or_invariant("source");
    session
}
fn coordinator() -> AnalysisCoordinator {
    AnalysisCoordinator::new(AnalysisLimits {
        capture: CaptureLimits {
            residency: Duration::from_secs(5),
            ..CaptureLimits::default()
        },
        ..AnalysisLimits::default()
    })
}
fn descriptor() -> CaptureDescriptor {
    CaptureDescriptor {
        request_id: 1,
        source_revision: 1,
        state_revision: 1,
        generation: 0,
        configuration: 1,
        scope: AnalysisScope::WholeUniverse,
    }
}

#[test]
fn delayed_region_capture_uses_dequeue_generation_and_frozen_rectangle() {
    let mut simulation = source();
    let mut coordinator = coordinator();
    coordinator
        .request(1, AnalysisScope::IsolatedRegion((0, 0, 2, 0)))
        .or_invariant("request");
    simulation
        .advance_hashlife_root(1)
        .or_invariant("continue while choosing scope");
    coordinator.poll(&simulation, 1, 2, &AtomicBool::new(false));
    let mut captured = None;
    while let Some(update) = coordinator.take_update() {
        captured = update.descriptor.or(captured);
    }
    let capture = captured.or_invariant("capture acknowledgment");
    assert_eq!(capture.generation, 1);
    assert_eq!(capture.scope, AnalysisScope::IsolatedRegion((0, 0, 2, 0)));
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        coordinator.poll(&simulation, 1, 2, &AtomicBool::new(false));
        if let Some(update) = coordinator.take_update() {
            if let AnalysisStatus::Completed(report) = update.status {
                assert_eq!(
                    report.outcome,
                    crate::classify::ClassificationOutcome::Extinct,
                    "captured vertical phase must contain only one cell, not the old horizontal render"
                );
                break;
            }
            assert!(
                update.status.is_active(),
                "unexpected terminal result: {update:?}"
            );
        }
        assert!(Instant::now() < deadline, "analysis did not finish");
        std::thread::yield_now();
    }
    assert_eq!(
        simulation.hashlife_generation(),
        1,
        "analysis mutated authoritative generation"
    );
}

#[test]
fn cancellation_is_terminal_and_pending_replacement_waits_for_owned_work() {
    let simulation = source();
    let mut coordinator = coordinator();
    let (tx, rx) = std::sync::mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    coordinator.next_id = 1;
    coordinator.active = Some(Active {
        descriptor: descriptor(),
        cancelled: Arc::clone(&cancelled),
        terminal: false,
        thread: thread::spawn(move || {
            rx.recv().or_invariant("release old work");
            Ok(crate::classify::ClassificationReport::from_legacy(
                &crate::classify::Classification::Unknown { simulated: 0 },
            ))
        }),
    });
    coordinator
        .request(1, AnalysisScope::WholeUniverse)
        .or_invariant("replacement");
    coordinator.poll(&simulation, 1, 1, &AtomicBool::new(false));
    assert!(cancelled.load(Ordering::Relaxed));
    assert!(
        coordinator.pending.is_some(),
        "replacement captured while old work still owned storage"
    );
    let first = coordinator
        .take_update()
        .or_invariant("terminal supersession");
    assert_eq!(first.request_id, 1);
    assert_eq!(first.status, AnalysisStatus::Superseded);
    tx.send(()).or_invariant("release old work");
    let deadline = Instant::now() + Duration::from_secs(3);
    while coordinator.pending.is_some() {
        coordinator.poll(&simulation, 1, 1, &AtomicBool::new(false));
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    while let Some(update) = coordinator.take_update() {
        assert_ne!(
            update.request_id, 1,
            "late completion resurrected superseded work"
        );
    }
}

#[test]
fn stale_source_and_aggregate_denial_do_not_capture_or_mutate_simulation() {
    let simulation = source();
    let mut coordinator = AnalysisCoordinator::new(AnalysisLimits {
        aggregate_bytes: 1,
        ..AnalysisLimits::default()
    });
    coordinator
        .request(1, AnalysisScope::WholeUniverse)
        .or_invariant("request");
    coordinator.poll(&simulation, 1, 1, &AtomicBool::new(false));
    assert!(coordinator.active.is_none());
    assert!(
        coordinator
            .updates
            .iter()
            .any(|update| matches!(update.status, AnalysisStatus::Unavailable(_)))
    );
    coordinator.updates.clear();
    coordinator
        .request(1, AnalysisScope::WholeUniverse)
        .or_invariant("old source request");
    coordinator.poll(&simulation, 2, 2, &AtomicBool::new(false));
    assert!(
        coordinator
            .updates
            .iter()
            .any(|update| update.status == AnalysisStatus::Superseded)
    );
    assert_eq!(simulation.hashlife_generation(), 0);
}

#[test]
fn absolute_timestamp_overflow_retains_relative_evidence() {
    let mut capture = descriptor();
    capture.generation = u64::MAX - 20;
    let report = ClassificationReport::from_legacy(&crate::classify::Classification::DiesOut {
        at_generation: 42,
    });
    let update = AnalysisUpdate {
        request_id: 1,
        descriptor: Some(capture),
        status: AnalysisStatus::Completed(report),
    };
    let message = update.describe();
    assert!(
        message.contains("after 42") && message.contains("absolute=out of range"),
        "relative evidence was lost: {message}"
    );
}

#[test]
fn configuration_supersedes_pending_work_and_exhaustion_preserves_it() {
    let mut coordinator = coordinator();
    coordinator
        .request(1, AnalysisScope::WholeUniverse)
        .or_invariant("request");
    assert_eq!(coordinator.configure_generations(42), Ok(2));
    assert!(
        coordinator.pending.is_none(),
        "old configuration remained pending"
    );
    let terminal: Vec<_> = coordinator
        .updates
        .iter()
        .filter(|update| !update.status.is_active())
        .collect();
    assert_eq!(
        terminal.len(),
        1,
        "configuration change must terminate its request once"
    );
    assert_eq!(terminal[0].status, AnalysisStatus::Superseded);
    coordinator
        .request(1, AnalysisScope::WholeUniverse)
        .or_invariant("new request");
    coordinator.configuration = u64::MAX;
    assert!(coordinator.configure_generations(43).is_err());
    assert_eq!(coordinator.limits.generations, 42);
    assert!(
        coordinator.pending.is_some(),
        "failed configuration change destroyed accepted work"
    );
    coordinator.next_id = u64::MAX;
    assert!(
        coordinator
            .request(1, AnalysisScope::WholeUniverse)
            .is_err()
    );
    assert_eq!(
        coordinator.pending.as_ref().map(|request| request.id),
        Some(2)
    );
}
