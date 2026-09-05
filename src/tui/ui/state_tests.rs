use super::*;
use crate::RequiredExt;
use crate::tui::analysis::{AnalysisStatus, CaptureDescriptor};

fn status(sequence: u64, generation: u64, running: bool) -> WorkerStatus {
    WorkerStatus {
        worker_state_seq: sequence,
        source_revision: 1,
        state_revision: sequence,
        generation,
        running,
        quantum: 1,
    }
}

#[test]
fn newer_worker_state_wins_independently_of_rejected_viewport_frames() {
    let mut state = UiState::new(&Config::default());
    accept_status(&mut state, status(42, 100, true));
    accept_status(&mut state, status(41, 99, false));
    state.viewport_request = Some(ViewportRequest {
        revision: 5,
        width: 80,
        height: 20,
        origin: Some((10, 20)),
        auto: false,
        recenter: false,
    });
    accept_frame(&mut state, tests::snapshot(1, (0, 0)));
    assert!(state.frame.is_none(), "stale camera frame was accepted");
    assert_eq!(
        state.authoritative,
        Some(status(42, 100, true)),
        "stale pause or rejected frame regressed authoritative status"
    );
}

#[test]
fn historical_analysis_survives_advancement_but_terminal_cancellation_cannot_resurrect() {
    let mut state = UiState::new(&Config::default());
    accept_status(&mut state, status(200, 1_000_000, true));
    let (tx, rx) = mpsc::channel();
    let descriptor = CaptureDescriptor {
        request_id: 7,
        source_revision: 1,
        state_revision: 1,
        generation: 100,
        configuration: 1,
        scope: AnalysisScope::WholeUniverse,
    };
    let completed = AnalysisUpdate {
        request_id: 7,
        descriptor: Some(descriptor),
        status: AnalysisStatus::Completed(crate::classify::ClassificationReport::from_legacy(
            &crate::classify::Classification::Repeats {
                period: 1,
                first_seen: 0,
            },
        )),
    };
    tx.send(WorkerEvent::Analysis(completed.clone()))
        .or_invariant("historical result");
    drain_worker_events(&rx, &mut state).or_invariant("drain");
    assert_eq!(
        state.analysis,
        Some(completed.clone()),
        "continued simulation incorrectly invalidated historical evidence"
    );
    state.analysis = Some(AnalysisUpdate {
        status: AnalysisStatus::Cancelled,
        ..completed.clone()
    });
    tx.send(WorkerEvent::Analysis(completed))
        .or_invariant("late result");
    drain_worker_events(&rx, &mut state).or_invariant("drain");
    assert_eq!(
        state.analysis.as_ref().map(|update| &update.status),
        Some(&AnalysisStatus::Cancelled)
    );
    assert_eq!(state.authoritative, Some(status(200, 1_000_000, true)));
}

#[test]
fn changed_configuration_rejects_old_completion_without_changing_camera_or_generation() {
    let mut state = UiState::new(&Config::default());
    accept_status(&mut state, status(9, 100, true));
    state.manual_origin = Some((123, 456));
    let (tx, rx) = mpsc::channel();
    tx.send(WorkerEvent::AnalysisConfiguration(2))
        .or_invariant("configuration");
    tx.send(WorkerEvent::Analysis(AnalysisUpdate {
        request_id: 1,
        descriptor: Some(CaptureDescriptor {
            request_id: 1,
            source_revision: 1,
            state_revision: 1,
            generation: 0,
            configuration: 1,
            scope: AnalysisScope::WholeUniverse,
        }),
        status: AnalysisStatus::Completed(crate::classify::ClassificationReport::from_legacy(
            &crate::classify::Classification::Unknown { simulated: 1 },
        )),
    }))
    .or_invariant("obsolete completion");
    drain_worker_events(&rx, &mut state).or_invariant("drain");
    assert!(
        state.analysis.is_none(),
        "obsolete configuration resurrected evidence"
    );
    assert_eq!(state.authoritative, Some(status(9, 100, true)));
    assert_eq!(state.manual_origin, Some((123, 456)));
}

#[test]
fn stale_camera_result_cannot_revert_a_newer_manual_origin() {
    let mut state = UiState::new(&Config::default());
    state.auto_viewport = false;
    state.manual_origin = Some((123, 456));
    let mut current = tests::snapshot(1, (123, 456));
    current.camera_revision = 10;
    current.worker_state_seq = 12;
    accept_frame(&mut state, current);
    let mut old = tests::snapshot(1, (-500, -500));
    old.camera_revision = 9;
    old.worker_state_seq = 13;
    accept_frame(&mut state, old);
    assert_eq!(
        state.frame.as_ref().map(|frame| frame.origin),
        Some((123, 456))
    );
    assert_eq!(state.manual_origin, Some((123, 456)));
}
