use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use super::analysis::{AnalysisCoordinator, AnalysisStatus};

#[cfg(test)]
use crate::bitgrid::BitGrid;
use crate::engine::SimulationSession;
use crate::render::activity::ActiveFocus;
use crate::render::{ViewportController, ViewportMode};

use super::protocol::{
    ControlCommand, LatestValue, MAX_CONTINUOUS_QUANTUM, RenderSnapshot, SharedLatest,
    ViewportRequest, WorkerEvent, WorkerStatus, WorkerTuning,
};

const WORKER_IDLE_WAIT: Duration = Duration::from_millis(2);
const INTERRUPTIBLE_ADVANCE_CHUNK: u64 = 1 << 20;

mod client;
mod presentation;
use presentation::{ObservationToken, Presentation};
#[cfg(test)]
mod navigation_tests;
use client::{CommandClient, Envelope};

pub struct WorkerHandle {
    commands: Arc<CommandClient>,
    events: Receiver<WorkerEvent>,
    tuning: SharedLatest<WorkerTuning>,
    viewport: SharedLatest<ViewportRequest>,
    frames: SharedLatest<RenderSnapshot>,
    status: SharedLatest<WorkerStatus>,
    stop_requested: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl WorkerHandle {
    pub fn shutdown(mut self) -> Result<(), String> {
        self.stop_requested.store(true, Ordering::Relaxed);
        let _ = self.commands.send(ControlCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|panic| format!("simulation worker panicked: {}", panic_message(panic)))?;
        }
        Ok(())
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send + 'static>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.stop_requested.store(true, Ordering::Relaxed);
        let _ = self.commands.send(ControlCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct WorkerState {
    simulation: SimulationSession,
    source: String,
    source_revision: u64,
    running: bool,
    quantum: u64,
    interval: Duration,
    presentation: Presentation,
    status: String,
    analysis: AnalysisCoordinator,
    state_revision: u64,
    state_seq: u64,
    output: String,
}

impl WorkerState {
    fn observation(&self) -> ObservationToken {
        ObservationToken {
            source_revision: self.source_revision,
            state_revision: self.state_revision,
            generation: self.simulation.hashlife_generation(),
        }
    }
}

pub fn start_worker(
    initial: super::protocol::PreparedSource,
    initial_quantum: u64,
    initial_interval_ms: u64,
) -> WorkerHandle {
    start_worker_with_analysis_limits(
        initial,
        initial_quantum,
        initial_interval_ms,
        super::analysis::AnalysisLimits::default(),
    )
}

pub(crate) fn start_worker_with_analysis_limits(
    initial: super::protocol::PreparedSource,
    initial_quantum: u64,
    initial_interval_ms: u64,
    analysis_limits: super::analysis::AnalysisLimits,
) -> WorkerHandle {
    let (command_tx, command_rx) = mpsc::channel::<Envelope>();
    let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();
    let tuning = Arc::new(LatestValue::default());
    let viewport = Arc::new(LatestValue::default());
    let frames = Arc::new(LatestValue::default());
    let status = Arc::new(LatestValue::default());
    let worker_status = Arc::clone(&status);
    let pause_requested = Arc::new(AtomicBool::new(false));
    let stop_requested = Arc::new(AtomicBool::new(false));
    let worker_tuning = Arc::clone(&tuning);
    let worker_viewport = Arc::clone(&viewport);
    let worker_frames = Arc::clone(&frames);
    let command_client = Arc::new(CommandClient::new(command_tx, Arc::clone(&pause_requested)));
    let worker_client = Arc::clone(&command_client);
    let worker_stop = Arc::clone(&stop_requested);

    let thread = thread::spawn(move || {
        run_worker(
            initial,
            initial_quantum,
            initial_interval_ms,
            WorkerPorts {
                commands: command_rx,
                events: event_tx,
                tuning: worker_tuning,
                viewport: worker_viewport,
                frames: worker_frames,
                status: worker_status,
                command_client: worker_client,
                stop_requested: worker_stop,
                analysis_limits,
            },
        );
    });

    WorkerHandle {
        commands: command_client,
        events: event_rx,
        tuning,
        viewport,
        frames,
        status,
        stop_requested,
        thread: Some(thread),
    }
}

struct WorkerPorts {
    analysis_limits: super::analysis::AnalysisLimits,
    commands: Receiver<Envelope>,
    events: Sender<WorkerEvent>,
    tuning: SharedLatest<WorkerTuning>,
    viewport: SharedLatest<ViewportRequest>,
    frames: SharedLatest<RenderSnapshot>,
    status: SharedLatest<WorkerStatus>,
    command_client: Arc<CommandClient>,
    stop_requested: Arc<AtomicBool>,
}

fn run_worker(
    initial: super::protocol::PreparedSource,
    initial_quantum: u64,
    initial_interval_ms: u64,
    ports: WorkerPorts,
) {
    let WorkerPorts {
        analysis_limits,
        commands,
        events,
        tuning,
        viewport,
        frames,
        status,
        command_client,
        stop_requested,
    } = ports;
    let pause_requested = Arc::clone(&command_client.interrupt);
    let viewport_controller = match ViewportController::new(80, 20) {
        Ok(controller) => controller,
        Err(error) => {
            let _ = events.send(WorkerEvent::Error(error.to_string()));
            let _ = events.send(WorkerEvent::Stopped);
            return;
        }
    };
    let mut state = WorkerState {
        simulation: initial.session,
        source: initial.label,
        source_revision: 1,
        running: true,
        quantum: initial_quantum.clamp(1, MAX_CONTINUOUS_QUANTUM),
        interval: Duration::from_millis(initial_interval_ms.max(1)),
        presentation: Presentation::new(viewport_controller),
        status: "running".to_string(),
        analysis: AnalysisCoordinator::new(analysis_limits),
        state_revision: 1,
        state_seq: 0,
        output: String::new(),
    };
    let mut dirty = true;
    let mut last_tick = Instant::now();

    loop {
        if stop_requested.load(Ordering::Relaxed) {
            break;
        }
        if state.state_seq == u64::MAX || state.state_revision == u64::MAX {
            let _ = events.send(WorkerEvent::Error(
                "worker revisions exhausted; no further state was committed".into(),
            ));
            break;
        }
        let before_generation = state.simulation.hashlife_generation();
        let before_source = state.source_revision;
        if let Some(next) = tuning.take() {
            state.quantum = next.quantum.clamp(1, MAX_CONTINUOUS_QUANTUM);
            state.interval = Duration::from_millis(next.interval_ms.max(1));
            dirty = true;
        }
        if let Some(next) = viewport.take() {
            if !apply_viewport(&mut state, next) {
                let _ = events.send(WorkerEvent::Error(state.status.clone()));
            }
            dirty = true;
        }

        match commands.try_recv() {
            Ok(envelope) => {
                let analysis_only = matches!(
                    envelope.command,
                    ControlCommand::Classify { .. }
                        | ControlCommand::CancelClassification
                        | ControlCommand::ConfigureAnalysis { .. }
                );
                command_client.acknowledge(envelope.id);
                let command = envelope.command;
                if handle_command(
                    command,
                    &mut state,
                    &events,
                    &command_client.pause,
                    &stop_requested,
                ) {
                    break;
                }
                let _ = events.send(WorkerEvent::Acknowledged {
                    request_id: envelope.id,
                });
                dirty |= !analysis_only;
            }
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }

        if state.running
            && !pause_requested.load(Ordering::Relaxed)
            && last_tick.elapsed() >= state.interval
        {
            let requested = state.quantum;
            match advance_interruptibly(
                &mut state.simulation,
                requested,
                &pause_requested,
                &stop_requested,
            ) {
                Ok(completed) => {
                    state.status = if completed == requested {
                        "running".to_string()
                    } else {
                        format!("interrupted after {completed} committed generations")
                    };
                    // Interrupting a pacing quantum is not itself a run-state
                    // command. Preserve the prior state until its ordered
                    // Pause/Toggle arrives, otherwise Toggle would resume it.
                }
                Err(error) => {
                    state.running = false;
                    state.status = format!("advance failed: {error}");
                    let _ = events.send(WorkerEvent::Error(state.status.clone()));
                }
            }
            last_tick = Instant::now();
            dirty = true;
        }

        if before_generation != state.simulation.hashlife_generation()
            || before_source != state.source_revision
        {
            let Some(revision) = state.state_revision.checked_add(1) else {
                let _ = events.send(WorkerEvent::Error("state revisions exhausted".into()));
                break;
            };
            state.state_revision = revision;
            state.presentation.pending = true;
        }
        let observation = state.observation();
        if state.presentation.pending
            && let Err(error) = state
                .presentation
                .prepare_sample(&mut state.simulation, observation)
        {
            state.presentation.pending = false;
            let _ = events.send(WorkerEvent::Error(error));
        }
        if state.presentation.pending && !stop_requested.load(Ordering::Relaxed) {
            let observation = state.observation();
            let base_revision = state.presentation.revision;
            let mut candidate = state.presentation.viewport.clone();
            let selection = state.presentation.focus.selection_checkpoint();
            match state.presentation.focus.refresh(
                &mut state.simulation,
                &mut candidate,
                Instant::now(),
                false,
            ) {
                Ok(changed) => {
                    if candidate.origin() != state.presentation.viewport.origin()
                        || candidate.dimensions() != state.presentation.viewport.dimensions()
                        || candidate.mode() != state.presentation.viewport.mode()
                    {
                        match state.presentation.prepare_camera(
                            candidate,
                            &mut state.simulation,
                            observation,
                            base_revision,
                        ) {
                            Ok(prepared) => {
                                let current = state.observation();
                                match state.presentation.commit_camera(prepared, current) {
                                    Ok(true) => dirty = true,
                                    Ok(false) => {
                                        state.presentation.focus.restore_selection(selection);
                                    }
                                    Err(error) => {
                                        state.presentation.focus.restore_selection(selection);
                                        let _ = events.send(WorkerEvent::Error(error));
                                    }
                                }
                            }
                            Err(error) => {
                                state.presentation.focus.restore_selection(selection);
                                let _ = events.send(WorkerEvent::Error(error));
                            }
                        }
                    }
                    dirty |= changed;
                }
                Err(error) => {
                    state.presentation.focus.restore_selection(selection);
                    let _ = events.send(WorkerEvent::Error(error.to_string()));
                }
            }
            state.presentation.pending = state.presentation.focus.discovery_pending();
        }
        state.analysis.poll(
            &state.simulation,
            state.source_revision,
            state.state_revision,
            &pause_requested,
        );
        while let Some(update) = state.analysis.take_update() {
            let _ = events.send(WorkerEvent::Analysis(update));
        }
        if dirty {
            let Some(sequence) = state.state_seq.checked_add(1) else {
                let _ = events.send(WorkerEvent::Error("worker state sequence exhausted".into()));
                break;
            };
            state.state_seq = sequence;
            status.replace(WorkerStatus {
                worker_state_seq: sequence,
                source_revision: state.source_revision,
                state_revision: state.state_revision,
                generation: state.simulation.hashlife_generation(),
                running: state.running,
                quantum: state.quantum,
            });
            match render_snapshot(&mut state) {
                Ok(frame) => frames.replace(frame),
                Err(error) => {
                    let _ = events.send(WorkerEvent::Error(error));
                }
            }
            dirty = false;
        }
        thread::sleep(WORKER_IDLE_WAIT);
    }
    state.simulation.finish();
    let _ = events.send(WorkerEvent::Stopped);
}

fn apply_viewport(state: &mut WorkerState, next: ViewportRequest) -> bool {
    let observation = state.observation();
    if let Err(error) = state
        .presentation
        .apply_request(next, &mut state.simulation, observation)
    {
        state.status = error;
        return false;
    }
    true
}

fn handle_command(
    command: ControlCommand,
    state: &mut WorkerState,
    events: &Sender<WorkerEvent>,
    pause_requested: &Arc<AtomicBool>,
    stop_requested: &Arc<AtomicBool>,
) -> bool {
    match command {
        ControlCommand::Pause => {
            state.running = false;
            state.status = "paused".to_string();
        }
        ControlCommand::ToggleRunning => {
            state.running = !state.running;
            state.status = if state.running { "running" } else { "paused" }.into();
        }
        ControlCommand::Resume => {
            state.running = true;
            state.status = "running".to_string();
        }
        ControlCommand::StepOne => {
            run_exact_delta(state, 1, events, pause_requested, stop_requested)
        }
        ControlCommand::AdvanceBy(delta) => {
            run_exact_delta(state, delta, events, pause_requested, stop_requested)
        }
        ControlCommand::AdvanceTo(target) => {
            let current = state.simulation.hashlife_generation();
            if target < current {
                let message = format!("target {target} already passed at generation {current}");
                state.status = message.clone();
                let _ = events.send(WorkerEvent::Error(message));
            } else {
                run_exact_delta(
                    state,
                    target - current,
                    events,
                    pause_requested,
                    stop_requested,
                );
            }
        }
        ControlCommand::FocusNext(request) | ControlCommand::FocusPrevious(request) => {
            let previous = matches!(command, ControlCommand::FocusPrevious(_));
            // Stale resize mailboxes cannot overwrite this ordered navigation.
            // Accepted navigation still executes even if a later presentation
            // request has already superseded its display acknowledgement.
            let superseded = request.revision < state.presentation.request.revision;
            if !apply_viewport(state, request) {
                let _ = events.send(WorkerEvent::Error(state.status.clone()));
                return false;
            }
            let base_revision = state.presentation.revision;
            let observation = state.observation();
            let mut candidate = state.presentation.viewport.clone();
            let selection = state.presentation.focus.selection_checkpoint();
            match state
                .presentation
                .focus
                .navigate(&mut state.simulation, &mut candidate, previous)
            {
                Ok(true) => {
                    if superseded {
                        state.presentation.focus.restore_selection(selection);
                        state.status = "active group selection was superseded".to_string();
                    } else {
                        match state.presentation.prepare_camera(
                            candidate,
                            &mut state.simulation,
                            observation,
                            base_revision,
                        ) {
                            Ok(prepared) => {
                                let current = state.observation();
                                match state.presentation.commit_camera(prepared, current) {
                                    Ok(true) => state.status = "active group selected".to_string(),
                                    Ok(false) => {
                                        state.presentation.focus.restore_selection(selection);
                                        state.status =
                                            "active group selection was superseded".to_string()
                                    }
                                    Err(error) => {
                                        state.presentation.focus.restore_selection(selection);
                                        state.status = error;
                                    }
                                }
                            }
                            Err(error) => {
                                state.presentation.focus.restore_selection(selection);
                                state.status = error;
                            }
                        }
                    }
                }
                Ok(false) => {
                    state.presentation.focus.restore_selection(selection);
                    state.status = "no next active group found".to_string();
                }
                Err(error) => {
                    state.presentation.focus.restore_selection(selection);
                    state.status = error.to_string();
                }
            }
        }
        ControlCommand::ResetAutoFocus => {
            let Some(revision) = state.presentation.revision.checked_add(1) else {
                let _ = events.send(WorkerEvent::Error("camera revisions exhausted".into()));
                return true;
            };
            state.presentation.revision = revision;
            state.presentation.focus.release_selection();
            state.presentation.pending = true;
        }
        ControlCommand::Save(path) => save_committed_snapshot(state, path, events),
        ControlCommand::ReplaceSource(prepared) => {
            if state.source_revision == u64::MAX {
                let _ = events.send(WorkerEvent::Error("source revisions exhausted".into()));
                return true;
            }
            state.simulation.finish();
            state.simulation = prepared.session;
            state.presentation.invalidate_samples();
            state.source = prepared.label;
            state.source_revision += 1;
            state.presentation.focus = ActiveFocus::default();
            state.analysis.cancel(AnalysisStatus::Superseded);
            if let Ok(controller) = ViewportController::new(
                usize::from(state.presentation.request.width),
                usize::from(state.presentation.request.height),
            ) {
                let observation = state.observation();
                let base_revision = state.presentation.revision;
                match state.presentation.prepare_camera(
                    controller,
                    &mut state.simulation,
                    observation,
                    base_revision,
                ) {
                    Ok(prepared) => {
                        let current = state.observation();
                        if let Err(error) = state.presentation.commit_camera(prepared, current) {
                            let _ = events.send(WorkerEvent::Error(error));
                        }
                    }
                    Err(error) => {
                        let _ = events.send(WorkerEvent::Error(error));
                    }
                }
            }
            state.output.clear();
            state.status = "source replaced".to_string();
            let _ = events.send(WorkerEvent::SourceReplaced {
                label: state.source.clone(),
                revision: state.source_revision,
            });
        }
        ControlCommand::Classify {
            source_revision,
            scope,
        } => {
            if let Err(error) = state.analysis.request(source_revision, scope) {
                let _ = events.send(WorkerEvent::Error(error.to_string()));
            }
        }
        ControlCommand::CancelClassification => {
            state.analysis.cancel(AnalysisStatus::Cancelled);
        }
        ControlCommand::ConfigureAnalysis { max_generations } => {
            match state.analysis.configure_generations(max_generations) {
                Ok(revision) => {
                    let _ = events.send(WorkerEvent::AnalysisConfiguration(revision));
                }
                Err(error) => {
                    let _ = events.send(WorkerEvent::Error(error.to_string()));
                }
            }
        }
        ControlCommand::Shutdown => {
            stop_requested.store(true, Ordering::Relaxed);
            return true;
        }
    }
    false
}

fn run_exact_delta(
    state: &mut WorkerState,
    delta: u64,
    events: &Sender<WorkerEvent>,
    pause: &Arc<AtomicBool>,
    stop: &Arc<AtomicBool>,
) {
    state.running = false;
    match advance_interruptibly(&mut state.simulation, delta, pause, stop) {
        Ok(completed) => {
            state.status = format!("advanced {completed} of {delta} requested generation(s)");
            let _ = events.send(WorkerEvent::CommandCompleted(state.status.clone()));
        }
        Err(error) => {
            state.status = format!("advance failed: {error:?}");
            let _ = events.send(WorkerEvent::Error(state.status.clone()));
        }
    }
}

fn advance_interruptibly(
    simulation: &mut SimulationSession,
    requested: u64,
    pause_requested: &Arc<AtomicBool>,
    stop_requested: &Arc<AtomicBool>,
) -> Result<u64, String> {
    let mut completed = 0_u64;
    while completed < requested {
        if pause_requested.load(Ordering::Relaxed) || stop_requested.load(Ordering::Relaxed) {
            break;
        }
        let remaining = requested - completed;
        let chunk = remaining.min(INTERRUPTIBLE_ADVANCE_CHUNK);
        match simulation
            .advance_hashlife_root_controlled(chunk, Some((pause_requested, stop_requested)))
        {
            Ok(stats) => completed = completed.saturating_add(stats.completed_generations),
            Err(error) => {
                completed = completed.saturating_add(error.completed_generations());
                if matches!(
                    error,
                    crate::hashlife::HashLifeAdvanceError::Cancelled { .. }
                ) {
                    return Ok(completed);
                }
                return Err(format!("{error:?}; committed={completed}"));
            }
        }
    }
    Ok(completed)
}

fn save_committed_snapshot(state: &mut WorkerState, path: PathBuf, events: &Sender<WorkerEvent>) {
    let generation = state.simulation.hashlife_generation();
    let mut bytes = Vec::new();
    match state.simulation.write_hashlife_snapshot(&mut bytes) {
        Ok(true) => {
            state.status = format!("saving committed generation {generation}");
            let sender = events.clone();
            thread::spawn(move || match write_atomic(&path, &bytes) {
                Ok(()) => {
                    let _ = sender.send(WorkerEvent::Saved { path, generation });
                }
                Err(error) => {
                    let _ = sender.send(WorkerEvent::Error(format!("save failed: {error}")));
                }
            });
        }
        Ok(false) => {
            state.status = "save failed: no loaded state".to_string();
            let _ = events.send(WorkerEvent::Error(state.status.clone()));
        }
        Err(error) => {
            state.status = format!("save failed: {error}");
            let _ = events.send(WorkerEvent::Error(state.status.clone()));
        }
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "save path must name a file".to_string())?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    let file = File::create(&temporary)
        .map_err(|error| format!("failed to create temporary snapshot: {error}"))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .and_then(|()| writer.flush())
        .map_err(|error| format!("failed to write temporary snapshot: {error}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| format!("failed to sync temporary snapshot: {error}"))?;
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!("failed to commit snapshot: {error}")
    })
}

fn render_snapshot(state: &mut WorkerState) -> Result<RenderSnapshot, String> {
    let observation = state.observation();
    let generation = state.simulation.hashlife_generation();
    let population = state
        .simulation
        .hashlife_population_count()
        .map_or(0, |count| count.lower_bound());
    let sample = state
        .presentation
        .sample(&mut state.simulation, observation)?;
    if observation != state.observation() {
        return Err("viewport observation changed before frame publication".into());
    }
    // Tracking uncertainty restricts automatic camera movement, not inspection
    // of the current universe. Growth, splits and merges must not freeze a
    // successfully sampled nonempty view or its generation/population counters.
    if sample.grid.is_empty()
        && !state.presentation.focus.accepts_sample(
            state.presentation.viewport.mode(),
            generation,
            &sample,
        )
    {
        return Err("auto viewport reacquiring selected group; retaining previous view".into());
    }
    let (origin, grid) = (sample.origin, sample.grid);
    Ok(RenderSnapshot {
        worker_state_seq: state.state_seq,
        camera_revision: state.presentation.revision,
        viewport_revision: state.presentation.request.revision,
        source_revision: state.source_revision,
        generation,
        population,
        backend: "hashlife",
        source: state.source.clone(),
        running: state.running,
        quantum: state.quantum,
        origin,
        grid,
        status: format!(
            "{} | {}",
            state.status,
            state
                .presentation
                .focus
                .status(state.presentation.viewport.mode())
        ),
        output: state.output.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    use crate::generators::pattern_by_name;
    use crate::tui::protocol::PreparedSource;

    fn prepared_block() -> PreparedSource {
        let mut session = SimulationSession::new();
        session
            .try_load_hashlife_state(&pattern_by_name("block").unwrap_or_else(BitGrid::empty))
            .or_invariant("test block should load into HashLife");
        PreparedSource {
            session,
            label: "block".to_string(),
        }
    }

    #[test]
    fn ordered_relative_commands_are_cumulative() {
        let handle = start_worker(prepared_block(), 1, 1_000);
        let _ = handle.commands.send(ControlCommand::Pause);
        let _ = handle.commands.send(ControlCommand::AdvanceBy(10));
        let _ = handle.commands.send(ControlCommand::AdvanceBy(20));
        let _ = handle.commands.send(ControlCommand::Shutdown);
        let mut completed = 0;
        while let Ok(event) = handle.events.recv_timeout(Duration::from_secs(2)) {
            if matches!(event, WorkerEvent::CommandCompleted(_)) {
                completed += 1;
            }
            if event == WorkerEvent::Stopped {
                break;
            }
        }
        assert_eq!(completed, 2, "both accepted relative commands must execute");
        let frame = handle
            .frames
            .take()
            .or_invariant("worker should publish its latest committed frame");
        assert_eq!(
            frame.generation, 30,
            "relative advances must be cumulative at worker dequeue time"
        );
    }

    #[test]
    fn absolute_target_rejects_a_generation_already_passed() {
        let handle = start_worker(prepared_block(), 1, 1_000);
        let _ = handle.commands.send(ControlCommand::Pause);
        let _ = handle.commands.send(ControlCommand::AdvanceBy(10));
        let _ = handle.commands.send(ControlCommand::AdvanceTo(5));
        let _ = handle.commands.send(ControlCommand::Shutdown);

        let mut target_error = None;
        while let Ok(event) = handle.events.recv_timeout(Duration::from_secs(2)) {
            if let WorkerEvent::Error(message) = &event
                && message.contains("already passed")
            {
                target_error = Some(message.clone());
            }
            if event == WorkerEvent::Stopped {
                break;
            }
        }
        assert_eq!(
            target_error.as_deref(),
            Some("target 5 already passed at generation 10")
        );
    }

    #[test]
    fn save_barrier_writes_one_committed_generation() {
        let handle = start_worker(prepared_block(), 1, 1_000);
        let path = std::env::temp_dir().join(format!(
            "asimplelife-worker-save-{}-4.hls",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let _ = handle.commands.send(ControlCommand::Pause);
        let _ = handle.commands.send(ControlCommand::AdvanceBy(4));
        let _ = handle.commands.send(ControlCommand::Save(path.clone()));

        let saved_generation = loop {
            let event = handle
                .events
                .recv_timeout(Duration::from_secs(2))
                .or_invariant("save worker should report completion");
            if let WorkerEvent::Saved { generation, .. } = event {
                break generation;
            }
        };
        let mut restored = SimulationSession::new();
        let file = File::open(&path).or_invariant("saved snapshot should exist");
        restored
            .load_hashlife_snapshot_reader(file)
            .or_invariant("saved snapshot should load");
        let _ = fs::remove_file(path);
        assert_eq!(saved_generation, 4);
        assert_eq!(restored.hashlife_generation(), 4);
    }

    #[test]
    fn shutdown_reports_a_worker_panic() {
        let (commands, _command_rx) = mpsc::channel();
        let (_event_tx, events) = mpsc::channel();
        let handle = WorkerHandle {
            commands: Arc::new(CommandClient::new(
                commands,
                Arc::new(AtomicBool::new(false)),
            )),
            events,
            tuning: Arc::new(LatestValue::default()),
            viewport: Arc::new(LatestValue::default()),
            frames: Arc::new(LatestValue::default()),
            status: Arc::new(LatestValue::default()),
            stop_requested: Arc::new(AtomicBool::new(false)),
            thread: Some(thread::spawn(|| {
                std::panic::resume_unwind(Box::new("worker test panic"));
            })),
        };

        assert_eq!(
            handle.shutdown(),
            Err("simulation worker panicked: worker test panic".to_string()),
            "worker unwind must be visible to the UI"
        );
    }
}
