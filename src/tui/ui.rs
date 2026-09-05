use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::bitgrid::{Cell, Coord};
use crate::cli::{Config, RunMode};

use super::protocol::{
    AnalysisScope, AnalysisUpdate, ControlCommand, MAX_CONTINUOUS_QUANTUM, RenderSnapshot,
    ViewportRequest, WorkerEvent, WorkerStatus, WorkerTuning,
};
use super::source::{
    prepare_bf_source, prepare_config_source, prepare_file_source, prepare_named_source,
};
use super::termination::TuiExit;
use super::worker::{WorkerHandle, start_worker};

mod layout;
#[cfg(test)]
mod state_tests;
mod viewport;
use layout::{draw_ui, universe_content};
use viewport::request_viewport;

const FRAME_INTERVALS_MS: [u64; 7] = [1_000, 500, 250, 100, 50, 33, 16];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptKind {
    Forward,
    Target,
    Save,
    Open,
    Seed,
    Brainfuck,
    Classification {
        source_revision: u64,
        rectangle: Option<crate::bitgrid::Bounds>,
    },
}

#[derive(Debug)]
struct Prompt {
    kind: PromptKind,
    value: String,
}

#[derive(Debug)]
struct SourcePreparation {
    receiver: mpsc::Receiver<Result<super::protocol::PreparedSource, String>>,
}

#[derive(Debug)]
struct UiState {
    frame: Option<RenderSnapshot>,
    authoritative: Option<WorkerStatus>,
    analysis: Option<AnalysisUpdate>,
    analysis_configuration: u64,
    pending_commands: Vec<u64>,
    auto_viewport: bool,
    manual_origin: Option<Cell>,
    recenter: bool,
    quantum: u64,
    speed_index: usize,
    prompt: Option<Prompt>,
    source_preparation: Option<SourcePreparation>,
    source_revision: u64,
    notice: String,
    viewport_request: Option<ViewportRequest>,
    status_scroll: u16,
    help: bool,
}

impl UiState {
    fn new(config: &Config) -> Self {
        let speed_index = FRAME_INTERVALS_MS
            .iter()
            .position(|value| *value <= config.delay_ms.max(1))
            .unwrap_or(FRAME_INTERVALS_MS.len() - 1);
        Self {
            frame: None,
            authoritative: None,
            analysis: None,
            analysis_configuration: 1,
            pending_commands: Vec::new(),
            auto_viewport: true,
            manual_origin: None,
            recenter: true,
            quantum: config.step_generations.clamp(1, MAX_CONTINUOUS_QUANTUM),
            speed_index,
            prompt: None,
            source_preparation: None,
            source_revision: 1,
            notice: "starting".to_string(),
            viewport_request: None,
            status_scroll: 0,
            help: false,
        }
    }

    fn interval(&self) -> Duration {
        Duration::from_millis(FRAME_INTERVALS_MS[self.speed_index])
    }
}

struct TerminalRestore;

impl TerminalRestore {
    fn enter() -> Result<Self, String> {
        enable_raw_mode()
            .map_err(|error| format!("failed to enable terminal raw mode: {error}"))?;
        let restore = Self;
        execute!(io::stdout(), EnterAlternateScreen, crossterm::cursor::Hide)
            .map_err(|error| format!("failed to enter alternate terminal screen: {error}"))?;
        Ok(restore)
    }
}

impl Drop for TerminalRestore {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}

pub fn should_run_tui(config: &Config) -> bool {
    if config.classify_only {
        return false;
    }
    match config.run_mode {
        RunMode::Tui => true,
        RunMode::Headless => false,
        RunMode::Auto => {
            !config.steps_explicit && io::stdin().is_terminal() && io::stdout().is_terminal()
        }
    }
}

pub fn run(config: Config) -> Result<TuiExit, String> {
    super::termination::prepare()?;
    // Read the cause after worker shutdown and terminal restoration, including
    // when a hangup also caused terminal I/O to fail.
    let result = run_session(config);
    match super::termination::exit() {
        TuiExit::Completed => result.map(|()| TuiExit::Completed),
        exit => Ok(exit),
    }
}

fn run_session(config: Config) -> Result<(), String> {
    let initial = prepare_config_source(&config)?;
    if super::termination::requested() {
        return Ok(());
    }
    let _restore = TerminalRestore::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)
        .map_err(|error| format!("failed to initialize terminal renderer: {error}"))?;
    terminal
        .clear()
        .map_err(|error| format!("failed to clear terminal: {error}"))?;

    let worker = start_worker(initial, config.step_generations, config.delay_ms);
    super::termination::register_worker(worker.termination_token());
    let mut state = UiState::new(&config);
    if let Some(max_generations) = config.max_generations {
        send_control(
            &worker,
            ControlCommand::ConfigureAnalysis { max_generations },
            &mut state,
        );
    }
    let mut next_draw = Instant::now();
    let mut exiting = false;
    let mut session_error = None;

    while !exiting && !super::termination::requested() {
        match drain_worker_events(worker.events(), &mut state) {
            Ok(stopped) => exiting = stopped,
            Err(error) => {
                session_error = Some(error);
                break;
            }
        }
        poll_source_preparation(&worker, &mut state);
        if let Some(update) = worker.next_status() {
            accept_status(&mut state, update);
        }
        let frame_updated = if let Some(frame) = worker.next_frame() {
            accept_frame(&mut state, frame);
            true
        } else {
            false
        };

        let size = terminal
            .size()
            .map_err(|error| format!("failed to read terminal size: {error}"))?;
        let viewport_changed = request_viewport(
            &worker,
            &mut state,
            universe_content(Rect::new(0, 0, size.width, size.height)),
        );

        if next_draw.elapsed() >= state.interval()
            || state.prompt.is_some()
            || viewport_changed
            || frame_updated
        {
            terminal
                .draw(|frame| draw_ui(frame, &mut state))
                .map_err(|error| format!("terminal draw failed: {error}"))?;
            next_draw = Instant::now();
        }

        let timeout = state
            .interval()
            .saturating_sub(next_draw.elapsed())
            .min(Duration::from_millis(25));
        if event::poll(timeout).map_err(|error| format!("terminal input poll failed: {error}"))? {
            match event::read().map_err(|error| format!("terminal input failed: {error}"))? {
                Event::Key(key) if key.kind != crossterm::event::KeyEventKind::Release => {
                    exiting = handle_key(key, &worker, &mut state, &config);
                    next_draw = Instant::now() - state.interval();
                }
                Event::Resize(_, _) => next_draw = Instant::now() - state.interval(),
                _ => {}
            }
        }
    }

    let shutdown = worker.shutdown();
    shutdown.and_then(|()| session_error.map_or(Ok(()), Err))
}

fn drain_worker_events(
    events: &mpsc::Receiver<WorkerEvent>,
    state: &mut UiState,
) -> Result<bool, String> {
    let mut stopped = false;
    loop {
        match events.try_recv() {
            Ok(WorkerEvent::Acknowledged { request_id }) => {
                state.pending_commands.retain(|id| *id != request_id);
            }
            Ok(WorkerEvent::Analysis(update)) => {
                let valid_source = update.descriptor.is_none_or(|descriptor| {
                    descriptor.source_revision == state.source_revision
                        && descriptor.configuration == state.analysis_configuration
                });
                let valid_request = state.analysis.as_ref().is_none_or(|current| {
                    update.request_id > current.request_id
                        || (update.request_id == current.request_id && current.status.is_active())
                });
                if valid_source && valid_request {
                    state.analysis = Some(update);
                }
            }
            Ok(WorkerEvent::CommandCompleted(message) | WorkerEvent::Error(message)) => {
                state.notice = message;
                state.status_scroll = 0;
            }
            Ok(WorkerEvent::Saved { path, generation }) => {
                state.notice = format!("saved generation {generation} to {}", path.display());
            }
            Ok(WorkerEvent::AnalysisConfiguration(revision)) => {
                if revision > state.analysis_configuration {
                    state.analysis_configuration = revision;
                    state.analysis = None;
                }
            }
            Ok(WorkerEvent::SourceReplaced { label, revision }) => {
                state.notice = format!("loaded {label} (revision {revision})");
                state.source_revision = revision;
                state.analysis = None;
                state.manual_origin = None;
                state.recenter = true;
            }
            Ok(WorkerEvent::Stopped) => stopped = true,
            Err(mpsc::TryRecvError::Empty) => return Ok(stopped),
            Err(mpsc::TryRecvError::Disconnected) if stopped => return Ok(true),
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err("simulation worker disconnected unexpectedly".to_string());
            }
        }
    }
}

fn accept_frame(state: &mut UiState, frame: RenderSnapshot) {
    if frame.source_revision < state.source_revision {
        return;
    }
    if state
        .viewport_request
        .is_some_and(|request| frame.viewport_revision != request.revision)
    {
        return;
    }
    if state.frame.as_ref().is_some_and(|current| {
        frame.worker_state_seq < current.worker_state_seq
            || frame.camera_revision < current.camera_revision
    }) {
        return;
    }
    state.source_revision = frame.source_revision;
    state.recenter = false;
    if state.manual_origin.is_none() || state.auto_viewport {
        state.manual_origin = Some(frame.origin);
        if !state.auto_viewport
            && let Some(request) = &mut state.viewport_request
        {
            request.origin = Some(frame.origin);
        }
    }
    state.frame = Some(frame);
}

fn accept_status(state: &mut UiState, update: WorkerStatus) {
    if state
        .authoritative
        .as_ref()
        .is_none_or(|current| update.worker_state_seq > current.worker_state_seq)
    {
        state.authoritative = Some(update);
    }
}

fn handle_key(key: KeyEvent, worker: &WorkerHandle, state: &mut UiState, config: &Config) -> bool {
    if is_termination_key(key) {
        super::termination::request();
        return true;
    }
    if let Some(prompt) = state.prompt.as_mut() {
        match key.code {
            KeyCode::Esc => state.prompt = None,
            KeyCode::Enter => submit_prompt(worker, state, config),
            KeyCode::Backspace => {
                prompt.value.pop();
            }
            KeyCode::Char(ch) => prompt.value.push(ch),
            _ => {}
        }
        return false;
    }
    if state.help && matches!(key.code, KeyCode::Esc | KeyCode::Char('?')) {
        state.help = false;
        state.status_scroll = 0;
        return false;
    }

    if state.source_preparation.is_some() && key.code == KeyCode::Esc {
        state.source_preparation = None;
        state.notice = "source preparation cancelled; current universe unchanged".to_string();
        return false;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s') {
        state.prompt = Some(Prompt {
            kind: PromptKind::Save,
            value: "state.hls".to_string(),
        });
        return false;
    }
    if state.source_preparation.is_some()
        && matches!(
            key.code,
            KeyCode::Char('.')
                | KeyCode::Char('f')
                | KeyCode::Char('g')
                | KeyCode::Tab
                | KeyCode::BackTab
        )
    {
        state.notice = "exact controls are disabled while a source is being prepared".to_string();
        return false;
    }
    match key.code {
        KeyCode::Char('?') => {
            state.help = true;
            state.status_scroll = 0;
            false
        }
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Char(' ') => {
            send_control(worker, ControlCommand::ToggleRunning, state);
            false
        }
        KeyCode::Char('.') => {
            send_control(worker, ControlCommand::StepOne, state);
            false
        }
        KeyCode::Char('+') | KeyCode::Char('=') => {
            state.speed_index = (state.speed_index + 1).min(FRAME_INTERVALS_MS.len() - 1);
            send_tuning(worker, state);
            false
        }
        KeyCode::Char('-') => {
            state.speed_index = state.speed_index.saturating_sub(1);
            send_tuning(worker, state);
            false
        }
        KeyCode::Char(']') => {
            state.quantum = state
                .quantum
                .checked_mul(2)
                .unwrap_or(MAX_CONTINUOUS_QUANTUM)
                .min(MAX_CONTINUOUS_QUANTUM);
            send_tuning(worker, state);
            false
        }
        KeyCode::Char('[') => {
            state.quantum = (state.quantum / 2).max(1);
            send_tuning(worker, state);
            false
        }
        KeyCode::Char('f') => open_prompt(state, PromptKind::Forward),
        KeyCode::Char('g') => open_prompt(state, PromptKind::Target),
        KeyCode::Char('o') => open_prompt(state, PromptKind::Open),
        KeyCode::Char('n') => open_prompt(state, PromptKind::Seed),
        KeyCode::Char('b') => open_prompt(state, PromptKind::Brainfuck),
        KeyCode::Char('c') => {
            if state
                .analysis
                .as_ref()
                .is_some_and(|update| update.status.is_active())
            {
                send_control(worker, ControlCommand::CancelClassification, state);
            } else {
                let rectangle = state.viewport_request.and_then(|request| {
                    let origin = request
                        .origin
                        .or_else(|| state.frame.as_ref().map(|frame| frame.origin))?;
                    Some((
                        origin.0,
                        origin.1,
                        origin
                            .0
                            .checked_add(i64::from(request.width).checked_sub(1)?)?,
                        origin.1.checked_add(
                            i64::from(request.height).checked_mul(2)?.checked_sub(1)?,
                        )?,
                    ))
                });
                open_prompt(
                    state,
                    PromptKind::Classification {
                        source_revision: state.source_revision,
                        rectangle,
                    },
                );
                state.notice =
                    "Classify: enter whole or region; region uses frozen rectangle at capture time"
                        .into();
            }
            false
        }
        KeyCode::Char('a') => {
            state.auto_viewport = !state.auto_viewport;
            if state.auto_viewport {
                // Releasing an explicit selection is an ordered action even
                // if rapid off/on presentation requests coalesce.
                send_control(worker, ControlCommand::ResetAutoFocus, state);
            }
            state.notice = format!(
                "automatic viewport {}",
                if state.auto_viewport {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            false
        }
        KeyCode::Home => {
            state.recenter = true;
            state.manual_origin = None;
            false
        }
        KeyCode::Tab | KeyCode::BackTab => {
            if let Some(mut request) = state.viewport_request {
                request.revision = request.revision.saturating_add(1);
                request.origin = None;
                request.recenter = false;
                request.auto = state.auto_viewport;
                state.viewport_request = Some(request);
                state.manual_origin = None;
                state.recenter = false;
                let previous =
                    key.code == KeyCode::BackTab || key.modifiers.contains(KeyModifiers::SHIFT);
                send_control(
                    worker,
                    if previous {
                        ControlCommand::FocusPrevious(request)
                    } else {
                        ControlCommand::FocusNext(request)
                    },
                    state,
                );
            }
            false
        }
        KeyCode::PageUp => {
            state.status_scroll = state.status_scroll.saturating_sub(3);
            false
        }
        KeyCode::PageDown => {
            state.status_scroll = state.status_scroll.saturating_add(3);
            false
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down => {
            pan_viewport(key, state);
            false
        }
        _ => false,
    }
}

fn is_termination_key(key: KeyEvent) -> bool {
    key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c')
}

fn send_tuning(worker: &WorkerHandle, state: &UiState) {
    worker.set_tuning(WorkerTuning {
        quantum: state.quantum,
        interval_ms: FRAME_INTERVALS_MS[state.speed_index],
    });
}

fn open_prompt(state: &mut UiState, kind: PromptKind) -> bool {
    if state.source_preparation.is_some()
        && matches!(
            kind,
            PromptKind::Open | PromptKind::Seed | PromptKind::Brainfuck
        )
    {
        state.notice = "a source is already being prepared".to_string();
        return false;
    }
    state.prompt = Some(Prompt {
        kind,
        value: String::new(),
    });
    false
}

fn pan_viewport(key: KeyEvent, state: &mut UiState) {
    let scale = if key.modifiers.contains(KeyModifiers::SHIFT) {
        10
    } else {
        1
    };
    let (dx, dy) = match key.code {
        KeyCode::Left => (-scale, 0),
        KeyCode::Right => (scale, 0),
        KeyCode::Up => (0, -2 * scale),
        KeyCode::Down => (0, 2 * scale),
        _ => (0, 0),
    };
    let origin = state
        .manual_origin
        .or_else(|| state.frame.as_ref().map(|frame| frame.origin))
        .unwrap_or((0, 0));
    state.manual_origin = Some((origin.0.saturating_add(dx), origin.1.saturating_add(dy)));
    state.auto_viewport = false;
}

fn submit_prompt(worker: &WorkerHandle, state: &mut UiState, config: &Config) {
    let Some(prompt) = state.prompt.take() else {
        return;
    };
    let value = prompt.value.trim().to_string();
    match prompt.kind {
        PromptKind::Classification {
            source_revision,
            rectangle,
        } => {
            let scope = match value.as_str() {
                "" | "whole" | "w" => Some(AnalysisScope::WholeUniverse),
                "region" | "r" => rectangle.map(AnalysisScope::IsolatedRegion),
                _ => None,
            };
            if let Some(scope) = scope {
                send_control(
                    worker,
                    ControlCommand::Classify {
                        source_revision,
                        scope,
                    },
                    state,
                );
            } else {
                state.notice =
                    "choose whole or region (region requires a valid camera rectangle)".into();
            }
        }
        PromptKind::Forward => match value.parse::<u64>() {
            Ok(delta) => send_control(worker, ControlCommand::AdvanceBy(delta), state),
            Err(_) => state.notice = format!("invalid forward delta {value:?}"),
        },
        PromptKind::Target => match value.parse::<u64>() {
            Ok(target) => send_control(worker, ControlCommand::AdvanceTo(target), state),
            Err(_) => state.notice = format!("invalid target generation {value:?}"),
        },
        PromptKind::Save => {
            send_control(worker, ControlCommand::Save(PathBuf::from(value)), state);
        }
        PromptKind::Open | PromptKind::Seed | PromptKind::Brainfuck => {
            state.notice = "preparing source; current universe continues".to_string();
            let (sender, receiver) = mpsc::channel();
            state.source_preparation = Some(SourcePreparation { receiver });
            let config = config.clone();
            if let Err(error) = thread::Builder::new()
                .name("source-preparation".to_string())
                .spawn(move || {
                    let result = match prompt.kind {
                        PromptKind::Open => prepare_file_source(&value),
                        PromptKind::Seed => prepare_named_source(&value, &config),
                        PromptKind::Brainfuck => prepare_bf_source(&value),
                        _ => Err("invalid source prompt".to_string()),
                    };
                    let _ = sender.send(result);
                })
            {
                state.source_preparation = None;
                state.notice = format!("failed to start source preparation: {error}");
            }
        }
    }
}

fn poll_source_preparation(worker: &WorkerHandle, state: &mut UiState) {
    let result = match state.source_preparation.as_ref() {
        Some(preparation) => match preparation.receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("source preparation worker stopped unexpectedly".to_string())
            }
        },
        None => return,
    };
    state.source_preparation = None;
    match result {
        Ok(prepared) => send_control(
            worker,
            ControlCommand::ReplaceSource(Box::new(prepared)),
            state,
        ),
        Err(error) => {
            state.notice = error;
            state.status_scroll = 0;
        }
    }
}

fn send_control(worker: &WorkerHandle, command: ControlCommand, state: &mut UiState) {
    match worker.submit(command) {
        Ok(id) => state.pending_commands.push(id),
        Err(error) => state.notice = error.to_string(),
    }
}

fn draw_universe(frame: &mut ratatui::Frame<'_>, area: Rect, snapshot: Option<&RenderSnapshot>) {
    let inner = Block::default().borders(Borders::ALL).title("Universe");
    let content = inner.inner(area);
    frame.render_widget(inner, area);
    let Some(snapshot) = snapshot else {
        return;
    };
    let mut lines = Vec::with_capacity(usize::from(content.height));
    for row in 0..content.height {
        let mut text = String::with_capacity(usize::from(content.width));
        for column in 0..content.width {
            let x = snapshot.origin.0.saturating_add(Coord::from(column));
            let y = snapshot
                .origin
                .1
                .saturating_add(Coord::from(row).saturating_mul(2));
            let lower_y = y.saturating_add(1);
            text.push(
                match (snapshot.grid.get(x, y), snapshot.grid.get(x, lower_y)) {
                    (false, false) => ' ',
                    (true, false) => '▀',
                    (false, true) => '▄',
                    (true, true) => '█',
                },
            );
        }
        lines.push(Line::raw(text));
    }
    frame.render_widget(Paragraph::new(lines), content);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    use ratatui::backend::TestBackend;

    pub(super) fn snapshot(revision: u64, origin: Cell) -> RenderSnapshot {
        RenderSnapshot {
            worker_state_seq: 1,
            camera_revision: 0,
            viewport_revision: 0,
            source_revision: revision,
            generation: 0,
            population: 0,
            backend: "hashlife",
            source: "test".to_string(),
            running: false,
            quantum: 1,
            origin,
            grid: crate::bitgrid::BitGrid::empty(),
            status: String::new(),
            output: String::new(),
        }
    }

    #[test]
    fn explicit_steps_select_headless_auto_mode() {
        let config = Config {
            steps_explicit: true,
            ..Config::default()
        };
        assert!(!should_run_tui(&config));
    }

    #[test]
    fn forced_modes_override_terminal_detection() {
        let tui = Config {
            run_mode: RunMode::Tui,
            ..Config::default()
        };
        let headless = Config {
            run_mode: RunMode::Headless,
            ..Config::default()
        };
        assert!(should_run_tui(&tui));
        assert!(!should_run_tui(&headless));
    }

    #[test]
    fn control_c_is_recognized_before_prompt_input() {
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(
            is_termination_key(key),
            "raw-mode Ctrl-C must terminate instead of becoming prompt text"
        );
        assert!(!is_termination_key(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::NONE
        )));
    }

    #[test]
    fn worker_disconnect_without_stopped_event_is_an_error() {
        let (sender, receiver) = mpsc::channel();
        drop(sender);
        let mut state = UiState::new(&Config::default());
        assert_eq!(
            drain_worker_events(&receiver, &mut state),
            Err("simulation worker disconnected unexpectedly".to_string())
        );
    }

    #[test]
    fn stopped_event_allows_worker_channel_to_disconnect() {
        let (sender, receiver) = mpsc::channel();
        assert!(sender.send(WorkerEvent::Stopped).is_ok());
        drop(sender);
        let mut state = UiState::new(&Config::default());
        assert_eq!(drain_worker_events(&receiver, &mut state), Ok(true));
    }

    #[test]
    fn cancelled_source_preparation_cannot_publish_a_late_result() {
        let (sender, receiver) = mpsc::channel();
        let mut state = UiState::new(&Config::default());
        state.source_preparation = Some(SourcePreparation { receiver });
        state.notice = "preparing".to_string();

        state.source_preparation = None;
        assert!(sender.send(Err("late failure".to_string())).is_err());
        assert_eq!(state.notice, "preparing");
    }

    #[test]
    fn source_failure_reveals_the_notice_without_replacing_or_pausing_the_universe() {
        let initial =
            prepare_named_source("block", &Config::default()).or_invariant("block source");
        let worker = start_worker(initial, 1, 1000);
        let (sender, receiver) = mpsc::channel();
        let mut state = UiState::new(&Config::default());
        state.source_preparation = Some(SourcePreparation { receiver });
        state.status_scroll = 50;
        state.frame = Some(snapshot(1, (7, 8)));
        sender
            .send(Err("unknown Life seed \"bad\"".to_string()))
            .or_invariant("source failure");
        poll_source_preparation(&worker, &mut state);
        assert_eq!(
            state.status_scroll, 0,
            "new errors must not stay below the scrolled viewport"
        );
        assert!(
            state.authoritative.is_none(),
            "termination must not fabricate a worker update"
        );
        assert_eq!(state.source_revision, 1);
        assert_eq!(
            state.frame.as_ref().or_invariant("retained frame").origin,
            (7, 8)
        );
        assert!(state.source_preparation.is_none());
        assert_eq!(state.notice, "unknown Life seed \"bad\"");
        worker.shutdown().or_invariant("worker shutdown");
    }

    #[test]
    fn stale_frame_cannot_restore_the_previous_source_viewport() {
        let mut state = UiState::new(&Config::default());
        state.source_revision = 2;
        state.manual_origin = None;
        accept_frame(&mut state, snapshot(1, (123, 456)));
        assert!(state.frame.is_none());
        assert!(state.manual_origin.is_none());

        accept_frame(&mut state, snapshot(2, (7, 8)));
        assert_eq!(state.manual_origin, Some((7, 8)));
    }

    #[test]
    fn universe_render_saturates_at_coordinate_bounds() {
        let backend = TestBackend::new(3, 3);
        let mut terminal = Terminal::new(backend).expect("test terminal should initialize");
        let snapshot = snapshot(1, (Coord::MAX, Coord::MAX));
        terminal
            .draw(|frame| draw_universe(frame, frame.area(), Some(&snapshot)))
            .expect("coordinate bounds should not panic or fail rendering");
    }
}
