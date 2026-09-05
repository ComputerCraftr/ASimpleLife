#![cfg(unix)]

use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use nix::fcntl::{FcntlArg, FdFlag, OFlag, fcntl};
use nix::pty::{Winsize, openpty};
use nix::sys::signal::{Signal, kill};
use nix::sys::termios::{LocalFlags, Termios, tcgetattr};
use nix::unistd::Pid;

#[path = "support/pty_session.rs"]
mod pty_session;

type TestResult = Result<(), Box<dyn Error>>;
const TIMEOUT: Duration = Duration::from_secs(10);

struct TerminalProcess {
    child: Child,
    master: File,
    slave: File,
    original: Termios,
    output: Vec<u8>,
    cursor_queries_answered: usize,
    answer_cursor_queries: bool,
    screen: vt100::Parser,
    parent_terminal: Option<(File, Termios)>,
}

impl TerminalProcess {
    fn start(answer_cursor_queries: bool) -> Result<Self, Box<dyn Error>> {
        Self::start_with_size(answer_cursor_queries, 24, 120)
    }

    fn start_with_size(
        answer_cursor_queries: bool,
        rows: u16,
        columns: u16,
    ) -> Result<Self, Box<dyn Error>> {
        let parent_terminal = File::open("/dev/tty")
            .ok()
            .map(|file| {
                tcgetattr(&file)
                    .map(|attributes| (file, attributes))
                    .map_err(|error| format!("read parent tty attributes: {error}"))
            })
            .transpose()?;
        let size = Winsize {
            ws_row: rows,
            ws_col: columns,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let pty = openpty(Some(&size), None)?;
        let master = File::from(pty.master);
        let slave = File::from(pty.slave);
        for descriptor in [&master, &slave] {
            fcntl(descriptor, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))?;
        }
        let original =
            tcgetattr(&master).map_err(|error| format!("read initial PTY attributes: {error}"))?;
        let flags = OFlag::from_bits_truncate(fcntl(&master, FcntlArg::F_GETFL)?);
        fcntl(&master, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
        let mut command = Command::new(env!("CARGO_BIN_EXE_a_simple_life"));
        command
            .args(["--tui", "--pattern", "block", "--delay-ms", "16"])
            .env("TERM", "xterm-256color")
            .stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?))
            .stderr(Stdio::from(slave.try_clone()?));
        let child = pty_session::spawn(command)
            .map_err(|error| format!("spawn controlling PTY child: {error}"))?;
        let mut process = Self {
            child,
            master,
            slave,
            original,
            output: Vec::new(),
            cursor_queries_answered: 0,
            answer_cursor_queries,
            screen: vt100::Parser::new(size.ws_row, size.ws_col, 0),
            parent_terminal,
        };
        process.wait_for_text(if answer_cursor_queries {
            b"Universe"
        } else {
            b"\x1b[6n"
        })?;
        let attributes = tcgetattr(&process.master)
            .map_err(|error| format!("read running PTY attributes: {error}"))?;
        assert!(
            !attributes
                .local_flags
                .intersects(LocalFlags::ICANON | LocalFlags::ECHO | LocalFlags::ISIG),
            "raw mode was not applied to the test PTY: {attributes:?}"
        );
        process.assert_parent_terminal_unchanged()?;
        Ok(process)
    }

    fn assert_parent_terminal_unchanged(&self) -> TestResult {
        if let Some((terminal, original)) = &self.parent_terminal {
            assert_eq!(
                &tcgetattr(terminal)
                    .map_err(|error| format!("recheck parent terminal: {error}"))?,
                original,
                "test child modified the caller's terminal"
            );
        }
        Ok(())
    }

    fn resize(&mut self, rows: u16, columns: u16, expected: &str) -> TestResult {
        self.drain()?;
        let start = self.output.len();
        // A real terminal retains parser/cursor state across a resize; creating
        // a fresh parser can lose a split escape sequence or a pending redraw.
        self.screen.screen_mut().set_size(rows, columns);
        let status = Command::new("stty")
            .args(["rows", &rows.to_string(), "cols", &columns.to_string()])
            .stdin(Stdio::from(self.slave.try_clone()?))
            .status()?;
        assert!(
            status.success(),
            "PTY resize failed for {columns}x{rows}: {status}"
        );
        self.signal(Signal::SIGWINCH)?;
        self.wait_for_text_since(start, expected.as_bytes())?;
        if rows >= 12 && columns >= 24 {
            self.wait_until("a complete resized border", |process| {
                let screen = process.screen.screen();
                screen
                    .cell(0, columns - 1)
                    .is_some_and(|cell| cell.contents() == "┐")
                    && screen
                        .cell(rows - 1, columns - 1)
                        .is_some_and(|cell| cell.contents() == "┘")
            })?;
        }
        Ok(())
    }

    fn drain(&mut self) -> io::Result<()> {
        let mut buffer = [0; 8192];
        loop {
            match self.master.read(&mut buffer) {
                Ok(0) => return Ok(()),
                Ok(count) => {
                    self.screen.process(&buffer[..count]);
                    self.output.extend_from_slice(&buffer[..count]);
                    // A PTY transports bytes but does not emulate the terminal's
                    // response to Crossterm's cursor-position query.
                    let queries = self
                        .output
                        .windows(4)
                        .filter(|text| *text == b"\x1b[6n")
                        .count();
                    while self.answer_cursor_queries && self.cursor_queries_answered < queries {
                        self.master.write_all(b"\x1b[1;1R")?;
                        self.cursor_queries_answered += 1;
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn wait_for_text(&mut self, expected: &[u8]) -> TestResult {
        self.wait_for_text_since(0, expected)
    }

    fn wait_for_text_since(&mut self, start: usize, expected: &[u8]) -> TestResult {
        self.wait_until(
            &format!("rendered {:?}", String::from_utf8_lossy(expected)),
            |process| {
                if expected.starts_with(b"\x1b") {
                    process.output[start..]
                        .windows(expected.len())
                        .any(|text| text == expected)
                } else {
                    process.output.len() > start
                        && process
                            .screen
                            .screen()
                            .contents()
                            .contains(String::from_utf8_lossy(expected).trim_end())
                }
            },
        )
    }

    fn wait_until(&mut self, expected: &str, ready: impl Fn(&Self) -> bool) -> TestResult {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            self.drain()?;
            if ready(self) {
                return Ok(());
            }
            let status = self.child.try_wait()?;
            if status.is_some() || Instant::now() >= deadline {
                return Err(format!(
                    "TUI never {expected}; child={status:?}; size={:?}; screen={:?}",
                    self.screen.screen().size(),
                    self.screen.screen().contents()
                )
                .into());
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn signal(&self, signal: Signal) -> TestResult {
        kill(Pid::from_raw(i32::try_from(self.child.id())?), signal)?;
        Ok(())
    }

    fn wait_for_exit(&mut self) -> Result<ExitStatus, Box<dyn Error>> {
        let deadline = Instant::now() + TIMEOUT;
        let status: ExitStatus = loop {
            self.drain()?;
            if let Some(status) = self.child.try_wait()? {
                break status;
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "TUI did not shut down; output={:?}",
                    String::from_utf8_lossy(&self.output)
                )
                .into());
            }
            thread::sleep(Duration::from_millis(5));
        };
        self.drain()?;
        Ok(status)
    }

    fn finish(&mut self, expected_code: i32) -> TestResult {
        let status = self.wait_for_exit()?;
        assert_eq!(
            status.code(),
            Some(expected_code),
            "status={status}; output={:?}",
            String::from_utf8_lossy(&self.output)
        );
        assert_eq!(
            // macOS revokes slave descriptors when their session leader exits.
            // The retained master still exposes the same PTY's termios state.
            tcgetattr(&self.master)
                .map_err(|error| format!("read restored PTY attributes: {error}"))?,
            self.original,
            "exit {expected_code} left terminal attributes changed"
        );
        self.assert_parent_terminal_unchanged()?;
        for sequence in [b"\x1b[?1049l".as_slice(), b"\x1b[?25h".as_slice()] {
            assert!(
                self.output
                    .windows(sequence.len())
                    .any(|text| text == sequence),
                "exit {expected_code} omitted terminal restoration {:?}; output={:?}",
                String::from_utf8_lossy(sequence),
                String::from_utf8_lossy(&self.output)
            );
        }
        Ok(())
    }
}

#[test]
fn repeated_pty_resizing_redraws_layout_and_remains_interactive() -> TestResult {
    let mut process = TerminalProcess::start(true)?;
    for (rows, columns, expected) in [
        (8, 20, "Terminal too small"),
        (24, 80, "Universe"),
        (18, 40, "all controls"),
        (24, 120, "Universe"),
    ] {
        process.resize(rows, columns, expected)?;
    }
    process.master.write_all(b"q")?;
    process.finish(0)
}

#[test]
fn active_group_keys_and_help_render_in_a_real_terminal() -> TestResult {
    let mut process = TerminalProcess::start(true)?;
    process.master.write_all(b"?")?;
    process.wait_for_text(b"Tab / Shift+Tab: next / previous active group")?;
    process.master.write_all(b"\x1b")?;
    process.wait_for_text(b"Universe")?;
    process.master.write_all(b" \t\x1b[Z\x1b[6~")?;
    process.wait_for_text(b"no next active group found")?;
    process.wait_for_text(b"paused")?;
    process.master.write_all(b"q")?;
    process.finish(0)
}

#[test]
fn classification_scope_dialog_and_result_leave_paused_simulation_unchanged() -> TestResult {
    let mut process = TerminalProcess::start_with_size(true, 36, 180)?;
    process.master.write_all(b" ")?;
    process.wait_for_text(b"paused")?;
    let before = process.screen.screen().contents();
    let generation = before
        .split("generation=")
        .nth(1)
        .ok_or("generation missing")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    process.master.write_all(b"c")?;
    process.wait_for_text(b"enter whole or region")?;
    process.master.write_all(b"whole\r")?;
    process.wait_for_text(b"still life at capture")?;
    process.wait_for_text(b"captured generation=")?;
    let after = process.screen.screen().contents();
    assert!(
        after.contains(&format!(
            "generation={generation} | view generation={generation}"
        )),
        "classification changed committed/displayed progress: {after}"
    );
    assert!(
        after.contains("paused"),
        "classification changed run state: {after}"
    );
    process.master.write_all(b"q")?;
    process.finish(0)
}

impl Drop for TerminalProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        // Darwin may wait for terminal output to drain during session teardown.
        // Never block failure cleanup in wait(): keep consuming output while
        // reaping, and release the PTY descriptors if teardown exceeds the bound.
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let _ = self.drain();
            if !matches!(self.child.try_wait(), Ok(None)) || Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn unix_signals_preserve_exit_status_and_restore_terminal() -> TestResult {
    for (signal, code) in [
        (Signal::SIGHUP, 129),
        (Signal::SIGINT, 130),
        (Signal::SIGTERM, 143),
    ] {
        let mut process = TerminalProcess::start(true)?;
        process.signal(signal)?;
        process.finish(code)?;
    }
    Ok(())
}

#[test]
fn raw_control_c_exits_from_a_dialog_and_restores_terminal() -> TestResult {
    let mut process = TerminalProcess::start(true)?;
    process.master.write_all(b"f")?;
    process.wait_for_text(b"Forward> ")?;
    process.master.write_all(b"\x03")?;
    process.finish(130)
}

#[test]
fn escape_cancels_a_dialog_before_it_quits_the_tui() -> TestResult {
    let mut process = TerminalProcess::start(true)?;
    process.master.write_all(b"f")?;
    process.wait_for_text(b"Forward> ")?;
    process.master.write_all(b"\x1b")?;
    thread::sleep(Duration::from_millis(50));
    process.drain()?;
    assert!(
        process.child.try_wait()?.is_none(),
        "Escape in a dialog unexpectedly exited the TUI"
    );
    process.master.write_all(b"q")?;
    process.finish(0)
}

#[test]
fn failed_source_preparation_keeps_the_tui_usable() -> TestResult {
    let mut process = TerminalProcess::start_with_size(true, 24, 80)?;
    process.master.write_all(b"n")?;
    process.wait_for_text(b"Seed> ")?;
    process.master.write_all(b"not-a-life-seed\r")?;
    process.wait_for_text(b"\"not-a-life-seed\"")?;
    process.wait_for_text(b"source=block")?;
    process.wait_for_text(b"population=4")?;
    process.master.write_all(b" ")?;
    process.wait_for_text(b"paused")?;
    process.master.write_all(b"q")?;
    process.finish(0)
}

#[test]
fn normal_quit_keys_exit_successfully_and_restore_terminal() -> TestResult {
    for key in [b"q".as_slice(), b"\x1b".as_slice()] {
        let mut process = TerminalProcess::start(true)?;
        process.master.write_all(key)?;
        process.finish(0)?;
    }
    Ok(())
}

#[test]
fn restoration_check_detects_raw_mode_left_by_an_abrupt_exit() -> TestResult {
    let mut process = TerminalProcess::start(true)?;
    // SIGKILL cannot run the application's RAII guard. Prove our retained-master
    // check sees that failure rather than a kernel reset masquerading as cleanup.
    process.child.kill()?;
    let status = process.wait_for_exit()?;
    assert!(
        !status.success(),
        "forced exit unexpectedly succeeded: {status}"
    );
    assert_ne!(
        tcgetattr(&process.master)?,
        process.original,
        "PTY restoration check cannot distinguish SIGKILL from orderly cleanup"
    );
    process.assert_parent_terminal_unchanged()
}

#[test]
fn termination_status_survives_terminal_initialization_errors() -> TestResult {
    for (signal, code) in [(Signal::SIGHUP, 129), (Signal::SIGTERM, 143)] {
        let mut process = TerminalProcess::start(false)?;
        process.signal(signal)?;
        process.finish(code)?;
    }
    Ok(())
}

#[test]
fn terminal_initialization_failure_without_a_signal_remains_an_error() -> TestResult {
    let mut process = TerminalProcess::start(false)?;
    process.finish(1)
}
