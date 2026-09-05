//! Bounded pipe capture and owned process groups. No child survives its owner.
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
mod tree {
    use super::*;
    use std::os::unix::process::CommandExt;
    pub struct Tree {
        pub child: Child,
        terminated: bool,
    }
    impl Tree {
        pub fn spawn(command: &mut Command) -> io::Result<Self> {
            command.process_group(0);
            Ok(Self {
                child: command.spawn()?,
                terminated: false,
            })
        }
        pub fn terminate(&mut self) {
            if self.terminated {
                return;
            }
            self.terminated = true;
            if let Ok(pid) = i32::try_from(self.child.id()) {
                let _ = nix::sys::signal::killpg(
                    nix::unistd::Pid::from_raw(pid),
                    nix::sys::signal::Signal::SIGKILL,
                );
            }
            let _ = self.child.wait();
        }
    }
    impl Drop for Tree {
        fn drop(&mut self) {
            self.terminate();
        }
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
#[path = "process_windows.rs"]
mod tree;

#[derive(Debug)]
pub enum ProcessFailure {
    Io(io::Error),
    Timeout {
        phase: &'static str,
        diagnostics: String,
    },
    OutputLimitExceeded {
        phase: &'static str,
        diagnostics: String,
    },
}
impl From<io::Error> for ProcessFailure {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
impl std::fmt::Display for ProcessFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "child I/O failed: {error}"),
            Self::Timeout { phase, diagnostics } => write!(f, "{phase} timed out: {diagnostics}"),
            Self::OutputLimitExceeded { phase, diagnostics } => {
                write!(f, "{phase} output limit exceeded: {diagnostics}")
            }
        }
    }
}
impl std::error::Error for ProcessFailure {}

fn spool(
    mut input: impl Read,
    file: std::fs::File,
    path: &Path,
    limit: usize,
    exceeded: &AtomicBool,
    failed: &AtomicBool,
) -> io::Result<Vec<u8>> {
    let result = spool_inner(&mut input, file, path, limit, exceeded);
    if result.is_err() {
        failed.store(true, Ordering::Relaxed);
    }
    result
}

fn spool_inner(
    mut input: impl Read,
    mut file: std::fs::File,
    path: &Path,
    limit: usize,
    exceeded: &AtomicBool,
) -> io::Result<Vec<u8>> {
    let mut size = 0;
    let mut buffer = [0_u8; 8192];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let keep = read.min(limit - size);
        file.write_all(&buffer[..keep])?;
        size += keep;
        if keep != read {
            exceeded.store(true, Ordering::Relaxed);
            break;
        }
    }
    file.flush()?;
    // Loading is bounded by the same spool limit, never by child output size.
    std::fs::read(path)
}

pub fn run(
    command: &mut Command,
    directory: &Path,
    phase: &'static str,
    timeout: Duration,
    stdout_limit: usize,
) -> Result<Output, ProcessFailure> {
    // Reject unusable spools before launching a process tree. Reader-thread
    // scheduling must not determine how long a setup failure takes to surface.
    let out_path = directory.join(format!("{phase}.stdout"));
    let err_path = directory.join(format!("{phase}.stderr"));
    let out_file = std::fs::File::create(&out_path)?;
    let err_file = std::fs::File::create(&err_path)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut tree = tree::Tree::spawn(command)?;
    let stdout = tree
        .child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("stdout pipe absent"))?;
    let stderr = tree
        .child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("stderr pipe absent"))?;
    let exceeded = Arc::new(AtomicBool::new(false));
    let failed = AtomicBool::new(false);
    let started = Instant::now();
    thread::scope(|scope| {
        // Own the tree inside the scope closure: unwind must kill children
        // before thread::scope waits for pipe-reader threads to finish.
        let mut tree = tree;
        let out = scope.spawn(|| {
            spool(
                stdout,
                out_file,
                &out_path,
                stdout_limit,
                &exceeded,
                &failed,
            )
        });
        let err = scope.spawn(|| {
            spool(
                stderr,
                err_file,
                &err_path,
                4 * 1024 * 1024,
                &exceeded,
                &failed,
            )
        });
        let result = loop {
            if exceeded.load(Ordering::Relaxed)
                || failed.load(Ordering::Relaxed)
                || started.elapsed() >= timeout
            {
                break None;
            }
            match tree.child.try_wait() {
                Ok(Some(status)) => break Some(Ok(status)),
                Ok(None) => thread::sleep(Duration::from_millis(2)),
                Err(error) => break Some(Err(error)),
            }
        };
        // Terminate remaining group members even when the leader exited, before
        // joining readers that a descendant could otherwise keep blocked.
        tree.terminate();
        let stdout = out
            .join()
            .map_err(|_| io::Error::other("stdout reader panicked"))??;
        let stderr = err
            .join()
            .map_err(|_| io::Error::other("stderr reader panicked"))??;
        let diagnostics = || {
            let start = stderr.len().saturating_sub(8192);
            format!(
                "command={command:?} elapsed={:?}\nstderr tail={}",
                started.elapsed(),
                String::from_utf8_lossy(&stderr[start..])
            )
        };
        if exceeded.load(Ordering::Relaxed) {
            return Err(ProcessFailure::OutputLimitExceeded {
                phase,
                diagnostics: diagnostics(),
            });
        }
        let status = result.ok_or_else(|| ProcessFailure::Timeout {
            phase,
            diagnostics: diagnostics(),
        })??;
        Ok(Output {
            status,
            stdout,
            stderr,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;

    struct FailedReader;
    impl Read for FailedReader {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("injected pipe read failure"))
        }
    }

    #[test]
    fn running_spool_io_failures_notify_the_process_owner() {
        let root = super::super::tests::TestRoot::new();
        let work = super::super::storage::Workspace::create(&root.0).or_invariant("workspace");
        let path = work.path.join("spool");
        let exceeded = AtomicBool::new(false);
        let failed = AtomicBool::new(false);
        let file = std::fs::File::create(&path).or_invariant("spool");
        assert!(spool(FailedReader, file, &path, 1024, &exceeded, &failed).is_err());
        assert!(failed.load(Ordering::Relaxed));
        assert!(!exceeded.load(Ordering::Relaxed));

        let failed = AtomicBool::new(false);
        let read_only = std::fs::File::open(&path).or_invariant("read-only spool");
        assert!(spool(&b"output"[..], read_only, &path, 1024, &exceeded, &failed).is_err());
        assert!(failed.load(Ordering::Relaxed));
        assert!(!exceeded.load(Ordering::Relaxed));
        work.finish().or_invariant("cleanup after reader failure");
    }
}
