use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiExit {
    Completed,
    Interrupted,
    Terminated,
    Hangup,
}

impl TuiExit {
    pub const fn process_code(self) -> i32 {
        match self {
            Self::Completed => 0,
            Self::Interrupted => 130,
            Self::Terminated => 143,
            Self::Hangup => 129,
        }
    }
}

struct TerminationState {
    reason: AtomicU8,
    worker_stop: Mutex<Option<Weak<AtomicBool>>>,
}

impl TerminationState {
    const fn new() -> Self {
        Self {
            reason: AtomicU8::new(0),
            worker_stop: Mutex::new(None),
        }
    }

    fn register_worker(&self, stop: &Arc<AtomicBool>) {
        match self.worker_stop.lock() {
            Ok(mut registered) => *registered = Some(Arc::downgrade(stop)),
            Err(poisoned) => *poisoned.into_inner() = Some(Arc::downgrade(stop)),
        }
        if self.exit() != TuiExit::Completed {
            stop.store(true, Ordering::Release);
        }
    }

    fn request(&self, reason: TuiExit) {
        let value = match reason {
            TuiExit::Completed => return,
            TuiExit::Interrupted => 1,
            TuiExit::Terminated => 2,
            TuiExit::Hangup => 3,
        };
        // Preserve the first accepted cause, including during worker shutdown.
        let _ = self
            .reason
            .compare_exchange(0, value, Ordering::AcqRel, Ordering::Acquire);
        let worker = match self.worker_stop.lock() {
            Ok(registered) => registered.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        if let Some(stop) = worker.and_then(|registered| registered.upgrade()) {
            stop.store(true, Ordering::Release);
        }
    }

    fn exit(&self) -> TuiExit {
        match self.reason.load(Ordering::Acquire) {
            1 => TuiExit::Interrupted,
            2 => TuiExit::Terminated,
            3 => TuiExit::Hangup,
            _ => TuiExit::Completed,
        }
    }
}

static STATE: TerminationState = TerminationState::new();
static INSTALL_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

pub(super) fn prepare() -> Result<(), String> {
    INSTALL_HANDLER.get_or_init(install_handler).clone()
}

#[cfg(unix)]
fn install_handler() -> Result<(), String> {
    use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    // ctrlc's callback erases the Unix signal number; retain it for exit status.
    let mut signals = Signals::new([SIGINT, SIGTERM, SIGHUP])
        .map_err(|error| format!("failed to install termination handlers: {error}"))?;
    std::thread::Builder::new()
        .name("termination".to_string())
        .spawn(move || {
            for signal in signals.forever() {
                let reason = match signal {
                    SIGINT => TuiExit::Interrupted,
                    SIGTERM => TuiExit::Terminated,
                    SIGHUP => TuiExit::Hangup,
                    _ => continue,
                };
                STATE.request(reason);
            }
        })
        .map_err(|error| format!("failed to start termination handler: {error}"))?;
    Ok(())
}

#[cfg(windows)]
fn install_handler() -> Result<(), String> {
    ctrlc::set_handler(request)
        .map_err(|error| format!("failed to install Ctrl-C handler: {error}"))
}

#[cfg(not(any(unix, windows)))]
fn install_handler() -> Result<(), String> {
    Err("interactive termination handling is unsupported on this platform".to_string())
}

pub(super) fn register_worker(stop: &Arc<AtomicBool>) {
    STATE.register_worker(stop);
}

pub(super) fn request() {
    STATE.request(TuiExit::Interrupted);
}

pub(super) fn requested() -> bool {
    exit() != TuiExit::Completed
}

pub(super) fn exit() -> TuiExit {
    STATE.exit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_termination_cause_stops_the_worker_and_survives_later_requests() {
        for (reason, code) in [
            (TuiExit::Interrupted, 130),
            (TuiExit::Terminated, 143),
            (TuiExit::Hangup, 129),
        ] {
            let state = TerminationState::new();
            let stop = Arc::new(AtomicBool::new(false));
            state.register_worker(&stop);
            state.request(reason);
            state.request(TuiExit::Interrupted);
            assert!(
                stop.load(Ordering::Acquire),
                "{reason:?} did not stop worker"
            );
            assert_eq!(state.exit(), reason, "later request replaced {reason:?}");
            assert_eq!(state.exit().process_code(), code);
        }
    }

    #[test]
    fn termination_before_worker_registration_is_not_lost() {
        let state = TerminationState::new();
        state.request(TuiExit::Terminated);
        let stop = Arc::new(AtomicBool::new(false));
        state.register_worker(&stop);
        assert!(stop.load(Ordering::Acquire));
        assert_eq!(state.exit(), TuiExit::Terminated);
    }
}
