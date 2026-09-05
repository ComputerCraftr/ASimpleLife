use super::*;
use crate::tui::protocol::ProtocolError;
use std::sync::Mutex;

pub(super) struct Envelope {
    pub id: u64,
    pub command: ControlCommand,
}
struct Outbound {
    sender: Sender<Envelope>,
    sequence: u64,
    interruption: Option<u64>,
    pause: Option<u64>,
}

pub(super) struct CommandClient {
    outbound: Mutex<Outbound>,
    pub interrupt: Arc<AtomicBool>,
    pub pause: Arc<AtomicBool>,
}

impl CommandClient {
    pub fn new(sender: Sender<Envelope>, interrupt: Arc<AtomicBool>) -> Self {
        Self {
            outbound: Mutex::new(Outbound {
                sender,
                sequence: 0,
                interruption: None,
                pause: None,
            }),
            interrupt,
            pause: Arc::new(AtomicBool::new(false)),
        }
    }
    pub fn send(&self, command: ControlCommand) -> Result<u64, ProtocolError> {
        let mut state = self
            .outbound
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let id = state
            .sequence
            .checked_add(1)
            .ok_or(ProtocolError::CommandSequenceExhausted)?;
        let interrupts = matches!(
            command,
            ControlCommand::Pause
                | ControlCommand::ToggleRunning
                | ControlCommand::StepOne
                | ControlCommand::AdvanceBy(_)
                | ControlCommand::AdvanceTo(_)
                | ControlCommand::Shutdown
        );
        let old = state.interruption;
        let old_pause = state.pause;
        if matches!(command, ControlCommand::Pause) {
            state.pause = Some(id);
            self.pause.store(true, Ordering::Relaxed);
        }
        if interrupts {
            state.interruption = Some(id);
            self.interrupt.store(true, Ordering::Relaxed);
        }
        if state.sender.send(Envelope { id, command }).is_err() {
            state.interruption = old;
            state.pause = old_pause;
            self.pause.store(old_pause.is_some(), Ordering::Relaxed);
            self.interrupt.store(old.is_some(), Ordering::Relaxed);
            return Err(ProtocolError::WorkerDisconnected);
        }
        state.sequence = id;
        Ok(id)
    }
    pub fn acknowledge(&self, id: u64) {
        let mut state = self
            .outbound
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.pause == Some(id) {
            state.pause = None;
            self.pause.store(false, Ordering::Relaxed);
        }
        if state.interruption == Some(id) {
            state.interruption = None;
            self.interrupt.store(false, Ordering::Relaxed);
        }
    }
}

impl WorkerHandle {
    pub fn submit(&self, command: ControlCommand) -> Result<u64, ProtocolError> {
        self.commands.send(command)
    }
    pub fn next_frame(&self) -> Option<RenderSnapshot> {
        self.frames.take()
    }
    pub fn next_status(&self) -> Option<WorkerStatus> {
        self.status.take()
    }
    pub fn set_viewport(&self, request: ViewportRequest) {
        self.viewport.replace(request);
    }
    pub fn set_tuning(&self, tuning: WorkerTuning) {
        self.tuning.replace(tuning);
    }
    pub fn events(&self) -> &Receiver<WorkerEvent> {
        &self.events
    }
    pub fn termination_token(&self) -> &Arc<AtomicBool> {
        &self.stop_requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    #[test]
    fn unrelated_acknowledgments_cannot_consume_newer_interruptions() {
        let (tx, _rx) = mpsc::channel();
        let client = CommandClient::new(tx, Arc::new(AtomicBool::new(false)));
        let first = client.send(ControlCommand::Pause).or_invariant("pause");
        let next = client
            .send(ControlCommand::ToggleRunning)
            .or_invariant("toggle");
        client.acknowledge(first);
        assert!(
            client.interrupt.load(Ordering::Relaxed),
            "old acknowledgment consumed a newer interruption"
        );
        client.acknowledge(next);
        assert!(!client.interrupt.load(Ordering::Relaxed));
    }
}
