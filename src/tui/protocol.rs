use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub use super::analysis::{AnalysisScope, AnalysisUpdate};
use crate::bitgrid::{BitGrid, Cell};
use crate::engine::SimulationSession;

pub const MAX_CONTINUOUS_QUANTUM: u64 = 1_u64 << 63;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    CommandSequenceExhausted,
    AnalysisRequestExhausted,
    AnalysisConfigurationExhausted,
    WorkerDisconnected,
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::CommandSequenceExhausted => "command sequence exhausted",
            Self::AnalysisRequestExhausted => "analysis request IDs exhausted",
            Self::AnalysisConfigurationExhausted => "analysis configuration revisions exhausted",
            Self::WorkerDisconnected => "simulation worker disconnected",
        })
    }
}

impl std::error::Error for ProtocolError {}

#[derive(Debug)]
pub struct PreparedSource {
    pub session: SimulationSession,
    pub label: String,
}

#[derive(Debug)]
pub enum ControlCommand {
    Pause,
    Resume,
    ToggleRunning,
    StepOne,
    AdvanceBy(u64),
    AdvanceTo(u64),
    FocusNext(ViewportRequest),
    FocusPrevious(ViewportRequest),
    ResetAutoFocus,
    Save(PathBuf),
    ReplaceSource(Box<PreparedSource>),
    Classify {
        source_revision: u64,
        scope: AnalysisScope,
    },
    CancelClassification,
    ConfigureAnalysis {
        max_generations: u64,
    },
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerTuning {
    pub quantum: u64,
    pub interval_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewportRequest {
    pub revision: u64,
    pub width: u16,
    pub height: u16,
    pub origin: Option<Cell>,
    pub auto: bool,
    pub recenter: bool,
}

#[derive(Clone, Debug)]
pub struct RenderSnapshot {
    pub worker_state_seq: u64,
    pub camera_revision: u64,
    pub viewport_revision: u64,
    pub source_revision: u64,
    pub generation: u64,
    pub population: u128,
    pub backend: &'static str,
    pub source: String,
    pub running: bool,
    pub quantum: u64,
    pub origin: Cell,
    pub grid: BitGrid,
    pub status: String,
    pub output: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerStatus {
    pub worker_state_seq: u64,
    pub source_revision: u64,
    pub state_revision: u64,
    pub generation: u64,
    pub running: bool,
    pub quantum: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerEvent {
    Acknowledged { request_id: u64 },
    Analysis(AnalysisUpdate),
    AnalysisConfiguration(u64),
    CommandCompleted(String),
    Error(String),
    Saved { path: PathBuf, generation: u64 },
    SourceReplaced { label: String, revision: u64 },
    Stopped,
}

#[derive(Debug)]
pub struct LatestValue<T> {
    inner: Mutex<Option<T>>,
}

impl<T> Default for LatestValue<T> {
    fn default() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }
}

impl<T> LatestValue<T> {
    pub fn replace(&self, value: T) {
        match self.inner.lock() {
            Ok(mut slot) => *slot = Some(value),
            Err(poisoned) => *poisoned.into_inner() = Some(value),
        }
    }

    pub fn take(&self) -> Option<T> {
        match self.inner.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
    }
}

pub type SharedLatest<T> = Arc<LatestValue<T>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_value_replaces_stale_presentation_state() {
        let latest = LatestValue::default();
        latest.replace(1);
        latest.replace(2);
        latest.replace(3);
        assert_eq!(latest.take(), Some(3));
        assert_eq!(latest.take(), None);
    }

    #[test]
    fn continuous_quantum_limit_is_a_representable_u64_delta() {
        assert_eq!(MAX_CONTINUOUS_QUANTUM, 9_223_372_036_854_775_808);
        assert_eq!(MAX_CONTINUOUS_QUANTUM.checked_mul(2), None);
    }
}
