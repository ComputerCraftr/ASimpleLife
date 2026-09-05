use super::protocol::ProtocolError;
use crate::bitgrid::Bounds;
use crate::classify::{
    ClassificationLimits, ClassificationReport,
    analysis::{AnalysisFailure, classify_capture, describe_report},
};
use crate::engine::SimulationSession;
use crate::hashlife::session::capture::{CaptureError, CaptureLimits};
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisScope {
    WholeUniverse,
    IsolatedRegion(Bounds),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureDescriptor {
    pub request_id: u64,
    pub source_revision: u64,
    pub state_revision: u64,
    pub generation: u64,
    pub configuration: u64,
    pub scope: AnalysisScope,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnalysisStatus {
    Pending,
    Running,
    Completed(ClassificationReport),
    Cancelled,
    Unavailable(String),
    Failed(String),
    Superseded,
}

impl AnalysisStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Pending | Self::Running)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisUpdate {
    pub request_id: u64,
    pub descriptor: Option<CaptureDescriptor>,
    pub status: AnalysisStatus,
}

impl AnalysisUpdate {
    pub fn describe(&self) -> String {
        let result = match &self.status {
            AnalysisStatus::Completed(report) => {
                let absolute = self
                    .descriptor
                    .and_then(|capture| capture.generation.checked_add(report.observed_through));
                format!(
                    "{}; observed after={} absolute={}",
                    describe_report(report),
                    report.observed_through,
                    absolute.map_or_else(
                        || "out of range".to_string(),
                        |generation| generation.to_string()
                    )
                )
            }
            other => format!("{other:?}"),
        };
        self.descriptor.map_or(result.clone(), |capture| {
            format!(
                "{result}; captured generation={} scope={:?}{}",
                capture.generation,
                capture.scope,
                if matches!(capture.scope, AnalysisScope::IsolatedRegion(_)) {
                    " (isolated seed; outside initially dead, no wrapping)"
                } else {
                    ""
                }
            )
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AnalysisLimits {
    pub aggregate_bytes: u128,
    pub capture: CaptureLimits,
    pub generations: u64,
}

impl Default for AnalysisLimits {
    fn default() -> Self {
        Self {
            aggregate_bytes: 128 * 1024 * 1024,
            capture: CaptureLimits::default(),
            generations: 512,
        }
    }
}

struct Request {
    id: u64,
    source: u64,
    scope: AnalysisScope,
}
struct Active {
    descriptor: CaptureDescriptor,
    cancelled: Arc<AtomicBool>,
    terminal: bool,
    thread: JoinHandle<Result<ClassificationReport, AnalysisFailure>>,
}

pub(crate) struct AnalysisCoordinator {
    next_id: u64,
    configuration: u64,
    pending: Option<Request>,
    active: Option<Active>,
    updates: VecDeque<AnalysisUpdate>,
    limits: AnalysisLimits,
}

impl Default for AnalysisCoordinator {
    fn default() -> Self {
        Self::new(AnalysisLimits::default())
    }
}

impl AnalysisCoordinator {
    pub fn new(limits: AnalysisLimits) -> Self {
        Self {
            next_id: 0,
            configuration: 1,
            pending: None,
            active: None,
            updates: VecDeque::new(),
            limits,
        }
    }
    pub fn request(&mut self, source: u64, scope: AnalysisScope) -> Result<(), ProtocolError> {
        let id = self
            .next_id
            .checked_add(1)
            .ok_or(ProtocolError::AnalysisRequestExhausted)?;
        self.cancel(AnalysisStatus::Superseded);
        self.next_id = id;
        self.pending = Some(Request { id, source, scope });
        self.updates.push_back(AnalysisUpdate {
            request_id: id,
            descriptor: None,
            status: AnalysisStatus::Pending,
        });
        Ok(())
    }
    pub fn cancel(&mut self, status: AnalysisStatus) {
        if let Some(request) = self.pending.take() {
            self.updates.push_back(AnalysisUpdate {
                request_id: request.id,
                descriptor: None,
                status: status.clone(),
            });
        }
        if let Some(active) = &mut self.active
            && !active.terminal
        {
            active.cancelled.store(true, Ordering::Relaxed);
            active.terminal = true;
            self.updates.push_back(AnalysisUpdate {
                request_id: active.descriptor.request_id,
                descriptor: Some(active.descriptor),
                status,
            });
        }
    }
    pub fn configure_generations(&mut self, generations: u64) -> Result<u64, ProtocolError> {
        if self.limits.generations != generations {
            let next = self
                .configuration
                .checked_add(1)
                .ok_or(ProtocolError::AnalysisConfigurationExhausted)?;
            self.cancel(AnalysisStatus::Superseded);
            self.configuration = next;
            self.limits.generations = generations;
        }
        Ok(self.configuration)
    }
    pub fn poll(
        &mut self,
        simulation: &SimulationSession,
        source: u64,
        revision: u64,
        interrupt: &AtomicBool,
    ) {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.thread.is_finished())
            && let Some(active) = self.active.take()
        {
            let result = active.thread.join();
            if !active.terminal {
                let status = match result {
                    Ok(Ok(report)) => AnalysisStatus::Completed(report),
                    Ok(Err(
                        AnalysisFailure::Cancelled
                        | AnalysisFailure::Resource(CaptureError::Cancelled),
                    )) => AnalysisStatus::Cancelled,
                    Ok(Err(error)) => AnalysisStatus::Unavailable(format!("{error:?}")),
                    Err(_) => AnalysisStatus::Failed("analysis worker panicked".to_string()),
                };
                self.updates.push_back(AnalysisUpdate {
                    request_id: active.descriptor.request_id,
                    descriptor: Some(active.descriptor),
                    status,
                });
            }
        }
        // No new capture overlaps storage still owned by an obsolete worker.
        if self.active.is_some() || interrupt.load(Ordering::Relaxed) {
            return;
        }
        let Some(request) = self.pending.take() else {
            return;
        };
        if request.source != source {
            self.updates.push_back(AnalysisUpdate {
                request_id: request.id,
                descriptor: None,
                status: AnalysisStatus::Superseded,
            });
            return;
        }
        let descriptor = CaptureDescriptor {
            request_id: request.id,
            source_revision: source,
            state_revision: revision,
            generation: simulation.hashlife_generation(),
            configuration: self.configuration,
            scope: request.scope,
        };
        let region = match request.scope {
            AnalysisScope::WholeUniverse => None,
            AnalysisScope::IsolatedRegion(rect) => Some(rect),
        };
        let captured = if self.limits.aggregate_bytes < self.limits.capture.bytes as u128 {
            Err(CaptureError::TooLarge)
        } else {
            simulation.capture_analysis(region, self.limits.capture, interrupt)
        };
        match captured {
            Err(error) => self.updates.push_back(AnalysisUpdate {
                request_id: request.id,
                descriptor: Some(descriptor),
                status: AnalysisStatus::Unavailable(format!("AnalysisCapture{error:?}")),
            }),
            Ok(capture) => {
                let cancelled = Arc::new(AtomicBool::new(false));
                let cancel = Arc::clone(&cancelled);
                let limits = self.limits;
                let thread =
                    thread::Builder::new()
                        .name("classification".into())
                        .spawn(move || {
                            classify_capture(
                                capture,
                                &ClassificationLimits {
                                    max_generations: limits.generations,
                                },
                                limits.aggregate_bytes,
                                &cancel,
                            )
                        });
                let thread = match thread {
                    Ok(thread) => thread,
                    Err(error) => {
                        self.updates.push_back(AnalysisUpdate {
                            request_id: request.id,
                            descriptor: Some(descriptor),
                            status: AnalysisStatus::Failed(format!(
                                "cannot start analysis: {error}"
                            )),
                        });
                        return;
                    }
                };
                self.active = Some(Active {
                    descriptor,
                    cancelled,
                    terminal: false,
                    thread,
                });
                self.updates.push_back(AnalysisUpdate {
                    request_id: request.id,
                    descriptor: Some(descriptor),
                    status: AnalysisStatus::Running,
                });
            }
        }
    }
    pub fn take_update(&mut self) -> Option<AnalysisUpdate> {
        self.updates.pop_front()
    }
}

impl Drop for AnalysisCoordinator {
    fn drop(&mut self) {
        if let Some(active) = self.active.take() {
            active.cancelled.store(true, Ordering::Relaxed);
            let _ = active.thread.join();
        }
    }
}
