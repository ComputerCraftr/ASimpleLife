//! Cancellable analysis of an independent, exact DAG occurrence.
use super::*;
use crate::hashlife::session::capture::{CaptureError, OwnedDag};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AnalysisFailure {
    Cancelled,
    Resource(CaptureError),
    Advance(String),
}

pub(crate) fn classify_capture(
    capture: OwnedDag,
    limits: &ClassificationLimits,
    memory_bytes: u128,
    cancelled: &Arc<AtomicBool>,
) -> Result<ClassificationReport, AnalysisFailure> {
    // This reserve covers the bounded recurrence evidence, heuristic samples,
    // their scratch storage, and the returned report, not just the engine DAG.
    let engine_budget = memory_bytes
        .checked_sub(16 * 1024 * 1024)
        .ok_or(AnalysisFailure::Resource(CaptureError::TooLarge))?;
    let mut session = capture
        .into_analysis_session(engine_budget, cancelled)
        .map_err(AnalysisFailure::Resource)?;
    classify_session(&mut session, limits, cancelled)
}

pub(crate) fn classify_session(
    session: &mut crate::hashlife::HashLifeSession,
    limits: &ClassificationLimits,
    cancelled: &Arc<AtomicBool>,
) -> Result<ClassificationReport, AnalysisFailure> {
    let lineage = Lineage::fresh();
    let mut evidence = super::evaluator::EvidenceEvaluator::new(
        ExactRecurrenceTracker::new(lineage),
        limits,
        limits.max_generations,
    );
    let mut observed = 0;
    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Err(AnalysisFailure::Cancelled);
        }
        let observation = session.try_recurrence_observation(lineage);
        let empty = session
            .population_count()
            .is_some_and(|count| count.lower_bound() == 0);
        // Optional heuristics see only a complete, bounded independent universe.
        let grid = session
            .extract_grid(crate::hashlife::GridExtractionPolicy::FullGridIfUnder {
                max_population: 4096,
                max_chunks: 512,
                max_bounds_span: 4096,
            })
            .ok();
        if let Some(report) = evidence.observe(observed, observation, empty, grid.as_ref()) {
            return Ok(report);
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(AnalysisFailure::Cancelled);
        }
        session
            .advance_root_cancellable(1, cancelled)
            .map_err(|error| {
                if matches!(
                    error,
                    crate::hashlife::HashLifeAdvanceError::Cancelled { .. }
                ) {
                    AnalysisFailure::Cancelled
                } else {
                    AnalysisFailure::Advance(format!("{error:?}"))
                }
            })?;
        observed = observed
            .checked_add(1)
            .ok_or_else(|| AnalysisFailure::Advance("analysis clock exhausted".into()))?;
    }
}

pub(crate) fn describe_report(report: &ClassificationReport) -> String {
    match report.evidence {
        ClassificationEvidence::Extinction { at_generation: 0 } => "empty at capture".to_string(),
        ClassificationEvidence::Extinction { at_generation } => {
            format!("becomes extinct after {at_generation} generations")
        }
        ClassificationEvidence::Recurrence {
            period,
            first_seen,
            displacement,
            ..
        } => {
            let kind = if displacement != (0, 0) {
                "spaceship"
            } else if period == 1 {
                "still life"
            } else {
                "oscillator"
            };
            if first_seen == 0 {
                format!("{kind} at capture (period={period}, displacement={displacement:?})")
            } else {
                format!(
                    "settles into {kind} after {first_seen} generations (period={period}, displacement={displacement:?})"
                )
            }
        }
        _ => format!("{:?} ({:?})", report.outcome, report.certainty),
    }
}
