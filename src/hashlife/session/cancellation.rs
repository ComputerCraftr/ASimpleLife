use super::*;

impl HashLifeSession {
    pub(crate) fn advance_root_cancellable(
        &mut self,
        delta: u64,
        cancelled: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<SessionAdvanceStats, HashLifeAdvanceError> {
        self.advance_root_interruptible(delta, cancelled, cancelled)
    }

    pub(crate) fn advance_root_interruptible(
        &mut self,
        delta: u64,
        cancelled: &std::sync::Arc<std::sync::atomic::AtomicBool>,
        stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<SessionAdvanceStats, HashLifeAdvanceError> {
        if cancelled.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
            return Err(HashLifeAdvanceError::Cancelled {
                starting_generation: self.current_generation,
                requested_delta: delta,
                completed_generations: 0,
                reached_generation: self.current_generation,
            });
        }
        self.engine.advance_cancellation = Some([
            std::sync::Arc::clone(cancelled),
            std::sync::Arc::clone(stop),
        ]);
        let result = self.advance_root(delta);
        self.engine.advance_cancellation = None;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariant::RequiredErrorExt;
    use std::sync::{Arc, atomic::AtomicBool};
    #[test]
    fn cancelled_advance_retains_generation_and_is_retryable() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]))
            .or_invariant("blinker");
        let cancelled = Arc::new(AtomicBool::new(true));
        let error = session
            .advance_root_cancellable(100, &cancelled)
            .error_or_invariant("cancelled advance");
        assert!(matches!(
            error,
            HashLifeAdvanceError::Cancelled {
                completed_generations: 0,
                reached_generation: 0,
                ..
            }
        ));
        assert_eq!(session.generation(), 0);
        cancelled.store(false, Ordering::Relaxed);
        session
            .advance_root_cancellable(100, &cancelled)
            .or_invariant("retry after cancellation");
        assert_eq!(session.generation(), 100);
    }
}
