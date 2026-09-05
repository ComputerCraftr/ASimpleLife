//! Camera intent and samples are separate from simulation and analysis state.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ObservationToken {
    // Valid only while the worker retains exclusive access to its session; no
    // root handle or mutable simulation borrow crosses a presentation commit.
    pub source_revision: u64,
    pub state_revision: u64,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedSample {
    observation: ObservationToken,
    camera_revision: u64,
    sample: crate::render::ViewportSample,
}

pub(super) struct PreparedCamera {
    viewport: ViewportController,
    observation: ObservationToken,
    base_revision: u64,
    sample: crate::render::ViewportSample,
}

pub(super) struct Presentation {
    pub request: ViewportRequest,
    pub viewport: ViewportController,
    pub focus: ActiveFocus,
    pub revision: u64,
    pub pending: bool,
    cached: Option<PreparedSample>,
}

impl Presentation {
    pub fn new(viewport: ViewportController) -> Self {
        Self {
            viewport,
            request: ViewportRequest {
                revision: 0,
                width: 80,
                height: 20,
                origin: None,
                auto: true,
                recenter: true,
            },
            focus: ActiveFocus::default(),
            revision: 0,
            pending: true,
            cached: None,
        }
    }

    pub fn apply_request(
        &mut self,
        next: ViewportRequest,
        session: &mut SimulationSession,
        observation: ObservationToken,
    ) -> Result<(), String> {
        if next.revision < self.request.revision {
            return Ok(());
        }
        let mut candidate = self.viewport.clone();
        candidate
            .resize(usize::from(next.width), usize::from(next.height))
            .map_err(|error| error.to_string())?;
        candidate.set_mode(if next.auto {
            ViewportMode::Auto
        } else {
            ViewportMode::Manual
        });
        if !next.auto
            && let Some(origin) = next.origin
        {
            candidate.set_origin(origin);
        }
        if next.recenter {
            candidate
                .recenter(session)
                .map_err(|error| error.to_string())?;
        }
        let base_revision = self.revision;
        let prepared = self.prepare_camera(candidate, session, observation, base_revision)?;
        if !self.commit_camera(prepared, observation)? {
            return Err("viewport request was superseded before camera commit".into());
        }
        self.request = next;
        self.pending = true;
        Ok(())
    }

    pub fn prepare_camera(
        &self,
        mut candidate: ViewportController,
        session: &mut SimulationSession,
        observation: ObservationToken,
        base_revision: u64,
    ) -> Result<PreparedCamera, String> {
        if observation.generation != session.hashlife_generation() {
            return Err("viewport observation changed before camera preparation".into());
        }
        let sample = candidate
            .sample_focus(session)
            .map_err(|error| format!("viewport sample failed: {error}"))?;
        Ok(PreparedCamera {
            viewport: candidate,
            observation,
            base_revision,
            sample,
        })
    }

    /// Publish only a sample prepared against the still-current simulation and
    /// camera. No simulation/root borrow crosses this synchronous safepoint.
    pub fn commit_camera(
        &mut self,
        prepared: PreparedCamera,
        current: ObservationToken,
    ) -> Result<bool, String> {
        if prepared.observation != current || self.revision != prepared.base_revision {
            return Ok(false);
        }
        let revision = prepared
            .base_revision
            .checked_add(1)
            .ok_or("camera revisions exhausted")?;
        self.viewport = prepared.viewport;
        self.revision = revision;
        self.cached = Some(PreparedSample {
            observation: prepared.observation,
            camera_revision: revision,
            sample: prepared.sample,
        });
        Ok(true)
    }

    pub fn sample(
        &mut self,
        session: &mut SimulationSession,
        observation: ObservationToken,
    ) -> Result<crate::render::ViewportSample, String> {
        self.prepare_sample(session, observation)?;
        self.cached
            .as_ref()
            .filter(|cached| {
                cached.observation == observation && cached.camera_revision == self.revision
            })
            .map(|cached| cached.sample.clone())
            .ok_or_else(|| "prepared viewport sample is stale".into())
    }

    pub fn invalidate_samples(&mut self) {
        self.cached = None;
    }

    pub fn prepare_sample(
        &mut self,
        session: &mut SimulationSession,
        observation: ObservationToken,
    ) -> Result<(), String> {
        if observation.generation != session.hashlife_generation() {
            return Err("viewport observation changed before sampling".into());
        }
        if self.cached.as_ref().is_some_and(|cached| {
            cached.observation == observation
                && cached.camera_revision == self.revision
                && Some(cached.sample.origin) == self.viewport.origin()
        }) {
            return Ok(());
        }
        let mut candidate = self.viewport.clone();
        let sample = candidate
            .sample_focus(session)
            .map_err(|error| format!("viewport sample failed: {error}"))?;
        if observation.generation != session.hashlife_generation() {
            return Err("viewport observation changed while sampling".into());
        }
        let camera_changed = candidate.origin() != self.viewport.origin()
            || candidate.dimensions() != self.viewport.dimensions()
            || candidate.mode() != self.viewport.mode();
        let camera_revision = if camera_changed {
            self.revision
                .checked_add(1)
                .ok_or("camera revisions exhausted")?
        } else {
            self.revision
        };
        self.viewport = candidate;
        self.revision = camera_revision;
        self.cached = Some(PreparedSample {
            observation,
            camera_revision,
            sample,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;

    fn loaded_session() -> SimulationSession {
        let mut session = SimulationSession::new();
        session
            .try_load_hashlife_state(&BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]))
            .or_invariant("source");
        session
    }

    fn token(session: &SimulationSession) -> ObservationToken {
        ObservationToken {
            source_revision: 1,
            state_revision: 1,
            generation: session.hashlife_generation(),
        }
    }

    fn request(revision: u64) -> ViewportRequest {
        ViewportRequest {
            revision,
            width: 40,
            height: 10,
            origin: Some((-10, -10)),
            auto: false,
            recenter: false,
        }
    }

    #[test]
    fn failed_region_sample_preserves_committed_camera_and_last_valid_sample() {
        let mut session = loaded_session();
        let mut presentation =
            Presentation::new(ViewportController::new(40, 10).or_invariant("camera"));
        let initial = request(1);
        let observation = token(&session);
        presentation
            .apply_request(initial, &mut session, observation)
            .or_invariant("valid sample");
        let saved = presentation.cached.clone();
        session = SimulationSession::new();
        let invalid = ViewportRequest {
            revision: 2,
            origin: Some((100, 100)),
            ..initial
        };
        let observation = token(&session);
        assert!(
            presentation
                .apply_request(invalid, &mut session, observation)
                .is_err(),
            "unloaded source was fabricated as an empty universe"
        );
        assert_eq!(presentation.request, initial);
        assert_eq!(presentation.viewport.origin(), initial.origin);
        assert_eq!(presentation.revision, 1);
        assert_eq!(
            presentation.cached, saved,
            "failed sample replaced valid evidence"
        );
    }

    #[test]
    fn stale_observation_fields_cannot_commit_a_sampled_camera() {
        let mut session = loaded_session();
        let observation = token(&session);
        let mut presentation =
            Presentation::new(ViewportController::new(40, 10).or_invariant("camera"));
        presentation
            .apply_request(request(1), &mut session, observation)
            .or_invariant("initial camera");
        let saved = presentation.cached.clone();
        let stale = [
            ObservationToken {
                source_revision: 2,
                ..observation
            },
            ObservationToken {
                state_revision: 2,
                ..observation
            },
            ObservationToken {
                generation: 1,
                ..observation
            },
        ];
        for current in stale {
            let mut candidate = presentation.viewport.clone();
            candidate.set_origin((500, 500));
            let prepared = presentation
                .prepare_camera(candidate, &mut session, observation, 1)
                .or_invariant("candidate sample");
            assert!(
                !presentation
                    .commit_camera(prepared, current)
                    .or_invariant("stale work is rejected")
            );
        }
        assert_eq!(presentation.viewport.origin(), request(1).origin);
        assert_eq!(presentation.revision, 1);
        assert_eq!(presentation.cached, saved);
    }

    #[test]
    fn newer_camera_revision_rejects_an_older_prepared_camera() {
        let mut session = loaded_session();
        let observation = token(&session);
        let mut presentation =
            Presentation::new(ViewportController::new(40, 10).or_invariant("camera"));
        presentation
            .apply_request(request(1), &mut session, observation)
            .or_invariant("initial camera");
        let mut old_candidate = presentation.viewport.clone();
        old_candidate.set_origin((500, 500));
        let prepared = presentation
            .prepare_camera(old_candidate, &mut session, observation, 1)
            .or_invariant("old candidate sample");
        let newer = ViewportRequest {
            revision: 2,
            origin: Some((100, 100)),
            ..request(1)
        };
        presentation
            .apply_request(newer, &mut session, observation)
            .or_invariant("newer camera");
        let saved = presentation.cached.clone();
        assert!(
            !presentation
                .commit_camera(prepared, observation)
                .or_invariant("older camera is rejected")
        );
        assert_eq!(presentation.viewport.origin(), newer.origin);
        assert_eq!(presentation.revision, 2);
        assert_eq!(presentation.cached, saved);
    }

    #[test]
    fn failed_automatic_sample_and_zero_dimensions_preserve_camera() {
        let mut session = loaded_session();
        let observation = token(&session);
        let mut presentation =
            Presentation::new(ViewportController::new(40, 10).or_invariant("camera"));
        presentation
            .apply_request(request(1), &mut session, observation)
            .or_invariant("initial camera");
        let saved = presentation.cached.clone();
        let saved_origin = presentation.viewport.origin();
        let mut unavailable = presentation.viewport.clone();
        unavailable.set_origin((i64::MAX, i64::MAX));
        assert!(
            presentation
                .prepare_camera(unavailable, &mut session, observation, 1)
                .is_err()
        );
        let zero = ViewportRequest {
            revision: 2,
            width: 0,
            ..request(1)
        };
        assert!(
            presentation
                .apply_request(zero, &mut session, observation)
                .is_err()
        );
        assert_eq!(presentation.viewport.origin(), saved_origin);
        assert_eq!(presentation.revision, 1);
        assert_eq!(presentation.cached, saved);
    }
}
