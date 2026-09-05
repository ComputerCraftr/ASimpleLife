use super::*;

pub(crate) struct SelectionCheckpoint {
    selected: Option<u64>,
    pinned: bool,
    selected_at: Option<Instant>,
    challenger: Option<PairedHistory>,
    missing: usize,
    negative_generation: Option<u64>,
    evidence: FocusEvidence,
    mode: Option<ViewportMode>,
    initial_generation: Option<u64>,
    last_observation: Option<Instant>,
    last_validation: Option<u64>,
}

impl ActiveFocus {
    pub(crate) fn selection_checkpoint(&self) -> SelectionCheckpoint {
        SelectionCheckpoint {
            selected: self.selected,
            pinned: self.pinned,
            selected_at: self.selected_at,
            challenger: self.challenger.clone(),
            missing: self.missing,
            negative_generation: self.negative_generation,
            evidence: self.evidence,
            mode: self.mode,
            initial_generation: self.initial_generation,
            last_observation: self.last_observation,
            last_validation: self.last_validation,
        }
    }

    pub(crate) fn restore_selection(&mut self, checkpoint: SelectionCheckpoint) {
        self.selected = checkpoint.selected;
        self.pinned = checkpoint.pinned;
        self.selected_at = checkpoint.selected_at;
        self.challenger = checkpoint.challenger;
        self.missing = checkpoint.missing;
        self.negative_generation = checkpoint.negative_generation;
        self.evidence = checkpoint.evidence;
        self.mode = checkpoint.mode;
        self.initial_generation = checkpoint.initial_generation;
        self.last_observation = checkpoint.last_observation;
        self.last_validation = checkpoint.last_validation;
    }

    pub(super) fn choose(
        &mut self,
        best: Option<usize>,
        generation: u64,
        now: Instant,
    ) -> Option<usize> {
        let best = best?;
        let Some(current) = self.selected_index() else {
            // A paused initial scan can finish before choosing; it must not
            // lock onto page one's group and then require future generations
            // just to discover that page two contained a larger organism.
            if self.initial_generation == Some(generation) && !self.catalog.finished {
                return None;
            }
            return Some(best);
        };
        if self.tracks[current].id == self.tracks[best].id {
            return None;
        }
        if self.missing >= 2
            && matches!(self.evidence, FocusEvidence::Stable | FocusEvidence::Absent)
        {
            return Some(best);
        }
        if self.pinned
            || self.evidence != FocusEvidence::Active
            || self.tracks[current].group.generation != generation
        {
            return None;
        }
        let id = self.tracks[best].id;
        if self
            .challenger
            .as_ref()
            .is_none_or(|history| history.id != id)
        {
            self.challenger = Some(PairedHistory::new(id));
        }
        let qualified = self.challenger.as_mut().is_some_and(|history| {
            history.observe(&self.tracks[current].group, &self.tracks[best].group)
        });
        (qualified
            && self
                .selected_at
                .is_some_and(|at| now.saturating_duration_since(at) >= Duration::from_secs(2)))
        .then_some(best)
    }

    pub fn navigate(
        &mut self,
        session: &mut SimulationSession,
        viewport: &mut ViewportController,
        previous: bool,
    ) -> Result<bool, ViewportError> {
        let now = Instant::now();
        // A key press owns its inspection budget and starting selection. An
        // automatic refresh here used to spend that budget and move selection
        // before the requested next/previous action even began.
        self.catalog.begin_observation();
        self.mode = Some(viewport.mode());
        if self.tracks.is_empty() {
            self.catalog.refresh_budgeted(session, now, true);
            self.merge_discovery()?;
        }
        let len = self.tracks.len();
        let current = self.selected_index();
        let current_group = current.and_then(|index| {
            match self
                .catalog
                .observe_navigation(session, &self.tracks[index].group)
            {
                GroupObservation::Complete(group) | GroupObservation::Inactive(group) => {
                    Some(group)
                }
                _ => None,
            }
        });
        for offset in 1..=len {
            let index = match (current, previous) {
                (Some(index), false) => (index + offset) % len,
                (Some(index), true) => (index + len - offset) % len,
                (None, false) => offset - 1,
                (None, true) => len - offset,
            };
            if Some(index) == current {
                continue;
            }
            if !self.tracks[index].group.active {
                continue;
            }
            if let GroupObservation::Complete(group) = self
                .catalog
                .observe_navigation(session, &self.tracks[index].group)
            {
                if current_group
                    .as_ref()
                    .is_some_and(|current| current.cells == group.cells)
                {
                    // Discovery can retain several old positions of one moving
                    // group. Different hint IDs are not different destinations.
                    continue;
                }
                if !self.can_retain(index, &group) {
                    continue;
                }
                self.tracks[index].observe(group);
                return self.select(
                    index,
                    viewport,
                    now,
                    viewport.mode() == ViewportMode::Auto,
                    session,
                );
            }
        }
        Ok(false)
    }

    pub(super) fn select(
        &mut self,
        index: usize,
        viewport: &mut ViewportController,
        now: Instant,
        pinned: bool,
        session: &mut SimulationSession,
    ) -> Result<bool, ViewportError> {
        if !self.follow(index, viewport, true, session)? {
            return Ok(false);
        }
        self.selected = Some(self.tracks[index].id);
        self.pinned = pinned;
        self.selected_at = Some(now);
        self.missing = 0;
        self.negative_generation = None;
        self.challenger = None;
        self.evidence = FocusEvidence::Active;
        self.last_validation = Some(self.tracks[index].group.generation);
        Ok(true)
    }

    pub(super) fn follow(
        &self,
        index: usize,
        viewport: &mut ViewportController,
        force: bool,
        session: &mut SimulationSession,
    ) -> Result<bool, ViewportError> {
        let track = &self.tracks[index];
        let mut candidate = viewport.clone();
        candidate.focus_group(track.focus_bounds(), force)?;
        if !force && candidate.origin() == viewport.origin() {
            return Ok(true);
        }
        let mut sample = candidate.sample_focus(session)?;
        let visible = |sample: &crate::render::ViewportSample| {
            track
                .group
                .cells
                .iter()
                .any(|&(x, y)| sample.grid.get(x, y))
        };
        if !visible(&sample) {
            // A large hollow component's bounding-box center may be empty.
            // Focus an exact member, never an unrelated visible organism.
            let Some(&(x, y)) = track.group.cells.get(track.group.cells.len() / 2) else {
                return Ok(false);
            };
            candidate.focus_group((x, y, x, y), true)?;
            sample = candidate.sample_focus(session)?;
        }
        if !visible(&sample) {
            return Ok(false);
        }
        *viewport = candidate;
        Ok(true)
    }

    /// Whether the sample shows the verified selection (or needs none). This
    /// can reject an empty automatic view, but is not a validity check for a
    /// nonempty current crop: inspection does not require organism identity.
    pub fn accepts_sample(
        &self,
        mode: ViewportMode,
        generation: u64,
        sample: &crate::render::ViewportSample,
    ) -> bool {
        if mode == ViewportMode::Manual || self.extinct {
            return true;
        }
        self.selected_index().is_none_or(|index| {
            let group = &self.tracks[index].group;
            group.generation == generation
                && group.cells.iter().any(|&(x, y)| sample.grid.get(x, y))
        })
    }

    pub fn status(&self, mode: ViewportMode) -> String {
        let index = self.selected_index().map_or(0, |index| index + 1);
        let mode = if mode == ViewportMode::Manual {
            "manual"
        } else if self.pinned {
            "auto pinned"
        } else {
            "auto largest verified active"
        };
        let evidence = if self.extinct {
            " extinct"
        } else {
            match self.evidence {
                FocusEvidence::Active => "",
                FocusEvidence::Stable => " stable",
                FocusEvidence::Absent | FocusEvidence::Reacquiring => " reacquiring",
            }
        };
        format!(
            "viewport={mode} active={index}/{}{evidence}{}",
            self.tracks.len(),
            if self.catalog.complete {
                ""
            } else {
                " (discovery incomplete)"
            }
        )
    }
}
