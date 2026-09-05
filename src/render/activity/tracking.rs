use super::*;
use crate::render::{ViewportController, ViewportError, ViewportMode};

mod history;
mod selection;
use history::{PairedHistory, Track};

#[cfg(test)]
mod tests;

const MAX_TRACKS: usize = 256;
const MAX_RETAINED_CELLS: usize = 65_536;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum FocusEvidence {
    Active,
    Stable,
    Absent,
    #[default]
    Reacquiring,
}

#[derive(Debug, Default)]
pub(crate) struct ActiveFocus {
    catalog: ActivityCatalog,
    tracks: Vec<Track>,
    selected: Option<u64>,
    next_id: u64,
    pinned: bool,
    selected_at: Option<Instant>,
    challenger: Option<PairedHistory>,
    missing: usize,
    negative_generation: Option<u64>,
    evidence: FocusEvidence,
    mode: Option<ViewportMode>,
    last_observation: Option<Instant>,
    last_validation: Option<u64>,
    extinct: bool,
    initial_generation: Option<u64>,
}

impl ActiveFocus {
    pub(crate) fn discovery_pending(&self) -> bool {
        !self.catalog.finished
    }

    pub fn release_selection(&mut self) {
        self.pinned = false;
        self.selected = None;
        self.challenger = None;
        self.missing = 0;
        self.negative_generation = None;
        self.initial_generation = None;
        self.last_observation = None;
        self.last_validation = None;
    }

    pub fn refresh(
        &mut self,
        session: &mut SimulationSession,
        viewport: &mut ViewportController,
        now: Instant,
        force: bool,
    ) -> Result<bool, ViewportError> {
        let mode_changed = self.mode != Some(viewport.mode());
        if mode_changed {
            if viewport.mode() == ViewportMode::Auto {
                self.release_selection();
            }
            self.mode = Some(viewport.mode());
        }
        let discovery_due = force
            || mode_changed
            || !self
                .last_observation
                .is_some_and(|last| now.saturating_duration_since(last) < INTERVAL);
        let generation = session.hashlife_generation();
        let selected_due = self.selected.is_some() && self.last_validation != Some(generation);
        if !discovery_due && !selected_due {
            return Ok(false);
        }
        if discovery_due {
            self.last_observation = Some(now);
        }
        self.catalog.begin_observation();
        // One shared borrow covers foreground validation, discovery and paired
        // evidence. No advancement, source swap or arena mutation can interleave.
        let state = &*session;
        self.initial_generation
            .get_or_insert(state.hashlife_generation());
        self.extinct =
            state.hashlife_population_count() == Some(crate::hashlife::PopulationCount::Exact(0));
        if let Some(index) = self.selected_index() {
            // Tab pins a user-chosen occurrence/path, not a claim of unique
            // historical identity among every similar glider in a causal box.
            // Still require the exact predicted current cells and complete halo.
            let observation = if self.pinned {
                self.catalog
                    .observe_navigation(state, &self.tracks[index].group)
            } else {
                self.catalog.observe(state, &self.tracks[index].group)
            };
            self.accept_selected(index, observation, state.hashlife_generation());
            self.last_validation = Some(generation);
        }
        let contender = self.best_candidate(state.hashlife_generation(), false);
        if let Some(index) = contender
            && discovery_due
            && self.selected.is_some()
            && Some(self.tracks[index].id) != self.selected
        {
            match self.catalog.observe(state, &self.tracks[index].group) {
                GroupObservation::Complete(group) | GroupObservation::Inactive(group)
                    if self.can_retain(index, &group) =>
                {
                    self.tracks[index].observe(group);
                }
                _ => {}
            }
        }
        if discovery_due {
            if self.selected.is_none() {
                self.catalog.refresh(state, now, force);
            } else {
                self.catalog.refresh_budgeted(state, now, force);
            }
            self.merge_discovery()?;
            self.recover_selected_from_discovery(generation);
        }
        if viewport.mode() == ViewportMode::Manual {
            return Ok(true);
        }
        let generation = state.hashlife_generation();
        let best = self.best_candidate(generation, true);
        let choice = self.choose(best, generation, now);
        if let Some(index) = choice {
            self.select(index, viewport, now, false, session)?;
        } else if let Some(index) = self.selected_index()
            && self.tracks[index].group.generation == generation
            && matches!(self.evidence, FocusEvidence::Active | FocusEvidence::Stable)
        {
            self.follow(index, viewport, false, session)?;
        }
        Ok(true)
    }

    fn selected_index(&self) -> Option<usize> {
        self.selected
            .and_then(|id| self.tracks.iter().position(|track| track.id == id))
    }

    fn recover_selected_from_discovery(&mut self, generation: u64) {
        if self.evidence != FocusEvidence::Reacquiring
            || !self.catalog.complete
            || self.catalog.generation != Some(generation)
        {
            return;
        }
        let Some(index) = self.selected_index() else {
            return;
        };
        let mut matches =
            self.catalog.groups.iter().filter(|group| {
                history::shares_cells(&self.tracks[index].group.cells, &group.cells)
            });
        let Some(group) = matches.next() else {
            return;
        };
        // Partial discovery cannot establish one-to-one membership. A complete
        // observation can recover a changed component, but never choose one
        // branch of a split or transfer a selected ID across an ambiguous merge.
        if matches.next().is_some()
            || self
                .tracks
                .iter()
                .filter(|track| history::shares_cells(&track.group.cells, &group.cells))
                .count()
                != 1
            || !self.can_retain(index, group)
        {
            return;
        }
        let group = group.clone();
        self.accept_selected(index, GroupObservation::Complete(group), generation);
    }

    fn accept_selected(&mut self, index: usize, observation: GroupObservation, generation: u64) {
        match observation {
            GroupObservation::Complete(group) => {
                if !self.can_retain(index, &group) {
                    self.evidence = FocusEvidence::Reacquiring;
                    return;
                }
                self.tracks[index].observe(group);
                self.evidence = FocusEvidence::Active;
                self.missing = 0;
                self.negative_generation = None;
            }
            GroupObservation::Inactive(group) => {
                if !self.can_retain(index, &group) {
                    self.evidence = FocusEvidence::Reacquiring;
                    return;
                }
                self.tracks[index].observe(group);
                self.evidence = FocusEvidence::Stable;
                self.record_negative(generation);
            }
            GroupObservation::Absent => {
                self.evidence = FocusEvidence::Absent;
                self.record_negative(generation);
            }
            GroupObservation::Incomplete | GroupObservation::Failed => {
                self.evidence = FocusEvidence::Reacquiring;
            }
        }
    }

    fn record_negative(&mut self, generation: u64) {
        if self.negative_generation != Some(generation) {
            self.missing = (self.missing + 1).min(2);
            self.negative_generation = Some(generation);
        }
    }

    fn can_retain(&self, index: usize, group: &ActiveGroup) -> bool {
        let retained: usize = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, track)| track.group.cells.capacity())
            .sum();
        retained + group.cells.capacity() <= MAX_RETAINED_CELLS
    }

    fn best_candidate(&self, generation: u64, current_only: bool) -> Option<usize> {
        if let Some(history) = &self.challenger
            && let Some(index) = self.tracks.iter().position(|track| {
                track.id == history.id
                    && track.group.active
                    && (!current_only || track.group.generation == generation)
            })
        {
            return Some(index);
        }
        self.tracks
            .iter()
            .enumerate()
            .filter(|(_, track)| {
                Some(track.id) != self.selected
                    && track.group.active
                    && (!current_only || track.group.generation == generation)
            })
            .max_by(|(_, a), (_, b)| {
                a.group
                    .population
                    .cmp(&b.group.population)
                    .then_with(|| b.anchor.cmp(&a.anchor))
            })
            .map(|(index, _)| index)
    }

    fn merge_discovery(&mut self) -> Result<(), ViewportError> {
        for group in &self.catalog.groups {
            // Membership continuity must be one-to-one. Splits and merges do
            // not steal a selected identity, even when their bounds overlap.
            let mut matches = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, track)| history::shares_cells(&track.group.cells, &group.cells));
            let first = matches.next().map(|(index, _)| index);
            let unique = matches.next().is_none();
            if let Some(index) = first {
                if unique
                    && self
                        .catalog
                        .groups
                        .iter()
                        .filter(|other| {
                            history::shares_cells(&self.tracks[index].group.cells, &other.cells)
                        })
                        .count()
                        == 1
                    && Some(self.tracks[index].id) != self.selected
                    && self.can_retain(index, group)
                {
                    self.tracks[index].observe(group.clone());
                }
                continue;
            }
            let retained: usize = self
                .tracks
                .iter()
                .map(|track| track.group.cells.capacity())
                .sum();
            if self.tracks.len() == MAX_TRACKS
                || retained + group.cells.capacity() > MAX_RETAINED_CELLS
            {
                // Forget an old unselected hint, never selected evidence.
                if let Some(index) = self
                    .tracks
                    .iter()
                    .enumerate()
                    .filter(|(_, track)| Some(track.id) != self.selected)
                    .min_by_key(|(_, track)| track.group.generation)
                    .map(|(i, _)| i)
                {
                    self.tracks.remove(index);
                }
                continue;
            }
            self.next_id = self
                .next_id
                .checked_add(1)
                .ok_or(ViewportError::RevisionExhausted)?;
            self.tracks.push(Track::new(self.next_id, group.clone()));
        }
        self.tracks.sort_by_key(|track| (track.anchor, track.id));
        Ok(())
    }
}
