use super::*;

const DIRECT_GENERATIONS: u64 = 16;
const MAX_PREDICTION_CELLS: usize = 1024;
const MAX_PREDICTION_EVIDENCE: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GroupObservation {
    Complete(ActiveGroup),
    Inactive(ActiveGroup),
    Absent,
    Incomplete,
    Failed,
}

enum Prediction {
    Cells(Vec<Cell>),
    Absent,
    Failed,
}

impl ActivityCatalog {
    pub fn observe(
        &mut self,
        session: &SimulationSession,
        previous: &ActiveGroup,
    ) -> GroupObservation {
        self.observe_candidate(session, previous, true)
    }

    /// Explicit navigation selects a verified current component; it does not
    /// assert that this is the unique historical continuation of an old track.
    pub fn observe_navigation(
        &mut self,
        session: &SimulationSession,
        previous: &ActiveGroup,
    ) -> GroupObservation {
        self.observe_candidate(session, previous, false)
    }

    fn observe_candidate(
        &mut self,
        session: &SimulationSession,
        previous: &ActiveGroup,
        require_continuity: bool,
    ) -> GroupObservation {
        let generation = session.hashlife_generation();
        if generation < previous.generation || previous.cells.is_empty() {
            return GroupObservation::Failed;
        }
        let predicted = match predict(&previous.cells, generation - previous.generation) {
            Prediction::Cells(cells) => cells,
            Prediction::Absent => {
                return match self.prove_absent(session, previous, generation) {
                    Some(true) => GroupObservation::Absent,
                    Some(false) | None => GroupObservation::Incomplete,
                };
            }
            Prediction::Failed => return GroupObservation::Failed,
        };
        if !is_component(&predicted) {
            return GroupObservation::Failed;
        }
        match self.verify_component(session, previous, predicted, generation, require_continuity) {
            Ok(Some(group)) if group.active => GroupObservation::Complete(group),
            Ok(Some(group)) => GroupObservation::Inactive(group),
            Ok(None) => GroupObservation::Absent,
            Err(true) => GroupObservation::Incomplete,
            Err(false) => GroupObservation::Failed,
        }
    }

    fn verify_component(
        &mut self,
        session: &SimulationSession,
        previous: &ActiveGroup,
        predicted: Vec<Cell>,
        generation: u64,
        require_continuity: bool,
    ) -> Result<Option<ActiveGroup>, bool> {
        if session.viewport_root_bounds().is_none() {
            return Err(false);
        }
        let bounds = cell_bounds(&predicted).ok_or(false)?;
        let envelope = expand_bounds(bounds, 2).ok_or(false)?;
        let (tx0, ty0, tx1, ty1) = (
            envelope.0.div_euclid(8),
            envelope.1.div_euclid(8),
            envelope.2.div_euclid(8),
            envelope.3.div_euclid(8),
        );
        let tile_count = (i128::from(tx1) - i128::from(tx0) + 1)
            .checked_mul(i128::from(ty1) - i128::from(ty0) + 1)
            .ok_or(false)?;
        if tile_count > i128::try_from(MAX_TILES).unwrap_or(i128::MAX) {
            return Err(false);
        }
        self.cell_workspace.clear();
        // One inspection already returns all nine exact current chunks. Use
        // them once instead of traversing the same deep DAG nine times.
        for ty in (ty0..=ty1).step_by(3) {
            for tx in (tx0..=tx1).step_by(3) {
                if self.evaluations == MAX_EVALUATIONS {
                    return Err(true);
                }
                let mut remaining = MAX_VISITS.saturating_sub(self.visits);
                if remaining == 0 {
                    return Err(true);
                }
                let before = remaining;
                let center = checked_cell_offset((tx, ty), 1, 1).ok_or(false)?;
                let chunks = session.inspect_viewport_neighborhood(center, &mut remaining);
                self.visits += before - remaining;
                self.evaluations += 1;
                let Some(chunks) = chunks else {
                    return Err(remaining == 0);
                };
                for dy in 0..3 {
                    for dx in 0..3 {
                        let tile = checked_cell_offset((tx, ty), dx, dy).ok_or(false)?;
                        if tile.0 <= tx1 && tile.1 <= ty1 {
                            let index = usize::try_from(dy * 3 + dx).map_err(|_| false)?;
                            append_tile_cells(tile, chunks[index], &mut self.cell_workspace);
                        }
                    }
                }
            }
        }
        self.cell_workspace.retain(|&(x, y)| {
            x >= envelope.0 && x <= envelope.2 && y >= envelope.1 && y <= envelope.3
        });
        self.cell_workspace.sort_unstable();
        if predicted.is_empty() {
            return Ok(None);
        }
        if self.cell_workspace != predicted {
            return Err(true);
        }
        if require_continuity
            // Exact retained membership supports local continuity (including
            // oscillator phases). A displaced occurrence still needs the full
            // uniqueness check; unrelated ash in a large causal box must not
            // invalidate a blinker that is still at its verified location.
            && !predicted.iter().any(|cell| previous.cells.binary_search(cell).is_ok())
            && !self.prove_unique_causal_region(session, previous, envelope, generation)?
        {
            return Err(true);
        }
        let future = evolve_cells(&predicted);
        let active = future != predicted;
        let active_tile = first_difference(&predicted, &future)
            .map(|(x, y)| (x.div_euclid(8), y.div_euclid(8)))
            .unwrap_or((bounds.0.div_euclid(8), bounds.1.div_euclid(8)));
        Ok(Some(ActiveGroup {
            population: u32::try_from(predicted.len()).map_err(|_| false)?,
            cells: predicted,
            active,
            bounds,
            active_tile,
            generation,
        }))
    }

    fn prove_absent(
        &mut self,
        session: &SimulationSession,
        previous: &ActiveGroup,
        generation: u64,
    ) -> Option<bool> {
        session.viewport_root_bounds()?;
        let causal = causal_bounds(
            previous.bounds,
            generation.checked_sub(previous.generation)?,
        )?;
        self.region_empty(session, causal)
    }

    fn prove_unique_causal_region(
        &mut self,
        session: &SimulationSession,
        previous: &ActiveGroup,
        inspected: Bounds,
        generation: u64,
    ) -> Result<bool, bool> {
        let delta = generation.saturating_sub(previous.generation);
        let causal = causal_bounds(previous.bounds, delta).ok_or(true)?;
        let cut = intersect_i128(causal, tuple_i128(inspected)).ok_or(true)?;
        let rectangles = [
            (causal.0, causal.1, causal.2, cut.1 - 1),
            (causal.0, cut.3 + 1, causal.2, causal.3),
            (causal.0, cut.1, cut.0 - 1, cut.3),
            (cut.2 + 1, cut.1, causal.2, cut.3),
        ];
        for rectangle in rectangles {
            if rectangle.0 <= rectangle.2 && rectangle.1 <= rectangle.3 {
                match self.region_empty(session, rectangle) {
                    Some(true) => {}
                    Some(false) => return Ok(false),
                    None => return Err(true),
                }
            }
        }
        Ok(true)
    }

    fn region_empty(
        &mut self,
        session: &SimulationSession,
        bounds: (i128, i128, i128, i128),
    ) -> Option<bool> {
        let mut remaining = MAX_VISITS.saturating_sub(self.visits);
        if remaining == 0 {
            return None;
        }
        let before = remaining;
        let occupied = session.inspect_viewport_region(bounds, &mut remaining);
        self.visits += before - remaining;
        occupied.map(|value| !value)
    }
}

fn predict(initial: &[Cell], generations: u64) -> Prediction {
    if initial.len() > MAX_PREDICTION_CELLS {
        return Prediction::Failed;
    }
    if generations == 0 {
        return Prediction::Cells(initial.to_vec());
    }
    let mut states = vec![initial.to_vec()];
    let mut evidence = initial.len();
    for step in 1..=generations.min(DIRECT_GENERATIONS) {
        let cells = evolve_cells(states.last().unwrap_or(&states[0]));
        if cells.is_empty() {
            return Prediction::Absent;
        }
        if cells.len() > MAX_PREDICTION_CELLS {
            return Prediction::Failed;
        }
        evidence = match evidence.checked_add(cells.len()) {
            Some(total) if total <= MAX_PREDICTION_EVIDENCE => total,
            _ => return Prediction::Failed,
        };
        states.push(cells);
        if step == generations {
            return Prediction::Cells(states.pop().unwrap_or_default());
        }
        for start in 0..usize::try_from(step).unwrap_or(0) {
            let Some(current) = states.last() else {
                return Prediction::Failed;
            };
            if normalized_equal(&states[start], current) {
                let period = usize::try_from(step).unwrap_or(0) - start;
                let Ok(start_generation) = u64::try_from(start) else {
                    return Prediction::Failed;
                };
                let Ok(period_generations) = u64::try_from(period) else {
                    return Prediction::Failed;
                };
                let offset = generations.saturating_sub(start_generation);
                let cycles = offset / period_generations;
                let remainder = usize::try_from(offset % period_generations).unwrap_or(0);
                let Some(a) = cell_bounds(&states[start]) else {
                    return Prediction::Absent;
                };
                let Some(b) = cell_bounds(states.last().unwrap_or(&states[start])) else {
                    return Prediction::Absent;
                };
                let dx = i128::from(b.0) - i128::from(a.0);
                let dy = i128::from(b.1) - i128::from(a.1);
                return translate(
                    &states[start + remainder],
                    dx * i128::from(cycles),
                    dy * i128::from(cycles),
                );
            }
        }
    }
    Prediction::Failed
}

fn translate(cells: &[Cell], dx: i128, dy: i128) -> Prediction {
    let mut translated = Vec::with_capacity(cells.len());
    for &(x, y) in cells {
        let Ok(x) = Coord::try_from(i128::from(x) + dx) else {
            return Prediction::Failed;
        };
        let Ok(y) = Coord::try_from(i128::from(y) + dy) else {
            return Prediction::Failed;
        };
        translated.push((x, y));
    }
    Prediction::Cells(translated)
}

fn normalized_equal(a: &[Cell], b: &[Cell]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let (Some(ab), Some(bb)) = (cell_bounds(a), cell_bounds(b)) else {
        return a.is_empty() && b.is_empty();
    };
    a.iter().zip(b).all(|(&(ax, ay), &(bx, by))| {
        i128::from(ax) - i128::from(ab.0) == i128::from(bx) - i128::from(bb.0)
            && i128::from(ay) - i128::from(ab.1) == i128::from(by) - i128::from(bb.1)
    })
}

fn is_component(cells: &[Cell]) -> bool {
    !cells.is_empty() && connected_cells(cells, 0).len() == cells.len()
}

fn connected_cells(cells: &[Cell], seed: usize) -> Vec<Cell> {
    let mut seen = vec![false; cells.len()];
    let mut queue = vec![seed];
    let mut result = Vec::new();
    seen[seed] = true;
    while let Some(index) = queue.pop() {
        let cell = cells[index];
        result.push(cell);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let Some(neighbor) = checked_cell_offset(cell, dx, dy) else {
                    continue;
                };
                if let Ok(found) = cells.binary_search(&neighbor)
                    && !seen[found]
                {
                    seen[found] = true;
                    queue.push(found);
                }
            }
        }
    }
    result.sort_unstable();
    result
}

fn causal_bounds(bounds: Bounds, generations: u64) -> Option<(i128, i128, i128, i128)> {
    let distance = i128::from(generations);
    Some((
        i128::from(bounds.0).checked_sub(distance)?,
        i128::from(bounds.1).checked_sub(distance)?,
        i128::from(bounds.2).checked_add(distance)?,
        i128::from(bounds.3).checked_add(distance)?,
    ))
}

fn tuple_i128(bounds: Bounds) -> (i128, i128, i128, i128) {
    (
        i128::from(bounds.0),
        i128::from(bounds.1),
        i128::from(bounds.2),
        i128::from(bounds.3),
    )
}

fn intersect_i128(
    a: (i128, i128, i128, i128),
    b: (i128, i128, i128, i128),
) -> Option<(i128, i128, i128, i128)> {
    let result = (a.0.max(b.0), a.1.max(b.1), a.2.min(b.2), a.3.min(b.3));
    (result.0 <= result.2 && result.1 <= result.3).then_some(result)
}

#[cfg(test)]
mod tests;
