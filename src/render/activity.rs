//! Bounded, presentation-only discovery over immutable HashLife state.
use crate::bitgrid::{Bounds, Cell, Coord};
use crate::engine::SimulationSession;
use crate::probe_table::{ProbeKey, ProbeMode, ProbeTable};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

mod observation;
#[cfg(test)]
mod tests;
mod tracking;
pub(crate) use observation::GroupObservation;
pub(crate) use tracking::ActiveFocus;

pub(crate) const MAX_VISITS: usize = 4096;
const DISCOVERY_VISITS: usize = MAX_VISITS / 2;
pub(crate) const MAX_TILES: usize = 4096;
pub(crate) const MAX_EVALUATIONS: usize = 256;
const MAX_CATALOG_CELLS: usize = 8192;
const INTERVAL: Duration = Duration::from_millis(250);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ActiveGroup {
    pub cells: Vec<Cell>,
    pub active: bool,
    pub bounds: Bounds,
    pub population: u32,
    pub active_tile: Cell,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    bits: u64,
    known: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileKey(Cell);

impl ProbeKey for TileKey {
    fn fingerprint(&self) -> u64 {
        crate::hashing::hash_chunk_coord_key(self.0.0, self.0.1)
    }
}

#[derive(Debug)]
pub(crate) struct ActivityCatalog {
    regions: Vec<Bounds>,
    pending: Vec<Cell>,
    tiles: ProbeTable<TileKey, Tile>,
    keys: Vec<Cell>,
    component_queue: Vec<usize>,
    component_seen: Vec<bool>,
    cell_workspace: Vec<Cell>,
    groups_dirty: bool,
    pub groups: Vec<ActiveGroup>,
    pub complete: bool,
    pub(crate) finished: bool,
    truncated: bool,
    generation: Option<u64>,
    last_scan: Option<Instant>,
    pub visits: usize,
    pub evaluations: usize,
}

impl Default for ActivityCatalog {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            pending: Vec::new(),
            tiles: ProbeTable::new(ProbeMode::Scratch),
            keys: Vec::new(),
            component_queue: Vec::new(),
            component_seen: Vec::new(),
            cell_workspace: Vec::new(),
            groups_dirty: false,
            groups: Vec::new(),
            complete: false,
            finished: false,
            truncated: false,
            generation: None,
            last_scan: None,
            visits: 0,
            evaluations: 0,
        }
    }
}

impl ActivityCatalog {
    pub fn begin_observation(&mut self) {
        self.visits = 0;
        self.evaluations = 0;
    }

    pub fn refresh(&mut self, session: &SimulationSession, now: Instant, force: bool) -> bool {
        self.begin_observation();
        self.refresh_budgeted(session, now, force)
    }

    pub fn refresh_budgeted(
        &mut self,
        session: &SimulationSession,
        now: Instant,
        force: bool,
    ) -> bool {
        let generation = session.hashlife_generation();
        if self.finished && self.generation == Some(generation) {
            return false;
        }
        if !force
            && self
                .last_scan
                .is_some_and(|last| now.duration_since(last) < INTERVAL)
        {
            return false;
        }
        self.last_scan = Some(now);
        if self.generation != Some(generation) {
            self.tiles.clear();
            self.keys.clear();
            self.groups.clear();
            if self.generation.is_some() && (!self.regions.is_empty() || !self.pending.is_empty()) {
                self.truncated = true;
            }
            self.generation = Some(generation);
            self.complete = false;
            self.finished = false;
        }
        if self.regions.is_empty() && self.pending.is_empty() {
            self.start_scan(session, generation);
        }
        let allowance = DISCOVERY_VISITS.min(MAX_VISITS.saturating_sub(self.visits));
        let mut remaining = allowance;
        let evaluation_limit = MAX_EVALUATIONS.saturating_sub(4);
        while remaining != 0 && self.evaluations < evaluation_limit && self.tiles.len() < MAX_TILES
        {
            if let Some(coord) = self.pending.pop() {
                if self
                    .tiles
                    .get(&TileKey(coord))
                    .is_some_and(|tile| tile.known)
                {
                    continue;
                }
                let Some(chunks) = session.inspect_viewport_neighborhood(coord, &mut remaining)
                else {
                    self.pending.push(coord);
                    break;
                };
                self.evaluations += 1;
                self.store_neighborhood(coord, chunks);
                self.groups_dirty = true;
                continue;
            }
            let Some(region) = self.regions.pop() else {
                break;
            };
            let (x0, y0, x1, y1) = region;
            let cells = (
                i128::from(x0) * 8,
                i128::from(y0) * 8,
                i128::from(x1) * 8 + 7,
                i128::from(y1) * 8 + 7,
            );
            match session.inspect_viewport_region(cells, &mut remaining) {
                None => {
                    if x0 != x1 || y0 != y1 {
                        split_region(&mut self.regions, region);
                    } else {
                        self.regions.push(region);
                    }
                    break;
                }
                Some(false) => continue,
                Some(true) if x0 != x1 || y0 != y1 => split_region(&mut self.regions, region),
                Some(true) => self.schedule_halo((x0, y0)),
            }
        }
        self.visits += allowance - remaining;
        self.finished = self.regions.is_empty() && self.pending.is_empty();
        self.complete = self.finished && !self.truncated && self.generation == Some(generation);
        if self.groups_dirty {
            self.rebuild_groups(generation);
            self.groups_dirty = false;
        }
        if self.tiles.len() == MAX_TILES {
            self.tiles.clear();
            self.keys.clear();
            self.truncated = true;
            self.complete = false;
        }
        true
    }

    fn start_scan(&mut self, session: &SimulationSession, generation: u64) {
        self.tiles.clear();
        self.keys.clear();
        self.groups.clear();
        self.generation = Some(generation);
        self.complete = false;
        self.finished = false;
        self.truncated = false;
        self.groups_dirty = false;
        if let Some((x0, y0, x1, y1)) = session.viewport_root_bounds() {
            self.regions.push((
                x0.div_euclid(8),
                y0.div_euclid(8),
                x1.div_euclid(8),
                y1.div_euclid(8),
            ));
        }
    }

    fn schedule_halo(&mut self, center: Cell) {
        if !self.tiles.contains_key(&TileKey(center)) && self.tiles.len() < MAX_TILES {
            self.tiles.insert(
                TileKey(center),
                Tile {
                    bits: 0,
                    known: false,
                },
            );
            self.keys.push(center);
            self.pending.push(center);
        }
    }

    fn store_neighborhood(&mut self, center: Cell, chunks: [u64; 9]) {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let Some(coord) = checked_tile_offset(center, dx, dy) else {
                    continue;
                };
                let index = usize::try_from((dy + 1) * 3 + dx + 1).unwrap_or(4);
                if !self.tiles.contains_key(&TileKey(coord)) {
                    if self.tiles.len() == MAX_TILES {
                        self.truncated = true;
                        continue;
                    }
                    self.keys.push(coord);
                }
                self.tiles.insert(
                    TileKey(coord),
                    Tile {
                        bits: chunks[index],
                        known: true,
                    },
                );
            }
        }
        self.groups_dirty = true;
    }

    fn rebuild_groups(&mut self, generation: u64) {
        self.cell_workspace.clear();
        for &coord in &self.keys {
            let Some(tile) = self.tiles.get(&TileKey(coord)).filter(|tile| tile.known) else {
                continue;
            };
            append_tile_cells(coord, tile.bits, &mut self.cell_workspace);
            if self.cell_workspace.len() > MAX_CATALOG_CELLS {
                self.cell_workspace.clear();
                self.groups.clear();
                self.truncated = true;
                self.complete = false;
                return;
            }
        }
        self.cell_workspace.sort_unstable();
        self.component_seen.clear();
        self.component_seen.resize(self.cell_workspace.len(), false);
        self.groups.clear();
        for start in 0..self.cell_workspace.len() {
            if self.component_seen[start] {
                continue;
            }
            self.component_seen[start] = true;
            self.component_queue.clear();
            self.component_queue.push(start);
            let mut component = Vec::new();
            while let Some(index) = self.component_queue.pop() {
                let cell = self.cell_workspace[index];
                component.push(cell);
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let Some(neighbor) = checked_cell_offset(cell, dx, dy) else {
                            continue;
                        };
                        if let Ok(found) = self.cell_workspace.binary_search(&neighbor)
                            && !self.component_seen[found]
                        {
                            self.component_seen[found] = true;
                            self.component_queue.push(found);
                        }
                    }
                }
            }
            component.sort_unstable();
            if let Some(group) = build_complete_group(&self.tiles, component, generation)
                && group.active
            {
                self.groups.push(group);
            }
        }
        self.groups
            .sort_by_key(|group| (group.bounds.0, group.bounds.1));
    }
}

fn build_complete_group(
    tiles: &ProbeTable<TileKey, Tile>,
    cells: Vec<Cell>,
    generation: u64,
) -> Option<ActiveGroup> {
    let bounds = cell_bounds(&cells)?;
    let envelope = expand_bounds(bounds, 2)?;
    let tile_count = (i128::from(envelope.2.div_euclid(8)) - i128::from(envelope.0.div_euclid(8))
        + 1)
    .checked_mul(i128::from(envelope.3.div_euclid(8)) - i128::from(envelope.1.div_euclid(8)) + 1)?;
    if tile_count > i128::try_from(MAX_TILES).ok()? {
        return None;
    }
    for ty in envelope.1.div_euclid(8)..=envelope.3.div_euclid(8) {
        for tx in envelope.0.div_euclid(8)..=envelope.2.div_euclid(8) {
            if !tiles.get(&TileKey((tx, ty))).is_some_and(|tile| tile.known) {
                return None;
            }
        }
    }
    let future = evolve_cells(&cells);
    let changed =
        first_difference(&cells, &future).map(|(x, y)| (x.div_euclid(8), y.div_euclid(8)));
    Some(ActiveGroup {
        population: u32::try_from(cells.len()).ok()?,
        active: changed.is_some(),
        cells,
        bounds,
        active_tile: changed.unwrap_or((bounds.0.div_euclid(8), bounds.1.div_euclid(8))),
        generation,
    })
}

pub(super) fn cell_bounds(cells: &[Cell]) -> Option<Bounds> {
    let &(mut x0, mut y0) = cells.first()?;
    let (mut x1, mut y1) = (x0, y0);
    for &(x, y) in &cells[1..] {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    Some((x0, y0, x1, y1))
}

pub(super) fn expand_bounds((x0, y0, x1, y1): Bounds, n: Coord) -> Option<Bounds> {
    Some((
        x0.checked_sub(n)?,
        y0.checked_sub(n)?,
        x1.checked_add(n)?,
        y1.checked_add(n)?,
    ))
}

pub(super) fn append_tile_cells((tx, ty): Cell, mut bits: u64, output: &mut Vec<Cell>) {
    while bits != 0 {
        let bit = bits.trailing_zeros();
        bits &= bits - 1;
        output.push((tx * 8 + Coord::from(bit % 8), ty * 8 + Coord::from(bit / 8)));
    }
}

fn checked_tile_offset((x, y): Cell, dx: Coord, dy: Coord) -> Option<Cell> {
    let result = (x.checked_add(dx)?, y.checked_add(dy)?);
    (result.0 >= Coord::MIN.div_euclid(8)
        && result.0 <= Coord::MAX.div_euclid(8)
        && result.1 >= Coord::MIN.div_euclid(8)
        && result.1 <= Coord::MAX.div_euclid(8))
    .then_some(result)
}

pub(super) fn checked_cell_offset((x, y): Cell, dx: Coord, dy: Coord) -> Option<Cell> {
    Some((x.checked_add(dx)?, y.checked_add(dy)?))
}

pub(super) fn evolve_cells(cells: &[Cell]) -> Vec<Cell> {
    let mut counts = BTreeMap::<Cell, u8>::new();
    for &cell in cells {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if let Some(neighbor) = checked_cell_offset(cell, dx, dy) {
                    *counts.entry(neighbor).or_default() += 1;
                }
            }
        }
    }
    counts
        .into_iter()
        .filter_map(|(cell, count)| {
            (count == 3 || (count == 2 && cells.binary_search(&cell).is_ok())).then_some(cell)
        })
        .collect()
}

pub(super) fn first_difference(a: &[Cell], b: &[Cell]) -> Option<Cell> {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => return Some(a[i]),
            std::cmp::Ordering::Greater => return Some(b[j]),
        }
    }
    a.get(i).copied().or_else(|| b.get(j).copied())
}

fn split_region(regions: &mut Vec<Bounds>, (x0, y0, x1, y1): Bounds) {
    if i128::from(x1) - i128::from(x0) >= i128::from(y1) - i128::from(y0) {
        let mid = Coord::try_from(i128::from(x0) + (i128::from(x1) - i128::from(x0)) / 2).ok();
        if let Some(mid) = mid {
            regions.push((mid + 1, y0, x1, y1));
            regions.push((x0, y0, mid, y1));
        }
    } else {
        let mid = Coord::try_from(i128::from(y0) + (i128::from(y1) - i128::from(y0)) / 2).ok();
        if let Some(mid) = mid {
            regions.push((x0, mid + 1, x1, y1));
            regions.push((x0, y0, x1, mid));
        }
    }
}
