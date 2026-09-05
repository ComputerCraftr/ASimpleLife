use super::*;

#[derive(Debug)]
pub(super) struct Track {
    pub id: u64,
    pub anchor: Cell,
    pub group: ActiveGroup,
    bounds: [Bounds; 4],
    samples: usize,
    next: usize,
}

impl Track {
    pub fn new(id: u64, group: ActiveGroup) -> Self {
        Self {
            id,
            anchor: (group.bounds.0, group.bounds.1),
            bounds: [group.bounds; 4],
            samples: 1,
            next: 1,
            group,
        }
    }

    pub fn observe(&mut self, group: ActiveGroup) {
        if group.generation < self.group.generation {
            return;
        }
        if normalized_equal(&self.group, &group) {
            let dx = i128::from(group.bounds.0) - i128::from(self.group.bounds.0);
            let dy = i128::from(group.bounds.1) - i128::from(self.group.bounds.1);
            for bounds in &mut self.bounds {
                *bounds = translated(*bounds, dx, dy).unwrap_or(group.bounds);
            }
        } else if !overlaps(self.group.bounds, group.bounds) {
            self.bounds.fill(group.bounds);
        }
        if group.generation != self.group.generation {
            self.bounds[self.next] = group.bounds;
            self.next = (self.next + 1) % 4;
            self.samples = (self.samples + 1).min(4);
        }
        self.group = group;
    }

    pub fn focus_bounds(&self) -> Bounds {
        self.bounds[..self.samples]
            .iter()
            .fold(self.group.bounds, |a, b| {
                (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
            })
    }
}

fn translated(b: Bounds, dx: i128, dy: i128) -> Option<Bounds> {
    Some((
        Coord::try_from(i128::from(b.0) + dx).ok()?,
        Coord::try_from(i128::from(b.1) + dy).ok()?,
        Coord::try_from(i128::from(b.2) + dx).ok()?,
        Coord::try_from(i128::from(b.3) + dy).ok()?,
    ))
}

fn normalized_equal(a: &ActiveGroup, b: &ActiveGroup) -> bool {
    a.cells.len() == b.cells.len()
        && a.cells.iter().zip(&b.cells).all(|(a_cell, b_cell)| {
            i128::from(a_cell.0) - i128::from(a.bounds.0)
                == i128::from(b_cell.0) - i128::from(b.bounds.0)
                && i128::from(a_cell.1) - i128::from(a.bounds.1)
                    == i128::from(b_cell.1) - i128::from(b.bounds.1)
        })
}

fn overlaps(a: Bounds, b: Bounds) -> bool {
    a.0 <= b.2 && b.0 <= a.2 && a.1 <= b.3 && b.1 <= a.3
}

pub(super) fn shares_cells(a: &[Cell], b: &[Cell]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

#[derive(Clone, Debug)]
pub(super) struct PairedHistory {
    pub id: u64,
    pairs: [(u32, u32); 4],
    samples: usize,
    next: usize,
    last_generation: Option<u64>,
    qualifying: usize,
}

impl PairedHistory {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            pairs: [(0, 0); 4],
            samples: 0,
            next: 0,
            last_generation: None,
            qualifying: 0,
        }
    }

    pub fn observe(&mut self, selected: &ActiveGroup, challenger: &ActiveGroup) -> bool {
        if selected.generation != challenger.generation
            || self.last_generation == Some(selected.generation)
        {
            return false;
        }
        self.last_generation = Some(selected.generation);
        self.pairs[self.next] = (selected.population, challenger.population);
        self.next = (self.next + 1) % 4;
        self.samples = (self.samples + 1).min(4);
        if self.samples != 4 {
            return false;
        }
        let (current, candidate) = self.pairs.iter().fold((0_u64, 0_u64), |(a, b), &(c, d)| {
            (a + u64::from(c), b + u64::from(d))
        });
        self.qualifying = if candidate * 4 > current * 5 {
            (self.qualifying + 1).min(3)
        } else {
            0
        };
        self.qualifying == 3
    }
}
