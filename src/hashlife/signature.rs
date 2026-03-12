use crate::bitgrid::{Cell, Coord};
use crate::normalize::NormalizedGridSignature;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HashLifeCheckpointSignature {
    pub width: Coord,
    pub height: Coord,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HashLifeCheckpointKey {
    pub width: Coord,
    pub height: Coord,
    pub population: u64,
    pub cell_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashLifeCheckpoint {
    pub generation: u64,
    pub origin: Cell,
    pub signature: HashLifeCheckpointSignature,
    pub population: u64,
    pub bounds: (Coord, Coord, Coord, Coord),
    pub bounds_span: Coord,
}

impl HashLifeCheckpointSignature {
    pub fn key(&self) -> HashLifeCheckpointKey {
        let mut hasher = DefaultHasher::new();
        self.width.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.cells.hash(&mut hasher);
        HashLifeCheckpointKey {
            width: self.width,
            height: self.height,
            population: u64::try_from(self.cells.len())
                .expect("checkpoint population exceeded u64"),
            cell_hash: hasher.finish(),
        }
    }

    pub fn matches_normalized(&self, normalized: &NormalizedGridSignature) -> bool {
        self.width == normalized.width
            && self.height == normalized.height
            && self.cells == normalized.cells
    }
}

impl From<&NormalizedGridSignature> for HashLifeCheckpointSignature {
    fn from(normalized: &NormalizedGridSignature) -> Self {
        Self {
            width: normalized.width,
            height: normalized.height,
            cells: normalized.cells.clone(),
        }
    }
}
