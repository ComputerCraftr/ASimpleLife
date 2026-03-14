use std::fmt;
use std::hash::{Hash, Hasher};

use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::hashing::hash_normalized_grid_signature;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedGridSignature {
    pub width: Coord,
    pub height: Coord,
    pub cells: Vec<Cell>,
}

impl fmt::Display for NormalizedGridSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} {:?}", self.width, self.height, self.cells)
    }
}

impl NormalizedGridSignature {
    pub fn fingerprint(&self) -> u64 {
        hash_normalized_grid_signature(self.width, self.height, &self.cells)
    }
}

impl Hash for NormalizedGridSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.fingerprint());
    }
}

pub fn normalize(grid: &BitGrid) -> (NormalizedGridSignature, Cell) {
    let Some((min_x, min_y, max_x, max_y)) = grid.bounds() else {
        return (
            NormalizedGridSignature {
                width: 0,
                height: 0,
                cells: Vec::new(),
            },
            (0, 0),
        );
    };

    let mut cells = grid.live_cells();
    for (x, y) in &mut cells {
        *x -= min_x;
        *y -= min_y;
    }
    cells.sort_unstable();

    (
        NormalizedGridSignature {
            width: max_x - min_x + 1,
            height: max_y - min_y + 1,
            cells,
        },
        (min_x, min_y),
    )
}
