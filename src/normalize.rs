use std::fmt;

use crate::bitgrid::BitGrid;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedGridSignature {
    pub width: i32,
    pub height: i32,
    pub cells: Vec<(i32, i32)>,
}

impl fmt::Display for NormalizedGridSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} {:?}", self.width, self.height, self.cells)
    }
}

pub fn normalize(grid: &BitGrid) -> (NormalizedGridSignature, (i32, i32)) {
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

    let mut cells = grid
        .live_cells()
        .into_iter()
        .map(|(x, y)| (x - min_x, y - min_y))
        .collect::<Vec<_>>();
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
