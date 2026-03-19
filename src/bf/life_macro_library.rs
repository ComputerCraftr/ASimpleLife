use crate::bitgrid::{BitGrid, Cell};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifeMacroKind {
    Clock,
    SplitterMerger,
    StateLatch,
    HeadTokenMover,
    BitIncrement,
    BitDecrement,
    ZeroDetector,
    OutputLatch,
    OutputBitSeedOne,
    OutputBitSeedZero,
    OutputBitTransducer,
    DivergeLatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifeMacroOrientation {
    R0,
    R90,
    R180,
    R270,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeMacroPort {
    pub name: &'static str,
    pub offset: Cell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeMacroTemplate {
    pub kind: LifeMacroKind,
    pub name: &'static str,
    pub anchor: Cell,
    pub bounds: (i64, i64),
    pub ports: &'static [LifeMacroPort],
    pub live_cells: &'static [Cell],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeMacroInstance {
    pub id: usize,
    pub kind: LifeMacroKind,
    pub name: &'static str,
    pub origin: Cell,
    pub orientation: LifeMacroOrientation,
}

const PORTS_CLOCK: &[LifeMacroPort] = &[LifeMacroPort {
    name: "tick",
    offset: (1, 0),
}];
const PORTS_SPLIT: &[LifeMacroPort] = &[
    LifeMacroPort {
        name: "in",
        offset: (0, 1),
    },
    LifeMacroPort {
        name: "out",
        offset: (2, 1),
    },
];
const PORTS_LATCH: &[LifeMacroPort] = &[
    LifeMacroPort {
        name: "set",
        offset: (0, 1),
    },
    LifeMacroPort {
        name: "q",
        offset: (2, 1),
    },
];
const PORTS_HEAD: &[LifeMacroPort] = &[
    LifeMacroPort {
        name: "left",
        offset: (0, 1),
    },
    LifeMacroPort {
        name: "right",
        offset: (2, 1),
    },
];
const PORTS_INCDEC: &[LifeMacroPort] = &[
    LifeMacroPort {
        name: "in",
        offset: (0, 1),
    },
    LifeMacroPort {
        name: "out",
        offset: (2, 1),
    },
];
const PORTS_ZERO: &[LifeMacroPort] = &[
    LifeMacroPort {
        name: "bits",
        offset: (0, 1),
    },
    LifeMacroPort {
        name: "zero",
        offset: (2, 1),
    },
];
const PORTS_OUTPUT: &[LifeMacroPort] = &[LifeMacroPort {
    name: "out",
    offset: (2, 1),
}];
const PORTS_OUTPUT_BIT: &[LifeMacroPort] = &[LifeMacroPort {
    name: "bit",
    offset: (0, 0),
}];
const PORTS_DIVERGE: &[LifeMacroPort] = &[LifeMacroPort {
    name: "loop",
    offset: (1, 1),
}];

const CLOCK_CELLS: &[Cell] = &[(0, 0), (1, 0), (2, 0), (1, 1)];
const SPLIT_CELLS: &[Cell] = &[(0, 0), (1, 0), (2, 0), (0, 2), (2, 2)];
const LATCH_CELLS: &[Cell] = &[(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (1, 2)];
const HEAD_CELLS: &[Cell] = &[(1, 0), (0, 1), (2, 1), (1, 2)];
const INC_CELLS: &[Cell] = &[(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)];
const DEC_CELLS: &[Cell] = &[(1, 0), (2, 0), (0, 1), (2, 1), (1, 2)];
const ZERO_CELLS: &[Cell] = &[(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)];
const OUTPUT_CELLS: &[Cell] = &[(0, 0), (1, 0), (0, 1), (1, 1)];
const OUTPUT_BIT_SEED_ONE_CELLS: &[Cell] = &[(0, 0), (0, 1), (1, 0)];
const OUTPUT_BIT_SEED_ZERO_CELLS: &[Cell] = &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (3, 0)];
const OUTPUT_BIT_TRANSDUCER_CELLS: &[Cell] = &[];
const DIVERGE_CELLS: &[Cell] = &[(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (1, 2)];

static LIFE_MACRO_TEMPLATES: &[LifeMacroTemplate] = &[
    LifeMacroTemplate {
        kind: LifeMacroKind::Clock,
        name: "clock",
        anchor: (0, 0),
        bounds: (3, 2),
        ports: PORTS_CLOCK,
        live_cells: CLOCK_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::SplitterMerger,
        name: "splitter_merger",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_SPLIT,
        live_cells: SPLIT_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::StateLatch,
        name: "state_latch",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_LATCH,
        live_cells: LATCH_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::HeadTokenMover,
        name: "head_token_mover",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_HEAD,
        live_cells: HEAD_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::BitIncrement,
        name: "bit_increment",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_INCDEC,
        live_cells: INC_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::BitDecrement,
        name: "bit_decrement",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_INCDEC,
        live_cells: DEC_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::ZeroDetector,
        name: "zero_detector",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_ZERO,
        live_cells: ZERO_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::OutputLatch,
        name: "output_latch",
        anchor: (0, 0),
        bounds: (2, 2),
        ports: PORTS_OUTPUT,
        live_cells: OUTPUT_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::OutputBitSeedOne,
        name: "output_bit_seed_one",
        anchor: (0, 0),
        bounds: (2, 2),
        ports: PORTS_OUTPUT_BIT,
        live_cells: OUTPUT_BIT_SEED_ONE_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::OutputBitSeedZero,
        name: "output_bit_seed_zero",
        anchor: (0, 0),
        bounds: (4, 3),
        ports: PORTS_OUTPUT_BIT,
        live_cells: OUTPUT_BIT_SEED_ZERO_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::OutputBitTransducer,
        name: "output_bit_transducer",
        anchor: (0, 0),
        bounds: (4, 3),
        ports: PORTS_OUTPUT_BIT,
        live_cells: OUTPUT_BIT_TRANSDUCER_CELLS,
    },
    LifeMacroTemplate {
        kind: LifeMacroKind::DivergeLatch,
        name: "diverge_latch",
        anchor: (0, 0),
        bounds: (3, 3),
        ports: PORTS_DIVERGE,
        live_cells: DIVERGE_CELLS,
    },
];

pub fn life_macro_templates() -> &'static [LifeMacroTemplate] {
    LIFE_MACRO_TEMPLATES
}

pub fn life_macro_template(kind: LifeMacroKind) -> &'static LifeMacroTemplate {
    LIFE_MACRO_TEMPLATES
        .iter()
        .find(|template| template.kind == kind)
        .expect("missing macro template")
}

pub fn transform_cell(cell: Cell, orientation: LifeMacroOrientation) -> Cell {
    match orientation {
        LifeMacroOrientation::R0 => cell,
        LifeMacroOrientation::R90 => (-cell.1, cell.0),
        LifeMacroOrientation::R180 => (-cell.0, -cell.1),
        LifeMacroOrientation::R270 => (cell.1, -cell.0),
    }
}

pub fn instantiate_macro_cells(instance: &LifeMacroInstance) -> Vec<Cell> {
    let template = life_macro_template(instance.kind);
    template
        .live_cells
        .iter()
        .map(|&(x, y)| {
            let (tx, ty) = transform_cell((x - template.anchor.0, y - template.anchor.1), instance.orientation);
            (instance.origin.0 + tx, instance.origin.1 + ty)
        })
        .collect()
}

pub fn macro_instance_grid(instances: &[LifeMacroInstance]) -> BitGrid {
    let mut cells = Vec::new();
    for instance in instances {
        cells.extend(instantiate_macro_cells(instance));
    }
    BitGrid::from_cells(&cells)
}
