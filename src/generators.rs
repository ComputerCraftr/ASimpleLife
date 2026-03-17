use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::hashing::{SPLITMIX64_GAMMA, mix_seed};
use crate::life_grid_format;

pub fn pattern_from_file(path: &str) -> Option<BitGrid> {
    let s = std::fs::read_to_string(path).ok()?;
    life_grid_format::deserialize(&s).ok()
}

pub fn pattern_by_name(name: &str) -> Option<BitGrid> {
    let cells = match name {
        "block" => vec![(0, 0), (1, 0), (0, 1), (1, 1)],
        "blinker" => vec![(0, 0), (1, 0), (2, 0)],
        "glider" => vec![(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
        "pulsar" => vec![
            (2, 0),
            (3, 0),
            (4, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (0, 2),
            (5, 2),
            (7, 2),
            (12, 2),
            (0, 3),
            (5, 3),
            (7, 3),
            (12, 3),
            (0, 4),
            (5, 4),
            (7, 4),
            (12, 4),
            (2, 5),
            (3, 5),
            (4, 5),
            (8, 5),
            (9, 5),
            (10, 5),
            (2, 7),
            (3, 7),
            (4, 7),
            (8, 7),
            (9, 7),
            (10, 7),
            (0, 8),
            (5, 8),
            (7, 8),
            (12, 8),
            (0, 9),
            (5, 9),
            (7, 9),
            (12, 9),
            (0, 10),
            (5, 10),
            (7, 10),
            (12, 10),
            (2, 12),
            (3, 12),
            (4, 12),
            (8, 12),
            (9, 12),
            (10, 12),
        ],
        "diehard" => vec![(6, 0), (0, 1), (1, 1), (1, 2), (5, 2), (6, 2), (7, 2)],
        "acorn" => vec![(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)],
        "r_pentomino" => vec![(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
        "gosper_glider_gun" => vec![
            (1, 5),
            (1, 6),
            (2, 5),
            (2, 6),
            (11, 5),
            (11, 6),
            (11, 7),
            (12, 4),
            (12, 8),
            (13, 3),
            (13, 9),
            (14, 3),
            (14, 9),
            (15, 6),
            (16, 4),
            (16, 8),
            (17, 5),
            (17, 6),
            (17, 7),
            (18, 6),
            (21, 3),
            (21, 4),
            (21, 5),
            (22, 3),
            (22, 4),
            (22, 5),
            (23, 2),
            (23, 6),
            (25, 1),
            (25, 2),
            (25, 6),
            (25, 7),
            (35, 3),
            (35, 4),
            (36, 3),
            (36, 4),
        ],
        "glider_producing_switch_engine" => parse_rle_cells(
            "bo65b$bo65b$bo65b$5bo61b$b3o2bo60b$o4bo61b$o3bo62b$b3o63b$15b2o50b$13b2o2bo49b$obo10b2o3bo48b$obo10b2o52b$obo64b$obo17bo46b$b2o14bo49b3$22bo2bo41b$26bo40b$22bo10b2o32b$26bo6b2o32b$22b4o41b$21bo45b$22b2o43b$23bo43b3$41b2o24b$41b2o24b3$13b2o52b$13b2o52b5$36b2o29b$36b3o9b2o17b$37bo10bobo16b$49bo17b$33bo33b$27bo5bo33b$26bobo11b3o24b$26bo2bo9bob2o24b$16b2o9b2o10b2o26b$15bo2bo48b$15b2ob2o47b$18bobo46b$20b2o45b$17bo3bo45b$17bo2bo44b2o$18b3o44b2o3$54bo12b$53bobo11b$40bo12b2o12b$39bobo25b$39b2o!",
        ),
        "blinker_puffer_1" => {
            parse_rle_cells("3bo$bo3bo$o$o4bo$5o4$b2o$2ob3o$b4o$2b2o2$5b2o$3bo4bo$2bo$2bo5bo$2b6o!")
        }
        _ => return None,
    };
    Some(BitGrid::from_cells(&cells))
}

fn parse_rle_cells(rle: &str) -> Vec<Cell> {
    let mut cells = Vec::new();
    let mut x = 0_i64;
    let mut y = 0_i64;
    let mut run = 0_i64;

    for ch in rle.chars() {
        match ch {
            '0'..='9' => {
                run = run * 10 + (ch as i64 - '0' as i64);
            }
            'b' => {
                x += run_length(run);
                run = 0;
            }
            'o' => {
                let len = run_length(run);
                for dx in 0..len {
                    cells.push((x + dx, y));
                }
                x += len;
                run = 0;
            }
            '$' => {
                y += run_length(run);
                x = 0;
                run = 0;
            }
            '!' => break,
            _ => {}
        }
    }

    cells
}

fn run_length(run: Coord) -> Coord {
    if run == 0 { 1 } else { run }
}

pub fn random_soup(width: Coord, height: Coord, fill_percent: u32, seed: u64) -> BitGrid {
    let mut rng = SplitMix64::new(seed);
    let mut cells = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if rng.next_u32() % 100 < fill_percent {
                cells.push((x, y));
            }
        }
    }

    BitGrid::from_cells(&cells)
}

#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        mix_seed(self.state)
    }
}
