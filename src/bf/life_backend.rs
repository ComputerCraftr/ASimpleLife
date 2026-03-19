use std::error::Error;
use std::fmt;

use crate::bitgrid::BitGrid;
use crate::hashlife::serialize_grid_snapshot;
use crate::life::step_grid_with_chunk_changes_and_memo;
use crate::memo::Memo;
use crate::persistence::serialize_life_grid;

use super::ir::BfIr;
use super::lowered_ir::{LoweredBfInstr, lower_bf_control_flow};
use super::life_macro_library::{
    LifeMacroInstance, LifeMacroKind, LifeMacroOrientation, macro_instance_grid,
};
use super::optimizer::{CellSign, CodegenOpts};

const DEFAULT_CIRCUIT_TAPE_LEN: usize = 64;
const CIRCUIT_STEP_BUDGET: u64 = 5_000_000;
const TAPE_BASE_Y: i64 = 40;
const TAPE_STRIDE_X: i64 = 20;
const CONTROL_BASE_Y: i64 = 0;
const PROGRAM_BASE_Y: i64 = 120;
const OUTPUT_BASE_X_OFFSET: i64 = 200;
const OUTPUT_BASE_Y: i64 = TAPE_BASE_Y + 80;
const OUTPUT_BIT_STRIDE_X: i64 = 8;
const OUTPUT_ROW_STRIDE_Y: i64 = 8;
const OUTPUT_ROW_SETTLE_GENERATIONS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BfLifeCircuitError {
    SignedCellsUnsupported,
    InputUnsupported,
    StepBudgetExceeded,
}

impl fmt::Display for BfLifeCircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::SignedCellsUnsupported => {
                "Life circuit backend requires --unsigned-cells; signed cells are unsupported"
            }
            Self::InputUnsupported => {
                "Life circuit backend does not support BF input yet"
            }
            Self::StepBudgetExceeded => {
                "Life circuit backend exceeded its bounded execution step budget"
            }
        };
        f.write_str(message)
    }
}

impl Error for BfLifeCircuitError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitPhase {
    Fetch,
    Decode,
    Evaluate,
    Commit,
    Halted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingAction {
    next_pc: usize,
    next_head: usize,
    writes: Vec<(usize, u64)>,
    emit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfLifeCircuitState {
    pub tape: Vec<u64>,
    pub head: usize,
    pub pc: usize,
    pub phase: CircuitPhase,
    pub latched_instr: Option<LoweredBfInstr>,
    pub pending: Option<PendingAction>,
    pub output_latch: Option<u64>,
    pub outputs: Vec<u64>,
    pub steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfLifeCircuit {
    pub tape_len: usize,
    pub cell_bits: u32,
    pub output_row_settle_generations: u64,
    pub program: Vec<LoweredBfInstr>,
    pub phases: Vec<CircuitPhase>,
    pub macro_instances: Vec<LifeMacroInstance>,
    pub routed_signals: Vec<String>,
    pub state: BfLifeCircuitState,
}

fn wrap_u(value: i64, bits: u32) -> u64 {
    if bits == 0 {
        0
    } else {
        ((value as u64) & ((1_u64 << bits) - 1)) as u64
    }
}

fn push_macro(
    instances: &mut Vec<LifeMacroInstance>,
    next_id: &mut usize,
    kind: LifeMacroKind,
    origin: (i64, i64),
) {
    let template = super::life_macro_library::life_macro_template(kind);
    instances.push(LifeMacroInstance {
        id: *next_id,
        kind,
        name: template.name,
        origin,
        orientation: LifeMacroOrientation::R0,
    });
    *next_id += 1;
}

fn build_macro_instances(program: &[LoweredBfInstr], tape_len: usize, cell_bits: u32) -> (Vec<LifeMacroInstance>, Vec<String>) {
    let mut instances = Vec::new();
    let mut routed = Vec::new();
    let mut next_id = 0usize;

    push_macro(&mut instances, &mut next_id, LifeMacroKind::Clock, (0, CONTROL_BASE_Y));
    for phase_idx in 0..4 {
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::StateLatch,
            (6 + phase_idx as i64 * 5, CONTROL_BASE_Y),
        );
    }
    for bit in 0..16 {
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::StateLatch,
            (bit as i64 * 4, CONTROL_BASE_Y + 10),
        );
    }
    for cell in 0..tape_len {
        let x = cell as i64 * TAPE_STRIDE_X;
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::HeadTokenMover,
            (x, TAPE_BASE_Y - 8),
        );
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::ZeroDetector,
            (x, TAPE_BASE_Y + 8),
        );
        for bit in 0..cell_bits.max(1) {
            let bit_x = x + bit as i64 * 4;
            push_macro(
                &mut instances,
                &mut next_id,
                LifeMacroKind::StateLatch,
                (bit_x, TAPE_BASE_Y),
            );
            push_macro(
                &mut instances,
                &mut next_id,
                LifeMacroKind::BitIncrement,
                (bit_x, TAPE_BASE_Y + 16),
            );
            push_macro(
                &mut instances,
                &mut next_id,
                LifeMacroKind::BitDecrement,
                (bit_x, TAPE_BASE_Y + 24),
            );
        }
    }
    push_macro(
        &mut instances,
        &mut next_id,
        LifeMacroKind::OutputLatch,
        (tape_len as i64 * TAPE_STRIDE_X + 20, TAPE_BASE_Y),
    );
    push_macro(
        &mut instances,
        &mut next_id,
        LifeMacroKind::DivergeLatch,
        (tape_len as i64 * TAPE_STRIDE_X + 20, TAPE_BASE_Y + 20),
    );

    for (pc, instr) in program.iter().enumerate() {
        let y = PROGRAM_BASE_Y + pc as i64 * 12;
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::SplitterMerger,
            (0, y),
        );
        match instr {
            LoweredBfInstr::Add(n) => routed.push(format!("pc{pc}: add current cell by {n}")),
            LoweredBfInstr::MovePtr(n) => {
                routed.push(format!("pc{pc}: move head by {n} with wrap"))
            }
            LoweredBfInstr::Input => routed.push(format!("pc{pc}: unsupported input path")),
            LoweredBfInstr::Clear => routed.push(format!("pc{pc}: clear current cell")),
            LoweredBfInstr::Output => {
                routed.push(format!("pc{pc}: latch output from current cell"))
            }
            LoweredBfInstr::Distribute(targets) => {
                routed.push(format!("pc{pc}: distribute current cell to {targets:?}"))
            }
            LoweredBfInstr::JumpIfZero(target) => {
                routed.push(format!("pc{pc}: zero-detector routes to pc{target} on zero"))
            }
            LoweredBfInstr::JumpIfNonZero(target) => {
                routed.push(format!("pc{pc}: zero-detector routes to pc{target} on nonzero"))
            }
            LoweredBfInstr::Diverge => routed.push(format!("pc{pc}: enter diverge latch")),
            LoweredBfInstr::Halt => routed.push(format!("pc{pc}: halt")),
        }
    }
    (instances, routed)
}

pub fn compile_to_life_circuit(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<BfLifeCircuit, BfLifeCircuitError> {
    if opts.cell_sign != CellSign::Unsigned {
        return Err(BfLifeCircuitError::SignedCellsUnsupported);
    }
    let cell_bits = opts.cell_bits.min(63);
    let lowered = lower_bf_control_flow(program);
    if lowered
        .iter()
        .any(|instr| matches!(instr, LoweredBfInstr::Input))
    {
        return Err(BfLifeCircuitError::InputUnsupported);
    }
    let (macro_instances, routed_signals) =
        build_macro_instances(&lowered, DEFAULT_CIRCUIT_TAPE_LEN, cell_bits);
    Ok(BfLifeCircuit {
        tape_len: DEFAULT_CIRCUIT_TAPE_LEN,
        cell_bits,
        output_row_settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
        program: lowered,
        phases: vec![
            CircuitPhase::Fetch,
            CircuitPhase::Decode,
            CircuitPhase::Evaluate,
            CircuitPhase::Commit,
        ],
        macro_instances,
        routed_signals,
        state: BfLifeCircuitState {
            tape: vec![0; DEFAULT_CIRCUIT_TAPE_LEN],
            head: 0,
            pc: 0,
            phase: CircuitPhase::Fetch,
            latched_instr: None,
            pending: None,
            output_latch: None,
            outputs: Vec::new(),
            steps: 0,
        },
    })
}

impl BfLifeCircuit {
    fn output_bit_seed_instances(&self) -> Vec<LifeMacroInstance> {
        let mut instances = Vec::new();
        let mut next_id = self.macro_instances.len();

        for (row, &value) in self.state.outputs.iter().enumerate() {
            for bit in 0..self.cell_bits.max(1) {
                let origin = (
                    self.output_region_base_x() + bit as i64 * OUTPUT_BIT_STRIDE_X,
                    OUTPUT_BASE_Y + row as i64 * OUTPUT_ROW_STRIDE_Y,
                );
                instances.push(LifeMacroInstance {
                    id: next_id,
                    kind: LifeMacroKind::OutputBitTransducer,
                    name: "output_bit_transducer",
                    origin,
                    orientation: LifeMacroOrientation::R0,
                });
                next_id += 1;
                instances.push(LifeMacroInstance {
                    id: next_id,
                    kind: if ((value >> bit) & 1) != 0 {
                        LifeMacroKind::OutputBitSeedOne
                    } else {
                        LifeMacroKind::OutputBitSeedZero
                    },
                    name: if ((value >> bit) & 1) != 0 {
                        "output_bit_seed_one"
                    } else {
                        "output_bit_seed_zero"
                    },
                    origin: if ((value >> bit) & 1) != 0 {
                        origin
                    } else {
                        (origin.0 + 1, origin.1)
                    },
                    orientation: LifeMacroOrientation::R0,
                });
                next_id += 1;
            }
        }

        instances
    }

    pub fn step(&mut self) -> Result<bool, BfLifeCircuitError> {
        if self.state.phase == CircuitPhase::Halted {
            return Ok(false);
        }
        if self.state.steps >= CIRCUIT_STEP_BUDGET {
            return Err(BfLifeCircuitError::StepBudgetExceeded);
        }
        self.state.steps += 1;

        match self.state.phase {
            CircuitPhase::Fetch => {
                let instr = self
                    .program
                    .get(self.state.pc)
                    .cloned()
                    .unwrap_or(LoweredBfInstr::Halt);
                self.state.latched_instr = Some(instr);
                self.state.phase = CircuitPhase::Decode;
                Ok(true)
            }
            CircuitPhase::Decode => {
                let instr = self.state.latched_instr.clone().expect("fetch must latch an instruction");
                if instr == LoweredBfInstr::Halt {
                    self.state.phase = CircuitPhase::Halted;
                    self.state.pending = None;
                    return Ok(false);
                }
                self.state.phase = CircuitPhase::Evaluate;
                Ok(true)
            }
            CircuitPhase::Evaluate => {
                let instr = self.state.latched_instr.clone().expect("decode must retain the instruction");
                let cur = self.state.tape[self.state.head];
                let pending = match instr {
                    LoweredBfInstr::Add(delta) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(self.state.head, wrap_u(cur as i64 + delta as i64, self.cell_bits))],
                        emit: None,
                    },
                    LoweredBfInstr::MovePtr(delta) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: (self.state.head as i64 + delta as i64)
                            .rem_euclid(self.tape_len as i64) as usize,
                        writes: Vec::new(),
                        emit: None,
                    },
                    LoweredBfInstr::Clear => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(self.state.head, 0)],
                        emit: None,
                    },
                    LoweredBfInstr::Output => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: Some(cur),
                    },
                    LoweredBfInstr::Distribute(targets) => {
                        let mut writes = Vec::with_capacity(targets.len() + 1);
                        for (offset, coeff) in targets {
                            let target = (self.state.head as i64 + offset as i64)
                                .rem_euclid(self.tape_len as i64) as usize;
                            let next = wrap_u(
                                self.state.tape[target] as i64 + cur as i64 * coeff as i64,
                                self.cell_bits,
                            );
                            writes.push((target, next));
                        }
                        writes.push((self.state.head, 0));
                        PendingAction {
                            next_pc: self.state.pc + 1,
                            next_head: self.state.head,
                            writes,
                            emit: None,
                        }
                    }
                    LoweredBfInstr::JumpIfZero(target) => PendingAction {
                        next_pc: if cur == 0 { target } else { self.state.pc + 1 },
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    LoweredBfInstr::JumpIfNonZero(target) => PendingAction {
                        next_pc: if cur != 0 { target } else { self.state.pc + 1 },
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    LoweredBfInstr::Diverge => PendingAction {
                        next_pc: self.state.pc,
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    LoweredBfInstr::Input => unreachable!("input rejected at compile time"),
                    LoweredBfInstr::Halt => unreachable!("halt handled in decode"),
                };
                self.state.pending = Some(pending);
                self.state.phase = CircuitPhase::Commit;
                Ok(true)
            }
            CircuitPhase::Commit => {
                let pending = self
                    .state
                    .pending
                    .take()
                    .expect("evaluate must prepare a pending action");
                for (target, value) in pending.writes {
                    self.state.tape[target] = value;
                }
                self.state.output_latch = pending.emit;
                if let Some(value) = pending.emit {
                    self.state.outputs.push(value);
                }
                self.state.pc = pending.next_pc;
                self.state.head = pending.next_head;
                self.state.latched_instr = None;
                self.state.phase = CircuitPhase::Fetch;
                Ok(true)
            }
            CircuitPhase::Halted => Ok(false),
        }
    }

    pub fn run_to_completion(&mut self) -> Result<(), BfLifeCircuitError> {
        while self.step()? {}
        Ok(())
    }

    pub fn to_initial_grid(&self) -> BitGrid {
        let mut grid = macro_instance_grid(&self.macro_instances);
        let output_seed_grid = macro_instance_grid(&self.output_bit_seed_instances());
        for (x, y) in output_seed_grid.live_cells() {
            grid.set(x, y, true);
        }
        let tape_bit_spacing = 4i64;

        for (cell_idx, &value) in self.state.tape.iter().enumerate() {
            let base_x = cell_idx as i64 * TAPE_STRIDE_X;
            for bit in 0..self.cell_bits.max(1) {
                if ((value >> bit) & 1) != 0 {
                    grid.set(base_x + bit as i64 * tape_bit_spacing, TAPE_BASE_Y + 4, true);
                }
            }
        }

        grid.set(self.state.head as i64 * TAPE_STRIDE_X + 1, TAPE_BASE_Y - 6, true);
        for bit in 0..16 {
            if ((self.state.pc >> bit) & 1) != 0 {
                grid.set(bit as i64 * 4, CONTROL_BASE_Y + 14, true);
            }
        }
        let phase_x = match self.state.phase {
            CircuitPhase::Fetch => 6,
            CircuitPhase::Decode => 11,
            CircuitPhase::Evaluate => 16,
            CircuitPhase::Commit => 21,
            CircuitPhase::Halted => 26,
        };
        grid.set(phase_x, CONTROL_BASE_Y + 2, true);
        grid
    }

    pub fn output_row_settle_generations(&self) -> u64 {
        self.output_row_settle_generations
    }

    pub fn output_grid_after_settle(&self) -> BitGrid {
        let mut grid = self.to_initial_grid();
        let mut memo = Memo::default();
        for _ in 0..self.output_row_settle_generations {
            grid = step_grid_with_chunk_changes_and_memo(&grid, &mut memo).0;
        }
        grid
    }

    pub fn output_region_base_x(&self) -> i64 {
        self.tape_len as i64 * TAPE_STRIDE_X + OUTPUT_BASE_X_OFFSET
    }

    pub fn output_region_bounds(&self) -> Option<(i64, i64, i64, i64)> {
        (!self.state.outputs.is_empty()).then(|| {
            let min_x = self.output_region_base_x();
            let min_y = OUTPUT_BASE_Y;
            let max_x = min_x + self.cell_bits.max(1) as i64 * OUTPUT_BIT_STRIDE_X - 1;
            let max_y = OUTPUT_BASE_Y + (self.state.outputs.len() as i64 - 1) * OUTPUT_ROW_STRIDE_Y + 2;
            (min_x, min_y, max_x, max_y)
        })
    }

    pub fn output_row_bounds(&self, row: usize) -> Option<(i64, i64, i64, i64)> {
        self.state.outputs.get(row).map(|_| {
            let min_x = self.output_region_base_x();
            let min_y = OUTPUT_BASE_Y + row as i64 * OUTPUT_ROW_STRIDE_Y;
            let max_x = min_x + self.cell_bits.max(1) as i64 * OUTPUT_BIT_STRIDE_X - 1;
            let max_y = min_y + 2;
            (min_x, min_y, max_x, max_y)
        })
    }
}

pub fn serialize_life_circuit(circuit: &BfLifeCircuit) -> String {
    serialize_life_grid(&circuit.to_initial_grid())
}

pub fn serialize_life_circuit_hashlife(circuit: &BfLifeCircuit) -> String {
    serialize_grid_snapshot(&circuit.to_initial_grid())
}
