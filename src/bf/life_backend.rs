use std::error::Error;
use std::fmt;

use crate::bitgrid::BitGrid;
use crate::hashlife::serialize_grid_snapshot;
use crate::life::step_grid_with_chunk_changes_and_memo;
use crate::memo::Memo;
use crate::persistence::serialize_life_grid;

use super::ir::BfIr;
use super::lowered_ir::{PhysicalBfInstr, lower_bf_physical_control_flow};
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
    pub latched_instr: Option<PhysicalBfInstr>,
    pub pending: Option<PendingAction>,
    pub output_latch: Option<u64>,
    pub outputs: Vec<u64>,
    pub steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RailGroup {
    ProgramControl,
    Phase,
    TapeData,
    HeadMove,
    ZeroDetectBranch,
    OutputTransducer,
    HaltDiverge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroTimingSpec {
    pub kind: LifeMacroKind,
    pub active_phase: CircuitPhase,
    pub settle_generations: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedRail {
    pub name: String,
    pub group: RailGroup,
    pub source: String,
    pub sink: String,
    pub phase: CircuitPhase,
    pub delay_generations: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedLifeMachine {
    pub phases: Vec<CircuitPhase>,
    pub macro_instances: Vec<LifeMacroInstance>,
    pub routed_rails: Vec<RoutedRail>,
    pub macro_timing_specs: Vec<MacroTimingSpec>,
    pub output_row_settle_generations: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfLifeCircuit {
    pub tape_len: usize,
    pub cell_bits: u32,
    pub output_row_settle_generations: u64,
    pub program: Vec<PhysicalBfInstr>,
    pub phases: Vec<CircuitPhase>,
    pub macro_instances: Vec<LifeMacroInstance>,
    pub routed_signals: Vec<String>,
    pub placed_machine: PlacedLifeMachine,
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

fn build_placed_machine(
    program: &[PhysicalBfInstr],
    tape_len: usize,
    cell_bits: u32,
) -> PlacedLifeMachine {
    let mut instances = Vec::new();
    let mut routed = Vec::new();
    let mut next_id = 0usize;
    let phases = vec![
        CircuitPhase::Fetch,
        CircuitPhase::Decode,
        CircuitPhase::Evaluate,
        CircuitPhase::Commit,
    ];

    push_macro(&mut instances, &mut next_id, LifeMacroKind::Clock, (0, CONTROL_BASE_Y));
    for phase_idx in 0..4 {
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::StateLatch,
            (6 + phase_idx as i64 * 5, CONTROL_BASE_Y),
        );
        routed.push(RoutedRail {
            name: format!("phase_{phase_idx}_tick"),
            group: RailGroup::Phase,
            source: "clock_tick".to_string(),
            sink: format!("phase_latch_{phase_idx}"),
            phase: CircuitPhase::Fetch,
            delay_generations: 1,
        });
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
        routed.push(RoutedRail {
            name: format!("pc{pc}_fetch"),
            group: RailGroup::ProgramControl,
            source: "program_counter".to_string(),
            sink: format!("pc{pc}_decode"),
            phase: CircuitPhase::Fetch,
            delay_generations: 1,
        });
        match instr {
            PhysicalBfInstr::Add(n) => routed.push(RoutedRail {
                name: format!("pc{pc}_add"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_write_add_{n}"),
                phase: CircuitPhase::Evaluate,
                delay_generations: 1,
            }),
            PhysicalBfInstr::MovePtr(n) => routed.push(RoutedRail {
                name: format!("pc{pc}_move"),
                group: RailGroup::HeadMove,
                source: format!("pc{pc}_decode"),
                sink: format!("head_move_{n}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Clear => routed.push(RoutedRail {
                name: format!("pc{pc}_clear"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_clear_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Output => routed.push(RoutedRail {
                name: format!("pc{pc}_output"),
                group: RailGroup::OutputTransducer,
                source: format!("pc{pc}_decode"),
                sink: format!("output_seed_row_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
            }),
            PhysicalBfInstr::JumpIfZero(target) => routed.push(RoutedRail {
                name: format!("pc{pc}_jump_zero"),
                group: RailGroup::ZeroDetectBranch,
                source: format!("pc{pc}_zero_detect"),
                sink: format!("pc{target}_fetch"),
                phase: CircuitPhase::Evaluate,
                delay_generations: 1,
            }),
            PhysicalBfInstr::JumpIfNonZero(target) => routed.push(RoutedRail {
                name: format!("pc{pc}_jump_nonzero"),
                group: RailGroup::ZeroDetectBranch,
                source: format!("pc{pc}_zero_detect"),
                sink: format!("pc{target}_fetch"),
                phase: CircuitPhase::Evaluate,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Diverge => routed.push(RoutedRail {
                name: format!("pc{pc}_diverge"),
                group: RailGroup::HaltDiverge,
                source: format!("pc{pc}_decode"),
                sink: "diverge_latch".to_string(),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Halt => routed.push(RoutedRail {
                name: format!("pc{pc}_halt"),
                group: RailGroup::HaltDiverge,
                source: format!("pc{pc}_decode"),
                sink: "halt_latch".to_string(),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
        }
    }
    let macro_timing_specs = vec![
        MacroTimingSpec {
            kind: LifeMacroKind::Clock,
            active_phase: CircuitPhase::Fetch,
            settle_generations: 0,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::StateLatch,
            active_phase: CircuitPhase::Commit,
            settle_generations: 0,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::HeadTokenMover,
            active_phase: CircuitPhase::Commit,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::BitIncrement,
            active_phase: CircuitPhase::Evaluate,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::BitDecrement,
            active_phase: CircuitPhase::Evaluate,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::ZeroDetector,
            active_phase: CircuitPhase::Evaluate,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::OutputLatch,
            active_phase: CircuitPhase::Commit,
            settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::OutputBitTransducer,
            active_phase: CircuitPhase::Commit,
            settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::DivergeLatch,
            active_phase: CircuitPhase::Commit,
            settle_generations: 0,
        },
    ];

    PlacedLifeMachine {
        phases,
        macro_instances: instances,
        routed_rails: routed,
        macro_timing_specs,
        output_row_settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
    }
}

pub fn compile_to_life_circuit(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<BfLifeCircuit, BfLifeCircuitError> {
    fn contains_input(nodes: &[BfIr]) -> bool {
        nodes.iter().any(|instr| match instr {
            BfIr::Input => true,
            BfIr::Loop(body) => contains_input(body),
            _ => false,
        })
    }

    if opts.cell_sign != CellSign::Unsigned {
        return Err(BfLifeCircuitError::SignedCellsUnsupported);
    }
    let cell_bits = opts.cell_bits.min(63);
    if contains_input(program) {
        return Err(BfLifeCircuitError::InputUnsupported);
    }
    let lowered = lower_bf_physical_control_flow(program);
    let placed_machine =
        build_placed_machine(&lowered, DEFAULT_CIRCUIT_TAPE_LEN, cell_bits);
    Ok(BfLifeCircuit {
        tape_len: DEFAULT_CIRCUIT_TAPE_LEN,
        cell_bits,
        output_row_settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
        program: lowered,
        phases: placed_machine.phases.clone(),
        macro_instances: placed_machine.macro_instances.clone(),
        routed_signals: placed_machine
            .routed_rails
            .iter()
            .map(|rail| format!("{:?}/{}: {} -> {}", rail.group, rail.name, rail.source, rail.sink))
            .collect(),
        placed_machine,
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
    pub fn placed_machine(&self) -> &PlacedLifeMachine {
        &self.placed_machine
    }

    pub fn routed_rails(&self) -> &[RoutedRail] {
        &self.placed_machine.routed_rails
    }

    pub fn macro_timing_specs(&self) -> &[MacroTimingSpec] {
        &self.placed_machine.macro_timing_specs
    }

    pub fn debug_routed_signals(&self) -> Vec<String> {
        self.placed_machine
            .routed_rails
            .iter()
            .map(|rail| format!("{:?}/{}: {} -> {}", rail.group, rail.name, rail.source, rail.sink))
            .collect()
    }

    pub fn reference_run_to_completion(&mut self) -> Result<(), BfLifeCircuitError> {
        while self.step()? {}
        Ok(())
    }
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
                    .unwrap_or(PhysicalBfInstr::Halt);
                self.state.latched_instr = Some(instr);
                self.state.phase = CircuitPhase::Decode;
                Ok(true)
            }
            CircuitPhase::Decode => {
                let instr = self.state.latched_instr.clone().expect("fetch must latch an instruction");
                if instr == PhysicalBfInstr::Halt {
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
                    PhysicalBfInstr::Add(delta) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(self.state.head, wrap_u(cur as i64 + delta as i64, self.cell_bits))],
                        emit: None,
                    },
                    PhysicalBfInstr::MovePtr(delta) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: (self.state.head as i64 + delta as i64)
                            .rem_euclid(self.tape_len as i64) as usize,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::Clear => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(self.state.head, 0)],
                        emit: None,
                    },
                    PhysicalBfInstr::Output => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: Some(cur),
                    },
                    PhysicalBfInstr::JumpIfZero(target) => PendingAction {
                        next_pc: if cur == 0 { target } else { self.state.pc + 1 },
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::JumpIfNonZero(target) => PendingAction {
                        next_pc: if cur != 0 { target } else { self.state.pc + 1 },
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::Diverge => PendingAction {
                        next_pc: self.state.pc,
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::Halt => unreachable!("halt handled in decode"),
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
