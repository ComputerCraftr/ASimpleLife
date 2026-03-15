use std::error::Error;
use std::fmt;

#[cfg(test)]
use crate::RequiredExt;
use crate::bitgrid::BitGrid;

use super::ir::BfIr;
#[cfg(test)]
use super::ir::ShiftDir;
#[cfg(test)]
use super::ir::validate_canonical_ir;
use super::life_macro_library::{LifeMacroInstance, LifeMacroKind};
#[cfg(test)]
use super::life_macro_library::{LifeMacroOrientation, macro_instance_grid};
use super::lowered_ir::PhysicalBfInstr;
#[cfg(test)]
use super::lowered_ir::{expand_distribute_to_primitive, lower_bf_control_flow};
#[cfg(test)]
use super::optimizer::CellSign;
use super::optimizer::CodegenOpts;
#[cfg(test)]
use super::tape::wrapped_index as wrap_tape_index;

#[cfg(test)]
const DEFAULT_CIRCUIT_TAPE_LEN: usize = super::BF_LIFE_TAPE_LEN;
#[cfg(test)]
const CIRCUIT_STEP_BUDGET: u64 = 5_000_000;
#[cfg(test)]
const TAPE_BASE_Y: i64 = 40;
#[cfg(test)]
const CONTROL_BASE_Y: i64 = 0;
#[cfg(test)]
const PROGRAM_BASE_Y: i64 = 120;
#[cfg(test)]
const OUTPUT_ROW_SETTLE_GENERATIONS: u64 = 1;

#[cfg(test)]
fn tape_stride_x(cell_bits: u32) -> i64 {
    (i64::from(cell_bits.max(1)) * 4 + 8).max(20)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BfLifeCircuitError {
    ExecutableCircuitUnavailable,
    UnsupportedCellWidth { requested: u32 },
    SignedCellsUnsupported,
    InputUnsupported,
    ProgramTooLarge { instructions: usize, maximum: usize },
    Geometry(String),
    Routing(String),
    Timing(String),
    StepBudgetExceeded,
    NonCanonicalIr(String),
}

impl fmt::Display for BfLifeCircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ExecutableCircuitUnavailable => {
                "BF-to-Life emission requires a repository-local, licensed, independently verified physical component library"
            }
            Self::UnsupportedCellWidth { requested } => {
                return write!(
                    f,
                    "Life circuit backend requires exactly 8-bit cells, got {requested}"
                );
            }
            Self::SignedCellsUnsupported => {
                "Life circuit backend requires --signed-cells false; signed cells are unsupported"
            }
            Self::InputUnsupported => "Life circuit backend does not support BF input yet",
            Self::ProgramTooLarge {
                instructions,
                maximum,
            } => {
                return write!(
                    f,
                    "Life circuit program has {instructions} instructions; maximum is {maximum}"
                );
            }
            Self::Geometry(message) => return write!(f, "Life circuit geometry error: {message}"),
            Self::Routing(message) => return write!(f, "Life circuit routing error: {message}"),
            Self::Timing(message) => return write!(f, "Life circuit timing error: {message}"),
            Self::StepBudgetExceeded => {
                "Life circuit backend exceeded its bounded execution step budget"
            }
            Self::NonCanonicalIr(message) => return f.write_str(message),
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
pub struct LifeTimingContract {
    generations_per_tick: u64,
    output_frame_generations: u64,
}

impl LifeTimingContract {
    pub fn generations_per_tick(&self) -> u64 {
        self.generations_per_tick
    }

    pub fn output_frame_generations(&self) -> u64 {
        self.output_frame_generations
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeObservationPort {
    name: String,
    origin: (i64, i64),
    direction: LifeSignalDirection,
    minimum_clearance: u64,
}

impl LifeObservationPort {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn origin(&self) -> (i64, i64) {
        self.origin
    }

    pub fn direction(&self) -> LifeSignalDirection {
        self.direction
    }

    pub fn minimum_clearance(&self) -> u64 {
        self.minimum_clearance
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifeSignalDirection {
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifeStaticBounds {
    min_x: i64,
    min_y: i64,
    max_x: i64,
    max_y: i64,
}

impl LifeStaticBounds {
    pub fn min_x(self) -> i64 {
        self.min_x
    }

    pub fn min_y(self) -> i64 {
        self.min_y
    }

    pub fn max_x(self) -> i64 {
        self.max_x
    }

    pub fn max_y(self) -> i64 {
        self.max_y
    }

    fn contains(self, point: (i64, i64)) -> bool {
        point.0 >= self.min_x
            && point.0 <= self.max_x
            && point.1 >= self.min_y
            && point.1 <= self.max_y
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeStaticLayout {
    machine_bounds: LifeStaticBounds,
}

impl LifeStaticLayout {
    pub fn machine_bounds(&self) -> LifeStaticBounds {
        self.machine_bounds
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeDecoderMetadata {
    sync_pulses: u8,
    data_bits: u8,
    least_significant_bit_first: bool,
    sample_cadence_generations: u64,
    sample_phase_generations: u64,
}

impl LifeDecoderMetadata {
    pub fn sync_pulses(&self) -> u8 {
        self.sync_pulses
    }

    pub fn data_bits(&self) -> u8 {
        self.data_bits
    }

    pub fn least_significant_bit_first(&self) -> bool {
        self.least_significant_bit_first
    }

    pub fn sample_cadence_generations(&self) -> u64 {
        self.sample_cadence_generations
    }

    pub fn sample_phase_generations(&self) -> u64 {
        self.sample_phase_generations
    }
}

/// Immutable output of the physical BF-to-Life compiler.
///
/// No interpreter tape, program counter, or host-produced output is retained in
/// this artifact. Evolution and observation are defined solely by `initial_grid`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledLifeProgram {
    initial_grid: BitGrid,
    tape_cells: u8,
    cell_bits: u8,
    static_layout: LifeStaticLayout,
    timing: LifeTimingContract,
    observation_ports: Vec<LifeObservationPort>,
    decoder: LifeDecoderMetadata,
}

impl CompiledLifeProgram {
    pub fn initial_grid(&self) -> &BitGrid {
        &self.initial_grid
    }

    pub fn tape_cells(&self) -> u8 {
        self.tape_cells
    }

    pub fn cell_bits(&self) -> u8 {
        self.cell_bits
    }

    pub fn static_layout(&self) -> &LifeStaticLayout {
        &self.static_layout
    }

    pub fn timing(&self) -> &LifeTimingContract {
        &self.timing
    }

    pub fn observation_ports(&self) -> &[LifeObservationPort] {
        &self.observation_ports
    }

    pub fn decoder(&self) -> &LifeDecoderMetadata {
        &self.decoder
    }

    pub(crate) fn validate_physical_contract(&self) -> Result<(), BfLifeCircuitError> {
        self.validate_session_contract()
            .map_err(|error| match error {
                LifeProgramValidationError::NoObservationPort
                | LifeProgramValidationError::ObservationPortInsideMachine { .. }
                | LifeProgramValidationError::ObservationPortClearance { .. } => {
                    BfLifeCircuitError::Routing(error.to_string())
                }
                LifeProgramValidationError::InvertedMachineBounds => {
                    BfLifeCircuitError::Geometry(error.to_string())
                }
                _ => BfLifeCircuitError::Timing(error.to_string()),
            })
    }

    pub fn validate_session_contract(&self) -> Result<(), LifeProgramValidationError> {
        if self.observation_ports.is_empty() {
            return Err(LifeProgramValidationError::NoObservationPort);
        }
        let bounds = self.static_layout.machine_bounds;
        if bounds.min_x > bounds.max_x || bounds.min_y > bounds.max_y {
            return Err(LifeProgramValidationError::InvertedMachineBounds);
        }
        if self.timing.generations_per_tick == 0 {
            return Err(LifeProgramValidationError::ZeroClockTick);
        }
        if self.decoder.sync_pulses != 1
            || self.decoder.data_bits != 8
            || !self.decoder.least_significant_bit_first
        {
            return Err(LifeProgramValidationError::UnsupportedFrame {
                sync_pulses: self.decoder.sync_pulses,
                data_bits: self.decoder.data_bits,
                least_significant_bit_first: self.decoder.least_significant_bit_first,
            });
        }
        if self.decoder.sample_cadence_generations == 0 {
            return Err(LifeProgramValidationError::ZeroSampleCadence);
        }
        if self.decoder.sample_phase_generations == 0
            || self.decoder.sample_phase_generations > self.decoder.sample_cadence_generations
        {
            return Err(LifeProgramValidationError::InvalidSamplePhase {
                phase: self.decoder.sample_phase_generations,
                cadence: self.decoder.sample_cadence_generations,
            });
        }
        let frame_pulses = u64::from(self.decoder.sync_pulses)
            .checked_add(u64::from(self.decoder.data_bits))
            .ok_or(LifeProgramValidationError::FrameTimingOverflow)?;
        let minimum_frame_generations = frame_pulses
            .checked_mul(self.decoder.sample_cadence_generations)
            .ok_or(LifeProgramValidationError::FrameTimingOverflow)?;
        if self.timing.output_frame_generations < minimum_frame_generations {
            return Err(LifeProgramValidationError::FrameTooShort {
                available: self.timing.output_frame_generations,
                required: minimum_frame_generations,
            });
        }
        for port in &self.observation_ports {
            if port.minimum_clearance == 0 || bounds.contains(port.origin) {
                return Err(LifeProgramValidationError::ObservationPortInsideMachine {
                    name: port.name.clone(),
                });
            }
            let x = i128::from(port.origin.0);
            let y = i128::from(port.origin.1);
            let min_x = i128::from(bounds.min_x);
            let min_y = i128::from(bounds.min_y);
            let max_x = i128::from(bounds.max_x);
            let max_y = i128::from(bounds.max_y);
            let clearance = i128::from(port.minimum_clearance);
            let clears_envelope = match port.direction {
                LifeSignalDirection::NorthEast => x - max_x >= clearance && min_y - y >= clearance,
                LifeSignalDirection::NorthWest => min_x - x >= clearance && min_y - y >= clearance,
                LifeSignalDirection::SouthEast => x - max_x >= clearance && y - max_y >= clearance,
                LifeSignalDirection::SouthWest => min_x - x >= clearance && y - max_y >= clearance,
            };
            if !clears_envelope {
                return Err(LifeProgramValidationError::ObservationPortClearance {
                    name: port.name.clone(),
                    clearance: port.minimum_clearance,
                });
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn test_fixture(initial_grid: BitGrid) -> Self {
        let (min_x, min_y, max_x, max_y) = initial_grid.bounds().unwrap_or((0, 0, 0, 0));
        let clearance = 8_i64;
        let program = Self {
            initial_grid,
            tape_cells: 64,
            cell_bits: 8,
            static_layout: LifeStaticLayout {
                machine_bounds: LifeStaticBounds {
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                },
            },
            timing: LifeTimingContract {
                generations_per_tick: 4,
                output_frame_generations: 36,
            },
            observation_ports: vec![LifeObservationPort {
                name: "framed_output".to_string(),
                origin: (
                    max_x.checked_add(clearance).unwrap_or(max_x),
                    max_y.checked_add(clearance).unwrap_or(max_y),
                ),
                direction: LifeSignalDirection::SouthEast,
                minimum_clearance: 8,
            }],
            decoder: LifeDecoderMetadata {
                sync_pulses: 1,
                data_bits: 8,
                least_significant_bit_first: true,
                sample_cadence_generations: 4,
                sample_phase_generations: 1,
            },
        };
        debug_assert!(program.validate_physical_contract().is_ok());
        program
    }
}

/// Compatibility name retained by the crate's existing re-export surface.
pub type BfLifeCircuit = CompiledLifeProgram;

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
#[cfg(test)]
pub(crate) struct ReferenceLifeScaffold {
    pub tape_len: usize,
    pub cell_bits: u32,
    pub program: Vec<PhysicalBfInstr>,
    pub placed_machine: PlacedLifeMachine,
    pub state: BfLifeCircuitState,
}

#[cfg(test)]
fn wrap_u64(value: u64, bits: u32) -> u64 {
    if bits == 0 {
        0
    } else {
        value & ((1_u64 << bits) - 1)
    }
}

#[cfg(test)]
fn wrap_add_product(base: u64, lhs: u64, rhs: u64, bits: u32) -> u64 {
    wrap_u64(base.wrapping_add(lhs.wrapping_mul(rhs)), bits)
}

#[cfg(test)]
fn layout_coord(value: usize) -> i64 {
    i64::try_from(value).or_invariant("BF Life layout coordinate exceeded i64")
}

#[cfg(test)]
fn wrap_shift_left(value: u64, amount: u32, bits: u32) -> u64 {
    if bits == 0 || amount >= 64 {
        0
    } else {
        wrap_u64((value & ((1_u64 << bits) - 1)) << amount, bits)
    }
}

#[cfg(test)]
fn wrap_shift_right(value: u64, amount: u32, bits: u32) -> u64 {
    if bits == 0 || amount >= 64 {
        0
    } else {
        ((value & ((1_u64 << bits) - 1)) >> amount) & ((1_u64 << bits) - 1)
    }
}

mod hashlife_session;
#[cfg(test)]
mod layout;
#[cfg(test)]
mod runtime;
mod serialize;
mod session;

pub use hashlife_session::{HashLifeProgramSession, HashLifeProgramSessionError};
#[cfg(test)]
use layout::build_placed_machine;
pub use serialize::{serialize_life_circuit, serialize_life_circuit_hashlife};
#[cfg(test)]
pub(crate) use serialize::{serialize_life_scaffold, serialize_life_scaffold_hashlife};
pub use session::{
    LifePortSample, LifeProgramSession, LifeProgramSessionError, LifeProgramValidationError,
};

pub fn compile_to_life_circuit(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<CompiledLifeProgram, BfLifeCircuitError> {
    validate_compile_contract(program, opts)?;
    Err(BfLifeCircuitError::ExecutableCircuitUnavailable)
}

const MAX_PHYSICAL_PROGRAM_INSTRUCTIONS: usize = 4_096;

fn validate_compile_contract(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<(), BfLifeCircuitError> {
    if opts.cell_sign != super::optimizer::CellSign::Unsigned {
        return Err(BfLifeCircuitError::SignedCellsUnsupported);
    }
    if opts.cell_bits != 8 {
        return Err(BfLifeCircuitError::UnsupportedCellWidth {
            requested: opts.cell_bits,
        });
    }
    let mut stack = vec![program];
    let mut instructions = 0usize;
    while let Some(nodes) = stack.pop() {
        for node in nodes {
            instructions += 1;
            if instructions > MAX_PHYSICAL_PROGRAM_INSTRUCTIONS {
                return Err(BfLifeCircuitError::ProgramTooLarge {
                    instructions,
                    maximum: MAX_PHYSICAL_PROGRAM_INSTRUCTIONS,
                });
            }
            match node {
                BfIr::Input => return Err(BfLifeCircuitError::InputUnsupported),
                BfIr::Loop(body) => stack.push(body),
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn compile_life_scaffold(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<ReferenceLifeScaffold, BfLifeCircuitError> {
    fn contains_input(nodes: &[BfIr]) -> bool {
        let mut stack: Vec<&[BfIr]> = vec![nodes];
        while let Some(cur) = stack.pop() {
            for instr in cur {
                match instr {
                    BfIr::Input => return true,
                    BfIr::Loop(body) => stack.push(body),
                    _ => {}
                }
            }
        }
        false
    }

    if opts.cell_sign != CellSign::Unsigned {
        return Err(BfLifeCircuitError::SignedCellsUnsupported);
    }
    validate_canonical_ir(program).map_err(|err| {
        BfLifeCircuitError::NonCanonicalIr(format!(
            "Life backend requires canonical richer IR: {err}"
        ))
    })?;
    let cell_bits = opts.cell_bits.min(63);
    if contains_input(program) {
        return Err(BfLifeCircuitError::InputUnsupported);
    }
    let primitive_program = expand_distribute_to_primitive(program);
    let lowered = lower_bf_control_flow(&primitive_program);
    let placed_machine = build_placed_machine(&lowered, DEFAULT_CIRCUIT_TAPE_LEN, cell_bits);
    Ok(ReferenceLifeScaffold {
        tape_len: DEFAULT_CIRCUIT_TAPE_LEN,
        cell_bits,
        program: lowered,
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
