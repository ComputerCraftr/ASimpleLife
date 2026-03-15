use std::error::Error;
use std::fmt;

use crate::bitgrid::BitGrid;
use crate::life::GameOfLife;

use super::{CompiledLifeProgram, LifeObservationPort};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifeProgramValidationError {
    NoObservationPort,
    InvertedMachineBounds,
    ZeroClockTick,
    UnsupportedFrame {
        sync_pulses: u8,
        data_bits: u8,
        least_significant_bit_first: bool,
    },
    ZeroSampleCadence,
    InvalidSamplePhase {
        phase: u64,
        cadence: u64,
    },
    FrameTimingOverflow,
    FrameTooShort {
        available: u64,
        required: u64,
    },
    ObservationPortInsideMachine {
        name: String,
    },
    ObservationPortClearance {
        name: String,
        clearance: u64,
    },
}

impl fmt::Display for LifeProgramValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObservationPort => {
                f.write_str("compiled Life program has no physical observation port")
            }
            Self::InvertedMachineBounds => {
                f.write_str("compiled Life program has inverted static machine bounds")
            }
            Self::ZeroClockTick => {
                f.write_str("compiled Life program has a zero-generation clock tick")
            }
            Self::UnsupportedFrame {
                sync_pulses,
                data_bits,
                least_significant_bit_first,
            } => write!(
                f,
                "unsupported Life output frame: sync={sync_pulses}, data_bits={data_bits}, lsb_first={least_significant_bit_first}"
            ),
            Self::ZeroSampleCadence => f.write_str("Life output sample cadence must be nonzero"),
            Self::InvalidSamplePhase { phase, cadence } => write!(
                f,
                "Life output sample phase {phase} must be in 1..={cadence}"
            ),
            Self::FrameTimingOverflow => f.write_str("Life output frame timing overflowed"),
            Self::FrameTooShort {
                available,
                required,
            } => write!(
                f,
                "output frame allows {available} generations but requires at least {required}"
            ),
            Self::ObservationPortInsideMachine { name } => write!(
                f,
                "observation port {name:?} does not clear the static machine envelope"
            ),
            Self::ObservationPortClearance { name, clearance } => write!(
                f,
                "observation port {name:?} does not provide its declared outward clearance of {clearance} cells"
            ),
        }
    }
}

impl Error for LifeProgramValidationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifeProgramSessionError {
    InvalidProgram(LifeProgramValidationError),
    UnknownObservationPort(String),
    GenerationRewind { current: u64, target: u64 },
    GenerationBudgetExceeded { required: u64, budget: u64 },
}

impl fmt::Display for LifeProgramSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProgram(error) => write!(f, "invalid compiled Life program: {error}"),
            Self::UnknownObservationPort(name) => {
                write!(
                    f,
                    "compiled Life program has no observation port named {name:?}"
                )
            }
            Self::GenerationRewind { current, target } => write!(
                f,
                "cannot advance Life session backward from generation {current} to {target}"
            ),
            Self::GenerationBudgetExceeded { required, budget } => write!(
                f,
                "advancing the Life session requires {required} generations but the budget is {budget}"
            ),
        }
    }
}

impl Error for LifeProgramSessionError {}

impl From<LifeProgramValidationError> for LifeProgramSessionError {
    fn from(value: LifeProgramValidationError) -> Self {
        Self::InvalidProgram(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifePortSample {
    generation: u64,
    pulse: bool,
    decoded_byte: Option<u8>,
}

impl LifePortSample {
    pub(super) fn new(generation: u64, pulse: bool, decoded_byte: Option<u8>) -> Self {
        Self {
            generation,
            pulse,
            decoded_byte,
        }
    }

    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn pulse(self) -> bool {
        self.pulse
    }

    pub fn decoded_byte(self) -> Option<u8> {
        self.decoded_byte
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    AwaitingSync,
    Data { value: u8, bits_read: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FramedByteDecoder {
    state: DecoderState,
    sample_cadence_generations: u64,
    sample_phase_generations: u64,
}

impl FramedByteDecoder {
    pub(super) fn new(sample_cadence_generations: u64, sample_phase_generations: u64) -> Self {
        Self {
            state: DecoderState::AwaitingSync,
            sample_cadence_generations,
            sample_phase_generations,
        }
    }

    fn should_sample(self, generation: u64) -> bool {
        generation >= self.sample_phase_generations
            && (generation - self.sample_phase_generations)
                .is_multiple_of(self.sample_cadence_generations)
    }

    pub(super) fn next_sample_after(self, generation: u64, target: u64) -> Option<u64> {
        let phase = u128::from(self.sample_phase_generations);
        let cadence = u128::from(self.sample_cadence_generations);
        let earliest = u128::from(generation) + 1;
        let sample = if earliest <= phase {
            phase
        } else {
            let distance = earliest - phase;
            phase + distance.div_ceil(cadence) * cadence
        };
        if sample > u128::from(target) {
            return None;
        }
        u64::try_from(sample).ok()
    }

    pub(super) fn observe(&mut self, pulse: bool) -> Option<u8> {
        match self.state {
            DecoderState::AwaitingSync => {
                if pulse {
                    self.state = DecoderState::Data {
                        value: 0,
                        bits_read: 0,
                    };
                }
                None
            }
            DecoderState::Data {
                mut value,
                bits_read,
            } => {
                if pulse {
                    value |= 1 << bits_read;
                }
                let bits_read = bits_read + 1;
                if bits_read == 8 {
                    self.state = DecoderState::AwaitingSync;
                    Some(value)
                } else {
                    self.state = DecoderState::Data { value, bits_read };
                    None
                }
            }
        }
    }
}

/// Independent production evolution and observation for a compiled Life grid.
///
/// The immutable compiled artifact is borrowed. The session owns only the
/// evolving Life state and incremental decoder state; decoded bytes are returned
/// to the caller and are not retained.
pub struct LifeProgramSession<'program> {
    port: &'program LifeObservationPort,
    life: GameOfLife,
    decoder: FramedByteDecoder,
}

impl<'program> LifeProgramSession<'program> {
    pub fn new(program: &'program CompiledLifeProgram) -> Result<Self, LifeProgramSessionError> {
        program.validate_session_contract()?;
        let port = program
            .observation_ports()
            .first()
            .ok_or(LifeProgramValidationError::NoObservationPort)?;
        Ok(Self::from_validated_program(program, port))
    }

    pub fn with_observation_port(
        program: &'program CompiledLifeProgram,
        port_name: &str,
    ) -> Result<Self, LifeProgramSessionError> {
        program.validate_session_contract()?;
        let port = program
            .observation_ports()
            .iter()
            .find(|port| port.name() == port_name)
            .ok_or_else(|| {
                LifeProgramSessionError::UnknownObservationPort(port_name.to_string())
            })?;
        Ok(Self::from_validated_program(program, port))
    }

    fn from_validated_program(
        program: &'program CompiledLifeProgram,
        port: &'program LifeObservationPort,
    ) -> Self {
        let metadata = program.decoder();
        Self {
            port,
            life: GameOfLife::new(program.initial_grid().clone()),
            decoder: FramedByteDecoder::new(
                metadata.sample_cadence_generations(),
                metadata.sample_phase_generations(),
            ),
        }
    }

    pub fn generation(&self) -> u64 {
        self.life.generation()
    }

    pub fn grid(&self) -> &BitGrid {
        self.life.grid()
    }

    pub fn observation_port(&self) -> &LifeObservationPort {
        self.port
    }

    pub fn advance_to_generation(
        &mut self,
        target: u64,
        generation_budget: u64,
    ) -> Result<Vec<LifePortSample>, LifeProgramSessionError> {
        let current = self.generation();
        let required = target
            .checked_sub(current)
            .ok_or(LifeProgramSessionError::GenerationRewind { current, target })?;
        if required > generation_budget {
            return Err(LifeProgramSessionError::GenerationBudgetExceeded {
                required,
                budget: generation_budget,
            });
        }

        let mut samples = Vec::new();
        for _ in 0..required {
            self.life.step();
            let generation = self.generation();
            if self.decoder.should_sample(generation) {
                let pulse = self
                    .life
                    .grid()
                    .get(self.port.origin().0, self.port.origin().1);
                samples.push(LifePortSample {
                    generation,
                    pulse,
                    decoded_byte: self.decoder.observe(pulse),
                });
            }
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    use crate::bf::life_backend::{
        LifeDecoderMetadata, LifeSignalDirection, LifeStaticBounds, LifeStaticLayout,
        LifeTimingContract,
    };

    fn pulse_fixture() -> CompiledLifeProgram {
        // A stable block keeps the declared detector cell alive through evolution.
        let initial_grid = BitGrid::from_cells(&[(8, 8), (9, 8), (8, 9), (9, 9)]);
        CompiledLifeProgram {
            initial_grid,
            tape_cells: 64,
            cell_bits: 8,
            static_layout: LifeStaticLayout {
                machine_bounds: LifeStaticBounds {
                    min_x: 0,
                    min_y: 0,
                    max_x: 0,
                    max_y: 0,
                },
            },
            timing: LifeTimingContract {
                generations_per_tick: 1,
                output_frame_generations: 9,
            },
            observation_ports: vec![LifeObservationPort {
                name: "framed_output".to_string(),
                origin: (8, 8),
                direction: LifeSignalDirection::SouthEast,
                minimum_clearance: 8,
            }],
            decoder: LifeDecoderMetadata {
                sync_pulses: 1,
                data_bits: 8,
                least_significant_bit_first: true,
                sample_cadence_generations: 1,
                sample_phase_generations: 1,
            },
        }
    }

    #[test]
    fn synthetic_pulses_resynchronize_and_decode_lsb_first() {
        let mut decoder = FramedByteDecoder::new(1, 1);
        let pulses = [
            false, false, true, true, false, true, false, false, true, false, true,
        ];
        let decoded: Vec<_> = pulses
            .into_iter()
            .filter_map(|pulse| decoder.observe(pulse))
            .collect();
        assert_eq!(decoded, vec![0xa5]);
    }

    #[test]
    fn generation_zero_cannot_decode_but_evolved_port_cells_do() {
        let program = pulse_fixture();
        let mut session = LifeProgramSession::new(&program).or_invariant("valid pulse fixture");

        assert_eq!(
            session
                .advance_to_generation(0, 0)
                .or_invariant("generation zero advance"),
            Vec::new()
        );
        let samples = session
            .advance_to_generation(9, 9)
            .or_invariant("bounded evolution");
        assert_eq!(samples.len(), 9);
        assert!(samples.iter().all(|sample| sample.pulse()));
        assert_eq!(
            samples.last().and_then(|sample| sample.decoded_byte()),
            Some(0xff)
        );
    }

    #[test]
    fn rejected_budget_does_not_partially_advance() {
        let program = pulse_fixture();
        let mut session = LifeProgramSession::new(&program).or_invariant("valid pulse fixture");
        assert_eq!(
            session.advance_to_generation(9, 8),
            Err(LifeProgramSessionError::GenerationBudgetExceeded {
                required: 9,
                budget: 8,
            })
        );
        assert_eq!(session.generation(), 0);
    }
}
