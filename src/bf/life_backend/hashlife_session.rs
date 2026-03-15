use std::error::Error;
use std::fmt;

use crate::hashlife::{
    HashLifeAdvanceError, HashLifeConversionError, HashLifeExecutionStats, HashLifeLimits,
    HashLifeSession,
};

use super::session::FramedByteDecoder;
use super::{CompiledLifeProgram, LifeObservationPort, LifePortSample, LifeProgramValidationError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashLifeProgramSessionError {
    InvalidProgram(LifeProgramValidationError),
    UnknownObservationPort(String),
    GenerationRewind { current: u64, target: u64 },
    GenerationBudgetExceeded { required: u64, budget: u64 },
    Load(HashLifeConversionError),
    Advance(HashLifeAdvanceError),
}

impl fmt::Display for HashLifeProgramSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProgram(error) => write!(f, "invalid compiled Life program: {error}"),
            Self::UnknownObservationPort(name) => write!(
                f,
                "compiled Life program has no observation port named {name:?}"
            ),
            Self::GenerationRewind { current, target } => write!(
                f,
                "cannot advance HashLife session backward from generation {current} to {target}"
            ),
            Self::GenerationBudgetExceeded { required, budget } => write!(
                f,
                "advancing the HashLife session requires {required} generations but the budget is {budget}"
            ),
            Self::Load(error) => write!(f, "failed to load the compiled Life grid: {error:?}"),
            Self::Advance(error) => {
                write!(f, "failed to advance the compiled Life grid: {error:?}")
            }
        }
    }
}

impl Error for HashLifeProgramSessionError {}

impl From<LifeProgramValidationError> for HashLifeProgramSessionError {
    fn from(value: LifeProgramValidationError) -> Self {
        Self::InvalidProgram(value)
    }
}

/// HashLife-backed evolution and framed observation for a compiled Life grid.
///
/// The session owns only HashLife grid state and decoder state. It never owns or
/// reconstructs a host BF tape, program counter, or output buffer.
pub struct HashLifeProgramSession<'program> {
    port: &'program LifeObservationPort,
    life: HashLifeSession,
    decoder: FramedByteDecoder,
}

impl<'program> HashLifeProgramSession<'program> {
    pub fn new(
        program: &'program CompiledLifeProgram,
    ) -> Result<Self, HashLifeProgramSessionError> {
        Self::new_with_limits(program, HashLifeLimits::default())
    }

    pub fn new_with_limits(
        program: &'program CompiledLifeProgram,
        limits: HashLifeLimits,
    ) -> Result<Self, HashLifeProgramSessionError> {
        program.validate_session_contract()?;
        let port = program
            .observation_ports()
            .first()
            .ok_or(LifeProgramValidationError::NoObservationPort)?;
        Self::from_validated_program(program, port, limits)
    }

    pub fn with_observation_port(
        program: &'program CompiledLifeProgram,
        port_name: &str,
    ) -> Result<Self, HashLifeProgramSessionError> {
        program.validate_session_contract()?;
        let port = program
            .observation_ports()
            .iter()
            .find(|port| port.name() == port_name)
            .ok_or_else(|| {
                HashLifeProgramSessionError::UnknownObservationPort(port_name.to_string())
            })?;
        Self::from_validated_program(program, port, HashLifeLimits::default())
    }

    fn from_validated_program(
        program: &'program CompiledLifeProgram,
        port: &'program LifeObservationPort,
        limits: HashLifeLimits,
    ) -> Result<Self, HashLifeProgramSessionError> {
        let metadata = program.decoder();
        let mut life = HashLifeSession::with_limits(limits);
        life.try_load_grid(program.initial_grid())
            .map_err(HashLifeProgramSessionError::Load)?;
        Ok(Self {
            port,
            life,
            decoder: FramedByteDecoder::new(
                metadata.sample_cadence_generations(),
                metadata.sample_phase_generations(),
            ),
        })
    }

    pub fn generation(&self) -> u64 {
        self.life.generation()
    }

    pub fn observation_port(&self) -> &LifeObservationPort {
        self.port
    }

    pub fn execution_stats(&self) -> HashLifeExecutionStats {
        self.life.execution_stats()
    }

    pub fn advance_to_generation(
        &mut self,
        target: u64,
        generation_budget: u64,
    ) -> Result<Vec<LifePortSample>, HashLifeProgramSessionError> {
        let current = self.generation();
        let required = target
            .checked_sub(current)
            .ok_or(HashLifeProgramSessionError::GenerationRewind { current, target })?;
        if required > generation_budget {
            return Err(HashLifeProgramSessionError::GenerationBudgetExceeded {
                required,
                budget: generation_budget,
            });
        }

        let mut samples = Vec::new();
        while let Some(sample_generation) =
            self.decoder.next_sample_after(self.generation(), target)
        {
            self.advance_by(sample_generation - self.generation())?;
            samples.push(self.sample_port());
        }
        self.advance_by(target - self.generation())?;
        Ok(samples)
    }

    fn advance_by(&mut self, generations: u64) -> Result<(), HashLifeProgramSessionError> {
        self.life
            .advance_root(generations)
            .map(|_| ())
            .map_err(HashLifeProgramSessionError::Advance)
    }

    fn sample_port(&mut self) -> LifePortSample {
        let (x, y) = self.port.origin();
        let pulse = self.life.sample_cell(x, y).unwrap_or(false);
        LifePortSample::new(self.generation(), pulse, self.decoder.observe(pulse))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    use crate::bf::life_backend::{
        LifeDecoderMetadata, LifeProgramSession, LifeSignalDirection, LifeStaticBounds,
        LifeStaticLayout, LifeTimingContract,
    };
    use crate::bitgrid::BitGrid;

    fn stable_pulse_fixture() -> CompiledLifeProgram {
        CompiledLifeProgram {
            initial_grid: BitGrid::from_cells(&[(8, 8), (9, 8), (8, 9), (9, 9)]),
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
                output_frame_generations: 36,
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
                sample_cadence_generations: 4,
                sample_phase_generations: 1,
            },
        }
    }

    #[test]
    fn stable_pulse_matches_scalar_life_session() {
        let program = stable_pulse_fixture();
        let mut scalar = LifeProgramSession::new(&program).or_invariant("valid scalar fixture");
        let mut hashlife =
            HashLifeProgramSession::new(&program).or_invariant("valid HashLife fixture");

        let scalar_samples = scalar
            .advance_to_generation(33, 33)
            .or_invariant("scalar evolution");
        let hashlife_samples = hashlife
            .advance_to_generation(33, 33)
            .or_invariant("HashLife evolution");
        let scalar_observations: Vec<_> = scalar_samples
            .into_iter()
            .map(|sample| (sample.generation(), sample.pulse(), sample.decoded_byte()))
            .collect();
        let hashlife_observations: Vec<_> = hashlife_samples
            .into_iter()
            .map(|sample| (sample.generation(), sample.pulse(), sample.decoded_byte()))
            .collect();

        assert_eq!(hashlife_observations, scalar_observations);
        assert_eq!(hashlife.generation(), scalar.generation());
        assert_eq!(
            hashlife_observations.last().and_then(|sample| sample.2),
            Some(0xff)
        );
    }

    #[test]
    fn rejected_budget_preserves_transactional_generation_accounting() {
        let program = stable_pulse_fixture();
        let mut session =
            HashLifeProgramSession::new(&program).or_invariant("valid HashLife fixture");

        assert_eq!(
            session.advance_to_generation(33, 32),
            Err(HashLifeProgramSessionError::GenerationBudgetExceeded {
                required: 33,
                budget: 32,
            })
        );
        assert_eq!(session.generation(), 0);

        let samples = session
            .advance_to_generation(5, 5)
            .or_invariant("accepted evolution");
        assert_eq!(session.generation(), 5);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].generation(), 1);
        assert_eq!(samples[1].generation(), 5);
    }

    #[test]
    fn output_observation_does_not_materialize_a_grid() {
        let program = stable_pulse_fixture();
        let mut session =
            HashLifeProgramSession::new(&program).or_invariant("valid HashLife fixture");
        let before = session.execution_stats().materializations;
        let samples = session
            .advance_to_generation(33, 33)
            .or_invariant("HashLife observation");
        assert_eq!(
            samples.last().and_then(|sample| sample.decoded_byte()),
            Some(0xff)
        );
        assert_eq!(session.execution_stats().materializations, before);
    }
}
