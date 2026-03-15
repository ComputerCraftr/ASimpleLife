use super::*;
#[cfg(test)]
use crate::hashlife::serialize_grid_snapshot;
#[cfg(test)]
use crate::persistence::serialize_life_grid;

pub fn serialize_life_circuit(program: &CompiledLifeProgram) -> Result<String, BfLifeCircuitError> {
    program.validate_physical_contract()?;
    Ok(crate::persistence::serialize_life_grid(
        program.initial_grid(),
    ))
}

pub fn serialize_life_circuit_hashlife(
    program: &CompiledLifeProgram,
) -> Result<String, BfLifeCircuitError> {
    program.validate_physical_contract()?;
    crate::hashlife::serialize_grid_snapshot(program.initial_grid())
        .map_err(|error| BfLifeCircuitError::Geometry(error.to_string()))
}

#[cfg(test)]
pub(crate) fn serialize_life_scaffold(circuit: &ReferenceLifeScaffold) -> String {
    serialize_life_grid(&circuit.compiled_grid())
}

#[cfg(test)]
pub(crate) fn serialize_life_scaffold_hashlife(circuit: &ReferenceLifeScaffold) -> String {
    serialize_grid_snapshot(&circuit.compiled_grid())
        .or_invariant("test Life scaffold should serialize as HashLife")
}
