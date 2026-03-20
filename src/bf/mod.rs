mod c_backend;
mod c_super_backend;
mod c_support;
mod cli;
mod ir;
mod life_backend;
mod life_macro_library;
mod lowered_ir;
mod optimizer;
#[cfg(test)]
mod tests;

pub use c_backend::{emit_c, format_ir};
pub use c_super_backend::emit_c_super;
pub use cli::run;
pub use ir::{BfIr, Parser};
pub use life_backend::{
    BfLifeCircuit, BfLifeCircuitError, BfLifeCircuitState, CircuitPhase, MacroTimingSpec,
    PlacedLifeMachine, RailGroup, RoutedRail, compile_to_life_circuit, serialize_life_circuit,
    serialize_life_circuit_hashlife,
};
pub use life_macro_library::{
    LifeMacroInstance, LifeMacroKind, LifeMacroOrientation, LifeMacroPort, LifeMacroTemplate,
    instantiate_macro_cells, life_macro_template, life_macro_templates, transform_cell,
};
pub use lowered_ir::{PhysicalBfInstr, expand_distribute_to_primitive, lower_bf_control_flow};
pub use optimizer::{CellSign, CodegenOpts, IoMode, optimize};
