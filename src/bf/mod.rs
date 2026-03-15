mod c_backend;
mod c_super_backend;
mod c_support;
mod cli;
mod ir;
mod ir_report;
mod life_assets;
mod life_backend;
mod life_macro_library;
mod lowered_ir;
mod optimizer;
mod summary;
mod symbolic;
mod tape;
#[cfg(test)]
mod tests;

pub(super) const BF_C_TAPE_LEN: usize = 30_000;
pub(super) const BF_LIFE_TAPE_LEN: usize = 64;

pub use c_backend::{emit_c, format_ir};
pub use c_super_backend::emit_c_super;
pub use cli::run;
pub use ir::{BfIr, BfOffset, Parser, ShiftDir};
pub use life_assets::{
    AssetBlocker, AssetBounds, AssetComponent, AssetManifest, AssetPattern, AssetPort,
    AssetRegistry, ComponentKind, Isolation, LifeAssetError, PatternFormat, PortDirection,
    PortRole, Provenance, REQUIRED_V1_COMPONENT_KINDS, VerificationRecord, VerifiedAssetRegistry,
};
#[cfg(test)]
pub(crate) use life_backend::compile_life_scaffold;
pub use life_backend::{
    BfLifeCircuit, BfLifeCircuitError, BfLifeCircuitState, CircuitPhase, HashLifeProgramSession,
    HashLifeProgramSessionError, LifePortSample, LifeProgramSession, LifeProgramSessionError,
    LifeProgramValidationError, MacroTimingSpec, PlacedLifeMachine, RailGroup, RoutedRail,
    compile_to_life_circuit, serialize_life_circuit, serialize_life_circuit_hashlife,
};
pub use life_macro_library::{
    LifeMacroInstance, LifeMacroKind, LifeMacroOrientation, LifeMacroPort, LifeMacroTemplate,
    instantiate_macro_cells, life_macro_template, life_macro_templates, transform_cell,
};
pub use lowered_ir::{PhysicalBfInstr, expand_distribute_to_primitive, lower_bf_control_flow};
pub use optimizer::{CellSign, CodegenOpts, IoMode, optimize_with_opts};
pub use summary::{
    DynamicLoopMetadata, LoopId, LoopSummary, SummaryEffect, SummaryGuard, SummaryProvenance,
};
