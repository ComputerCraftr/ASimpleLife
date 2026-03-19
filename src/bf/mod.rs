mod cli;
mod codegen;
mod ir;
mod optimizer;
#[cfg(test)]
mod tests;

pub use cli::run;
pub use codegen::{
    BfLifeEmitError, compile_to_life_grid, emit_c, format_ir, serialize_legacy_life_grid,
    serialize_life_grid,
};
pub use ir::{BfIr, Parser};
pub use optimizer::{CellSign, CodegenOpts, IoMode, optimize};
