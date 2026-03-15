pub mod app;
pub mod benchmark;
pub mod bf;
pub mod bitgrid;
pub(crate) mod cache_policy;
pub mod classify;
pub mod cli;
pub mod engine;
pub(crate) mod flat_table;
pub mod generators;
pub(crate) mod hashing;
pub mod hashlife;
pub mod life;
pub mod memo;
pub mod normalize;
pub mod oracle;
pub mod render;
pub(crate) mod simd_layout;
pub(crate) mod symmetry;
pub mod term;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
