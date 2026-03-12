pub mod app;
pub mod benchmark;
pub mod bitgrid;
pub(crate) mod cache_policy;
pub mod classify;
pub mod cli;
pub mod engine;
pub mod generators;
pub mod hashlife;
pub mod life;
pub mod memo;
pub mod normalize;
pub mod oracle;
pub mod render;
pub(crate) mod symmetry;
pub mod term;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
