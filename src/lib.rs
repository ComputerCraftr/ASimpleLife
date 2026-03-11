pub mod app;
pub mod benchmark;
pub mod bitgrid;
pub mod classify;
pub mod cli;
pub mod generators;
pub mod life;
pub mod memo;
pub mod normalize;
pub mod render;
pub mod term;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
