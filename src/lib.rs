#![deny(unsafe_code)]
#[cfg(all(
    not(doc),
    any(target_env = "musl", target_env = "msvc"),
    not(target_feature = "crt-static")
))]
compile_error!("musl and MSVC release targets require target-feature=+crt-static");

pub mod app;
pub mod benchmark;
pub mod bf;
pub mod bitgrid;
pub(crate) mod cache_policy;
pub mod classify;
pub mod cli;
pub mod engine;
pub mod generators;
pub(crate) mod hashing;
pub mod hashlife;
mod invariant;
pub mod life;
pub mod memo;
pub mod normalize;
pub mod oracle;
pub mod persistence;
#[allow(unsafe_code)]
pub(crate) mod probe_table;
pub mod recurrence;
pub mod render;
pub(crate) mod simd_layout;
pub(crate) mod symmetry;
pub mod term;
#[cfg(test)]
pub(crate) mod test_support;
pub mod tui;
pub(crate) mod wide_math;

pub use invariant::{RequiredErrorExt, RequiredExt, invariant_failure};

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
