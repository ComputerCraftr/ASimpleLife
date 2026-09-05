//! Shared exact recurrence detection.
//!
//! Fingerprints select candidate buckets only. A certificate is issued only
//! after exact normalized packed chunks or weak DAG identity compare equal.
//! The tracker owns witnesses but DAG witnesses contain numeric identities,
//! never arena references, so recurrence tracking cannot extend node liveness.
//! Powering computes a checked proposal; applying it remains the caller's
//! atomic commit responsibility.
//! `MAX_RECURRENCE_BYTES` caps retained tracker evidence, including retained
//! table and vector capacity. A candidate observation remains caller-owned
//! transient memory and is separately bounded by the cell and chunk limits.

mod tracker;
mod witness;

pub use tracker::{
    ExactRecurrenceTracker, ObserveOutcome, PeriodicCertificate, RecurrenceSkip, TrackerCounters,
};
pub use witness::{DagWitness, ExactWitness, Lineage, Observation, RecurrenceUnavailable};

pub const MAX_RECURRENCE_ENTRIES: usize = 4_096;
pub const MAX_RECURRENCE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_WITNESS_CELLS: usize = 4_096;
pub const MAX_WITNESS_CHUNKS: usize = 1_024;

#[cfg(test)]
mod tests;
