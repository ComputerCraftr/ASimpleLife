use super::{GridExtractionError, HashLifeSnapshotError};

pub(super) const DEFAULT_SOFT_MEMORY_BYTES: u128 = 704 * 1024 * 1024;
pub(super) const DEFAULT_HARD_MEMORY_BYTES: u128 = 896 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionAdvanceStats {
    pub requested_generations: u64,
    pub completed_generations: u64,
    pub starting_generation: u64,
    pub reached_generation: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HashLifeExecutionStats {
    pub allocated_bytes: u128,
    pub nodes: usize,
    pub arena_epoch: u64,
    pub gc_runs: usize,
    pub dependency_stalls: usize,
    pub materializations: usize,
    pub candidate_lanes: usize,
    pub portable_vector_lanes: usize,
    pub vectorized_structural_lanes: usize,
    pub scalar_fallback_lanes: usize,
    pub native_avx2_lanes: usize,
    pub native_neon_lanes: usize,
    pub d4_candidate_lanes: usize,
    pub native_d4_candidate_lanes: usize,
    pub native_d4_prefix_compare_lanes: usize,
    pub native_d4_exact_winner_lanes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashLifeLimits {
    pub soft_memory_bytes: u128,
    pub hard_memory_bytes: u128,
}

impl Default for HashLifeLimits {
    fn default() -> Self {
        Self {
            soft_memory_bytes: DEFAULT_SOFT_MEMORY_BYTES,
            hard_memory_bytes: DEFAULT_HARD_MEMORY_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashLifeAdvanceError {
    NotLoaded {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        reached_generation: u64,
    },
    MemoryBudgetExceeded {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        requested_generation: u64,
        reached_generation: u64,
        allocated_bytes: u128,
        limit_bytes: u128,
    },
    GenerationOverflow {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        reached_generation: u64,
    },
    CoordinateRangeExceeded {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        reached_generation: u64,
        required_level: u32,
    },
    AllocationFailed {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        reached_generation: u64,
        requested_bytes: u128,
    },
    NodeIdExhausted {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        reached_generation: u64,
    },
    CanonicalReferenceExhausted {
        starting_generation: u64,
        requested_delta: u64,
        completed_generations: u64,
        reached_generation: u64,
    },
}

impl HashLifeAdvanceError {
    pub const fn completed_generations(self) -> u64 {
        match self {
            Self::NotLoaded {
                completed_generations,
                ..
            }
            | Self::MemoryBudgetExceeded {
                completed_generations,
                ..
            }
            | Self::GenerationOverflow {
                completed_generations,
                ..
            }
            | Self::CoordinateRangeExceeded {
                completed_generations,
                ..
            }
            | Self::AllocationFailed {
                completed_generations,
                ..
            }
            | Self::NodeIdExhausted {
                completed_generations,
                ..
            }
            | Self::CanonicalReferenceExhausted {
                completed_generations,
                ..
            } => completed_generations,
        }
    }

    pub const fn reached_generation(self) -> u64 {
        match self {
            Self::NotLoaded {
                reached_generation, ..
            }
            | Self::MemoryBudgetExceeded {
                reached_generation, ..
            }
            | Self::GenerationOverflow {
                reached_generation, ..
            }
            | Self::CoordinateRangeExceeded {
                reached_generation, ..
            }
            | Self::AllocationFailed {
                reached_generation, ..
            }
            | Self::NodeIdExhausted {
                reached_generation, ..
            }
            | Self::CanonicalReferenceExhausted {
                reached_generation, ..
            } => reached_generation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashLifeConversionError {
    MemoryBudgetExceeded {
        retained_bytes: u128,
        candidate_bytes: u128,
        limit_bytes: u128,
    },
    AllocationFailed {
        requested_bytes: u128,
    },
    NodeIdExhausted,
    CanonicalReferenceExhausted,
    CoordinateRangeExceeded {
        axis: &'static str,
    },
    Snapshot(HashLifeSnapshotError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HashLifeMaterializationError {
    Conversion(HashLifeConversionError),
    Extraction(GridExtractionError),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HashLifeCheckpointProfile {
    pub metadata_reads: usize,
    pub subtree_visits: usize,
}
