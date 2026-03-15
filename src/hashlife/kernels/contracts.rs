use crate::symmetry::D4Symmetry;

use super::SIMD_BATCH_LANES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::hashlife) enum KernelOperation {
    OutputPresence,
    Fingerprint,
    ControlMatch,
    D4Candidate,
    D4SemanticPrefix,
    Population,
    BaseTransition,
    Dedup,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct D4CandidateBatch {
    pub(in crate::hashlife) children: [u32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct D4CandidateBatchResult {
    pub(in crate::hashlife) children: [[u32; 4]; 8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct FingerprintBatch {
    pub(in crate::hashlife) levels: [u32; SIMD_BATCH_LANES],
    pub(in crate::hashlife) words: [[u64; SIMD_BATCH_LANES]; 4],
    pub(in crate::hashlife) active_lanes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct D4PrefixBatch {
    pub(in crate::hashlife) words: [[u64; 8]; 2],
    pub(in crate::hashlife) complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::hashlife) struct D4PrefixDecision {
    pub(in crate::hashlife) transform: D4Symmetry,
    pub(in crate::hashlife) inverse: D4Symmetry,
    pub(in crate::hashlife) unresolved_mask: u8,
    pub(in crate::hashlife) exact: bool,
}

impl Default for D4PrefixDecision {
    fn default() -> Self {
        Self {
            transform: D4Symmetry::Identity,
            inverse: D4Symmetry::Identity,
            unresolved_mask: 1,
            exact: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct PopulationBatch {
    pub(in crate::hashlife) lo: [[u64; SIMD_BATCH_LANES]; 4],
    pub(in crate::hashlife) hi: [[u64; SIMD_BATCH_LANES]; 4],
    pub(in crate::hashlife) saturated: [[u64; SIMD_BATCH_LANES]; 4],
    pub(in crate::hashlife) active_lanes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct PopulationBatchResult {
    pub(in crate::hashlife) lo: [u64; SIMD_BATCH_LANES],
    pub(in crate::hashlife) hi: [u64; SIMD_BATCH_LANES],
    pub(in crate::hashlife) saturated: [bool; SIMD_BATCH_LANES],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::hashlife) struct DedupBatch {
    pub(in crate::hashlife) fingerprints: [u64; SIMD_BATCH_LANES],
    pub(in crate::hashlife) words: [[u64; SIMD_BATCH_LANES]; 4],
    pub(in crate::hashlife) active_lanes: usize,
}

pub(super) fn symmetry_from_index(index: usize) -> D4Symmetry {
    D4Symmetry::ALL[index]
}

pub(super) fn scalar_fingerprints(batch: &FingerprintBatch) -> [u64; SIMD_BATCH_LANES] {
    let mut result = [0; SIMD_BATCH_LANES];
    for (lane, output) in result[..batch.active_lanes].iter_mut().enumerate() {
        *output = crate::hashing::hash_u64_words_with_level(
            batch.levels[lane],
            batch.words.map(|words| words[lane]),
        );
    }
    result
}

pub(super) fn scalar_control_matches(
    control: &[u8; 16],
    tags: &[u8; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u16; SIMD_BATCH_LANES] {
    let mut result = [0; SIMD_BATCH_LANES];
    for (lane, output) in result[..active_lanes].iter_mut().enumerate() {
        for (slot, &value) in control.iter().enumerate() {
            *output |= u16::from(value == tags[lane]) << slot;
        }
    }
    result
}

pub(super) fn scalar_d4_prefix(batch: &D4PrefixBatch) -> D4PrefixDecision {
    let mut winner = 0;
    for candidate in 1..8 {
        if [batch.words[0][candidate], batch.words[1][candidate]]
            < [batch.words[0][winner], batch.words[1][winner]]
        {
            winner = candidate;
        }
    }
    let mut unresolved_mask = 0_u8;
    for candidate in 0..8 {
        if batch.words[0][candidate] == batch.words[0][winner]
            && batch.words[1][candidate] == batch.words[1][winner]
        {
            unresolved_mask |= 1 << candidate;
        }
    }
    let transform = symmetry_from_index(winner);
    D4PrefixDecision {
        transform,
        inverse: transform.inverse(),
        unresolved_mask,
        exact: batch.complete || unresolved_mask.count_ones() == 1,
    }
}

pub(super) fn scalar_d4_candidates(batch: &D4CandidateBatch) -> D4CandidateBatchResult {
    D4CandidateBatchResult {
        children: D4Symmetry::ALL
            .map(|symmetry| symmetry.quadrant_perm().map(|index| batch.children[index])),
    }
}

pub(super) fn scalar_population(batch: &PopulationBatch) -> PopulationBatchResult {
    let mut result = PopulationBatchResult::default();
    for lane in 0..batch.active_lanes {
        let mut lo = 0_u64;
        let mut hi = 0_u64;
        let mut saturated = false;
        for child in 0..4 {
            saturated |= batch.saturated[child][lane] != 0;
            let (next_lo, carry) = lo.overflowing_add(batch.lo[child][lane]);
            let (next_hi, overflow_a) = hi.overflowing_add(batch.hi[child][lane]);
            let (next_hi, overflow_b) = next_hi.overflowing_add(u64::from(carry));
            lo = next_lo;
            hi = next_hi;
            saturated |= overflow_a | overflow_b;
        }
        result.saturated[lane] = saturated;
        result.lo[lane] = if saturated { u64::MAX } else { lo };
        result.hi[lane] = if saturated { u64::MAX } else { hi };
    }
    result
}

pub(super) fn scalar_base_transition(
    neighborhoods: &[u16; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u8; SIMD_BATCH_LANES] {
    let mut result = [0; SIMD_BATCH_LANES];
    for (lane, output) in result[..active_lanes].iter_mut().enumerate() {
        let board = neighborhoods[lane];
        for (output_bit, (x, y)) in [(1_u32, 1_u32), (2, 1), (1, 2), (2, 2)]
            .into_iter()
            .enumerate()
        {
            let mut neighbors = 0_u8;
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let bit = (i32::try_from(y).unwrap_or_default() + dy) * 4
                        + i32::try_from(x).unwrap_or_default()
                        + dx;
                    neighbors += u8::from((board & (1_u16 << bit)) != 0);
                }
            }
            let alive = (board & (1_u16 << (y * 4 + x))) != 0;
            *output |= u8::from(neighbors == 3 || (alive && neighbors == 2)) << output_bit;
        }
    }
    result
}

pub(super) fn scalar_dedup(batch: &DedupBatch) -> [u8; SIMD_BATCH_LANES] {
    let mut result = [u8::MAX; SIMD_BATCH_LANES];
    for lane in 0..batch.active_lanes {
        for previous in 0..lane {
            if batch.fingerprints[lane] == batch.fingerprints[previous]
                && batch
                    .words
                    .iter()
                    .all(|words| words[lane] == words[previous])
            {
                result[lane] = u8::try_from(previous).unwrap_or(u8::MAX);
                break;
            }
        }
    }
    result
}
