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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::hashlife) struct D4CandidateBatch {
    /// Child identities after recursively applying each candidate symmetry,
    /// before the parent-quadrant permutation is applied.
    pub(in crate::hashlife) oriented_children: [[u32; 4]; 8],
    pub(in crate::hashlife) permutations: [[u8; 4]; 8],
    pub(in crate::hashlife) active_lanes: usize,
    /// Maps each packed lane back to the transform in the original D4 orbit.
    pub(in crate::hashlife) transforms: [D4Symmetry; 8],
}

impl Default for D4CandidateBatch {
    fn default() -> Self {
        Self {
            oriented_children: [[0; 4]; 8],
            permutations: [[0; 4]; 8],
            active_lanes: 8,
            transforms: D4Symmetry::ALL,
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::hashlife) struct D4PrefixBatch {
    pub(in crate::hashlife) words: [[u64; 8]; 2],
    pub(in crate::hashlife) complete: bool,
    pub(in crate::hashlife) active_lanes: usize,
    /// Maps each packed lane back to the transform in the original D4 orbit.
    pub(in crate::hashlife) transforms: [D4Symmetry; 8],
}

impl Default for D4PrefixBatch {
    fn default() -> Self {
        Self {
            words: [[0; 8]; 2],
            complete: false,
            active_lanes: 8,
            transforms: D4Symmetry::ALL,
        }
    }
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

pub(super) fn active_lane_mask(active_lanes: usize) -> u8 {
    debug_assert!(active_lanes <= 8, "D4 batches have at most eight lanes");
    if active_lanes == 0 {
        0
    } else {
        u8::MAX >> (8_usize.saturating_sub(active_lanes))
    }
}

pub(super) fn lowest_original_lane(packed_mask: u8, transforms: &[D4Symmetry; 8]) -> usize {
    let first = usize::try_from(packed_mask.trailing_zeros()).unwrap_or_default();
    let mut winner = first;
    for (lane, &transform) in transforms.iter().enumerate().skip(first + 1) {
        if packed_mask & (1 << lane) != 0 && transform < transforms[winner] {
            winner = lane;
        }
    }
    winner
}

pub(super) fn original_transform_mask(packed_mask: u8, transforms: &[D4Symmetry; 8]) -> u8 {
    transforms
        .iter()
        .enumerate()
        .fold(0_u8, |mask, (lane, &transform)| {
            mask | (u8::from(packed_mask & (1 << lane) != 0) << transform_index(transform))
        })
}

fn transform_index(transform: D4Symmetry) -> usize {
    transform as usize
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

#[cfg(test)]
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

pub(super) fn swar_control_matches(
    control: &[u8; 16],
    tags: &[u8; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u16; SIMD_BATCH_LANES] {
    crate::probe_table::match_control_groups_swar(control, tags, active_lanes)
}

pub(super) fn scalar_d4_prefix(batch: &D4PrefixBatch) -> D4PrefixDecision {
    if batch.active_lanes == 0 {
        return D4PrefixDecision {
            unresolved_mask: 0,
            ..D4PrefixDecision::default()
        };
    }

    let mut winner = 0;
    for candidate in 1..batch.active_lanes {
        let candidate_key = [batch.words[0][candidate], batch.words[1][candidate]];
        let winner_key = [batch.words[0][winner], batch.words[1][winner]];
        if candidate_key < winner_key
            || (candidate_key == winner_key
                && batch.transforms[candidate] < batch.transforms[winner])
        {
            winner = candidate;
        }
    }
    let mut packed_unresolved_mask = 0_u8;
    for candidate in 0..batch.active_lanes {
        if batch.words[0][candidate] == batch.words[0][winner]
            && batch.words[1][candidate] == batch.words[1][winner]
        {
            packed_unresolved_mask |= 1 << candidate;
        }
    }
    let unresolved_mask = original_transform_mask(packed_unresolved_mask, &batch.transforms);
    let transform = batch.transforms[winner];
    D4PrefixDecision {
        transform,
        inverse: transform.inverse(),
        unresolved_mask,
        exact: batch.complete || unresolved_mask.count_ones() == 1,
    }
}

pub(in crate::hashlife) fn scalar_d4_candidates(
    batch: &D4CandidateBatch,
) -> D4CandidateBatchResult {
    let mut result = D4CandidateBatchResult::default();
    for candidate in 0..batch.active_lanes {
        for output in 0..4 {
            result.children[candidate][output] = batch.oriented_children[candidate]
                [usize::from(batch.permutations[candidate][output])];
        }
    }
    result
}

pub(super) fn scalar_population(batch: &PopulationBatch) -> PopulationBatchResult {
    let mut result = PopulationBatchResult::default();
    for lane in 0..batch.active_lanes {
        let mut lo = 0_u64;
        let mut hi = 0_u64;
        let mut saturated = false;
        for child in 0..4 {
            saturated |= batch.saturated[child][lane] != 0;
            let (next_lo, carry) =
                crate::wide_math::add_u64_carry(lo, batch.lo[child][lane], false);
            let (next_hi, overflow) =
                crate::wide_math::add_u64_carry(hi, batch.hi[child][lane], carry);
            lo = next_lo;
            hi = next_hi;
            saturated |= overflow;
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
