#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use super::KernelAccounting;
#[cfg(target_arch = "aarch64")]
use super::contracts::{
    self, D4CandidateBatch, D4CandidateBatchResult, D4PrefixBatch, D4PrefixDecision, DedupBatch,
    FingerprintBatch, KernelOperation, PopulationBatch, PopulationBatchResult,
};
#[cfg(target_arch = "aarch64")]
use crate::simd_layout::SIMD_BATCH_LANES;

#[cfg(target_arch = "aarch64")]
pub(super) fn evaluate(
    populations: &[[u64; SIMD_BATCH_LANES]; 9],
    active_mask: u8,
    active_lanes: usize,
) -> ([u8; 4], KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let masks = unsafe { output_nonzero_masks(populations, active_mask) };
    (
        masks,
        KernelAccounting::neon(KernelOperation::OutputPresence, active_lanes),
    )
}

#[cfg(target_arch = "aarch64")]
pub(super) fn fingerprints(
    batch: &FingerprintBatch,
) -> ([u64; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { fingerprint_kernel(batch) };
    (
        result,
        KernelAccounting::neon(KernelOperation::Fingerprint, batch.active_lanes),
    )
}

#[cfg(target_arch = "aarch64")]
pub(super) fn control_matches(
    control: &[u8; 16],
    tags: &[u8; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> ([u16; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { control_match_kernel(control, tags, active_lanes) };
    let mut accounting = KernelAccounting::neon(KernelOperation::ControlMatch, active_lanes);
    accounting.native_neon_control_groups = 1;
    (result, accounting)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn d4_prefix(batch: &D4PrefixBatch) -> (D4PrefixDecision, KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { d4_prefix_kernel(batch) };
    let mut accounting =
        KernelAccounting::neon(KernelOperation::D4SemanticPrefix, batch.active_lanes);
    accounting.native_d4_prefix_compare_lanes = batch.active_lanes;
    accounting.native_d4_exact_winner_lanes = usize::from(result.exact);
    (result, accounting)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn d4_candidates(
    batch: &D4CandidateBatch,
) -> (D4CandidateBatchResult, KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { d4_candidate_kernel(batch) };
    let mut accounting = KernelAccounting::neon(KernelOperation::D4Candidate, batch.active_lanes);
    accounting.native_d4_candidate_lanes = batch.active_lanes;
    (result, accounting)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn population(batch: &PopulationBatch) -> (PopulationBatchResult, KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { population_kernel(batch) };
    (
        result,
        KernelAccounting::neon(KernelOperation::Population, batch.active_lanes),
    )
}

#[cfg(target_arch = "aarch64")]
pub(super) fn base_transition(
    neighborhoods: &[u16; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> ([u8; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { base_transition_kernel(neighborhoods, active_lanes) };
    (
        result,
        KernelAccounting::neon(KernelOperation::BaseTransition, active_lanes),
    )
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dedup(batch: &DedupBatch) -> ([u8; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: Advanced SIMD is mandatory for supported AArch64 targets.
    let result = unsafe { dedup_kernel(batch) };
    (
        result,
        KernelAccounting::neon(KernelOperation::Dedup, batch.active_lanes),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fingerprint_kernel(batch: &FingerprintBatch) -> [u64; SIMD_BATCH_LANES] {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
    const MUL1: u64 = 0xFF51_AFD7_ED55_8CCD;
    const MUL2: u64 = 0xC4CE_B9FE_1A85_EC53;
    let levels = batch.levels.map(u64::from);
    let mut output = [0; SIMD_BATCH_LANES];
    for offset in (0..batch.active_lanes).step_by(2) {
        // SAFETY: each two-lane load starts within the fixed eight-lane batch.
        let mut value = unsafe { vld1q_u64(levels[offset..].as_ptr()) };
        for words in &batch.words {
            // SAFETY: each input word has the same fixed eight-lane layout.
            value = veorq_u64(mullo_u64(value, vdupq_n_u64(GAMMA)), unsafe {
                vld1q_u64(words[offset..].as_ptr())
            });
        }
        value = mullo_u64(
            veorq_u64(value, vshrq_n_u64::<33>(value)),
            vdupq_n_u64(MUL1),
        );
        value = mullo_u64(
            veorq_u64(value, vshrq_n_u64::<33>(value)),
            vdupq_n_u64(MUL2),
        );
        value = veorq_u64(value, vshrq_n_u64::<33>(value));
        // SAFETY: the destination has two writable lanes from every loop offset.
        unsafe { vst1q_u64(output[offset..].as_mut_ptr(), value) };
    }
    output[batch.active_lanes..].fill(0);
    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn mullo_u64(left: uint64x2_t, right: uint64x2_t) -> uint64x2_t {
    let left_low = vmovn_u64(left);
    let right_low = vmovn_u64(right);
    let low = vmull_u32(left_low, right_low);
    let cross_left = vmull_u32(vshrn_n_u64::<32>(left), right_low);
    let cross_right = vmull_u32(left_low, vshrn_n_u64::<32>(right));
    vaddq_u64(low, vshlq_n_u64::<32>(vaddq_u64(cross_left, cross_right)))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn control_match_kernel(
    control: &[u8; 16],
    tags: &[u8; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u16; SIMD_BATCH_LANES] {
    // SAFETY: `control` contains exactly one complete 16-byte control group.
    let group = unsafe { vld1q_u8(control.as_ptr()) };
    let mut output = [0; SIMD_BATCH_LANES];
    for lane in 0..active_lanes {
        let equal = vceqq_u8(group, vdupq_n_u8(tags[lane]));
        let mut bytes = [0_u8; 16];
        // SAFETY: `bytes` is a complete writable 16-byte vector destination.
        unsafe { vst1q_u8(bytes.as_mut_ptr(), equal) };
        output[lane] = bytes
            .into_iter()
            .enumerate()
            .fold(0_u16, |mask, (slot, byte)| {
                mask | (u16::from(byte != 0) << slot)
            });
    }
    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn d4_prefix_kernel(batch: &D4PrefixBatch) -> D4PrefixDecision {
    let active_mask = contracts::active_lane_mask(batch.active_lanes);
    let mut winner = 0_usize;
    loop {
        let mut less_mask = 0_u8;
        for offset in (0..8).step_by(2) {
            // SAFETY: both prefix rows contain eight lanes and offsets advance by two.
            let first = unsafe { vld1q_u64(batch.words[0][offset..].as_ptr()) };
            // SAFETY: the second prefix row has the same fixed layout.
            let second = unsafe { vld1q_u64(batch.words[1][offset..].as_ptr()) };
            let less_first = vcltq_u64(first, vdupq_n_u64(batch.words[0][winner]));
            let equal_first = vceqq_u64(first, vdupq_n_u64(batch.words[0][winner]));
            let less_second = vcltq_u64(second, vdupq_n_u64(batch.words[1][winner]));
            let less = vorrq_u64(less_first, vandq_u64(equal_first, less_second));
            let mut lanes = [0_u64; 2];
            // SAFETY: `lanes` is exactly one writable NEON vector.
            unsafe { vst1q_u64(lanes.as_mut_ptr(), less) };
            less_mask |= u8::from(lanes[0] != 0) << offset;
            less_mask |= u8::from(lanes[1] != 0) << (offset + 1);
        }
        less_mask &= active_mask;
        if less_mask == 0 {
            break;
        }
        winner = contracts::lowest_original_lane(less_mask, &batch.transforms);
    }

    let mut unresolved_mask = 0_u8;
    for offset in (0..8).step_by(2) {
        // SAFETY: both prefix rows contain eight lanes and offsets advance by two.
        let first = unsafe { vld1q_u64(batch.words[0][offset..].as_ptr()) };
        // SAFETY: the second prefix row has the same fixed layout.
        let second = unsafe { vld1q_u64(batch.words[1][offset..].as_ptr()) };
        let equal = vandq_u64(
            vceqq_u64(first, vdupq_n_u64(batch.words[0][winner])),
            vceqq_u64(second, vdupq_n_u64(batch.words[1][winner])),
        );
        let mut lanes = [0_u64; 2];
        // SAFETY: `lanes` is exactly one writable NEON vector.
        unsafe { vst1q_u64(lanes.as_mut_ptr(), equal) };
        unresolved_mask |= u8::from(lanes[0] != 0) << offset;
        unresolved_mask |= u8::from(lanes[1] != 0) << (offset + 1);
    }
    let packed_unresolved_mask = unresolved_mask & active_mask;
    winner = contracts::lowest_original_lane(packed_unresolved_mask, &batch.transforms);
    let unresolved_mask =
        contracts::original_transform_mask(packed_unresolved_mask, &batch.transforms);
    let transform = batch.transforms[winner];
    D4PrefixDecision {
        transform,
        inverse: transform.inverse(),
        unresolved_mask,
        exact: batch.complete || unresolved_mask.count_ones() == 1,
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn d4_candidate_kernel(batch: &D4CandidateBatch) -> D4CandidateBatchResult {
    let mut output = D4CandidateBatchResult::default();
    for candidate in 0..batch.active_lanes {
        let permutation = d4_byte_permutation(batch.permutations[candidate]);
        // SAFETY: four `u32` children occupy exactly one complete 16-byte vector.
        let source = unsafe { vld1q_u8(batch.oriented_children[candidate].as_ptr().cast()) };
        // SAFETY: every permutation is a complete 16-byte table-index vector.
        let indices = unsafe { vld1q_u8(permutation.as_ptr()) };
        let oriented = vqtbl1q_u8(source, indices);
        // SAFETY: each candidate row contains exactly four writable `u32` children.
        unsafe { vst1q_u8(output.children[candidate].as_mut_ptr().cast(), oriented) };
    }
    output
}

#[cfg(target_arch = "aarch64")]
fn d4_byte_permutation(slots: [u8; 4]) -> [u8; 16] {
    const BYTE_INDEX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut output = [0_u8; 16];
    let mut output_slot = 0;
    while output_slot < 4 {
        let input = usize::from(slots[output_slot]) * 4;
        let mut byte = 0;
        while byte < 4 {
            output[output_slot * 4 + byte] = BYTE_INDEX[input + byte];
            byte += 1;
        }
        output_slot += 1;
    }
    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn population_kernel(batch: &PopulationBatch) -> PopulationBatchResult {
    let mut output = PopulationBatchResult::default();
    for offset in (0..batch.active_lanes).step_by(2) {
        let mut lo = vdupq_n_u64(0);
        let mut hi = vdupq_n_u64(0);
        let mut saturated = vdupq_n_u64(0);
        for child in 0..4 {
            // SAFETY: active offsets always leave two readable lanes in each child row.
            let child_lo = unsafe { vld1q_u64(batch.lo[child][offset..].as_ptr()) };
            // SAFETY: high-limb rows have the same fixed eight-lane layout.
            let child_hi = unsafe { vld1q_u64(batch.hi[child][offset..].as_ptr()) };
            let next_lo = vaddq_u64(lo, child_lo);
            let carry = vcltq_u64(next_lo, lo);
            let high_without_carry = vaddq_u64(hi, child_hi);
            let overflow_a = vcltq_u64(high_without_carry, hi);
            let next_hi = vaddq_u64(high_without_carry, vandq_u64(carry, vdupq_n_u64(1)));
            let overflow_b = vcltq_u64(next_hi, high_without_carry);
            saturated = vorrq_u64(
                saturated,
                vorrq_u64(
                    vcgtq_u64(
                        // SAFETY: saturation rows have the same fixed eight-lane layout.
                        unsafe { vld1q_u64(batch.saturated[child][offset..].as_ptr()) },
                        vdupq_n_u64(0),
                    ),
                    vorrq_u64(overflow_a, overflow_b),
                ),
            );
            lo = next_lo;
            hi = next_hi;
        }
        lo = vbslq_u64(saturated, vdupq_n_u64(u64::MAX), lo);
        hi = vbslq_u64(saturated, vdupq_n_u64(u64::MAX), hi);
        let mut saturated_words = [0_u64; 2];
        // SAFETY: all three destinations have at least two writable lanes.
        unsafe {
            vst1q_u64(output.lo[offset..].as_mut_ptr(), lo);
            vst1q_u64(output.hi[offset..].as_mut_ptr(), hi);
            vst1q_u64(saturated_words.as_mut_ptr(), saturated);
        }
        for (lane, &value) in saturated_words
            .iter()
            .enumerate()
            .take(2.min(batch.active_lanes - offset))
        {
            output.saturated[offset + lane] = value != 0;
        }
    }
    for lane in batch.active_lanes..SIMD_BATCH_LANES {
        output.lo[lane] = 0;
        output.hi[lane] = 0;
        output.saturated[lane] = false;
    }
    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn base_transition_kernel(
    neighborhoods: &[u16; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u8; SIMD_BATCH_LANES] {
    // SAFETY: eight `u16` neighborhoods occupy exactly one NEON vector.
    let boards = unsafe { vld1q_u16(neighborhoods.as_ptr()) };
    let one = vdupq_n_u16(1);
    let mut packed = vdupq_n_u16(0);
    for (output_bit, (x, y)) in [(1_i16, 1_i16), (2, 1), (1, 2), (2, 2)]
        .into_iter()
        .enumerate()
    {
        let mut count = vdupq_n_u16(0);
        for dy in -1_i16..=1 {
            for dx in -1_i16..=1 {
                if dx != 0 || dy != 0 {
                    let shift = -((y + dy) * 4 + x + dx);
                    count = vaddq_u16(count, vandq_u16(vshlq_u16(boards, vdupq_n_s16(shift)), one));
                }
            }
        }
        let center_shift = -(y * 4 + x);
        let alive = vandq_u16(vshlq_u16(boards, vdupq_n_s16(center_shift)), one);
        let survives = vorrq_u16(
            vceqq_u16(count, vdupq_n_u16(3)),
            vandq_u16(
                vceqq_u16(count, vdupq_n_u16(2)),
                vcgtq_u16(alive, vdupq_n_u16(0)),
            ),
        );
        packed = vorrq_u16(
            packed,
            vandq_u16(survives, vdupq_n_u16(1_u16 << output_bit)),
        );
    }
    let mut words = [0_u16; SIMD_BATCH_LANES];
    // SAFETY: `words` is exactly one writable eight-lane `u16` vector.
    unsafe { vst1q_u16(words.as_mut_ptr(), packed) };
    let mut output = [0_u8; SIMD_BATCH_LANES];
    for lane in 0..active_lanes {
        output[lane] = words[lane].to_le_bytes()[0];
    }
    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dedup_kernel(batch: &DedupBatch) -> [u8; SIMD_BATCH_LANES] {
    let fingerprints: [uint64x2_t; 4] = std::array::from_fn(|pair| {
        // SAFETY: each pair selects two lanes from the fixed eight-lane array.
        unsafe { vld1q_u64(batch.fingerprints[pair * 2..].as_ptr()) }
    });
    let words: [[uint64x2_t; 4]; 4] = batch.words.map(|values| {
        std::array::from_fn(|pair| {
            // SAFETY: every word row has the same fixed eight-lane layout.
            unsafe { vld1q_u64(values[pair * 2..].as_ptr()) }
        })
    });
    let mut output = [u8::MAX; SIMD_BATCH_LANES];
    for lane in 1..batch.active_lanes {
        let mut matches = [0_u64; SIMD_BATCH_LANES];
        for pair in 0..4 {
            let mut equal = vceqq_u64(fingerprints[pair], vdupq_n_u64(batch.fingerprints[lane]));
            for (word, word_pairs) in words.iter().enumerate() {
                equal = vandq_u64(
                    equal,
                    vceqq_u64(word_pairs[pair], vdupq_n_u64(batch.words[word][lane])),
                );
            }
            // SAFETY: each pair selects two writable lanes in `matches`.
            unsafe { vst1q_u64(matches[pair * 2..].as_mut_ptr(), equal) };
        }
        if let Some(previous) = matches[..lane].iter().position(|&matched| matched != 0) {
            output[lane] = u8::try_from(previous).unwrap_or(u8::MAX);
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn output_nonzero_masks(
    populations: &[[u64; SIMD_BATCH_LANES]; 9],
    active_mask: u8,
) -> [u8; 4] {
    const SOURCES: [[usize; 4]; 4] = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]];
    let mut result = [0_u8; 4];
    for (output, sources) in SOURCES.into_iter().enumerate() {
        let mut mask = 0_u8;
        for pair in 0..4 {
            let offset = pair * 2;
            // SAFETY: each population row contains eight lanes and offsets advance by two.
            let mut combined = unsafe { vld1q_u64(populations[sources[0]][offset..].as_ptr()) };
            for source in &sources[1..] {
                // SAFETY: every selected population row has the same fixed layout.
                let values = unsafe { vld1q_u64(populations[*source][offset..].as_ptr()) };
                combined = vorrq_u64(combined, values);
            }
            mask |= u8::from(vgetq_lane_u64(combined, 0) != 0) << offset;
            mask |= u8::from(vgetq_lane_u64(combined, 1) != 0) << (offset + 1);
        }
        result[output] = mask & active_mask;
    }
    result
}
