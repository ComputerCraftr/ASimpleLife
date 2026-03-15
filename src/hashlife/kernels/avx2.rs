#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::simd_layout::SIMD_BATCH_LANES;

#[cfg(target_arch = "x86_64")]
use super::KernelAccounting;
#[cfg(target_arch = "x86_64")]
use super::contracts::{
    self, D4CandidateBatch, D4CandidateBatchResult, D4PrefixBatch, D4PrefixDecision, DedupBatch,
    FingerprintBatch, KernelOperation, PopulationBatch, PopulationBatchResult,
};

#[cfg(target_arch = "x86_64")]
pub(super) fn evaluate(
    populations: &[[u64; SIMD_BATCH_LANES]; 9],
    active_mask: u8,
    active_lanes: usize,
) -> ([u8; 4], KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let masks = unsafe { output_nonzero_masks(populations, active_mask) };
    (
        masks,
        KernelAccounting::avx2(KernelOperation::OutputPresence, active_lanes),
    )
}

#[cfg(target_arch = "x86_64")]
pub(super) fn fingerprints(
    batch: &FingerprintBatch,
) -> ([u64; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { fingerprint_kernel(batch) };
    (
        result,
        KernelAccounting::avx2(KernelOperation::Fingerprint, batch.active_lanes),
    )
}

#[cfg(target_arch = "x86_64")]
pub(super) fn control_matches(
    control: &[u8; 16],
    tags: &[u8; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> ([u16; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { control_match_kernel(control, tags, active_lanes) };
    (
        result,
        KernelAccounting::avx2(KernelOperation::ControlMatch, active_lanes),
    )
}

#[cfg(target_arch = "x86_64")]
pub(super) fn d4_prefix(batch: &D4PrefixBatch) -> (D4PrefixDecision, KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { d4_prefix_kernel(batch) };
    let mut accounting = KernelAccounting::avx2(KernelOperation::D4SemanticPrefix, 8);
    accounting.native_d4_prefix_compare_lanes = 8;
    accounting.native_d4_exact_winner_lanes = usize::from(result.exact);
    (result, accounting)
}

#[cfg(target_arch = "x86_64")]
pub(super) fn d4_candidates(
    batch: &D4CandidateBatch,
) -> (D4CandidateBatchResult, KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { d4_candidate_kernel(batch) };
    let mut accounting = KernelAccounting::avx2(KernelOperation::D4Candidate, 8);
    accounting.native_d4_candidate_lanes = 8;
    (result, accounting)
}

#[cfg(target_arch = "x86_64")]
pub(super) fn population(batch: &PopulationBatch) -> (PopulationBatchResult, KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { population_kernel(batch) };
    (
        result,
        KernelAccounting::avx2(KernelOperation::Population, batch.active_lanes),
    )
}

#[cfg(target_arch = "x86_64")]
pub(super) fn base_transition(
    neighborhoods: &[u16; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> ([u8; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { base_transition_kernel(neighborhoods, active_lanes) };
    (
        result,
        KernelAccounting::avx2(KernelOperation::BaseTransition, active_lanes),
    )
}

#[cfg(target_arch = "x86_64")]
pub(super) fn dedup(batch: &DedupBatch) -> ([u8; SIMD_BATCH_LANES], KernelAccounting) {
    // SAFETY: the caller reaches this module only after AVX2 runtime detection.
    let result = unsafe { dedup_kernel(batch) };
    (
        result,
        KernelAccounting::avx2(KernelOperation::Dedup, batch.active_lanes),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fingerprint_kernel(batch: &FingerprintBatch) -> [u64; SIMD_BATCH_LANES] {
    const GAMMA: i64 = i64::from_ne_bytes(0x9E37_79B9_7F4A_7C15_u64.to_ne_bytes());
    const MUL1: i64 = i64::from_ne_bytes(0xFF51_AFD7_ED55_8CCD_u64.to_ne_bytes());
    const MUL2: i64 = i64::from_ne_bytes(0xC4CE_B9FE_1A85_EC53_u64.to_ne_bytes());
    let levels = batch.levels.map(u64::from);
    let mut output = [0_u64; SIMD_BATCH_LANES];
    for offset in (0..SIMD_BATCH_LANES).step_by(4) {
        // SAFETY: each four-lane load starts within the fixed eight-lane batch.
        let mut value = unsafe { _mm256_loadu_si256(levels[offset..].as_ptr().cast()) };
        for words in &batch.words {
            // SAFETY: each input word has the same fixed eight-lane layout.
            value = _mm256_xor_si256(mullo_epi64(value, _mm256_set1_epi64x(GAMMA)), unsafe {
                _mm256_loadu_si256(words[offset..].as_ptr().cast())
            });
        }
        value = mullo_epi64(
            _mm256_xor_si256(value, _mm256_srli_epi64::<33>(value)),
            _mm256_set1_epi64x(MUL1),
        );
        value = mullo_epi64(
            _mm256_xor_si256(value, _mm256_srli_epi64::<33>(value)),
            _mm256_set1_epi64x(MUL2),
        );
        value = _mm256_xor_si256(value, _mm256_srli_epi64::<33>(value));
        // SAFETY: the destination has four writable lanes from every loop offset.
        unsafe { _mm256_storeu_si256(output[offset..].as_mut_ptr().cast(), value) };
    }
    output[batch.active_lanes..].fill(0);
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn mullo_epi64(left: __m256i, right: __m256i) -> __m256i {
    let low = _mm256_mul_epu32(left, right);
    let cross_left = _mm256_mul_epu32(_mm256_srli_epi64::<32>(left), right);
    let cross_right = _mm256_mul_epu32(left, _mm256_srli_epi64::<32>(right));
    _mm256_add_epi64(
        low,
        _mm256_slli_epi64::<32>(_mm256_add_epi64(cross_left, cross_right)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn control_match_kernel(
    control: &[u8; 16],
    tags: &[u8; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u16; SIMD_BATCH_LANES] {
    // SAFETY: `control` contains exactly one complete 16-byte control group.
    let group = unsafe { _mm_loadu_si128(control.as_ptr().cast()) };
    let mut output = [0; SIMD_BATCH_LANES];
    for lane in 0..active_lanes {
        let equal = _mm_cmpeq_epi8(group, _mm_set1_epi8(i8::from_ne_bytes([tags[lane]])));
        output[lane] = u16::try_from(_mm_movemask_epi8(equal)).unwrap_or(u16::MAX);
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn d4_prefix_kernel(batch: &D4PrefixBatch) -> D4PrefixDecision {
    let sign = _mm256_set1_epi64x(i64::MIN);
    let mut winner = 0_usize;
    loop {
        let mut less_mask = 0_u8;
        for offset in [0_usize, 4] {
            // SAFETY: both prefix rows contain eight lanes and offsets advance by four.
            let first = unsafe { _mm256_loadu_si256(batch.words[0][offset..].as_ptr().cast()) };
            // SAFETY: the second prefix row has the same fixed layout.
            let second = unsafe { _mm256_loadu_si256(batch.words[1][offset..].as_ptr().cast()) };
            let best_first =
                _mm256_set1_epi64x(i64::from_ne_bytes(batch.words[0][winner].to_ne_bytes()));
            let best_second =
                _mm256_set1_epi64x(i64::from_ne_bytes(batch.words[1][winner].to_ne_bytes()));
            let less_first = _mm256_cmpgt_epi64(
                _mm256_xor_si256(best_first, sign),
                _mm256_xor_si256(first, sign),
            );
            let equal_first = _mm256_cmpeq_epi64(first, best_first);
            let less_second = _mm256_cmpgt_epi64(
                _mm256_xor_si256(best_second, sign),
                _mm256_xor_si256(second, sign),
            );
            let less = _mm256_or_si256(less_first, _mm256_and_si256(equal_first, less_second));
            let bits =
                u8::try_from(_mm256_movemask_pd(_mm256_castsi256_pd(less))).unwrap_or_default();
            less_mask |= bits << offset;
        }
        if less_mask == 0 {
            break;
        }
        winner = usize::try_from(less_mask.trailing_zeros()).unwrap_or_default();
    }

    let mut unresolved_mask = 0_u8;
    for offset in [0_usize, 4] {
        // SAFETY: both prefix rows contain eight lanes and offsets advance by four.
        let first = unsafe { _mm256_loadu_si256(batch.words[0][offset..].as_ptr().cast()) };
        // SAFETY: the second prefix row has the same fixed layout.
        let second = unsafe { _mm256_loadu_si256(batch.words[1][offset..].as_ptr().cast()) };
        let best_first =
            _mm256_set1_epi64x(i64::from_ne_bytes(batch.words[0][winner].to_ne_bytes()));
        let best_second =
            _mm256_set1_epi64x(i64::from_ne_bytes(batch.words[1][winner].to_ne_bytes()));
        let equal = _mm256_and_si256(
            _mm256_cmpeq_epi64(first, best_first),
            _mm256_cmpeq_epi64(second, best_second),
        );
        let bits = u8::try_from(_mm256_movemask_pd(_mm256_castsi256_pd(equal))).unwrap_or_default();
        unresolved_mask |= bits << offset;
    }
    let transform = contracts::symmetry_from_index(winner);
    D4PrefixDecision {
        transform,
        inverse: transform.inverse(),
        unresolved_mask,
        exact: batch.complete || unresolved_mask.count_ones() == 1,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn d4_candidate_kernel(batch: &D4CandidateBatch) -> D4CandidateBatchResult {
    const PERMUTATIONS: [[i32; 8]; 4] = [
        [0, 1, 2, 3, 1, 3, 0, 2],
        [3, 2, 1, 0, 2, 0, 3, 1],
        [1, 0, 3, 2, 3, 1, 2, 0],
        [2, 3, 0, 1, 0, 2, 1, 3],
    ];
    let mut output = D4CandidateBatchResult::default();
    for (pair, indices) in PERMUTATIONS.iter().enumerate() {
        // SAFETY: every permutation is one complete eight-lane `i32` vector.
        let permutation = unsafe { _mm256_loadu_si256(indices.as_ptr().cast()) };
        // SAFETY: every index is in 0..4 and addresses the fixed child array.
        let candidates = unsafe {
            _mm256_i32gather_epi32::<4>(batch.children.as_ptr().cast::<i32>(), permutation)
        };
        // SAFETY: each pair owns eight writable `u32` values across two candidate rows.
        unsafe {
            _mm256_storeu_si256(output.children[pair * 2..].as_mut_ptr().cast(), candidates);
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn population_kernel(batch: &PopulationBatch) -> PopulationBatchResult {
    let mut output = PopulationBatchResult::default();
    let sign = _mm256_set1_epi64x(i64::MIN);
    let zero = _mm256_setzero_si256();
    let all_ones = _mm256_set1_epi64x(-1);
    for offset in (0..SIMD_BATCH_LANES).step_by(4) {
        let mut lo = zero;
        let mut hi = zero;
        let mut saturated = zero;
        for child in 0..4 {
            // SAFETY: active offsets always leave four readable lanes in each child row.
            let child_lo = unsafe { _mm256_loadu_si256(batch.lo[child][offset..].as_ptr().cast()) };
            // SAFETY: high-limb rows have the same fixed eight-lane layout.
            let child_hi = unsafe { _mm256_loadu_si256(batch.hi[child][offset..].as_ptr().cast()) };
            let next_lo = _mm256_add_epi64(lo, child_lo);
            let carry =
                _mm256_cmpgt_epi64(_mm256_xor_si256(lo, sign), _mm256_xor_si256(next_lo, sign));
            let high_without_carry = _mm256_add_epi64(hi, child_hi);
            let overflow_a = _mm256_cmpgt_epi64(
                _mm256_xor_si256(hi, sign),
                _mm256_xor_si256(high_without_carry, sign),
            );
            let next_hi = _mm256_sub_epi64(high_without_carry, carry);
            let overflow_b = _mm256_cmpgt_epi64(
                _mm256_xor_si256(high_without_carry, sign),
                _mm256_xor_si256(next_hi, sign),
            );
            let child_saturated =
                // SAFETY: saturation rows have the same fixed eight-lane layout.
                unsafe { _mm256_loadu_si256(batch.saturated[child][offset..].as_ptr().cast()) };
            let child_saturated =
                _mm256_xor_si256(_mm256_cmpeq_epi64(child_saturated, zero), all_ones);
            saturated = _mm256_or_si256(
                saturated,
                _mm256_or_si256(child_saturated, _mm256_or_si256(overflow_a, overflow_b)),
            );
            lo = next_lo;
            hi = next_hi;
        }
        lo = _mm256_or_si256(lo, saturated);
        hi = _mm256_or_si256(hi, saturated);
        let mut saturated_words = [0_u64; 4];
        // SAFETY: all three destinations have at least four writable lanes.
        unsafe {
            _mm256_storeu_si256(output.lo[offset..].as_mut_ptr().cast(), lo);
            _mm256_storeu_si256(output.hi[offset..].as_mut_ptr().cast(), hi);
            _mm256_storeu_si256(saturated_words.as_mut_ptr().cast(), saturated);
        }
        for (lane, &value) in saturated_words
            .iter()
            .enumerate()
            .take(4.min(batch.active_lanes.saturating_sub(offset)))
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn base_transition_kernel(
    neighborhoods: &[u16; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> [u8; SIMD_BATCH_LANES] {
    // SAFETY: eight `u16` neighborhoods occupy exactly one 128-bit vector.
    let packed_boards = unsafe { _mm_loadu_si128(neighborhoods.as_ptr().cast()) };
    let boards = _mm256_cvtepu16_epi32(packed_boards);
    let one = _mm256_set1_epi32(1);
    let mut packed = _mm256_setzero_si256();
    for (output_bit, (x, y)) in [(1_i32, 1_i32), (2, 1), (1, 2), (2, 2)]
        .into_iter()
        .enumerate()
    {
        let mut count = _mm256_setzero_si256();
        for dy in -1_i32..=1 {
            for dx in -1_i32..=1 {
                if dx != 0 || dy != 0 {
                    let shifts = _mm256_set1_epi32((y + dy) * 4 + x + dx);
                    count = _mm256_add_epi32(
                        count,
                        _mm256_and_si256(_mm256_srlv_epi32(boards, shifts), one),
                    );
                }
            }
        }
        let alive = _mm256_and_si256(_mm256_srlv_epi32(boards, _mm256_set1_epi32(y * 4 + x)), one);
        let is_three = _mm256_cmpeq_epi32(count, _mm256_set1_epi32(3));
        let is_two_alive = _mm256_and_si256(
            _mm256_cmpeq_epi32(count, _mm256_set1_epi32(2)),
            _mm256_cmpeq_epi32(alive, one),
        );
        let survives = _mm256_or_si256(is_three, is_two_alive);
        packed = _mm256_or_si256(
            packed,
            _mm256_and_si256(survives, _mm256_set1_epi32(1 << output_bit)),
        );
    }
    let mut words = [0_u32; SIMD_BATCH_LANES];
    // SAFETY: `words` is exactly one writable eight-lane `u32` vector.
    unsafe { _mm256_storeu_si256(words.as_mut_ptr().cast(), packed) };
    let mut output = [0_u8; SIMD_BATCH_LANES];
    for lane in 0..active_lanes {
        output[lane] = words[lane].to_le_bytes()[0];
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dedup_kernel(batch: &DedupBatch) -> [u8; SIMD_BATCH_LANES] {
    let fingerprints = [
        // SAFETY: each load selects four lanes from the fixed eight-lane array.
        unsafe { _mm256_loadu_si256(batch.fingerprints.as_ptr().cast()) },
        // SAFETY: the second load starts at lane four and reads the remaining lanes.
        unsafe { _mm256_loadu_si256(batch.fingerprints[4..].as_ptr().cast()) },
    ];
    let words = batch.words.map(|values| {
        [
            // SAFETY: every word row has the same fixed eight-lane layout.
            unsafe { _mm256_loadu_si256(values.as_ptr().cast()) },
            // SAFETY: the second load starts at lane four and reads the remaining lanes.
            unsafe { _mm256_loadu_si256(values[4..].as_ptr().cast()) },
        ]
    });
    let mut output = [u8::MAX; SIMD_BATCH_LANES];
    for lane in 1..batch.active_lanes {
        let mut matches = [0_u64; SIMD_BATCH_LANES];
        for half in 0..2 {
            let mut equal = _mm256_cmpeq_epi64(
                fingerprints[half],
                _mm256_set1_epi64x(i64::from_ne_bytes(batch.fingerprints[lane].to_ne_bytes())),
            );
            for (word, word_halves) in words.iter().enumerate() {
                equal = _mm256_and_si256(
                    equal,
                    _mm256_cmpeq_epi64(
                        word_halves[half],
                        _mm256_set1_epi64x(i64::from_ne_bytes(
                            batch.words[word][lane].to_ne_bytes(),
                        )),
                    ),
                );
            }
            // SAFETY: each half selects four writable lanes in `matches`.
            unsafe { _mm256_storeu_si256(matches[half * 4..].as_mut_ptr().cast(), equal) };
        }
        if let Some(previous) = matches[..lane].iter().position(|&matched| matched != 0) {
            output[lane] = u8::try_from(previous).unwrap_or(u8::MAX);
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn output_nonzero_masks(
    populations: &[[u64; SIMD_BATCH_LANES]; 9],
    active_mask: u8,
) -> [u8; 4] {
    const SOURCES: [[usize; 4]; 4] = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]];
    let mut result = [0_u8; 4];
    for (output, sources) in SOURCES.into_iter().enumerate() {
        let mut mask = 0_u8;
        for half in 0..2 {
            let offset = half * 4;
            // SAFETY: each population row contains eight lanes and offsets advance by four.
            let mut combined =
                unsafe { _mm256_loadu_si256(populations[sources[0]][offset..].as_ptr().cast()) };
            for source in &sources[1..] {
                // SAFETY: every selected population row has the same fixed layout.
                let values =
                    unsafe { _mm256_loadu_si256(populations[*source][offset..].as_ptr().cast()) };
                combined = _mm256_or_si256(combined, values);
            }
            let zeros = _mm256_cmpeq_epi64(combined, _mm256_setzero_si256());
            let nonzero =
                u8::try_from(!_mm256_movemask_pd(_mm256_castsi256_pd(zeros)) & 0x0f).unwrap_or(0);
            mask |= nonzero << offset;
        }
        result[output] = mask & active_mask;
    }
    result
}
