use bytemuck::must_cast;
use wide::u64x8;

use crate::bitgrid::{Cell, Coord};
use crate::simd_layout::{AlignedU128Batch, SIMD_BATCH_LANES};

pub(crate) const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MURMUR3_FMIX_MUL1: u64 = 0xFF51_AFD7_ED55_8CCD;
const MURMUR3_FMIX_MUL2: u64 = 0xC4CE_B9FE_1A85_EC53;
const MURMUR3_FMIX_SHIFT: u32 = 33;
const STRUCTURAL_FP_LO_DOMAIN: u64 = 0x5354_5255_4354_4C4F;
const STRUCTURAL_FP_HI_DOMAIN: u64 = 0x5354_5255_4354_4849;
const STRUCTURAL_PROBE_DOMAIN: u64 = 0x5354_5255_4354_5052;
const SPLITMIX64_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX64_MUL2: u64 = 0x94D0_49BB_1331_11EB;
const MORTON_MASK_16: u64 = 0x0000_FFFF_0000_FFFF;
const MORTON_MASK_8: u64 = 0x00FF_00FF_00FF_00FF;
const MORTON_MASK_4: u64 = 0x0F0F_0F0F_0F0F_0F0F;
const MORTON_MASK_2: u64 = 0x3333_3333_3333_3333;
const MORTON_MASK_1: u64 = 0x5555_5555_5555_5555;
const MURMUR3_FMIX_MUL1_VEC: u64x8 = u64x8::splat(MURMUR3_FMIX_MUL1);
const MURMUR3_FMIX_MUL2_VEC: u64x8 = u64x8::splat(MURMUR3_FMIX_MUL2);
const SPLITMIX64_GAMMA_VEC: u64x8 = u64x8::splat(SPLITMIX64_GAMMA);
const MORTON_MASK_16_VEC: u64x8 = u64x8::splat(MORTON_MASK_16);
const MORTON_MASK_8_VEC: u64x8 = u64x8::splat(MORTON_MASK_8);
const MORTON_MASK_4_VEC: u64x8 = u64x8::splat(MORTON_MASK_4);
const MORTON_MASK_2_VEC: u64x8 = u64x8::splat(MORTON_MASK_2);
const MORTON_MASK_1_VEC: u64x8 = u64x8::splat(MORTON_MASK_1);

fn xor_shift_right(value: u64, shift: u32) -> u64 {
    value ^ (value >> shift)
}

fn xor_shift_right_batch(value: u64x8, shift: u32) -> u64x8 {
    value ^ (value >> shift)
}

/// MurmurHash3's 64-bit finalizer. Keep all structural hash avalanche here.
pub(crate) fn mix64(value: u64) -> u64 {
    let value = xor_shift_right(value, MURMUR3_FMIX_SHIFT).wrapping_mul(MURMUR3_FMIX_MUL1);
    let value = xor_shift_right(value, MURMUR3_FMIX_SHIFT).wrapping_mul(MURMUR3_FMIX_MUL2);
    xor_shift_right(value, MURMUR3_FMIX_SHIFT)
}

pub(crate) fn splitmix64_output(value: u64) -> u64 {
    let value = xor_shift_right(value, 30).wrapping_mul(SPLITMIX64_MUL1);
    let value = xor_shift_right(value, 27).wrapping_mul(SPLITMIX64_MUL2);
    xor_shift_right(value, 31)
}

pub(crate) fn mix64_batch(values: [u64; SIMD_BATCH_LANES]) -> [u64; SIMD_BATCH_LANES] {
    let value =
        xor_shift_right_batch(must_cast(values), MURMUR3_FMIX_SHIFT) * MURMUR3_FMIX_MUL1_VEC;
    let value = xor_shift_right_batch(value, MURMUR3_FMIX_SHIFT) * MURMUR3_FMIX_MUL2_VEC;
    must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>(xor_shift_right_batch(value, MURMUR3_FMIX_SHIFT))
}

#[inline]
fn fold_hash_word(state: u64, word: u64) -> u64 {
    state.wrapping_mul(SPLITMIX64_GAMMA) ^ word
}

pub(crate) fn hash_words(domain: u64, words: impl IntoIterator<Item = u64>) -> u64 {
    let mut state = domain;
    for word in words {
        state = fold_hash_word(state, word);
    }
    mix64(state)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StructuralFingerprint {
    lo: u64,
    hi: u64,
}

impl StructuralFingerprint {
    pub(crate) fn probe_hash(self) -> u64 {
        hash_words(STRUCTURAL_PROBE_DOMAIN, [self.lo, self.hi])
    }

    pub(crate) const fn words(self) -> [u64; 2] {
        [self.lo, self.hi]
    }
}

pub(crate) fn structural_leaf_fingerprint(alive: bool) -> StructuralFingerprint {
    structural_fingerprint(0, [StructuralFingerprint::default(); 4], u64::from(alive))
}

pub(crate) fn structural_node_fingerprint(
    level: u32,
    children: [StructuralFingerprint; 4],
) -> StructuralFingerprint {
    structural_fingerprint(level, children, 0)
}

fn structural_fingerprint(
    level: u32,
    children: [StructuralFingerprint; 4],
    leaf: u64,
) -> StructuralFingerprint {
    let words = std::iter::once(u64::from(level))
        .chain(std::iter::once(leaf))
        .chain(children.into_iter().flat_map(StructuralFingerprint::words));
    StructuralFingerprint {
        lo: hash_words(STRUCTURAL_FP_LO_DOMAIN, words.clone()),
        hi: hash_words(STRUCTURAL_FP_HI_DOMAIN, words),
    }
}

pub(crate) fn derive_seed(seed: u64, words: impl IntoIterator<Item = u64>) -> u64 {
    hash_words(seed ^ SPLITMIX64_GAMMA, words)
}

pub(crate) fn hash_u64_words_with_level(level: u32, words: [u64; 4]) -> u64 {
    hash_words(u64::from(level), words)
}

pub(crate) fn hash_leaf_population(population: u64) -> u64 {
    hash_u64_words_with_level(0, [population, 0, 0, 0])
}

pub(crate) fn hash_chunk_neighborhood_words(words: [u64; 9]) -> u64 {
    hash_words(9, words)
}

pub(crate) fn hash_packed_node_fingerprint(level: u32, children: [u64; 4]) -> u64 {
    hash_u64_words_with_level(level, children)
}

pub(crate) fn hash_packed_jump_fingerprint(packed_fingerprint: u64, step_exp: u32) -> u64 {
    hash_words(2, [packed_fingerprint, u64::from(step_exp)])
}

pub(crate) fn hash_chunk_coord_key(cx: Coord, cy: Coord) -> u64 {
    hash_words(2, [cx.cast_unsigned(), cy.cast_unsigned()])
}

pub(crate) fn hash_normalized_grid_signature(width: Coord, height: Coord, cells: &[Cell]) -> u64 {
    let dimensions = [width.cast_unsigned(), height.cast_unsigned()];
    let cell_words = cells
        .iter()
        .flat_map(|&(x, y)| [x.cast_unsigned(), y.cast_unsigned()]);
    hash_words(2, dimensions.into_iter().chain(cell_words))
}

pub(crate) fn hash_u64_words_with_level_batch(
    levels: [u32; SIMD_BATCH_LANES],
    words: [[u64; SIMD_BATCH_LANES]; 4],
) -> [u64; SIMD_BATCH_LANES] {
    let level_words = levels.map(u64::from);
    let mut value: u64x8 = must_cast(level_words);
    value = (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[0]);
    value = (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[1]);
    value = (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[2]);
    value = (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[3]);
    mix64_batch(must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>(value))
}

fn spread_bits_u32_to_u64_batch(values: [u64; SIMD_BATCH_LANES]) -> [u64; SIMD_BATCH_LANES] {
    let mut widened: u64x8 = must_cast(values);
    widened = (widened | (widened << 16)) & MORTON_MASK_16_VEC;
    widened = (widened | (widened << 8)) & MORTON_MASK_8_VEC;
    widened = (widened | (widened << 4)) & MORTON_MASK_4_VEC;
    widened = (widened | (widened << 2)) & MORTON_MASK_2_VEC;
    widened = (widened | (widened << 1)) & MORTON_MASK_1_VEC;
    must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>(widened)
}

pub(crate) fn morton_interleave_u64_batch(
    xs: [u64; SIMD_BATCH_LANES],
    ys: [u64; SIMD_BATCH_LANES],
) -> [u128; SIMD_BATCH_LANES] {
    let x_low = xs.map(|x| x & 0xFFFF_FFFF);
    let y_low = ys.map(|y| y & 0xFFFF_FFFF);
    let x_high = xs.map(|x| x >> 32);
    let y_high = ys.map(|y| y >> 32);
    let low = spread_bits_u32_to_u64_batch(x_low);
    let low_y = spread_bits_u32_to_u64_batch(y_low);
    let high = spread_bits_u32_to_u64_batch(x_high);
    let high_y = spread_bits_u32_to_u64_batch(y_high);
    let mut keys = AlignedU128Batch::default();
    let mut lane = 0;
    while lane < SIMD_BATCH_LANES {
        let low64 = low[lane] | (low_y[lane] << 1);
        let high64 = high[lane] | (high_y[lane] << 1);
        keys.0[lane] = u128::from(low64) | (u128::from(high64) << 64);
        lane += 1;
    }
    keys.0
}

#[cfg(test)]
mod tests {
    use crate::RequiredExt;

    #[cfg(not(miri))]
    use super::{SIMD_BATCH_LANES, mix64_batch};
    use super::{mix64, splitmix64_output};

    #[test]
    fn murmur3_fmix64_matches_reference_vectors() {
        assert_eq!(mix64(0), 0);
        assert_eq!(mix64(1), 0xb456_bcfc_34c2_cb2c);
    }

    #[test]
    fn splitmix64_output_preserves_generator_reference_sequence() {
        assert_eq!(splitmix64_output(0), 0);
        assert_eq!(splitmix64_output(1), 0x5692_161d_100b_05e5);
    }

    #[cfg(not(miri))]
    #[test]
    fn murmur3_fmix64_simd_matches_scalar() {
        let input = std::array::from_fn(|lane| {
            u64::try_from(lane)
                .or_invariant("SIMD lane exceeded u64")
                .wrapping_mul(0x0101_0101_0101_0101)
        });
        let expected = input.map(mix64);
        assert_eq!(mix64_batch(input), expected);
        assert_eq!(expected.len(), SIMD_BATCH_LANES);
    }

    #[test]
    fn murmur3_fmix64_spreads_adversarial_low_bit_grid_keys() {
        let mut low_buckets = [false; 256];
        let mut tag_buckets = [false; 128];
        for row in 0..64_u64 {
            for column in 0..64_u64 {
                let mixed = mix64((row << 32) | column);
                let low = usize::from(mixed.to_le_bytes()[0]);
                let tag = usize::try_from((mixed >> 57) & 0x7f)
                    .or_invariant("seven-bit hash tag exceeded usize");
                low_buckets[low] = true;
                tag_buckets[tag] = true;
            }
        }
        assert!(low_buckets.into_iter().filter(|used| *used).count() >= 250);
        assert!(tag_buckets.into_iter().filter(|used| *used).count() >= 126);
    }
}
