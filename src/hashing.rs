use bytemuck::must_cast;
use wide::u64x8;

use crate::bitgrid::{Cell, Coord};
use crate::simd_layout::{AlignedU128Batch, SIMD_BATCH_LANES};

pub(crate) const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX64_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX64_MUL2: u64 = 0x94D0_49BB_1331_11EB;
const SPLITMIX64_SHIFT1: u32 = 30;
const SPLITMIX64_SHIFT2: u32 = 27;
const SPLITMIX64_SHIFT3: u32 = 31;
const MORTON_MASK_16: u64 = 0x0000_FFFF_0000_FFFF;
const MORTON_MASK_8: u64 = 0x00FF_00FF_00FF_00FF;
const MORTON_MASK_4: u64 = 0x0F0F_0F0F_0F0F_0F0F;
const MORTON_MASK_2: u64 = 0x3333_3333_3333_3333;
const MORTON_MASK_1: u64 = 0x5555_5555_5555_5555;
const SPLITMIX64_MUL1_VEC: u64x8 = u64x8::splat(SPLITMIX64_MUL1);
const SPLITMIX64_MUL2_VEC: u64x8 = u64x8::splat(SPLITMIX64_MUL2);
const SPLITMIX64_GAMMA_VEC: u64x8 = u64x8::splat(SPLITMIX64_GAMMA);
const MORTON_MASK_16_VEC: u64x8 = u64x8::splat(MORTON_MASK_16);
const MORTON_MASK_8_VEC: u64x8 = u64x8::splat(MORTON_MASK_8);
const MORTON_MASK_4_VEC: u64x8 = u64x8::splat(MORTON_MASK_4);
const MORTON_MASK_2_VEC: u64x8 = u64x8::splat(MORTON_MASK_2);
const MORTON_MASK_1_VEC: u64x8 = u64x8::splat(MORTON_MASK_1);

pub(crate) fn mix_seed(seed: u64) -> u64 {
    let mut z = seed;
    z = (z ^ (z >> SPLITMIX64_SHIFT1)).wrapping_mul(SPLITMIX64_MUL1);
    z = (z ^ (z >> SPLITMIX64_SHIFT2)).wrapping_mul(SPLITMIX64_MUL2);
    z ^ (z >> SPLITMIX64_SHIFT3)
}

pub(crate) fn mix_seed_batch(values: [u64; SIMD_BATCH_LANES]) -> [u64; SIMD_BATCH_LANES] {
    let mut z: u64x8 = must_cast(values);
    z = (z ^ (z >> SPLITMIX64_SHIFT1)) * SPLITMIX64_MUL1_VEC;
    z = (z ^ (z >> SPLITMIX64_SHIFT2)) * SPLITMIX64_MUL2_VEC;
    must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>(z ^ (z >> SPLITMIX64_SHIFT3))
}

pub(crate) fn hash_u64_words_with_level(level: u32, words: [u64; 4]) -> u64 {
    let mut value = u64::from(level);
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ words[0];
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ words[1];
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ words[2];
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ words[3];
    mix_seed(value)
}

pub(crate) fn hash_leaf_population(population: u64) -> u64 {
    hash_u64_words_with_level(0, [population, 0, 0, 0])
}

pub(crate) fn hash_chunk_neighborhood_words(words: [u64; 9]) -> u64 {
    let mut value = 9_u64;
    for word in words {
        value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ word;
    }
    mix_seed(value)
}

pub(crate) fn hash_packed_node_fingerprint(level: u32, children: [u64; 4]) -> u64 {
    hash_u64_words_with_level(level, children)
}

pub(crate) fn hash_packed_jump_fingerprint(packed_fingerprint: u64, step_exp: u32) -> u64 {
    mix_seed(packed_fingerprint ^ (u64::from(step_exp) << 32))
}

pub(crate) fn hash_chunk_coord_key(cx: Coord, cy: Coord) -> u64 {
    let mut value = 2_u64;
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ (cx as u64);
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ (cy as u64);
    mix_seed(value)
}

pub(crate) fn hash_normalized_grid_signature(
    width: Coord,
    height: Coord,
    cells: &[Cell],
) -> u64 {
    let mut value = 2_u64;
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ (width as u64);
    value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ (height as u64);
    for &(x, y) in cells {
        value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ (x as u64);
        value = value.wrapping_mul(SPLITMIX64_GAMMA) ^ (y as u64);
    }
    mix_seed(value)
}

pub(crate) fn hash_u64_words_with_level_batch(
    levels: [u32; SIMD_BATCH_LANES],
    words: [[u64; SIMD_BATCH_LANES]; 4],
) -> [u64; SIMD_BATCH_LANES] {
    let level_words = levels.map(u64::from);
    let mut value: u64x8 = must_cast(level_words);
    value =
        (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[0]);
    value =
        (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[1]);
    value =
        (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[2]);
    value =
        (value * SPLITMIX64_GAMMA_VEC) ^ must_cast::<[u64; SIMD_BATCH_LANES], u64x8>(words[3]);
    mix_seed_batch(must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>(value))
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
