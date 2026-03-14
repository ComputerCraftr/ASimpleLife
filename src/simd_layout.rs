use bytemuck::{Pod, Zeroable, must_cast};
use wide::{u8x16, u16x16};

pub(crate) const SIMD_BATCH_LANES: usize = 8;

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU64LaneWords9(pub [[u64; 9]; SIMD_BATCH_LANES]);

impl Default for AlignedU64LaneWords9 {
    fn default() -> Self {
        Self([[0; 9]; SIMD_BATCH_LANES])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU64Batch(pub [u64; SIMD_BATCH_LANES]);

impl Default for AlignedU64Batch {
    fn default() -> Self {
        Self([0; SIMD_BATCH_LANES])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AlignedU64Value(pub u64);

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU32Batch(pub [u32; SIMD_BATCH_LANES]);

impl Default for AlignedU32Batch {
    fn default() -> Self {
        Self([0; SIMD_BATCH_LANES])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU128Batch(pub [u128; SIMD_BATCH_LANES]);

impl Default for AlignedU128Batch {
    fn default() -> Self {
        Self([0; SIMD_BATCH_LANES])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU64WordBatch4(pub [[u64; SIMD_BATCH_LANES]; 4]);

impl Default for AlignedU64WordBatch4 {
    fn default() -> Self {
        Self([[0; SIMD_BATCH_LANES]; 4])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU64WordBatch9(pub [[u64; SIMD_BATCH_LANES]; 9]);

impl Default for AlignedU64WordBatch9 {
    fn default() -> Self {
        Self([[0; SIMD_BATCH_LANES]; 9])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy)]
pub(crate) struct AlignedU16LaneChunkRows9(pub [[[u16; 8]; 9]; SIMD_BATCH_LANES]);

impl Default for AlignedU16LaneChunkRows9 {
    fn default() -> Self {
        Self([[[0; 8]; 9]; SIMD_BATCH_LANES])
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU16Rows2(pub [[u16; 8]; 2]);

unsafe impl Zeroable for AlignedU16Rows2 {}
unsafe impl Pod for AlignedU16Rows2 {}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedLaneIndexBatch(pub [usize; SIMD_BATCH_LANES]);

impl Default for AlignedLaneIndexBatch {
    fn default() -> Self {
        Self([0; SIMD_BATCH_LANES])
    }
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU8LaneBatch(pub [u8; SIMD_BATCH_LANES]);

impl Default for AlignedU8LaneBatch {
    fn default() -> Self {
        Self([0; SIMD_BATCH_LANES])
    }
}

pub(crate) fn widen_u64_pair_to_u16_rows(chunks: [u64; 2]) -> [[u16; 8]; 2] {
    let byte_lanes: u8x16 = must_cast(chunks);
    must_cast(u16x16::from(byte_lanes))
}

pub(crate) fn widen_u64_pair_to_aligned_u16_rows(chunks: [u64; 2]) -> AlignedU16Rows2 {
    AlignedU16Rows2(widen_u64_pair_to_u16_rows(chunks))
}

pub(crate) fn widen_u64_quad_to_u16_rows(chunks: [u64; 4]) -> [[u16; 8]; 4] {
    let byte_lane_halves: [u8x16; 2] = must_cast(chunks);
    must_cast([
        u16x16::from(byte_lane_halves[0]),
        u16x16::from(byte_lane_halves[1]),
    ])
}

#[cfg(test)]
pub(crate) fn transpose_u64_words_9xn<const N: usize>(
    active_lanes: usize,
    word_lanes: &[[u64; N]; 9],
) -> [[u64; 9]; N] {
    let mut lane_words = [[0; 9]; N];
    for lane in 0..active_lanes {
        lane_words[lane] = [
            word_lanes[0][lane],
            word_lanes[1][lane],
            word_lanes[2][lane],
            word_lanes[3][lane],
            word_lanes[4][lane],
            word_lanes[5][lane],
            word_lanes[6][lane],
            word_lanes[7][lane],
            word_lanes[8][lane],
        ];
    }
    lane_words
}

pub(crate) fn transpose_u64_lanes_9xn<const N: usize>(
    lane_words: &[[u64; 9]; N],
    active_lanes: usize,
) -> [[u64; N]; 9] {
    let mut word_lanes = [[0; N]; 9];
    for lane in 0..active_lanes {
        let words = lane_words[lane];
        word_lanes[0][lane] = words[0];
        word_lanes[1][lane] = words[1];
        word_lanes[2][lane] = words[2];
        word_lanes[3][lane] = words[3];
        word_lanes[4][lane] = words[4];
        word_lanes[5][lane] = words[5];
        word_lanes[6][lane] = words[6];
        word_lanes[7][lane] = words[7];
        word_lanes[8][lane] = words[8];
    }
    word_lanes
}

pub(crate) fn transpose_chunk_row_staging9(
    staging: &AlignedU16LaneChunkRows9,
    active_lanes: usize,
) -> [[[u16; SIMD_BATCH_LANES]; 8]; 9] {
    let mut chunks = [[[0; SIMD_BATCH_LANES]; 8]; 9];
    for lane in 0..active_lanes {
        for chunk_index in 0..9 {
            let rows = staging.0[lane][chunk_index];
            chunks[chunk_index][0][lane] = rows[0];
            chunks[chunk_index][1][lane] = rows[1];
            chunks[chunk_index][2][lane] = rows[2];
            chunks[chunk_index][3][lane] = rows[3];
            chunks[chunk_index][4][lane] = rows[4];
            chunks[chunk_index][5][lane] = rows[5];
            chunks[chunk_index][6][lane] = rows[6];
            chunks[chunk_index][7][lane] = rows[7];
        }
    }
    chunks
}

pub(crate) fn compact_nonzero_u8_lanes(
    row_bytes: [u64; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> (AlignedLaneIndexBatch, AlignedU8LaneBatch, usize) {
    let mut indices = AlignedLaneIndexBatch::default();
    let mut values = AlignedU8LaneBatch::default();
    let mut count = 0;
    for lane in 0..active_lanes {
        let row_bits = row_bytes[lane] as u8;
        if row_bits == 0 {
            continue;
        }
        indices.0[count] = lane;
        values.0[count] = row_bits;
        count += 1;
    }
    (indices, values, count)
}

#[cfg(test)]
mod tests {
    use super::{widen_u64_pair_to_u16_rows, widen_u64_quad_to_u16_rows};

    #[test]
    fn widening_helpers_match_bytewise_rows() {
        let pair = [0x0807_0605_0403_0201_u64, 0x100F_0E0D_0C0B_0A09_u64];
        let widened_pair = widen_u64_pair_to_u16_rows(pair);
        assert_eq!(widened_pair[0], [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(widened_pair[1], [9, 10, 11, 12, 13, 14, 15, 16]);

        let quad = [
            0x0807_0605_0403_0201_u64,
            0x100F_0E0D_0C0B_0A09_u64,
            0x1817_1615_1413_1211_u64,
            0x201F_1E1D_1C1B_1A19_u64,
        ];
        let widened_quad = widen_u64_quad_to_u16_rows(quad);
        assert_eq!(widened_quad[2], [17, 18, 19, 20, 21, 22, 23, 24]);
        assert_eq!(widened_quad[3], [25, 26, 27, 28, 29, 30, 31, 32]);
    }
}
