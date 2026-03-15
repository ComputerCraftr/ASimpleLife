use bytemuck::{Pod, Zeroable, must_cast};
use wide::{u8x16, u16x16, u16x32};

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
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedU16ChunkRowBatches9(pub [[u16x32; 2]; 9]);

impl Default for AlignedU16ChunkRowBatches9 {
    fn default() -> Self {
        Self([[u16x32::ZERO; 2]; 9])
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct AlignedU16Rows2(pub [[u16; 8]; 2]);

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

pub(crate) fn compact_nonzero_u8_lanes(
    row_bytes: [u64; SIMD_BATCH_LANES],
    active_lanes: usize,
) -> (AlignedLaneIndexBatch, AlignedU8LaneBatch, usize) {
    let mut indices = AlignedLaneIndexBatch::default();
    let mut values = AlignedU8LaneBatch::default();
    let mut count = 0;
    for (lane, &row) in row_bytes[..active_lanes].iter().enumerate() {
        let row_bits = row.to_le_bytes()[0];
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
