use crate::generators::SplitMix64;
use crate::memo::Memo;

use super::{
    ChunkNeighborhood, evolve_center_chunk_bitwise, evolve_center_chunk_naive,
    evolve_center_chunks_bitwise_batch,
};

#[test]
fn branchless_chunk_kernel_matches_naive_kernel() {
    let mut rng = SplitMix64::new(0x9E3779B97F4A7C15);
    for _ in 0..512 {
        let mut chunks = [0_u64; 9];
        for chunk in &mut chunks {
            *chunk = rng.next_u64();
        }
        let neighborhood = ChunkNeighborhood(chunks);
        assert_eq!(
            evolve_center_chunk_bitwise(&neighborhood),
            evolve_center_chunk_naive(&neighborhood)
        );
    }
}

#[test]
fn simd_chunk_batch_kernel_matches_scalar_kernel() {
    let mut rng = SplitMix64::new(0xD1B54A32D192ED03);
    for batch_len in 1..=8 {
        for _ in 0..128 {
            let mut neighborhoods = Vec::with_capacity(batch_len);
            for _ in 0..batch_len {
                let mut chunks = [0_u64; 9];
                for chunk in &mut chunks {
                    *chunk = rng.next_u64();
                }
                neighborhoods.push(ChunkNeighborhood(chunks));
            }

            let batch = evolve_center_chunks_bitwise_batch(&neighborhoods);
            let scalar = neighborhoods
                .iter()
                .map(evolve_center_chunk_bitwise)
                .collect::<Vec<_>>();
            assert_eq!(batch, scalar);
        }
    }
}

#[test]
fn symmetric_neighborhoods_share_chunk_transition_cache_entries() {
    let cells = [(2, 3), (4, 5), (10, 9), (11, 10), (14, 8)];
    let base = neighborhood_from_cells(&cells);
    let mirrored_cells = cells.map(|(x, y)| (23 - x, y));
    let mirrored = neighborhood_from_cells(&mirrored_cells);

    let next = evolve_center_chunk_bitwise(&base);
    let mirrored_next = evolve_center_chunk_bitwise(&mirrored);

    let mut memo = Memo::default();
    memo.insert_chunk_transition(base.clone(), next);
    assert_eq!(memo.chunk_transition_cache_len(), 1);
    assert_eq!(memo.chunk_canonicalization_cache_len(), 1);
    assert_eq!(memo.get_chunk_transition(&base), Some(next));
    assert_eq!(memo.chunk_canonicalization_cache_len(), 1);
    assert_eq!(memo.get_chunk_transition(&mirrored), Some(mirrored_next));
    assert_eq!(memo.chunk_canonicalization_cache_len(), 2);

    memo.insert_chunk_transition(mirrored.clone(), mirrored_next);
    assert_eq!(memo.chunk_transition_cache_len(), 1);
    assert_eq!(memo.chunk_canonicalization_cache_len(), 2);
}

fn neighborhood_from_cells(cells: &[(usize, usize)]) -> ChunkNeighborhood {
    let mut chunks = [0_u64; 9];
    for &(x, y) in cells {
        let chunk_x = x / 8;
        let chunk_y = y / 8;
        let local_x = x % 8;
        let local_y = y % 8;
        let bit = (local_y * 8 + local_x) as u32;
        chunks[chunk_y * 3 + chunk_x] |= 1_u64 << bit;
    }
    ChunkNeighborhood(chunks)
}
