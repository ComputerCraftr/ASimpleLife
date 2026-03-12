use crate::classify::Classification;
use crate::memo::Memo;
use crate::normalize::NormalizedGridSignature;

use super::{ChunkNeighborhood, evolve_center_chunk_bitwise};

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
    let canonicalization_after_base = memo.chunk_canonicalization_cache_len();
    assert!(canonicalization_after_base >= 1);
    assert_eq!(memo.get_chunk_transition(&base), Some(next));
    assert_eq!(
        memo.chunk_canonicalization_cache_len(),
        canonicalization_after_base
    );
    assert_eq!(memo.get_chunk_transition(&mirrored), Some(mirrored_next));
    let canonicalization_after_mirror = memo.chunk_canonicalization_cache_len();
    assert!(canonicalization_after_mirror >= canonicalization_after_base);

    memo.insert_chunk_transition(mirrored.clone(), mirrored_next);
    assert_eq!(memo.chunk_transition_cache_len(), 1);
    assert_eq!(
        memo.chunk_canonicalization_cache_len(),
        canonicalization_after_mirror
    );
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

#[test]
fn memo_transition_collection_preserves_classification_cache() {
    let mut memo = Memo::default();
    let signature = NormalizedGridSignature {
        width: 1,
        height: 1,
        cells: vec![(0, 0)],
    };
    let classification = Classification::Repeats {
        period: 1,
        first_seen: 0,
    };
    memo.insert_classification(signature.clone(), classification.clone());

    let neighborhood = neighborhood_from_cells(&[(0, 0), (9, 9), (17, 17)]);
    let next = evolve_center_chunk_bitwise(&neighborhood);
    memo.insert_chunk_transition(neighborhood, next);
    assert_eq!(memo.chunk_transition_cache_len(), 1);

    memo.force_collect_transition_caches();

    assert_eq!(memo.chunk_transition_cache_len(), 0);
    assert_eq!(memo.chunk_canonicalization_cache_len(), 0);
    assert_eq!(memo.get_classification(&signature), Some(classification));
}
