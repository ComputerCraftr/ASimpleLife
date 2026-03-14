use crate::classify::Classification;
use crate::memo::Memo;
use crate::normalize::{NormalizedGridSignature, normalize};

use super::{
    ChunkNeighborhood, build_neighborhood, evolve_center_chunk_bitwise,
    gather_neighborhoods_staged,
};
use crate::bitgrid::BitGrid;
use crate::life::step_grid_with_chunk_changes_and_memo;

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

#[test]
fn normalized_grid_signature_fingerprint_is_translation_stable() {
    let base = BitGrid::from_cells(&[(0, 0), (1, 0), (1, 1), (8, 8), (9, 8)]);
    let shifted = base.translated(37, -19);
    let base_signature = normalize(&base).0;
    let shifted_signature = normalize(&shifted).0;
    assert_eq!(base_signature, shifted_signature);
    assert_eq!(base_signature.fingerprint(), shifted_signature.fingerprint());
}

#[test]
fn batched_neighborhood_gather_matches_scalar_gather() {
    let grid = BitGrid::from_cells(&[
        (-9, -1),
        (-8, -1),
        (-1, 7),
        (0, 0),
        (1, 0),
        (7, 7),
        (8, 8),
        (15, 15),
        (16, 0),
        (24, 24),
    ]);
    let targets = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
    ];
    let batched = gather_neighborhoods_staged(&grid, &targets);
    for (index, &(cx, cy)) in targets.iter().enumerate() {
        assert_eq!(batched.neighborhoods[index], build_neighborhood(&grid, cx, cy));
    }
}

#[test]
fn batched_chunk_transition_probe_tracks_hits_misses_and_inserts() {
    let grid = BitGrid::from_cells(&[
        (-1, -1),
        (0, 0),
        (1, 0),
        (8, 8),
        (9, 8),
        (16, 16),
    ]);
    let mut memo = Memo::default();

    let (_next, _changed) = step_grid_with_chunk_changes_and_memo(&grid, &mut memo);
    let first = memo.runtime_stats();
    assert!(first.probe_batches > 0, "{first:?}");
    assert!(first.probe_miss_lanes > 0, "{first:?}");
    assert!(first.scalar_insert_slow_path_lanes > 0, "{first:?}");
    let first_hit_lanes = first.probe_hit_lanes;
    let first_miss_lanes = first.probe_miss_lanes;
    let first_insert_slow_path_lanes = first.scalar_insert_slow_path_lanes;

    let (_next, _changed) = step_grid_with_chunk_changes_and_memo(&grid, &mut memo);
    let second = memo.runtime_stats();
    let second_hit_delta = second.probe_hit_lanes.saturating_sub(first_hit_lanes);
    let second_miss_delta = second.probe_miss_lanes.saturating_sub(first_miss_lanes);
    let second_insert_delta = second
        .scalar_insert_slow_path_lanes
        .saturating_sub(first_insert_slow_path_lanes);
    assert!(second_hit_delta > 0, "{first:?} -> {second:?}");
    assert_eq!(second_miss_delta, 0, "{first:?} -> {second:?}");
    assert_eq!(second_insert_delta, 0, "{first:?} -> {second:?}");
}

#[test]
fn canonical_chunk_neighborhood_fingerprint_matches_symmetric_variants() {
    let base = neighborhood_from_cells(&[(2, 3), (4, 5), (10, 9), (11, 10), (14, 8)]);
    let mirrored = neighborhood_from_cells(&[(21, 3), (19, 5), (13, 9), (12, 10), (9, 8)]);
    let mut memo = Memo::default();

    let (base_canonical, _base_symmetry, base_fingerprint) =
        memo.canonicalize_chunk_neighborhood_for_tests(&base);
    let (mirrored_canonical, _mirrored_symmetry, mirrored_fingerprint) =
        memo.canonicalize_chunk_neighborhood_for_tests(&mirrored);

    assert_eq!(base_canonical, mirrored_canonical);
    assert_eq!(base_fingerprint, mirrored_fingerprint);
    assert_eq!(base_canonical.fingerprint(), base_fingerprint);
}
