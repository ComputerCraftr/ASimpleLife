use std::collections::HashMap;

use crate::cache_policy::{SIMD_RETAINED_CACHE_CAPACITY, should_collect_simd_transition_caches};
use crate::classify::Classification;
use crate::normalize::NormalizedGridSignature;
use crate::symmetry::D4Symmetry as Symmetry;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ChunkNeighborhood(pub [u64; 9]);

#[derive(Clone, Debug, Default)]
pub struct Memo {
    classification_cache: HashMap<NormalizedGridSignature, Classification>,
    chunk_transition_cache: HashMap<ChunkNeighborhood, u64>,
    chunk_canonicalization_cache: HashMap<ChunkNeighborhood, (ChunkNeighborhood, Symmetry)>,
}

impl Memo {
    pub fn get_classification(
        &self,
        signature: &NormalizedGridSignature,
    ) -> Option<Classification> {
        self.classification_cache.get(signature).cloned()
    }

    pub fn insert_classification(
        &mut self,
        signature: NormalizedGridSignature,
        classification: Classification,
    ) {
        self.classification_cache.insert(signature, classification);
    }

    pub(crate) fn get_chunk_transition(&mut self, neighborhood: &ChunkNeighborhood) -> Option<u64> {
        let (canonical, symmetry) = self.canonicalize_chunk_neighborhood(neighborhood);
        self.chunk_transition_cache
            .get(&canonical)
            .copied()
            .map(|next| transform_chunk_bits(next, symmetry.inverse()))
    }

    pub(crate) fn insert_chunk_transition(&mut self, neighborhood: ChunkNeighborhood, next: u64) {
        let (canonical, symmetry) = self.canonicalize_chunk_neighborhood(&neighborhood);
        self.chunk_transition_cache
            .insert(canonical, transform_chunk_bits(next, symmetry));
    }

    pub(crate) fn maybe_collect_transition_caches(&mut self) {
        if !should_collect_simd_transition_caches(
            self.chunk_transition_cache.len(),
            self.chunk_canonicalization_cache.len(),
        ) {
            return;
        }

        self.chunk_transition_cache.clear();
        self.chunk_transition_cache
            .shrink_to(SIMD_RETAINED_CACHE_CAPACITY);
        self.chunk_canonicalization_cache.clear();
        self.chunk_canonicalization_cache
            .shrink_to(SIMD_RETAINED_CACHE_CAPACITY);
    }

    fn canonicalize_chunk_neighborhood(
        &mut self,
        neighborhood: &ChunkNeighborhood,
    ) -> (ChunkNeighborhood, Symmetry) {
        if let Some(cached) = self.chunk_canonicalization_cache.get(neighborhood) {
            return *cached;
        }

        let canonical = canonicalize_neighborhood(neighborhood);
        self.chunk_canonicalization_cache
            .insert(*neighborhood, canonical);
        canonical
    }

    #[cfg(test)]
    pub(crate) fn chunk_transition_cache_len(&self) -> usize {
        self.chunk_transition_cache.len()
    }

    #[cfg(test)]
    pub(crate) fn chunk_canonicalization_cache_len(&self) -> usize {
        self.chunk_canonicalization_cache.len()
    }

    #[cfg(test)]
    pub(crate) fn force_collect_transition_caches(&mut self) {
        self.chunk_transition_cache.clear();
        self.chunk_canonicalization_cache.clear();
    }
}

fn canonicalize_neighborhood(neighborhood: &ChunkNeighborhood) -> (ChunkNeighborhood, Symmetry) {
    let mut best = transform_neighborhood(neighborhood, Symmetry::Identity);
    let mut best_symmetry = Symmetry::Identity;

    for symmetry in Symmetry::ALL.into_iter().skip(1) {
        let candidate = transform_neighborhood(neighborhood, symmetry);
        if candidate.0 < best.0 {
            best = candidate;
            best_symmetry = symmetry;
        }
    }

    (best, best_symmetry)
}

fn transform_neighborhood(
    neighborhood: &ChunkNeighborhood,
    symmetry: Symmetry,
) -> ChunkNeighborhood {
    let mut transformed = [0_u64; 9];

    for chunk_y in 0..3 {
        for chunk_x in 0..3 {
            let source_index = chunk_y * 3 + chunk_x;
            let mut source_bits = neighborhood.0[source_index];
            if source_bits == 0 {
                continue;
            }

            while source_bits != 0 {
                let bit = source_bits.trailing_zeros();
                let local_x = (bit % 8) as usize;
                let local_y = (bit / 8) as usize;
                let x = chunk_x * 8 + local_x;
                let y = chunk_y * 8 + local_y;
                let (tx, ty) = transform_coord(x, y, 24, symmetry);
                let target_chunk_x = tx / 8;
                let target_chunk_y = ty / 8;
                let target_local_x = tx % 8;
                let target_local_y = ty % 8;
                let target_bit = (target_local_y * 8 + target_local_x) as u32;
                transformed[target_chunk_y * 3 + target_chunk_x] |= 1_u64 << target_bit;
                source_bits &= source_bits - 1;
            }
        }
    }

    ChunkNeighborhood(transformed)
}

fn transform_chunk_bits(bits: u64, symmetry: Symmetry) -> u64 {
    let mut transformed = 0_u64;
    let mut remaining = bits;
    while remaining != 0 {
        let bit = remaining.trailing_zeros();
        let x = (bit % 8) as usize;
        let y = (bit / 8) as usize;
        let (tx, ty) = transform_coord(x, y, 8, symmetry);
        let target_bit = (ty * 8 + tx) as u32;
        transformed |= 1_u64 << target_bit;
        remaining &= remaining - 1;
    }
    transformed
}

fn transform_coord(x: usize, y: usize, size: usize, symmetry: Symmetry) -> (usize, usize) {
    symmetry.transform_coords(x, y, size - 1)
}
