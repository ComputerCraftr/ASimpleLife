use std::collections::HashMap;

use crate::classify::Classification;
use crate::normalize::NormalizedGridSignature;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ChunkNeighborhood(pub [u64; 9]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Symmetry {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    MirrorX,
    MirrorXRotate90,
    MirrorXRotate180,
    MirrorXRotate270,
}

impl Symmetry {
    const ALL: [Self; 8] = [
        Self::Identity,
        Self::Rotate90,
        Self::Rotate180,
        Self::Rotate270,
        Self::MirrorX,
        Self::MirrorXRotate90,
        Self::MirrorXRotate180,
        Self::MirrorXRotate270,
    ];

    const fn inverse(self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Rotate90 => Self::Rotate270,
            Self::Rotate180 => Self::Rotate180,
            Self::Rotate270 => Self::Rotate90,
            Self::MirrorX => Self::MirrorX,
            Self::MirrorXRotate90 => Self::MirrorXRotate90,
            Self::MirrorXRotate180 => Self::MirrorXRotate180,
            Self::MirrorXRotate270 => Self::MirrorXRotate270,
        }
    }
}

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

    fn canonicalize_chunk_neighborhood(
        &mut self,
        neighborhood: &ChunkNeighborhood,
    ) -> (ChunkNeighborhood, Symmetry) {
        if let Some(cached) = self.chunk_canonicalization_cache.get(neighborhood) {
            return cached.clone();
        }

        let canonical = canonicalize_neighborhood(neighborhood);
        self.chunk_canonicalization_cache
            .insert(neighborhood.clone(), canonical.clone());
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
    let max = size - 1;
    match symmetry {
        Symmetry::Identity => (x, y),
        Symmetry::Rotate90 => (max - y, x),
        Symmetry::Rotate180 => (max - x, max - y),
        Symmetry::Rotate270 => (y, max - x),
        Symmetry::MirrorX => (max - x, y),
        Symmetry::MirrorXRotate90 => (max - y, max - x),
        Symmetry::MirrorXRotate180 => (x, max - y),
        Symmetry::MirrorXRotate270 => (y, x),
    }
}
