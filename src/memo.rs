use std::collections::HashMap;

use crate::classify::Classification;
use crate::normalize::NormalizedGridSignature;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ChunkNeighborhood(pub [u64; 9]);

#[derive(Clone, Debug, Default)]
pub struct Memo {
    classification_cache: HashMap<NormalizedGridSignature, Classification>,
    chunk_transition_cache: HashMap<ChunkNeighborhood, u64>,
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

    pub(crate) fn get_chunk_transition(&self, neighborhood: &ChunkNeighborhood) -> Option<u64> {
        self.chunk_transition_cache.get(neighborhood).copied()
    }

    pub(crate) fn insert_chunk_transition(&mut self, neighborhood: ChunkNeighborhood, next: u64) {
        self.chunk_transition_cache.insert(neighborhood, next);
    }

    #[cfg(test)]
    pub(crate) fn chunk_transition_cache_len(&self) -> usize {
        self.chunk_transition_cache.len()
    }
}
