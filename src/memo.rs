use std::hash::{Hash, Hasher};

use crate::RequiredExt;
use crate::cache_policy::{SIMD_RETAINED_CACHE_CAPACITY, should_collect_simd_transition_caches};
use crate::hashing::hash_chunk_neighborhood_words;
use crate::probe_table::{ProbeKey, ProbeMode, ProbeTable};
use crate::simd_layout::{AlignedLaneIndexBatch, SIMD_BATCH_LANES};
use crate::symmetry::D4Symmetry as Symmetry;

const EMPTY_CHUNK_NEIGHBORHOOD: ChunkNeighborhood = ChunkNeighborhood([0; 9]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkNeighborhood(pub [u64; 9]);

impl ChunkNeighborhood {
    pub(crate) fn fingerprint(&self) -> u64 {
        hash_chunk_neighborhood_words(self.0)
    }
}

impl Hash for ChunkNeighborhood {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.fingerprint());
    }
}

impl ProbeKey for ChunkNeighborhood {
    fn fingerprint(&self) -> u64 {
        self.fingerprint()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkTransitionMemoIntent {
    pub canonical: ChunkNeighborhood,
    pub symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkTransitionProbeBatch {
    pub hits: [Option<u64>; SIMD_BATCH_LANES],
    pub miss_intents: [Option<ChunkTransitionMemoIntent>; SIMD_BATCH_LANES],
}

#[derive(Clone, Copy, Debug)]
struct CanonicalChunkNeighborhoodBatch {
    canonical: [ChunkNeighborhood; SIMD_BATCH_LANES],
    symmetries: [Symmetry; SIMD_BATCH_LANES],
    fingerprints: [u64; SIMD_BATCH_LANES],
}

#[derive(Clone, Copy, Debug)]
struct GroupedTransitionHits {
    lanes: [AlignedLaneIndexBatch; 8],
    bits: [[u64; SIMD_BATCH_LANES]; 8],
    counts: [usize; 8],
}

#[derive(Clone, Copy, Debug)]
struct ChunkNeighborhoodBatch([ChunkNeighborhood; SIMD_BATCH_LANES]);

impl Default for ChunkNeighborhoodBatch {
    fn default() -> Self {
        Self([EMPTY_CHUNK_NEIGHBORHOOD; SIMD_BATCH_LANES])
    }
}

#[derive(Clone, Copy, Debug)]
struct CanonicalChunkNeighborhoodEntry {
    canonical: ChunkNeighborhood,
    symmetry: Symmetry,
}

#[derive(Clone, Debug, Default)]
struct MemoStats {
    probe_batches: usize,
    probe_hit_lanes: usize,
    probe_miss_lanes: usize,
    scalar_insert_slow_path_lanes: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoRuntimeStats {
    pub probe_batches: usize,
    pub probe_hit_lanes: usize,
    pub probe_miss_lanes: usize,
    pub scalar_insert_slow_path_lanes: usize,
}

#[derive(Clone, Debug)]
pub struct Memo {
    chunk_transition_cache: ProbeTable<ChunkNeighborhood, u64>,
    chunk_canonicalization_cache: ProbeTable<ChunkNeighborhood, CanonicalChunkNeighborhoodEntry>,
    stats: MemoStats,
}

impl Default for Memo {
    fn default() -> Self {
        Self {
            chunk_transition_cache: ProbeTable::new(ProbeMode::Mutable),
            chunk_canonicalization_cache: ProbeTable::new(ProbeMode::Mutable),
            stats: MemoStats::default(),
        }
    }
}

#[cfg(test)]
impl Memo {
    pub(crate) fn get_chunk_transition(&mut self, neighborhood: &ChunkNeighborhood) -> Option<u64> {
        let (canonical, symmetry) = self.canonicalize_chunk_neighborhood(neighborhood);
        self.chunk_transition_cache
            .get_with_fingerprint(&canonical, canonical.fingerprint())
            .map(|next| transform_chunk_bits(next, symmetry.inverse()))
    }
}

impl Memo {
    pub(crate) fn canonicalize_and_probe_chunk_transitions_staged(
        &mut self,
        neighborhoods: &[ChunkNeighborhood; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> ChunkTransitionProbeBatch {
        debug_assert!(active_lanes <= SIMD_BATCH_LANES);
        let mut hits = [None; SIMD_BATCH_LANES];
        let mut miss_intents = [None; SIMD_BATCH_LANES];
        if active_lanes == 0 {
            return ChunkTransitionProbeBatch { hits, miss_intents };
        }

        self.stats.probe_batches += 1;
        let canonicalized =
            self.canonicalize_chunk_neighborhoods_staged(neighborhoods, active_lanes);
        let cached = self.chunk_transition_cache.get_many_with_fingerprints(
            &canonicalized.canonical,
            &canonicalized.fingerprints,
            active_lanes,
        );
        let grouped_hits = Self::group_cached_transition_hits_by_inverse_symmetry(
            &cached,
            &canonicalized.symmetries,
            active_lanes,
        );
        for inverse in Symmetry::ALL {
            let symmetry_index = symmetry_index(inverse);
            let transformed = transform_chunk_bits_grouped(
                &grouped_hits.bits[symmetry_index],
                &grouped_hits.lanes[symmetry_index],
                grouped_hits.counts[symmetry_index],
                inverse,
            );
            for (&lane, &bits) in grouped_hits.lanes[symmetry_index].0
                [..grouped_hits.counts[symmetry_index]]
                .iter()
                .zip(&transformed)
            {
                self.stats.probe_hit_lanes += 1;
                hits[lane] = Some(bits);
            }
        }
        for lane in 0..active_lanes {
            if hits[lane].is_none() {
                self.stats.probe_miss_lanes += 1;
                miss_intents[lane] = Some(ChunkTransitionMemoIntent {
                    canonical: canonicalized.canonical[lane],
                    symmetry: canonicalized.symmetries[lane],
                });
            }
        }

        ChunkTransitionProbeBatch { hits, miss_intents }
    }

    fn group_cached_transition_hits_by_inverse_symmetry(
        cached: &[Option<u64>; SIMD_BATCH_LANES],
        symmetries: &[Symmetry; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> GroupedTransitionHits {
        let mut grouped_hits = GroupedTransitionHits {
            lanes: [AlignedLaneIndexBatch::default(); 8],
            bits: [[0_u64; SIMD_BATCH_LANES]; 8],
            counts: [0; 8],
        };
        for lane in 0..active_lanes {
            if let Some(next) = cached[lane] {
                let symmetry_index = symmetry_index(symmetries[lane].inverse());
                let index = grouped_hits.counts[symmetry_index];
                grouped_hits.lanes[symmetry_index].0[index] = lane;
                grouped_hits.bits[symmetry_index][index] = next;
                grouped_hits.counts[symmetry_index] += 1;
            }
        }
        grouped_hits
    }

    fn canonicalize_chunk_neighborhoods_staged(
        &mut self,
        neighborhoods: &[ChunkNeighborhood; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> CanonicalChunkNeighborhoodBatch {
        let mut canonical = ChunkNeighborhoodBatch::default().0;
        let mut symmetries = [Symmetry::Identity; SIMD_BATCH_LANES];
        let mut fingerprints = [0_u64; SIMD_BATCH_LANES];
        let mut input_fingerprints = [0_u64; SIMD_BATCH_LANES];
        let mut miss_lanes = AlignedLaneIndexBatch::default();
        let mut miss_count = 0;

        for lane in 0..active_lanes {
            input_fingerprints[lane] = neighborhoods[lane].fingerprint();
        }
        let cached = self
            .chunk_canonicalization_cache
            .get_many_with_fingerprints(neighborhoods, &input_fingerprints, active_lanes);
        for lane in 0..active_lanes {
            if let Some(entry) = cached[lane] {
                canonical[lane] = entry.canonical;
                symmetries[lane] = entry.symmetry;
                fingerprints[lane] = entry.canonical.fingerprint();
            } else {
                miss_lanes.0[miss_count] = lane;
                miss_count += 1;
                canonical[lane] = neighborhoods[lane];
                fingerprints[lane] = neighborhoods[lane].fingerprint();
            }
        }

        if miss_count != 0 {
            for symmetry in Symmetry::ALL.into_iter().skip(1) {
                let candidates = transform_neighborhoods_staged(
                    neighborhoods,
                    &miss_lanes,
                    miss_count,
                    symmetry,
                );
                for (&lane, &candidate) in miss_lanes.0[..miss_count].iter().zip(&candidates) {
                    if candidate.0 < canonical[lane].0 {
                        canonical[lane] = candidate;
                        symmetries[lane] = symmetry;
                        fingerprints[lane] = candidate.fingerprint();
                    }
                }
            }
            for index in 0..miss_count {
                let lane = miss_lanes.0[index];
                self.chunk_canonicalization_cache.insert_with_fingerprint(
                    neighborhoods[lane],
                    input_fingerprints[lane],
                    CanonicalChunkNeighborhoodEntry {
                        canonical: canonical[lane],
                        symmetry: symmetries[lane],
                    },
                );
            }
        }

        CanonicalChunkNeighborhoodBatch {
            canonical,
            symmetries,
            fingerprints,
        }
    }
}

#[cfg(test)]
impl Memo {
    pub(crate) fn insert_chunk_transition(&mut self, neighborhood: ChunkNeighborhood, next: u64) {
        let (canonical, symmetry) = self.canonicalize_chunk_neighborhood(&neighborhood);
        self.insert_chunk_transition_from_intent(
            ChunkTransitionMemoIntent {
                canonical,
                symmetry,
            },
            next,
        );
    }
}

impl Memo {
    pub(crate) fn insert_chunk_transition_from_intent(
        &mut self,
        intent: ChunkTransitionMemoIntent,
        next: u64,
    ) {
        self.stats.scalar_insert_slow_path_lanes += 1;
        self.chunk_transition_cache.insert_with_fingerprint(
            intent.canonical,
            intent.canonical.fingerprint(),
            transform_chunk_bits(next, intent.symmetry),
        );
    }

    pub(crate) fn maybe_collect_transition_caches(&mut self) {
        if !should_collect_simd_transition_caches(
            self.chunk_transition_cache.len(),
            self.chunk_canonicalization_cache.len(),
        ) {
            return;
        }

        self.chunk_transition_cache =
            ProbeTable::with_capacity(ProbeMode::Mutable, SIMD_RETAINED_CACHE_CAPACITY);
        self.chunk_canonicalization_cache =
            ProbeTable::with_capacity(ProbeMode::Mutable, SIMD_RETAINED_CACHE_CAPACITY);
    }
}

#[cfg(test)]
impl Memo {
    fn canonicalize_chunk_neighborhood(
        &mut self,
        neighborhood: &ChunkNeighborhood,
    ) -> (ChunkNeighborhood, Symmetry) {
        let fingerprint = neighborhood.fingerprint();
        if let Some(cached) = self
            .chunk_canonicalization_cache
            .get_with_fingerprint(neighborhood, fingerprint)
        {
            return (cached.canonical, cached.symmetry);
        }

        let canonical = canonicalize_neighborhood(neighborhood);
        self.chunk_canonicalization_cache.insert_with_fingerprint(
            *neighborhood,
            fingerprint,
            CanonicalChunkNeighborhoodEntry {
                canonical: canonical.0,
                symmetry: canonical.1,
            },
        );
        canonical
    }

    pub(crate) fn canonicalize_chunk_neighborhood_for_tests(
        &mut self,
        neighborhood: &ChunkNeighborhood,
    ) -> (ChunkNeighborhood, Symmetry, u64) {
        let (canonical, symmetry) = self.canonicalize_chunk_neighborhood(neighborhood);
        (canonical, symmetry, canonical.fingerprint())
    }

    pub(crate) fn chunk_transition_cache_len(&self) -> usize {
        self.chunk_transition_cache.len()
    }

    pub(crate) fn chunk_canonicalization_cache_len(&self) -> usize {
        self.chunk_canonicalization_cache.len()
    }

    pub(crate) fn force_collect_transition_caches(&mut self) {
        self.chunk_transition_cache.clear();
        self.chunk_canonicalization_cache.clear();
    }

    pub(crate) fn runtime_stats(&self) -> MemoRuntimeStats {
        MemoRuntimeStats {
            probe_batches: self.stats.probe_batches,
            probe_hit_lanes: self.stats.probe_hit_lanes,
            probe_miss_lanes: self.stats.probe_miss_lanes,
            scalar_insert_slow_path_lanes: self.stats.scalar_insert_slow_path_lanes,
        }
    }
}

#[cfg(test)]
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

fn transform_neighborhoods_staged(
    neighborhoods: &[ChunkNeighborhood; SIMD_BATCH_LANES],
    lanes: &AlignedLaneIndexBatch,
    active_lanes: usize,
    symmetry: Symmetry,
) -> [ChunkNeighborhood; SIMD_BATCH_LANES] {
    let mut transformed = ChunkNeighborhoodBatch::default().0;
    for (output, &lane) in transformed.iter_mut().zip(&lanes.0).take(active_lanes) {
        *output = transform_neighborhood(&neighborhoods[lane], symmetry);
    }
    transformed
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
                let local_x = usize::try_from(bit % 8).or_invariant("local x exceeded usize");
                let local_y = usize::try_from(bit / 8).or_invariant("local y exceeded usize");
                let x = chunk_x * 8 + local_x;
                let y = chunk_y * 8 + local_y;
                let (tx, ty) = symmetry.transform_coords(x, y, 23);
                let target_chunk_x = tx / 8;
                let target_chunk_y = ty / 8;
                let target_local_x = tx % 8;
                let target_local_y = ty % 8;
                let target_bit = u32::try_from(target_local_y * 8 + target_local_x)
                    .or_invariant("chunk target bit exceeded u32");
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
        let x = usize::try_from(bit % 8).or_invariant("local x exceeded usize");
        let y = usize::try_from(bit / 8).or_invariant("local y exceeded usize");
        let (tx, ty) = symmetry.transform_coords(x, y, 7);
        let target_bit = u32::try_from(ty * 8 + tx).or_invariant("chunk target bit exceeded u32");
        transformed |= 1_u64 << target_bit;
        remaining &= remaining - 1;
    }
    transformed
}

fn transform_chunk_bits_grouped(
    bits: &[u64; SIMD_BATCH_LANES],
    _lanes: &AlignedLaneIndexBatch,
    active_lanes: usize,
    symmetry: Symmetry,
) -> [u64; SIMD_BATCH_LANES] {
    let mut transformed = [0_u64; SIMD_BATCH_LANES];
    for lane in 0..active_lanes {
        transformed[lane] = transform_chunk_bits(bits[lane], symmetry);
    }
    transformed
}

const fn symmetry_index(symmetry: Symmetry) -> usize {
    match symmetry {
        Symmetry::Identity => 0,
        Symmetry::Rotate90 => 1,
        Symmetry::Rotate180 => 2,
        Symmetry::Rotate270 => 3,
        Symmetry::MirrorX => 4,
        Symmetry::MirrorXRotate90 => 5,
        Symmetry::MirrorXRotate180 => 6,
        Symmetry::MirrorXRotate270 => 7,
    }
}
