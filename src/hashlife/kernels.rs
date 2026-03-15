use std::sync::OnceLock;

use bytemuck::must_cast;
use wide::u64x8;

use super::{SIMD_BATCH_LANES, SimdBatchResult, SimdLaneResult, SimdPackedBatch};
use contracts::{
    D4CandidateBatch, D4CandidateBatchResult, D4PrefixBatch, D4PrefixDecision, DedupBatch,
    FingerprintBatch, KernelOperation, PopulationBatch, PopulationBatchResult,
};

#[allow(unsafe_code)]
mod avx2;
pub(super) mod contracts;
#[allow(unsafe_code)]
mod neon;

const VECTOR_BREAK_EVEN_LANES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KernelFlavor {
    Scalar,
    #[cfg(any(target_arch = "aarch64", test))]
    Neon,
    #[cfg(any(target_arch = "x86_64", test))]
    Avx2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct KernelAccounting {
    pub(super) operation: Option<KernelOperation>,
    pub(super) candidate_lanes: usize,
    pub(super) portable_vector_lanes: usize,
    pub(super) scalar_lanes: usize,
    pub(super) native_avx2_lanes: usize,
    pub(super) native_neon_lanes: usize,
    pub(super) native_d4_candidate_lanes: usize,
    pub(super) native_d4_prefix_compare_lanes: usize,
    pub(super) native_d4_exact_winner_lanes: usize,
}

impl KernelAccounting {
    fn scalar(operation: KernelOperation, lanes: usize) -> Self {
        Self {
            operation: Some(operation),
            candidate_lanes: lanes,
            scalar_lanes: lanes,
            ..Self::default()
        }
    }

    fn portable(operation: KernelOperation, lanes: usize) -> Self {
        Self {
            operation: Some(operation),
            candidate_lanes: lanes,
            portable_vector_lanes: lanes,
            ..Self::default()
        }
    }

    pub(super) fn accumulate(&mut self, other: Self) {
        debug_assert!(
            self.operation.is_none() || self.operation == other.operation,
            "kernel accounting cannot combine different operations"
        );
        self.operation = other.operation.or(self.operation);
        self.candidate_lanes += other.candidate_lanes;
        self.portable_vector_lanes += other.portable_vector_lanes;
        self.scalar_lanes += other.scalar_lanes;
        self.native_avx2_lanes += other.native_avx2_lanes;
        self.native_neon_lanes += other.native_neon_lanes;
        self.native_d4_candidate_lanes += other.native_d4_candidate_lanes;
        self.native_d4_prefix_compare_lanes += other.native_d4_prefix_compare_lanes;
        self.native_d4_exact_winner_lanes += other.native_d4_exact_winner_lanes;
    }

    #[cfg(target_arch = "x86_64")]
    fn avx2(operation: KernelOperation, lanes: usize) -> Self {
        Self {
            operation: Some(operation),
            candidate_lanes: lanes,
            native_avx2_lanes: lanes,
            ..Self::default()
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn neon(operation: KernelOperation, lanes: usize) -> Self {
        Self {
            operation: Some(operation),
            candidate_lanes: lanes,
            native_neon_lanes: lanes,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct KernelSet {
    flavor: KernelFlavor,
}

impl KernelSet {
    pub(super) fn selected() -> &'static Self {
        static SELECTED: OnceLock<KernelSet> = OnceLock::new();
        SELECTED.get_or_init(|| Self {
            flavor: detect_kernel_flavor(),
        })
    }

    pub(super) fn evaluate(self, batch: &SimdPackedBatch) -> (SimdBatchResult, KernelAccounting) {
        let vectorized =
            self.flavor != KernelFlavor::Scalar && batch.active_lanes >= VECTOR_BREAK_EVEN_LANES;
        if !vectorized {
            return (
                evaluate_scalar(batch),
                KernelAccounting::scalar(KernelOperation::OutputPresence, batch.active_lanes),
            );
        }
        let populations = batch.populations.map(must_cast);
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                let (masks, accounting) =
                    avx2::evaluate(&populations, batch.active_mask, batch.active_lanes);
                (assemble_results(batch, masks), accounting)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => {
                let (masks, accounting) =
                    neon::evaluate(&populations, batch.active_mask, batch.active_lanes);
                (assemble_results(batch, masks), accounting)
            }
            _ => (
                evaluate_vector(batch),
                KernelAccounting::portable(KernelOperation::OutputPresence, batch.active_lanes),
            ),
        }
    }

    pub(super) fn fingerprints(
        self,
        batch: &FingerprintBatch,
    ) -> ([u64; SIMD_BATCH_LANES], KernelAccounting) {
        if !self.use_native(batch.active_lanes) {
            return (
                contracts::scalar_fingerprints(batch),
                KernelAccounting::scalar(KernelOperation::Fingerprint, batch.active_lanes),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                avx2::fingerprints(batch)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::fingerprints(batch),
            _ => (
                contracts::scalar_fingerprints(batch),
                KernelAccounting::scalar(KernelOperation::Fingerprint, batch.active_lanes),
            ),
        }
    }

    pub(super) fn control_matches(
        self,
        control: &[u8; 16],
        tags: &[u8; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> ([u16; SIMD_BATCH_LANES], KernelAccounting) {
        if !self.use_native(active_lanes) {
            return (
                contracts::scalar_control_matches(control, tags, active_lanes),
                KernelAccounting::scalar(KernelOperation::ControlMatch, active_lanes),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                avx2::control_matches(control, tags, active_lanes)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::control_matches(control, tags, active_lanes),
            _ => (
                contracts::scalar_control_matches(control, tags, active_lanes),
                KernelAccounting::scalar(KernelOperation::ControlMatch, active_lanes),
            ),
        }
    }

    pub(super) fn compare_d4_prefixes(
        self,
        batch: &D4PrefixBatch,
    ) -> (D4PrefixDecision, KernelAccounting) {
        const CANDIDATES: usize = 8;
        if !self.use_native(CANDIDATES) {
            return (
                contracts::scalar_d4_prefix(batch),
                KernelAccounting::scalar(KernelOperation::D4SemanticPrefix, CANDIDATES),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                avx2::d4_prefix(batch)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::d4_prefix(batch),
            _ => (
                contracts::scalar_d4_prefix(batch),
                KernelAccounting::scalar(KernelOperation::D4SemanticPrefix, CANDIDATES),
            ),
        }
    }

    pub(super) fn construct_d4_candidates(
        self,
        batch: &D4CandidateBatch,
    ) -> (D4CandidateBatchResult, KernelAccounting) {
        const CANDIDATES: usize = 8;
        if !self.use_native(CANDIDATES) {
            return (
                contracts::scalar_d4_candidates(batch),
                KernelAccounting::scalar(KernelOperation::D4Candidate, CANDIDATES),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                avx2::d4_candidates(batch)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::d4_candidates(batch),
            _ => (
                contracts::scalar_d4_candidates(batch),
                KernelAccounting::scalar(KernelOperation::D4Candidate, CANDIDATES),
            ),
        }
    }

    pub(super) fn aggregate_population(
        self,
        batch: &PopulationBatch,
    ) -> (PopulationBatchResult, KernelAccounting) {
        if !self.use_native(batch.active_lanes) {
            return (
                contracts::scalar_population(batch),
                KernelAccounting::scalar(KernelOperation::Population, batch.active_lanes),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                avx2::population(batch)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::population(batch),
            _ => (
                contracts::scalar_population(batch),
                KernelAccounting::scalar(KernelOperation::Population, batch.active_lanes),
            ),
        }
    }

    pub(super) fn base_transition(
        self,
        neighborhoods: &[u16; SIMD_BATCH_LANES],
        active_lanes: usize,
    ) -> ([u8; SIMD_BATCH_LANES], KernelAccounting) {
        if !self.use_native(active_lanes) {
            return (
                contracts::scalar_base_transition(neighborhoods, active_lanes),
                KernelAccounting::scalar(KernelOperation::BaseTransition, active_lanes),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
                avx2::base_transition(neighborhoods, active_lanes)
            }
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::base_transition(neighborhoods, active_lanes),
            _ => (
                contracts::scalar_base_transition(neighborhoods, active_lanes),
                KernelAccounting::scalar(KernelOperation::BaseTransition, active_lanes),
            ),
        }
    }

    pub(super) fn dedup(self, batch: &DedupBatch) -> ([u8; SIMD_BATCH_LANES], KernelAccounting) {
        if !self.use_native(batch.active_lanes) {
            return (
                contracts::scalar_dedup(batch),
                KernelAccounting::scalar(KernelOperation::Dedup, batch.active_lanes),
            );
        }
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 if std::arch::is_x86_feature_detected!("avx2") => avx2::dedup(batch),
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => neon::dedup(batch),
            _ => (
                contracts::scalar_dedup(batch),
                KernelAccounting::scalar(KernelOperation::Dedup, batch.active_lanes),
            ),
        }
    }

    fn use_native(self, active_lanes: usize) -> bool {
        self.flavor != KernelFlavor::Scalar && active_lanes >= VECTOR_BREAK_EVEN_LANES
    }

    pub(super) fn supports_native_d4_prefix(self) -> bool {
        match self.flavor {
            #[cfg(target_arch = "x86_64")]
            KernelFlavor::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "aarch64")]
            KernelFlavor::Neon => true,
            _ => false,
        }
    }

    #[cfg(test)]
    fn with_flavor(flavor: KernelFlavor) -> Self {
        Self { flavor }
    }
}

fn detect_kernel_flavor() -> KernelFlavor {
    if std::env::var("ASIMPLELIFE_HASHLIFE_KERNEL").is_ok_and(|value| value == "scalar") {
        return KernelFlavor::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        KernelFlavor::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            KernelFlavor::Avx2
        } else {
            KernelFlavor::Scalar
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        KernelFlavor::Scalar
    }
}

fn evaluate_vector(batch: &SimdPackedBatch) -> SimdBatchResult {
    let output_presence = output_presence(batch);
    let output_nonzero_masks = output_presence.map(|population| {
        let zero_lanes: [u64; SIMD_BATCH_LANES] = must_cast(population.simd_eq(u64x8::ZERO));
        zero_lanes
            .into_iter()
            .enumerate()
            .fold(0_u8, |mask, (lane, zero)| {
                mask | (u8::from(zero == 0) << lane)
            })
            & batch.active_mask
    });
    assemble_results(batch, output_nonzero_masks)
}

fn evaluate_scalar(batch: &SimdPackedBatch) -> SimdBatchResult {
    let populations = batch
        .populations
        .map(must_cast::<u64x8, [u64; SIMD_BATCH_LANES]>);
    let mut masks = [0_u8; 4];
    for (lane, _) in populations[0][..batch.active_lanes].iter().enumerate() {
        let lane_bit = 1_u8 << lane;
        masks[0] |= lane_bit
            * u8::from(
                populations[0][lane]
                    | populations[1][lane]
                    | populations[3][lane]
                    | populations[4][lane]
                    != 0,
            );
        masks[1] |= lane_bit
            * u8::from(
                populations[1][lane]
                    | populations[2][lane]
                    | populations[4][lane]
                    | populations[5][lane]
                    != 0,
            );
        masks[2] |= lane_bit
            * u8::from(
                populations[3][lane]
                    | populations[4][lane]
                    | populations[6][lane]
                    | populations[7][lane]
                    != 0,
            );
        masks[3] |= lane_bit
            * u8::from(
                populations[4][lane]
                    | populations[5][lane]
                    | populations[7][lane]
                    | populations[8][lane]
                    != 0,
            );
    }
    assemble_results(batch, masks)
}

fn output_presence(batch: &SimdPackedBatch) -> [u64x8; 4] {
    [
        batch.populations[0] | batch.populations[1] | batch.populations[3] | batch.populations[4],
        batch.populations[1] | batch.populations[2] | batch.populations[4] | batch.populations[5],
        batch.populations[3] | batch.populations[4] | batch.populations[6] | batch.populations[7],
        batch.populations[4] | batch.populations[5] | batch.populations[7] | batch.populations[8],
    ]
}

fn assemble_results(batch: &SimdPackedBatch, masks: [u8; 4]) -> SimdBatchResult {
    let mut lanes = [SimdLaneResult {
        output_nonzero_mask: 0,
    }; SIMD_BATCH_LANES];
    for (lane, result) in lanes[..batch.active_lanes].iter_mut().enumerate() {
        let lane_bit = 1_u8 << lane;
        result.output_nonzero_mask = u8::from((masks[0] & lane_bit) != 0)
            | (u8::from((masks[1] & lane_bit) != 0) << 1)
            | (u8::from((masks[2] & lane_bit) != 0) << 2)
            | (u8::from((masks[3] & lane_bit) != 0) << 3);
    }
    SimdBatchResult { lanes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt as _;
    use contracts::{
        D4CandidateBatch, D4PrefixBatch, DedupBatch, FingerprintBatch, PopulationBatch,
    };

    fn active_mask(active_lanes: usize) -> u8 {
        u8::try_from((1_u16 << active_lanes) - 1)
            .or_invariant("eight-lane active mask should fit u8")
    }

    #[test]
    fn vector_contract_matches_scalar_for_full_and_partial_waves() {
        for active_lanes in 1..=SIMD_BATCH_LANES {
            let mut words = [[0_u64; SIMD_BATCH_LANES]; 9];
            for (word, lanes) in words.iter_mut().enumerate() {
                for (lane, value) in lanes.iter_mut().enumerate() {
                    *value = match (word * 11 + lane * 7) % 7 {
                        0 => u64::MAX,
                        1 => 1,
                        _ => 0,
                    };
                }
            }
            let batch = SimdPackedBatch {
                active_lanes,
                active_mask: active_mask(active_lanes),
                populations: words.map(must_cast),
            };
            let scalar = KernelSet::with_flavor(KernelFlavor::Scalar)
                .evaluate(&batch)
                .0;
            for flavor in [KernelFlavor::Neon, KernelFlavor::Avx2] {
                let vector = KernelSet::with_flavor(flavor).evaluate(&batch).0;
                for lane in 0..active_lanes {
                    assert_eq!(
                        vector.lanes[lane].output_nonzero_mask,
                        scalar.lanes[lane].output_nonzero_mask,
                        "structural kernel mismatch flavor={flavor:?} active_lanes={active_lanes} lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn structural_fingerprint_contract_matches_scalar_and_ignores_poisoned_tail() {
        for active_lanes in 1..=SIMD_BATCH_LANES {
            let batch = FingerprintBatch {
                levels: std::array::from_fn(|lane| {
                    if lane < active_lanes {
                        u32::try_from(lane).or_invariant("kernel lane exceeded u32")
                    } else {
                        u32::MAX
                    }
                }),
                words: std::array::from_fn(|word| {
                    std::array::from_fn(|lane| {
                        if lane < active_lanes {
                            (u64::try_from(word).or_invariant("kernel word exceeded u64") << 48)
                                .wrapping_add(
                                    u64::try_from(lane)
                                        .or_invariant("kernel lane exceeded u64")
                                        .wrapping_mul(0x9e37_79b9),
                                )
                        } else {
                            u64::MAX - u64::try_from(word).or_invariant("kernel word exceeded u64")
                        }
                    })
                }),
                active_lanes,
            };
            let expected = contracts::scalar_fingerprints(&batch);
            let (actual, accounting) = KernelSet::selected().fingerprints(&batch);
            assert_eq!(&actual[..active_lanes], &expected[..active_lanes]);
            assert!(actual[active_lanes..].iter().all(|&value| value == 0));
            assert_eq!(accounting.operation, Some(KernelOperation::Fingerprint));
        }
    }

    #[test]
    fn control_match_contract_handles_collisions_and_poisoned_tags() {
        let control = [7, 3, 7, 9, 0x80, 3, 4, 7, 1, 2, 3, 4, 5, 6, 7, 8];
        let tags = [7, 3, 0x80, 0xff, 0xaa, 0xaa, 0xaa, 0xaa];
        for active_lanes in 1..=SIMD_BATCH_LANES {
            let expected = contracts::scalar_control_matches(&control, &tags, active_lanes);
            let (actual, accounting) =
                KernelSet::selected().control_matches(&control, &tags, active_lanes);
            assert_eq!(
                actual, expected,
                "control match mismatch active_lanes={active_lanes}"
            );
            assert_eq!(accounting.operation, Some(KernelOperation::ControlMatch));
        }
    }

    #[test]
    fn d4_prefix_contract_selects_exact_prefix_winner_and_lowest_tie() {
        assert!(
            !KernelSet::with_flavor(KernelFlavor::Scalar).supports_native_d4_prefix(),
            "forced scalar kernels must bypass semantic-prefix construction"
        );
        let batch = D4PrefixBatch {
            words: [[9, 7, 3, 3, 8, 6, 4, 5], [0, 0, 8, 8, 0, 0, 0, 0]],
            complete: false,
        };
        let expected = contracts::scalar_d4_prefix(&batch);
        let (actual, accounting) = KernelSet::selected().compare_d4_prefixes(&batch);
        assert_eq!(actual, expected);
        assert_eq!(actual.transform, crate::symmetry::D4Symmetry::Rotate180);
        assert_eq!(actual.inverse, actual.transform.inverse());
        assert_eq!(actual.unresolved_mask, (1 << 2) | (1 << 3));
        assert!(!actual.exact);
        assert_eq!(
            accounting.operation,
            Some(KernelOperation::D4SemanticPrefix)
        );
        let native_lanes = accounting.native_avx2_lanes + accounting.native_neon_lanes;
        assert_eq!(accounting.native_d4_candidate_lanes, 0);
        assert_eq!(
            accounting.native_d4_prefix_compare_lanes,
            if native_lanes == 0 { 0 } else { 8 }
        );
        assert_eq!(accounting.native_d4_exact_winner_lanes, 0);

        let complete = D4PrefixBatch {
            complete: true,
            ..batch
        };
        let (actual, accounting) = KernelSet::selected().compare_d4_prefixes(&complete);
        assert!(actual.exact);
        assert_eq!(actual.transform, crate::symmetry::D4Symmetry::Rotate180);
        assert_eq!(
            accounting.native_d4_exact_winner_lanes,
            usize::from(accounting.native_avx2_lanes + accounting.native_neon_lanes != 0)
        );

        let all_symmetric = D4PrefixBatch {
            words: [[0x55aa; 8], [0xaa55; 8]],
            complete: true,
        };
        let (actual, _) = KernelSet::selected().compare_d4_prefixes(&all_symmetric);
        assert!(actual.exact);
        assert_eq!(actual.unresolved_mask, u8::MAX);
        assert_eq!(actual.transform, crate::symmetry::D4Symmetry::Identity);
        assert_eq!(actual.inverse, crate::symmetry::D4Symmetry::Identity);
    }

    #[test]
    fn d4_candidate_contract_constructs_every_orientation_without_ordering_ids() {
        let batch = D4CandidateBatch {
            children: [u32::MAX, 7, 0x8000_0000, 1],
        };
        let expected = contracts::scalar_d4_candidates(&batch);

        for flavor in [KernelFlavor::Scalar, KernelFlavor::Neon, KernelFlavor::Avx2] {
            let (actual, accounting) =
                KernelSet::with_flavor(flavor).construct_d4_candidates(&batch);
            assert_eq!(
                actual, expected,
                "D4 candidate construction mismatch for {flavor:?}"
            );
            for (index, symmetry) in crate::symmetry::D4Symmetry::ALL.into_iter().enumerate() {
                assert_eq!(
                    actual.children[index],
                    symmetry.quadrant_perm().map(|slot| batch.children[slot]),
                    "D4 candidate permutation mismatch for {symmetry:?}"
                );
            }
            if flavor == KernelFlavor::Scalar {
                assert_eq!(accounting.native_d4_candidate_lanes, 0);
                assert_eq!(accounting.scalar_lanes, 8);
            }
        }

        let (_, selected_accounting) = KernelSet::selected().construct_d4_candidates(&batch);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        if KernelSet::selected().supports_native_d4_prefix() {
            assert_eq!(
                selected_accounting.native_d4_candidate_lanes, 8,
                "the selected native kernel must account for all eight constructed candidates"
            );
        }
    }

    #[test]
    fn population_contract_matches_cross_limb_and_saturation_oracle() {
        let mut batch = PopulationBatch {
            ..PopulationBatch::default()
        };
        for lane in 0..SIMD_BATCH_LANES {
            let lane_word = u64::try_from(lane).or_invariant("kernel lane exceeded u64");
            batch.lo[0][lane] = u64::MAX - lane_word;
            batch.lo[1][lane] = lane_word + 1;
            batch.hi[2][lane] = if lane % 2 == 0 { u64::MAX } else { 4 };
            batch.hi[3][lane] = if lane % 2 == 0 { 1 } else { 8 };
        }
        batch.saturated[1][5] = 1;
        for active_lanes in 1..=SIMD_BATCH_LANES {
            batch.active_lanes = active_lanes;
            let expected = contracts::scalar_population(&batch);
            let (actual, accounting) = KernelSet::selected().aggregate_population(&batch);
            assert_eq!(actual, expected, "population mismatch lanes={active_lanes}");
            assert_eq!(accounting.operation, Some(KernelOperation::Population));
        }
    }

    #[test]
    fn base_transition_contract_matches_scalar_for_entire_four_by_four_state_space() {
        for base in (0..=u32::from(u16::MAX)).step_by(SIMD_BATCH_LANES) {
            let boards = std::array::from_fn(|lane| {
                u16::try_from(base + u32::try_from(lane).unwrap_or(u32::MAX)).unwrap_or(u16::MAX)
            });
            let expected = contracts::scalar_base_transition(&boards, SIMD_BATCH_LANES);
            let (actual, accounting) =
                KernelSet::selected().base_transition(&boards, SIMD_BATCH_LANES);
            assert_eq!(actual, expected, "base transition mismatch base={base}");
            assert_eq!(accounting.operation, Some(KernelOperation::BaseTransition));
        }
    }

    #[test]
    fn dedup_contract_requires_fingerprint_and_full_key_equality() {
        let batch = DedupBatch {
            fingerprints: [11, 11, 12, 11, 99, 99, 99, 0],
            words: [
                [1, 1, 1, 1, 5, 5, 5, u64::MAX],
                [2, 2, 2, 2, 6, 6, 6, u64::MAX],
                [3, 3, 3, 9, 7, 7, 7, u64::MAX],
                [4, 4, 4, 4, 8, 8, 9, u64::MAX],
            ],
            active_lanes: 7,
        };
        let expected = contracts::scalar_dedup(&batch);
        let (actual, accounting) = KernelSet::selected().dedup(&batch);
        assert_eq!(actual, expected);
        assert_eq!(actual[1], 0);
        assert_eq!(
            actual[3],
            u8::MAX,
            "full key mismatch must defeat hash collision"
        );
        assert_eq!(actual[5], 4);
        assert_eq!(actual[6], u8::MAX);
        assert_eq!(
            actual[7],
            u8::MAX,
            "inactive poison lane must remain untouched"
        );
        assert_eq!(accounting.operation, Some(KernelOperation::Dedup));
    }

    #[test]
    fn native_break_even_keeps_short_batches_scalar() {
        let batch = FingerprintBatch {
            active_lanes: VECTOR_BREAK_EVEN_LANES - 1,
            ..FingerprintBatch::default()
        };
        let (_, accounting) = KernelSet::selected().fingerprints(&batch);
        assert_eq!(accounting.scalar_lanes, VECTOR_BREAK_EVEN_LANES - 1);
        assert_eq!(
            accounting.native_avx2_lanes + accounting.native_neon_lanes,
            0
        );
    }

    #[test]
    fn selected_kernel_reports_only_work_executed_by_its_actual_path() {
        let batch = SimdPackedBatch {
            active_lanes: SIMD_BATCH_LANES,
            active_mask: u8::MAX,
            populations: [u64x8::splat(1); 9],
        };
        let selected = *KernelSet::selected();
        let (_, accounting) = selected.evaluate(&batch);
        assert_eq!(accounting.candidate_lanes, SIMD_BATCH_LANES);
        assert_eq!(
            accounting.candidate_lanes,
            accounting.portable_vector_lanes
                + accounting.scalar_lanes
                + accounting.native_avx2_lanes
                + accounting.native_neon_lanes,
            "kernel counters must partition every candidate lane exactly once"
        );
        if selected.flavor != KernelFlavor::Scalar {
            assert_eq!(
                accounting.native_avx2_lanes + accounting.native_neon_lanes,
                SIMD_BATCH_LANES,
                "a supported full native wave must not be attributed to scheduler or scalar work"
            );
        }
    }
}
