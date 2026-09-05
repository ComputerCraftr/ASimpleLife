use std::mem::size_of;

use crate::RequiredExt;
use crate::bitgrid::{BitGrid, GridTranslationError};
use crate::probe_table::{ProbeKey, ProbeMode, ProbeTable};

use super::witness::{ExactWitness, Lineage, Observation, RecurrenceUnavailable};
use super::{MAX_RECURRENCE_BYTES, MAX_RECURRENCE_ENTRIES};

const ENTRY_BLOCK_LEN: usize = 16;
const MAX_ENTRY_BLOCKS: usize = MAX_RECURRENCE_ENTRIES.div_ceil(ENTRY_BLOCK_LEN);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackerCounters {
    /// Every evidence-capture attempt, including unavailable observations.
    pub observations: u64,
    /// Eligible observations that were either retained or certified.
    pub repeat_candidates: u64,
    pub fingerprint_misses: u64,
    pub exact_witness_misses: u64,
    pub certificates_produced: u64,
    pub ineligible: u64,
    pub unavailable: u64,
    pub same_generation: u64,
}

impl TrackerCounters {
    pub const fn partitions_hold(self) -> bool {
        self.repeat_candidates
            == self.fingerprint_misses + self.exact_witness_misses + self.certificates_produced
            && self.observations
                == self.repeat_candidates
                    + self.ineligible
                    + self.unavailable
                    + self.same_generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrenceSkip {
    committed_generations: u64,
    displacement: (i128, i128),
}

impl RecurrenceSkip {
    pub const fn committed_generations(self) -> u64 {
        self.committed_generations
    }

    pub const fn displacement(self) -> (i128, i128) {
        self.displacement
    }

    pub fn try_translate_grid(self, grid: &BitGrid) -> Result<BitGrid, RecurrenceUnavailable> {
        grid.try_translated(self.displacement.0, self.displacement.1)
            .map_err(|error| match error {
                GridTranslationError::CoordinateOverflow => {
                    RecurrenceUnavailable::CoordinateOverflow
                }
                GridTranslationError::Allocation => RecurrenceUnavailable::Allocation,
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeriodicCertificate {
    lineage: Lineage,
    period: u64,
    first_seen: u64,
    detected_at: u64,
    delta: (i128, i128),
}

impl PeriodicCertificate {
    pub(super) fn new(
        lineage: Lineage,
        period: u64,
        first_seen: u64,
        detected_at: u64,
        delta: (i128, i128),
    ) -> Self {
        Self {
            lineage,
            period,
            first_seen,
            detected_at,
            delta,
        }
    }

    pub const fn period(self) -> u64 {
        self.period
    }

    pub const fn lineage(self) -> Lineage {
        self.lineage
    }

    pub const fn matches_lineage(self, lineage: Lineage) -> bool {
        self.lineage.session == lineage.session && self.lineage.epoch == lineage.epoch
    }

    pub const fn first_seen(self) -> u64 {
        self.first_seen
    }

    pub const fn detected_at(self) -> u64 {
        self.detected_at
    }

    pub const fn delta(self) -> (i128, i128) {
        self.delta
    }

    pub fn checked_power(self, current: u64, target: u64) -> Option<RecurrenceSkip> {
        if current < self.detected_at || target <= current {
            return None;
        }
        // Conway evolution commutes with translation. Once a whole-universe
        // recurrence is proven, it holds at every later phase, not just at
        // generations congruent to the original observation.
        let cycles = target.checked_sub(current)?.checked_div(self.period)?;
        if cycles == 0 {
            return None;
        }
        let committed_generations = cycles.checked_mul(self.period)?;
        let wide_cycles = i128::from(cycles);
        let displacement = (
            self.delta.0.checked_mul(wide_cycles)?,
            self.delta.1.checked_mul(wide_cycles)?,
        );
        Some(RecurrenceSkip {
            committed_generations,
            displacement,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserveOutcome {
    Recorded,
    Repeated(PeriodicCertificate),
    DuplicateGeneration,
    Unavailable(RecurrenceUnavailable),
}

impl ObserveOutcome {
    pub const fn certificate(self) -> Option<PeriodicCertificate> {
        match self {
            Self::Repeated(certificate) => Some(certificate),
            Self::Recorded | Self::DuplicateGeneration | Self::Unavailable(_) => None,
        }
    }
}

#[derive(Debug)]
struct Entry {
    generation: u64,
    anchor: (i128, i128),
    witness: ExactWitness,
    next: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FingerprintKey(u64);

impl ProbeKey for FingerprintKey {
    fn fingerprint(&self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
pub struct ExactRecurrenceTracker {
    lineage: Lineage,
    index: Option<ProbeTable<FingerprintKey, usize>>,
    entry_blocks: [Option<Box<[Option<Entry>]>>; MAX_ENTRY_BLOCKS],
    entry_count: usize,
    max_entries: usize,
    max_bytes: usize,
    retained_bytes: usize,
    storage_failure: Option<RecurrenceUnavailable>,
    dag_arena: Option<(u64, u64)>,
    last_generation: Option<u64>,
    counters: TrackerCounters,
}

impl ExactRecurrenceTracker {
    pub fn new(lineage: Lineage) -> Self {
        Self::with_limits(lineage, MAX_RECURRENCE_ENTRIES, MAX_RECURRENCE_BYTES)
    }

    fn with_limits(lineage: Lineage, max_entries: usize, max_bytes: usize) -> Self {
        let max_entries = max_entries.min(MAX_RECURRENCE_ENTRIES);
        let (index, retained_bytes, storage_failure) = Self::allocate_index(max_entries, max_bytes);
        Self {
            lineage,
            index,
            entry_blocks: [const { None }; MAX_ENTRY_BLOCKS],
            entry_count: 0,
            max_entries,
            max_bytes,
            retained_bytes,
            storage_failure,
            dag_arena: None,
            last_generation: None,
            counters: TrackerCounters::default(),
        }
    }

    fn allocate_index(
        max_entries: usize,
        max_bytes: usize,
    ) -> (
        Option<ProbeTable<FingerprintKey, usize>>,
        usize,
        Option<RecurrenceUnavailable>,
    ) {
        if max_entries == 0 {
            return (None, 0, None);
        }
        let initial_capacity = max_entries.min(ENTRY_BLOCK_LEN);
        let Ok(required_bytes) =
            ProbeTable::<FingerprintKey, usize>::allocation_bytes_for_capacity(initial_capacity)
        else {
            return (None, 0, Some(RecurrenceUnavailable::Allocation));
        };
        if required_bytes > max_bytes as u128 {
            return (None, 0, Some(RecurrenceUnavailable::ByteLimit));
        }
        let Ok(index) = ProbeTable::try_with_capacity(ProbeMode::AppendOnly, initial_capacity)
        else {
            return (None, 0, Some(RecurrenceUnavailable::Allocation));
        };
        let retained_bytes = index.allocated_bytes();
        debug_assert!(retained_bytes as u128 <= required_bytes);
        (Some(index), retained_bytes, None)
    }

    pub fn reset(&mut self, lineage: Lineage) {
        self.lineage = lineage;
        self.clear_evidence();
        self.dag_arena = None;
        self.last_generation = None;
        self.counters = TrackerCounters::default();
        self.audit_retained_bytes();
    }

    fn clear_evidence(&mut self) {
        if let Some(index) = &mut self.index {
            index.clear();
        }
        let mut witness_bytes = 0_usize;
        for block in self.entry_blocks.iter_mut().flatten() {
            for entry in block.iter_mut().filter_map(Option::take) {
                witness_bytes = witness_bytes.saturating_add(entry.witness.heap_bytes());
            }
        }
        self.entry_count = 0;
        self.retained_bytes = self.retained_bytes.saturating_sub(witness_bytes);
    }

    pub const fn lineage(&self) -> Lineage {
        self.lineage
    }

    pub const fn counters(&self) -> TrackerCounters {
        self.counters
    }

    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    pub const fn allocated_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn observe(&mut self, observation: Observation) -> ObserveOutcome {
        let fingerprint = observation.witness.fingerprint();
        self.observe_with_fingerprint(observation, fingerprint)
    }

    pub fn observe_result(
        &mut self,
        observation: Result<Observation, RecurrenceUnavailable>,
    ) -> ObserveOutcome {
        match observation {
            Ok(observation) => self.observe(observation),
            Err(reason) => {
                self.counters.observations += 1;
                self.unavailable(reason)
            }
        }
    }

    fn observe_with_fingerprint(
        &mut self,
        observation: Observation,
        fingerprint: u64,
    ) -> ObserveOutcome {
        self.counters.observations += 1;
        if observation.lineage != self.lineage {
            return self.ineligible(RecurrenceUnavailable::LineageMismatch);
        }
        if let Some(last) = self.last_generation {
            if observation.generation == last {
                self.counters.same_generation += 1;
                self.assert_counter_conservation();
                return ObserveOutcome::DuplicateGeneration;
            }
            if observation.generation < last {
                return self.ineligible(RecurrenceUnavailable::NonMonotonicGeneration);
            }
        }
        if let Some(arena) = observation.witness.dag_arena() {
            if self.dag_arena.is_some_and(|previous| previous != arena) {
                self.clear_evidence();
            }
            self.dag_arena = Some(arena);
        }

        let key = FingerprintKey(fingerprint);
        let head = self.index.as_ref().and_then(|index| index.get(&key));
        let mut candidate = head;
        while let Some(entry_index) = candidate {
            let (entry_generation, entry_anchor, next, witness_matches) = {
                let entry = self.entry(entry_index);
                (
                    entry.generation,
                    entry.anchor,
                    entry.next,
                    entry.witness == observation.witness,
                )
            };
            if witness_matches {
                let period = observation.generation.saturating_sub(entry_generation);
                if period == 0 {
                    self.counters.same_generation += 1;
                    self.assert_counter_conservation();
                    return ObserveOutcome::DuplicateGeneration;
                }
                let Some(dx) = observation.anchor.0.checked_sub(entry_anchor.0) else {
                    return self.unavailable(RecurrenceUnavailable::CoordinateOverflow);
                };
                let Some(dy) = observation.anchor.1.checked_sub(entry_anchor.1) else {
                    return self.unavailable(RecurrenceUnavailable::CoordinateOverflow);
                };
                self.counters.repeat_candidates += 1;
                self.counters.certificates_produced += 1;
                self.last_generation = Some(observation.generation);
                self.assert_counter_conservation();
                return ObserveOutcome::Repeated(PeriodicCertificate::new(
                    self.lineage,
                    period,
                    entry_generation,
                    observation.generation,
                    (dx, dy),
                ));
            }
            candidate = next;
        }

        if self.entry_count == self.max_entries {
            return self.unavailable(RecurrenceUnavailable::EntryLimit);
        }
        if let Some(reason) = self.storage_failure {
            return self.unavailable(reason);
        }
        let witness_heap_bytes = observation.witness.heap_bytes();
        let Some(next_bytes) = self.retained_bytes.checked_add(witness_heap_bytes) else {
            return self.unavailable(RecurrenceUnavailable::ByteLimit);
        };
        if next_bytes > self.max_bytes {
            return self.unavailable(RecurrenceUnavailable::ByteLimit);
        }
        let entry_index = self.entry_count;
        let block_index = entry_index / ENTRY_BLOCK_LEN;
        let block_bytes = if self.entry_blocks[block_index].is_none() {
            ENTRY_BLOCK_LEN.saturating_mul(size_of::<Option<Entry>>())
        } else {
            0
        };
        let index = self
            .index
            .as_ref()
            .or_invariant("available recurrence storage must include an index");
        let Ok(index_reservation) =
            index.reservation_bytes_for_insert_with_fingerprint(&key, fingerprint)
        else {
            return self.unavailable(RecurrenceUnavailable::Allocation);
        };
        let retained_peak = (self.retained_bytes as u128)
            .saturating_add(witness_heap_bytes as u128)
            .saturating_add(block_bytes as u128)
            .saturating_add(index_reservation);
        if retained_peak > self.max_bytes as u128 {
            return self.unavailable(RecurrenceUnavailable::ByteLimit);
        }
        let allocated_block = if block_bytes == 0 {
            None
        } else {
            let mut block = Vec::new();
            if block.try_reserve_exact(ENTRY_BLOCK_LEN).is_err() {
                return self.unavailable(RecurrenceUnavailable::Allocation);
            }
            block.resize_with(ENTRY_BLOCK_LEN, || None);
            Some(block.into_boxed_slice())
        };
        let old_index_bytes = index.allocated_bytes();
        let insertion = self
            .index
            .as_mut()
            .or_invariant("available recurrence storage must include an index")
            .try_insert(key, entry_index);
        if insertion.is_err() {
            return self.unavailable(RecurrenceUnavailable::Allocation);
        }
        let new_index_bytes = self
            .index
            .as_ref()
            .or_invariant("inserted recurrence index disappeared")
            .allocated_bytes();
        if let Some(block) = allocated_block {
            self.entry_blocks[block_index] = Some(block);
        }
        let block = self.entry_blocks[block_index]
            .as_mut()
            .or_invariant("preflighted recurrence entry block disappeared");
        block[entry_index % ENTRY_BLOCK_LEN] = Some(Entry {
            generation: observation.generation,
            anchor: observation.anchor,
            witness: observation.witness,
            next: head,
        });
        self.entry_count += 1;
        self.retained_bytes = next_bytes
            .saturating_add(block_bytes)
            .saturating_sub(old_index_bytes)
            .saturating_add(new_index_bytes);
        self.last_generation = Some(observation.generation);
        self.counters.repeat_candidates += 1;
        if head.is_some() {
            self.counters.exact_witness_misses += 1;
        } else {
            self.counters.fingerprint_misses += 1;
        }
        self.assert_counter_conservation();
        self.audit_retained_bytes();
        ObserveOutcome::Recorded
    }

    fn unavailable(&mut self, reason: RecurrenceUnavailable) -> ObserveOutcome {
        self.counters.unavailable += 1;
        self.assert_counter_conservation();
        ObserveOutcome::Unavailable(reason)
    }

    fn ineligible(&mut self, reason: RecurrenceUnavailable) -> ObserveOutcome {
        self.counters.ineligible += 1;
        self.assert_counter_conservation();
        ObserveOutcome::Unavailable(reason)
    }

    fn assert_counter_conservation(&self) {
        debug_assert!(self.counters.partitions_hold());
    }

    fn entry(&self, index: usize) -> &Entry {
        self.entry_blocks[index / ENTRY_BLOCK_LEN]
            .as_ref()
            .and_then(|block| block[index % ENTRY_BLOCK_LEN].as_ref())
            .or_invariant("recurrence fingerprint chain references a missing entry")
    }

    #[cfg(debug_assertions)]
    fn audit_retained_bytes(&self) {
        let index_bytes = self.index.as_ref().map_or(0, ProbeTable::allocated_bytes);
        let entry_bytes = self
            .entry_blocks
            .iter()
            .flatten()
            .map(|block| block.len().saturating_mul(size_of::<Option<Entry>>()))
            .sum::<usize>();
        let witness_bytes = self
            .entry_blocks
            .iter()
            .flatten()
            .flat_map(|block| block.iter().flatten())
            .map(|entry| entry.witness.heap_bytes())
            .sum::<usize>();
        debug_assert_eq!(
            self.retained_bytes,
            index_bytes
                .saturating_add(entry_bytes)
                .saturating_add(witness_bytes)
        );
        debug_assert!(self.retained_bytes <= self.max_bytes);
    }

    #[cfg(not(debug_assertions))]
    const fn audit_retained_bytes(&self) {}
}

#[cfg(test)]
impl ExactRecurrenceTracker {
    pub(super) fn limited(lineage: Lineage, max_entries: usize, max_bytes: usize) -> Self {
        Self::with_limits(lineage, max_entries, max_bytes)
    }

    pub(super) fn observe_forced_fingerprint(
        &mut self,
        observation: Observation,
        fingerprint: u64,
    ) -> ObserveOutcome {
        self.observe_with_fingerprint(observation, fingerprint)
    }
}
