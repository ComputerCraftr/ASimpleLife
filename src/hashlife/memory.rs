use std::mem::size_of;

use super::{CanonicalNodeRef, HashLifeEngine, NodeId, PackedTransformId};
use crate::flat_table::{FlatKey, FlatTable};
use crate::probe_table::ProbeTable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EngineAllocationFailure {
    Allocation { requested_bytes: u128 },
    NodeIdExhausted,
    CanonicalReferenceExhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EngineIdCapacity {
    pub(super) node_count: usize,
    pub(super) canonical_count: usize,
}

impl EngineIdCapacity {
    pub(super) const FULL: Self = Self {
        // The maximum value remains reserved as the arena remap sentinel.
        node_count: NodeId::MAX as usize,
        canonical_count: CanonicalNodeRef::MAX as usize,
    };
}

impl HashLifeEngine {
    pub(super) fn begin_allocation_transaction(&mut self, hard_limit: u128) {
        debug_assert!(
            !self.allocation_transaction_active,
            "HashLife allocation transactions must not overlap"
        );
        self.allocation_hard_limit = hard_limit;
        self.allocation_transient_reserved = 0;
        self.allocation_failure = None;
        self.allocation_transaction_active = true;
    }

    pub(super) fn take_allocation_failure(&mut self) -> Option<EngineAllocationFailure> {
        self.allocation_transaction_active = false;
        self.allocation_transient_reserved = 0;
        self.allocation_failure.take()
    }

    pub(super) fn at_gc_safepoint(&self) -> bool {
        !self.scheduler_active && !self.allocation_transaction_active
    }

    pub(super) fn allocation_failed(&self) -> bool {
        self.allocation_failure.is_some()
    }

    fn reserve_transient_bytes(&mut self, bytes: u128) -> bool {
        let retained = wide_allocated_bytes(self.allocated_bytes());
        let Some(projected) = retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(bytes))
        else {
            return self.reject_allocation(u128::MAX);
        };
        if projected > self.allocation_hard_limit {
            return self.reject_allocation(bytes);
        }
        self.allocation_transient_reserved += bytes;
        true
    }

    pub(super) fn try_transient_vec<T>(&mut self, capacity: usize) -> Option<Vec<T>> {
        let Some(bytes) = (capacity as u128).checked_mul(size_of::<T>() as u128) else {
            self.reject_allocation(u128::MAX);
            return None;
        };
        if !self.reserve_transient_bytes(bytes) {
            return None;
        }
        let mut values = Vec::new();
        if values.try_reserve_exact(capacity).is_err() {
            self.reject_allocation(bytes);
            return None;
        }
        Some(values)
    }

    pub(super) fn try_transient_string(&mut self, capacity: usize) -> Option<String> {
        if !self.reserve_transient_bytes(capacity as u128) {
            return None;
        }
        let mut value = String::new();
        if value.try_reserve_exact(capacity).is_err() {
            self.reject_allocation(capacity as u128);
            return None;
        }
        Some(value)
    }

    fn try_reserve_transient_vec<T>(&mut self, values: &mut Vec<T>, additional: usize) -> bool {
        let Some(required) = values.len().checked_add(additional) else {
            return self.reject_allocation(u128::MAX);
        };
        if required <= values.capacity() {
            return true;
        }
        let target = required.max(values.capacity().saturating_mul(2)).max(8);
        let Some(extra) = target.checked_sub(values.capacity()) else {
            return self.reject_allocation(u128::MAX);
        };
        let Some(bytes) = (extra as u128).checked_mul(size_of::<T>() as u128) else {
            return self.reject_allocation(u128::MAX);
        };
        if !self.reserve_transient_bytes(bytes) {
            return false;
        }
        if values.try_reserve_exact(extra).is_err() {
            return self.reject_allocation(bytes);
        }
        true
    }

    pub(super) fn try_push_transient<T>(&mut self, values: &mut Vec<T>, value: T) -> bool {
        if !self.try_reserve_transient_vec(values, 1) {
            return false;
        }
        values.push(value);
        true
    }

    pub(super) fn try_transient_flat_table<K: FlatKey, V: Copy>(
        &mut self,
        capacity: usize,
    ) -> Option<FlatTable<K, V>> {
        let Some(bytes) = FlatTable::<K, V>::allocation_bytes_for_capacity(capacity) else {
            self.reject_allocation(u128::MAX);
            return None;
        };
        if !self.reserve_transient_bytes(bytes) {
            return None;
        }
        match FlatTable::try_with_capacity(capacity) {
            Ok(table) => Some(table),
            Err(_) => {
                self.reject_allocation(bytes);
                None
            }
        }
    }

    pub(super) fn try_insert_transient_table<K: FlatKey, V: Copy>(
        &mut self,
        table: &mut FlatTable<K, V>,
        key: K,
        value: V,
    ) -> bool {
        let bytes = match table.growth_bytes_for_insert() {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        if bytes != 0 && !self.reserve_transient_bytes(bytes) {
            return false;
        }
        if table.try_reserve_insert().is_err() || table.try_insert(key, value).is_err() {
            return self.reject_allocation(bytes.max(1));
        }
        true
    }

    pub(super) fn publish_optional_cache<K: Copy + Eq, V: Copy>(
        &mut self,
        table: fn(&Self) -> &ProbeTable<K, V>,
        table_mut: fn(&mut Self) -> &mut ProbeTable<K, V>,
        key: K,
        fingerprint: u64,
        value: V,
    ) -> bool {
        let required_bytes =
            match table(self).reservation_bytes_for_insert_with_fingerprint(&key, fingerprint) {
                Ok(bytes) => bytes,
                Err(_) => return false,
            };
        if required_bytes != 0 && self.allocation_transaction_active {
            let retained = wide_allocated_bytes(self.allocated_bytes());
            let permitted = retained
                .checked_add(self.allocation_transient_reserved)
                .and_then(|used| used.checked_add(required_bytes))
                .is_some_and(|projected| projected <= self.allocation_hard_limit);
            if !permitted {
                return false;
            }
        }
        let inserted = table_mut(self)
            .try_insert_with_fingerprint(key, fingerprint, value)
            .is_ok();
        if !inserted {
            return false;
        }
        if required_bytes != 0
            && self.allocation_transaction_active
            && wide_allocated_bytes(self.allocated_bytes()) > self.allocation_hard_limit
        {
            table_mut(self).release_storage();
            return false;
        }
        true
    }

    pub(super) fn record_active_jump_result(
        &mut self,
        key: super::CanonicalJumpKey,
        fingerprint: u64,
        value: super::PackedSymmetryKey,
    ) -> bool {
        let required_bytes = match self
            .active_jump_results
            .reservation_bytes_for_insert_with_fingerprint(&key, fingerprint)
        {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        if required_bytes != 0 {
            let retained = wide_allocated_bytes(self.allocated_bytes());
            let permitted = retained
                .checked_add(self.allocation_transient_reserved)
                .and_then(|used| used.checked_add(required_bytes))
                .is_some_and(|projected| projected <= self.allocation_hard_limit);
            if !permitted {
                return self.reject_allocation(required_bytes);
            }
        }
        if self
            .active_jump_results
            .try_insert_with_fingerprint(key, fingerprint, value)
            .is_err()
        {
            return self.reject_allocation(required_bytes.max(1));
        }
        true
    }

    pub(super) fn record_mandatory_symmetry_ref(
        &mut self,
        key: super::SymmetryRefKey,
        value: CanonicalNodeRef,
    ) -> bool {
        let fingerprint = crate::flat_table::FlatKey::fingerprint(&key);
        let required_bytes = match self
            .canonical_caches
            .symmetry_refs
            .reservation_bytes_for_insert_with_fingerprint(&key, fingerprint)
        {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        if required_bytes != 0 {
            let retained = wide_allocated_bytes(self.allocated_bytes());
            let permitted = retained
                .checked_add(self.allocation_transient_reserved)
                .and_then(|used| used.checked_add(required_bytes))
                .is_some_and(|projected| projected <= self.allocation_hard_limit);
            if !permitted {
                return self.reject_allocation(required_bytes);
            }
        }
        if self
            .canonical_caches
            .symmetry_refs
            .try_insert_with_fingerprint(key, fingerprint, value)
            .is_err()
        {
            return self.reject_allocation(required_bytes.max(1));
        }
        true
    }

    pub(super) fn prepare_mandatory_node_growth(&mut self) -> bool {
        if self.allocation_failure.is_some() {
            return false;
        }
        if self.node_columns.len() >= self.id_capacity.node_count {
            return self.reject_node_id_exhaustion();
        }
        let arena_bytes = self.node_columns.growth_reservation_bytes();
        let intern_bytes = match self.intern.reservation_bytes(1) {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        let shape_bytes = match self.canonical_caches.shape_intern.reservation_bytes(1) {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        let candidate = match arena_bytes
            .checked_add(intern_bytes)
            .and_then(|bytes| bytes.checked_add(shape_bytes))
        {
            Some(bytes) => bytes,
            None => return self.reject_allocation(u128::MAX),
        };
        let retained = wide_allocated_bytes(self.allocated_bytes());
        if retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(candidate))
            .is_none_or(|projected| projected > self.allocation_hard_limit)
        {
            return self.reject_allocation(candidate);
        }
        if self.intern.try_reserve(1).is_err()
            || self.canonical_caches.shape_intern.try_reserve(1).is_err()
            || self.node_columns.try_reserve_node().is_err()
        {
            return self.reject_allocation(candidate);
        }
        true
    }

    pub(super) fn prepare_mandatory_shape_growth(&mut self) -> bool {
        if self.allocation_failure.is_some() {
            return false;
        }
        if self.canonical_caches.shape_intern.len() >= self.id_capacity.canonical_count {
            return self.reject_canonical_reference_exhaustion();
        }
        let candidate = match self.canonical_caches.shape_intern.reservation_bytes(1) {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        let retained = wide_allocated_bytes(self.allocated_bytes());
        if retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(candidate))
            .is_none_or(|projected| projected > self.allocation_hard_limit)
            || self.canonical_caches.shape_intern.try_reserve(1).is_err()
        {
            return self.reject_allocation(candidate);
        }
        true
    }

    pub(super) fn prepare_mandatory_transform_growth(&mut self, additional: usize) -> bool {
        if self.allocation_failure.is_some() {
            return false;
        }
        if self
            .transform_state
            .nodes
            .len()
            .checked_add(additional)
            .is_none_or(|len| len >= PackedTransformId::MAX as usize)
        {
            return self.reject_allocation(u128::MAX);
        }
        let nodes = match vector_growth::<super::PackedTransformNode>(
            self.transform_state.nodes.len(),
            self.transform_state.nodes.capacity(),
            additional,
        ) {
            Some(growth) => growth,
            None => return self.reject_allocation(u128::MAX),
        };
        let materialized = match vector_growth::<Option<NodeId>>(
            self.transform_state.materialized.len(),
            self.transform_state.materialized.capacity(),
            additional,
        ) {
            Some(growth) => growth,
            None => return self.reject_allocation(u128::MAX),
        };
        let packed_roots = match vector_growth::<Option<super::PackedNodeKey>>(
            self.transform_state.packed_roots.len(),
            self.transform_state.packed_roots.capacity(),
            additional,
        ) {
            Some(growth) => growth,
            None => return self.reject_allocation(u128::MAX),
        };
        let intern = match self.transform_state.intern.reservation_bytes(additional) {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        let candidate = match nodes
            .1
            .checked_add(materialized.1)
            .and_then(|bytes| bytes.checked_add(packed_roots.1))
            .and_then(|bytes| bytes.checked_add(intern))
        {
            Some(bytes) => bytes,
            None => return self.reject_allocation(u128::MAX),
        };
        let retained = wide_allocated_bytes(self.allocated_bytes());
        if retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(candidate))
            .is_none_or(|projected| projected > self.allocation_hard_limit)
        {
            return self.reject_allocation(candidate);
        }
        if self
            .transform_state
            .nodes
            .try_reserve_exact(nodes.0)
            .is_err()
            || self
                .transform_state
                .materialized
                .try_reserve_exact(materialized.0)
                .is_err()
            || self
                .transform_state
                .packed_roots
                .try_reserve_exact(packed_roots.0)
                .is_err()
            || self.transform_state.intern.try_reserve(additional).is_err()
        {
            return self.reject_allocation(candidate);
        }
        true
    }

    pub(super) fn reject_allocation(&mut self, requested_bytes: u128) -> bool {
        self.allocation_failure = Some(EngineAllocationFailure::Allocation { requested_bytes });
        false
    }

    fn reject_node_id_exhaustion(&mut self) -> bool {
        self.allocation_failure = Some(EngineAllocationFailure::NodeIdExhausted);
        false
    }

    pub(super) fn reject_canonical_reference_exhaustion(&mut self) -> bool {
        self.allocation_failure = Some(EngineAllocationFailure::CanonicalReferenceExhausted);
        false
    }
}

fn vector_growth<T>(len: usize, capacity: usize, additional: usize) -> Option<(usize, u128)> {
    let required = len.checked_add(additional)?;
    if required <= capacity {
        return Some((0, 0));
    }
    let target = required.max(capacity.checked_mul(2)?).max(8);
    let extra = target.checked_sub(capacity)?;
    let bytes = (extra as u128).checked_mul(size_of::<T>() as u128)?;
    Some((extra, bytes))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocationBoundaryError {
    ByteCountOverflow,
    CapacityOverflow,
    BudgetExceeded { requested: u128, available: u128 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashLifeAllocationClass {
    Embed,
    Materialize,
    SnapshotImport,
    SnapshotExport,
    ArenaGrowth,
    CachePublication,
    Scheduler,
    GarbageCollection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashLifeAllocationFailure {
    pub class: HashLifeAllocationClass,
    pub ordinal: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AllocationGate {
    failure: Option<HashLifeAllocationFailure>,
    observed: u64,
}

impl AllocationGate {
    pub(crate) fn configure(&mut self, failure: Option<HashLifeAllocationFailure>) {
        self.failure = failure;
        self.observed = 0;
    }

    pub(crate) fn check(
        &mut self,
        class: HashLifeAllocationClass,
        requested_bytes: u128,
    ) -> Result<(), AllocationBoundaryError> {
        let Some(failure) = self.failure else {
            return Ok(());
        };
        if failure.class != class {
            return Ok(());
        }
        self.observed = self.observed.saturating_add(1);
        if self.observed == failure.ordinal {
            return Err(AllocationBoundaryError::BudgetExceeded {
                requested: requested_bytes,
                available: 0,
            });
        }
        Ok(())
    }
}

#[doc = "source-policy: checked-narrowing-boundary"]
pub(crate) fn checked_alloc_bytes(bytes: u128) -> Result<usize, AllocationBoundaryError> {
    usize::try_from(bytes).map_err(|_| AllocationBoundaryError::ByteCountOverflow)
}

#[doc = "source-policy: checked-narrowing-boundary"]
pub(crate) fn checked_capacity<T>(elements: u128) -> Result<usize, AllocationBoundaryError> {
    let element_bytes = size_of::<T>() as u128;
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or(AllocationBoundaryError::CapacityOverflow)?;
    checked_alloc_bytes(bytes)?;
    usize::try_from(elements).map_err(|_| AllocationBoundaryError::CapacityOverflow)
}

pub(crate) fn wide_allocated_bytes(bytes: usize) -> u128 {
    bytes as u128
}

pub(crate) fn split_u128(value: u128) -> (u64, u64) {
    let [
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        b9,
        b10,
        b11,
        b12,
        b13,
        b14,
        b15,
    ] = value.to_le_bytes();
    (
        u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
        u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HashLifeMemoryBudget {
    retained: u128,
    transient: u128,
    reserved: u128,
    peak: u128,
    hard_limit: u128,
}

impl HashLifeMemoryBudget {
    pub(crate) const fn new(hard_limit: u128) -> Self {
        Self {
            retained: 0,
            transient: 0,
            reserved: 0,
            peak: 0,
            hard_limit,
        }
    }

    pub(crate) fn sync_retained(&mut self, retained: u128) {
        self.retained = retained;
        self.record_peak();
    }

    pub(crate) fn set_hard_limit(&mut self, hard_limit: u128) {
        self.hard_limit = hard_limit;
    }

    pub(crate) fn reserve_candidate(
        &mut self,
        candidate: u128,
    ) -> Result<(), AllocationBoundaryError> {
        let used = self
            .retained
            .checked_add(self.transient)
            .and_then(|value| value.checked_add(self.reserved))
            .ok_or(AllocationBoundaryError::ByteCountOverflow)?;
        let projected = used
            .checked_add(candidate)
            .ok_or(AllocationBoundaryError::ByteCountOverflow)?;
        if projected > self.hard_limit {
            return Err(AllocationBoundaryError::BudgetExceeded {
                requested: candidate,
                available: self.hard_limit.saturating_sub(used),
            });
        }
        self.reserved += candidate;
        self.record_peak();
        Ok(())
    }

    pub(crate) fn release_candidate(&mut self, candidate: u128) {
        self.reserved = self.reserved.saturating_sub(candidate);
    }

    pub(crate) fn commit_replacement(&mut self, old: u128, new: u128, reservation: u128) {
        self.reserved = self.reserved.saturating_sub(reservation);
        self.retained = self.retained.saturating_sub(old).saturating_add(new);
        self.record_peak();
    }

    #[cfg(test)]
    pub(crate) const fn retained(self) -> u128 {
        self.retained
    }

    #[cfg(test)]
    pub(crate) const fn peak(self) -> u128 {
        self.peak
    }

    fn record_peak(&mut self) {
        let total = self
            .retained
            .saturating_add(self.transient)
            .saturating_add(self.reserved);
        self.peak = self.peak.max(total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_reservation_does_not_charge_retained_bytes_twice() {
        let mut budget = HashLifeMemoryBudget::new(1_000);
        budget.sync_retained(600);
        let reserved = budget.reserve_candidate(350);
        assert!(reserved.is_ok(), "candidate should fit once: {reserved:?}");
        budget.commit_replacement(600, 350, 350);
        assert_eq!(budget.retained(), 350);
        assert_eq!(budget.peak(), 950);
    }

    #[test]
    fn checked_allocator_boundaries_reject_unrepresentable_values() {
        assert_eq!(
            checked_alloc_bytes(u128::MAX),
            Err(AllocationBoundaryError::ByteCountOverflow)
        );
        assert_eq!(
            checked_capacity::<u64>(u128::MAX),
            Err(AllocationBoundaryError::CapacityOverflow)
        );
    }
}
