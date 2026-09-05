use std::mem::size_of;

use super::{CanonicalNodeRef, HashLifeEngine, NodeId, PackedTransformId};
use crate::probe_table::{ProbeKey, ProbeMode, ProbeTable};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EngineAllocationFailure {
    Cancelled,
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
        node_count: NodeId::MAX_COUNT,
        canonical_count: CanonicalNodeRef::MAX_COUNT,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MandatoryShapeBatchReservation {
    metadata_extra: usize,
    bytes: u128,
}

impl HashLifeEngine {
    pub(super) fn poll_advance_cancellation(&mut self) -> bool {
        if self.advance_cancellation.as_ref().is_some_and(|tokens| {
            tokens
                .iter()
                .any(|token| token.load(std::sync::atomic::Ordering::Relaxed))
        }) {
            self.allocation_failure = Some(EngineAllocationFailure::Cancelled);
        }
        self.allocation_failed()
    }
    pub(super) fn with_transient_allocation_scope<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let enclosing_reserved = self.allocation_transient_reserved;
        let result = operation(self);
        self.allocation_transient_reserved = enclosing_reserved;
        result
    }

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

    pub(super) fn reserve_transient_bytes(&mut self, bytes: u128) -> bool {
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

    pub(super) fn reserve_optional_transient_bytes(&mut self, bytes: u128) -> bool {
        let retained = wide_allocated_bytes(self.allocated_bytes());
        let Some(projected) = retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(bytes))
        else {
            return false;
        };
        if projected > self.allocation_hard_limit {
            return false;
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
            self.allocation_transient_reserved -= bytes;
            self.reject_allocation(bytes);
            return None;
        }
        Some(values)
    }

    fn try_reserve_transient_vec<T>(&mut self, values: &mut Vec<T>, additional: usize) -> bool {
        let Some(required) = values.len().checked_add(additional) else {
            return self.reject_allocation(u128::MAX);
        };
        if required <= values.capacity() {
            return true;
        }
        let target = required.max(values.capacity().saturating_mul(2)).max(8);
        let old_bytes = (values.capacity() as u128) * (size_of::<T>() as u128);
        let Some(bytes) = (target as u128).checked_mul(size_of::<T>() as u128) else {
            return self.reject_allocation(u128::MAX);
        };
        if !self.reserve_transient_bytes(bytes) {
            return false;
        }
        if values.try_reserve_exact(target - values.len()).is_err() {
            self.allocation_transient_reserved -= bytes;
            return self.reject_allocation(bytes);
        }
        self.allocation_transient_reserved -= old_bytes;
        true
    }

    pub(super) fn try_push_transient<T>(&mut self, values: &mut Vec<T>, value: T) -> bool {
        if !self.try_reserve_transient_vec(values, 1) {
            return false;
        }
        values.push(value);
        true
    }

    pub(super) fn try_transient_probe_table<K: ProbeKey, V: Copy>(
        &mut self,
        capacity: usize,
    ) -> Option<ProbeTable<K, V>> {
        let bytes = match ProbeTable::<K, V>::allocation_bytes_for_capacity(capacity) {
            Ok(bytes) => bytes,
            Err(_) => {
                self.reject_allocation(u128::MAX);
                return None;
            }
        };
        if !self.reserve_transient_bytes(bytes) {
            return None;
        }
        match ProbeTable::try_with_capacity(ProbeMode::Scratch, capacity) {
            Ok(table) => Some(table),
            Err(_) => {
                self.allocation_transient_reserved -= bytes;
                self.reject_allocation(bytes);
                None
            }
        }
    }

    pub(super) fn try_insert_transient_table<K: ProbeKey, V: Copy>(
        &mut self,
        table: &mut ProbeTable<K, V>,
        key: K,
        value: V,
    ) -> bool {
        let fingerprint = key.fingerprint();
        let bytes = match table.reservation_bytes_for_insert_with_fingerprint(&key, fingerprint) {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        if bytes != 0 && !self.reserve_transient_bytes(bytes) {
            return false;
        }
        let old_bytes = wide_allocated_bytes(table.allocated_bytes());
        let result = table.try_insert_with_fingerprint(key, fingerprint, value);
        if bytes != 0 {
            self.allocation_transient_reserved -= bytes + old_bytes;
            self.allocation_transient_reserved += wide_allocated_bytes(table.allocated_bytes());
        }
        if result.is_err() {
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
        // A failed dependency may have returned a sentinel while the transaction
        // unwinds. Such values must never become persistent acceleration data.
        if self.allocation_failed() {
            return false;
        }
        let required_bytes =
            match table(self).reservation_bytes_for_insert_with_fingerprint(&key, fingerprint) {
                Ok(bytes) => bytes,
                Err(_) => return false,
            };
        if required_bytes != 0 {
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

    pub(super) fn prepare_mandatory_node_growth(&mut self) -> bool {
        self.prepare_mandatory_node_batch_growth(1)
    }

    pub(super) fn prepare_mandatory_node_batch_growth(&mut self, additional: usize) -> bool {
        if self.allocation_failure.is_some() {
            return false;
        }
        if self
            .node_columns
            .len()
            .checked_add(additional)
            .is_none_or(|required| required > self.id_capacity.node_count)
        {
            return self.reject_node_id_exhaustion();
        }
        let shape = match self.mandatory_shape_batch_reservation(additional) {
            Some(reservation) => reservation,
            None => return false,
        };
        let arena_bytes = self.node_columns.growth_reservation_bytes_for(additional);
        let intern_bytes = match self.intern.reservation_bytes(additional) {
            Ok(bytes) => bytes,
            Err(_) => return self.reject_allocation(u128::MAX),
        };
        let candidate = match shape
            .bytes
            .checked_add(arena_bytes)
            .and_then(|bytes| bytes.checked_add(intern_bytes))
        {
            Some(bytes) => bytes,
            None => return self.reject_allocation(u128::MAX),
        };
        // Quiescent GC may rebuild all mandatory indexes inside retained storage
        // even when a subsequently lowered limit is below that storage.
        if candidate == 0 {
            return true;
        }
        let retained = wide_allocated_bytes(self.allocated_bytes());
        if retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(candidate))
            .is_none_or(|projected| projected > self.allocation_hard_limit)
        {
            return self.reject_allocation(candidate);
        }
        if self
            .canonical_caches
            .shape_intern
            .try_reserve(additional)
            .is_err()
            || (shape.metadata_extra != 0
                && self
                    .canonical_caches
                    .shapes
                    .try_reserve_exact(shape.metadata_extra)
                    .is_err())
            || self.intern.try_reserve(additional).is_err()
            || self.node_columns.try_reserve_nodes(additional).is_err()
        {
            return self.reject_allocation(candidate);
        }
        true
    }

    pub(super) fn prepare_mandatory_shape_growth(&mut self) -> bool {
        self.prepare_mandatory_shape_batch_growth(1)
    }

    fn prepare_mandatory_shape_batch_growth(&mut self, additional: usize) -> bool {
        let reservation = match self.mandatory_shape_batch_reservation(additional) {
            Some(reservation) => reservation,
            None => return false,
        };
        // Quiescent GC must rebuild inside already-owned capacity even when a
        // lowered limit is below retained storage. No allocation occurs here.
        if reservation.bytes == 0 {
            return true;
        }
        let retained = wide_allocated_bytes(self.allocated_bytes());
        if retained
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(reservation.bytes))
            .is_none_or(|projected| projected > self.allocation_hard_limit)
            || self
                .canonical_caches
                .shape_intern
                .try_reserve(additional)
                .is_err()
            || (reservation.metadata_extra != 0
                && self
                    .canonical_caches
                    .shapes
                    .try_reserve_exact(reservation.metadata_extra)
                    .is_err())
        {
            return self.reject_allocation(reservation.bytes);
        }
        true
    }

    fn mandatory_shape_batch_reservation(
        &mut self,
        additional: usize,
    ) -> Option<MandatoryShapeBatchReservation> {
        if self.allocation_failure.is_some() {
            return None;
        }
        if self
            .canonical_caches
            .shapes
            .len()
            .checked_add(additional)
            .is_none_or(|required| required > self.id_capacity.canonical_count)
        {
            self.reject_canonical_reference_exhaustion();
            return None;
        }
        let table_bytes = match self
            .canonical_caches
            .shape_intern
            .reservation_bytes(additional)
        {
            Ok(bytes) => bytes,
            Err(_) => {
                self.reject_allocation(u128::MAX);
                return None;
            }
        };
        let (metadata_extra, metadata_bytes) = match vector_growth::<super::CanonicalShapeMeta>(
            self.canonical_caches.shapes.len(),
            self.canonical_caches.shapes.capacity(),
            additional,
        ) {
            Some(growth) => growth,
            None => {
                self.reject_allocation(u128::MAX);
                return None;
            }
        };
        let Some(candidate) = table_bytes.checked_add(metadata_bytes) else {
            self.reject_allocation(u128::MAX);
            return None;
        };
        Some(MandatoryShapeBatchReservation {
            metadata_extra,
            bytes: candidate,
        })
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
            .is_none_or(|len| len >= PackedTransformId::MAX_COUNT)
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
    // Vec reservation is relative to length, not capacity. The old allocation
    // remains charged while the complete replacement is allocated.
    let additional_from_len = target.checked_sub(len)?;
    let bytes = (target as u128).checked_mul(size_of::<T>() as u128)?;
    Some((additional_from_len, bytes))
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
    use crate::RequiredExt;

    #[test]
    fn persistent_vector_reservation_covers_replacement_and_spare_capacity() {
        assert_eq!(vector_growth::<u64>(6, 8, 3), Some((10, 128)));
        assert_eq!(vector_growth::<u64>(8, 8, 1), Some((8, 128)));
        assert_eq!(vector_growth::<u64>(6, 8, 2), Some((0, 0)));
        let mut values = vec![0_u64; 8];
        values.truncate(6);
        let (additional, bytes) = vector_growth::<u64>(values.len(), values.capacity(), 3)
            .or_invariant("bounded fixture should fit");
        values
            .try_reserve_exact(additional)
            .or_invariant("fixture should reserve");
        assert_eq!(values.capacity(), 16);
        assert_eq!(bytes, values.capacity() as u128 * size_of::<u64>() as u128);
    }

    #[test]
    fn optional_cache_growth_respects_limit_after_transaction_ends() {
        let mut engine = HashLifeEngine::default();
        engine.result_caches.bounds.release_storage();
        let retained = wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);
        assert_eq!(engine.take_allocation_failure(), None);
        assert!(!engine.publish_optional_cache(
            |engine| &engine.result_caches.bounds,
            |engine| &mut engine.result_caches.bounds,
            engine.live_leaf,
            ProbeKey::fingerprint(&engine.live_leaf),
            super::super::RelativeBounds {
                min_x: 0,
                min_y: 0,
                max_x: 0,
                max_y: 0
            },
        ));
        assert_eq!(wide_allocated_bytes(engine.allocated_bytes()), retained);
        assert_eq!(engine.take_allocation_failure(), None);
    }

    #[test]
    fn mandatory_node_batch_preflight_rejects_before_any_capacity_changes() {
        let mut engine = HashLifeEngine::default();
        let additional = engine
            .node_columns
            .capacity()
            .checked_sub(engine.node_columns.len())
            .and_then(|remaining| remaining.checked_add(1))
            .or_invariant("fixture node batch should fit usize");
        let shape = engine
            .mandatory_shape_batch_reservation(additional)
            .or_invariant("fixture shape reservation should be representable");
        let arena_bytes = engine.node_columns.growth_reservation_bytes_for(additional);
        let intern_bytes = engine
            .intern
            .reservation_bytes(additional)
            .or_invariant("fixture intern reservation should be representable");
        let combined = shape
            .bytes
            .checked_add(arena_bytes)
            .and_then(|bytes| bytes.checked_add(intern_bytes))
            .or_invariant("fixture aggregate reservation should be representable");
        assert!(shape.bytes != 0, "shape reservation must require growth");
        assert!(
            arena_bytes != 0,
            "node arena reservation must require growth"
        );
        assert!(combined > shape.bytes);

        let retained = wide_allocated_bytes(engine.allocated_bytes());
        let capacities_before = (
            engine.node_columns.capacity(),
            engine.node_columns.allocated_bytes(),
            engine.intern.capacity(),
            engine.intern.allocated_bytes(),
            engine.canonical_caches.shape_intern.capacity(),
            engine.canonical_caches.shape_intern.allocated_bytes(),
            engine.canonical_caches.shapes.capacity(),
            engine.allocated_bytes(),
        );
        engine.begin_allocation_transaction(retained + shape.bytes);
        assert!(
            !engine.prepare_mandatory_node_batch_growth(additional),
            "shape growth alone fits, but the joint mandatory reservation must not"
        );
        assert_eq!(
            (
                engine.node_columns.capacity(),
                engine.node_columns.allocated_bytes(),
                engine.intern.capacity(),
                engine.intern.allocated_bytes(),
                engine.canonical_caches.shape_intern.capacity(),
                engine.canonical_caches.shape_intern.allocated_bytes(),
                engine.canonical_caches.shapes.capacity(),
                engine.allocated_bytes(),
            ),
            capacities_before,
            "a preflight denial must not mutate any retained capacity"
        );
        assert_eq!(
            engine.take_allocation_failure(),
            Some(EngineAllocationFailure::Allocation {
                requested_bytes: combined
            })
        );

        engine.begin_allocation_transaction(retained + combined);
        assert!(engine.prepare_mandatory_node_batch_growth(additional));
        assert_eq!(engine.take_allocation_failure(), None);
        assert!(engine.node_columns.capacity() >= engine.node_columns.len() + additional);
        assert!(engine.canonical_caches.shapes.capacity() >= additional);
    }

    #[test]
    fn mandatory_node_batch_allows_zero_allocation_below_retained_limit() {
        let mut engine = HashLifeEngine::default();
        let shape = engine
            .mandatory_shape_batch_reservation(1)
            .or_invariant("fixture shape reservation should be representable");
        let arena_bytes = engine.node_columns.growth_reservation_bytes_for(1);
        let intern_bytes = engine
            .intern
            .reservation_bytes(1)
            .or_invariant("fixture intern reservation should be representable");
        assert_eq!((shape.bytes, arena_bytes, intern_bytes), (0, 0, 0));

        let retained = wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained.saturating_sub(1));
        assert!(engine.prepare_mandatory_node_batch_growth(1));
        assert_eq!(engine.take_allocation_failure(), None);
    }

    #[test]
    fn transient_vector_growth_checks_replacement_peak_before_mutating() {
        let mut engine = HashLifeEngine::default();
        let retained = wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained + 16);
        let mut values = engine
            .try_transient_vec::<u8>(8)
            .or_invariant("initial buffer fits");
        values.extend_from_slice(&[3; 8]);
        assert!(
            !engine.try_push_transient(&mut values, 9),
            "replacement needs old 8 plus new 16 bytes, not just the growth delta"
        );
        assert_eq!(values, [3; 8]);
        assert_eq!(engine.allocation_transient_reserved, 8);
        assert!(engine.take_allocation_failure().is_some());
    }

    #[test]
    fn transient_rehash_releases_the_replaced_table_charge() {
        let mut engine = HashLifeEngine::default();
        engine.begin_allocation_transaction(u128::MAX);
        let mut table = engine
            .try_transient_probe_table::<u64, u64>(1)
            .or_invariant("initial scratch table should allocate");
        let original_bytes = table.allocated_bytes();
        for key in 0..100_u64 {
            assert!(engine.try_insert_transient_table(&mut table, key, key * 2));
            assert_eq!(
                engine.allocation_transient_reserved,
                wide_allocated_bytes(table.allocated_bytes()),
                "rehash kept a freed table charged after key {key}"
            );
        }
        assert!(table.allocated_bytes() > original_bytes);
        assert_eq!(engine.take_allocation_failure(), None);
    }

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
