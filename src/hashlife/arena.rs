use std::collections::TryReserveError;
use std::mem::size_of;

use crate::RequiredExt;
use crate::hashing::{hash_leaf_population, hash_u64_words_with_level};

use super::{CanonicalNodeRef, NodeId, PackedNodeKey, PopulationCount, PopulationStat};

pub(super) const NODE_SEGMENT_SHIFT: usize = 12;
pub(super) const NODE_SEGMENT_LEN: usize = 1 << NODE_SEGMENT_SHIFT;
const NODE_SEGMENT_MASK: usize = NODE_SEGMENT_LEN - 1;

#[derive(Debug)]
struct NodeSegment {
    levels: Box<[u8]>,
    population_los: Box<[u64]>,
    population_his: Box<[u64]>,
    population_saturation: Box<[u8]>,
    nws: Box<[NodeId]>,
    nes: Box<[NodeId]>,
    sws: Box<[NodeId]>,
    ses: Box<[NodeId]>,
    fingerprints: Box<[u64]>,
    identity_refs: Box<[CanonicalNodeRef]>,
    mark_words: Box<[u64]>,
    remap: Box<[NodeId]>,
}

impl NodeSegment {
    fn try_new() -> Result<Self, TryReserveError> {
        Ok(Self {
            levels: zeroed_boxed_slice()?,
            population_los: zeroed_boxed_slice()?,
            population_his: zeroed_boxed_slice()?,
            population_saturation: zeroed_boxed_slice()?,
            nws: zeroed_boxed_slice()?,
            nes: zeroed_boxed_slice()?,
            sws: zeroed_boxed_slice()?,
            ses: zeroed_boxed_slice()?,
            fingerprints: zeroed_boxed_slice()?,
            identity_refs: zeroed_boxed_slice()?,
            mark_words: zeroed_boxed_len(NODE_SEGMENT_LEN / 64, 0)?,
            remap: zeroed_boxed_len(NODE_SEGMENT_LEN, NodeId::MAX)?,
        })
    }

    const fn allocated_bytes() -> usize {
        NODE_SEGMENT_LEN
            * (size_of::<u8>() * 2
                + size_of::<u64>() * 3
                + size_of::<NodeId>() * 4
                + size_of::<CanonicalNodeRef>()
                + size_of::<NodeId>())
            + NODE_SEGMENT_LEN / 64 * size_of::<u64>()
    }
}

fn zeroed_boxed_slice<T: Copy + Default>() -> Result<Box<[T]>, TryReserveError> {
    zeroed_boxed_len(NODE_SEGMENT_LEN, T::default())
}

fn zeroed_boxed_len<T: Copy>(len: usize, value: T) -> Result<Box<[T]>, TryReserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    values.resize(len, value);
    Ok(values.into_boxed_slice())
}

#[derive(Debug, Default)]
pub(super) struct NodeColumns {
    segments: Vec<NodeSegment>,
    len: usize,
}

impl NodeColumns {
    pub(super) fn growth_reservation_bytes_for(&self, additional: usize) -> u128 {
        let required = self.len.saturating_add(additional);
        let missing_rows = required.saturating_sub(self.capacity());
        let missing_segments = missing_rows.div_ceil(NODE_SEGMENT_LEN);
        let segment_bytes =
            (missing_segments as u128).saturating_mul(NodeSegment::allocated_bytes() as u128);
        let missing_headers = self
            .segments
            .len()
            .saturating_add(missing_segments)
            .saturating_sub(self.segments.capacity());
        segment_bytes.saturating_add(
            (missing_headers as u128).saturating_mul(size_of::<NodeSegment>() as u128),
        )
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(super) fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub(super) fn capacity(&self) -> usize {
        self.segments.len() * NODE_SEGMENT_LEN
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        self.segments.len() * NodeSegment::allocated_bytes()
            + self.segments.capacity() * size_of::<NodeSegment>()
    }

    pub(super) fn try_reserve_nodes(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let required = self.len.saturating_add(additional);
        if required <= self.capacity() {
            return Ok(());
        }
        let missing_segments = (required - self.capacity()).div_ceil(NODE_SEGMENT_LEN);
        self.segments.try_reserve_exact(missing_segments)?;
        for _ in 0..missing_segments {
            self.segments.push(NodeSegment::try_new()?);
        }
        Ok(())
    }

    pub(super) fn push(
        &mut self,
        level: u32,
        population: PopulationStat,
        children: [NodeId; 4],
        identity_ref: CanonicalNodeRef,
    ) {
        debug_assert!(
            self.len < self.capacity(),
            "node segment must be reserved before push"
        );
        let index = self.len;
        self.len += 1;
        let (segment, row) = self.location_mut(index);
        let [nw, ne, sw, se] = children;
        segment.levels[row] =
            u8::try_from(level).or_invariant("HashLife node level exceeded u8 capacity");
        segment.population_los[row] = population.lo;
        segment.population_his[row] = population.hi;
        segment.population_saturation[row] = u8::from(population.saturated);
        segment.nws[row] = nw;
        segment.nes[row] = ne;
        segment.sws[row] = sw;
        segment.ses[row] = se;
        segment.fingerprints[row] = if level == 0 {
            hash_leaf_population(population.lo)
        } else {
            hash_u64_words_with_level(level, [nw, ne, sw, se].map(u64::from))
        };
        segment.identity_refs[row] = identity_ref;
    }

    pub(super) fn level(&self, node: NodeId) -> u32 {
        let (segment, row) = self.location(node.index());
        u32::from(segment.levels[row])
    }

    pub(super) fn population(&self, node: NodeId) -> u128 {
        self.population_stat(node).value()
    }

    pub(super) fn population_stat(&self, node: NodeId) -> PopulationStat {
        let (segment, row) = self.location(node.index());
        PopulationStat {
            lo: segment.population_los[row],
            hi: segment.population_his[row],
            saturated: segment.population_saturation[row] != 0,
        }
    }

    pub(super) fn population_count(&self, node: NodeId) -> PopulationCount {
        self.population_stat(node).count()
    }

    pub(super) fn quadrants(&self, node: NodeId) -> [NodeId; 4] {
        let (segment, row) = self.location(node.index());
        [
            segment.nws[row],
            segment.nes[row],
            segment.sws[row],
            segment.ses[row],
        ]
    }

    pub(super) fn set_quadrants(&mut self, node: NodeId, children: [NodeId; 4]) {
        let (segment, row) = self.location_mut(node.index());
        let [nw, ne, sw, se] = children;
        segment.nws[row] = nw;
        segment.nes[row] = ne;
        segment.sws[row] = sw;
        segment.ses[row] = se;
    }

    pub(super) fn fingerprint(&self, node: NodeId) -> u64 {
        let (segment, row) = self.location(node.index());
        segment.fingerprints[row]
    }

    pub(super) fn set_fingerprint(&mut self, node: NodeId, fingerprint: u64) {
        let (segment, row) = self.location_mut(node.index());
        segment.fingerprints[row] = fingerprint;
    }

    pub(super) fn identity_ref(&self, node: NodeId) -> CanonicalNodeRef {
        let (segment, row) = self.location(node.index());
        segment.identity_refs[row]
    }

    pub(super) fn set_identity_ref(&mut self, node: NodeId, identity: CanonicalNodeRef) {
        let (segment, row) = self.location_mut(node.index());
        segment.identity_refs[row] = identity;
    }

    pub(super) fn packed_key(&self, node: NodeId) -> PackedNodeKey {
        if self.level(node) == 0 {
            return PackedNodeKey::new(
                0,
                [
                    NodeId::from(self.population_stat(node).lo != 0),
                    NodeId::ZERO,
                    NodeId::ZERO,
                    NodeId::ZERO,
                ],
            );
        }
        PackedNodeKey::new(self.level(node), self.quadrants(node))
    }

    pub(super) fn packed_key_and_fingerprint(&self, node: NodeId) -> (PackedNodeKey, u64) {
        (self.packed_key(node), self.fingerprint(node))
    }

    pub(super) fn copy_node(&mut self, source: usize, target: usize) {
        if source == target {
            return;
        }
        let source_id = node_id(source);
        let level = self.level(source_id);
        let population = self.population_stat(source_id);
        let children = self.quadrants(source_id);
        let fingerprint = self.fingerprint(source_id);
        let identity = self.identity_ref(source_id);
        let (segment, row) = self.location_mut(target);
        segment.levels[row] = u8::try_from(level).or_invariant("node level exceeds u8");
        segment.population_los[row] = population.lo;
        segment.population_his[row] = population.hi;
        segment.population_saturation[row] = u8::from(population.saturated);
        let [nw, ne, sw, se] = children;
        segment.nws[row] = nw;
        segment.nes[row] = ne;
        segment.sws[row] = sw;
        segment.ses[row] = se;
        segment.fingerprints[row] = fingerprint;
        segment.identity_refs[row] = identity;
    }

    pub(super) fn set_len_after_compaction(&mut self, len: usize) {
        debug_assert!(len <= self.len);
        self.len = len;
    }

    pub(super) fn release_tail_segments(&mut self) {
        self.segments.truncate(self.len.div_ceil(NODE_SEGMENT_LEN));
    }

    pub(super) fn clear_marks(&mut self) {
        for segment in &mut self.segments {
            segment.mark_words.fill(0);
        }
    }

    pub(super) fn mark(&mut self, node: NodeId) {
        let index = node.index();
        if node == NodeId::MAX || index >= self.len {
            return;
        }
        let segment = &mut self.segments[index >> NODE_SEGMENT_SHIFT];
        let row = index & NODE_SEGMENT_MASK;
        segment.mark_words[row / 64] |= 1_u64 << (row % 64);
    }

    pub(super) fn is_marked(&self, index: usize) -> bool {
        if index >= self.capacity() {
            return false;
        }
        let segment = &self.segments[index >> NODE_SEGMENT_SHIFT];
        let row = index & NODE_SEGMENT_MASK;
        (segment.mark_words[row / 64] & (1_u64 << (row % 64))) != 0
    }

    pub(super) fn marked_count(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| {
                segment
                    .mark_words
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>()
            })
            .sum()
    }

    pub(super) fn clear_remap(&mut self) {
        for segment in &mut self.segments {
            segment.remap.fill(NodeId::MAX);
        }
    }

    pub(super) fn set_remap(&mut self, old_index: usize, new_node: NodeId) {
        let segment = &mut self.segments[old_index >> NODE_SEGMENT_SHIFT];
        segment.remap[old_index & NODE_SEGMENT_MASK] = new_node;
    }

    pub(super) fn remap(&self, old_node: NodeId) -> Option<NodeId> {
        let index = old_node.index();
        if index >= self.capacity() {
            return None;
        }
        let segment = &self.segments[index >> NODE_SEGMENT_SHIFT];
        let remapped = segment.remap[index & NODE_SEGMENT_MASK];
        (remapped != NodeId::MAX).then_some(remapped)
    }

    fn location(&self, index: usize) -> (&NodeSegment, usize) {
        debug_assert!(index < self.len);
        (
            &self.segments[index >> NODE_SEGMENT_SHIFT],
            index & NODE_SEGMENT_MASK,
        )
    }

    fn location_mut(&mut self, index: usize) -> (&mut NodeSegment, usize) {
        debug_assert!(index < self.capacity());
        (
            &mut self.segments[index >> NODE_SEGMENT_SHIFT],
            index & NODE_SEGMENT_MASK,
        )
    }
}

fn node_id(index: usize) -> NodeId {
    NodeId::try_from(index).or_invariant("HashLife node arena exceeded u32 capacity")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_leaf(columns: &mut NodeColumns, alive: bool) {
        columns
            .try_reserve_nodes(1)
            .or_invariant("test segment allocation failed");
        let id = node_id(columns.len());
        columns.push(
            0,
            PopulationStat::exact(u128::from(alive)),
            [id; 4],
            if alive {
                CanonicalNodeRef::LIVE
            } else {
                CanonicalNodeRef::DEAD
            },
        );
    }

    #[test]
    fn segmented_columns_preserve_rows_across_segment_boundaries() {
        let mut columns = NodeColumns::default();
        for index in 0..=NODE_SEGMENT_LEN {
            push_leaf(&mut columns, index % 2 != 0);
        }

        assert_eq!(columns.segment_count(), 2);
        assert_eq!(columns.population(node_id(NODE_SEGMENT_LEN - 1)), 1);
        assert_eq!(columns.population(node_id(NODE_SEGMENT_LEN)), 0);
    }

    #[test]
    fn truncation_drops_unreachable_tail_segment_storage() {
        let mut columns = NodeColumns::default();
        for index in 0..=NODE_SEGMENT_LEN {
            push_leaf(&mut columns, index % 2 != 0);
        }
        let bytes_before = columns.allocated_bytes();

        columns.set_len_after_compaction(2);
        columns.release_tail_segments();

        assert_eq!(columns.segment_count(), 1);
        assert!(columns.allocated_bytes() < bytes_before);
        push_leaf(&mut columns, true);
        assert_eq!(columns.population(node_id(2)), 1);
    }
}
