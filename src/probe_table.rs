use std::collections::TryReserveError;
use std::mem::MaybeUninit;

use crate::RequiredExt;

const GROUP_WIDTH: usize = 16;
const EMPTY: u8 = 0x80;
const DELETED: u8 = 0xfe;
const MAX_LOAD_NUMERATOR: usize = 7;
const MAX_LOAD_DENOMINATOR: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeReserveError {
    CapacityOverflow,
    Allocation,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeMode {
    AppendOnly,
    RebuildOnGc,
    Mutable,
    Scratch,
}

pub(crate) trait ProbeKey: Copy + Eq {
    fn fingerprint(&self) -> u64;
}

#[derive(Clone, Copy)]
struct Entry<K, V> {
    key: K,
    value: V,
    fingerprint: u64,
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct ControlGroup([u8; GROUP_WIDTH]);

impl ControlGroup {
    const EMPTY: Self = Self([EMPTY; GROUP_WIDTH]);

    #[inline]
    fn matching(self, byte: u8) -> u16 {
        match_control_byte_swar(&self.0, byte)
    }
}

pub(crate) fn match_control_groups_swar<const N: usize>(
    control: &[u8; GROUP_WIDTH],
    tags: &[u8; N],
    active_lanes: usize,
) -> [u16; N] {
    let mut matches = [0_u16; N];
    for lane in 0..active_lanes.min(N) {
        matches[lane] = match_control_byte_swar(control, tags[lane]);
    }
    matches
}

#[inline]
fn match_control_byte_swar(control: &[u8; GROUP_WIDTH], byte: u8) -> u16 {
    u16::from(swar_match_8(
        control[..8].try_into().or_invariant("low control half"),
        byte,
    )) | (u16::from(swar_match_8(
        control[8..].try_into().or_invariant("high control half"),
        byte,
    )) << 8)
}

#[inline]
fn swar_match_8(bytes: [u8; 8], byte: u8) -> u8 {
    let word = u64::from_le_bytes(bytes);
    let repeated = u64::from(byte).wrapping_mul(0x0101_0101_0101_0101);
    let different = word ^ repeated;
    // This exact per-byte zero test avoids the cross-byte borrow false
    // positives of the usual "has any zero byte" expression.
    let high_bits = !(((different & 0x7f7f_7f7f_7f7f_7f7f).wrapping_add(0x7f7f_7f7f_7f7f_7f7f))
        | different
        | 0x7f7f_7f7f_7f7f_7f7f)
        & 0x8080_8080_8080_8080;
    ((high_bits >> 7).wrapping_mul(0x0102_0408_1020_4080)).to_le_bytes()[7]
}

pub(crate) struct ProbeTable<K: Copy + Eq, V: Copy> {
    mode: ProbeMode,
    controls: Vec<ControlGroup>,
    entries: Vec<MaybeUninit<Entry<K, V>>>,
    len: usize,
    deleted: usize,
}

impl<K: Copy + Eq, V: Copy> std::fmt::Debug for ProbeTable<K, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProbeTable")
            .field("mode", &self.mode)
            .field("len", &self.len)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl<K: Copy + Eq, V: Copy> Clone for ProbeTable<K, V> {
    fn clone(&self) -> Self {
        let mut cloned = Self::with_capacity(self.mode, self.len);
        for index in 0..self.entries.len() {
            if !is_full(self.control(index)) {
                continue;
            }
            let entry = *self.entry(index);
            cloned
                .insert_no_grow(entry.key, entry.fingerprint, entry.value)
                .or_invariant("cloned probe table retained insufficient capacity");
        }
        cloned
    }
}

impl<K: Copy + Eq, V: Copy> ProbeTable<K, V> {
    pub(crate) fn new(mode: ProbeMode) -> Self {
        Self::with_capacity(mode, GROUP_WIDTH)
    }

    pub(crate) fn with_capacity(mode: ProbeMode, capacity: usize) -> Self {
        Self::try_with_capacity(mode, capacity).or_invariant("probe table allocation failed")
    }

    pub(crate) fn try_with_capacity(
        mode: ProbeMode,
        capacity: usize,
    ) -> Result<Self, ProbeReserveError> {
        let slots = slots_for_capacity_checked(capacity)?;
        let mut controls = Vec::new();
        controls
            .try_reserve_exact(slots / GROUP_WIDTH)
            .map_err(|_: TryReserveError| ProbeReserveError::Allocation)?;
        controls.resize(slots / GROUP_WIDTH, ControlGroup::EMPTY);
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(slots)
            .map_err(|_: TryReserveError| ProbeReserveError::Allocation)?;
        entries.resize(slots, MaybeUninit::uninit());
        Ok(Self {
            mode,
            controls,
            entries,
            len: 0,
            deleted: 0,
        })
    }

    pub(crate) fn allocation_bytes_for_capacity(
        capacity: usize,
    ) -> Result<u128, ProbeReserveError> {
        table_allocation_bytes::<K, V>(slots_for_capacity_checked(capacity)?)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.entries.len() * MAX_LOAD_NUMERATOR / MAX_LOAD_DENOMINATOR
    }

    /// Returns whether a cleared table can accept `entries` without allocating.
    /// GC uses this before destructive arena repacking so mandatory indexes can
    /// be rebuilt entirely inside their existing storage.
    pub(crate) fn can_rebuild_without_allocation(&self, entries: usize) -> bool {
        entries <= self.capacity()
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.controls.capacity() * std::mem::size_of::<ControlGroup>()
            + self.entries.capacity() * std::mem::size_of::<MaybeUninit<Entry<K, V>>>()
    }

    pub(crate) fn reservation_bytes(&self, additional: usize) -> Result<u128, ProbeReserveError> {
        let Some(rehash_slots) = self.rehash_slots_for(additional)? else {
            return Ok(0);
        };
        table_allocation_bytes::<K, V>(rehash_slots)
    }

    pub(crate) fn reservation_bytes_for_insert_with_fingerprint(
        &self,
        key: &K,
        fingerprint: u64,
    ) -> Result<u128, ProbeReserveError> {
        if self.find_index(key, fingerprint).is_some() {
            Ok(0)
        } else {
            self.reservation_bytes(1)
        }
    }

    pub(crate) fn clear(&mut self) {
        self.controls.fill(ControlGroup::EMPTY);
        self.len = 0;
        self.deleted = 0;
    }

    pub(crate) fn reset(&mut self) {
        self.clear();
    }

    pub(crate) fn release_storage(&mut self) {
        self.controls = Vec::new();
        self.entries = Vec::new();
        self.len = 0;
        self.deleted = 0;
    }

    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), ProbeReserveError> {
        if let Some(rehash_slots) = self.rehash_slots_for(additional)? {
            self.try_rehash(rehash_slots)?;
        }
        Ok(())
    }

    pub(crate) fn get_with_fingerprint(&self, key: &K, fingerprint: u64) -> Option<V> {
        if self.controls.is_empty() {
            return None;
        }
        let tag = tag(fingerprint);
        let mut group = group_index(fingerprint, self.controls.len());

        for _ in 0..self.controls.len() {
            let controls = self.controls[group];
            let mut candidates = controls.matching(tag);
            while candidates != 0 {
                let lane = candidates.trailing_zeros() as usize;
                let index = group * GROUP_WIDTH + lane;
                let entry = self.entry(index);
                if entry.fingerprint == fingerprint && entry.key == *key {
                    return Some(entry.value);
                }
                candidates &= candidates - 1;
            }
            if controls.matching(EMPTY) != 0 {
                return None;
            }
            group = (group + 1) & (self.controls.len() - 1);
        }
        None
    }

    pub(crate) fn get_many_with_fingerprints<const N: usize>(
        &self,
        keys: &[K; N],
        fingerprints: &[u64; N],
        active_lanes: usize,
    ) -> [Option<V>; N] {
        self.get_many_with_fingerprints_using(
            keys,
            fingerprints,
            active_lanes,
            match_control_groups_swar,
        )
    }

    pub(crate) fn get_many_with_fingerprints_using<const N: usize>(
        &self,
        keys: &[K; N],
        fingerprints: &[u64; N],
        active_lanes: usize,
        mut match_controls: impl FnMut(&[u8; GROUP_WIDTH], &[u8; N], usize) -> [u16; N],
    ) -> [Option<V>; N] {
        assert!(active_lanes <= N, "active lane count exceeds batch width");
        let mut values = [None; N];
        if self.controls.is_empty() {
            return values;
        }
        let mut groups = fingerprints.map(|hash| group_index(hash, self.controls.len()));
        let mut probes = [0_usize; N];
        let mut pending = [false; N];
        pending[..active_lanes].fill(true);
        let mut remaining = active_lanes;

        while remaining != 0 {
            let leader = pending
                .iter()
                .position(|&is_pending| is_pending)
                .or_invariant("pending probe batch lost its leader");
            let group = groups[leader];
            let controls = self.controls[group];
            let has_empty = controls.matching(EMPTY) != 0;
            let mut grouped_lanes = [0_usize; N];
            let mut grouped_query = [0_usize; N];
            let mut grouped_tags = [0_u8; N];
            let mut grouped_count = 0;

            for lane in 0..active_lanes {
                if !pending[lane] || groups[lane] != group {
                    continue;
                }
                let query = (0..grouped_count).find(|&query| {
                    let representative = grouped_lanes[query];
                    fingerprints[representative] == fingerprints[lane]
                        && keys[representative] == keys[lane]
                });
                grouped_query[lane] = query.unwrap_or_else(|| {
                    let query = grouped_count;
                    grouped_lanes[query] = lane;
                    grouped_tags[query] = tag(fingerprints[lane]);
                    grouped_count += 1;
                    query
                });
            }
            let grouped_matches = match_controls(&controls.0, &grouped_tags, grouped_count);
            let mut grouped_values = [None; N];

            for query in 0..grouped_count {
                let lane = grouped_lanes[query];
                let mut candidates = grouped_matches[query];
                while candidates != 0 {
                    let control_lane = candidates.trailing_zeros() as usize;
                    let index = group * GROUP_WIDTH + control_lane;
                    let entry = self.entry(index);
                    if entry.fingerprint == fingerprints[lane] && entry.key == keys[lane] {
                        grouped_values[query] = Some(entry.value);
                        break;
                    }
                    candidates &= candidates - 1;
                }
            }

            for lane in 0..active_lanes {
                if !pending[lane] || groups[lane] != group {
                    continue;
                }
                values[lane] = grouped_values[grouped_query[lane]];
                probes[lane] += 1;
                if values[lane].is_some() || has_empty || probes[lane] == self.controls.len() {
                    pending[lane] = false;
                    remaining -= 1;
                } else {
                    groups[lane] = (group + 1) & (self.controls.len() - 1);
                }
            }
        }
        values
    }

    pub(crate) fn insert_with_fingerprint(
        &mut self,
        key: K,
        fingerprint: u64,
        value: V,
    ) -> Option<V> {
        match self.try_insert_with_fingerprint(key, fingerprint, value) {
            Ok(previous) => previous,
            Err(error) => {
                if self.mode == ProbeMode::AppendOnly {
                    Result::<(), _>::Err(error)
                        .or_invariant("mandatory append-only probe table allocation failed");
                }
                None
            }
        }
    }

    pub(crate) fn try_insert_with_fingerprint(
        &mut self,
        key: K,
        fingerprint: u64,
        value: V,
    ) -> Result<Option<V>, ProbeReserveError> {
        if self.rehash_slots_for(1)?.is_some()
            && let Some(index) = self.find_index(&key, fingerprint)
        {
            let entry = self.entry_mut(index);
            let previous = entry.value;
            entry.value = value;
            return Ok(Some(previous));
        }
        self.try_reserve(1)?;
        self.insert_no_grow(key, fingerprint, value)
    }

    pub(crate) fn remove_with_fingerprint(&mut self, key: &K, fingerprint: u64) -> Option<V> {
        if !matches!(self.mode, ProbeMode::Mutable | ProbeMode::Scratch) {
            return None;
        }
        let index = self.find_index(key, fingerprint)?;
        let value = self.entry(index).value;
        self.set_control(index, DELETED);
        self.len -= 1;
        self.deleted += 1;
        Some(value)
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(K, V) -> bool) -> bool {
        if self.mode == ProbeMode::AppendOnly {
            return false;
        }
        for index in 0..self.entries.len() {
            if !is_full(self.control(index)) {
                continue;
            }
            let entry = *self.entry(index);
            if !keep(entry.key, entry.value) {
                self.set_control(index, DELETED);
                self.len -= 1;
                self.deleted += 1;
            }
        }
        true
    }

    /// GC-only filtering for mandatory append-only indexes. Removed slots become
    /// reusable tombstones, but no allocation or key movement occurs.
    pub(crate) fn retain_for_gc(&mut self, mut keep: impl FnMut(K, V) -> bool) {
        for index in 0..self.entries.len() {
            if !is_full(self.control(index)) {
                continue;
            }
            let entry = *self.entry(index);
            if !keep(entry.key, entry.value) {
                self.set_control(index, DELETED);
                self.len -= 1;
                self.deleted += 1;
            }
        }
    }

    /// Iterates in physical bucket order, which is deliberately not stable.
    /// Callers requiring deterministic output must impose semantic ordering.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (K, V)> + '_ {
        (0..self.entries.len())
            .filter(|&index| is_full(self.control(index)))
            .map(|index| {
                let entry = self.entry(index);
                (entry.key, entry.value)
            })
    }

    fn find_index(&self, key: &K, fingerprint: u64) -> Option<usize> {
        if self.controls.is_empty() {
            return None;
        }
        let mut group = group_index(fingerprint, self.controls.len());
        let expected_tag = tag(fingerprint);
        for _ in 0..self.controls.len() {
            let controls = self.controls[group];
            let mut candidates = controls.matching(expected_tag);
            while candidates != 0 {
                let lane = candidates.trailing_zeros() as usize;
                let index = group * GROUP_WIDTH + lane;
                let entry = self.entry(index);
                if entry.fingerprint == fingerprint && entry.key == *key {
                    return Some(index);
                }
                candidates &= candidates - 1;
            }
            if controls.matching(EMPTY) != 0 {
                return None;
            }
            group = (group + 1) & (self.controls.len() - 1);
        }
        None
    }

    fn insert_no_grow(
        &mut self,
        key: K,
        fingerprint: u64,
        value: V,
    ) -> Result<Option<V>, ProbeReserveError> {
        let expected_tag = tag(fingerprint);
        let mut group = group_index(fingerprint, self.controls.len());
        let mut first_deleted = None;

        for _ in 0..self.controls.len() {
            let controls = self.controls[group];
            let mut candidates = controls.matching(expected_tag);
            while candidates != 0 {
                let lane = candidates.trailing_zeros() as usize;
                let index = group * GROUP_WIDTH + lane;
                let entry = self.entry_mut(index);
                if entry.fingerprint == fingerprint && entry.key == key {
                    let previous = entry.value;
                    entry.value = value;
                    return Ok(Some(previous));
                }
                candidates &= candidates - 1;
            }

            if first_deleted.is_none() {
                let deleted = controls.matching(DELETED);
                if deleted != 0 {
                    first_deleted = Some(group * GROUP_WIDTH + deleted.trailing_zeros() as usize);
                }
            }
            let empty = controls.matching(EMPTY);
            if empty != 0 {
                let index =
                    first_deleted.unwrap_or(group * GROUP_WIDTH + empty.trailing_zeros() as usize);
                self.write_entry(
                    index,
                    Entry {
                        key,
                        value,
                        fingerprint,
                    },
                    expected_tag,
                );
                return Ok(None);
            }
            group = (group + 1) & (self.controls.len() - 1);
        }

        let Some(index) = first_deleted else {
            return Err(ProbeReserveError::Full);
        };
        self.write_entry(
            index,
            Entry {
                key,
                value,
                fingerprint,
            },
            expected_tag,
        );
        Ok(None)
    }

    fn write_entry(&mut self, index: usize, entry: Entry<K, V>, control: u8) {
        if self.control(index) == DELETED {
            self.deleted -= 1;
        }
        self.entries[index].write(entry);
        self.set_control(index, control);
        self.len += 1;
    }

    fn try_rehash(&mut self, slots: usize) -> Result<(), ProbeReserveError> {
        let capacity = slots
            .checked_mul(MAX_LOAD_NUMERATOR)
            .ok_or(ProbeReserveError::CapacityOverflow)?
            / MAX_LOAD_DENOMINATOR;
        let mut rebuilt = Self::try_with_capacity(self.mode, capacity)?;
        for index in 0..self.entries.len() {
            if !is_full(self.control(index)) {
                continue;
            }
            let entry = *self.entry(index);
            rebuilt.insert_no_grow(entry.key, entry.fingerprint, entry.value)?;
        }
        *self = rebuilt;
        Ok(())
    }

    fn rehash_slots_for(&self, additional: usize) -> Result<Option<usize>, ProbeReserveError> {
        let required = self
            .len
            .checked_add(additional)
            .ok_or(ProbeReserveError::CapacityOverflow)?;
        let required_slots = slots_for_capacity_checked(required)?;
        if required_slots > self.entries.len() {
            return Ok(Some(required_slots));
        }
        let occupied = self
            .len
            .checked_add(self.deleted)
            .and_then(|value| value.checked_add(additional))
            .ok_or(ProbeReserveError::CapacityOverflow)?;
        if occupied > self.capacity()
            && !(self.mode == ProbeMode::AppendOnly && self.deleted >= additional)
        {
            return Ok(Some(self.entries.len()));
        }
        Ok(None)
    }

    #[inline]
    fn control(&self, index: usize) -> u8 {
        self.controls[index / GROUP_WIDTH].0[index % GROUP_WIDTH]
    }

    #[inline]
    fn set_control(&mut self, index: usize, control: u8) {
        self.controls[index / GROUP_WIDTH].0[index % GROUP_WIDTH] = control;
    }

    #[inline]
    fn entry(&self, index: usize) -> &Entry<K, V> {
        debug_assert!(is_full(self.control(index)));
        // SAFETY: a full control byte is published only after its entry is initialized.
        unsafe { self.entries[index].assume_init_ref() }
    }

    #[inline]
    fn entry_mut(&mut self, index: usize) -> &mut Entry<K, V> {
        debug_assert!(is_full(self.control(index)));
        // SAFETY: exclusive table access and a full control byte guarantee initialization.
        unsafe { self.entries[index].assume_init_mut() }
    }
}

impl<K: ProbeKey, V: Copy> ProbeTable<K, V> {
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        self.get_with_fingerprint(key, key.fingerprint())
    }

    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_with_fingerprint(key, key.fingerprint(), value)
    }

    pub(crate) fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, ProbeReserveError> {
        self.try_insert_with_fingerprint(key, key.fingerprint(), value)
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        self.remove_with_fingerprint(key, key.fingerprint())
    }
}

#[inline]
fn is_full(control: u8) -> bool {
    control & 0x80 == 0
}

fn slots_for_capacity_checked(capacity: usize) -> Result<usize, ProbeReserveError> {
    let required = capacity
        .checked_mul(MAX_LOAD_DENOMINATOR)
        .and_then(|value| value.checked_add(MAX_LOAD_NUMERATOR - 1))
        .map(|value| value / MAX_LOAD_NUMERATOR)
        .ok_or(ProbeReserveError::CapacityOverflow)?;
    required
        .max(GROUP_WIDTH)
        .checked_next_power_of_two()
        .ok_or(ProbeReserveError::CapacityOverflow)
}

fn table_allocation_bytes<K, V>(slots: usize) -> Result<u128, ProbeReserveError> {
    let control_groups = slots / GROUP_WIDTH;
    let control_bytes = control_groups
        .checked_mul(std::mem::size_of::<ControlGroup>())
        .ok_or(ProbeReserveError::CapacityOverflow)?;
    let entry_bytes = slots
        .checked_mul(std::mem::size_of::<MaybeUninit<Entry<K, V>>>())
        .ok_or(ProbeReserveError::CapacityOverflow)?;
    u128::try_from(
        control_bytes
            .checked_add(entry_bytes)
            .ok_or(ProbeReserveError::CapacityOverflow)?,
    )
    .map_err(|_| ProbeReserveError::CapacityOverflow)
}

#[inline]
fn group_index(hash: u64, group_count: usize) -> usize {
    let mask = u64::try_from(group_count - 1).or_invariant("probe group mask exceeds u64");
    usize::try_from(hash & mask).or_invariant("masked probe group exceeds usize")
}

#[inline]
fn tag(hash: u64) -> u8 {
    ((hash >> 57) as u8) & 0x7f
}

#[cfg(test)]
mod tests;
