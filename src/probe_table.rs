use std::collections::TryReserveError;
use std::mem::MaybeUninit;

use bytemuck::must_cast;
use wide::u8x16;

use crate::RequiredExt;
use crate::flat_table::FlatKey;
use crate::hashing::mix64;

const GROUP_WIDTH: usize = 16;
const EMPTY: u8 = 0x80;
const DELETED: u8 = 0xfe;
const MAX_LOAD_NUMERATOR: usize = 7;
const MAX_LOAD_DENOMINATOR: usize = 8;

#[derive(Debug)]
pub(crate) enum ProbeReserveError {
    CapacityOverflow,
    Allocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeMode {
    AppendOnly,
    RebuildOnGc,
    Mutable,
}

pub(crate) trait ProbeKey: Copy + Eq {
    fn fingerprint(&self) -> u64;
}

impl<T: FlatKey> ProbeKey for T {
    fn fingerprint(&self) -> u64 {
        FlatKey::fingerprint(self)
    }
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
        let controls: u8x16 = must_cast(self.0);
        let matches: [u8; GROUP_WIDTH] = must_cast(controls.simd_eq(u8x16::splat(byte)));
        matches
            .into_iter()
            .enumerate()
            .fold(0, |mask, (lane, matched)| {
                mask | (u16::from(matched != 0) << lane)
            })
    }
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
            cloned.insert_no_grow(entry.key, entry.fingerprint, entry.value);
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
        let required = self
            .len
            .checked_add(additional)
            .ok_or(ProbeReserveError::CapacityOverflow)?;
        let required_slots = slots_for_capacity_checked(required)?;
        let rehash_slots = if required_slots > self.entries.len() {
            required_slots
        } else if self.len + self.deleted + additional > self.capacity()
            && !(self.mode == ProbeMode::AppendOnly && self.deleted >= additional)
        {
            self.entries.len()
        } else {
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
        let required = self
            .len
            .checked_add(additional)
            .ok_or(ProbeReserveError::CapacityOverflow)?;
        let required_slots = slots_for_capacity_checked(required)?;
        if required_slots > self.entries.len() {
            self.try_rehash(required_slots)?;
        } else if self.len + self.deleted + additional > self.capacity()
            && !(self.mode == ProbeMode::AppendOnly && self.deleted >= additional)
        {
            self.try_rehash(self.entries.len())?;
        }
        Ok(())
    }

    pub(crate) fn get_with_fingerprint(&self, key: &K, fingerprint: u64) -> Option<V> {
        if self.controls.is_empty() {
            return None;
        }
        let mixed = mix64(fingerprint);
        let tag = tag(mixed);
        let mut group = group_index(mixed, self.controls.len());

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
            |controls, tags, lanes| {
                let group = ControlGroup(*controls);
                let mut matches = [0_u16; N];
                for lane in 0..lanes {
                    matches[lane] = group.matching(tags[lane]);
                }
                matches
            },
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
        let mixed = fingerprints.map(mix64);
        let mut groups = mixed.map(|hash| group_index(hash, self.controls.len()));
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
            let mut grouped_tags = [0_u8; N];
            let mut grouped_count = 0;

            for lane in 0..active_lanes {
                if !pending[lane] || groups[lane] != group {
                    continue;
                }
                grouped_lanes[grouped_count] = lane;
                grouped_tags[grouped_count] = tag(mixed[lane]);
                grouped_count += 1;
            }
            let grouped_matches = match_controls(&controls.0, &grouped_tags, grouped_count);

            for grouped_lane in 0..grouped_count {
                let lane = grouped_lanes[grouped_lane];
                let mut candidates = grouped_matches[grouped_lane];
                while candidates != 0 {
                    let control_lane = candidates.trailing_zeros() as usize;
                    let index = group * GROUP_WIDTH + control_lane;
                    let entry = self.entry(index);
                    if entry.fingerprint == fingerprints[lane] && entry.key == keys[lane] {
                        values[lane] = Some(entry.value);
                        break;
                    }
                    candidates &= candidates - 1;
                }

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
        if let Err(error) = self.try_reserve(1) {
            if self.mode == ProbeMode::AppendOnly {
                Result::<(), _>::Err(error)
                    .or_invariant("mandatory append-only probe table allocation failed");
            }
            return None;
        }
        self.insert_no_grow(key, fingerprint, value)
    }

    pub(crate) fn try_insert_with_fingerprint(
        &mut self,
        key: K,
        fingerprint: u64,
        value: V,
    ) -> Result<Option<V>, ProbeReserveError> {
        self.try_reserve(1)?;
        Ok(self.insert_no_grow(key, fingerprint, value))
    }

    pub(crate) fn remove_with_fingerprint(&mut self, key: &K, fingerprint: u64) -> Option<V> {
        if self.mode != ProbeMode::Mutable {
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
        let mixed = mix64(fingerprint);
        let mut group = group_index(mixed, self.controls.len());
        let expected_tag = tag(mixed);
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

    fn insert_no_grow(&mut self, key: K, fingerprint: u64, value: V) -> Option<V> {
        let mixed = mix64(fingerprint);
        let expected_tag = tag(mixed);
        let mut group = group_index(mixed, self.controls.len());
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
                    return Some(previous);
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
                return None;
            }
            group = (group + 1) & (self.controls.len() - 1);
        }

        let index = first_deleted.or_invariant("probe table must retain insertion capacity");
        self.write_entry(
            index,
            Entry {
                key,
                value,
                fingerprint,
            },
            expected_tag,
        );
        None
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
            rebuilt.insert_no_grow(entry.key, entry.fingerprint, entry.value);
        }
        *self = rebuilt;
        Ok(())
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
mod tests {
    use std::collections::HashMap;

    use crate::RequiredExt;

    use super::{ControlGroup, ProbeKey, ProbeMode, ProbeReserveError, ProbeTable};

    impl ProbeKey for u64 {
        fn fingerprint(&self) -> u64 {
            *self
        }
    }

    #[test]
    fn control_group_matches_all_equal_lanes() {
        let group = ControlGroup([3, 9, 3, 0, 3, 9, 7, 3, 1, 2, 3, 4, 5, 6, 3, 3]);
        assert_eq!(group.matching(3), 0b1100_0100_1001_0101);
        assert_eq!(group.matching(8), 0);
    }

    #[test]
    fn collision_heavy_batch_lookup_checks_full_keys() {
        const N: usize = 32;
        let n = u64::try_from(N).or_invariant("test batch width exceeds u64");
        let mut table = ProbeTable::with_capacity(ProbeMode::AppendOnly, N);
        for key in 0..n {
            table.insert_with_fingerprint(key, 0xdead_beef, key * 11);
        }
        assert_eq!(table.get_with_fingerprint(&17, 0xdead_beef), Some(187));

        let keys = std::array::from_fn(|lane| {
            (u64::try_from(lane).or_invariant("test lane exceeds u64") * 7) % 41
        });
        let fingerprints = [0xdead_beef; N];
        let found = table.get_many_with_fingerprints(&keys, &fingerprints, 29);
        for lane in 0..29 {
            assert_eq!(found[lane], (keys[lane] < n).then_some(keys[lane] * 11));
        }
        assert!(found[29..].iter().all(Option::is_none));
    }

    #[test]
    fn mutable_table_matches_hash_map_under_adversarial_operations() {
        #[cfg(miri)]
        const STEPS: u64 = 1_000;
        #[cfg(not(miri))]
        const STEPS: u64 = 40_000;
        let mut table = ProbeTable::with_capacity(ProbeMode::Mutable, 1);
        let mut expected = HashMap::new();
        let mut state = 0x1234_5678_9abc_def0_u64;

        for step in 0..STEPS {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let key = state % 997;
            let fingerprint = key & 7;
            match state >> 61 {
                0..=3 => {
                    let value = state ^ step;
                    assert_eq!(
                        table.insert_with_fingerprint(key, fingerprint, value),
                        expected.insert(key, value)
                    );
                }
                4..=5 => assert_eq!(
                    table.remove_with_fingerprint(&key, fingerprint),
                    expected.remove(&key)
                ),
                6 => {
                    let additional = usize::try_from((state >> 8) & 31)
                        .or_invariant("bounded reserve count exceeds usize");
                    assert!(
                        table.try_reserve(additional).is_ok(),
                        "adversarial mutable-table reserve failed additional={additional}"
                    );
                }
                _ => assert_eq!(
                    table.get_with_fingerprint(&key, fingerprint),
                    expected.get(&key).copied()
                ),
            }

            if step % 257 == 0 {
                assert_eq!(table.len(), expected.len());
                for (&key, &value) in &expected {
                    assert_eq!(table.get_with_fingerprint(&key, key & 7), Some(value));
                }
            }
        }
    }

    #[test]
    fn gc_filter_keeps_probe_storage_and_removes_dead_entries_in_place() {
        let mut table = ProbeTable::with_capacity(ProbeMode::RebuildOnGc, 128);
        for key in 0..96_u64 {
            table.insert(key, key * 3);
        }
        let bytes_before = table.allocated_bytes();
        table.retain(|key, _| key % 3 == 0);
        assert_eq!(table.len(), 32);
        assert_eq!(table.allocated_bytes(), bytes_before);
        for key in 0..96_u64 {
            assert_eq!(table.get(&key), (key % 3 == 0).then_some(key * 3));
        }
    }

    #[test]
    fn append_only_gc_tombstones_are_reused_without_rehash_allocation() {
        let mut table = ProbeTable::with_capacity(ProbeMode::AppendOnly, 64);
        let initial_capacity = table.capacity();
        for key in 0..initial_capacity {
            let key = u64::try_from(key).or_invariant("test key exceeds u64");
            table.insert(key, key);
        }
        let bytes_before = table.allocated_bytes();
        table.retain_for_gc(|key, _| key % 2 == 0);

        for key in 0..initial_capacity / 2 {
            let key = u64::try_from(initial_capacity + key).or_invariant("test key exceeds u64");
            table.insert(key, key);
        }

        assert_eq!(
            table.allocated_bytes(),
            bytes_before,
            "mandatory append-only index rehashed instead of reusing GC tombstones"
        );
    }

    #[test]
    fn fallible_reserve_rejects_capacity_overflow_without_mutation() {
        let mut table = ProbeTable::with_capacity(ProbeMode::AppendOnly, 16);
        table.insert(7_u64, 11_u64);
        let bytes_before = table.allocated_bytes();

        let result = table.try_reserve(usize::MAX);

        assert!(matches!(result, Err(ProbeReserveError::CapacityOverflow)));
        assert_eq!(table.get(&7), Some(11));
        assert_eq!(table.allocated_bytes(), bytes_before);
    }

    #[test]
    fn rebuild_on_gc_filters_and_purges_without_losing_entries() {
        let mut table = ProbeTable::new(ProbeMode::RebuildOnGc);
        for key in 0..500_u64 {
            table.insert(key, key + 1);
        }
        table.retain(|key, _| key % 3 == 0);
        assert_eq!(table.len(), 167);
        for key in 0..500_u64 {
            assert_eq!(table.get(&key), (key % 3 == 0).then_some(key + 1));
        }
        assert_eq!(table.iter().count(), table.len());
    }

    #[test]
    fn clear_reuses_all_modes() {
        for mode in [
            ProbeMode::AppendOnly,
            ProbeMode::RebuildOnGc,
            ProbeMode::Mutable,
        ] {
            let mut table = ProbeTable::new(mode);
            table.insert(1_u64, 2_u64);
            assert_eq!(table.get(&1_u64), Some(2_u64));
            table.clear();
            assert_eq!(table.len(), 0);
            assert_eq!(table.insert(3_u64, 4_u64), None);
            assert_eq!(table.get(&3_u64), Some(4_u64));
        }
    }

    #[test]
    fn released_optional_storage_is_empty_and_reallocates_lazily() {
        let mut table = ProbeTable::with_capacity(ProbeMode::Mutable, 512);
        table.insert(1_u64, 2_u64);
        assert!(table.allocated_bytes() > 0);

        table.release_storage();
        assert_eq!(table.allocated_bytes(), 0);
        assert_eq!(table.get(&1), None);

        table.insert(3, 4);
        assert_eq!(table.get(&3), Some(4));
    }

    #[test]
    fn append_only_rejects_removal_without_mutation() {
        let mut table = ProbeTable::new(ProbeMode::AppendOnly);
        table.insert(1_u64, 2_u64);
        assert_eq!(table.remove(&1), None);
        assert_eq!(table.get(&1), Some(2));
    }

    #[test]
    fn append_only_rejects_filtered_rebuild_without_mutation() {
        let mut table = ProbeTable::<u64, u64>::new(ProbeMode::AppendOnly);
        table.insert(1, 2);
        assert!(!table.retain(|_, _| false));
        assert_eq!(table.get(&1), Some(2));
    }

    #[test]
    fn miri_probe_table_storage_lifecycle_preserves_initialized_slots() {
        let mut table = ProbeTable::with_capacity(ProbeMode::Mutable, 1);
        for key in 0..128_u64 {
            table.insert_with_fingerprint(key, key & 3, key + 10);
        }
        for key in (0..128_u64).step_by(2) {
            assert_eq!(table.remove_with_fingerprint(&key, key & 3), Some(key + 10));
        }
        assert!(
            table.try_reserve(256).is_ok(),
            "Miri probe-table reserve failed"
        );
        let cloned = table.clone();
        for key in 0..128_u64 {
            let expected = (key % 2 == 1).then_some(key + 10);
            assert_eq!(table.get_with_fingerprint(&key, key & 3), expected);
            assert_eq!(cloned.get_with_fingerprint(&key, key & 3), expected);
        }
        table.clear();
        table.insert(900, 901);
        assert_eq!(table.get(&900), Some(901));
    }
}
