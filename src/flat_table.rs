use crate::RequiredExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlatTableAllocationError {
    CapacityOverflow,
    AllocationFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FlatEntry<K, V> {
    key: K,
    value: V,
}

pub(crate) trait FlatKey: Copy + Eq {
    fn fingerprint(&self) -> u64;
}

fn bucket_index(fingerprint: u64, mask: usize) -> usize {
    let mask = u64::try_from(mask).or_invariant("flat-table mask exceeds u64");
    usize::try_from(crate::hashing::mix64(fingerprint) & mask)
        .or_invariant("masked flat-table bucket exceeds usize")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlatTable<K: FlatKey, V: Copy> {
    entries: Vec<Option<FlatEntry<K, V>>>,
    len: usize,
}

impl<K: FlatKey, V: Copy> FlatTable<K, V> {
    #[cfg(test)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).or_invariant("flat-table allocation failed")
    }

    pub(crate) fn try_with_capacity(capacity: usize) -> Result<Self, FlatTableAllocationError> {
        let slots = capacity
            .checked_next_power_of_two()
            .ok_or(FlatTableAllocationError::CapacityOverflow)?
            .max(16);
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(slots)
            .map_err(|_| FlatTableAllocationError::AllocationFailed)?;
        entries.resize(slots, None);
        Ok(Self { entries, len: 0 })
    }

    pub(crate) fn allocation_bytes_for_capacity(capacity: usize) -> Option<u128> {
        let slots = capacity.checked_next_power_of_two()?.max(16);
        (slots as u128).checked_mul(std::mem::size_of::<Option<FlatEntry<K, V>>>() as u128)
    }

    pub(crate) fn growth_bytes_for_insert(&self) -> Result<u128, FlatTableAllocationError> {
        if (self.len + 1) * 10 < self.entries.len() * 7 {
            return Ok(0);
        }
        let new_slots = self
            .entries
            .len()
            .checked_mul(2)
            .ok_or(FlatTableAllocationError::CapacityOverflow)?;
        (new_slots as u128)
            .checked_mul(std::mem::size_of::<Option<FlatEntry<K, V>>>() as u128)
            .ok_or(FlatTableAllocationError::CapacityOverflow)
    }

    pub(crate) fn try_reserve_insert(&mut self) -> Result<(), FlatTableAllocationError> {
        if (self.len + 1) * 10 >= self.entries.len() * 7 {
            self.try_rehash(
                self.entries
                    .len()
                    .checked_mul(2)
                    .ok_or(FlatTableAllocationError::CapacityOverflow)?,
            )?;
        }
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (K, V)> + '_ {
        self.entries
            .iter()
            .flatten()
            .copied()
            .map(|entry| (entry.key, entry.value))
    }

    fn find_index(&self, key: &K, fingerprint: u64) -> Option<usize> {
        let mask = self.entries.len() - 1;
        let mut index = bucket_index(fingerprint, mask);
        loop {
            match self.entries[index] {
                Some(entry) if entry.key == *key => return Some(index),
                Some(_) => index = (index + 1) & mask,
                None => return None,
            }
        }
    }

    fn insertion_index(&self, key: &K, fingerprint: u64) -> usize {
        let mask = self.entries.len() - 1;
        let mut index = bucket_index(fingerprint, mask);
        loop {
            match self.entries[index] {
                Some(entry) if entry.key == *key => return index,
                Some(_) => index = (index + 1) & mask,
                None => return index,
            }
        }
    }

    pub(crate) fn get(&self, key: &K) -> Option<V> {
        self.get_with_fingerprint(key, key.fingerprint())
    }

    pub(crate) fn get_with_fingerprint(&self, key: &K, fingerprint: u64) -> Option<V> {
        self.find_index(key, fingerprint)
            .and_then(|index| self.entries[index].map(|entry| entry.value))
    }

    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub(crate) fn try_insert(&mut self, key: K, value: V) -> Result<(), FlatTableAllocationError> {
        if (self.len + 1) * 10 >= self.entries.len() * 7 {
            self.try_rehash(self.entries.len() * 2)?;
        }
        self.insert_no_grow_with_fingerprint(key, key.fingerprint(), value);
        Ok(())
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let fingerprint = key.fingerprint();
        let index = self.find_index(key, fingerprint)?;
        let removed = self.entries[index].take().map(|entry| entry.value)?;
        self.len -= 1;

        let mask = self.entries.len() - 1;
        let mut cursor = (index + 1) & mask;
        while let Some(entry) = self.entries[cursor].take() {
            self.len -= 1;
            self.insert_no_grow(entry.key, entry.value);
            cursor = (cursor + 1) & mask;
        }
        Some(removed)
    }

    fn insert_no_grow(&mut self, key: K, value: V) {
        self.insert_no_grow_with_fingerprint(key, key.fingerprint(), value);
    }

    fn insert_no_grow_with_fingerprint(&mut self, key: K, fingerprint: u64, value: V) {
        let index = self.insertion_index(&key, fingerprint);
        match &mut self.entries[index] {
            Some(entry) => {
                entry.value = value;
            }
            slot @ None => {
                *slot = Some(FlatEntry { key, value });
                self.len += 1;
            }
        }
    }

    fn try_rehash(&mut self, new_capacity: usize) -> Result<(), FlatTableAllocationError> {
        let mut rebuilt = Self::try_with_capacity(new_capacity)?;
        for entry in self.entries.iter().flatten().copied() {
            rebuilt.insert_no_grow(entry.key, entry.value);
        }
        *self = rebuilt;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{FlatKey, FlatTable, FlatTableAllocationError, bucket_index};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct GridKey {
        x: u16,
        y: u16,
    }

    impl FlatKey for GridKey {
        fn fingerprint(&self) -> u64 {
            (u64::from(self.x) << 48) | (u64::from(self.y) << 32)
        }
    }

    #[test]
    fn flat_table_finalizer_spreads_adversarial_grid_fingerprints() {
        const SIDE: u16 = 64;
        const CAPACITY: usize = 8_192;
        let mask = CAPACITY - 1;
        let mut buckets = HashSet::new();
        let mut table = FlatTable::with_capacity(CAPACITY);

        for x in 0..SIDE {
            for y in 0..SIDE {
                let key = GridKey { x, y };
                buckets.insert(bucket_index(key.fingerprint(), mask));
                assert!(
                    table
                        .try_insert(key, u32::from(x) * u32::from(SIDE) + u32::from(y))
                        .is_ok(),
                    "adversarial mixer fixture should fit its preallocated table"
                );
            }
        }

        assert!(
            buckets.len() > 3_000,
            "hash finalization should spread a 64x64 high-bit grid; distinct_buckets={}",
            buckets.len()
        );
        for x in 0..SIDE {
            for y in 0..SIDE {
                let key = GridKey { x, y };
                assert_eq!(
                    table.get(&key),
                    Some(u32::from(x) * u32::from(SIDE) + u32::from(y)),
                    "grid key lookup failed for ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn flat_table_rejects_capacity_overflow_before_allocation() {
        assert_eq!(
            FlatTable::<GridKey, u32>::try_with_capacity(usize::MAX),
            Err(FlatTableAllocationError::CapacityOverflow)
        );
    }
}
