#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FlatEntry<K, V> {
    key: K,
    value: V,
}

pub(crate) trait FlatKey: Copy + Eq {
    fn fingerprint(&self) -> u64;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlatTable<K: FlatKey, V: Copy> {
    entries: Vec<Option<FlatEntry<K, V>>>,
    len: usize,
}

impl<K: FlatKey, V: Copy> FlatTable<K, V> {
    pub(crate) fn new() -> Self {
        Self::with_capacity(64)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let slots = capacity.next_power_of_two().max(16);
        Self {
            entries: vec![None; slots],
            len: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(None);
        self.len = 0;
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
        let mut index = (fingerprint as usize) & mask;
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
        let mut index = (fingerprint as usize) & mask;
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

    pub(crate) fn get_many_with_fingerprints<const N: usize>(
        &self,
        keys: &[K; N],
        fingerprints: &[u64; N],
        active_lanes: usize,
    ) -> [Option<V>; N] {
        let mut values = [None; N];
        for lane in 0..active_lanes {
            values[lane] = self.get_with_fingerprint(&keys[lane], fingerprints[lane]);
        }
        values
    }

    pub(crate) fn insert(&mut self, key: K, value: V) {
        if (self.len + 1) * 10 >= self.entries.len() * 7 {
            self.rehash(self.entries.len() * 2);
        }
        self.insert_no_grow_with_fingerprint(key, key.fingerprint(), value);
    }

    pub(crate) fn insert_with_fingerprint(&mut self, key: K, fingerprint: u64, value: V) {
        if (self.len + 1) * 10 >= self.entries.len() * 7 {
            self.rehash(self.entries.len() * 2);
        }
        self.insert_no_grow_with_fingerprint(key, fingerprint, value);
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

    fn rehash(&mut self, new_capacity: usize) {
        let mut rebuilt = Self::with_capacity(new_capacity);
        for entry in self.entries.iter().flatten().copied() {
            rebuilt.insert_no_grow(entry.key, entry.value);
        }
        *self = rebuilt;
    }
}
