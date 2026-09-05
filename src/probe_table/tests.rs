use std::cell::Cell;
use std::collections::HashMap;

use crate::RequiredExt;

use super::{ControlGroup, Entry, ProbeKey, ProbeMode, ProbeReserveError, ProbeTable, tag};

impl ProbeKey for u64 {
    fn fingerprint(&self) -> u64 {
        crate::hashing::mix64(*self)
    }
}

#[test]
fn control_group_matches_all_equal_lanes() {
    let group = ControlGroup([3, 9, 3, 0, 3, 9, 7, 3, 1, 2, 3, 4, 5, 6, 3, 3]);
    assert_eq!(group.matching(3), 0b1100_0100_1001_0101);
    assert_eq!(group.matching(8), 0);
}

#[test]
fn swar_control_matching_matches_scalar_for_every_byte_value() {
    for needle in 0..=u8::MAX {
        let bytes = std::array::from_fn(|lane| {
            needle
                .wrapping_add(u8::try_from(lane).or_invariant("control lane exceeds u8"))
                .wrapping_sub(3)
        });
        let actual = ControlGroup(bytes).matching(needle);
        let expected = bytes.iter().enumerate().fold(0_u16, |mask, (lane, &byte)| {
            mask | (u16::from(byte == needle) << lane)
        });
        assert_eq!(
            actual, expected,
            "SWAR control match disagreed with scalar lanes needle={needle} bytes={bytes:?}"
        );
    }
}

#[test]
fn swar_control_matching_preserves_every_lane_mask() {
    for mask in 0..=u16::MAX {
        let controls = std::array::from_fn(|lane| if mask & (1 << lane) != 0 { 7 } else { 8 });
        assert_eq!(ControlGroup(controls).matching(7), mask, "mask={mask:#06x}");
    }
}

#[test]
fn saturated_control_groups_terminate_with_typed_full_result() {
    let mut table = ProbeTable::with_capacity(ProbeMode::Scratch, 1);
    for index in 0..table.entries.len() {
        let key = u64::try_from(index).or_invariant("test slot exceeds u64");
        let fingerprint = key;
        table.entries[index].write(Entry {
            key,
            value: key,
            fingerprint,
        });
        table.set_control(index, tag(fingerprint));
    }
    table.len = table.entries.len();

    assert_eq!(table.get_with_fingerprint(&u64::MAX, u64::MAX), None);
    assert_eq!(
        table.insert_no_grow(u64::MAX, u64::MAX, 1),
        Err(ProbeReserveError::Full),
        "a table with no empty or deleted control byte must report Full"
    );
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
fn batch_lookup_matches_duplicate_full_keys_once_per_group() {
    const N: usize = 8;
    let fingerprint = 0xdead_beef_u64;
    let mut table = ProbeTable::with_capacity(ProbeMode::AppendOnly, N);
    table.insert_with_fingerprint(1, fingerprint, 11);
    table.insert_with_fingerprint(2, fingerprint, 22);
    table.insert_with_fingerprint(3, fingerprint, 33);

    let keys = [1, 1, 2, 2, 3, 3, 4, 4];
    let fingerprints = [fingerprint; N];
    let matched_queries = Cell::new(0);
    let found = table.get_many_with_fingerprints_using(
        &keys,
        &fingerprints,
        N,
        |controls, tags, active_lanes| {
            matched_queries.set(matched_queries.get() + active_lanes);
            super::match_control_groups_swar(controls, tags, active_lanes)
        },
    );

    assert_eq!(
        found,
        [
            Some(11),
            Some(11),
            Some(22),
            Some(22),
            Some(33),
            Some(33),
            None,
            None
        ]
    );
    assert_eq!(
        matched_queries.get(),
        4,
        "duplicate full keys were redundantly sent through the control matcher"
    );
}

#[test]
fn updating_at_growth_threshold_does_not_rehash() {
    for mode in [
        ProbeMode::AppendOnly,
        ProbeMode::RebuildOnGc,
        ProbeMode::Mutable,
        ProbeMode::Scratch,
    ] {
        let mut table = ProbeTable::with_capacity(mode, 1);
        for key in 0..table.capacity() {
            let key = u64::try_from(key).or_invariant("test key exceeds u64");
            assert_eq!(table.insert(key, key + 10), None);
        }
        let bytes_before = table.allocated_bytes();

        assert_eq!(table.insert(7, 700), Some(17));
        assert_eq!(table.get(&7), Some(700));
        assert_eq!(table.allocated_bytes(), bytes_before, "mode={mode:?}");
    }
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
fn scratch_table_matches_hash_map_under_reuse_and_tombstones() {
    let mut table = ProbeTable::with_capacity(ProbeMode::Scratch, 1);
    let mut expected = HashMap::new();
    for round in 0..64_u64 {
        for key in 0..97_u64 {
            let value = round * 100 + key;
            assert_eq!(table.insert(key, value), expected.insert(key, value));
        }
        for key in (round % 3..97_u64).step_by(3) {
            assert_eq!(table.remove(&key), expected.remove(&key));
        }
        for key in 0..97_u64 {
            assert_eq!(
                table.get(&key),
                expected.get(&key).copied(),
                "scratch lookup drifted after round={round} key={key}"
            );
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
        ProbeMode::Scratch,
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
