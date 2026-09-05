use super::*;

const EXHAUSTIVE_BATCH_SIZE: u32 = 4_096;
const MORTON_4X4: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

fn node_from_4x4_bits(engine: &mut HashLifeEngine, bits: u16) -> NodeId {
    let quadrants: [NodeId; 4] = std::array::from_fn(|quadrant| {
        let origin_x = (quadrant % 2) * 2;
        let origin_y = (quadrant / 2) * 2;
        let leaves: [NodeId; 4] = std::array::from_fn(|child| {
            let x = origin_x + child % 2;
            let y = origin_y + child / 2;
            if bits & (1_u16 << (y * 4 + x)) == 0 {
                engine.dead_leaf
            } else {
                engine.live_leaf
            }
        });
        engine.join(leaves[0], leaves[1], leaves[2], leaves[3])
    });
    engine.join(quadrants[0], quadrants[1], quadrants[2], quadrants[3])
}

fn orient_4x4(bits: u16, symmetry: Symmetry) -> u16 {
    let mut output = 0_u16;
    for y in 0..4 {
        for x in 0..4 {
            let (source_x, source_y) = oracle_source(symmetry, x, y, 3);
            if bits & (1_u16 << (source_y * 4 + source_x)) != 0 {
                output |= 1_u16 << (y * 4 + x);
            }
        }
    }
    output
}

fn oracle_source(symmetry: Symmetry, x: usize, y: usize, max: usize) -> (usize, usize) {
    match symmetry {
        Symmetry::Identity => (x, y),
        Symmetry::Rotate90 => (max - y, x),
        Symmetry::Rotate180 => (max - x, max - y),
        Symmetry::Rotate270 => (y, max - x),
        Symmetry::MirrorX => (max - x, y),
        Symmetry::MirrorXRotate90 => (max - y, max - x),
        Symmetry::MirrorXRotate180 => (x, max - y),
        Symmetry::MirrorXRotate270 => (y, x),
    }
}

fn structural_rank_4x4(bits: u16) -> u16 {
    MORTON_4X4
        .into_iter()
        .fold(0, |rank, bit| (rank << 1) | ((bits >> bit) & 1))
}

fn oracle_4x4(bits: u16, base: Symmetry) -> ([u16; 8], usize, u8, u8) {
    let base_bits = orient_4x4(bits, base);
    let candidates = Symmetry::ALL.map(|symmetry| orient_4x4(base_bits, symmetry));
    let winner = (1..Symmetry::ALL.len()).fold(0, |winner, candidate| {
        if structural_rank_4x4(candidates[candidate]) < structural_rank_4x4(candidates[winner]) {
            candidate
        } else {
            winner
        }
    });
    let aliases = candidates
        .iter()
        .enumerate()
        .fold(0_u8, |mask, (index, candidate)| {
            if *candidate == candidates[winner] {
                mask | (1_u8 << index)
            } else {
                mask
            }
        });
    let stabilizer = Symmetry::ALL
        .iter()
        .enumerate()
        .fold(0_u8, |mask, (index, symmetry)| {
            if orient_4x4(candidates[winner], *symmetry) == candidates[winner] {
                mask | (1_u8 << index)
            } else {
                mask
            }
        });
    (candidates, winner, aliases, stabilizer)
}

fn assert_scan_matches_4x4_oracle(
    engine: &mut HashLifeEngine,
    packed: PackedNodeKey,
    bits: u16,
    base: Symmetry,
) -> (u16, usize, CanonicalStructKey) {
    let (expected_bits, winner_index, aliases, stabilizer) = oracle_4x4(bits, base);
    let (winner, entry, candidates) = engine.scan_canonical_transform_winner(packed, base, false);
    assert_eq!(
        winner,
        Symmetry::ALL[winner_index],
        "bits={bits:#06x} base={base:?}"
    );
    assert_eq!(
        entry.structural, candidates[winner_index],
        "bits={bits:#06x} base={base:?}"
    );
    assert_eq!(entry.aliases, aliases, "bits={bits:#06x} base={base:?}");
    assert_eq!(
        entry.stabilizer, stabilizer,
        "bits={bits:#06x} base={base:?}"
    );
    assert_eq!(
        winner_index,
        usize::try_from(aliases.trailing_zeros()).or_invariant("alias index exceeds usize"),
        "bits={bits:#06x} base={base:?}"
    );

    for left in 0..Symmetry::ALL.len() {
        let expected_order = structural_rank_4x4(expected_bits[left])
            .cmp(&structural_rank_4x4(expected_bits[winner_index]));
        assert_eq!(
            engine.compare_canonical_keys(candidates[left], entry.structural),
            expected_order,
            "bits={bits:#06x} base={base:?} candidate={:?}",
            Symmetry::ALL[left]
        );
        if base == Symmetry::Identity {
            for right in 0..Symmetry::ALL.len() {
                let expected_pair_order = structural_rank_4x4(expected_bits[left])
                    .cmp(&structural_rank_4x4(expected_bits[right]));
                assert_eq!(
                    engine.compare_canonical_keys(candidates[left], candidates[right]),
                    expected_pair_order,
                    "bits={bits:#06x} left={:?} right={:?}",
                    Symmetry::ALL[left],
                    Symmetry::ALL[right]
                );
            }
        } else {
            assert_eq!(
                candidates[left] == entry.structural,
                expected_bits[left] == expected_bits[winner_index],
                "bits={bits:#06x} base={base:?} candidate={:?}",
                Symmetry::ALL[left]
            );
        }
    }

    let base_bits = orient_4x4(bits, base);
    let canonical_bits = expected_bits[winner_index];
    assert_eq!(
        orient_4x4(canonical_bits, winner.inverse()),
        base_bits,
        "bits={bits:#06x} base={base:?} inverse={:?}",
        winner.inverse()
    );
    assert_eq!(
        orient_4x4(orient_4x4(base_bits, winner), winner.inverse()),
        base_bits,
        "bits={bits:#06x} base={base:?}"
    );

    let mut orbit = [0_u16; 8];
    let mut orbit_len = 0_usize;
    for candidate in expected_bits {
        if !orbit[..orbit_len].contains(&candidate) {
            orbit[orbit_len] = candidate;
            orbit_len += 1;
        }
    }
    let stabilizer_size =
        usize::try_from(stabilizer.count_ones()).or_invariant("D4 stabilizer size exceeds usize");
    assert_eq!(
        orbit_len * stabilizer_size,
        8,
        "bits={bits:#06x} base={base:?}"
    );
    assert_eq!(
        aliases.count_ones(),
        stabilizer.count_ones(),
        "bits={bits:#06x} base={base:?}"
    );
    (canonical_bits, winner_index, entry.structural)
}

#[test]
fn exhaustive_4x4_quotient_semantics_match_independent_cell_oracle() {
    for batch_start in (0..=u32::from(u16::MAX))
        .step_by(usize::try_from(EXHAUSTIVE_BATCH_SIZE).or_invariant("batch size exceeds usize"))
    {
        let batch_end = (batch_start + EXHAUSTIVE_BATCH_SIZE).min(1_u32 << 16);
        let mut engine = HashLifeEngine::default();
        let mut roots = Vec::with_capacity(
            usize::try_from(batch_end - batch_start).or_invariant("batch length exceeds usize"),
        );
        for raw_bits in batch_start..batch_end {
            let bits = u16::try_from(raw_bits).or_invariant("4x4 pattern exceeds u16");
            roots.push(node_from_4x4_bits(&mut engine, bits));
        }

        for (offset, root) in roots.into_iter().enumerate() {
            let raw_bits =
                batch_start + u32::try_from(offset).or_invariant("batch offset exceeds u32");
            let bits = u16::try_from(raw_bits).or_invariant("4x4 pattern exceeds u16");
            let packed = engine.node_columns.packed_key(root);
            for base in Symmetry::ALL {
                let (canonical_bits, winner_index, canonical_structure) =
                    assert_scan_matches_4x4_oracle(&mut engine, packed, bits, base);
                let expected = node_from_4x4_bits(&mut engine, canonical_bits);
                let expected_packed = engine.node_columns.packed_key(expected);
                let result = engine.canonicalize_packed_direct(packed, base, false);
                assert_eq!(
                    result.node.packed, expected_packed,
                    "bits={bits:#06x} base={base:?}"
                );
                assert_eq!(
                    result.node.structural, canonical_structure,
                    "bits={bits:#06x} base={base:?}"
                );
                assert_eq!(
                    result.node.symmetry,
                    Symmetry::ALL[winner_index],
                    "bits={bits:#06x} base={base:?}"
                );
            }
        }
    }
}

fn quadrant_4x4(bits: u64, quadrant: usize) -> u16 {
    let origin_x = (quadrant % 2) * 4;
    let origin_y = (quadrant / 2) * 4;
    let mut result = 0_u16;
    for y in 0..4 {
        for x in 0..4 {
            if bits & (1_u64 << ((origin_y + y) * 8 + origin_x + x)) != 0 {
                result |= 1_u16 << (y * 4 + x);
            }
        }
    }
    result
}

fn node_from_8x8_bits(engine: &mut HashLifeEngine, bits: u64) -> NodeId {
    let quadrants: [NodeId; 4] =
        std::array::from_fn(|index| node_from_4x4_bits(engine, quadrant_4x4(bits, index)));
    engine.join(quadrants[0], quadrants[1], quadrants[2], quadrants[3])
}

fn orient_8x8(bits: u64, symmetry: Symmetry) -> u64 {
    let mut output = 0_u64;
    for y in 0..8 {
        for x in 0..8 {
            let (source_x, source_y) = oracle_source(symmetry, x, y, 7);
            if bits & (1_u64 << (source_y * 8 + source_x)) != 0 {
                output |= 1_u64 << (y * 8 + x);
            }
        }
    }
    output
}

fn structural_rank_8x8(bits: u64) -> u64 {
    let mut rank = 0_u64;
    for quadrant in 0..4 {
        let quadrant_bits = quadrant_4x4(bits, quadrant);
        for bit in MORTON_4X4 {
            rank = (rank << 1) | u64::from((quadrant_bits >> bit) & 1);
        }
    }
    rank
}

#[test]
fn recursively_asymmetric_8x8_has_exact_quotient_winner_and_inverse() {
    let bits = 0x8124_0a03_4880_2519_u64;
    let mut engine = HashLifeEngine::default();
    let root = node_from_8x8_bits(&mut engine, bits);
    let packed = engine.node_columns.packed_key(root);
    let oriented = Symmetry::ALL.map(|symmetry| orient_8x8(bits, symmetry));
    assert_eq!(
        oriented
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        8,
        "fixture must have a trivial stabilizer"
    );
    let canonical_bits = oriented
        .into_iter()
        .min_by_key(|candidate| structural_rank_8x8(*candidate))
        .or_invariant("D4 orbit must not be empty");
    let expected = node_from_8x8_bits(&mut engine, canonical_bits);
    let expected_packed = engine.node_columns.packed_key(expected);

    for base in Symmetry::ALL {
        let base_bits = orient_8x8(bits, base);
        let candidates = Symmetry::ALL.map(|symmetry| orient_8x8(base_bits, symmetry));
        let winner_index = candidates
            .iter()
            .position(|candidate| *candidate == canonical_bits)
            .or_invariant("asymmetric orbit must contain its canonical member");
        let (winner, entry, structures) =
            engine.scan_canonical_transform_winner(packed, base, false);
        assert_eq!(winner, Symmetry::ALL[winner_index], "base={base:?}");
        assert_eq!(entry.structural, structures[winner_index], "base={base:?}");
        assert_eq!(entry.aliases, 1_u8 << winner_index, "base={base:?}");
        assert_eq!(entry.stabilizer, 1, "base={base:?}");
        assert_eq!(
            orient_8x8(canonical_bits, winner.inverse()),
            base_bits,
            "base={base:?}"
        );

        let result = engine.canonicalize_packed_direct(packed, base, false);
        assert_eq!(result.node.packed, expected_packed, "base={base:?}");
        assert_eq!(result.node.symmetry, winner, "base={base:?}");
    }
}
