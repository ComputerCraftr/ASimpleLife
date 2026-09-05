use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
struct OracleGrid {
    side: usize,
    cells: Vec<bool>,
}

impl OracleGrid {
    fn empty(side: usize) -> Self {
        Self {
            side,
            cells: vec![false; side * side],
        }
    }

    fn get(&self, x: usize, y: usize) -> bool {
        self.cells[y * self.side + x]
    }

    fn set(&mut self, x: usize, y: usize, alive: bool) {
        self.cells[y * self.side + x] = alive;
    }

    fn transformed(&self, symmetry: Symmetry) -> Self {
        let mut output = Self::empty(self.side);
        let max = self.side - 1;
        for y in 0..self.side {
            for x in 0..self.side {
                let (source_x, source_y) = oracle_source(symmetry, x, y, max);
                output.set(x, y, self.get(source_x, source_y));
            }
        }
        output
    }

    fn place(&mut self, source: &Self, origin_x: usize, origin_y: usize) {
        for y in 0..source.side {
            for x in 0..source.side {
                self.set(origin_x + x, origin_y + y, source.get(x, y));
            }
        }
    }

    fn morton_bits(&self) -> Vec<bool> {
        let mut bits = Vec::with_capacity(self.cells.len());
        let levels = usize::try_from(self.side.trailing_zeros())
            .or_invariant("test grid depth exceeds usize");
        for code in 0..self.cells.len() {
            let mut x = 0_usize;
            let mut y = 0_usize;
            for level in 0..levels {
                x |= ((code >> (level * 2)) & 1) << level;
                y |= ((code >> (level * 2 + 1)) & 1) << level;
            }
            bits.push(self.get(x, y));
        }
        bits
    }
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

fn node_from_grid(engine: &mut HashLifeEngine, grid: &OracleGrid) -> NodeId {
    let mut frames = vec![(0_usize, 0_usize, grid.side, false)];
    let mut nodes = Vec::with_capacity(grid.cells.len() * 2);
    while let Some((origin_x, origin_y, size, ready)) = frames.pop() {
        if size == 1 {
            nodes.push(if grid.get(origin_x, origin_y) {
                engine.live_leaf
            } else {
                engine.dead_leaf
            });
            continue;
        }
        if !ready {
            let half = size / 2;
            frames.push((origin_x, origin_y, size, true));
            frames.push((origin_x + half, origin_y + half, half, false));
            frames.push((origin_x, origin_y + half, half, false));
            frames.push((origin_x + half, origin_y, half, false));
            frames.push((origin_x, origin_y, half, false));
            continue;
        }
        let se = nodes
            .pop()
            .or_invariant("postorder stack is missing SE child");
        let sw = nodes
            .pop()
            .or_invariant("postorder stack is missing SW child");
        let ne = nodes
            .pop()
            .or_invariant("postorder stack is missing NE child");
        let nw = nodes
            .pop()
            .or_invariant("postorder stack is missing NW child");
        nodes.push(engine.join(nw, ne, sw, se));
    }
    assert_eq!(
        nodes.len(),
        1,
        "postorder construction must produce one root"
    );
    nodes
        .pop()
        .or_invariant("postorder construction produced no root")
}

fn grid_from_node(engine: &HashLifeEngine, node: NodeId, side: usize) -> OracleGrid {
    let mut grid = OracleGrid::empty(side);
    for y in 0..side {
        for x in 0..side {
            let coord_x = i64::try_from(x).or_invariant("test x coordinate exceeds i64");
            let coord_y = i64::try_from(y).or_invariant("test y coordinate exceeds i64");
            grid.set(x, y, engine.node_cell_alive(node, 0, 0, coord_x, coord_y));
        }
    }
    grid
}

fn symmetry_bit(symmetry: Symmetry) -> u8 {
    1 << (symmetry as u8)
}

fn assert_subgroup(mask: u8, case: &str, base: Symmetry) {
    assert_ne!(mask, 0, "case={case} base={base:?}");
    assert_ne!(
        mask & symmetry_bit(Symmetry::Identity),
        0,
        "identity missing: case={case} base={base:?}"
    );
    for left in Symmetry::ALL {
        if mask & symmetry_bit(left) == 0 {
            continue;
        }
        assert_ne!(
            mask & symmetry_bit(left.inverse()),
            0,
            "inverse missing: case={case} base={base:?} member={left:?}"
        );
        for right in Symmetry::ALL {
            if mask & symmetry_bit(right) != 0 {
                assert_ne!(
                    mask & symmetry_bit(left.then(right)),
                    0,
                    "closure failure: case={case} base={base:?} left={left:?} right={right:?}"
                );
            }
        }
    }
}

fn oracle_candidates(grid: &OracleGrid, base: Symmetry) -> [OracleGrid; 8] {
    let based = grid.transformed(base);
    Symmetry::ALL.map(|candidate| based.transformed(candidate))
}

fn oracle_winner(candidates: &[OracleGrid; 8]) -> usize {
    let morton = candidates.each_ref().map(OracleGrid::morton_bits);
    (1..Symmetry::ALL.len()).fold(0, |winner, candidate| {
        if morton[candidate] < morton[winner] {
            candidate
        } else {
            winner
        }
    })
}

fn oracle_mask(candidates: &[OracleGrid; 8], expected: &OracleGrid) -> u8 {
    candidates
        .iter()
        .enumerate()
        .fold(0_u8, |mask, (index, candidate)| {
            if candidate == expected {
                mask | (1_u8 << index)
            } else {
                mask
            }
        })
}

fn oracle_stabilizer(canonical: &OracleGrid) -> u8 {
    Symmetry::ALL
        .iter()
        .enumerate()
        .fold(0_u8, |mask, (index, symmetry)| {
            if canonical.transformed(*symmetry) == *canonical {
                mask | (1_u8 << index)
            } else {
                mask
            }
        })
}

fn assert_deep_case(case: &str, grid: &OracleGrid) {
    for base in Symmetry::ALL {
        // A fresh engine makes canonicalize_packed_direct exercise the cold resolver path.
        let mut engine = HashLifeEngine::default();
        let root = node_from_grid(&mut engine, grid);
        let packed = engine.node_columns.packed_key(root);
        let candidates = oracle_candidates(grid, base);
        let winner_index = oracle_winner(&candidates);
        let canonical = &candidates[winner_index];
        let aliases = oracle_mask(&candidates, canonical);
        let stabilizer = oracle_stabilizer(canonical);

        let result = engine.canonicalize_packed_direct(packed, base, false);
        assert_eq!(
            result.node.symmetry,
            Symmetry::ALL[winner_index],
            "case={case} base={base:?}"
        );
        let canonical_node = engine.materialize_packed_node_key(result.node.packed);
        assert_eq!(
            grid_from_node(&engine, canonical_node, grid.side),
            *canonical,
            "canonical materialization: case={case} base={base:?}"
        );

        let (winner, entry, structures) =
            engine.scan_canonical_transform_winner(packed, base, false);
        assert_eq!(
            winner,
            Symmetry::ALL[winner_index],
            "case={case} base={base:?}"
        );
        assert_eq!(
            entry.structural, structures[winner_index],
            "case={case} base={base:?}"
        );
        assert_eq!(entry.aliases, aliases, "case={case} base={base:?}");
        assert_eq!(entry.stabilizer, stabilizer, "case={case} base={base:?}");
        assert_subgroup(entry.stabilizer, case, base);

        let orbit_size = candidates
            .iter()
            .enumerate()
            .filter(|(index, candidate)| !candidates[..*index].contains(candidate))
            .count();
        let stabilizer_size =
            usize::try_from(stabilizer.count_ones()).or_invariant("stabilizer size exceeds usize");
        assert_eq!(orbit_size * stabilizer_size, 8, "case={case} base={base:?}");
        assert_eq!(aliases.count_ones(), stabilizer.count_ones());

        let morton = candidates.each_ref().map(OracleGrid::morton_bits);
        for left in 0..Symmetry::ALL.len() {
            for right in 0..Symmetry::ALL.len() {
                assert_eq!(
                    engine.compare_canonical_keys(structures[left], structures[right]),
                    morton[left].cmp(&morton[right]),
                    "structure order: case={case} base={base:?} left={:?} right={:?}",
                    Symmetry::ALL[left],
                    Symmetry::ALL[right]
                );
            }

            let effective = base.then(Symmetry::ALL[left]);
            let transformed = engine.transform_packed_node_key(packed, effective);
            let materialized = engine.materialize_packed_transform_root(transformed);
            assert_eq!(
                grid_from_node(&engine, materialized, grid.side),
                candidates[left],
                "oriented materialization: case={case} base={base:?} candidate={:?}",
                Symmetry::ALL[left]
            );
        }

        assert_eq!(
            canonical.transformed(winner.inverse()),
            grid.transformed(base),
            "winner inverse: case={case} base={base:?}"
        );
        let inverse_node = engine.transform_node(canonical_node, winner.inverse());
        assert_eq!(
            grid_from_node(&engine, inverse_node, grid.side),
            grid.transformed(base),
            "materialized inverse: case={case} base={base:?}"
        );
    }
}

fn random_grid(side: usize, mut state: u64, threshold: u8) -> OracleGrid {
    let mut grid = OracleGrid::empty(side);
    for y in 0..side {
        for x in 0..side {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            grid.set(x, y, state.to_le_bytes()[0] < threshold);
        }
    }
    grid
}

fn repeated_child_grid() -> OracleGrid {
    let child = random_grid(8, 0x4d59_5df4_d0f3_3173, 91);
    let mut grid = OracleGrid::empty(32);
    for quadrant_y in 0..4 {
        for quadrant_x in 0..4 {
            grid.place(&child, quadrant_x * 8, quadrant_y * 8);
        }
    }
    grid
}

fn symmetric_grid() -> OracleGrid {
    let mut grid = OracleGrid::empty(32);
    for y in 0..16 {
        for x in 0..16 {
            if (x * 19 + y * 23 + x * y) % 17 < 5 {
                for symmetry in Symmetry::ALL {
                    let (target_x, target_y) = oracle_source(symmetry, x, y, 31);
                    grid.set(target_x, target_y, true);
                }
            }
        }
    }
    grid
}

#[test]
fn deep_cold_d4_semantics_match_independent_grid_oracle() {
    let repeated = repeated_child_grid();
    let mut dag_engine = HashLifeEngine::default();
    let repeated_root = node_from_grid(&mut dag_engine, &repeated);
    let repeated_quadrants = dag_engine.node_columns.quadrants(repeated_root);
    assert!(
        repeated_quadrants[1..]
            .iter()
            .all(|quadrant| *quadrant == repeated_quadrants[0]),
        "repeated-child fixture must exercise an exact shared DAG child"
    );
    let cases = [
        (
            "random-dense-16",
            random_grid(16, 0xa076_1d64_78bd_642f, 173),
        ),
        (
            "random-sparse-32",
            random_grid(32, 0xe703_7ed1_a0b4_28db, 29),
        ),
        (
            "random-deep-32",
            random_grid(32, 0x8ebc_6af0_9c88_c6e3, 127),
        ),
        ("repeated-child-32", repeated),
        ("d4-symmetric-32", symmetric_grid()),
    ];
    for (case, grid) in &cases {
        assert_deep_case(case, grid);
    }
}

#[test]
fn parent_quadrant_and_child_local_orientation_composition_is_exact() {
    let child = random_grid(8, 0xd6e8_feb8_6659_fd93, 103);
    let mut engine = HashLifeEngine::default();

    for quadrant in 0..4 {
        let origin_x = (quadrant % 2) * child.side;
        let origin_y = (quadrant / 2) * child.side;
        for local in Symmetry::ALL {
            let mut parent = OracleGrid::empty(16);
            parent.place(&child.transformed(local), origin_x, origin_y);
            let root = node_from_grid(&mut engine, &parent);
            let packed = engine.node_columns.packed_key(root);
            let shape = engine.node_columns.identity_ref(root);
            let record = engine
                .resolve_shape_orientations(shape, u8::MAX)
                .or_invariant("deep orientation resolution should succeed");

            for parent_transform in Symmetry::ALL {
                let expected_grid = parent.transformed(parent_transform);
                let expected_node = node_from_grid(&mut engine, &expected_grid);
                assert_eq!(
                    record.reference(parent_transform),
                    engine.node_columns.identity_ref(expected_node),
                    "parent={parent_transform:?} quadrant={quadrant} local={local:?}"
                );

                let transformed = engine.transform_packed_node_key(packed, parent_transform);
                let materialized = engine.materialize_packed_transform_root(transformed);
                assert_eq!(
                    grid_from_node(&engine, materialized, parent.side),
                    expected_grid,
                    "materialized parent={parent_transform:?} quadrant={quadrant} local={local:?}"
                );
            }
        }
    }
}

fn assert_partial_rotational_proof_stays_closed(force_scalar: bool) {
    let mut pinwheel = OracleGrid::empty(4);
    for (x, y) in [(0, 1), (2, 0), (3, 2), (1, 3)] {
        pinwheel.set(x, y, true);
    }
    assert_eq!(pinwheel.transformed(Symmetry::Rotate90), pinwheel);
    assert_ne!(pinwheel.transformed(Symmetry::MirrorX), pinwheel);

    let mut engine = HashLifeEngine::default();
    let root = node_from_grid(&mut engine, &pinwheel);
    let shape = engine.node_columns.identity_ref(root);
    let rotation_record = if force_scalar {
        engine.resolve_shape_orientations_with_quotient_limit(
            shape,
            symmetry_bit(Symmetry::Rotate90),
            0,
        )
    } else {
        engine.resolve_shape_orientations(shape, symmetry_bit(Symmetry::Rotate90))
    }
    .or_invariant("partial rotational orientation proof should succeed");
    assert_eq!(rotation_record.reference(Symmetry::Rotate90), shape);

    let rotations = symmetry_bit(Symmetry::Identity)
        | symmetry_bit(Symmetry::Rotate90)
        | symmetry_bit(Symmetry::Rotate180)
        | symmetry_bit(Symmetry::Rotate270);
    let after_rotation = engine.canonical_caches.shapes[shape.index()].stabilizer;
    assert_eq!(
        after_rotation, rotations,
        "a proved generator must close C4"
    );
    assert_subgroup(after_rotation, "partial-pinwheel-r90", Symmetry::Identity);

    let reflection_record = if force_scalar {
        engine.resolve_shape_orientations_with_quotient_limit(
            shape,
            symmetry_bit(Symmetry::MirrorX),
            0,
        )
    } else {
        engine.resolve_shape_orientations(shape, symmetry_bit(Symmetry::MirrorX))
    }
    .or_invariant("separate reflection orientation proof should succeed");
    let mirrored = pinwheel.transformed(Symmetry::MirrorX);
    let mirrored_node = node_from_grid(&mut engine, &mirrored);
    assert_eq!(
        reflection_record.reference(Symmetry::MirrorX),
        engine.node_columns.identity_ref(mirrored_node)
    );
    assert_ne!(reflection_record.reference(Symmetry::MirrorX), shape);

    let after_reflection = engine.canonical_caches.shapes[shape.index()].stabilizer;
    assert_eq!(
        after_reflection, rotations,
        "proving a distinct reflection must preserve the exact C4 stabilizer"
    );
    assert_subgroup(
        after_reflection,
        "partial-pinwheel-reflection",
        Symmetry::Identity,
    );
}

#[test]
fn partial_rotational_proofs_store_closed_subgroups_on_both_resolvers() {
    assert_partial_rotational_proof_stays_closed(false);
    assert_partial_rotational_proof_stays_closed(true);
}

#[test]
fn asymmetric_work_cutoff_bounds_retry_work_without_extra_shapes() {
    let grid = random_grid(64, 0x7612_0459_a38b_09cf, 127);
    for transform in Symmetry::ALL.into_iter().skip(1) {
        assert_ne!(
            grid.transformed(transform),
            grid,
            "fixture must be asymmetric"
        );
    }
    let mut bounded = HashLifeEngine::default();
    let root = node_from_grid(&mut bounded, &grid);
    let children = bounded.node_columns.quadrants(root);
    assert!(
        children
            .iter()
            .enumerate()
            .all(|(index, child)| !children[..index].contains(child)),
        "fixture must have four distinct root children"
    );
    let shape = bounded.node_columns.identity_ref(root);
    let actual = bounded
        .resolve_shape_orientations(shape, u8::MAX)
        .or_invariant("bounded asymmetric orientation proof");
    let stats = bounded.stats.transform;
    assert_eq!(stats.orientation_work_bypasses, 1);
    assert_eq!(stats.orientation_scratch_bypasses, 0);

    let mut unrestricted = HashLifeEngine::default();
    let expected_root = node_from_grid(&mut unrestricted, &grid);
    let expected_shape = unrestricted.node_columns.identity_ref(expected_root);
    let expected = unrestricted
        .resolve_shape_orientations_with_quotient_limit(expected_shape, u8::MAX, usize::MAX)
        .or_invariant("unrestricted asymmetric orientation proof");
    let baseline = unrestricted.stats.transform;
    assert_eq!(baseline.orientation_work_bypasses, 0);
    assert_eq!(
        bounded.canonical_caches.shapes.len(),
        unrestricted.canonical_caches.shapes.len(),
        "cutoff must not intern extra structural identities"
    );
    for transform in Symmetry::ALL {
        let expected_grid = grid.transformed(transform);
        let expected_node = node_from_grid(&mut bounded, &expected_grid);
        assert_eq!(
            actual.reference(transform),
            bounded.node_columns.identity_ref(expected_node),
            "cutoff changed exact cells for {transform:?}"
        );
        let expected_node = node_from_grid(&mut unrestricted, &expected_grid);
        assert_eq!(
            expected.reference(transform),
            unrestricted.node_columns.identity_ref(expected_node)
        );
    }
    // Only unfinished ancestor frames may need fresh uncached proof requests.
    // Completed descendants must survive the switch through optional records.
    let depth = 6;
    assert!(
        stats.orientation_uncached_requests <= baseline.orientation_uncached_requests + 8 * depth,
        "cutoff repeated completed proofs: bounded={stats:?} baseline={baseline:?}"
    );
    assert!(
        stats.orientation_requests <= baseline.orientation_requests + 8 * 4 * depth,
        "restart must not revisit more than the unfinished ancestor frontier: bounded={stats:?} baseline={baseline:?}"
    );
    eprintln!(
        "D4 asymmetric cutoff requests={}/{} uncached={}/{} signatures={}/{}",
        stats.orientation_requests,
        baseline.orientation_requests,
        stats.orientation_uncached_requests,
        baseline.orientation_uncached_requests,
        stats.orientation_resolved_signatures,
        baseline.orientation_resolved_signatures,
    );
}
