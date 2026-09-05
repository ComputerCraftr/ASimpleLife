use super::*;

#[cfg(test)]
impl HashLifeEngine {
    pub(crate) fn verify_overlap_batch_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut nodes = [super::NodeId::ZERO; crate::simd_layout::SIMD_BATCH_LANES];
        let mut active = 0;
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) < 2 {
                continue;
            }
            nodes[active] = node;
            active += 1;
            if active == crate::simd_layout::SIMD_BATCH_LANES {
                break;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.push(se);
            stack.push(sw);
            stack.push(ne);
            stack.push(nw);
        }

        if active == 0 {
            return true;
        }
        let batched = self.probe_and_build_overlaps_staged(&nodes, active);
        (0..active).all(|lane| batched[lane] == self.overlapping_subnodes(nodes[lane]))
    }

    pub(crate) fn verify_canonical_overlap_batch_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let mut nodes = [super::NodeId::ZERO; crate::simd_layout::SIMD_BATCH_LANES];
        let mut canonical_keys =
            [super::CanonicalJumpKey::empty(); crate::simd_layout::SIMD_BATCH_LANES];
        let mut canonical_packed = [super::PackedNodeKey::new(0, [super::NodeId::ZERO; 4]);
            crate::simd_layout::SIMD_BATCH_LANES];
        let mut active = 0;
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if self.node_columns.level(node) < 2 {
                continue;
            }
            nodes[active] = node;
            let canonical = self.canonicalize_packed_node(node);
            canonical_keys[active] = super::CanonicalJumpKey {
                structural: canonical.node.structural,
                step_exp: 1,
                symmetry_admitted: true,
            };
            canonical_packed[active] = canonical.node.packed;
            active += 1;
            if active == crate::simd_layout::SIMD_BATCH_LANES {
                break;
            }
            let [nw, ne, sw, se] = self.node_columns.quadrants(node);
            stack.push(se);
            stack.push(sw);
            stack.push(ne);
            stack.push(nw);
        }

        if active == 0 {
            return true;
        }
        let mut canonical_nodes = [super::NodeId::ZERO; crate::simd_layout::SIMD_BATCH_LANES];
        for lane in 0..active {
            canonical_nodes[lane] = self.materialize_packed_node_key(canonical_packed[lane]);
        }
        let raw = self.probe_and_build_overlaps_staged(&canonical_nodes, active);
        let mut identities = [super::CanonicalNodeIdentity {
            packed: super::PackedNodeKey::new(0, [super::NodeId::ZERO; 4]),
            structural: super::CanonicalStructKey::leaf(false),
            symmetry: super::Symmetry::Identity,
        }; crate::simd_layout::SIMD_BATCH_LANES];
        let mut fingerprints = [0_u64; crate::simd_layout::SIMD_BATCH_LANES];
        for lane in 0..active {
            identities[lane] = super::CanonicalNodeIdentity {
                packed: canonical_packed[lane],
                structural: canonical_keys[lane].structural,
                symmetry: super::Symmetry::Identity,
            };
            fingerprints[lane] = ProbeKey::fingerprint(&canonical_keys[lane].structural);
        }
        self.stats.scheduler.cache_probe_batches += 1;
        self.stats.scheduler.scheduler_probe_batches += 1;
        self.stats.simd.overlap_prep_batches += 1;
        self.stats.transform.packed_overlap_outputs_produced += active;
        let Some(canonical) =
            self.probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, active)
        else {
            return false;
        };
        (0..active).all(|lane| raw[lane] == canonical[lane])
    }

    pub(crate) fn verify_canonical_child_key_batch_parity(&mut self, grid: &BitGrid) -> bool {
        let (root, _, _) = self.embed_grid_state(grid);
        let overlaps = self.overlapping_subnodes(root);
        let nodes = [
            overlaps[8],
            overlaps[7],
            overlaps[6],
            overlaps[5],
            overlaps[4],
            overlaps[3],
            overlaps[2],
            overlaps[1],
            overlaps[0],
        ];
        let batched = self.discovered_jump_tasks_from_nodes(nodes, 2);
        (0..9).all(|lane| batched[lane].key == self.canonical_jump_probe((nodes[lane], 2)).key)
    }

    pub(crate) fn duplicate_overlap_batch_dedupe_stats(
        &mut self,
        grid: &BitGrid,
    ) -> (usize, usize) {
        let (root, _, _) = self.embed_grid_state(grid);
        let before = (
            self.stats.result_cache.overlap_cache_misses,
            self.stats.simd.overlap_local_reuse_lanes,
        );
        let mut nodes = [super::NodeId::ZERO; crate::simd_layout::SIMD_BATCH_LANES];
        nodes[0] = root;
        nodes[1] = root;
        let overlaps = self.probe_and_build_overlaps_staged(&nodes, 2);
        assert_eq!(overlaps[0], overlaps[1]);
        (
            self.stats.result_cache.overlap_cache_misses - before.0,
            self.stats.simd.overlap_local_reuse_lanes - before.1,
        )
    }

    pub(crate) fn centered_overlap_full_batch_work(
        &mut self,
        grids: &[BitGrid; crate::simd_layout::SIMD_BATCH_LANES],
    ) -> (bool, bool, usize) {
        let mut overlap_lanes = [[super::NodeId::ZERO; 9]; crate::simd_layout::SIMD_BATCH_LANES];
        for (lane, grid) in grids.iter().enumerate() {
            let (root, _, _) = self.embed_grid_state(grid);
            overlap_lanes[lane] = self.overlapping_subnodes(root);
        }
        let distinct_inputs = overlap_lanes[1..]
            .iter()
            .any(|overlaps| overlaps != &overlap_lanes[0]);
        let before = self.stats.scheduler.cache_probe_batches;
        let (centered, populations) = self.build_centered_population_lanes_9xn(
            &overlap_lanes,
            crate::simd_layout::SIMD_BATCH_LANES,
        );
        let probe_batches = self.stats.scheduler.cache_probe_batches - before;

        let mut matches_scalar = true;
        for lane in 0..crate::simd_layout::SIMD_BATCH_LANES {
            for index in 0..9 {
                let scalar = self.centered_subnode(overlap_lanes[lane][index]);
                matches_scalar &= centered[lane][index] == scalar;
                matches_scalar &= populations[lane][index]
                    == u64::from(self.node_columns.population(scalar) != 0);
            }
        }
        (matches_scalar, distinct_inputs, probe_batches)
    }
}
