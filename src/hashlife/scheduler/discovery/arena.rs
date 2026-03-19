use super::*;

impl HashLifeEngine {
    pub(super) fn drain_discover_batch<const N: usize>(
        discover: &mut Vec<DiscoveredJumpTask>,
        batch: &mut [DiscoveredJumpTask; N],
        batch_keys: &mut [CanonicalJumpKey; N],
    ) -> usize {
        let mut batch_len = 0;
        while batch_len < N {
            let Some(entry) = discover.pop() else {
                break;
            };
            batch[batch_len] = entry;
            batch_keys[batch_len] = entry.key;
            batch_len += 1;
        }
        batch_len
    }

    pub(super) fn build_chunk_child_states<const N: usize>(
        &mut self,
        compacted: [CompactedDiscoveredTask; N],
        unique_count: usize,
        task_index: &FlatTable<CanonicalJumpKey, usize>,
    ) -> [ChunkChildState; N] {
        let mut keys = [CanonicalJumpKey::empty(); N];
        for lane in 0..unique_count {
            keys[lane] = compacted[lane].task.key;
        }
        let present = self.probe_jump_cache_presence_batch(&keys, unique_count);
        let mut states = [ChunkChildState {
            compacted: CompactedDiscoveredTask {
                task: DiscoveredJumpTask {
                    key: CanonicalJumpKey::empty(),
                    source_node: 0,
                    canonical_packed: PackedNodeKey::new(0, [0; 4]),
                },
                duplicate_count: 0,
            },
            present: false,
            blocked: false,
            enqueued: false,
        }; N];
        for index in 0..unique_count {
            let compacted_task = compacted[index];
            let blocked = present[index] || task_index.contains_key(&compacted_task.task.key);
            states[index] = ChunkChildState {
                compacted: compacted_task,
                present: present[index],
                blocked,
                enqueued: false,
            };
        }
        states
    }

    pub(super) fn finalize_chunk_child_states<const N: usize>(
        &mut self,
        states: &mut [ChunkChildState; N],
        unique_count: usize,
        task_index: &FlatTable<CanonicalJumpKey, usize>,
    ) {
        let mut keys = [CanonicalJumpKey::empty(); N];
        for lane in 0..unique_count {
            keys[lane] = states[lane].compacted.task.key;
        }
        let present = self.probe_jump_cache_presence_batch(&keys, unique_count);
        for lane in 0..unique_count {
            let child_key = states[lane].compacted.task.key;
            states[lane].present = present[lane];
            states[lane].blocked = present[lane] || task_index.contains_key(&child_key);
        }
    }

    pub(super) fn complete_recursive_fast_exit(
        &mut self,
        discovered: DiscoveredJumpTask,
        result: NodeId,
        tasks: &mut Vec<Option<TaskRecord>>,
        dependents: &mut FlatTable<CanonicalJumpKey, usize>,
        dependent_edges: &mut Vec<DependentEdge>,
        ready: &mut Vec<usize>,
    ) {
        self.insert_jump_result((discovered.source_node, discovered.key.step_exp), result);
        notify_dependents(
            &discovered.key,
            tasks,
            dependents,
            dependent_edges,
            ready,
        );
        self.stats.scheduler_ready_max = self.stats.scheduler_ready_max.max(ready.len());
    }

    pub(super) fn compact_discovered_jump_tasks<const N: usize>(
        &mut self,
        child_tasks: [DiscoveredJumpTask; N],
    ) -> ([CompactedDiscoveredTask; N], usize) {
        let mut compacted = [CompactedDiscoveredTask {
            task: DiscoveredJumpTask {
                key: CanonicalJumpKey::empty(),
                source_node: 0,
                canonical_packed: PackedNodeKey::new(0, [0; 4]),
            },
            duplicate_count: 0,
        }; N];
        let mut unique_count = 0;
        let mut unique_lookup = FlatTable::<CanonicalJumpKey, usize>::with_capacity(N.max(4));

        for child_key in child_tasks {
            if let Some(index) = unique_lookup.get(&child_key.key) {
                compacted[index].duplicate_count += 1;
            } else {
                compacted[unique_count] = CompactedDiscoveredTask {
                    task: child_key,
                    duplicate_count: 1,
                };
                unique_lookup.insert(child_key.key, unique_count);
                unique_count += 1;
            }
        }
        (compacted, unique_count)
    }

    pub(super) fn probe_and_attach_recursive_parent_overlaps(
        &mut self,
        parent_records: &mut [RecursiveParentBatchRecord],
    ) {
        let parent_count = parent_records.len();
        let mut identities = [CanonicalNodeIdentity {
            packed: PackedNodeKey::new(0, [0; 4]),
            structural: CanonicalStructKey::new(0, [0; 4]),
            symmetry: Symmetry::Identity,
        }; DISCOVER_BATCH];
        let mut fingerprints = [0_u64; DISCOVER_BATCH];
        for lane in 0..parent_count {
            identities[lane] = CanonicalNodeIdentity {
                packed: parent_records[lane].discovered.canonical_packed,
                structural: parent_records[lane].canonical_structural,
                symmetry: Symmetry::Identity,
            };
            fingerprints[lane] = parent_records[lane].canonical_fingerprint;
        }
        let overlaps =
            self.probe_and_build_canonical_overlaps_staged(&identities, &fingerprints, parent_count);
        for lane in 0..parent_count {
            parent_records[lane].overlaps = overlaps[lane];
        }
    }

    pub(super) fn build_recursive_parent_chunk_child_states<const C: usize>(
        &mut self,
        parent_records: &mut [RecursiveParentBatchRecord],
        task_index: &FlatTable<CanonicalJumpKey, usize>,
        child_arena: &mut Vec<RecursiveParentChildRef>,
    ) -> ([ChunkChildState; C], usize) {
        let parent_count = parent_records.len();
        let mut chunk_query_lookup = FlatTable::<JumpQuery, usize>::with_capacity(C.max(8));
        let mut chunk_child_states = [ChunkChildState {
            compacted: CompactedDiscoveredTask {
                task: DiscoveredJumpTask {
                    key: CanonicalJumpKey::empty(),
                    source_node: 0,
                    canonical_packed: PackedNodeKey::new(0, [0; 4]),
                },
                duplicate_count: 0,
            },
            present: false,
            blocked: false,
            enqueued: false,
        }; C];
        let mut chunk_queries = [JumpQuery { node: 0, step_exp: 0 }; C];
        let mut chunk_unique_count = 0usize;
        child_arena.clear();
        const REVERSED_OVERLAP_INDEX: [usize; 9] = [8, 7, 6, 5, 4, 3, 2, 1, 0];

        for lane in 0..parent_count {
            let parent_child_start = child_arena.len();
            let next_exp = parent_records[lane].next_exp;
            let mut parent_query_indices = [u16::MAX; 9];
            let mut parent_unique_count = 0usize;

            for overlap_index in REVERSED_OVERLAP_INDEX {
                let child = parent_records[lane].overlaps[overlap_index];
                let query = JumpQuery {
                    node: child,
                    step_exp: next_exp,
                };

                let query_index = if let Some(existing) = chunk_query_lookup.get(&query) {
                    existing
                } else {
                    chunk_queries[chunk_unique_count] = query;
                    chunk_query_lookup.insert(query, chunk_unique_count);
                    let inserted = chunk_unique_count;
                    chunk_unique_count += 1;
                    inserted
                } as u16;

                let mut reused_parent_slot = None;
                for local_index in 0..parent_unique_count {
                    if parent_query_indices[local_index] == query_index {
                        reused_parent_slot = Some(local_index);
                        break;
                    }
                }

                if let Some(local_index) = reused_parent_slot {
                    let arena_index = parent_child_start + local_index;
                    child_arena[arena_index].duplicate_count =
                        child_arena[arena_index].duplicate_count.saturating_add(1);
                } else {
                    parent_query_indices[parent_unique_count] = query_index;
                    parent_unique_count += 1;
                    child_arena.push(RecursiveParentChildRef {
                        query_index,
                        duplicate_count: 1,
                    });
                }
            }

            parent_records[lane].child_arena_start = parent_child_start as u16;
            parent_records[lane].child_arena_len = (child_arena.len() - parent_child_start) as u8;
        }

        for chunk_index in 0..chunk_unique_count {
            let jump_probe = self.canonical_jump_probe((
                chunk_queries[chunk_index].node,
                chunk_queries[chunk_index].step_exp,
            ));
            chunk_child_states[chunk_index] = ChunkChildState {
                compacted: CompactedDiscoveredTask {
                    task: DiscoveredJumpTask {
                        key: jump_probe.key,
                        source_node: chunk_queries[chunk_index].node,
                        canonical_packed: jump_probe.node.packed,
                    },
                    duplicate_count: 1,
                },
                present: false,
                blocked: false,
                enqueued: false,
            };
        }

        self.finalize_chunk_child_states(&mut chunk_child_states, chunk_unique_count, task_index);
        (chunk_child_states, chunk_unique_count)
    }

    pub(in crate::hashlife) fn discovered_jump_tasks_from_nodes<const N: usize>(
        &mut self,
        child_nodes: [NodeId; N],
        step_exp: u32,
    ) -> [DiscoveredJumpTask; N] {
        let mut child_keys = [DiscoveredJumpTask {
            key: CanonicalJumpKey::empty(),
            source_node: 0,
            canonical_packed: PackedNodeKey::new(0, [0; 4]),
        }; N];
        let mut reuse = FlatTable::<NodeId, DiscoveredJumpTask>::with_capacity(N.max(4));
        for lane in 0..N {
            let child = child_nodes[lane];
            if let Some(reused) = reuse.get(&child) {
                child_keys[lane] = reused;
                continue;
            }
            let jump_probe = self.canonical_jump_probe((child, step_exp));
            let discovered = DiscoveredJumpTask {
                key: jump_probe.key,
                source_node: child,
                canonical_packed: jump_probe.node.packed,
            };
            child_keys[lane] = discovered;
            reuse.insert(child, discovered);
        }
        child_keys
    }

    pub(super) fn probe_jump_cache_presence_batch<const N: usize>(
        &mut self,
        keys: &[CanonicalJumpKey; N],
        active_lanes: usize,
    ) -> [bool; N] {
        let mut present = [false; N];
        if active_lanes == 0 {
            return present;
        }

        let mut fingerprints = [0_u64; N];
        self.stats.jump_presence_probe_batches += 1;
        self.stats.jump_presence_probe_lanes += active_lanes;
        self.stats.scheduler_probe_batches += 1;
        for lane in 0..active_lanes {
            fingerprints[lane] = hash_packed_jump_fingerprint(
                keys[lane].structural.fingerprint(),
                keys[lane].step_exp,
            );
        }
        let cached =
            self.jump_cache
                .get_many_with_fingerprints(keys, &fingerprints, active_lanes);
        let mut hit_count = 0usize;
        for lane in 0..active_lanes {
            present[lane] = cached[lane].is_some();
            hit_count += usize::from(present[lane]);
        }
        self.stats.jump_presence_probe_hits += hit_count;
        present
    }
}
