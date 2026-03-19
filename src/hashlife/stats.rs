use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct HashLifeStats {
    pub(super) jump_result_cache_lookups: usize,
    pub(super) jump_result_cache_hits: usize,
    pub(super) jump_result_cache_misses: usize,
    pub(super) symmetric_jump_result_cache_hits: usize,
    pub(super) oriented_result_cache_lookups: usize,
    pub(super) oriented_result_cache_hits: usize,
    pub(super) oriented_result_cache_misses: usize,
    pub(super) root_result_cache_lookups: usize,
    pub(super) root_result_cache_hits: usize,
    pub(super) root_result_cache_misses: usize,
    pub(super) overlap_cache_hits: usize,
    pub(super) overlap_cache_misses: usize,
    pub(super) nodes_before_mark: usize,
    pub(super) nodes_after_mark: usize,
    pub(super) nodes_before_compact: usize,
    pub(super) nodes_after_compact: usize,
    pub(super) jump_cache_before_clear: usize,
    pub(super) gc_runs: usize,
    pub(super) gc_skips: usize,
    pub(super) gc_reason: &'static str,
    pub(super) builder_frames: usize,
    pub(super) builder_partitions: usize,
    pub(super) builder_max_stack: usize,
    pub(super) scheduler_tasks: usize,
    pub(super) scheduler_ready_max: usize,
    pub(super) simd_disabled_fast_exits: usize,
    pub(super) step0_simd_lanes: usize,
    pub(super) phase1_simd_lanes: usize,
    pub(super) phase2_simd_lanes: usize,
    pub(super) step0_simd_batches: usize,
    pub(super) phase1_simd_batches: usize,
    pub(super) phase2_simd_batches: usize,
    pub(super) step0_provisional_records: usize,
    pub(super) phase1_provisional_records: usize,
    pub(super) phase2_provisional_records: usize,
    pub(super) scalar_commit_lanes: usize,
    pub(super) join_shortcut_avoided: usize,
    pub(super) dependency_stalls: usize,
    pub(super) step0_ready_max: usize,
    pub(super) phase1_ready_max: usize,
    pub(super) phase2_ready_max: usize,
    pub(super) canonical_batch_lanes: usize,
    pub(super) canonical_batch_batches: usize,
    pub(super) overlap_prep_lanes: usize,
    pub(super) overlap_prep_batches: usize,
    pub(super) recursive_overlap_batch_lanes: usize,
    pub(super) recursive_overlap_batch_batches: usize,
    pub(super) overlap_local_reuse_lanes: usize,
    pub(super) cache_probe_batches: usize,
    pub(super) scheduler_probe_batches: usize,
    pub(super) symmetry_gate_allowed: usize,
    pub(super) symmetry_gate_blocked: usize,
    pub(super) symmetry_gate_canonical_cache_bypasses: usize,
    pub(super) structural_fast_path_lookups: usize,
    pub(super) structural_fast_path_hits: usize,
    pub(super) structural_fast_path_misses: usize,
    pub(super) canonical_result_insert_bypasses: usize,
    pub(super) symmetry_aware_result_canonicalization_lookups: usize,
    pub(super) canonical_node_cache_hits: usize,
    pub(super) canonical_node_cache_misses: usize,
    pub(super) canonical_packed_cache_lookups: usize,
    pub(super) canonical_packed_cache_hits: usize,
    pub(super) canonical_packed_cache_misses: usize,
    pub(super) canonical_oriented_cache_lookups: usize,
    pub(super) canonical_oriented_cache_hits: usize,
    pub(super) canonical_oriented_cache_misses: usize,
    pub(super) direct_parent_winner_lookups: usize,
    pub(super) direct_parent_winner_hits: usize,
    pub(super) direct_parent_winner_misses: usize,
    pub(super) direct_parent_winner_fallbacks: usize,
    pub(super) direct_parent_cached_result_hits: usize,
    pub(super) symmetry_scan_fallbacks: usize,
    pub(super) canonical_phase2_fallbacks: usize,
    pub(super) canonical_result_batch_fallbacks: usize,
    pub(super) canonical_blocked_structural_fallbacks: usize,
    pub(super) canonical_result_unique_inputs: usize,
    pub(super) canonical_result_unique_parent_shapes: usize,
    pub(super) canonical_result_batch_local_reuses: usize,
    pub(super) canonical_transform_root_reconstructions: usize,
    pub(super) oriented_transform_root_reconstructions: usize,
    pub(super) jump_presence_probe_batches: usize,
    pub(super) jump_presence_probe_lanes: usize,
    pub(super) jump_presence_probe_hits: usize,
    pub(super) jump_batch_unique_queries: usize,
    pub(super) jump_batch_reused_queries: usize,
    pub(super) cached_fingerprint_probes: usize,
    pub(super) recomputed_fingerprint_probes: usize,
    pub(super) gc_mark_batches: usize,
    pub(super) gc_remap_batches: usize,
    pub(super) gc_transient_entries_before: usize,
    pub(super) gc_canonical_cache_entries_before: usize,
    pub(super) gc_skipped_with_transient_growth: usize,
    pub(super) packed_d4_canonicalization_misses: usize,
    pub(super) packed_inverse_transform_hits: usize,
    pub(super) packed_recursive_transform_hits: usize,
    pub(super) packed_recursive_transform_misses: usize,
    pub(super) packed_overlap_outputs_produced: usize,
    pub(super) packed_cache_result_materializations: usize,
    pub(crate) session_full_grid_materializations: usize,
    pub(super) session_bounded_grid_extractions: usize,
    pub(super) embedded_result_bounded_extractions: usize,
    pub(super) clipped_viewport_extractions: usize,
    pub(super) checkpoint_cell_materializations: usize,
    pub(super) oracle_confirmation_materializations: usize,
    pub(super) dense_fallback_invocations: usize,
    #[cfg(test)]
    pub(super) transformed_node_materializations: usize,
}

#[derive(Clone, Copy)]
pub(super) enum PendingTask {
    PhaseOne {
        next_exp: u32,
        a: NodeId,
        b: NodeId,
        c: NodeId,
        d: NodeId,
        e: NodeId,
        f: NodeId,
        g: NodeId,
        h: NodeId,
        i: NodeId,
    },
    PhaseTwo {
        next_exp: u32,
        nw: NodeId,
        ne: NodeId,
        sw: NodeId,
        se: NodeId,
    },
}

#[derive(Clone, Copy)]
pub(super) struct TaskRecord {
    pub(super) remaining: u8,
    pub(super) task: PendingTask,
}

#[derive(Clone, Copy)]
pub(super) struct Step0TaskRecord {
    pub(super) remaining: u8,
    pub(super) children: [NodeId; 4],
}

#[derive(Clone, Copy)]
pub(super) struct RecursiveParentBatchRecord {
    pub(super) discovered: DiscoveredJumpTask,
    pub(super) next_exp: u32,
    pub(super) canonical_structural: CanonicalStructKey,
    pub(super) canonical_fingerprint: u64,
    pub(super) overlaps: [NodeId; 9],
    pub(super) child_arena_start: u16,
    pub(super) child_arena_len: u8,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RecursiveParentChildRef {
    pub(super) query_index: u16,
    pub(super) duplicate_count: u8,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DiscoveredJumpTask {
    pub(super) key: CanonicalJumpKey,
    pub(super) source_node: NodeId,
    pub(super) canonical_packed: PackedNodeKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step0LaneDispatch {
    SimdChild,
}

#[derive(Clone, Copy)]
pub(super) struct SimdProvisionalRecord {
    pub(super) cache_key: CanonicalJumpKey,
    pub(super) level: u32,
    pub(super) inputs: SimdProvisionalInputs,
    pub(super) payload: SimdProvisionalPayload,
}

#[derive(Clone, Copy)]
pub(super) struct Phase1ReadyLane {
    pub(super) task_id: usize,
    pub(super) key: CanonicalJumpKey,
    pub(super) next_exp: u32,
    pub(super) inputs: [NodeId; 9],
}

#[derive(Clone, Copy)]
pub(super) struct Phase2ReadyLane {
    pub(super) key: CanonicalJumpKey,
    pub(super) next_exp: u32,
    pub(super) inputs: [NodeId; 4],
}

#[derive(Clone, Copy)]
pub(super) struct Phase1CommitLane {
    pub(super) provisional: SimdProvisionalRecord,
    pub(super) task_id: usize,
    pub(super) next_exp: u32,
    pub(super) next_children: [NodeId; 4],
}

#[derive(Clone, Copy)]
pub(super) struct Phase2CommitLane {
    pub(super) key: CanonicalJumpKey,
    pub(super) fallback: NodeId,
    pub(super) result: NodeId,
    pub(super) unique_input_index: usize,
    pub(super) packed_input: PackedSymmetryKey,
    pub(super) canonical_entry: PackedSymmetryKey,
}

#[derive(Clone, Copy)]
pub(super) enum SimdProvisionalInputs {
    Nine {
        nodes: [NodeId; 9],
        populations: [u64; 9],
    },
    Four {
        nodes: [NodeId; 4],
        populations: [u64; 4],
    },
}

#[derive(Clone, Copy)]
pub(super) enum SimdProvisionalPayload {
    Step0 {
        dispatch: Step0LaneDispatch,
    },
    PhaseOne {
        next_exp: u32,
        source_task_id: usize,
    },
    PhaseTwo,
}

#[derive(Clone, Copy)]
pub(super) struct SimdPackedBatch {
    pub(super) active_lanes: usize,
    pub(super) active_mask: u8,
    pub(super) populations: [u64x8; 9],
}

#[derive(Clone, Copy)]
pub(super) struct SimdLaneResult {
    pub(super) output_nonzero_mask: u8,
}

pub(super) struct SimdBatchResult {
    pub(super) lanes: [SimdLaneResult; SIMD_BATCH_LANES],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct JoinIntent {
    pub(super) level: u32,
    pub(super) children: [NodeId; 4],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UniqueJumpQueryRecord {
    pub(super) query: JumpQuery,
    pub(super) cache_key: CanonicalJumpKey,
    pub(super) inverse_symmetry: Symmetry,
    pub(super) fingerprint: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UniqueOrientedResultRecord {
    pub(super) packed: PackedNodeKey,
    pub(super) symmetry: Symmetry,
    pub(super) node: NodeId,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EmbeddedCell {
    pub(super) key: u128,
}

pub(super) const NO_DEPENDENT: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
pub(super) struct DependentEdge {
    pub(super) task_id: usize,
    pub(super) next: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HashLifeRuntimeStats {
    pub nodes: usize,
    pub intern: usize,
    pub empty_levels: usize,
    pub jump_cache: usize,
    pub retained_roots: usize,
    pub overlap_cache: usize,
    pub canonical_packed_cache_entries: usize,
    pub canonical_oriented_cache_entries: usize,
    pub direct_parent_cache_entries: usize,
    pub structural_fast_path_cache_entries: usize,
    pub packed_structural_fast_path_cache_entries: usize,
    pub oriented_result_cache_entries: usize,
    pub packed_transform_intern_entries: usize,
    pub jump_result_cache_lookups: usize,
    pub jump_result_cache_hits: usize,
    pub jump_result_cache_misses: usize,
    pub symmetric_jump_result_cache_hits: usize,
    pub oriented_result_cache_lookups: usize,
    pub oriented_result_cache_hits: usize,
    pub oriented_result_cache_misses: usize,
    pub root_result_cache_lookups: usize,
    pub root_result_cache_hits: usize,
    pub root_result_cache_misses: usize,
    pub overlap_cache_hits: usize,
    pub overlap_cache_misses: usize,
    pub gc_runs: usize,
    pub gc_skips: usize,
    pub nodes_before_mark: usize,
    pub nodes_after_mark: usize,
    pub nodes_before_compact: usize,
    pub nodes_after_compact: usize,
    pub jump_cache_before_clear: usize,
    pub gc_reason: &'static str,
    pub builder_frames: usize,
    pub builder_partitions: usize,
    pub builder_max_stack: usize,
    pub scheduler_tasks: usize,
    pub scheduler_ready_max: usize,
    pub simd_disabled_fast_exits: usize,
    pub step0_simd_lanes: usize,
    pub phase1_simd_lanes: usize,
    pub phase2_simd_lanes: usize,
    pub step0_simd_batches: usize,
    pub phase1_simd_batches: usize,
    pub phase2_simd_batches: usize,
    pub step0_provisional_records: usize,
    pub phase1_provisional_records: usize,
    pub phase2_provisional_records: usize,
    pub scalar_commit_lanes: usize,
    pub join_shortcut_avoided: usize,
    pub dependency_stalls: usize,
    pub step0_ready_max: usize,
    pub phase1_ready_max: usize,
    pub phase2_ready_max: usize,
    pub canonical_batch_lanes: usize,
    pub canonical_batch_batches: usize,
    pub overlap_prep_lanes: usize,
    pub overlap_prep_batches: usize,
    pub recursive_overlap_batch_lanes: usize,
    pub recursive_overlap_batch_batches: usize,
    pub overlap_local_reuse_lanes: usize,
    pub cache_probe_batches: usize,
    pub scheduler_probe_batches: usize,
    pub symmetry_gate_allowed: usize,
    pub symmetry_gate_blocked: usize,
    pub symmetry_gate_canonical_cache_bypasses: usize,
    pub structural_fast_path_lookups: usize,
    pub structural_fast_path_hits: usize,
    pub structural_fast_path_misses: usize,
    pub canonical_result_insert_bypasses: usize,
    pub symmetry_aware_result_canonicalization_lookups: usize,
    pub canonical_node_cache_hits: usize,
    pub canonical_node_cache_misses: usize,
    pub canonical_packed_cache_lookups: usize,
    pub canonical_packed_cache_hits: usize,
    pub canonical_packed_cache_misses: usize,
    pub canonical_oriented_cache_lookups: usize,
    pub canonical_oriented_cache_hits: usize,
    pub canonical_oriented_cache_misses: usize,
    pub direct_parent_winner_lookups: usize,
    pub direct_parent_winner_hits: usize,
    pub direct_parent_winner_misses: usize,
    pub direct_parent_winner_fallbacks: usize,
    pub direct_parent_cached_result_hits: usize,
    pub symmetry_scan_fallbacks: usize,
    pub canonical_phase2_fallbacks: usize,
    pub canonical_result_batch_fallbacks: usize,
    pub canonical_blocked_structural_fallbacks: usize,
    pub jump_presence_probe_batches: usize,
    pub jump_presence_probe_lanes: usize,
    pub jump_presence_probe_hits: usize,
    pub jump_batch_unique_queries: usize,
    pub jump_batch_reused_queries: usize,
    pub cached_fingerprint_probes: usize,
    pub recomputed_fingerprint_probes: usize,
    pub gc_mark_batches: usize,
    pub gc_remap_batches: usize,
    pub gc_transient_entries_before: usize,
    pub gc_canonical_cache_entries_before: usize,
    pub gc_skipped_with_transient_growth: usize,
    pub packed_d4_canonicalization_misses: usize,
    pub packed_inverse_transform_hits: usize,
    pub packed_recursive_transform_hits: usize,
    pub packed_recursive_transform_misses: usize,
    pub packed_overlap_outputs_produced: usize,
    pub packed_cache_result_materializations: usize,
    pub session_full_grid_materializations: usize,
    pub session_bounded_grid_extractions: usize,
    pub embedded_result_bounded_extractions: usize,
    pub clipped_viewport_extractions: usize,
    pub checkpoint_cell_materializations: usize,
    pub oracle_confirmation_materializations: usize,
    pub dense_fallback_invocations: usize,
    #[cfg(test)]
    pub transformed_node_materializations: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HashLifeDiagnosticSummary {
    pub total_nodes: usize,
    pub retained_roots: usize,
    pub nodes_match_intern: bool,
    pub dependency_stalls: usize,
    pub jump_result_hit_rate: f64,
    pub jump_result_miss_count: usize,
    pub oriented_result_hit_rate: f64,
    pub root_result_hit_rate: f64,
    pub jump_presence_hit_rate: f64,
    pub overlap_hit_rate: f64,
    pub overlap_local_reuse_rate: f64,
    pub symmetry_gate_allow_rate: f64,
    pub symmetry_gate_canonical_cache_bypasses: usize,
    pub structural_fast_path_hit_rate: f64,
    pub canonical_cache_hit_rate: f64,
    pub canonical_packed_cache_hit_rate: f64,
    pub canonical_oriented_cache_hit_rate: f64,
    pub direct_parent_winner_hit_rate: f64,
    pub direct_parent_cached_result_hits: usize,
    pub direct_parent_winner_fallbacks: usize,
    pub symmetry_scan_fallbacks: usize,
    pub symmetry_jump_result_hits: usize,
    pub simd_lane_coverage: f64,
    pub scalar_commit_ratio: f64,
    pub probes_per_scheduler_task: f64,
    pub recursive_overlap_batch_rate: f64,
    pub gc_reclaim_ratio: f64,
    pub gc_compact_ratio: f64,
    pub gc_reason: &'static str,
    pub gc_runs: usize,
    pub gc_skips: usize,
    pub gc_transient_entries_before: usize,
    pub gc_canonical_cache_entries_before: usize,
    pub gc_skipped_with_transient_growth: usize,
    pub canonical_packed_cache_entries: usize,
    pub canonical_oriented_cache_entries: usize,
    pub direct_parent_cache_entries: usize,
    pub structural_fast_path_cache_entries: usize,
    pub packed_structural_fast_path_cache_entries: usize,
    pub oriented_result_cache_entries: usize,
    pub packed_transform_intern_entries: usize,
    pub packed_d4_canonicalization_misses: usize,
    pub packed_inverse_transform_hits: usize,
    pub packed_recursive_transform_hits: usize,
    pub packed_recursive_transform_misses: usize,
    pub packed_overlap_outputs_produced: usize,
    pub packed_cache_result_materializations: usize,
    pub session_full_grid_materializations: usize,
    pub session_bounded_grid_extractions: usize,
    pub embedded_result_bounded_extractions: usize,
    pub clipped_viewport_extractions: usize,
    pub checkpoint_cell_materializations: usize,
    pub oracle_confirmation_materializations: usize,
    pub dense_fallback_invocations: usize,
    #[cfg(test)]
    pub transformed_node_materializations: usize,
}

pub(super) const DISCOVER_BATCH: usize = 4;
pub(super) const JUMP_SYMMETRY_MAX_LEVEL: u32 = 8;
pub(super) const JUMP_SYMMETRY_MAX_POPULATION: u64 = 4_096;
pub const HASHLIFE_FULL_GRID_MAX_POPULATION: u64 = 250_000;
pub const HASHLIFE_FULL_GRID_MAX_CHUNKS: usize = 100_000;
pub const HASHLIFE_CHECKPOINT_MAX_POPULATION: u64 = 250_000;
pub const HASHLIFE_CHECKPOINT_MAX_BOUNDS_SPAN: Coord = 65_536;
