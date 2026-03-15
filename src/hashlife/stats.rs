use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResultCacheStats {
    pub(crate) jump_result_cache_lookups: usize,
    pub(crate) jump_result_cache_hits: usize,
    pub(crate) jump_result_cache_misses: usize,
    pub(crate) symmetric_jump_result_cache_hits: usize,
    pub(crate) oriented_result_cache_lookups: usize,
    pub(crate) oriented_result_cache_hits: usize,
    pub(crate) oriented_result_cache_misses: usize,
    pub(crate) root_result_cache_lookups: usize,
    pub(crate) root_result_cache_hits: usize,
    pub(crate) root_result_cache_misses: usize,
    pub(crate) overlap_cache_hits: usize,
    pub(crate) overlap_cache_misses: usize,
    pub(crate) structural_fast_path_lookups: usize,
    pub(crate) structural_fast_path_hits: usize,
    pub(crate) structural_fast_path_misses: usize,
    pub(crate) canonical_result_insert_bypasses: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SchedulerTraversalStats {
    pub(crate) builder_frames: usize,
    pub(crate) builder_partitions: usize,
    pub(crate) builder_max_stack: usize,
    pub(crate) scheduler_tasks: usize,
    pub(crate) scheduler_ready_max: usize,
    pub(crate) simd_disabled_fast_exits: usize,
    pub(crate) join_shortcut_avoided: usize,
    pub(crate) dependency_stalls: usize,
    pub(crate) step0_ready_max: usize,
    pub(crate) phase1_ready_max: usize,
    pub(crate) phase2_ready_max: usize,
    pub(crate) cache_probe_batches: usize,
    pub(crate) scheduler_probe_batches: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReadyWaveStats {
    pub(crate) samples: usize,
    pub(crate) total_lanes: usize,
    pub(crate) max: usize,
    pub(crate) histogram: [usize; SIMD_BATCH_LANES + 1],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KernelWorkStats {
    pub(crate) candidate_lanes: usize,
    pub(crate) portable_vector_lanes: usize,
    pub(crate) vectorized_structural_lanes: usize,
    pub(crate) scalar_fallback_lanes: usize,
    pub(crate) native_avx2_lanes: usize,
    pub(crate) native_neon_lanes: usize,
    pub(crate) output_presence_kernel_lanes: usize,
    pub(crate) fingerprint_kernel_lanes: usize,
    pub(crate) control_match_kernel_lanes: usize,
    pub(crate) d4_candidate_lanes: usize,
    pub(crate) native_d4_candidate_lanes: usize,
    pub(crate) native_d4_prefix_compare_lanes: usize,
    pub(crate) native_d4_exact_winner_lanes: usize,
    pub(crate) population_kernel_lanes: usize,
    pub(crate) base_transition_kernel_lanes: usize,
    pub(crate) dedup_kernel_lanes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SimdWorkStats {
    pub(crate) step0_kernel_candidate_lanes: usize,
    pub(crate) phase1_kernel_candidate_lanes: usize,
    pub(crate) phase2_kernel_candidate_lanes: usize,
    pub(crate) step0_kernel_candidate_batches: usize,
    pub(crate) phase1_kernel_candidate_batches: usize,
    pub(crate) phase2_kernel_candidate_batches: usize,
    pub(crate) step0_provisional_records: usize,
    pub(crate) phase1_provisional_records: usize,
    pub(crate) phase2_provisional_records: usize,
    pub(crate) scalar_commit_lanes: usize,
    pub(crate) canonical_batch_lanes: usize,
    pub(crate) canonical_batch_batches: usize,
    pub(crate) overlap_prep_lanes: usize,
    pub(crate) overlap_prep_batches: usize,
    pub(crate) recursive_overlap_batch_lanes: usize,
    pub(crate) recursive_overlap_batch_batches: usize,
    pub(crate) overlap_local_reuse_lanes: usize,
    pub(crate) kernel: KernelWorkStats,
    pub(crate) ready_wave: ReadyWaveStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CanonicalCacheStats {
    pub(crate) symmetry_gate_allowed: usize,
    pub(crate) symmetry_gate_blocked: usize,
    pub(crate) symmetry_gate_canonical_cache_bypasses: usize,
    pub(crate) symmetry_aware_result_canonicalization_lookups: usize,
    pub(crate) canonical_node_cache_hits: usize,
    pub(crate) canonical_node_cache_misses: usize,
    pub(crate) canonical_packed_cache_lookups: usize,
    pub(crate) canonical_packed_cache_hits: usize,
    pub(crate) canonical_packed_cache_misses: usize,
    pub(crate) canonical_oriented_cache_lookups: usize,
    pub(crate) canonical_oriented_cache_hits: usize,
    pub(crate) canonical_oriented_cache_misses: usize,
    pub(crate) direct_parent_winner_lookups: usize,
    pub(crate) direct_parent_winner_hits: usize,
    pub(crate) direct_parent_winner_misses: usize,
    pub(crate) direct_parent_winner_fallbacks: usize,
    pub(crate) direct_parent_cached_result_hits: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CanonicalFallbackStats {
    pub(crate) symmetry_scan_fallbacks: usize,
    pub(crate) canonical_phase2_fallbacks: usize,
    pub(crate) canonical_result_batch_fallbacks: usize,
    pub(crate) canonical_result_unique_inputs: usize,
    pub(crate) canonical_result_unique_parent_shapes: usize,
    pub(crate) canonical_result_batch_local_reuses: usize,
    pub(crate) canonical_transform_root_reconstructions: usize,
    pub(crate) oriented_transform_root_reconstructions: usize,
    pub(crate) jump_presence_probe_batches: usize,
    pub(crate) jump_presence_probe_lanes: usize,
    pub(crate) jump_presence_probe_hits: usize,
    pub(crate) jump_batch_unique_queries: usize,
    pub(crate) jump_batch_reused_queries: usize,
    pub(crate) cached_fingerprint_probes: usize,
    pub(crate) recomputed_fingerprint_probes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GcStats {
    pub(crate) nodes_before_mark: usize,
    pub(crate) nodes_after_mark: usize,
    pub(crate) nodes_before_compact: usize,
    pub(crate) nodes_after_compact: usize,
    pub(crate) jump_cache_before_clear: usize,
    pub(crate) gc_runs: usize,
    pub(crate) gc_skips: usize,
    pub(crate) gc_reason: &'static str,
    pub(crate) gc_mark_batches: usize,
    pub(crate) gc_remap_batches: usize,
    pub(crate) gc_transient_pressure_entries_before: usize,
    pub(crate) gc_canonical_cache_entries_before: usize,
    pub(crate) gc_skipped_with_transient_growth: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TransformStats {
    pub(crate) packed_d4_canonicalization_misses: usize,
    pub(crate) d4_semantic_prefix_attempts: usize,
    pub(crate) d4_semantic_prefix_leaf_visits: usize,
    pub(crate) d4_semantic_prefix_cache_bypasses: usize,
    pub(crate) d4_semantic_prefix_cost_bypasses: usize,
    pub(crate) packed_inverse_transform_hits: usize,
    pub(crate) packed_recursive_transform_hits: usize,
    pub(crate) packed_recursive_transform_misses: usize,
    pub(crate) packed_overlap_outputs_produced: usize,
    pub(crate) packed_cache_result_materializations: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MaterializationStats {
    pub(crate) session_full_grid_materializations: usize,
    pub(crate) embedded_result_full_extractions: usize,
    pub(crate) clipped_viewport_extractions: usize,
    pub(crate) checkpoint_cell_materializations: usize,
    pub(crate) oracle_confirmation_materializations: usize,
    #[cfg(test)]
    pub(crate) transformed_node_materializations: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct HashLifeStats {
    pub(super) result_cache: ResultCacheStats,
    pub(super) scheduler: SchedulerTraversalStats,
    pub(super) simd: SimdWorkStats,
    pub(super) canonical_cache: CanonicalCacheStats,
    pub(super) canonical_fallback: CanonicalFallbackStats,
    pub(super) gc: GcStats,
    pub(super) transform: TransformStats,
    pub(super) materialization: MaterializationStats,
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
pub(crate) struct RuntimeEngineStats {
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
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HashLifeRuntimeStats {
    pub engine: RuntimeEngineStats,
    pub result_cache: ResultCacheStats,
    pub scheduler: SchedulerTraversalStats,
    pub simd: SimdWorkStats,
    pub canonical_cache: CanonicalCacheStats,
    pub canonical_fallback: CanonicalFallbackStats,
    pub gc: GcStats,
    pub transform: TransformStats,
    pub materialization: MaterializationStats,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticOverview {
    pub total_nodes: usize,
    pub retained_roots: usize,
    pub nodes_match_intern: bool,
    pub dependency_stalls: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticCacheRates {
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
    pub canonical_node_cache_hit_rate: f64,
    pub canonical_packed_cache_hit_rate: f64,
    pub canonical_oriented_cache_hit_rate: f64,
    pub direct_parent_winner_hit_rate: f64,
    pub direct_parent_cached_result_hits: usize,
    pub direct_parent_winner_fallbacks: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticFallbacks {
    pub blocked_symmetry_scan_fallbacks: usize,
    pub admitted_symmetry_scan_fallbacks: usize,
    pub total_symmetry_scan_fallbacks: usize,
    pub symmetry_jump_result_hits: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticWorkRates {
    pub simd_lane_coverage: f64,
    pub scalar_commit_ratio: f64,
    pub probes_per_scheduler_task: f64,
    pub recursive_overlap_batch_rate: f64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticGc {
    pub gc_reclaim_ratio: f64,
    pub gc_compact_ratio: f64,
    pub gc_reason: &'static str,
    pub gc_runs: usize,
    pub gc_skips: usize,
    pub gc_transient_pressure_entries_before: usize,
    pub gc_canonical_cache_entries_before: usize,
    pub gc_skipped_with_transient_growth: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticCacheSizes {
    pub canonical_packed_cache_entries: usize,
    pub canonical_oriented_cache_entries: usize,
    pub direct_parent_cache_entries: usize,
    pub structural_fast_path_cache_entries: usize,
    pub packed_structural_fast_path_cache_entries: usize,
    pub oriented_result_cache_entries: usize,
    pub packed_transform_intern_entries: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticTransforms {
    pub packed_d4_canonicalization_misses: usize,
    pub packed_inverse_transform_hits: usize,
    pub packed_recursive_transform_hits: usize,
    pub packed_recursive_transform_misses: usize,
    pub packed_overlap_outputs_produced: usize,
    pub packed_cache_result_materializations: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticMaterializations {
    pub session_full_grid_materializations: usize,
    pub embedded_result_full_extractions: usize,
    pub clipped_viewport_extractions: usize,
    pub checkpoint_cell_materializations: usize,
    pub oracle_confirmation_materializations: usize,
    #[cfg(test)]
    pub transformed_node_materializations: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HashLifeDiagnosticSummary {
    pub overview: DiagnosticOverview,
    pub cache_rates: DiagnosticCacheRates,
    pub fallbacks: DiagnosticFallbacks,
    pub work_rates: DiagnosticWorkRates,
    pub gc: DiagnosticGc,
    pub cache_sizes: DiagnosticCacheSizes,
    pub transforms: DiagnosticTransforms,
    pub materializations: DiagnosticMaterializations,
}

pub(super) const DISCOVER_BATCH: usize = SIMD_BATCH_LANES;
pub(super) const JUMP_SYMMETRY_MAX_LEVEL: u32 = 8;
pub(super) const JUMP_SYMMETRY_MAX_POPULATION: u64 = 4_096;
pub const HASHLIFE_FULL_GRID_MAX_POPULATION: u64 = 250_000;
pub const HASHLIFE_FULL_GRID_MAX_CHUNKS: usize = 100_000;
pub const HASHLIFE_CHECKPOINT_MAX_POPULATION: u64 = 250_000;
