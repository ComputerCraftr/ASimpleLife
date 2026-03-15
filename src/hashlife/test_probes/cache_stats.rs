use super::*;
use crate::RequiredExt;
use crate::simd_layout::SIMD_BATCH_LANES;

#[cfg(test)]
impl HashLifeEngine {
    pub(crate) fn repeated_active_gc_without_growth_is_skipped(&mut self) -> bool {
        self.maybe_garbage_collect_with_budget("active", u128::MAX);
        let runs = self.stats.gc.gc_runs;
        let mark_batches = self.stats.gc.gc_mark_batches;
        let root = self.dead_leaf;
        let reason = self.gc_reason(Some(root), Some(root));
        self.maybe_garbage_collect_with_budget(reason, u128::MAX);
        self.stats.gc.gc_runs == runs && self.stats.gc.gc_mark_batches == mark_batches
    }

    pub(crate) fn node_symmetry_metadata_size_for_tests() -> usize {
        std::mem::size_of::<CanonicalNodeRef>()
    }

    pub(crate) fn runtime_stats(&self) -> super::HashLifeRuntimeStats {
        super::HashLifeRuntimeStats {
            engine: super::RuntimeEngineStats {
                nodes: self.node_count(),
                intern: self.intern.len(),
                empty_levels: self.empty_by_level.len(),
                jump_cache: self.result_caches.jump.len(),
                retained_roots: self.retained_roots.len(),
                overlap_cache: self.result_caches.overlap.len(),
                canonical_packed_cache_entries: self.canonical_caches.packed.len()
                    + self.canonical_caches.hot_packed.len(),
                canonical_oriented_cache_entries: self.canonical_caches.oriented.len()
                    + self.canonical_caches.hot_oriented.len(),
                direct_parent_cache_entries: self.canonical_caches.direct_parent.len()
                    + self.canonical_caches.hot_direct_parent.len(),
                structural_fast_path_cache_entries: self.result_caches.structural_fast_path.len(),
                packed_structural_fast_path_cache_entries: self
                    .result_caches
                    .packed_structural_fast_path
                    .len(),
                oriented_result_cache_entries: self.result_caches.oriented.len(),
                packed_transform_intern_entries: self.transform_state.intern.len(),
            },
            result_cache: self.stats.result_cache,
            scheduler: self.stats.scheduler,
            simd: self.stats.simd,
            canonical_cache: self.stats.canonical_cache,
            canonical_fallback: self.stats.canonical_fallback,
            gc: self.stats.gc,
            transform: self.stats.transform,
            materialization: self.stats.materialization,
        }
    }

    pub(crate) fn diagnostic_summary(&self) -> super::HashLifeDiagnosticSummary {
        let stats = self.runtime_stats();
        let jump_presence_total = stats.canonical_fallback.jump_presence_probe_hits
            + stats
                .canonical_fallback
                .jump_presence_probe_lanes
                .saturating_sub(stats.canonical_fallback.jump_presence_probe_hits);
        let overlap_total =
            stats.result_cache.overlap_cache_hits + stats.result_cache.overlap_cache_misses;
        let symmetry_gate_total = stats.canonical_cache.symmetry_gate_allowed
            + stats.canonical_cache.symmetry_gate_blocked;
        let canonical_cache_total = stats.canonical_cache.canonical_node_cache_hits
            + stats.canonical_cache.canonical_node_cache_misses;
        let canonical_packed_total = stats.canonical_cache.canonical_packed_cache_hits
            + stats.canonical_cache.canonical_packed_cache_misses;
        let canonical_oriented_total = stats.canonical_cache.canonical_oriented_cache_hits
            + stats.canonical_cache.canonical_oriented_cache_misses;
        let direct_parent_total = stats.canonical_cache.direct_parent_winner_hits
            + stats.canonical_cache.direct_parent_winner_misses;
        let oriented_result_total = stats.result_cache.oriented_result_cache_hits
            + stats.result_cache.oriented_result_cache_misses;
        let structural_fast_path_total = stats.result_cache.structural_fast_path_hits
            + stats.result_cache.structural_fast_path_misses;
        let total_provisionals = stats.simd.step0_provisional_records
            + stats.simd.phase1_provisional_records
            + stats.simd.phase2_provisional_records;

        super::HashLifeDiagnosticSummary {
            overview: super::DiagnosticOverview {
                total_nodes: stats.engine.nodes,
                retained_roots: stats.engine.retained_roots,
                nodes_match_intern: stats.engine.nodes == stats.engine.intern,
                dependency_stalls: stats.scheduler.dependency_stalls,
            },
            cache_rates: super::DiagnosticCacheRates {
                jump_result_hit_rate: stats.result_cache.jump_result_cache_hits as f64
                    / stats.result_cache.jump_result_cache_lookups.max(1) as f64,
                jump_result_miss_count: stats.result_cache.jump_result_cache_misses,
                oriented_result_hit_rate: stats.result_cache.oriented_result_cache_hits as f64
                    / oriented_result_total.max(1) as f64,
                root_result_hit_rate: stats.result_cache.root_result_cache_hits as f64
                    / stats.result_cache.root_result_cache_lookups.max(1) as f64,
                jump_presence_hit_rate: stats.canonical_fallback.jump_presence_probe_hits as f64
                    / jump_presence_total.max(1) as f64,
                overlap_hit_rate: stats.result_cache.overlap_cache_hits as f64
                    / overlap_total.max(1) as f64,
                overlap_local_reuse_rate: stats.simd.overlap_local_reuse_lanes as f64
                    / stats.simd.overlap_prep_lanes.max(1) as f64,
                symmetry_gate_allow_rate: stats.canonical_cache.symmetry_gate_allowed as f64
                    / symmetry_gate_total.max(1) as f64,
                symmetry_gate_canonical_cache_bypasses: stats
                    .canonical_cache
                    .symmetry_gate_canonical_cache_bypasses,
                structural_fast_path_hit_rate: stats.result_cache.structural_fast_path_hits as f64
                    / structural_fast_path_total.max(1) as f64,
                canonical_node_cache_hit_rate: stats.canonical_cache.canonical_node_cache_hits
                    as f64
                    / canonical_cache_total.max(1) as f64,
                canonical_packed_cache_hit_rate: stats.canonical_cache.canonical_packed_cache_hits
                    as f64
                    / canonical_packed_total.max(1) as f64,
                canonical_oriented_cache_hit_rate: stats
                    .canonical_cache
                    .canonical_oriented_cache_hits
                    as f64
                    / canonical_oriented_total.max(1) as f64,
                direct_parent_winner_hit_rate: stats.canonical_cache.direct_parent_winner_hits
                    as f64
                    / direct_parent_total.max(1) as f64,
                direct_parent_cached_result_hits: stats
                    .canonical_cache
                    .direct_parent_cached_result_hits,
                direct_parent_winner_fallbacks: stats
                    .canonical_cache
                    .direct_parent_winner_fallbacks,
            },
            fallbacks: super::DiagnosticFallbacks {
                blocked_symmetry_scan_fallbacks: 0,
                admitted_symmetry_scan_fallbacks: stats.canonical_fallback.symmetry_scan_fallbacks,
                total_symmetry_scan_fallbacks: stats.canonical_fallback.symmetry_scan_fallbacks,
                symmetry_jump_result_hits: stats.result_cache.symmetric_jump_result_cache_hits,
            },
            work_rates: super::DiagnosticWorkRates {
                simd_lane_coverage: (stats.simd.kernel.portable_vector_lanes
                    + stats.simd.kernel.native_avx2_lanes
                    + stats.simd.kernel.native_neon_lanes)
                    as f64
                    / stats.simd.kernel.candidate_lanes.max(1) as f64,
                scalar_commit_ratio: stats.simd.scalar_commit_lanes as f64
                    / total_provisionals.max(1) as f64,
                probes_per_scheduler_task: stats.scheduler.scheduler_probe_batches as f64
                    / stats.scheduler.scheduler_tasks.max(1) as f64,
                recursive_overlap_batch_rate: stats.simd.recursive_overlap_batch_batches as f64
                    / stats.simd.overlap_prep_batches.max(1) as f64,
            },
            gc: super::DiagnosticGc {
                gc_reclaim_ratio: stats
                    .gc
                    .nodes_before_mark
                    .saturating_sub(stats.gc.nodes_after_mark)
                    as f64
                    / stats.gc.nodes_before_mark.max(1) as f64,
                gc_compact_ratio: stats
                    .gc
                    .nodes_before_compact
                    .saturating_sub(stats.gc.nodes_after_compact)
                    as f64
                    / stats.gc.nodes_before_compact.max(1) as f64,
                gc_reason: stats.gc.gc_reason,
                gc_runs: stats.gc.gc_runs,
                gc_skips: stats.gc.gc_skips,
                gc_transient_pressure_entries_before: stats.gc.gc_transient_pressure_entries_before,
                gc_canonical_cache_entries_before: stats.gc.gc_canonical_cache_entries_before,
                gc_skipped_with_transient_growth: stats.gc.gc_skipped_with_transient_growth,
            },
            cache_sizes: super::DiagnosticCacheSizes {
                canonical_packed_cache_entries: stats.engine.canonical_packed_cache_entries,
                canonical_oriented_cache_entries: stats.engine.canonical_oriented_cache_entries,
                direct_parent_cache_entries: stats.engine.direct_parent_cache_entries,
                structural_fast_path_cache_entries: stats.engine.structural_fast_path_cache_entries,
                packed_structural_fast_path_cache_entries: stats
                    .engine
                    .packed_structural_fast_path_cache_entries,
                oriented_result_cache_entries: stats.engine.oriented_result_cache_entries,
                packed_transform_intern_entries: stats.engine.packed_transform_intern_entries,
            },
            transforms: super::DiagnosticTransforms {
                packed_d4_canonicalization_misses: stats
                    .transform
                    .packed_d4_canonicalization_misses,
                packed_inverse_transform_hits: stats.transform.packed_inverse_transform_hits,
                packed_recursive_transform_hits: stats.transform.packed_recursive_transform_hits,
                packed_recursive_transform_misses: stats
                    .transform
                    .packed_recursive_transform_misses,
                packed_overlap_outputs_produced: stats.transform.packed_overlap_outputs_produced,
                packed_cache_result_materializations: stats
                    .transform
                    .packed_cache_result_materializations,
            },
            materializations: super::DiagnosticMaterializations {
                session_full_grid_materializations: stats
                    .materialization
                    .session_full_grid_materializations,
                embedded_result_full_extractions: stats
                    .materialization
                    .embedded_result_full_extractions,
                clipped_viewport_extractions: stats.materialization.clipped_viewport_extractions,
                checkpoint_cell_materializations: stats
                    .materialization
                    .checkpoint_cell_materializations,
                oracle_confirmation_materializations: stats
                    .materialization
                    .oracle_confirmation_materializations,
                transformed_node_materializations: stats
                    .materialization
                    .transformed_node_materializations,
            },
        }
    }

    pub(crate) fn verify_node_fingerprint_invariants(&self) -> bool {
        (0..self.node_count()).all(|i| {
            let id = NodeId::try_from(i).or_invariant("test node id exceeds u32");
            self.node_columns.fingerprint(id)
                == super::FlatKey::fingerprint(&self.node_columns.packed_key(id))
        })
    }

    pub(crate) fn verify_intern_fingerprint_fast_path_parity(&self) -> bool {
        (0..self.node_count()).all(|i| {
            let id = NodeId::try_from(i).or_invariant("test node id exceeds u32");
            let key = self.node_columns.packed_key(id);
            let fp = self.node_columns.fingerprint(id);
            self.intern.get(&key) == self.intern.get_with_fingerprint(&key, fp)
        })
    }
}

#[test]
fn diagnostic_simd_coverage_counts_executed_vector_kernels_not_scheduler_candidates() {
    let mut engine = HashLifeEngine::default();
    engine.stats.simd.step0_kernel_candidate_batches = 1;
    engine.stats.simd.step0_kernel_candidate_lanes = SIMD_BATCH_LANES;
    engine.stats.simd.kernel.candidate_lanes = SIMD_BATCH_LANES;
    engine.stats.simd.kernel.scalar_fallback_lanes = SIMD_BATCH_LANES;

    assert_eq!(
        engine.diagnostic_summary().work_rates.simd_lane_coverage,
        0.0,
        "scheduler admission must not be reported as SIMD execution"
    );

    engine.stats.simd.kernel.scalar_fallback_lanes = 0;
    engine.stats.simd.kernel.portable_vector_lanes = SIMD_BATCH_LANES;
    assert_eq!(
        engine.diagnostic_summary().work_rates.simd_lane_coverage,
        1.0,
        "executed portable-vector lanes should provide full kernel coverage"
    );
}
