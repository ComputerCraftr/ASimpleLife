use crate::bitgrid::{BitGrid, Coord};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeDiagnosticSummary;
use crate::life::{GameOfLife, step_grid};
use crate::normalize::normalize;

pub(super) const SMALL_SOUP_DIM: Coord = 24;
pub(super) const MEDIUM_SOUP_DIM: Coord = 48;
pub(super) const LARGE_SOUP_DIM: Coord = 64;
pub(super) const XL_SOUP_DIM: Coord = 96;
pub(super) const SMALL_SOUP_FILL: u32 = 20;
pub(super) const MEDIUM_SOUP_FILL: u32 = 22;
pub(super) const DEEP_SOUP_FILL: u32 = 24;

pub(super) const GEN_SINGLE_STEP: u64 = 1;
pub(super) const GEN_STEP0_STRESS: u64 = 1;
pub(super) const GEN_POWER_OF_TWO_SMALL: u64 = 64;
pub(super) const GEN_POWER_OF_TWO: u64 = 256;
pub(super) const GEN_SEGMENTED_REMAINDER: u64 = 253;
pub(super) const GEN_MEDIUM_PRIME: u64 = 269;
pub(super) const GEN_LARGE_PRIME: u64 = 509;
pub(super) const GEN_MIRROR_REUSE: u64 = 16;
pub(super) const GEN_BLINKER_EVEN: u64 = 1_000;
pub(super) const GEN_BLINKER_ODD: u64 = 1_001;
pub(super) const GEN_SNAPSHOT: u64 = 4_096;
pub(super) const GEN_RESUME_DELTA: u64 = 1_024;
pub(super) const GEN_DEEP_DIAGNOSTIC: u64 = 3_000_000;
pub(super) const GEN_STRUCTURED_PROMOTION_DIAGNOSTIC: u64 = 1_000_000;
pub(super) const GEN_STRUCTURED_TUNING_DIAGNOSTIC: u64 = 8_192;
pub(super) const GEN_STRUCTURED_PROMOTION_REGRESSION: u64 = 262_144;
pub(super) const GEN_GUN_PERIOD_MULTIPLE: u64 = 10_000_020;
pub(super) const GEN_SPARSE_SESSION_REGRESSION: u64 = 1_000_000;
pub(super) const GEN_SPARSE_SESSION_COVERAGE: u64 = 262_144;
pub(super) const DEEP_RUN_REPETITIONS: usize = 3;

pub(super) const SYMMETRY_GATE_DEFAULT_LEVEL: u32 = 8;
pub(super) const SYMMETRY_GATE_DEFAULT_POPULATION: u64 = 4_096;
pub(super) const SYMMETRY_GATE_WIDE_LEVEL: u32 = 12;
pub(super) const SYMMETRY_GATE_WIDE_POPULATION: u64 = 65_536;
pub(super) const SYMMETRY_GATE_UNGATED_LEVEL: u32 = u32::MAX;
pub(super) const SYMMETRY_GATE_UNGATED_POPULATION: u64 = u64::MAX;

pub(super) const STEP0_GLIDER_OFFSET_X: Coord = 160;
pub(super) const STEP0_GLIDER_OFFSET_Y: Coord = 24;
pub(super) const STEP0_BLINKER_OFFSET_X: Coord = 24;
pub(super) const STEP0_BLINKER_OFFSET_Y: Coord = 160;
pub(super) const STRUCTURED_MIRROR_OFFSET_X: Coord = 256;
pub(super) const STRUCTURED_MIRROR_OFFSET_Y: Coord = 0;
pub(super) const STRUCTURED_GLIDER_OFFSET_X: Coord = 128;
pub(super) const STRUCTURED_GLIDER_OFFSET_Y: Coord = 128;

pub(super) const SEED_CACHE_BASELINE: u64 = 0xA5A5_5A5A_DEAD_BEEF;
pub(super) const SEED_CACHE_VARIANT: u64 = 0x0DDC_0FFE_EE11_2233;
pub(super) const SEED_FINGERPRINT_FAST_PATH: u64 = 0xF1A5_C0DE_1234_5678;
pub(super) const SEED_GC_REBUILD: u64 = 0xBAD5_EED5_2468_ACE0;
pub(super) const SEED_PACKED_JUMP_ROUNDTRIP: u64 = 0xA55A_5AA5_FACE_CAFE;
pub(super) const SEED_CANONICAL_RESULT_INSERT: u64 = 0x1357_9BDF_2468_ACE0;
pub(super) const SEED_JUMP_RESULT_INSERT: u64 = 0x1122_3344_5566_7788;
pub(super) const SEED_ORIENTED_RESULT_CACHE: u64 = 0x0DDC_0FFE_EE11_BAAD;
pub(super) const SEED_CANONICAL_PACKED_CACHE: u64 = 0xA0B1_C2D3_E4F5_1020;
pub(super) const SEED_IDENTITY_CANONICALIZATION: u64 = 0x9A12_BC34_DE56_F078;
pub(super) const SEED_GATE_BLOCKED_PROBE: u64 = 0xCAF0_F00D_BAAD_F00D;
pub(super) const SEED_DIRECT_PARENT_WINNER: u64 = 0xD1EC_7CA1_100F_EEDE;
pub(super) const SEED_PACKED_TRANSFORM_PARITY: u64 = 0xC001_CAFE_FEED_FACE;
pub(super) const SEED_PACKED_ROOT_PARITY: u64 = 0x0F0F_F0F0_AAAA_5555;
pub(super) const SEED_CANONICAL_SYMMETRY_PARITY: u64 = 0xABCD_EF01_2345_6789;
pub(super) const SEED_SINGLE_STEP_SOUP: u64 = 0xD1B5_4A32_D192_ED03;
pub(super) const SEED_POWER_OF_TWO_SOUP: u64 = 0xDEAD_BEEF_CAFE_BABE;
pub(super) const SEED_PRIME_JUMP_SOUP: u64 = 0x1234_5678_9ABC_DEF0;
pub(super) const SEED_OVERLAP_BATCH_PARITY: u64 = 0x0BAD_F00D_CAFE_1234;
pub(super) const SEED_CANONICAL_OVERLAP_PARITY: u64 = 0x0ACE_F00D_BAAD_C0DE;
pub(super) const SEED_OVERLAP_BATCH_DEDUPE: u64 = 0x1234_5678_0BAD_F00D;
pub(super) const SEED_JUMP_BATCH_DEDUPE: u64 = 0xFACE_FEED_0BAD_F00D;
pub(super) const SEED_CHILD_KEY_PARITY: u64 = 0xACED_BEEF_1234_5678;
pub(super) const SEED_STEP0_STRESS: u64 = 0xBAD5_EED5_1234_5678;
pub(super) const SEED_STEP0_SMALL_BATCH: u64 = 0xFACE_FEED_1234_5678;
pub(super) const SEED_RECURSIVE_SIMD: u64 = 0x1234_5678_AAAA_5555;

pub(super) fn assert_hashlife_matches_stepper(grid: BitGrid, generations: u64) {
    let mut game = GameOfLife::new(grid.clone());
    for _ in 0..generations {
        game.step_with_changes();
    }

    let advanced = crate::hashlife::HashLifeEngine::default().advance(&grid, generations);
    assert_eq!(normalize(&advanced).0, normalize(game.grid()).0);
}

pub(super) fn grid_from_mask(width: Coord, height: Coord, mask: u32) -> BitGrid {
    let mut cells = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let bit = u32::try_from(y * width + x).expect("grid mask index exceeded u32");
            if (mask >> bit) & 1 == 1 {
                cells.push((x, y));
            }
        }
    }
    BitGrid::from_cells(&cells)
}

pub(super) fn mirror_grid_x(grid: &BitGrid) -> BitGrid {
    let (min_x, _, max_x, _) = grid.bounds().expect("grid should be non-empty");
    let mirrored = grid
        .live_cells()
        .into_iter()
        .map(|(x, y)| (max_x - (x - min_x), y))
        .collect::<Vec<_>>();
    BitGrid::from_cells(&mirrored)
}

pub(super) fn build_hashlife_structured_symmetry_grid() -> BitGrid {
    let mut cells = pattern_by_name("gosper_glider_gun").unwrap().live_cells();
    cells.extend(
        mirror_grid_x(&pattern_by_name("gosper_glider_gun").unwrap())
            .translated(STRUCTURED_MIRROR_OFFSET_X, STRUCTURED_MIRROR_OFFSET_Y)
            .live_cells(),
    );
    cells.extend(
        pattern_by_name("glider")
            .unwrap()
            .translated(STRUCTURED_GLIDER_OFFSET_X, STRUCTURED_GLIDER_OFFSET_Y)
            .live_cells(),
    );
    BitGrid::from_cells(&cells)
}

pub(super) fn build_hashlife_step0_stress_grid() -> BitGrid {
    let mut cells = random_soup(
        XL_SOUP_DIM,
        XL_SOUP_DIM,
        MEDIUM_SOUP_FILL,
        SEED_STEP0_STRESS,
    )
    .live_cells();
    cells.extend(
        pattern_by_name("glider")
            .unwrap()
            .translated(STEP0_GLIDER_OFFSET_X, STEP0_GLIDER_OFFSET_Y)
            .live_cells(),
    );
    cells.extend(
        pattern_by_name("blinker")
            .unwrap()
            .translated(STEP0_BLINKER_OFFSET_X, STEP0_BLINKER_OFFSET_Y)
            .live_cells(),
    );
    BitGrid::from_cells(&cells)
}

pub(super) fn print_hashlife_summary(
    label: &str,
    elapsed: std::time::Duration,
    summary: HashLifeDiagnosticSummary,
) {
    eprintln!(
        "{} total_us={} nodes={} retained_roots={} nodes_match_intern={} dependency_stalls={} jump_result_hit_rate={:.3} jump_result_misses={} oriented_result_hit_rate={:.3} root_result_hit_rate={:.3} jump_presence_hit_rate={:.3} overlap_hit_rate={:.3} overlap_local_reuse_rate={:.3} symmetry_gate_allow_rate={:.3} symmetry_gate_canonical_cache_bypasses={} structural_fast_path_hit_rate={:.3} canonical_node_cache_hit_rate={:.3} canonical_packed_cache_hit_rate={:.3} canonical_oriented_cache_hit_rate={:.3} direct_parent_winner_hit_rate={:.3} direct_parent_cached_result_hits={} direct_parent_winner_fallbacks={} blocked_symmetry_scan_fallbacks={} admitted_symmetry_scan_fallbacks={} total_symmetry_scan_fallbacks={} symmetry_jump_result_hits={} simd_lane_coverage={:.3} scalar_commit_ratio={:.3} probes_per_scheduler_task={:.3} recursive_overlap_batch_rate={:.3} gc_reclaim_ratio={:.3} gc_compact_ratio={:.3} gc_transient_pressure_entries_before={} gc_canonical_cache_entries_before={} gc_skipped_with_transient_growth={} canonical_packed_cache_entries={} canonical_oriented_cache_entries={} direct_parent_cache_entries={} structural_fast_path_cache_entries={} packed_structural_fast_path_cache_entries={} oriented_result_cache_entries={} packed_transform_intern_entries={} packed_d4_misses={} packed_inverse_transform_hits={} packed_recursive_transform_hits={} packed_recursive_transform_misses={} packed_overlap_outputs={} packed_result_materializations={} session_full_grid_materializations={} embedded_result_bounded_extractions={} clipped_viewport_extractions={} checkpoint_cell_materializations={} oracle_confirmation_materializations={} dense_fallback_invocations={} transformed_node_materializations={} gc_reason={} gc_runs={} gc_skips={}",
        label,
        elapsed.as_micros(),
        summary.total_nodes,
        summary.retained_roots,
        summary.nodes_match_intern,
        summary.dependency_stalls,
        summary.jump_result_hit_rate,
        summary.jump_result_miss_count,
        summary.oriented_result_hit_rate,
        summary.root_result_hit_rate,
        summary.jump_presence_hit_rate,
        summary.overlap_hit_rate,
        summary.overlap_local_reuse_rate,
        summary.symmetry_gate_allow_rate,
        summary.symmetry_gate_canonical_cache_bypasses,
        summary.structural_fast_path_hit_rate,
        summary.canonical_node_cache_hit_rate,
        summary.canonical_packed_cache_hit_rate,
        summary.canonical_oriented_cache_hit_rate,
        summary.direct_parent_winner_hit_rate,
        summary.direct_parent_cached_result_hits,
        summary.direct_parent_winner_fallbacks,
        summary.blocked_symmetry_scan_fallbacks,
        summary.admitted_symmetry_scan_fallbacks,
        summary.total_symmetry_scan_fallbacks,
        summary.symmetry_jump_result_hits,
        summary.simd_lane_coverage,
        summary.scalar_commit_ratio,
        summary.probes_per_scheduler_task,
        summary.recursive_overlap_batch_rate,
        summary.gc_reclaim_ratio,
        summary.gc_compact_ratio,
        summary.gc_transient_pressure_entries_before,
        summary.gc_canonical_cache_entries_before,
        summary.gc_skipped_with_transient_growth,
        summary.canonical_packed_cache_entries,
        summary.canonical_oriented_cache_entries,
        summary.direct_parent_cache_entries,
        summary.structural_fast_path_cache_entries,
        summary.packed_structural_fast_path_cache_entries,
        summary.oriented_result_cache_entries,
        summary.packed_transform_intern_entries,
        summary.packed_d4_canonicalization_misses,
        summary.packed_inverse_transform_hits,
        summary.packed_recursive_transform_hits,
        summary.packed_recursive_transform_misses,
        summary.packed_overlap_outputs_produced,
        summary.packed_cache_result_materializations,
        summary.session_full_grid_materializations,
        summary.embedded_result_bounded_extractions,
        summary.clipped_viewport_extractions,
        summary.checkpoint_cell_materializations,
        summary.oracle_confirmation_materializations,
        summary.dense_fallback_invocations,
        summary.transformed_node_materializations,
        summary.gc_reason,
        summary.gc_runs,
        summary.gc_skips
    );
}

pub(super) fn print_hashlife_gate_comparison(
    label: &str,
    mode: &str,
    elapsed: std::time::Duration,
    summary: HashLifeDiagnosticSummary,
) {
    eprintln!(
        "{} mode={} total_us={} jump_result_hit_rate={:.3} oriented_result_hit_rate={:.3} root_result_hit_rate={:.3} jump_presence_hit_rate={:.3} symmetry_gate_allow_rate={:.3} symmetry_gate_canonical_cache_bypasses={} structural_fast_path_hit_rate={:.3} canonical_node_cache_hit_rate={:.3} canonical_packed_cache_hit_rate={:.3} canonical_oriented_cache_hit_rate={:.3} direct_parent_winner_hit_rate={:.3} direct_parent_cached_result_hits={} direct_parent_winner_fallbacks={} blocked_symmetry_scan_fallbacks={} admitted_symmetry_scan_fallbacks={} total_symmetry_scan_fallbacks={} symmetry_jump_result_hits={} probes_per_scheduler_task={:.3} simd_lane_coverage={:.3} overlap_hit_rate={:.3} gc_reason={} gc_runs={} gc_transient_pressure_entries_before={} gc_canonical_cache_entries_before={} canonical_packed_cache_entries={} canonical_oriented_cache_entries={} direct_parent_cache_entries={} structural_fast_path_cache_entries={} packed_structural_fast_path_cache_entries={} oriented_result_cache_entries={}",
        label,
        mode,
        elapsed.as_micros(),
        summary.jump_result_hit_rate,
        summary.oriented_result_hit_rate,
        summary.root_result_hit_rate,
        summary.jump_presence_hit_rate,
        summary.symmetry_gate_allow_rate,
        summary.symmetry_gate_canonical_cache_bypasses,
        summary.structural_fast_path_hit_rate,
        summary.canonical_node_cache_hit_rate,
        summary.canonical_packed_cache_hit_rate,
        summary.canonical_oriented_cache_hit_rate,
        summary.direct_parent_winner_hit_rate,
        summary.direct_parent_cached_result_hits,
        summary.direct_parent_winner_fallbacks,
        summary.blocked_symmetry_scan_fallbacks,
        summary.admitted_symmetry_scan_fallbacks,
        summary.total_symmetry_scan_fallbacks,
        summary.symmetry_jump_result_hits,
        summary.probes_per_scheduler_task,
        summary.simd_lane_coverage,
        summary.overlap_hit_rate,
        summary.gc_reason,
        summary.gc_runs,
        summary.gc_transient_pressure_entries_before,
        summary.gc_canonical_cache_entries_before,
        summary.canonical_packed_cache_entries,
        summary.canonical_oriented_cache_entries,
        summary.direct_parent_cache_entries,
        summary.structural_fast_path_cache_entries,
        summary.packed_structural_fast_path_cache_entries,
        summary.oriented_result_cache_entries,
    );
}

pub(super) fn jump_presence_probe_misses(
    stats: &crate::hashlife::HashLifeRuntimeStats,
) -> usize {
    stats
        .jump_presence_probe_lanes
        .saturating_sub(stats.jump_presence_probe_hits)
}

pub(super) fn assert_stepper_matches_single_step(grid: &BitGrid) {
    let advanced = crate::hashlife::HashLifeEngine::default().advance(grid, 1);
    assert_eq!(normalize(&advanced).0, normalize(&step_grid(grid)).0);
}
