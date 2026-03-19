use crate::bitgrid::{BitGrid, Coord};
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::{HashLifeEngine, HashLifeSession};
use crate::life::{GameOfLife, step_grid};
use crate::normalize::normalize;
use std::time::Instant;

fn assert_hashlife_matches_stepper(grid: crate::bitgrid::BitGrid, generations: u64) {
    let mut game = GameOfLife::new(grid.clone());
    for _ in 0..generations {
        game.step_with_changes();
    }

    let advanced = HashLifeEngine::default().advance(&grid, generations);
    assert_eq!(normalize(&advanced).0, normalize(game.grid()).0);
}

fn grid_from_mask(width: Coord, height: Coord, mask: u32) -> BitGrid {
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

fn mirror_grid_x(grid: &BitGrid) -> BitGrid {
    let (min_x, _, max_x, _) = grid.bounds().expect("grid should be non-empty");
    let mirrored = grid
        .live_cells()
        .into_iter()
        .map(|(x, y)| (max_x - (x - min_x), y))
        .collect::<Vec<_>>();
    BitGrid::from_cells(&mirrored)
}

fn print_hashlife_summary(
    label: &str,
    elapsed: std::time::Duration,
    summary: crate::hashlife::HashLifeDiagnosticSummary,
) {
    eprintln!(
        "{} total_us={} nodes={} retained_roots={} nodes_match_intern={} dependency_stalls={} jump_result_hit_rate={:.3} root_result_hit_rate={:.3} jump_presence_hit_rate={:.3} overlap_hit_rate={:.3} overlap_local_reuse_rate={:.3} symmetry_gate_allow_rate={:.3} canonical_cache_hit_rate={:.3} symmetry_jump_result_hits={} simd_lane_coverage={:.3} scalar_commit_ratio={:.3} probes_per_scheduler_task={:.3} recursive_overlap_batch_rate={:.3} gc_reclaim_ratio={:.3} gc_compact_ratio={:.3} packed_d4_misses={} packed_inverse_transform_hits={} packed_recursive_transform_hits={} packed_recursive_transform_misses={} packed_overlap_outputs={} packed_result_materializations={} transformed_node_materializations={} gc_reason={} gc_runs={} gc_skips={}",
        label,
        elapsed.as_micros(),
        summary.total_nodes,
        summary.retained_roots,
        summary.nodes_match_intern,
        summary.dependency_stalls,
        summary.jump_result_hit_rate,
        summary.root_result_hit_rate,
        summary.jump_presence_hit_rate,
        summary.overlap_hit_rate,
        summary.overlap_local_reuse_rate,
        summary.symmetry_gate_allow_rate,
        summary.canonical_cache_hit_rate,
        summary.symmetry_jump_result_hits,
        summary.simd_lane_coverage,
        summary.scalar_commit_ratio,
        summary.probes_per_scheduler_task,
        summary.recursive_overlap_batch_rate,
        summary.gc_reclaim_ratio,
        summary.gc_compact_ratio,
        summary.packed_d4_canonicalization_misses,
        summary.packed_inverse_transform_hits,
        summary.packed_recursive_transform_hits,
        summary.packed_recursive_transform_misses,
        summary.packed_overlap_outputs_produced,
        summary.packed_cache_result_materializations,
        summary.transformed_node_materializations,
        summary.gc_reason,
        summary.gc_runs,
        summary.gc_skips
    );
}

fn print_hashlife_gate_comparison(
    label: &str,
    mode: &str,
    elapsed: std::time::Duration,
    summary: crate::hashlife::HashLifeDiagnosticSummary,
) {
    eprintln!(
        "{} mode={} total_us={} jump_result_hit_rate={:.3} root_result_hit_rate={:.3} jump_presence_hit_rate={:.3} symmetry_gate_allow_rate={:.3} canonical_cache_hit_rate={:.3} symmetry_jump_result_hits={} probes_per_scheduler_task={:.3} simd_lane_coverage={:.3} overlap_hit_rate={:.3}",
        label,
        mode,
        elapsed.as_micros(),
        summary.jump_result_hit_rate,
        summary.root_result_hit_rate,
        summary.jump_presence_hit_rate,
        summary.symmetry_gate_allow_rate,
        summary.canonical_cache_hit_rate,
        summary.symmetry_jump_result_hits,
        summary.probes_per_scheduler_task,
        summary.simd_lane_coverage,
        summary.overlap_hit_rate,
    );
}

#[test]
fn hashlife_matches_glider_after_single_step() {
    let grid = pattern_by_name("glider").unwrap();
    let advanced = HashLifeEngine::default().advance(&grid, 1);
    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_matches_blinker_after_large_even_and_odd_jumps() {
    let grid = pattern_by_name("blinker").unwrap();
    let mut oracle = HashLifeEngine::default();
    let even = oracle.advance(&grid, 1_000);
    let odd = oracle.advance(&grid, 1_001);

    assert_eq!(normalize(&even).0, normalize(&grid).0);
    assert_eq!(normalize(&odd).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_session_gosper_gun_core_survives_ten_million_scale_period_multiple() {
    let initial = pattern_by_name("gosper_glider_gun").unwrap();
    let (min_x, min_y, max_x, max_y) = initial
        .bounds()
        .expect("gosper glider gun should be non-empty");
    let target = 10_000_020_u64;
    assert_eq!(target % 30, 0, "test target must land on the gun period");

    let mut session = HashLifeSession::new();
    session.load_grid(&initial);
    session.advance_root(target);

    let sampled_core = session
        .sample_region(min_x, min_y, max_x, max_y)
        .expect("bounded gun core sample should remain available");

    assert_eq!(
        normalize(&sampled_core).0,
        normalize(&initial).0,
        "gosper gun core should still match generation 0 after a deep period-aligned jump"
    );
    assert!(
        session.sample_materializations() <= 1,
        "bounded gun-core regression should avoid full-grid materialization, got {} materializations",
        session.sample_materializations()
    );
}

#[test]
fn hashlife_snapshot_roundtrips_session_state() {
    let initial = pattern_by_name("glider").unwrap();
    let mut session = HashLifeSession::new();
    session.load_grid(&initial);
    session.advance_root(4_096);

    let snapshot = session
        .export_snapshot_string()
        .expect("loaded session should export a snapshot");
    let expected_generation = session.generation();
    let expected_origin = session.origin();
    let expected_population = session.population();
    let expected_bounds = session.bounds();
    let expected_checkpoint = session.signature_checkpoint().cloned();
    let expected_grid = session
        .sample_grid()
        .expect("snapshot source should be materializable");

    let mut restored = HashLifeSession::new();
    restored
        .load_snapshot_string(&snapshot)
        .expect("snapshot should reload");

    assert_eq!(restored.generation(), expected_generation);
    assert_eq!(restored.origin(), expected_origin);
    assert_eq!(restored.population(), expected_population);
    assert_eq!(restored.bounds(), expected_bounds);
    assert_eq!(
        restored.signature_checkpoint().cloned(),
        expected_checkpoint
    );
    assert_eq!(
        normalize(
            &restored
                .sample_grid()
                .expect("restored snapshot should be materializable")
        )
        .0,
        normalize(&expected_grid).0
    );
}

#[test]
fn hashlife_snapshot_persists_deep_run_resume() {
    let initial = pattern_by_name("glider").unwrap();
    let mut uninterrupted = HashLifeSession::new();
    uninterrupted.load_grid(&initial);
    uninterrupted.advance_root(1_000_000);
    let snapshot = uninterrupted
        .export_snapshot_string()
        .expect("deep run should export a snapshot");

    uninterrupted.advance_root(1_024);
    let expected = uninterrupted
        .signature_checkpoint()
        .cloned()
        .expect("continued deep run should have a checkpoint");

    let mut resumed = HashLifeSession::new();
    resumed
        .load_snapshot_string(&snapshot)
        .expect("snapshot should reload");
    resumed.advance_root(1_024);
    let actual = resumed
        .signature_checkpoint()
        .cloned()
        .expect("resumed deep run should have a checkpoint");

    assert_eq!(actual, expected);
}

#[test]
fn hashlife_snapshot_export_is_deterministic_for_same_session() {
    let initial = pattern_by_name("gosper_glider_gun").unwrap();
    let mut session = HashLifeSession::new();
    session.load_grid(&initial);
    session.advance_root(1_024);

    let first = session
        .export_snapshot_string()
        .expect("loaded session should export a snapshot");
    let second = session
        .export_snapshot_string()
        .expect("repeated export should succeed");
    assert_eq!(first, second);
}

#[test]
fn hashlife_matches_single_step_on_random_soup() {
    let grid = random_soup(24, 24, 20, 0xD1B54A32D192ED03);
    let advanced = HashLifeEngine::default().advance(&grid, 1);
    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
}

#[test]
fn hashlife_matches_glider_after_power_of_two_jump() {
    let grid = pattern_by_name("glider").unwrap();
    assert_hashlife_matches_stepper(grid, 256);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_power_of_two_jump() {
    let grid = random_soup(24, 24, 20, 0xDEADBEEFCAFEBABE);
    assert_hashlife_matches_stepper(grid, 64);
}

#[test]
fn hashlife_matches_stepper_on_glider_prime_jump() {
    let grid = pattern_by_name("glider").unwrap();
    assert_hashlife_matches_stepper(grid, 269);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_prime_jump() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    assert_hashlife_matches_stepper(grid, 269);
}

#[test]
fn hashlife_matches_random_soup_after_large_power_of_two_jump() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    assert_hashlife_matches_stepper(grid, 256);
}

#[test]
fn hashlife_matches_stepper_on_random_soup_large_prime_jump() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    assert_hashlife_matches_stepper(grid, 509);
}

#[test]
fn hashlife_segmented_prime_equivalence_matches_single_advance() {
    let grid = random_soup(24, 24, 20, 0x1234_5678_9ABC_DEF0);
    let mut oracle = HashLifeEngine::default();
    let combined = oracle.advance(&grid, 509);
    let intermediate = oracle.advance(&grid, 256);
    let segmented = oracle.advance(&intermediate, 253);
    assert_eq!(normalize(&combined).0, normalize(&segmented).0);
}

#[test]
fn hashlife_gc_keeps_state_bounded_and_reusable() {
    let grid = random_soup(24, 24, 20, 0xA5A5_5A5A_DEAD_BEEF);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, 509);
    let first = oracle.runtime_stats();
    assert!(first.nodes > 2);
    assert!(first.retained_roots >= 1);
    assert_eq!(first.nodes, first.intern);
    assert!(first.jump_cache_before_clear >= first.jump_cache);

    oracle.advance(&grid, 509);
    let second = oracle.runtime_stats();
    assert!(second.retained_roots >= 1);
    assert!(second.jump_cache_before_clear > 0);
    assert_eq!(second.nodes, second.intern);
    assert!(
        second.nodes <= first.nodes.saturating_mul(2),
        "repeated identical deep runs should keep retained canonical state bounded, first={first:?} second={second:?}"
    );

    oracle.advance(&grid, 509);
    let third = oracle.runtime_stats();
    assert!(third.retained_roots >= 1);
    assert_eq!(third.nodes, third.intern);
    assert!(third.jump_cache_before_clear >= third.jump_cache);
    assert!(
        third.nodes <= second.nodes,
        "repeated identical deep runs should stabilize retained canonical state, second={second:?} third={third:?}"
    );

    let other = random_soup(24, 24, 20, 0x0DDC_0FFE_EE11_2233);
    oracle.advance(&other, 509);
    let fourth = oracle.runtime_stats();
    assert!(fourth.retained_roots >= 1);
    assert_eq!(fourth.nodes, fourth.intern);
    assert!(fourth.jump_cache_before_clear >= fourth.jump_cache);
    assert!(
        fourth.nodes > 2,
        "advancing a different grid should still leave a valid reusable retained state, fourth={fourth:?}"
    );
}

#[test]
fn hashlife_node_store_fingerprints_match_recomputed_keys() {
    let grid = random_soup(24, 24, 20, 0xF1A5_C0DE_1234_5678);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, 256);
    assert!(oracle.verify_node_fingerprint_invariants());
    assert!(oracle.verify_intern_fingerprint_fast_path_parity());
    let stats = oracle.runtime_stats();
    assert!(stats.cached_fingerprint_probes > 0, "{stats:?}");
}

#[test]
fn hashlife_gc_rebuild_preserves_fingerprint_invariants() {
    let grid = random_soup(24, 24, 20, 0xBAD5_EED5_2468_ACE0);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, 509);
    oracle.advance(&grid, 509);
    assert!(oracle.verify_node_fingerprint_invariants());
    assert!(oracle.verify_intern_fingerprint_fast_path_parity());
}

#[test]
fn hashlife_overlap_batch_matches_scalar_overlap_builder() {
    let grid = random_soup(48, 48, 22, 0x0BAD_F00D_CAFE_1234);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_overlap_batch_parity(&grid));
}

#[test]
fn hashlife_canonical_overlap_batch_matches_raw_overlap_builder() {
    let grid = random_soup(48, 48, 22, 0x0ACE_F00D_BAAD_C0DE);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_canonical_overlap_batch_parity(&grid));
}

#[test]
fn hashlife_overlap_batch_dedupes_duplicate_miss_parents_locally() {
    let grid = random_soup(48, 48, 22, 0x1234_5678_0BAD_F00D);
    let mut oracle = HashLifeEngine::default();
    let (misses, local_reuse) = oracle.duplicate_overlap_batch_dedupe_stats(&grid);

    assert_eq!(
        misses, 1,
        "duplicate parent overlap miss path should build once"
    );
    assert_eq!(
        local_reuse, 1,
        "second duplicate parent should reuse staged overlap work"
    );
}

#[test]
fn hashlife_jump_result_batch_reuses_duplicate_queries() {
    let grid = random_soup(64, 64, 20, 0xFACE_FEED_0BAD_F00D);
    let mut oracle = HashLifeEngine::default();
    let (unique, reused) = oracle.duplicate_jump_batch_query_stats(&grid);

    assert_eq!(
        unique, 1,
        "grouped jump-result batch should probe one unique key"
    );
    assert_eq!(
        reused, 3,
        "remaining duplicate lanes should reuse the grouped result"
    );
}

#[test]
fn hashlife_packed_jump_cache_roundtrip_matches_materialized_result() {
    let grid = random_soup(64, 64, 20, 0xA55A_5AA5_FACE_CAFE);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_packed_jump_cache_roundtrip(&grid, 2));
    let stats = oracle.runtime_stats();
    assert!(stats.jump_result_cache_hits > 0, "{stats:?}");
    assert!(stats.packed_cache_result_materializations > 0, "{stats:?}");
}

#[test]
fn hashlife_oriented_result_cache_reuses_materialized_output() {
    let grid = random_soup(64, 64, 20, 0x0DDC_0FFE_EE11_BAAD);
    let mut oracle = HashLifeEngine::default();
    let (first_delta, second_delta) = oracle.duplicate_oriented_result_cache_stats(&grid);
    assert!(
        first_delta.0 > 0,
        "first oriented result batch should materialize once"
    );
    assert!(
        first_delta.1 > 0,
        "first oriented result batch should perform inverse transform work"
    );
    assert_eq!(
        second_delta.1, 0,
        "second identical oriented batch should hit oriented-result cache"
    );
}

#[test]
fn hashlife_packed_recursive_transform_matches_node_transform() {
    let grid = random_soup(64, 64, 20, 0xC001_CAFE_FEED_FACE);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_packed_transform_parity(&grid));
}

#[test]
fn hashlife_packed_transform_root_key_matches_materialized_root() {
    let grid = random_soup(64, 64, 20, 0x0F0F_F0F0_AAAA_5555);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_packed_transform_root_key_parity(&grid));
}

#[test]
fn hashlife_packed_canonicalization_matches_all_symmetry_variants() {
    let grid = random_soup(64, 64, 20, 0xABCD_EF01_2345_6789);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_packed_canonicalization_symmetry_parity(&grid));
}

#[test]
fn hashlife_canonical_child_key_batch_matches_scalar_keys() {
    let grid = random_soup(64, 64, 20, 0xACED_BEEF_1234_5678);
    let mut oracle = HashLifeEngine::default();
    assert!(oracle.verify_canonical_child_key_batch_parity(&grid));
}

#[test]
fn hashlife_matches_stepper_on_all_4x4_single_steps() {
    let mut oracle = HashLifeEngine::default();
    for mask in 0_u32..(1 << 16) {
        let grid = grid_from_mask(4, 4, mask);
        let advanced = oracle.advance(&grid, 1);
        assert_eq!(
            normalize(&advanced).0,
            normalize(&step_grid(&grid)).0,
            "4x4 mask {mask:#06x}"
        );
    }
}

#[test]
fn hashlife_symmetric_workloads_reduce_work_against_fresh_mirror_run() {
    let grid = pattern_by_name("glider").unwrap();
    let mirrored = mirror_grid_x(&grid);
    let generations = 16_u64;
    assert_ne!(
        grid.live_cells(),
        mirrored.live_cells(),
        "sanity check failed: mirrored workload should not be identical to the original grid"
    );
    let mut engine = HashLifeEngine::default();

    engine.advance(&grid, generations);
    let second = engine.advance(&mirrored, generations);
    let second_stats = engine.runtime_stats();

    let mut game = GameOfLife::new(mirrored.clone());
    for _ in 0..generations {
        game.step_with_changes();
    }
    let mut fresh_engine = HashLifeEngine::default();
    let fresh_mirrored = fresh_engine.advance(&mirrored, generations);
    let fresh_stats = fresh_engine.runtime_stats();

    assert_eq!(normalize(&second).0, normalize(game.grid()).0);
    assert_eq!(normalize(&fresh_mirrored).0, normalize(game.grid()).0);
    assert!(
        second_stats.jump_presence_misses < fresh_stats.jump_presence_misses
            || second_stats.jump_presence_probe_hits > fresh_stats.jump_presence_probe_hits
            || second_stats.packed_recursive_transform_misses
                < fresh_stats.packed_recursive_transform_misses
            || second_stats.packed_cache_result_materializations
                < fresh_stats.packed_cache_result_materializations,
        "expected mirrored workload to reduce work against a fresh mirror run, second={second_stats:?} fresh={fresh_stats:?}"
    );
}

#[test]
fn hashlife_matches_stepper_on_sampled_5x5_small_jumps() {
    const MASKS: [u32; 8] = [
        0x0000000, 0x0000001, 0x001f000, 0x0108421, 0x1555555, 0x0f0f0f0, 0x1249249, 0x1ffffff,
    ];

    for mask in MASKS {
        let grid = grid_from_mask(5, 5, mask);
        for generations in [1_u64, 2, 4, 5] {
            let mut game = GameOfLife::new(grid.clone());
            for _ in 0..generations {
                game.step_with_changes();
            }
            let advanced = HashLifeEngine::default().advance(&grid, generations);
            assert_eq!(
                normalize(&advanced).0,
                normalize(game.grid()).0,
                "5x5 mask {mask:#08x} generations={generations}"
            );
        }
    }
}

#[test]
fn hashlife_step0_simd_batches_are_exercised_on_large_single_step() {
    let mut cells = random_soup(96, 96, 22, 0xBAD5_EED5_1234_5678).live_cells();
    cells.extend(
        pattern_by_name("glider")
            .unwrap()
            .translated(160, 24)
            .live_cells(),
    );
    cells.extend(
        pattern_by_name("blinker")
            .unwrap()
            .translated(24, 160)
            .live_cells(),
    );
    let grid = BitGrid::from_cells(&cells);

    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, 1);
    let stats = oracle.runtime_stats();
    let provisional_records_built = stats.step0_provisional_records
        + stats.phase1_provisional_records
        + stats.phase2_provisional_records;
    let simd_candidate_lanes = stats.step0_provisional_records
        + stats.phase1_provisional_records
        + stats.phase2_provisional_records;
    let simd_batches =
        stats.step0_simd_batches + stats.phase1_simd_batches + stats.phase2_simd_batches;

    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
    assert!(provisional_records_built > 0, "{stats:?}");
    assert!(simd_candidate_lanes > 0, "{stats:?}");
    assert!(simd_batches > 0, "{stats:?}");
    assert!(stats.simd_disabled_fast_exits > 0, "{stats:?}");
    assert!(stats.overlap_prep_batches > 0, "{stats:?}");
    assert!(stats.cache_probe_batches > 0, "{stats:?}");
}

#[test]
fn hashlife_step0_small_candidate_batches_still_run_through_simd() {
    let grid = random_soup(40, 40, 20, 0xFACE_FEED_1234_5678);
    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, 1);
    let stats = oracle.runtime_stats();
    let provisional_records_built = stats.step0_provisional_records
        + stats.phase1_provisional_records
        + stats.phase2_provisional_records;

    assert_eq!(normalize(&advanced).0, normalize(&step_grid(&grid)).0);
    assert!(provisional_records_built >= 1, "{stats:?}");
    assert!(stats.step0_simd_batches >= 1, "{stats:?}");
    assert!(stats.step0_simd_lanes >= 1, "{stats:?}");
}

#[test]
fn hashlife_recursive_phase_simd_batches_are_exercised_on_deep_jump() {
    let grid = random_soup(64, 64, 24, 0x1234_5678_AAAA_5555);
    let mut oracle = HashLifeEngine::default();
    let advanced = oracle.advance(&grid, 256);
    let stats = oracle.runtime_stats();

    let mut game = GameOfLife::new(grid.clone());
    for _ in 0..256 {
        game.step_with_changes();
    }

    assert_eq!(normalize(&advanced).0, normalize(game.grid()).0);
    assert!(stats.phase1_provisional_records > 0, "{stats:?}");
    assert!(stats.phase2_provisional_records > 0, "{stats:?}");
    assert!(stats.phase1_simd_lanes > 0, "{stats:?}");
    assert!(stats.phase2_simd_lanes > 0, "{stats:?}");
    assert!(stats.phase1_simd_batches > 0, "{stats:?}");
    assert!(stats.phase2_simd_batches > 0, "{stats:?}");
    assert!(stats.scalar_commit_lanes > 0, "{stats:?}");
}

#[test]
fn hashlife_summary_tracks_symmetry_reuse_invariants() {
    let grid = pattern_by_name("glider").unwrap();
    let mirrored = mirror_grid_x(&grid);
    let mut oracle = HashLifeEngine::default();
    oracle.advance(&grid, 16);
    oracle.advance(&mirrored, 16);
    let summary = oracle.diagnostic_summary();

    assert!(summary.nodes_match_intern, "{summary:?}");
    assert_eq!(summary.dependency_stalls, 0, "{summary:?}");
    assert!(summary.symmetry_jump_result_hits > 0, "{summary:?}");
    assert!(
        summary.jump_result_hit_rate >= 0.0 && summary.jump_result_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.root_result_hit_rate >= 0.0 && summary.root_result_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.jump_presence_hit_rate >= 0.0 && summary.jump_presence_hit_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.symmetry_gate_allow_rate >= 0.0 && summary.symmetry_gate_allow_rate <= 1.0,
        "{summary:?}"
    );
    assert!(
        summary.canonical_cache_hit_rate >= 0.0 && summary.canonical_cache_hit_rate <= 1.0,
        "{summary:?}"
    );
}

#[test]
fn hashlife_summary_tracks_gc_reclaim_invariants() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let mut oracle = HashLifeEngine::default();
    for _ in 0..3 {
        oracle.advance(&grid, 1_000_000);
    }
    let summary = oracle.diagnostic_summary();

    assert!(summary.nodes_match_intern, "{summary:?}");
    assert_eq!(summary.dependency_stalls, 0, "{summary:?}");
    assert!(summary.gc_runs > 0, "{summary:?}");
    assert!(summary.gc_reclaim_ratio > 0.9, "{summary:?}");
    assert!(summary.gc_compact_ratio > 0.9, "{summary:?}");
}

#[test]
#[ignore = "diagnostic symmetry-gate random benchmark"]
fn hashlife_diagnostic_symmetry_gate_random_soup() {
    let grid = random_soup(96, 96, 20, 0x1357_9BDF_2468_ACE0);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, 509);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_sym_gate_random", elapsed, summary);
}

#[test]
#[ignore = "diagnostic symmetry-gate structured benchmark"]
fn hashlife_diagnostic_symmetry_gate_structured_workload() {
    let mut cells = pattern_by_name("gosper_glider_gun").unwrap().live_cells();
    cells.extend(
        mirror_grid_x(&pattern_by_name("gosper_glider_gun").unwrap())
            .translated(256, 0)
            .live_cells(),
    );
    cells.extend(
        pattern_by_name("glider")
            .unwrap()
            .translated(128, 128)
            .live_cells(),
    );
    let grid = BitGrid::from_cells(&cells);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, 3_000_000);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_sym_gate_structured", elapsed, summary);
}

#[test]
#[ignore = "diagnostic symmetry-gate comparison benchmark"]
fn hashlife_diagnostic_symmetry_gate_comparison_random_soup() {
    let grid = random_soup(96, 96, 20, 0x1357_9BDF_2468_ACE0);
    let configs = [
        ("default", 8_u32, 4_096_u64),
        ("wider", 12_u32, 65_536_u64),
        ("ungated", u32::MAX, u64::MAX),
    ];
    for (mode, max_level, max_population) in configs {
        let mut oracle = HashLifeEngine::with_symmetry_gate_for_tests(max_level, max_population);
        let start = Instant::now();
        oracle.advance(&grid, 509);
        let elapsed = start.elapsed();
        let summary = oracle.diagnostic_summary();
        print_hashlife_gate_comparison("hashlife_sym_gate_compare_random", mode, elapsed, summary);
    }
}

#[test]
#[ignore = "diagnostic symmetry-gate comparison benchmark"]
fn hashlife_diagnostic_symmetry_gate_comparison_structured() {
    let mut cells = pattern_by_name("gosper_glider_gun").unwrap().live_cells();
    cells.extend(
        mirror_grid_x(&pattern_by_name("gosper_glider_gun").unwrap())
            .translated(256, 0)
            .live_cells(),
    );
    cells.extend(
        pattern_by_name("glider")
            .unwrap()
            .translated(128, 128)
            .live_cells(),
    );
    let grid = BitGrid::from_cells(&cells);
    let configs = [
        ("default", 8_u32, 4_096_u64),
        ("wider", 12_u32, 65_536_u64),
        ("ungated", u32::MAX, u64::MAX),
    ];
    for (mode, max_level, max_population) in configs {
        let mut oracle = HashLifeEngine::with_symmetry_gate_for_tests(max_level, max_population);
        let start = Instant::now();
        oracle.advance(&grid, 3_000_000);
        let elapsed = start.elapsed();
        let summary = oracle.diagnostic_summary();
        print_hashlife_gate_comparison(
            "hashlife_sym_gate_compare_structured",
            mode,
            elapsed,
            summary,
        );
    }
}

#[test]
#[ignore = "diagnostic benchmark"]
fn hashlife_diagnostic_medium_prime_jump_benchmark() {
    let grid = random_soup(64, 64, 20, 0xDEADBEEFCAFEBABE);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, 509);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_bench", elapsed, summary);
}

#[test]
#[ignore = "diagnostic GC/runtime benchmark"]
fn hashlife_gc_diagnostic_repeated_deep_runs() {
    let grid = pattern_by_name("gosper_glider_gun").unwrap();
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    for _ in 0..3 {
        oracle.advance(&grid, 3_000_000);
    }
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_gc_diag", elapsed, summary);
}

#[test]
#[ignore = "diagnostic symmetry benchmark"]
fn hashlife_diagnostic_symmetric_mirror_reuse() {
    let grid = pattern_by_name("glider").unwrap();
    let mirrored = mirror_grid_x(&grid);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, 16);
    oracle.advance(&mirrored, 16);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_symmetry_diag", elapsed, summary);
}

#[test]
#[ignore = "diagnostic step0-heavy benchmark"]
fn hashlife_diagnostic_step0_heavy_single_step() {
    let mut cells = random_soup(96, 96, 22, 0xBAD5_EED5_1234_5678).live_cells();
    cells.extend(
        pattern_by_name("glider")
            .unwrap()
            .translated(160, 24)
            .live_cells(),
    );
    cells.extend(
        pattern_by_name("blinker")
            .unwrap()
            .translated(24, 160)
            .live_cells(),
    );
    let grid = BitGrid::from_cells(&cells);
    let mut oracle = HashLifeEngine::default();
    let start = Instant::now();
    oracle.advance(&grid, 1);
    let elapsed = start.elapsed();
    let summary = oracle.diagnostic_summary();
    print_hashlife_summary("hashlife_step0_diag", elapsed, summary);
}
