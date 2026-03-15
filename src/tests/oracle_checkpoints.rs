use crate::RequiredExt;
use crate::classify::Classification;
use crate::engine::SimulationSession;
use crate::generators::{pattern_by_name, random_soup};
use crate::hashlife::HashLifeSession;
use crate::oracle::{OracleRuntimeState, OracleSession};

fn large_stable_block_field() -> crate::bitgrid::BitGrid {
    let mut cells = Vec::new();
    for block_y in 0..36_i64 {
        for block_x in 0..36_i64 {
            let x = block_x * 4;
            let y = block_y * 4;
            cells.extend([(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]);
        }
    }
    crate::bitgrid::BitGrid::from_cells(&cells)
}

#[test]
fn hashlife_root_checkpoint_reuses_exact_stable_identity_without_traversal() {
    let grid = pattern_by_name("block").or_invariant("required value");
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    session
        .advance_root(1)
        .or_invariant("first checkpoint generation should complete");
    let first = *session
        .signature_checkpoint()
        .or_invariant("first centered checkpoint");
    session
        .advance_root(1)
        .or_invariant("second checkpoint generation should complete");
    let repeated = *session
        .signature_checkpoint()
        .or_invariant("repeated checkpoint");

    assert_eq!(repeated.identity, first.identity);
    assert_eq!(repeated.origin, first.origin);
    assert_eq!(session.checkpoint_profile().metadata_reads, 2);
    assert_eq!(session.checkpoint_profile().subtree_visits, 0);
    assert_eq!(session.sample_materializations(), 0);
    assert_eq!(
        session
            .runtime_stats()
            .materialization
            .checkpoint_cell_materializations,
        0
    );
}

#[test]
fn hashlife_root_checkpoint_keeps_translation_in_origin_when_root_repeats() {
    let grid = pattern_by_name("glider").or_invariant("required value");
    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(1_000, None);

    assert_eq!(
        outcome.classification,
        Classification::Spaceship {
            period: 4,
            first_seen: 0,
            delta: (1, 1),
            detected_at: 4,
        },
        "the oracle must classify glider translation rather than accepting a false stable repeat"
    );
    assert_eq!(simulation.hashlife_sample_materializations(), 0);
    assert_eq!(
        simulation
            .hashlife_runtime_stats()
            .materialization
            .checkpoint_cell_materializations,
        0
    );
}

#[test]
fn r_pentomino_runtime_target_projects_periodic_ash_and_retains_exact_state() {
    let grid = pattern_by_name("r_pentomino").or_invariant("required value");
    let target = 100_000_000;
    let mut simulation = SimulationSession::new();

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target(target, None);

    assert_eq!(outcome.final_generation, target, "outcome={outcome:?}");
    assert_eq!(outcome.failure, None, "outcome={outcome:?}");
    assert_eq!(outcome.population, 116, "outcome={outcome:?}");
    assert_eq!(
        outcome.classification,
        Classification::Unknown { simulated: target },
        "the exact finite target should not be overclassified as a globally repeating state: {outcome:?}"
    );
    assert_eq!(outcome.state, OracleRuntimeState::RetainedHashLife);
    assert_eq!(simulation.hashlife_generation(), target);
}

#[test]
fn hashlife_first_runtime_checkpoint_confirms_stable_cycle_and_syncs_generation() {
    let grid = pattern_by_name("block").or_invariant("required value");
    let target = 1_000_000_000_000;
    let mut simulation = SimulationSession::new();

    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target_hashlife_first(target, None);

    assert_eq!(outcome.final_generation, target);
    assert_eq!(simulation.hashlife_generation(), target);
    assert_eq!(outcome.state, OracleRuntimeState::RetainedHashLife);
    assert!(
        matches!(
            outcome.classification,
            Classification::Repeats {
                period,
                first_seen: 0
            } if period == 1
        ),
        "HashLife checkpoint repeat should confirm the stable cycle, got {:?}",
        outcome.classification
    );
    assert_eq!(
        simulation.hashlife_sample_materializations(),
        0,
        "retained-root checkpoints should avoid full-grid confirmation"
    );
    let runtime_stats = simulation.hashlife_runtime_stats();
    assert_eq!(
        runtime_stats
            .materialization
            .oracle_confirmation_materializations,
        0,
        "checkpoint repeat verification should not request a full-grid oracle confirmation: {runtime_stats:?}"
    );
    assert_eq!(
        runtime_stats
            .materialization
            .checkpoint_cell_materializations,
        0,
        "retained-root checkpointing must not materialize cells: {runtime_stats:?}"
    );
}

#[test]
fn hashlife_root_identity_epoch_changes_after_engine_gc_remapping() {
    let block = pattern_by_name("block").or_invariant("required value");
    let mut session = HashLifeSession::new();
    session
        .try_load_grid(&block)
        .or_invariant("test HashLife grid should load");
    let before = session
        .signature_checkpoint()
        .or_invariant("checkpoint before GC")
        .identity;
    session.finish();

    let mut observed_gc = false;
    for seed in 0..64_u64 {
        session
            .try_load_grid(&random_soup(96, 96, 35, seed))
            .or_invariant("test HashLife grid should load");
        session
            .advance_root(16)
            .or_invariant("checkpoint stress segment should complete");
        session.finish();
        if session.runtime_stats().gc.gc_runs > 0 {
            observed_gc = true;
            break;
        }
    }
    assert!(
        observed_gc,
        "production calls must trigger the GC remap in this regression"
    );

    session
        .try_load_grid(&block)
        .or_invariant("test HashLife grid should load");
    let after = session
        .signature_checkpoint()
        .or_invariant("checkpoint after GC")
        .identity;
    assert!(
        !after.same_epoch(before),
        "post-GC root ids must be isolated from pre-remap checkpoint history"
    );
}

#[test]
fn large_stable_checkpoint_detects_repeat_without_cell_or_subtree_work() {
    let grid = large_stable_block_field();
    assert!(grid.population() > 4_096);

    let mut root_session = HashLifeSession::new();
    root_session
        .try_load_grid(&grid)
        .or_invariant("test HashLife grid should load");
    let first = *root_session
        .signature_checkpoint()
        .or_invariant("large initial checkpoint");
    root_session
        .advance_root(1)
        .or_invariant("root checkpoint generation should complete");
    let repeated = *root_session
        .signature_checkpoint()
        .or_invariant("large repeated checkpoint");
    assert_eq!(repeated.identity, first.identity);
    assert_eq!(root_session.checkpoint_profile().subtree_visits, 0);
    assert_eq!(root_session.sample_materializations(), 0);
    assert_eq!(
        root_session
            .runtime_stats()
            .materialization
            .checkpoint_cell_materializations,
        0
    );

    let mut simulation = SimulationSession::new();
    let outcome = OracleSession::new(grid, 0, Default::default(), &mut simulation)
        .advance_runtime_target_hashlife_first(1_000_000, None);
    assert_eq!(
        outcome.classification,
        Classification::Repeats {
            period: 1,
            first_seen: 0
        },
        "large stable state must be classified from O(1) retained-root checkpoints"
    );
    assert_eq!(simulation.hashlife_sample_materializations(), 0);
    assert_eq!(
        simulation
            .hashlife_runtime_stats()
            .materialization
            .checkpoint_cell_materializations,
        0
    );
}
