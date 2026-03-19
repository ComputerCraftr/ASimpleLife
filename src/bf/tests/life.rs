use super::*;
use crate::bf::life_backend::{
    BfLifeCircuitError, CircuitPhase, compile_to_life_circuit, serialize_life_circuit,
    serialize_life_circuit_hashlife,
};
use crate::bf::LoweredBfInstr;
use crate::bf::life_macro_library::{
    LifeMacroKind, LifeMacroOrientation, instantiate_macro_cells, life_macro_template,
    life_macro_templates, transform_cell,
};
use crate::bitgrid::BitGrid;
use crate::generators::pattern_from_file;
use crate::life::step_grid;
use crate::persistence::{HASHLIFE_SNAPSHOT_MAGIC, LIFE_GRID_MAGIC, deserialize_grid, deserialize_life_grid};
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn crop_region(grid: &BitGrid, min_x: i64, min_y: i64, max_x: i64, max_y: i64) -> BitGrid {
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|(x, y)| *x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y)
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn step_n(grid: &BitGrid, generations: u64) -> BitGrid {
    let mut current = grid.clone();
    for _ in 0..generations {
        current = step_grid(&current);
    }
    current
}

fn connected_components(grid: &BitGrid) -> Vec<Vec<(i64, i64)>> {
    let live = grid.live_cells();
    let mut remaining = live.iter().copied().collect::<HashSet<_>>();
    let mut components = Vec::new();
    while let Some(&start) = remaining.iter().next() {
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        remaining.remove(&start);
        while let Some((x, y)) = queue.pop_front() {
            component.push((x, y));
            for ny in (y - 1)..=(y + 1) {
                for nx in (x - 1)..=(x + 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    if remaining.remove(&(nx, ny)) {
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

fn unique_temp_path(stem: &str, ext: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "a_simple_life_{}_{}_{}_{}.{}",
        stem,
        std::process::id(),
        timestamp,
        counter,
        ext
    ))
}

#[test]
fn life_circuit_compiles_basic_loop_machine() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+++[->++<]>.<."), life_opts()).unwrap();
    assert_eq!(circuit.tape_len, 64);
    assert_eq!(circuit.cell_bits, 8);
    assert_eq!(circuit.state.phase, CircuitPhase::Fetch);
    assert_eq!(circuit.program.last(), Some(&LoweredBfInstr::Halt));
    assert!(circuit
        .program
        .iter()
        .any(|instr| matches!(instr, LoweredBfInstr::Distribute(_))));
    assert!(!circuit
        .program
        .iter()
        .any(|instr| matches!(instr, LoweredBfInstr::JumpIfZero(_))));
}

#[test]
fn life_circuit_runs_add_clear_move_loop_and_output() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+++[->++<]>.<."), life_opts()).unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.phase, CircuitPhase::Halted);
    assert_eq!(circuit.state.outputs, vec![6, 0]);
    assert_eq!(circuit.state.tape[0], 0);
    assert_eq!(circuit.state.tape[1], 6);
}

#[test]
fn life_circuit_preserves_explicit_clear() {
    let mut circuit = compile_to_life_circuit(&[BfIr::Add(5), BfIr::Clear, BfIr::Output], life_opts())
        .unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.outputs, vec![0]);
}

#[test]
fn life_circuit_uses_explicit_phases() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+."), life_opts()).unwrap();
    assert_eq!(circuit.state.phase, CircuitPhase::Fetch);
    assert!(circuit.step().unwrap());
    assert_eq!(circuit.state.phase, CircuitPhase::Decode);
    assert!(circuit.step().unwrap());
    assert_eq!(circuit.state.phase, CircuitPhase::Evaluate);
    assert!(circuit.step().unwrap());
    assert_eq!(circuit.state.phase, CircuitPhase::Commit);
    assert!(circuit.step().unwrap());
    assert_eq!(circuit.state.phase, CircuitPhase::Fetch);
}

#[test]
fn life_circuit_serializer_uses_standard_life_grid_format() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+."), life_opts()).unwrap();
    let serialized = serialize_life_circuit(&circuit);
    assert!(serialized.starts_with(LIFE_GRID_MAGIC));
    assert_eq!(deserialize_life_grid(&serialized).unwrap(), circuit.to_initial_grid());
}

#[test]
fn life_circuit_can_be_serialized_as_hashlife_snapshot() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+."), life_opts()).unwrap();
    let serialized = serialize_life_circuit_hashlife(&circuit);
    assert!(serialized.starts_with(HASHLIFE_SNAPSHOT_MAGIC));
    assert_eq!(deserialize_grid(&serialized).unwrap(), circuit.to_initial_grid());
}

#[test]
fn life_circuit_rejects_input_ops() {
    let err = compile_to_life_circuit(&parse_only(",."), life_opts()).unwrap_err();
    assert_eq!(err, BfLifeCircuitError::InputUnsupported);
}

#[test]
fn life_circuit_rejects_signed_cells() {
    let err = compile_to_life_circuit(
        &parse_and_opt("+++"),
        CodegenOpts {
            cell_sign: CellSign::Signed,
            ..life_opts()
        },
    )
    .unwrap_err();
    assert_eq!(err, BfLifeCircuitError::SignedCellsUnsupported);
}

#[test]
fn life_circuit_rejects_step_budget_exceeded_programs() {
    let mut circuit = compile_to_life_circuit(&[BfIr::Diverge], life_opts()).unwrap();
    circuit.state.steps = 5_000_000;
    let err = circuit.step().unwrap_err();
    assert_eq!(err, BfLifeCircuitError::StepBudgetExceeded);
}

#[test]
fn life_circuit_compiles_diverge_as_stable_non_halting_state() {
    let mut circuit = compile_to_life_circuit(&[BfIr::Diverge], life_opts()).unwrap();
    for _ in 0..8 {
        assert!(circuit.step().unwrap());
    }
    assert_ne!(circuit.state.phase, CircuitPhase::Halted);
    assert_eq!(circuit.state.pc, 0);
}

#[test]
fn life_circuit_compiles_optimized_distribute_ir() {
    let mut circuit = compile_to_life_circuit(
        &[BfIr::Add(3), BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
        }, BfIr::MovePtr(2), BfIr::Output],
        life_opts(),
    )
    .unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.outputs, vec![6]);
    assert_eq!(circuit.state.tape[0], 0);
    assert_eq!(circuit.state.tape[1], 3);
    assert_eq!(circuit.state.tape[2], 6);
}

#[test]
fn life_circuit_macro_library_exposes_required_templates() {
    let templates = life_macro_templates();
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::Clock));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::SplitterMerger));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::StateLatch));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::HeadTokenMover));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::BitIncrement));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::BitDecrement));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::ZeroDetector));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::OutputBitSeedOne));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::OutputBitSeedZero));
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::OutputBitTransducer));
}

#[test]
fn life_circuit_macro_template_has_anchor_bounds_and_ports() {
    let zero = life_macro_template(LifeMacroKind::ZeroDetector);
    assert_eq!(zero.anchor, (0, 0));
    assert!(zero.bounds.0 > 0 && zero.bounds.1 > 0);
    assert!(zero.ports.iter().any(|port| port.name == "zero"));
}

#[test]
fn life_circuit_macro_rotation_and_instantiation_are_stable() {
    assert_eq!(transform_cell((2, 1), LifeMacroOrientation::R90), (-1, 2));
    let cells = instantiate_macro_cells(&crate::bf::LifeMacroInstance {
        id: 0,
        kind: LifeMacroKind::Clock,
        name: "clock",
        origin: (10, 20),
        orientation: LifeMacroOrientation::R180,
    });
    assert!(cells.contains(&(10, 20)));
}

#[test]
fn life_circuit_contains_explicit_machine_components() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+++[->++<]>.<."), life_opts()).unwrap();
    assert!(circuit
        .macro_instances
        .iter()
        .any(|instance| instance.kind == LifeMacroKind::Clock));
    assert!(circuit
        .macro_instances
        .iter()
        .any(|instance| instance.kind == LifeMacroKind::HeadTokenMover));
    assert!(circuit
        .macro_instances
        .iter()
        .any(|instance| instance.kind == LifeMacroKind::ZeroDetector));
    assert!(circuit
        .routed_signals
        .iter()
        .any(|signal| signal.contains("distribute")));
    assert_eq!(circuit.output_row_settle_generations(), 1);
}

#[test]
fn life_circuit_end_to_end_matches_unsigned_interpreter_for_small_programs() {
    for src in ["+++", ">+<", "[-]", "+++[->++<]>.<."] {
        let ir = parse_and_opt(src);
        let (expected_tape, expected_ptr) =
            interpret_unsigned_for_tests(&ir, life_opts().cell_bits).unwrap();
        let mut circuit = compile_to_life_circuit(&ir, life_opts()).unwrap();
        if !ir.iter().any(|node| matches!(node, BfIr::Diverge)) {
            circuit.run_to_completion().unwrap();
            assert_eq!(circuit.state.head, expected_ptr, "head mismatch for {src}");
            let expected_u64 = expected_tape
                .iter()
                .map(|&value| value as u64)
                .collect::<Vec<_>>();
            assert_eq!(
                &circuit.state.tape[..8],
                &expected_u64[..8],
                "tape mismatch for {src}"
            );
        }
    }
}

#[test]
fn life_circuit_output_region_uses_still_lifes_for_one_bits_and_is_simulator_stable() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+."), life_opts()).unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.output_latch, Some(1));
    assert_eq!(circuit.state.outputs, vec![1]);

    let grid = circuit.to_initial_grid();
    let (min_x, min_y, max_x, max_y) = circuit.output_row_bounds(0).unwrap();
    let initial_output = crop_region(&grid, min_x, min_y, max_x, max_y);
    assert_ne!(
        initial_output,
        step_grid(&initial_output),
        "initial output seed should not already be a stable still life row"
    );

    let settled_grid = circuit.output_grid_after_settle();
    let output = crop_region(&settled_grid, min_x, min_y, max_x, max_y);
    let next = step_grid(&output);
    assert_eq!(output, next);

    let cells = output.live_cells();
    let base_x = circuit.output_region_base_x();
    assert!(cells.contains(&(base_x, min_y)));
    assert!(cells.contains(&(base_x + 1, min_y)));
    assert!(cells.contains(&(base_x, min_y + 1)));
    assert!(cells.contains(&(base_x + 1, min_y + 1)));
}

#[test]
fn life_circuit_output_region_uses_still_lifes_for_zero_bits_and_is_simulator_stable() {
    let mut circuit =
        compile_to_life_circuit(&[BfIr::Clear, BfIr::Output], life_opts()).unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.output_latch, Some(0));
    assert_eq!(circuit.state.outputs, vec![0]);

    let grid = circuit.to_initial_grid();
    let (min_x, min_y, max_x, max_y) = circuit.output_row_bounds(0).unwrap();
    let initial_output = crop_region(&grid, min_x, min_y, max_x, max_y);
    assert_ne!(
        initial_output,
        step_grid(&initial_output),
        "initial output seed should not already be a stable still life row"
    );

    let settled_grid = circuit.output_grid_after_settle();
    let output = crop_region(&settled_grid, min_x, min_y, max_x, max_y);
    let next = step_grid(&output);
    assert_eq!(output, next);

    let cells = output.live_cells();
    let base_x = circuit.output_region_base_x();
    assert!(cells.contains(&(base_x + 1, min_y)));
    assert!(cells.contains(&(base_x + 2, min_y)));
    assert!(cells.contains(&(base_x, min_y + 1)));
    assert!(cells.contains(&(base_x + 3, min_y + 1)));
    assert!(cells.contains(&(base_x + 1, min_y + 2)));
    assert!(cells.contains(&(base_x + 2, min_y + 2)));
}

#[test]
fn life_circuit_emits_a_still_life_row_for_each_output_value_in_binary() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+.>++.>+++."), life_opts()).unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.outputs, vec![1, 2, 3]);

    let grid = circuit.output_grid_after_settle();
    for (row, &value) in circuit.state.outputs.iter().enumerate() {
        let (min_x, min_y, max_x, max_y) = circuit.output_row_bounds(row).unwrap();
        let output = crop_region(&grid, min_x, min_y, max_x, max_y);
        assert_eq!(output, step_grid(&output), "output row {row} must be stable");

        let cells = output.live_cells();
        let base_x = circuit.output_region_base_x();
        if (value & 1) != 0 {
            assert!(cells.contains(&(base_x, min_y)));
            assert!(cells.contains(&(base_x + 1, min_y)));
            assert!(cells.contains(&(base_x, min_y + 1)));
            assert!(cells.contains(&(base_x + 1, min_y + 1)));
        } else {
            assert!(cells.contains(&(base_x + 1, min_y)));
            assert!(cells.contains(&(base_x + 2, min_y)));
            assert!(cells.contains(&(base_x, min_y + 1)));
            assert!(cells.contains(&(base_x + 3, min_y + 1)));
            assert!(cells.contains(&(base_x + 1, min_y + 2)));
            assert!(cells.contains(&(base_x + 2, min_y + 2)));
        }
    }
}

#[test]
fn life_circuit_32bit_output_row_contains_32_stable_still_lifes_matching_bits() {
    let opts = CodegenOpts {
        cell_bits: 32,
        ..life_opts()
    };
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+."), opts).unwrap();
    circuit.run_to_completion().unwrap();
    assert_eq!(circuit.state.outputs, vec![1]);

    let grid = circuit.output_grid_after_settle();
    let (min_x, min_y, max_x, max_y) = circuit.output_row_bounds(0).unwrap();
    let output = crop_region(&grid, min_x, min_y, max_x, max_y);
    assert_eq!(output, step_grid(&output));

    let mut components = connected_components(&output);
    components.sort_by_key(|cells| cells.iter().map(|(x, _)| *x).min().unwrap_or(i64::MIN));
    assert_eq!(components.len(), 32, "expected one still life per 32-bit output cell");

    for (bit, component) in components.iter().enumerate() {
        let component_grid = BitGrid::from_cells(component);
        assert_eq!(component_grid, step_grid(&component_grid), "bit {bit} component must be stable");
        let expected_size = if bit == 0 { 4 } else { 6 };
        assert_eq!(
            component.len(),
            expected_size,
            "bit {bit} should be encoded as {}",
            if bit == 0 { "block(1)" } else { "beehive(0)" }
        );
    }
}

#[test]
fn life_circuit_output_row_is_not_ready_before_settle_generations() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+."), life_opts()).unwrap();
    circuit.run_to_completion().unwrap();

    let initial = circuit.to_initial_grid();
    let premature = step_n(
        &initial,
        circuit.output_row_settle_generations().saturating_sub(1),
    );
    let settled = circuit.output_grid_after_settle();
    let bounds = circuit.output_row_bounds(0).unwrap();
    let premature_output = crop_region(&premature, bounds.0, bounds.1, bounds.2, bounds.3);
    let settled_output = crop_region(&settled, bounds.0, bounds.1, bounds.2, bounds.3);

    assert_ne!(
        premature_output, settled_output,
        "output row should not claim to be ready before the settle interval"
    );
}

#[test]
fn life_circuit_file_can_be_saved_imported_and_stepped_in_simple_life() {
    let opts = CodegenOpts {
        cell_bits: 32,
        ..life_opts()
    };
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+."), opts).unwrap();
    circuit.run_to_completion().unwrap();

    let serialized = serialize_life_circuit(&circuit);
    let path = unique_temp_path("bf_life_circuit", "life");
    fs::write(&path, serialized).unwrap();

    let imported = pattern_from_file(path.to_str().unwrap()).expect("circuit file should import");
    fs::remove_file(&path).unwrap();

    assert_eq!(imported, circuit.to_initial_grid());

    let settled = step_n(&imported, circuit.output_row_settle_generations());
    let (min_x, min_y, max_x, max_y) = circuit.output_row_bounds(0).unwrap();
    let output = crop_region(&settled, min_x, min_y, max_x, max_y);
    assert_eq!(output, step_grid(&output));

    let components = connected_components(&output);
    assert_eq!(components.len(), 32);
}

#[test]
fn life_circuit_hashlife_file_can_be_saved_imported_and_stepped_in_simple_life() {
    let opts = CodegenOpts {
        cell_bits: 32,
        ..life_opts()
    };
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+."), opts).unwrap();
    circuit.run_to_completion().unwrap();

    let serialized = serialize_life_circuit_hashlife(&circuit);
    let path = unique_temp_path("bf_life_circuit_hashlife", "snapshot");
    fs::write(&path, serialized).unwrap();

    let imported = pattern_from_file(path.to_str().unwrap()).expect("hashlife circuit file should import");
    fs::remove_file(&path).unwrap();

    assert_eq!(imported, circuit.to_initial_grid());

    let settled = crate::hashlife::HashLifeEngine::default()
        .advance(&imported, circuit.output_row_settle_generations());
    let (min_x, min_y, max_x, max_y) = circuit.output_row_bounds(0).unwrap();
    let output = crop_region(&settled, min_x, min_y, max_x, max_y);
    assert_eq!(output, step_grid(&output));

    let components = connected_components(&output);
    assert_eq!(components.len(), 32);
}
