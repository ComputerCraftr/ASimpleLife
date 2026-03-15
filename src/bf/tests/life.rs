use super::*;
use crate::bf::ShiftDir;
use crate::bf::life_backend::{
    BfLifeCircuitError, CircuitPhase, compile_life_scaffold as compile_to_life_circuit,
    serialize_life_scaffold as serialize_life_circuit,
    serialize_life_scaffold_hashlife as serialize_life_circuit_hashlife,
};
use crate::bf::life_macro_library::{
    LifeMacroKind, LifeMacroOrientation, instantiate_macro_cells, life_macro_template,
    life_macro_templates, transform_cell,
};
use crate::bf::{BfIr, PhysicalBfInstr, RailGroup, expand_distribute_to_primitive};
use crate::generators::pattern_from_file;
use crate::life::step_grid;
use crate::persistence::{
    HASHLIFE_SNAPSHOT_MAGIC, LIFE_GRID_MAGIC, deserialize_grid, deserialize_life_grid,
};
use crate::{RequiredErrorExt, RequiredExt};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

mod advanced;

fn unique_temp_path(stem: &str, ext: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .or_invariant("required value")
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
fn life_scaffold_builds_basic_loop_metadata() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+++[->+>++<<]>>."), life_opts())
        .or_invariant("required value");
    assert_eq!(circuit.tape_len, 64);
    assert_eq!(circuit.cell_bits, 8);
    assert_eq!(circuit.state.phase, CircuitPhase::Fetch);
    assert_eq!(circuit.program.last(), Some(&PhysicalBfInstr::Halt));
    assert!(
        circuit
            .program
            .iter()
            .any(|instr| matches!(instr, PhysicalBfInstr::JumpIfZero(_)))
    );
    assert!(
        !circuit
            .program
            .iter()
            .any(|instr| matches!(instr, PhysicalBfInstr::Diverge))
    );
}

#[test]
fn life_scaffold_reference_executes_add_clear_move_loop_and_output() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+++[->++<]>.<."), life_opts())
        .or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.phase, CircuitPhase::Halted);
    assert_eq!(circuit.state.outputs, vec![6, 0]);
    assert_eq!(circuit.state.tape[0], 0);
    assert_eq!(circuit.state.tape[1], 6);
}

#[test]
fn life_scaffold_reference_preserves_explicit_clear() {
    let mut circuit =
        compile_to_life_circuit(&[BfIr::Add(5), BfIr::Clear, BfIr::Output], life_opts())
            .or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, vec![0]);
}

#[test]
fn life_scaffold_reference_uses_explicit_host_phases() {
    let mut circuit =
        compile_to_life_circuit(&parse_and_opt("+."), life_opts()).or_invariant("required value");
    assert_eq!(circuit.state.phase, CircuitPhase::Fetch);
    assert!(circuit.step().or_invariant("required value"));
    assert_eq!(circuit.state.phase, CircuitPhase::Decode);
    assert!(circuit.step().or_invariant("required value"));
    assert_eq!(circuit.state.phase, CircuitPhase::Evaluate);
    assert!(circuit.step().or_invariant("required value"));
    assert_eq!(circuit.state.phase, CircuitPhase::Commit);
    assert!(circuit.step().or_invariant("required value"));
    assert_eq!(circuit.state.phase, CircuitPhase::Fetch);
}

#[test]
fn life_scaffold_serializer_uses_standard_life_grid_format() {
    let circuit =
        compile_to_life_circuit(&parse_and_opt("+."), life_opts()).or_invariant("required value");
    let serialized = serialize_life_circuit(&circuit);
    assert!(serialized.starts_with(LIFE_GRID_MAGIC));
    assert_eq!(
        deserialize_life_grid(&serialized).or_invariant("required value"),
        circuit.compiled_grid()
    );
}

#[test]
fn life_scaffold_layout_can_be_serialized_as_hashlife_snapshot() {
    let circuit =
        compile_to_life_circuit(&parse_and_opt("+."), life_opts()).or_invariant("required value");
    let serialized = serialize_life_circuit_hashlife(&circuit);
    assert!(serialized.starts_with(HASHLIFE_SNAPSHOT_MAGIC));
    assert_eq!(
        deserialize_grid(&serialized).or_invariant("required value"),
        circuit.compiled_grid()
    );
}

#[test]
fn life_scaffold_serialization_is_independent_of_reference_state() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+.>++."), life_opts())
        .or_invariant("required value");
    let life_before = serialize_life_circuit(&circuit);
    let snapshot_before = serialize_life_circuit_hashlife(&circuit);

    circuit
        .reference_run_to_completion()
        .or_invariant("reference model should terminate");
    assert_eq!(circuit.state.outputs, vec![1, 2]);
    assert_eq!(serialize_life_circuit(&circuit), life_before);
    assert_eq!(serialize_life_circuit_hashlife(&circuit), snapshot_before);
}

#[test]
fn life_scaffold_rejects_input_ops() {
    let err = compile_to_life_circuit(&parse_only(",."), life_opts())
        .error_or_invariant("expected error");
    assert_eq!(err, BfLifeCircuitError::InputUnsupported);
}

#[test]
fn life_scaffold_rejects_nested_input_ops() {
    let err = compile_to_life_circuit(&parse_only("[,+.]"), life_opts())
        .error_or_invariant("expected error");
    assert_eq!(err, BfLifeCircuitError::InputUnsupported);
}

#[test]
fn life_scaffold_rejects_signed_cells() {
    let err = compile_to_life_circuit(
        &parse_and_opt("+++"),
        CodegenOpts {
            cell_sign: CellSign::Signed,
            ..life_opts()
        },
    )
    .error_or_invariant("expected error");
    assert_eq!(err, BfLifeCircuitError::SignedCellsUnsupported);
}

#[test]
fn life_scaffold_reference_rejects_step_budget_exceeded_programs() {
    let mut circuit =
        compile_to_life_circuit(&[BfIr::Diverge], life_opts()).or_invariant("required value");
    circuit.state.steps = 5_000_000;
    let err = circuit.step().error_or_invariant("expected error");
    assert_eq!(err, BfLifeCircuitError::StepBudgetExceeded);
}

#[test]
fn life_scaffold_bounded_prefix_does_not_claim_halt_for_diverge_instruction() {
    let mut circuit =
        compile_to_life_circuit(&[BfIr::Diverge], life_opts()).or_invariant("required value");
    for _ in 0..8 {
        assert!(circuit.step().or_invariant("required value"));
    }
    assert_ne!(circuit.state.phase, CircuitPhase::Halted);
    assert_eq!(circuit.state.pc, 0);
}

#[test]
fn life_scaffold_lowers_optimized_distribute_ir() {
    let optimized = vec![
        BfIr::Add(3),
        BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: false,
        },
        BfIr::MovePtr(2),
        BfIr::Output,
    ];
    let expanded = expand_distribute_to_primitive(&optimized);
    assert!(expanded.iter().any(|node| matches!(node, BfIr::Loop(_))));
    let mut circuit = compile_to_life_circuit(
        &[
            BfIr::Add(3),
            BfIr::Distribute {
                targets: vec![(1, 1), (2, 2)],
                preserve_src: false,
            },
            BfIr::MovePtr(2),
            BfIr::Output,
        ],
        life_opts(),
    )
    .or_invariant("required value");
    assert!(
        !circuit
            .program
            .iter()
            .any(|instr| matches!(instr, PhysicalBfInstr::Diverge))
    );
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, vec![6]);
    assert_eq!(circuit.state.tape[0], 0);
    assert_eq!(circuit.state.tape[1], 3);
    assert_eq!(circuit.state.tape[2], 6);
}

#[test]
fn life_scaffold_metadata_lists_placeholder_component_kinds() {
    let templates = life_macro_templates();
    assert!(templates.iter().any(|t| t.kind == LifeMacroKind::Clock));
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::SplitterMerger)
    );
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::StateLatch)
    );
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::HeadTokenMover)
    );
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::BitIncrement)
    );
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::BitDecrement)
    );
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::ZeroDetector)
    );
    assert!(
        templates
            .iter()
            .any(|t| t.kind == LifeMacroKind::OutputBitTransducer)
    );
    assert!(
        life_macro_template(LifeMacroKind::OutputBitTransducer)
            .live_cells
            .is_empty(),
        "an unverified placeholder transducer must not masquerade as executable Life logic"
    );
}

#[test]
fn life_scaffold_macro_metadata_has_anchor_bounds_and_ports() {
    let zero = life_macro_template(LifeMacroKind::ZeroDetector);
    assert_eq!(zero.anchor, (0, 0));
    assert!(zero.bounds.0 > 0 && zero.bounds.1 > 0);
    assert!(zero.ports.iter().any(|port| port.name == "zero"));
}

#[test]
fn life_scaffold_macro_metadata_rotation_and_instantiation_are_stable() {
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
fn life_scaffold_layout_contains_machine_component_metadata() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+++[->+>++<<]>>."), life_opts())
        .or_invariant("required value");
    assert!(
        circuit
            .macro_instances()
            .iter()
            .any(|instance| instance.kind == LifeMacroKind::Clock)
    );
    assert!(
        circuit
            .macro_instances()
            .iter()
            .any(|instance| instance.kind == LifeMacroKind::HeadTokenMover)
    );
    assert!(
        circuit
            .macro_instances()
            .iter()
            .any(|instance| instance.kind == LifeMacroKind::ZeroDetector)
    );
    assert!(
        circuit
            .routed_rails()
            .iter()
            .any(|rail| rail.name.contains("jump_zero"))
    );
    assert!(
        circuit
            .routed_rails()
            .iter()
            .any(|rail| rail.group == RailGroup::OutputTransducer)
    );
    assert!(
        circuit
            .macro_timing_specs()
            .iter()
            .any(|spec| spec.kind == LifeMacroKind::OutputBitTransducer)
    );
    assert_eq!(
        circuit.placed_machine().output_row_settle_generations,
        circuit.output_row_settle_generations()
    );
    assert_eq!(circuit.output_row_settle_generations(), 1);
}

#[test]
fn life_scaffold_placement_and_routing_metadata_are_deterministic() {
    let a = compile_to_life_circuit(&parse_and_opt("+++[->+>++<<]>>."), life_opts())
        .or_invariant("required value");
    let b = compile_to_life_circuit(&parse_and_opt("+++[->+>++<<]>>."), life_opts())
        .or_invariant("required value");
    assert_eq!(a.program, b.program);
    assert_eq!(a.placed_machine(), b.placed_machine());
    assert_eq!(a.compiled_grid(), b.compiled_grid());
    assert_eq!(a.debug_routed_signals(), b.debug_routed_signals());
}

#[test]
fn life_scaffold_routed_rail_metadata_covers_required_groups() {
    let circuit = compile_to_life_circuit(&parse_and_opt("+++[->+>++<<]>>."), life_opts())
        .or_invariant("required value");
    let rails = circuit.routed_rails();
    assert!(rails.iter().any(|rail| rail.group == RailGroup::TapeData));
    assert!(rails.iter().any(|rail| rail.group == RailGroup::HeadMove));
    assert!(
        rails
            .iter()
            .any(|rail| rail.group == RailGroup::ZeroDetectBranch)
    );
    assert!(
        rails
            .iter()
            .any(|rail| rail.group == RailGroup::OutputTransducer)
    );
}

#[test]
fn life_scaffold_output_program_exposes_transducer_timing_metadata() {
    let circuit =
        compile_to_life_circuit(&parse_and_opt("+."), life_opts()).or_invariant("required value");
    assert!(
        circuit
            .macro_timing_specs()
            .iter()
            .any(|spec| spec.kind == LifeMacroKind::OutputBitTransducer
                && spec.settle_generations == circuit.output_row_settle_generations())
    );
}

#[test]
fn life_scaffold_reference_model_matches_unsigned_interpreter_for_small_programs() {
    for src in ["+++", ">+<", "[-]", "+++[->++<]>.<."] {
        let ir = parse_and_opt(src);
        let (expected_tape, expected_ptr) =
            interpret_unsigned_for_tests(&ir, life_opts().cell_bits).or_invariant("required value");
        let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
        if !ir.iter().any(|node| matches!(node, BfIr::Diverge)) {
            circuit
                .reference_run_to_completion()
                .or_invariant("required value");
            assert_eq!(circuit.state.head, expected_ptr, "head mismatch for {src}");
            let expected_u64 = expected_tape
                .iter()
                .map(|&value| value.cast_unsigned())
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
fn life_scaffold_layout_roundtrips_through_life_format_and_steps() {
    let opts = CodegenOpts {
        cell_bits: 32,
        ..life_opts()
    };
    let circuit =
        compile_to_life_circuit(&parse_and_opt("+."), opts).or_invariant("required value");

    let serialized = serialize_life_circuit(&circuit);
    let path = unique_temp_path("bf_life_circuit", "life");
    fs::write(&path, serialized).or_invariant("required value");

    let imported = pattern_from_file(path.to_str().or_invariant("required value"))
        .or_invariant("circuit file should import");
    fs::remove_file(&path).or_invariant("required value");

    assert_eq!(imported, circuit.compiled_grid());

    assert_eq!(step_grid(&imported), step_grid(&circuit.compiled_grid()));
}

#[test]
fn life_scaffold_layout_roundtrips_through_hashlife_format_and_matches_scalar_step() {
    let opts = CodegenOpts {
        cell_bits: 32,
        ..life_opts()
    };
    let circuit =
        compile_to_life_circuit(&parse_and_opt("+."), opts).or_invariant("required value");

    let serialized = serialize_life_circuit_hashlife(&circuit);
    let path = unique_temp_path("bf_life_circuit_hashlife", "snapshot");
    fs::write(&path, serialized).or_invariant("required value");

    let imported = pattern_from_file(path.to_str().or_invariant("required value"))
        .or_invariant("hashlife circuit file should import");
    fs::remove_file(&path).or_invariant("required value");

    assert_eq!(imported, circuit.compiled_grid());

    let scalar = step_grid(&imported);
    let hashlife = crate::hashlife::HashLifeEngine::default().advance(&imported, 1);
    assert_eq!(
        hashlife, scalar,
        "imported compiled grid evolved differently under HashLife"
    );
}

#[test]
fn life_scaffold_reference_runtime_matches_explicit_fixture_outputs() {
    for src in ["+.", "+++[->++<]>.<.", "+.>++.>+++."] {
        let ir = parse_and_opt(src);
        let expected_outputs: &[u64] = match src {
            "+." => &[1],
            "+++[->++<]>.<." => &[6, 0],
            "+.>++.>+++." => &[1, 2, 3],
            _ => crate::invariant_failure!("unlisted scaffold fixture {src}"),
        };

        let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
        circuit
            .reference_run_to_completion()
            .or_invariant("required value");
        assert_eq!(
            circuit.state.outputs, expected_outputs,
            "reference outputs mismatch for {src}"
        );
    }
}

#[test]
fn life_scaffold_reference_runtime_matches_explicit_richer_ir_outputs() {
    let ir = vec![
        BfIr::Add(3),
        BfIr::MovePtr(1),
        BfIr::Add(5),
        BfIr::MovePtr(1),
        BfIr::Add(2),
        BfIr::MovePtr(-2),
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 2,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::Shift {
            src: 1,
            dst: 2,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Shift {
            src: 2,
            dst: 3,
            amount: 2,
            dir: ShiftDir::Right,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Square {
            src: 0,
            dst: 4,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MulAdd {
            lhs: 3,
            rhs: 4,
            dst: 5,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: true,
        },
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let expected_outputs = vec![11, 22, 5, 9, 45];

    let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, expected_outputs);
}

#[test]
fn life_scaffold_rejects_noncanonical_aliased_muladd_ir() {
    let ir = vec![BfIr::MulAdd {
        lhs: 1,
        rhs: 1,
        dst: 2,
        preserve_lhs: true,
        preserve_rhs: false,
        set_dst: true,
    }];
    let err = compile_to_life_circuit(&ir, life_opts()).error_or_invariant("expected error");
    assert!(err.to_string().contains("canonical richer IR"));
    assert!(err.to_string().contains("Square"));
}

#[test]
fn life_scaffold_reference_executes_manual_canonical_square_ir() {
    let ir = vec![
        BfIr::Add(3),
        BfIr::Square {
            src: 0,
            dst: 0,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::MovePtr(1),
        BfIr::Add(4),
        BfIr::MovePtr(-1),
        BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 1,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        },
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let expected_outputs = vec![12, 52];
    let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, expected_outputs);
}

#[test]
fn life_scaffold_reference_executes_exponentiation_by_squaring_pattern() {
    let src = "++++[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<>>.";
    let ir = parse_and_opt(src);
    assert!(ir.iter().any(|node| matches!(node, BfIr::Square { .. })));

    let expected_outputs = vec![16];

    let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, expected_outputs);
}

#[test]
fn life_scaffold_reference_executes_non_unit_odd_affine_loop_pattern() {
    let mut circuit = compile_to_life_circuit(&parse_and_opt("+++[--->+++<]>."), life_opts())
        .or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, vec![3]);
}

#[test]
fn life_scaffold_reference_executes_scan_summary_pattern() {
    let ir = parse_and_opt("+>+>+<<[>].");
    assert!(
        ir.iter()
            .any(|node| matches!(node, BfIr::Scan { stride: 1 })),
        "scan fixture should reach the dedicated IR: {ir:?}"
    );

    let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, vec![0]);
}

#[test]
fn life_scaffold_reference_executes_temp_consuming_polynomial_sequence() {
    let ir = vec![
        BfIr::Add(5),
        BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::MovePtr(2),
        BfIr::Output,
    ];

    let expected_outputs = vec![25];

    let mut circuit = compile_to_life_circuit(&ir, life_opts()).or_invariant("required value");
    circuit
        .reference_run_to_completion()
        .or_invariant("required value");
    assert_eq!(circuit.state.outputs, expected_outputs);
}
