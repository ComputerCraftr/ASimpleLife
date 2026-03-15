use super::*;
use crate::bf::{
    BfLifeCircuitError, LifeMacroKind, ShiftDir, compile_life_scaffold as compile_to_life_circuit,
};
use crate::{RequiredErrorExt, RequiredExt};

#[test]
fn life_reference_runtime_wraps_wide_richer_arithmetic_without_overflow() {
    let program = vec![
        BfIr::Add(1),
        BfIr::Shift {
            src: 0,
            dst: 0,
            amount: 62,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Add(1),
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(-1),
        BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 3,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: true,
        },
        BfIr::Affine {
            src: 0,
            dst: 4,
            coeff: 3,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Shift {
            src: 0,
            dst: 5,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MovePtr(2),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 63,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let mut circuit = compile_to_life_circuit(&program, opts).or_invariant("required value");

    circuit
        .reference_run_to_completion()
        .or_invariant("required value");

    assert_eq!(
        circuit.state.outputs,
        vec![1, (1_u64 << 62) + 3, (1_u64 << 62) + 3, 2]
    );
    let stride = tape_stride_between_latches(&circuit, 0, 1);
    assert!(
        stride >= 260,
        "63-bit tape cells must not overlap, observed latch stride={stride}"
    );
}

#[test]
fn life_reference_scan_preserves_nonzero_zero_congruence_orbit() {
    let program = vec![
        BfIr::Add(1),
        BfIr::Scan {
            stride: crate::bf::BfOffset::MIN,
        },
        BfIr::Output,
    ];
    let mut circuit = compile_to_life_circuit(&program, life_opts()).or_invariant("required value");

    assert_eq!(
        circuit
            .reference_run_to_completion()
            .error_or_invariant("expected error"),
        BfLifeCircuitError::StepBudgetExceeded
    );
    assert!(circuit.state.outputs.is_empty());
}

#[test]
fn life_reference_scan_wraps_adjacent_extreme_stride_without_overflow() {
    let program = vec![
        BfIr::Add(1),
        BfIr::Scan {
            stride: crate::bf::BfOffset::MIN + 1,
        },
        BfIr::Output,
    ];
    let mut circuit = compile_to_life_circuit(&program, life_opts()).or_invariant("required value");

    circuit
        .reference_run_to_completion()
        .or_invariant("required value");

    assert_eq!(circuit.state.outputs, vec![0]);
}

fn tape_stride_between_latches(
    circuit: &crate::bf::life_backend::ReferenceLifeScaffold,
    first_cell: usize,
    second_cell: usize,
) -> i64 {
    let latch_origins = circuit
        .macro_instances()
        .iter()
        .filter(|instance| instance.kind == LifeMacroKind::HeadTokenMover)
        .map(|instance| instance.origin.0)
        .collect::<Vec<_>>();
    latch_origins[second_cell] - latch_origins[first_cell]
}
