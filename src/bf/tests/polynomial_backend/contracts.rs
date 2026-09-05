use super::*;

fn assert_full_state_matches(program: &[BfIr], opts: CodegenOpts, cc_args: &[&str]) {
    let (expected_tape, expected_pointer) =
        interpret_for_tests(program, opts).or_invariant("contract fixture terminates");
    let (plain, _) = run_observed(program, opts, TestCBackend::Plain, cc_args);
    let (super_state, _) = run_observed(program, opts, TestCBackend::Super, cc_args);

    assert_eq!(
        plain.pointer, expected_pointer,
        "plain pointer opts={opts:?}"
    );
    assert_eq!(plain.tape, expected_tape, "plain tape opts={opts:?}");
    assert_eq!(super_state, plain, "super/plain full state opts={opts:?}");
}

fn evolving_loop_body(guard_delta: i32) -> Vec<BfIr> {
    vec![
        BfIr::Add(guard_delta),
        BfIr::Affine {
            src: 1,
            dst: 2,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::MovePtr(1),
        BfIr::Add(1),
        BfIr::MovePtr(-1),
    ]
}

fn variable_trip_program() -> Vec<BfIr> {
    let mut program = Vec::new();
    for trips in [0, 1, 23, 85, 255] {
        program.extend([
            BfIr::Add(trips),
            BfIr::MovePtr(1),
            BfIr::Add(3),
            BfIr::MovePtr(1),
            BfIr::Add(7),
            BfIr::MovePtr(-2),
            BfIr::Loop(evolving_loop_body(-1)),
            BfIr::MovePtr(1),
            BfIr::Output,
            BfIr::MovePtr(1),
            BfIr::Output,
            BfIr::MovePtr(2),
        ]);
    }
    program
}

#[test]
fn variable_trip_bits_use_entry_guard_and_each_power_uses_committed_state() {
    let program = variable_trip_program();
    let opts = numeric_opts(8, CellSign::Unsigned);
    let (expected_tape, expected_pointer) =
        interpret_for_tests(&program, opts).or_invariant("variable-trip fixture terminates");
    let (plain, _) = run_observed(&program, opts, TestCBackend::Plain, &[]);
    let (super_state, super_output) = run_observed(&program, opts, TestCBackend::Super, &[]);

    assert_eq!(plain.pointer, expected_pointer);
    assert_eq!(plain.tape, expected_tape);
    let expected_output = [0_u32, 1, 23, 85, 255]
        .into_iter()
        .flat_map(|n| {
            [
                (3 + n) % 256,
                (7 + 3 * n + n * n.saturating_sub(1) / 2) % 256,
            ]
        })
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        plain.payload, expected_output,
        "independent closed-form output oracle"
    );
    assert_eq!(super_state, plain);
    assert!(
        memo_stat(&super_output, "symbolic power hits:") >= 17,
        "nonzero bit-decomposed fixtures did not execute every selected power:\n{super_output}"
    );
}

#[test]
fn variable_trip_native_and_forced_scalar_match_at_o3() {
    let program = variable_trip_program();
    let opts = numeric_opts(8, CellSign::Unsigned);
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let (native, native_output) = run_observed(&program, opts, backend, &["-O3"]);
        let (scalar, scalar_output) = run_observed(
            &program,
            opts,
            backend,
            &["-O3", "-DBF_FORCE_SCALAR_KERNEL"],
        );
        assert_eq!(native, scalar, "backend={backend:?}");
        if matches!(backend, TestCBackend::Super) {
            assert!(memo_stat(&native_output, "symbolic power hits:") > 0);
            assert!(memo_stat(&scalar_output, "symbolic power hits:") > 0);
        }
    }
}

#[test]
fn signed_negative_guard_uses_positive_delta_for_exact_trip_count() {
    let program = vec![
        BfIr::Add(-5),
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(1),
        BfIr::Add(7),
        BfIr::MovePtr(-2),
        BfIr::Loop(evolving_loop_body(1)),
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let opts = numeric_opts(8, CellSign::Signed);
    let (expected_tape, expected_pointer) =
        interpret_for_tests(&program, opts).or_invariant("signed guard fixture terminates");
    let (plain, _) = run_observed(&program, opts, TestCBackend::Plain, &[]);
    let (super_state, super_output) = run_observed(&program, opts, TestCBackend::Super, &[]);

    assert_eq!(plain.pointer, expected_pointer);
    assert_eq!(plain.tape, expected_tape);
    assert_eq!(plain.payload, "8\n32", "signed five-iteration output");
    assert_eq!(super_state, plain);
    assert!(memo_stat(&super_output, "symbolic power hits:") > 0);
}

fn factoring_program(cell_bits: u32) -> Vec<BfIr> {
    let coefficients = [0, 2, 4, 1_i32 << (cell_bits - 1), 3];
    let mut program = vec![
        BfIr::Add(3),
        BfIr::MovePtr(1),
        BfIr::Add(5),
        BfIr::MovePtr(1),
        BfIr::Add(7),
        BfIr::MovePtr(-2),
    ];
    for (index, coefficient) in coefficients.into_iter().enumerate() {
        let temp = i64::try_from(index).or_invariant("small factoring index") + 3;
        let dst = temp + 5;
        program.extend([
            BfIr::Affine {
                src: 1,
                dst: temp,
                coeff: coefficient,
                preserve_src: true,
                set_dst: true,
            },
            BfIr::MulAdd {
                lhs: 0,
                rhs: temp,
                dst,
                preserve_lhs: true,
                preserve_rhs: true,
                set_dst: true,
            },
            BfIr::MulAdd {
                lhs: 0,
                rhs: 2,
                dst,
                preserve_lhs: true,
                preserve_rhs: true,
                set_dst: false,
            },
        ]);
        let filler_start = 20 + i64::try_from(index).or_invariant("small factoring index") * 6;
        for filler in filler_start..filler_start + 6 {
            program.push(BfIr::Affine {
                src: 0,
                dst: filler,
                coeff: 1,
                preserve_src: true,
                set_dst: true,
            });
        }
        program.extend([BfIr::MovePtr(1), BfIr::MovePtr(-1)]);
    }
    program.push(BfIr::MovePtr(8));
    for index in 0..coefficients.len() {
        program.push(BfIr::Output);
        if index + 1 != coefficients.len() {
            program.push(BfIr::MovePtr(1));
        }
    }
    program
}

#[test]
fn factoring_is_distributive_without_dividing_ring_coefficients() {
    for cell_sign in [CellSign::Unsigned, CellSign::Signed] {
        for bits in [4, 5] {
            let program = factoring_program(bits);
            let opts = numeric_opts(bits, cell_sign);
            let mut emitted_factoring = false;
            for backend in [TestCBackend::Plain, TestCBackend::Super] {
                let source = emit_backend_ir(&program, opts, backend);
                emitted_factoring |=
                    CSource::parse(&source).has_identifier_prefix("bf_poly_factor_value_");
            }
            assert!(
                emitted_factoring,
                "fixture missed factored emission: bits={bits} sign={cell_sign:?}"
            );
            assert_full_state_matches(&program, opts, &[]);
        }
    }
}

#[test]
fn guard_address_must_be_stable_and_free_of_wrapped_writes() {
    let opts = numeric_opts(8, CellSign::Unsigned);
    let moving_guard = vec![BfIr::Loop(vec![BfIr::Add(-1), BfIr::MovePtr(1)])];
    let wrapped_guard_write = vec![BfIr::Loop(vec![
        BfIr::Add(-1),
        BfIr::Affine {
            src: 1,
            dst: 30_000,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        },
    ])];

    for (label, program) in [
        ("moving guard", moving_guard),
        ("wrapped guard write", wrapped_guard_write),
    ] {
        let plans = crate::bf::tests::planner_shape::program_plans(&program, opts);
        assert!(
            !plans.iter().any(|plan| matches!(
                plan,
                crate::bf::c_super_backend::ExecPlan::ExactPoweredLoopMemo { .. }
            )),
            "unproven {label} shape reached powered execution"
        );
    }
}

#[test]
fn antipode_offsets_compose_as_one_physical_cell() {
    let program = vec![
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(-1),
        BfIr::Square {
            src: 1,
            dst: 15_000,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Affine {
            src: -15_000,
            dst: 2,
            coeff: 2,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MovePtr(2),
        BfIr::Output,
        BfIr::MovePtr(-2),
    ];
    let opts = numeric_opts(8, CellSign::Unsigned);
    let super_source = emit_backend_ir(&program, opts, TestCBackend::Super);
    assert!(
        CSource::parse(&super_source).has_identifier_prefix("bf_poly_src_"),
        "antipode fixture did not exercise symbolic composition"
    );
    assert_full_state_matches(&program, opts, &[]);
}

#[test]
fn high_wrap_powered_execution_matches_full_state() {
    assert_high_wrap_powered_execution(&["-O3"]);
}

#[test]
#[cfg(not(target_os = "windows"))]
fn high_wrap_powered_execution_is_ubsan_clean() {
    assert_high_wrap_powered_execution(&[
        "-O3",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
    ]);
}

fn assert_high_wrap_powered_execution(cc_args: &[&str]) {
    let program = vec![
        BfIr::Add(23),
        BfIr::MovePtr(1),
        BfIr::Add(i32::MAX),
        BfIr::MovePtr(1),
        BfIr::Add(i32::MAX),
        BfIr::MovePtr(-2),
        BfIr::Square {
            src: 1,
            dst: 1,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::Affine {
                src: 1,
                dst: 1,
                coeff: 3,
                preserve_src: true,
                set_dst: true,
            },
            BfIr::Affine {
                src: 1,
                dst: 2,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            },
        ]),
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let opts = numeric_opts(63, CellSign::Signed);
    assert_full_state_matches(&program, opts, cc_args);
}

#[test]
fn huge_trip_count_with_tiny_semantic_fuel_preserves_observable_prefix() {
    let program = vec![
        BfIr::Output,
        BfIr::Add(-1),
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(-1),
        BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::MovePtr(1),
            BfIr::Add(1),
            BfIr::MovePtr(-1),
        ]),
        BfIr::Output,
    ];
    let opts = numeric_opts(16, CellSign::Unsigned);
    let fuel = "-DBF_TEST_SEMANTIC_FUEL_LIMIT=12";
    let plain = generated_c_outcome(compile_and_run_c_source_capture(
        &emit_backend_ir(&program, opts, TestCBackend::Plain),
        &[fuel],
    ));
    let super_out = generated_c_outcome(compile_and_run_c_source_capture(
        &emit_backend_ir(&program, opts, TestCBackend::Super),
        &[fuel],
    ));

    assert!(matches!(plain, GeneratedCOutcome::WorkLimitReached { .. }));
    assert_eq!(super_out, plain);
}

#[test]
fn emitter_width_contract_rejects_instead_of_clamping() {
    let program = [BfIr::Output];
    for bits in [0, 1, 63] {
        let opts = numeric_opts(bits, CellSign::Unsigned);
        for backend in [TestCBackend::Plain, TestCBackend::Super] {
            let source = emit_backend_ir(&program, opts, backend);
            assert!(
                CSource::parse(&source).unconditional_errors().is_empty(),
                "bits={bits} backend={backend:?}"
            );
            assert!(
                CSource::parse(&source)
                    .define_values("BF_TEMPLATE_CELL_BITS")
                    .contains(&bits.to_string().as_str())
            );
        }
    }

    let invalid = [
        (64, None, None, "cell width 64"),
        (u32::MAX, None, None, "cell width 4294967295"),
        (63, Some(64), None, "input width 64"),
        (63, None, Some(u32::MAX), "output width 4294967295"),
    ];
    for (cell_bits, input_bits, output_bits, diagnostic) in invalid {
        let opts = CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits,
            input_bits,
            output_bits,
            cell_sign: CellSign::Unsigned,
        };
        assert!(
            interpret_for_tests(&program, opts).is_err(),
            "interpreter accepted invalid options: {opts:?}"
        );
        for backend in [TestCBackend::Plain, TestCBackend::Super] {
            let source = emit_backend_ir(&program, opts, backend);
            let syntax = CSource::parse(&source);
            let errors = syntax.unconditional_errors();
            assert_eq!(
                errors.len(),
                1,
                "expected one typed compiler rejection: {source}"
            );
            assert!(
                errors[0].starts_with("unsupported BF ") && errors[0].contains(diagnostic),
                "diagnostic={diagnostic:?} backend={backend:?} source={source:?}"
            );
        }
    }
}
