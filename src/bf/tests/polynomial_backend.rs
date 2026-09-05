use super::*;
use crate::bf::c_super_backend::summarize_c_region;

mod contracts;

const FINAL_STATE_SENTINEL: &str = "=== BF FINAL STATE ===";
const OUTPUT_OFFSETS: [usize; 6] = [3, 4, 5, 6, 7, 2];

#[derive(Debug, PartialEq, Eq)]
struct ObservedState {
    payload: String,
    pointer: usize,
    tape: Vec<i64>,
}

fn numeric_opts(bits: u32, cell_sign: CellSign) -> CodegenOpts {
    CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: bits,
        input_bits: None,
        output_bits: None,
        cell_sign,
    }
}

fn rich_degree_region() -> Vec<BfIr> {
    vec![
        BfIr::Square {
            src: 1,
            dst: 2,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MulAdd {
            lhs: 2,
            rhs: 1,
            dst: 3,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: true,
        },
        BfIr::Square {
            src: 2,
            dst: 4,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MulAdd {
            lhs: 3,
            rhs: 2,
            dst: 5,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: true,
        },
        BfIr::Square {
            src: 3,
            dst: 6,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::MulAdd {
            lhs: 4,
            rhs: 3,
            dst: 7,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: true,
        },
        BfIr::Square {
            src: 4,
            dst: 2,
            preserve_src: true,
            set_dst: true,
        },
    ]
}

fn adversarial_program() -> (Vec<BfIr>, Vec<BfIr>) {
    let rich = rich_degree_region();
    let mut body = vec![BfIr::Add(-1)];
    body.extend(rich.iter().cloned());
    let mut program = vec![
        BfIr::Add(5),
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(-1),
        BfIr::Loop(body),
        BfIr::MovePtr(3),
    ];
    for _ in 3..=7 {
        program.extend([BfIr::Output, BfIr::MovePtr(1)]);
    }
    program.extend([BfIr::MovePtr(-6), BfIr::Output, BfIr::MovePtr(-2)]);
    (program, rich)
}

fn instrument_final_state(source: &str, backend: TestCBackend) -> String {
    let dump = match backend {
        TestCBackend::Plain => {
            "printf(\"\\n=== BF FINAL STATE ===\\n%td\\n\", ptr); for (size_t bf_test_i = 0; bf_test_i < BF_TAPE_LEN; ++bf_test_i) printf(\"%lld%c\", (long long)tape[bf_test_i], bf_test_i + 1 == BF_TAPE_LEN ? '\\n' : ','); print_work_stats();"
        }
        TestCBackend::Super => {
            "printf(\"\\n=== BF FINAL STATE ===\\n%td\\n\", ptr); for (size_t bf_test_i = 0; bf_test_i < BF_TAPE_LEN; ++bf_test_i) printf(\"%lld%c\", (long long)tape[bf_test_i], bf_test_i + 1 == BF_TAPE_LEN ? '\\n' : ','); print_memo_stats();"
        }
    };
    CSource::parse(source)
        .replace_main_call(backend.stats_function(), dump)
        .or_invariant("unique final-state instrumentation point")
}

fn parse_observed_state(output: &str) -> ObservedState {
    let (payload, state) = output
        .split_once(FINAL_STATE_SENTINEL)
        .or_invariant("instrumented C emitted final state");
    let mut lines = state.trim_start().lines();
    let pointer = lines
        .next()
        .or_invariant("final pointer line")
        .parse::<usize>()
        .or_invariant("numeric final pointer");
    let tape = lines
        .next()
        .or_invariant("final tape line")
        .split(',')
        .map(|value| value.parse::<i64>().or_invariant("numeric final tape cell"))
        .collect::<Vec<_>>();
    assert_eq!(tape.len(), crate::bf::BF_C_TAPE_LEN);
    ObservedState {
        payload: payload.trim().to_string(),
        pointer,
        tape,
    }
}

fn run_observed(
    program: &[BfIr],
    opts: CodegenOpts,
    backend: TestCBackend,
    extra_cc_args: &[&str],
) -> (ObservedState, String) {
    let source = instrument_final_state(&emit_backend_ir(program, opts, backend), backend);
    let output = compile_and_run_c_source_with_args(&source, extra_cc_args);
    (parse_observed_state(&output), output)
}

fn expected_payload(tape: &[i64]) -> String {
    OUTPUT_OFFSETS
        .into_iter()
        .map(|offset| tape[offset].to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn represented_steps(program: &[BfIr]) -> u64 {
    let mut steps = 0_u64;
    let mut stack = program.iter().rev().collect::<Vec<_>>();
    while let Some(node) = stack.pop() {
        match node {
            BfIr::MovePtr(delta) => steps += delta.unsigned_abs(),
            BfIr::Add(delta) => steps += u64::from(delta.unsigned_abs()),
            BfIr::Input | BfIr::Output => steps += 1,
            BfIr::Loop(body) => {
                steps += 1;
                stack.extend(body.iter().rev());
            }
            BfIr::Clear
            | BfIr::ClearAt { .. }
            | BfIr::Scan { .. }
            | BfIr::Distribute { .. }
            | BfIr::Shift { .. }
            | BfIr::Affine { .. }
            | BfIr::Square { .. }
            | BfIr::MulAdd { .. }
            | BfIr::Diverge => {}
        }
    }
    steps
}

fn polynomial_stat(output: &str, backend: TestCBackend, prefix: &str) -> u64 {
    match backend {
        TestCBackend::Plain => plain_stat(output, prefix),
        TestCBackend::Super => memo_stat(output, prefix),
    }
}

#[test]
fn manual_rich_composition_reaches_real_degrees_three_through_eight() {
    let (_, rich) = adversarial_program();
    let opts = numeric_opts(16, CellSign::Unsigned);
    let transfer = summarize_c_region(&rich, opts).or_invariant("rich region is summarizable");
    for (offset, degree) in [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (2, 8)] {
        assert_eq!(
            transfer.effects[&offset].degree(),
            degree,
            "offset={offset} degree={degree}"
        );
    }
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let source = emit_backend_ir(&rich, opts, backend);
        assert!(
            CSource::parse(&source).has_call_using_identifier_prefix(
                "bf_polynomial_mul_batch",
                0,
                "bf_poly_lhs_"
            ),
            "backend={backend:?} did not emit the higher-degree batch path"
        );
    }
}

#[test]
fn manual_rich_degree_three_through_eight_matches_interpreter_and_both_backends() {
    let (program, _) = adversarial_program();
    for cell_sign in [CellSign::Unsigned, CellSign::Signed] {
        for bits in [1, 8, 16, 32, 33, 63] {
            let opts = numeric_opts(bits, cell_sign);
            let (expected_tape, expected_pointer) =
                interpret_for_tests(&program, opts).or_invariant("manual rich program terminates");
            let expected_payload = expected_payload(&expected_tape);
            let (plain, _) = run_observed(&program, opts, TestCBackend::Plain, &["-O3"]);
            let (super_state, _) = run_observed(&program, opts, TestCBackend::Super, &["-O3"]);
            assert_eq!(
                plain.pointer, expected_pointer,
                "bits={bits} sign={cell_sign:?} plain pointer"
            );
            assert_eq!(
                plain.tape, expected_tape,
                "bits={bits} sign={cell_sign:?} plain tape"
            );
            assert_eq!(
                plain.payload, expected_payload,
                "bits={bits} sign={cell_sign:?} plain payload"
            );
            assert_eq!(
                super_state, plain,
                "bits={bits} sign={cell_sign:?} super full state"
            );
        }
    }
}

#[test]
fn manual_rich_native_and_forced_scalar_paths_match_full_state() {
    let (program, _) = adversarial_program();
    let opts = numeric_opts(16, CellSign::Unsigned);
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let (native, native_output) = run_observed(&program, opts, backend, &["-O3"]);
        let (scalar, scalar_output) = run_observed(
            &program,
            opts,
            backend,
            &["-O3", "-DBF_FORCE_SCALAR_KERNEL"],
        );
        assert_eq!(native, scalar, "backend={backend:?}");
        assert_eq!(
            polynomial_stat(
                &scalar_output,
                backend,
                "bf native polynomial kernel lanes:"
            ),
            0,
            "backend={backend:?}"
        );
        let total = polynomial_stat(&native_output, backend, "bf polynomial kernel lanes:");
        assert!(total >= 4, "backend={backend:?}\n{native_output}");
        assert_eq!(
            polynomial_stat(
                &scalar_output,
                backend,
                "bf scalar polynomial kernel lanes:"
            ),
            total,
            "backend={backend:?}"
        );
        #[cfg(target_arch = "aarch64")]
        if !cfg!(miri) {
            assert!(
                polynomial_stat(
                    &native_output,
                    backend,
                    "bf native polynomial kernel lanes:"
                ) >= 4,
                "backend={backend:?}\n{native_output}"
            );
        }
        #[cfg(target_arch = "x86_64")]
        if !cfg!(miri) && std::arch::is_x86_feature_detected!("avx2") {
            assert!(
                polynomial_stat(
                    &native_output,
                    backend,
                    "bf native polynomial kernel lanes:"
                ) >= 4,
                "backend={backend:?}\n{native_output}"
            );
        }
    }
}

#[test]
fn signed_63bit_polynomial_native_execution_matches_interpreter() {
    assert_signed_polynomial_execution(&["-O3"]);
}

#[test]
#[cfg(not(target_os = "windows"))]
fn signed_63bit_polynomial_native_execution_matches_interpreter_under_sanitizers() {
    assert_signed_polynomial_execution(&[
        "-O3",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
    ]);
}

fn assert_signed_polynomial_execution(cc_args: &[&str]) {
    let (program, _) = adversarial_program();
    let opts = numeric_opts(63, CellSign::Signed);
    let (expected_tape, expected_pointer) =
        interpret_for_tests(&program, opts).or_invariant("bounded polynomial fixture terminates");
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let (observed, _) = run_observed(&program, opts, backend, cc_args);
        assert_eq!(observed.pointer, expected_pointer, "backend={backend:?}");
        assert_eq!(observed.tape, expected_tape, "backend={backend:?}");
        assert_eq!(
            observed.payload,
            expected_payload(&expected_tape),
            "backend={backend:?}"
        );
    }
}

#[test]
fn semantic_fuel_keeps_rich_regions_on_the_exact_primitive_fallback_boundary() {
    let (program, _) = adversarial_program();
    let opts = numeric_opts(8, CellSign::Unsigned);
    let fuel_arg = format!(
        "-DBF_TEST_SEMANTIC_FUEL_LIMIT={}",
        represented_steps(&program)
    );
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let source = instrument_final_state(&emit_backend_ir(&program, opts, backend), backend);
        let fuel_branch = match backend {
            TestCBackend::Plain => "!bf_semantic_fuel_enabled()",
            TestCBackend::Super => "bf_semantic_fuel_enabled()",
        };
        assert!(
            CSource::parse(&source).has_condition("if_statement", fuel_branch),
            "backend={backend:?}"
        );
        let outcome = generated_c_outcome(compile_and_run_c_source_capture(&source, &[&fuel_arg]));
        let GeneratedCOutcome::SemanticError {
            exit_code, stderr, ..
        } = outcome
        else {
            crate::invariant_failure!(
                "semantic fuel admitted richer summary for backend={backend:?}: {outcome:?}"
            );
        };
        assert_eq!(exit_code, Some(125), "backend={backend:?}");
        assert!(
            String::from_utf8_lossy(&stderr).contains("BF_TEST_SEMANTIC_FUEL_UNSUPPORTED_SUMMARY"),
            "backend={backend:?} stderr={}",
            String::from_utf8_lossy(&stderr)
        );
    }
}

#[test]
fn common_factor_mixed_sources_matches_interpreter_and_both_backends() {
    // Eight setup nodes followed by an eight-node rich tile make the same factored
    // transfer visible to both the plain region scanner and the super DAG emitter.
    let mut program = vec![
        BfIr::Add(7),
        BfIr::MovePtr(1),
        BfIr::Add(11),
        BfIr::MovePtr(1),
        BfIr::Add(13),
        BfIr::MovePtr(1),
        BfIr::Add(5),
        BfIr::MovePtr(-3),
        BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 3,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: true,
        },
        BfIr::MulAdd {
            lhs: 0,
            rhs: 2,
            dst: 3,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        },
    ];
    for dst in 4..10 {
        program.push(BfIr::Affine {
            src: 0,
            dst,
            coeff: 1,
            preserve_src: true,
            set_dst: true,
        });
    }
    program.extend([BfIr::MovePtr(3), BfIr::Output]);
    for cell_sign in [CellSign::Unsigned, CellSign::Signed] {
        let opts = numeric_opts(8, cell_sign);
        let (expected_tape, expected_pointer) =
            interpret_for_tests(&program, opts).or_invariant("factored rich program terminates");
        for backend in [TestCBackend::Plain, TestCBackend::Super] {
            let source = emit_backend_ir(&program, opts, backend);
            assert!(
                CSource::parse(&source).has_identifier("bf_poly_factor_value_0"),
                "backend={backend:?} did not select the common factor"
            );
            let (observed, _) = run_observed(&program, opts, backend, &["-O3"]);
            assert_eq!(observed.pointer, expected_pointer, "backend={backend:?}");
            assert_eq!(observed.tape, expected_tape, "backend={backend:?}");
            assert_eq!(observed.payload, expected_tape[3].to_string());
        }
    }
}
