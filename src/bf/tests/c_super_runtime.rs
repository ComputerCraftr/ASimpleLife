use super::*;

mod emitted_simd;
mod safety;

fn backend_kernel_stat(output: &str, backend: TestCBackend, prefix: &str) -> u64 {
    match backend {
        TestCBackend::Plain => plain_stat(output, prefix),
        TestCBackend::Super => memo_stat(output, prefix),
    }
}

fn assert_kernel_lane_conservation(output: &str, backend: TestCBackend, family: &str) {
    let total = backend_kernel_stat(output, backend, &format!("bf {family} kernel lanes:"));
    let native = backend_kernel_stat(
        output,
        backend,
        &format!("bf native {family} kernel lanes:"),
    );
    let scalar = backend_kernel_stat(
        output,
        backend,
        &format!("bf scalar {family} kernel lanes:"),
    );
    assert_eq!(
        total,
        native + scalar,
        "backend={backend:?} family={family} total={total} native={native} scalar={scalar}\n{output}"
    );
}

#[test]
fn primitive_guard_restoring_loop_exhausts_equal_semantic_fuel_in_both_backends() {
    let program = vec![BfIr::Add(1), BfIr::Loop(vec![BfIr::Add(-1), BfIr::Add(1)])];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_c_source_capture(
        &emit_backend_ir(&program, opts, TestCBackend::Plain),
        &["-DBF_TEST_SEMANTIC_FUEL_LIMIT=10000"],
    );
    let super_c = emit_backend_ir(&program, opts, TestCBackend::Super);
    assert!(
        !super_c.contains("dynamic_spec_"),
        "guard-writing loop must not be admitted to the runtime reducer"
    );
    let optimized =
        compile_and_run_c_source_capture(&super_c, &["-DBF_TEST_SEMANTIC_FUEL_LIMIT=10000"]);
    let plain = generated_c_outcome(plain);
    let optimized = generated_c_outcome(optimized);
    assert!(
        matches!(plain, GeneratedCOutcome::WorkLimitReached { .. }),
        "plain C must exhaust deterministic work rather than terminate or fail: {plain:?}"
    );
    assert_eq!(
        optimized, plain,
        "super C must exhaust work with the same observable prefix as plain C"
    );
}

#[test]
fn semantic_fuel_rejects_richer_summary_without_exact_source_debit() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let output = compile_and_run_c_source_capture(
            &emit_backend_ir(&[BfIr::Clear], opts, backend),
            &["-DBF_TEST_SEMANTIC_FUEL_LIMIT=10"],
        );
        assert_eq!(output.status.code(), Some(125), "backend={backend:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("BF_TEST_SEMANTIC_FUEL_UNSUPPORTED_SUMMARY"),
            "backend={backend:?} stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn wrapped_pointer_moves_preserve_semantic_fuel_before_address_normalization() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    for delta in [30_000, -30_000, 30_001] {
        let program = [BfIr::MovePtr(delta), BfIr::Output];
        let mut outcomes = Vec::new();
        for backend in [TestCBackend::Plain, TestCBackend::Super] {
            outcomes.push(generated_c_outcome(compile_and_run_c_source_capture(
                &emit_backend_ir(&program, opts, backend),
                &["-DBF_TEST_SEMANTIC_FUEL_LIMIT=30000"],
            )));
        }
        assert_eq!(
            outcomes[0], outcomes[1],
            "wrapped move delta={delta} consumed different semantic prefixes"
        );
        assert!(
            matches!(outcomes[0], GeneratedCOutcome::WorkLimitReached { .. }),
            "wrapped move delta={delta} bypassed semantic fuel: {:?}",
            outcomes[0]
        );
    }
}

#[test]
fn circular_scan_without_reachable_zero_exhausts_work_for_all_strides() {
    for template_path in ["src/bf/bf.c.in", "src/bf/bf_super.c.in"] {
        let template = fs::read_to_string(template_path).or_invariant("C template");
        for stride in [0, 1, 2] {
            let source = format!(
                "#define main bf_template_main\n{template}\n#undef main\n\
                 int main(void) {{\n\
                     int64_t tape[BF_TEMPLATE_TAPE_LEN];\n\
                     for (size_t i = 0; i < BF_TEMPLATE_TAPE_LEN; ++i) tape[i] = 1;\n\
                     uint64_t steps = 0;\n\
                     bf_kernel_init();\n\
                     (void)bf_scan_zero(tape, 0, {stride}, BF_TEMPLATE_TAPE_LEN, &steps);\n\
                     return 0;\n\
                 }}\n"
            );
            let outcome = generated_c_outcome(compile_and_run_c_source_capture(
                &source,
                &["-DBF_TEST_WORK_LIMIT=1000"],
            ));
            assert!(
                matches!(outcome, GeneratedCOutcome::WorkLimitReached { .. }),
                "template={template_path} stride={stride} did not exhaust deterministic work: {outcome:?}"
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn generated_c_avx2_kernel_is_isolated_from_baseline_translation_unit() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let program = [
        BfIr::Add(1),
        BfIr::Scan { stride: 1 },
        BfIr::Add(7),
        BfIr::Output,
    ];
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let output = compile_and_run_c_source_with_args(
            &emit_backend_ir(&program, opts, backend),
            &["-march=x86-64", "-mno-avx2"],
        );
        let payload = match backend {
            TestCBackend::Plain => split_plain_c_payload(&output),
            TestCBackend::Super => split_super_c_payload(&output),
        };
        assert_eq!(payload.trim(), "7", "backend={backend:?}");
    }
}

#[test]
fn positive_guard_invariant_product_matches_plain_c() {
    let loop_body = BfIr::Loop(vec![
        BfIr::Add(1),
        BfIr::Affine {
            src: 1,
            dst: 2,
            coeff: 2,
            preserve_src: true,
            set_dst: false,
        },
    ]);
    let mut program = vec![BfIr::MovePtr(1), BfIr::Add(3), BfIr::MovePtr(-1)];
    for _ in 0..9 {
        program.extend([BfIr::Add(1), loop_body.clone()]);
    }
    program.extend([BfIr::MovePtr(2), BfIr::Output]);
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };

    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);
    assert_super_payload_matches_plain("positive guard invariant product", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "202");
}

#[test]
fn runtime_summary_positive_guard_negates_invariant_product_trip_count() {
    let template = fs::read_to_string("src/bf/bf_super.c.in").or_invariant("required value");
    let source = format!(
        "#define BF_TEMPLATE_SIGNED_CELLS 0\n#define main bf_template_main\n{template}\n#undef main\n\
         int main(void) {{\n\
             static const DynamicOp ops[] = {{\n\
                 {{ BF_DYNAMIC_OP_ADD, 1, 0, 0, 0 }},\n\
                 {{ BF_DYNAMIC_OP_AFFINE, 1, 2, 2, 1 }}\n\
             }};\n\
             const DynamicLoopSpec spec = {{ 0U, UINT64_C(7), 0, 3U, 2U, ops }};\n\
             int64_t tape[BF_TAPE_LEN] = {{0}};\n\
             tape[0] = 1; tape[1] = 3;\n\
             if (!bf_trace_dynamic_loop(&spec) || !bf_apply_runtime_summary(&spec, tape, 0)) return 2;\n\
             printf(\"%\" PRId64 \" %\" PRId64 \"\\n\", tape[2], tape[0]);\n\
             return 0;\n\
         }}\n"
    );
    let output = compile_and_run_c_source(&source);
    assert_eq!(
        split_super_c_payload(&output).trim(),
        "250 0",
        "a +1 guard executes -guard iterations modulo 2^w"
    );
}

#[test]
fn memo_window_offsets_beyond_i8_range_match_plain_c() {
    let mut repeated = vec![BfIr::MovePtr(200), BfIr::Add(1), BfIr::MovePtr(-200)];
    repeated.extend(std::iter::repeat_n(BfIr::MovePtr(0), 5));
    let mut program = repeated.clone();
    program.extend(repeated);
    program.extend([BfIr::MovePtr(200), BfIr::Output]);
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };

    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);
    assert_super_payload_matches_plain("memo offset 200", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "2");
}

#[test]
fn wrapped_richer_ir_aliases_fall_back_without_symbolic_miscompilation() {
    let program = vec![
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(-1),
        BfIr::MulAdd {
            lhs: 1,
            rhs: 30_001,
            dst: 2,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        },
        BfIr::Affine {
            src: 1,
            dst: 30_001,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        },
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };

    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);
    assert_super_payload_matches_plain("wrapped richer-IR aliases", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "6\n9");
}

#[test]
fn emit_c_backends_scan_across_both_tape_boundaries() {
    let program = vec![
        BfIr::MovePtr(-1),
        BfIr::Add(1),
        BfIr::Scan { stride: 1 },
        BfIr::Output,
        BfIr::Add(1),
        BfIr::Scan { stride: -1 },
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);

    assert_super_payload_matches_plain("bidirectional boundary scan", &plain, &super_out);
    assert_eq!(split_plain_c_payload(&plain).trim(), "0\n0");
}

#[test]
fn generated_c_scan_uses_the_runtime_selected_kernel() {
    let mut program = Vec::new();
    for _ in 0..8 {
        program.extend([BfIr::Add(1), BfIr::MovePtr(1)]);
    }
    program.extend([BfIr::MovePtr(-8), BfIr::Scan { stride: 1 }, BfIr::Output]);
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };

    for (label, backend) in [
        ("plain C", TestCBackend::Plain),
        ("super C", TestCBackend::Super),
    ] {
        let output = compile_and_run_ir_backend(&program, opts, backend);
        let stat = |prefix| match backend {
            TestCBackend::Plain => plain_stat(&output, prefix),
            TestCBackend::Super => memo_stat(&output, prefix),
        };
        let native = stat("bf native avx2 lanes:") + stat("bf native neon lanes:");
        let scalar = stat("bf scalar kernel lanes:");
        assert_kernel_lane_conservation(&output, backend, "scan");
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            assert!(
                native >= 8 && stat("bf native scan kernel lanes:") >= 8,
                "{label} did not execute a native scan kernel: {output}"
            );
        } else {
            assert!(
                scalar >= 9,
                "{label} scalar fallback did not inspect the scan: {output}"
            );
        }
    }
}

#[test]
fn generated_c_scan_forced_scalar_path_does_not_claim_native_lanes() {
    let program = vec![
        BfIr::Add(1),
        BfIr::MovePtr(1),
        BfIr::Add(1),
        BfIr::MovePtr(-1),
        BfIr::Scan { stride: 1 },
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };

    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let source = emit_backend_ir(&program, opts, backend);
        let output = compile_and_run_c_source_with_args(&source, &["-DBF_FORCE_SCALAR_KERNEL"]);
        let stat = |prefix| match backend {
            TestCBackend::Plain => plain_stat(&output, prefix),
            TestCBackend::Super => memo_stat(&output, prefix),
        };
        assert_eq!(stat("bf native avx2 lanes:"), 0);
        assert_eq!(stat("bf native neon lanes:"), 0);
        assert_eq!(stat("bf native scan kernel lanes:"), 0);
        assert_kernel_lane_conservation(&output, backend, "scan");
        assert!(
            stat("bf scalar kernel lanes:") > 0,
            "forced scalar scan did not attribute work to the scalar kernel: {output}"
        );
    }
}

#[test]
fn generated_c_batched_kernels_match_forced_scalar_with_wrapped_addresses() {
    for signed_cells in [0, 1] {
        for (template_path, stats_call, stats_prefix) in [
            (
                "src/bf/bf.c.in",
                "print_work_stats();",
                "=== BF PLAIN RUNTIME STATS ===",
            ),
            (
                "src/bf/bf_super.c.in",
                "print_memo_stats();",
                "=== BF SUPER RUNTIME STATS ===",
            ),
        ] {
            let template = fs::read_to_string(template_path).or_invariant("C template");
            let source = format!(
                "#define BF_TEMPLATE_SIGNED_CELLS {signed_cells}\n#define main bf_template_main\n{template}\n#undef main\n\
             int main(void) {{\n\
                 int64_t tape[BF_TEMPLATE_TAPE_LEN] = {{0}};\n\
                 const ptrdiff_t add_offsets[] = {{0, 1, BF_TEMPLATE_TAPE_LEN + 2, -1}};\n\
                 const int64_t add_deltas[] = {{255, -2, 257, 3}};\n\
                 const ptrdiff_t set_offsets[] = {{3, 4, 5, 6}};\n\
                 const int64_t set_values[] = {{7, 8, 9, 10}};\n\
                 const ptrdiff_t transfer_offsets[] = {{7, 8, 9, 10}};\n\
                 const int64_t coefficients[] = {{1, -2, 3, 4}};\n\
                 tape[0] = 3; bf_kernel_init(); bf_semantic_step(UINT64_C(23));\n\
                 bf_add_at_batch(tape, 0, add_offsets, add_deltas, 4, BF_TEMPLATE_TAPE_LEN, 8, 0);\n\
                 bf_clear_set_batch(tape, 0, set_offsets, set_values, 4, BF_TEMPLATE_TAPE_LEN);\n\
                 bf_transfer_batch(tape, 0, 0, transfer_offsets, coefficients, 4, 1, BF_TEMPLATE_TAPE_LEN, 8, 0);\n\
                 for (size_t i = 0; i <= 10; ++i) printf(\"%\" PRId64 \" \", tape[i]);\n\
                 printf(\"%\" PRId64 \"\\n\", tape[BF_TEMPLATE_TAPE_LEN - 1]);\n\
                 {stats_call} return 0;\n\
             }}\n"
            );
            let native = compile_and_run_c_source_with_args(&source, &["-O3"]);
            let scalar =
                compile_and_run_c_source_with_args(&source, &["-O3", "-DBF_FORCE_SCALAR_KERNEL"]);
            let payload = |output: &str| {
                output
                    .split(stats_prefix)
                    .next()
                    .unwrap_or(output)
                    .trim()
                    .to_owned()
            };
            assert_eq!(
                payload(&native),
                payload(&scalar),
                "template={template_path} signed={signed_cells}"
            );
            let stat = |output: &str, prefix: &str| {
                output
                    .lines()
                    .find_map(|line| line.strip_prefix(prefix))
                    .and_then(|value| value.trim().parse::<u64>().ok())
                    .unwrap_or(0)
            };
            for family in [
                "bf add kernel lanes:",
                "bf clear/set kernel lanes:",
                "bf transfer kernel lanes:",
            ] {
                assert_eq!(
                    stat(&native, family),
                    4,
                    "template={template_path} {native}"
                );
                assert_eq!(
                    stat(&scalar, family),
                    4,
                    "template={template_path} {scalar}"
                );
            }
            assert_eq!(stat(&scalar, "bf native avx2 lanes:"), 0);
            assert_eq!(stat(&scalar, "bf native neon lanes:"), 0);
            assert_eq!(stat(&native, "semantic steps:"), 23, "{native}");
            assert_eq!(stat(&scalar, "semantic steps:"), 23, "{scalar}");
            for family in ["add", "clear/set", "transfer"] {
                let total_prefix = format!("bf {family} kernel lanes:");
                let native_prefix = format!("bf native {family} kernel lanes:");
                let scalar_prefix = format!("bf scalar {family} kernel lanes:");
                let total = stat(&native, &total_prefix);
                let native_lanes = stat(&native, &native_prefix);
                let scalar_lanes = stat(&native, &scalar_prefix);
                assert_eq!(total, native_lanes + scalar_lanes, "{native}");
                assert_eq!(stat(&scalar, &native_prefix), 0, "{scalar}");
                assert_eq!(stat(&scalar, &scalar_prefix), total, "{scalar}");
            }
            if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
                for family in ["add", "clear/set", "transfer"] {
                    let native_prefix = format!("bf native {family} kernel lanes:");
                    assert_eq!(
                        stat(&native, &native_prefix),
                        4,
                        "template={template_path} family={family} did not execute natively: {native}"
                    );
                }
            }
        }
    }
}

#[test]
fn generated_c_add_and_set_wrapped_aliases_use_sequential_scalar_fallbacks() {
    for (template_path, stats_call, stats_prefix) in [
        (
            "src/bf/bf.c.in",
            "print_work_stats();",
            "=== BF PLAIN RUNTIME STATS ===",
        ),
        (
            "src/bf/bf_super.c.in",
            "print_memo_stats();",
            "=== BF SUPER RUNTIME STATS ===",
        ),
    ] {
        let template = fs::read_to_string(template_path).or_invariant("C template");
        let source = format!(
            "#define main bf_template_main\n{template}\n#undef main\n\
             int main(void) {{\n\
                 int64_t tape[BF_TEMPLATE_TAPE_LEN] = {{0}};\n\
                 const ptrdiff_t add_offsets[] = {{1, BF_TEMPLATE_TAPE_LEN + 1, 2, 3}};\n\
                 const int64_t add_deltas[] = {{1, 2, 3, 4}};\n\
                 const ptrdiff_t set_offsets[] = {{4, BF_TEMPLATE_TAPE_LEN + 4, 5, 6}};\n\
                 const int64_t set_values[] = {{7, 8, 9, 10}};\n\
                 bf_kernel_init();\n\
                 bf_add_at_batch(tape, 0, add_offsets, add_deltas, 4, BF_TEMPLATE_TAPE_LEN, 8, 0);\n\
                 bf_clear_set_batch(tape, 0, set_offsets, set_values, 4, BF_TEMPLATE_TAPE_LEN);\n\
                 printf(\"%\" PRId64 \" %\" PRId64 \" %\" PRId64 \" %\" PRId64 \" %\" PRId64 \" %\" PRId64 \"\\n\",\n\
                        tape[1], tape[2], tape[3], tape[4], tape[5], tape[6]);\n\
                 {stats_call} return 0;\n\
             }}\n"
        );
        let output = compile_and_run_c_source_with_args(&source, &["-O3"]);
        let payload = output.split(stats_prefix).next().unwrap_or(&output).trim();
        assert_eq!(
            payload, "3 3 4 8 9 10",
            "template={template_path}\n{output}"
        );
        let stat = |prefix: &str| {
            output
                .lines()
                .find_map(|line| line.strip_prefix(prefix))
                .and_then(|value| value.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        for family in ["add", "clear/set"] {
            let total_prefix = format!("bf {family} kernel lanes:");
            let native_prefix = format!("bf native {family} kernel lanes:");
            let scalar_prefix = format!("bf scalar {family} kernel lanes:");
            assert_eq!(stat(&total_prefix), 4, "{output}");
            assert_eq!(stat(&native_prefix), 0, "{output}");
            assert_eq!(stat(&scalar_prefix), 4, "{output}");
        }
    }
}

#[test]
fn richer_simd_entrypoints_remain_fail_closed_under_semantic_fuel() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let program = [
        BfIr::Add(4),
        BfIr::Distribute {
            targets: vec![(1, 1), (2, 2), (3, 3), (4, 4)],
            preserve_src: false,
        },
    ];
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let output = compile_and_run_c_source_capture(
            &emit_backend_ir(&program, opts, backend),
            &["-DBF_TEST_SEMANTIC_FUEL_LIMIT=100"],
        );
        assert_eq!(output.status.code(), Some(125), "backend={backend:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("BF_TEST_SEMANTIC_FUEL_UNSUPPORTED_SUMMARY"),
            "backend={backend:?} stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn emitted_distribute_uses_transfer_kernel_and_wrapped_aliases_fall_back() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let disjoint = [
        BfIr::Add(3),
        BfIr::Distribute {
            targets: vec![(1, 1), (2, -2), (3, 3), (4, 4)],
            preserve_src: true,
        },
    ];
    let aliased = [
        BfIr::Add(3),
        BfIr::Distribute {
            targets: vec![(1, 1), (30_001, 2), (2, 3), (3, 4)],
            preserve_src: true,
        },
    ];
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let output = compile_and_run_ir_backend(&disjoint, opts, backend);
        let stat = |prefix| match backend {
            TestCBackend::Plain => plain_stat(&output, prefix),
            TestCBackend::Super => memo_stat(&output, prefix),
        };
        assert_eq!(
            stat("bf transfer kernel lanes:"),
            4,
            "backend={backend:?} {output}"
        );
        assert_kernel_lane_conservation(&output, backend, "transfer");
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            assert_eq!(
                stat("bf native transfer kernel lanes:"),
                4,
                "backend={backend:?} {output}"
            );
        }

        let output = compile_and_run_ir_backend(&aliased, opts, backend);
        let stat = |prefix| match backend {
            TestCBackend::Plain => plain_stat(&output, prefix),
            TestCBackend::Super => memo_stat(&output, prefix),
        };
        assert_eq!(
            stat("bf transfer kernel lanes:"),
            4,
            "backend={backend:?} {output}"
        );
        assert_kernel_lane_conservation(&output, backend, "transfer");
        assert_eq!(
            stat("bf native transfer kernel lanes:"),
            0,
            "wrapped aliases must not be attributed to native transfer: backend={backend:?} {output}"
        );
        assert!(
            stat("bf scalar kernel lanes:") >= 4,
            "wrapped duplicate destinations must force scalar transfer: backend={backend:?} {output}"
        );
    }
}

#[test]
fn super_c_runtime_summary_and_probe_use_their_kernel_families() {
    let template = fs::read_to_string("src/bf/bf_super.c.in").or_invariant("C template");
    let source = format!(
        "#define BF_TEMPLATE_SIGNED_CELLS 0\n#define main bf_template_main\n{template}\n#undef main\n\
         int main(void) {{\n\
             int64_t tape[BF_TAPE_LEN] = {{0}}; tape[0] = 3;\n\
             const DynamicLoopSpec spec = {{0U, UINT64_C(9), 0, 5U, 0U, NULL}};\n\
             RuntimeLoopSummary *summary = &runtime_loop_summaries[0];\n\
             summary->state = BF_DYNAMIC_INSTALLED; summary->body_hash = UINT64_C(9);\n\
             summary->guard_delta = -1; summary->effect_count = 5;\n\
             for (int32_t i = 0; i < 4; ++i) summary->effects[i] = (RuntimeSummaryEffect){{BF_SUMMARY_EFFECT_ADD_SCALED, i + 1, 0, i + 1}};\n\
             summary->effects[4] = (RuntimeSummaryEffect){{BF_SUMMARY_EFFECT_CLEAR, 0, 0, 0}};\n\
             bf_kernel_init(); if (!bf_apply_runtime_summary(&spec, tape, 0)) return 2;\n\
             MemoKey key = {{0}}; MemoVal value; (void)bf_memo_lookup(&key, &value);\n\
             printf(\"%\" PRId64 \" %\" PRId64 \" %\" PRId64 \" %\" PRId64 \" %\" PRId64 \"\\n\", tape[0], tape[1], tape[2], tape[3], tape[4]);\n\
             print_memo_stats(); return 0;\n\
         }}\n"
    );
    let output = compile_and_run_c_source(&source);
    assert_eq!(split_super_c_payload(&output).trim(), "0 3 6 9 12");
    assert_eq!(
        memo_stat(&output, "bf summary kernel lanes:"),
        4,
        "{output}"
    );
    assert_kernel_lane_conservation(&output, TestCBackend::Super, "summary");
    assert_kernel_lane_conservation(&output, TestCBackend::Super, "probe");
    assert_eq!(memo_stat(&output, "bf probe kernel lanes:"), 16, "{output}");
    if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        assert_eq!(
            memo_stat(&output, "bf native summary kernel lanes:"),
            4,
            "{output}"
        );
        assert_eq!(
            memo_stat(&output, "bf native probe kernel lanes:"),
            16,
            "{output}"
        );
    }
}

#[test]
fn emit_c_backends_count_each_scan_move_as_runtime_work() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_emitted_backend("+[>].", opts, TestCBackend::Plain, false);
    let super_out = compile_and_run_emitted_backend("+[>].", opts, TestCBackend::Super, false);

    assert_super_payload_matches_plain("single-move scan work", &plain, &super_out);
    assert_eq!(plain_stat(&plain, "work ops:"), 3, "{plain}");
    assert_eq!(super_work_stat(&super_out, "work ops:"), 3, "{super_out}");
}

#[test]
fn emit_c_backends_accept_extreme_scan_stride_without_host_overflow() {
    let program = vec![
        BfIr::Add(1),
        BfIr::Scan {
            stride: crate::bf::BfOffset::MIN,
        },
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);

    assert_super_payload_matches_plain("minimum-stride scan", &plain, &super_out);
    assert_eq!(split_plain_c_payload(&plain).trim(), "0");
}

#[test]
#[cfg(not(target_os = "windows"))]
fn emit_c_super_signed_minimum_guard_power_is_ubsan_clean() {
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
        BfIr::Loop(vec![BfIr::Add(1)]),
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 63,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    };
    let super_c = emit_c_super(&program, opts);
    let super_out = compile_and_run_c_source_sanitized(&super_c);

    assert_eq!(split_super_c_payload(&super_out).trim(), "0");
    assert!(
        memo_stat(&super_out, "symbolic power hits:") > 0,
        "signed minimum guard must be reduced by compact powers:\n{super_out}"
    );
}

#[test]
fn emit_c_super_transducer_capacity_fails_closed_to_exact_memo_path() {
    let mut program = Vec::new();
    for guard in 2..=6 {
        program.extend([
            BfIr::Add(guard),
            BfIr::MovePtr(1),
            BfIr::Add(2),
            BfIr::MovePtr(-1),
            BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::Square {
                    src: 1,
                    dst: 1,
                    preserve_src: true,
                    set_dst: true,
                },
            ]),
            BfIr::MovePtr(2),
        ]);
    }
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_c = emit_c_super(&program, opts).replacen(
        "#define BF_TEMPLATE_DYNAMIC_HOT_THRESHOLD 8",
        "#define BF_TEMPLATE_DYNAMIC_HOT_THRESHOLD 1",
        1,
    );
    let super_out = compile_and_run_c_source(&super_c);

    assert_super_payload_matches_plain("saturated guarded transducer", &plain, &super_out);
    assert_eq!(
        memo_stat(&super_out, "dynamic transducer installs:"),
        4,
        "bounded transducer must not exceed its generated capacity:\n{super_out}"
    );
    assert!(
        memo_stat(&super_out, "dynamic memo after rejection:") > 0,
        "a fifth distinct state must fail closed to exact memo/fallback:\n{super_out}"
    );
}

#[test]
fn emit_c_super_cold_dynamic_loop_uses_exact_memo_before_fallback() {
    let loop_body = vec![
        BfIr::Add(-1),
        BfIr::Square {
            src: 1,
            dst: 1,
            preserve_src: true,
            set_dst: true,
        },
    ];
    let program = vec![
        BfIr::Add(2),
        BfIr::MovePtr(1),
        BfIr::Add(2),
        BfIr::MovePtr(-1),
        BfIr::Loop(loop_body.clone()),
        BfIr::MovePtr(2),
        BfIr::Add(2),
        BfIr::MovePtr(1),
        BfIr::Add(2),
        BfIr::MovePtr(-1),
        BfIr::Loop(loop_body),
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);

    assert!(
        memo_hits(&super_out) > 0,
        "cold repeated loop state should hit exact memo before the hot threshold:\n{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic trace attempts:"),
        0,
        "two entries must remain below the default hot threshold:\n{super_out}"
    );
}

#[test]
fn emit_c_super_repeated_loop_regions_do_less_work_than_plain_c() {
    assert_super_c_does_less_work_than_plain_c(
        "++++[--]++++[--]++++[--]",
        default_c_opts(),
        "work ops:",
    );
}

#[test]
fn emit_c_super_repeated_powered_loop_regions_do_less_work_than_plain_c() {
    assert_super_c_does_less_work_than_plain_c(
        "++++[--]>++++[--]>++++[--]<<.>.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        "work ops:",
    );
}

#[test]
fn emit_c_super_composed_richer_summary_program_does_less_work_than_plain_c() {
    assert_super_c_does_less_work_than_plain_c(
        "++++[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<>>.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        "work ops:",
    );
}

#[test]
fn emit_c_super_reports_recursive_fallbacks_for_exact_loop_memo() {
    let stdout = compile_and_run_ir_backend(
        &[
            BfIr::Add(3),
            BfIr::Loop(vec![BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: false,
                set_dst: false,
            }]),
            BfIr::MovePtr(1),
            BfIr::Output,
        ],
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Super,
    );
    assert!(
        memo_stat(&stdout, "recursion fallbacks:") > 0,
        "expected exact loop memo execution to recurse through the loop body on memo miss, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_non_powered_loop_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("+++[->+<].", opts);
}

#[test]
fn emit_c_backends_are_deterministic_for_unsigned_number_programs() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    for src in [
        "++>+<.>.",
        "++++[--].",
        "+++[->++<]>.<.",
        ">+++<[--]>.",
        "+++[->+<].",
        "+[>+<,]>.",
    ] {
        assert_super_c_matches_plain_c(src, opts);
    }
}

#[test]
fn emit_c_backends_are_deterministic_for_signed_number_programs() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    };
    for src in ["-.", "--.", "+++.---.", "++++[--]."] {
        assert_super_c_matches_plain_c(src, opts);
    }
}

#[test]
fn emit_c_backends_are_deterministic_for_signed_63bit_number_programs() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 63,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    };
    for src in ["-.", "--.", "++++[--].", "+++[->++<]>.<."] {
        assert_super_c_matches_plain_c(src, opts);
    }
}
