use super::*;
use crate::bf::ShiftDir;

#[test]
fn emit_c_super_contains_symbolic_memo_runtime() {
    let c = emit_c_super(&parse_and_opt("+++[->+<]>+."), default_c_opts());
    let syntax = CSource::parse(&c);
    assert!(syntax.has_identifier("MemoKey"));
    assert!(syntax.has_call("bf_memo_lookup"));
    assert!(syntax.has_call("print_memo_stats"));
    assert!(syntax.has_identifier("node_hits"));
}

fn compile_and_run_emitted_c_super_with_memo_capacity(
    src: &str,
    opts: CodegenOpts,
    memo_capacity: usize,
) -> String {
    let c = emit_c_super(&parse_and_opt(src), opts);
    let c = CSource::parse(&c)
        .replace_define("BF_TEMPLATE_MEMO_CAPACITY", &memo_capacity.to_string())
        .or_invariant("unique memo capacity definition");
    compile_and_run_c_source(&c)
}

#[test]
fn emit_c_super_symbolic_seq_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("++>+<.>.", opts);
}

#[test]
fn emit_c_scan_summary_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let optimized = parse_and_opt("+>+>+<<[>].");

    assert!(
        optimized
            .iter()
            .any(|node| matches!(node, BfIr::Scan { stride: 1 })),
        "scan fixture should reach the dedicated IR: {optimized:?}"
    );
    let plain = compile_and_run_ir_backend(&optimized, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&optimized, opts, TestCBackend::Super);
    assert_super_payload_matches_plain("scan summary", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "0");
}

#[test]
fn emit_c_scan_only_program_retains_pointer_wrap_helper() {
    let plain =
        compile_and_run_emitted_backend("[>]", default_c_opts(), TestCBackend::Plain, false);
    let super_out =
        compile_and_run_emitted_backend("[>]", default_c_opts(), TestCBackend::Super, false);

    assert_super_payload_matches_plain("scan-only zero-entry loop", &plain, &super_out);
}

#[test]
fn emit_c_super_powered_loop_handles_65536_iterations() {
    let program = vec![
        BfIr::Add(65_536),
        BfIr::Loop(vec![BfIr::Add(-1)]),
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 17,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);

    assert_super_payload_matches_plain("17-bit powered clear", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "0");
    assert!(
        memo_stat(&super_out, "symbolic power hits:") > 0,
        "the 65,536-iteration guard must use a proven high power:\n{super_out}"
    );
}

#[test]
fn emit_c_super_powered_distribute_accumulates_duplicate_targets() {
    let program = vec![
        BfIr::Add(3),
        BfIr::Loop(vec![
            BfIr::Distribute {
                targets: vec![(1, 1), (1, 2)],
                preserve_src: true,
            },
            BfIr::Add(-1),
        ]),
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

    assert_super_payload_matches_plain("duplicate distribute targets", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "18");
}

#[test]
fn emit_c_manual_richer_summary_ir_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let program = vec![
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
    assert_super_c_matches_plain_c_ir("richer summary IR", &program, opts);
}

#[test]
fn emit_c_backends_reject_noncanonical_aliased_muladd_ir() {
    let ir = vec![BfIr::MulAdd {
        lhs: 1,
        rhs: 1,
        dst: 2,
        preserve_lhs: true,
        preserve_rhs: false,
        set_dst: true,
    }];
    assert_panics_quietly(|| {
        emit_c(&ir, default_c_opts());
    });
    assert_panics_quietly(|| {
        emit_c_super(&ir, default_c_opts());
    });
}

#[test]
fn emit_c_manual_canonical_square_and_alias_to_destination_ir_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let program = vec![
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
    assert_super_c_matches_plain_c_ir("canonical alias-rich IR", &program, opts);
}

#[test]
fn emit_c_exponentiation_by_squaring_source_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c(
        "++++[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<>>.",
        opts,
    );
}

#[test]
fn emit_c_non_unit_odd_affine_loop_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("+++[--->+++<]>.", opts);
}

#[test]
fn emit_c_preserving_multitarget_copy_source_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("+++[->+>++>+<<<]>>>[-<<<+>>>]<<<>.>.", opts);
}

#[test]
fn emit_c_signed_exponentiation_by_squaring_source_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    };
    assert_super_c_matches_plain_c(
        "++++[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<>>.",
        opts,
    );
}

#[test]
fn emit_c_manual_temp_consuming_polynomial_cleanup_ir_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let optimized = vec![
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
    assert_super_c_matches_plain_c_ir("temp-consuming polynomial cleanup", &optimized, opts);
}

#[test]
fn emit_c_manual_mixed_linear_and_product_sequence_runtime_matches_between_backends() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let optimized = vec![
        BfIr::Add(4),
        BfIr::MovePtr(1),
        BfIr::Add(7),
        BfIr::MovePtr(-1),
        BfIr::Shift {
            src: 0,
            dst: 2,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 2,
            preserve_lhs: false,
            preserve_rhs: true,
            set_dst: false,
        },
        BfIr::MovePtr(2),
        BfIr::Output,
    ];
    assert_super_c_matches_plain_c_ir("ssa mixed linear/product sequence", &optimized, opts);
}

#[test]
fn emit_c_super_loop_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("+++[->++<]>.<.", opts);
}

#[test]
fn emit_c_super_powered_loop_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("++++[--].", opts);
}

#[test]
fn emit_c_super_multi_decrement_loop_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("++++++[--].", opts);
}

#[test]
fn emit_c_super_nested_powered_loop_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c(">+++<[--]>.", opts);
}

#[test]
fn emit_c_super_residual_loop_runtime_matches_plain_c() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c("+[>+<,]>.", opts);
}

#[test]
fn emit_c_super_unsigned_runtime_executes_distribute_loop_correctly() {
    let stdout = compile_and_run_emitted_backend(
        "+++[->++<]>.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Super,
        false,
    );
    assert!(stdout.starts_with("6\n"));
    assert!(stdout.contains("work dispatches:"));
    assert!(stdout.contains("work loop iterations:"));
    assert!(stdout.contains("work ops:"));
    assert!(stdout.contains("memo hits:"));
    assert!(stdout.contains("memo misses:"));
    assert!(stdout.contains("per-node stats:"));
}

#[test]
fn emit_c_super_powered_loop_zero_guard_skips_iterations() {
    let stdout = compile_and_run_emitted_backend(
        "[--].",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Super,
        false,
    );
    assert!(stdout.starts_with("0\n"));
}

#[test]
fn emit_c_super_powered_loop_stops_exactly_at_zero() {
    let stdout = compile_and_run_emitted_backend(
        "++++[--].",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Super,
        false,
    );
    assert!(stdout.starts_with("0\n"));
}

#[test]
fn emit_c_super_powered_loop_emits_overshoot_guard() {
    let c = emit_c_super(
        &parse_and_opt("[--]"),
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert!(CSource::parse(&c).has_syntax("remaining_iters = (uint64_t)guard / UINT64_C(2);"));
    assert!(CSource::parse(&c).has_syntax("((uint64_t)guard % UINT64_C(2)) == 0"));
    assert!(CSource::parse(&c).has_condition("while_statement", "tape[ptr] != 0"));
}

#[test]
fn emit_c_super_powered_loops_produce_memo_hits_on_reuse() {
    let stdout = compile_and_run_emitted_backend(
        "++++[--]++++[--]",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Super,
        false,
    );
    assert!(
        memo_hits(&stdout) > 0,
        "expected powered loop memo hits, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_repeated_powered_loops_hit_memo_multiple_times() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c_and_has_memo_hits("++++[--]>++++[--]>++++[--]<<.>.", opts, 2);
}

#[test]
fn emit_c_super_repeated_nested_powered_loops_hit_memo_multiple_times() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    assert_super_c_matches_plain_c_and_has_memo_hits(
        ">++++[--]<>>++++[--]<<>>++++[--]<<.>.",
        opts,
        2,
    );
}

#[test]
fn emit_c_super_reports_evictions_and_probe_stats_under_pressure() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let mut src = String::new();
    for count in 1..=12 {
        src.push_str("[-]");
        for _ in 0..(count * 2) {
            src.push('+');
        }
        src.push_str("[--]");
    }
    src.push_str("[-]++++[--][-]++++[--].");

    let stdout = compile_and_run_emitted_c_super_with_memo_capacity(&src, opts, 2);
    assert!(
        memo_stat(&stdout, "memo evictions:") > 0,
        "expected memo evictions under pressure, got:\n{stdout}"
    );
    assert!(
        memo_stat(&stdout, "memo lookup max probe:") > 0,
        "expected lookup probe stats under pressure, got:\n{stdout}"
    );
    assert!(
        memo_stat(&stdout, "memo store max probe:") > 0,
        "expected store probe stats under pressure, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_repeated_powered_loops_keep_hitting_under_small_memo_capacity() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let stdout = compile_and_run_emitted_c_super_with_memo_capacity(
        "[-]++++[--][-]++++[--][-]++++[--].",
        opts,
        4,
    );
    assert!(
        memo_hits(&stdout) >= 2,
        "expected repeated powered loops to keep producing hits with a tiny memo table, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_repeated_leaf_ops_produce_exact_memo_hits() {
    let program = (0..4)
        .flat_map(|_| [BfIr::Add(1), BfIr::Output, BfIr::Add(-1), BfIr::Output])
        .collect::<Vec<_>>();
    let stdout = compile_and_run_ir_backend(
        &program,
        CodegenOpts {
            io_mode: IoMode::Number,
            ..default_c_opts()
        },
        TestCBackend::Super,
    );
    assert_eq!(
        split_super_c_payload(&stdout).trim(),
        "1\n0\n1\n0\n1\n0\n1\n0"
    );
    assert!(
        memo_stat(&stdout, "memo hits:") > 0,
        "expected exact memo hits for repeated leaf ops, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_repeated_pure_seq_is_composed_into_one_operation() {
    let stdout = compile_and_run_emitted_backend(
        "++>+<".repeat(8).as_str(),
        default_c_opts(),
        TestCBackend::Super,
        false,
    );
    assert_eq!(
        memo_stat(&stdout, "work ops:"),
        1,
        "pure repeated sequence should be composed rather than replayed:\n{stdout}"
    );
    assert!(
        memo_stat(&stdout, "max recursion depth:") > 0,
        "expected recursive execution depth to be tracked, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_repeated_seq_regions_do_less_work_than_plain_c() {
    assert_super_c_does_less_work_than_plain_c(
        "++>+<++>+<++>+<++>+<++>+<++>+<++>+<++>+<",
        default_c_opts(),
        "work ops:",
    );
}

#[test]
fn emit_c_super_repeated_loop_regions_produce_exact_loop_hits() {
    let stdout = compile_and_run_emitted_backend(
        "++++[--]++++[--]++++[--]",
        default_c_opts(),
        TestCBackend::Super,
        false,
    );
    assert!(
        memo_stat(&stdout, "memo hits:") > 0,
        "expected exact memo hits for repeated loops, got:\n{stdout}"
    );
    assert!(
        memo_stat(&stdout, "symbolic power hits:") > 0,
        "expected powered execution to record symbolic power hits, got:\n{stdout}"
    );
}

#[test]
fn emit_c_super_static_power_loop_bypasses_dynamic_runtime_summary() {
    let mut program = Vec::new();
    for _ in 0..10 {
        program.extend([
            BfIr::Add(3),
            BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ]),
            BfIr::MovePtr(2),
        ]);
    }
    program.extend([BfIr::MovePtr(-19), BfIr::Output]);
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);
    assert_super_payload_matches_plain("static powered transfer loop", &plain, &super_out);
    assert_eq!(
        memo_stat(&super_out, "dynamic trace attempts:"),
        0,
        "{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic trace successes:"),
        0,
        "{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic fallback entries:"),
        0,
        "{super_out}"
    );
    assert!(
        memo_stat(&super_out, "symbolic power hits:") > 0,
        "{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "recursion fallbacks:"),
        0,
        "{super_out}"
    );
}

#[test]
fn emit_c_super_invariant_product_loop_uses_static_powered_summary() {
    let mut program = Vec::new();
    for _ in 0..10 {
        program.extend([
            BfIr::Add(3),
            BfIr::MovePtr(1),
            BfIr::Add(4),
            BfIr::MovePtr(-1),
            BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::Affine {
                    src: 1,
                    dst: 2,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
            ]),
            BfIr::MovePtr(4),
        ]);
    }
    program.extend([BfIr::MovePtr(-38), BfIr::Output]);
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(&program, opts, TestCBackend::Super);
    assert_super_payload_matches_plain("static powered product loop", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "12");
    assert_eq!(
        memo_stat(&super_out, "dynamic trace attempts:"),
        0,
        "{super_out}"
    );
    assert!(
        memo_stat(&super_out, "symbolic power hits:") > 0,
        "{super_out}"
    );
}

#[test]
fn emit_c_super_composes_square_child_summary_into_powered_loop() {
    let program = vec![
        BfIr::Add(5),
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(-1),
        BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::Square {
                src: 1,
                dst: 2,
                preserve_src: true,
                set_dst: true,
            },
        ]),
        BfIr::MovePtr(2),
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

    assert_super_payload_matches_plain("powered square child summary", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "9");
    assert!(
        memo_stat(&super_out, "symbolic power hits:") > 0,
        "degree-2 child summary should be powered:\n{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic trace attempts:"),
        0,
        "statically composed square should bypass runtime tracing:\n{super_out}"
    );
}

#[test]
fn emit_c_super_value_guarded_transducer_handles_noncomposable_square_loop() {
    let mut program = Vec::new();
    for _ in 0..8 {
        program.extend([
            BfIr::Add(2),
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
    program.extend([BfIr::MovePtr(-15), BfIr::Output]);
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let plain = compile_and_run_ir_backend(&program, opts, TestCBackend::Plain);
    let source = emit_c_super(&program, opts);
    let super_c = CSource::parse(&source)
        .replace_define("BF_TEMPLATE_DYNAMIC_HOT_THRESHOLD", "3")
        .or_invariant("unique hot threshold definition");
    let super_out = compile_and_run_c_source(&super_c);

    assert_super_payload_matches_plain("value-guarded square transducer", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "16");
    assert_eq!(
        memo_stat(&super_out, "dynamic trace attempts:"),
        1,
        "{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic trace rejections:"),
        1,
        "{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic transducer installs:"),
        1,
        "{super_out}"
    );
    assert!(
        memo_stat(&super_out, "dynamic transducer hits:") > 0,
        "installed transducer should handle a repeated guarded state:\n{super_out}"
    );
    assert_eq!(
        memo_stat(&super_out, "dynamic memo after rejection:"),
        0,
        "guarded transducer should precede general memoization:\n{super_out}"
    );
}

#[test]
fn emit_c_super_contiguous_zero_summary_wraps_memset_at_tape_end() {
    let program = vec![
        BfIr::MovePtr(-1),
        BfIr::Add(3),
        BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 0,
                preserve_src: true,
                set_dst: true,
            },
            BfIr::Affine {
                src: 0,
                dst: 2,
                coeff: 0,
                preserve_src: true,
                set_dst: true,
            },
        ]),
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
    let super_c = emit_c_super(&program, opts);
    assert!(
        CSource::parse(&super_c).has_call("bf_zero_region"),
        "contiguous proven clears should select the wrap-safe memset lowering"
    );
    let super_out = compile_and_run_c_source(&super_c);

    assert_super_payload_matches_plain("wrapped contiguous zero summary", &plain, &super_out);
    assert_eq!(split_super_c_payload(&super_out).trim(), "0\n0");
}
