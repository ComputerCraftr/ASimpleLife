use super::*;

#[test]
fn emit_c_super_contains_symbolic_memo_runtime() {
    let c = emit_c_super(&parse_and_opt("+++[->+<]>+."), default_c_opts());
    assert!(c.contains("typedef struct {"));
    assert!(c.contains("MemoKey"));
    assert!(c.contains("bf_memo_lookup"));
    assert!(c.contains("print_memo_stats"));
    assert!(c.contains("memo hits:"));
    assert!(c.contains("memo evictions:"));
    assert!(c.contains("memo lookup max probe:"));
    assert!(c.contains("memo store max probe:"));
    assert!(c.contains("node_hits[MAX_NODES]"));
    assert!(c.contains(" cost="));
    assert!(c.contains("Distribute([(1, 1)]) plan=DirectKernel(\"distribute\")"));
    assert!(c.contains("Seq([NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)]) plan=Residual"));
}

fn compile_and_run_emitted_c_super_with_memo_capacity(
    src: &str,
    opts: CodegenOpts,
    memo_capacity: usize,
) -> String {
    let c = emit_c_super(&parse_and_opt(src), opts);
    let c = c.replacen(
        "#define BF_TEMPLATE_MEMO_CAPACITY 4096",
        &format!("#define BF_TEMPLATE_MEMO_CAPACITY {memo_capacity}"),
        1,
    );
    compile_and_run_c_source(&c)
}

#[test]
fn emit_c_super_can_plan_symbolic_loop_regions() {
    let c = emit_c_super(&parse_and_opt("[->+<+]"), default_c_opts());
    assert!(c.contains("Seq([NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(2)]) plan=SymbolicMemo"));
    assert!(c.contains("Loop(NodeId(4)) plan=DirectKernel(\"loop_kernel\")"));
    assert!(c.contains("while (tape[ptr] != 0) {"));
}

#[test]
fn emit_c_super_can_plan_powered_loop_regions() {
    let c = emit_c_super(&parse_and_opt("[--]"), default_c_opts());
    assert!(c.contains("Loop(NodeId(0)) plan=PoweredSymbolicLoop"));
    assert!(c.contains("remaining_iters"));
    assert!(c.contains("powered_ok"));
}

#[test]
fn emit_c_super_refuses_powered_loop_for_distribute_style_body() {
    let c = emit_c_super(&parse_and_opt("[->+<]"), default_c_opts());
    assert!(!c.contains("plan=PoweredSymbolicLoop"));
}

#[test]
fn emit_c_super_refuses_powered_loop_for_io_body() {
    let c = emit_c_super(
        &parse_and_opt("[,.]"),
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert!(!c.contains("plan=PoweredSymbolicLoop"));
}

#[test]
fn emit_c_super_can_plan_symbolic_seq_regions() {
    let c = emit_c_super(&parse_and_opt("++>+<"), default_c_opts());
    assert!(c.contains("Seq([NodeId(0), NodeId(1), NodeId(2), NodeId(3)]) plan=SymbolicMemo"));
    assert!(c.contains("int64_t cell_0 = tape[bf_wrap_ptr(ptr, (ptrdiff_t)(0), BF_TAPE_LEN)];"));
    assert!(c.contains("int64_t cell_1 = tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)];"));
    assert!(c.contains(
        "tape[bf_wrap_ptr(ptr, (ptrdiff_t)(0), BF_TAPE_LEN)] = bf_wrap_add(cell_0, INT64_C(2), BF_CELL_BITS, BF_SIGNED_CELLS);"
    ));
    assert!(c.contains(
        "tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)] = bf_wrap_add(cell_1, INT64_C(1), BF_CELL_BITS, BF_SIGNED_CELLS);"
    ));
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
    let stdout = compile_and_run_emitted_c_super(
        "+++[->++<]>.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert!(stdout.starts_with("6\n"));
    assert!(stdout.contains("memo hits:"));
    assert!(stdout.contains("memo misses:"));
    assert!(stdout.contains("per-node stats:"));
}

#[test]
fn emit_c_super_powered_loop_zero_guard_skips_iterations() {
    let stdout = compile_and_run_emitted_c_super(
        "[--].",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert!(stdout.starts_with("0\n"));
}

#[test]
fn emit_c_super_powered_loop_stops_exactly_at_zero() {
    let stdout = compile_and_run_emitted_c_super(
        "++++[--].",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
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
    assert!(c.contains("remaining_iters = (uint64_t)guard / UINT64_C(2);"));
    assert!(c.contains("((uint64_t)guard % UINT64_C(2)) == 0"));
    assert!(c.contains("while (tape[ptr] != 0) {"));
}

#[test]
fn emit_c_super_powered_loops_produce_memo_hits_on_reuse() {
    let stdout = compile_and_run_emitted_c_super(
        "++++[--]++++[--]",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert!(memo_hits(&stdout) > 0, "expected powered loop memo hits, got:\n{stdout}");
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
    assert_super_c_matches_plain_c_and_has_memo_hits(
        "++++[--]>++++[--]>++++[--]<<.>.",
        opts,
        2,
    );
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
    assert!(memo_hits(&stdout) > 0, "expected memo hits under pressure, got:\n{stdout}");
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

#[test]
fn emit_c_super_signed_runtime_outputs_negative_value() {
    let stdout = compile_and_run_emitted_c_super(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
    );
    assert!(stdout.starts_with("-1\n"));
    assert!(stdout.contains("memo hits:"));
    assert!(stdout.contains("memo misses:"));
}

#[test]
fn emit_c_super_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_c_super_sanitized(
        "++++[--]++++[--]",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert!(stdout.contains("memo hits:"));
    assert!(stdout.contains("memo evictions:"));
}

#[test]
fn emit_c_super_signed_63bit_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_c_super_sanitized(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 63,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
    );
    assert!(stdout.starts_with("-1\n"));
}
