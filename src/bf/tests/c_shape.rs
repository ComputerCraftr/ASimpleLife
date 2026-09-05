use super::*;

#[test]
fn generated_backends_parse_as_c_syntax_without_recovery() {
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let source = emit_backend_ir(&parse_and_opt(",++[->+<]>.[>]"), default_c_opts(), backend);
        let errors = CSource::parse(&source).errors();
        assert!(
            errors.is_empty(),
            "backend={backend:?} parse errors={errors:?}"
        );
    }
}

#[test]
fn super_c_sequence_memo_emits_lookup_store_and_a_stable_base_pointer() {
    let source = emit_c_super(&parse_and_opt("++>+<"), default_c_opts());
    let syntax = CSource::parse(&source);
    assert!(syntax.has_syntax("ptrdiff_t memo_base_ptr = ptr;"));
    assert!(syntax.has_syntax("bf_memo_lookup(&key, &value)"));
    assert!(syntax.has_syntax("bf_memo_store(&key, &value);"));
}

#[test]
fn plain_c_template_compiles_standalone_under_strict_c23_compatibility_mode() {
    let output = compile_and_run_c_template(TestCBackend::Plain);
    assert!(
        split_plain_c_payload(&output).is_empty(),
        "unexpected standalone template output: {output}"
    );
    assert!(output.contains(BF_PLAIN_STATS_SENTINEL));
    assert!(output.contains("work dispatches:"));
    assert!(output.contains("work loop iterations:"));
    assert!(output.contains("work ops:"));
}

#[test]
fn super_c_template_compiles_standalone_under_strict_c23_compatibility_mode() {
    let output = compile_and_run_c_template(TestCBackend::Super);
    assert!(
        output.contains("memo hits:"),
        "unexpected standalone super template output: {output}"
    );
}

#[test]
fn emit_c_signed_runtime_outputs_negative_value() {
    let stdout = compile_and_run_emitted_backend(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
        TestCBackend::Plain,
        false,
    );
    assert_eq!(split_plain_c_payload(&stdout), "-1\n");
}

#[test]
fn emit_c_char_only_program_omits_numeric_format_constants() {
    let c = emit_c(&parse_and_opt("+++."), default_c_opts());
    assert!(!CSource::parse(&c).has_identifier("BF_SCANF_FMT"));
    assert!(!CSource::parse(&c).has_identifier("BF_PRINTF_FMT"));
}

#[test]
fn emit_c_wraps_tape_pointer_moves() {
    let c = emit_c(&parse_and_opt(">>"), default_c_opts());
    assert!(CSource::parse(&c).has_identifier("bf_wrap_ptr"));
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_TAPE_LEN")
            .contains(&"30000")
    );
    assert!(CSource::parse(&c).has_syntax("ptr = bf_wrap_ptr(ptr, 2, BF_TAPE_LEN);"));
}

#[test]
fn emit_c_unsigned_runtime_executes_distribute_loop_correctly() {
    let stdout = compile_and_run_emitted_backend(
        "+++[->++<]>.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Plain,
        false,
    );
    assert_eq!(split_plain_c_payload(&stdout), "6\n");
}

#[test]
fn emit_c_unsigned_runtime_wraps_underflow_correctly() {
    let stdout = compile_and_run_emitted_backend(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Plain,
        false,
    );
    assert_eq!(split_plain_c_payload(&stdout), "255\n");
}

#[test]
fn emit_c_uses_unsigned_wrapping_when_requested() {
    let c = emit_c(
        &parse_and_opt("++[->+<]"),
        CodegenOpts {
            cell_sign: CellSign::Unsigned,
            ..default_c_opts()
        },
    );
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_SIGNED_CELLS")
            .contains(&"0")
    );
    assert!(CSource::parse(&c).has_identifier("bf_wrap_add_i64_unsigned"));
    assert!(CSource::parse(&c).has_identifier("bf_wrap_sub_i64_unsigned"));
    assert!(CSource::parse(&c).has_identifier("bf_wrap_mul_i64_unsigned"));
    assert!(CSource::parse(&c).has_identifier("bf_wrap_from_u64_unsigned"));
}

#[test]
fn emit_c_uses_requested_cell_width() {
    let c = emit_c(
        &parse_and_opt("+"),
        CodegenOpts {
            cell_bits: 63,
            ..default_c_opts()
        },
    );
    assert!(CSource::parse(&c).has_syntax("int64_t tape[BF_TAPE_LEN] = {0};"));
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_CELL_BITS")
            .contains(&"63")
    );
    assert!(!CSource::parse(&c).declares("BF_CELL_MASK"));
    assert!(CSource::parse(&c).has_syntax("ptrdiff_t ptr = 0;"));
}

#[test]
fn emit_c_applies_custom_char_masks() {
    let c = emit_c(
        &parse_and_opt(",."),
        CodegenOpts {
            cell_bits: 16,
            input_bits: Some(5),
            output_bits: Some(6),
            ..default_c_opts()
        },
    );
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_INPUT_MASK")
            .contains(&"UINT64_C(31)")
    );
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_OUTPUT_MASK")
            .contains(&"UINT64_C(63)")
    );
    assert!(CSource::parse(&c).has_syntax("BF_SIGNED_CELLS ? bf_wrap_from_u64_signed(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS) : bf_wrap_from_u64_unsigned(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS)"));
    assert!(
        CSource::parse(&c)
            .has_syntax("putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));")
    );
}

#[test]
fn emit_c_applies_custom_number_masks() {
    let c = emit_c(
        &parse_and_opt(",."),
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 32,
            input_bits: Some(3),
            output_bits: Some(4),
            cell_sign: CellSign::Signed,
        },
    );
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_INPUT_MASK")
            .contains(&"UINT64_C(7)")
    );
    assert!(
        CSource::parse(&c)
            .define_values("BF_TEMPLATE_OUTPUT_MASK")
            .contains(&"UINT64_C(15)")
    );
    assert!(CSource::parse(&c).has_syntax("{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"));
    assert!(
        CSource::parse(&c)
            .has_syntax("bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS)")
    );
    assert!(CSource::parse(&c).has_syntax("printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"));
}

#[test]
fn emit_c_runtime_wraps_pointer_moves_correctly() {
    let stdout = compile_and_run_emitted_backend(
        "<+.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Plain,
        false,
    );
    assert_eq!(split_plain_c_payload(&stdout), "1\n");
}

#[test]
#[cfg(not(target_os = "windows"))]
fn emit_c_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_backend(
        "+++[->++<]>.<.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Plain,
        true,
    );
    assert_eq!(split_plain_c_payload(&stdout), "6\n0\n");
}

#[test]
#[cfg(not(target_os = "windows"))]
fn emit_c_signed_63bit_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_backend(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 63,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
        TestCBackend::Plain,
        true,
    );
    assert_eq!(split_plain_c_payload(&stdout), "-1\n");
}

#[test]
fn emit_c_for_empty_loop_stays_guarded() {
    let c = emit_c(&parse_and_opt("[]"), default_c_opts());
    assert!(CSource::parse(&c).has_condition("while_statement", "tape[ptr] != 0"));
    assert!(CSource::parse(&c).has_syntax("bf_diverge_forever();"));
}
