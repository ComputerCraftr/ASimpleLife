use super::c_backend::{emit_c, format_ir, interpret_for_tests, interpret_unsigned_for_tests};
use super::c_super_backend::emit_c_super;
use super::cli::{parse_opts, read_input};
use super::ir::{BfIr, Parser};
use super::optimizer::{CellSign, CodegenOpts, IoMode, optimize};
use std::fs;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

mod c_super_backend;
mod life;
mod optimizer;

static BF_TEST_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn parse_and_opt(src: &str) -> Vec<BfIr> {
    optimize(Parser::new(src).parse().unwrap())
}

fn parse_only(src: &str) -> Vec<BfIr> {
    Parser::new(src).parse().unwrap()
}

fn default_c_opts() -> CodegenOpts {
    CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    }
}

fn parse_opt_and_emit_c(src: &str) -> String {
    emit_c(&parse_and_opt(src), default_c_opts())
}

fn life_opts() -> CodegenOpts {
    CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    }
}

fn assert_optimized_matches_unoptimized_unsigned(src: &str) {
    let parsed = parse_only(src);
    let optimized = optimize(parsed.clone());
    let parsed_result = interpret_unsigned_for_tests(&parsed, life_opts().cell_bits);
    let optimized_result = interpret_unsigned_for_tests(&optimized, life_opts().cell_bits);
    match (optimized_result, parsed_result) {
        (Ok((optimized_tape, optimized_ptr)), Ok((parsed_tape, parsed_ptr))) => {
            assert_eq!(optimized_ptr, parsed_ptr, "pointer mismatch for {src}");
            assert_eq!(optimized_tape, parsed_tape, "tape mismatch for {src}");
        }
        (Err(_), Err(_)) => {}
        (optimized, parsed) => {
            panic!("semantic mismatch for {src}: optimized={optimized:?} parsed={parsed:?}");
        }
    }
}

fn signed_test_opts() -> CodegenOpts {
    CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    }
}

fn assert_optimized_matches_unoptimized_with_opts(src: &str, opts: CodegenOpts) {
    let parsed = parse_only(src);
    let optimized = optimize(parsed.clone());
    let parsed_result = interpret_for_tests(&parsed, opts);
    let optimized_result = interpret_for_tests(&optimized, opts);
    match (optimized_result, parsed_result) {
        (Ok((optimized_tape, optimized_ptr)), Ok((parsed_tape, parsed_ptr))) => {
            assert_eq!(optimized_ptr, parsed_ptr, "pointer mismatch for {src}");
            assert_eq!(optimized_tape, parsed_tape, "tape mismatch for {src}");
        }
        (Err(_), Err(_)) => {}
        (optimized, parsed) => {
            panic!("semantic mismatch for {src}: optimized={optimized:?} parsed={parsed:?}");
        }
    }
}

fn compile_and_run_emitted_c(src: &str, opts: CodegenOpts) -> String {
    let c = emit_c(&parse_and_opt(src), opts);
    compile_and_run_c_source(&c)
}

fn compile_and_run_emitted_c_super(src: &str, opts: CodegenOpts) -> String {
    let c = emit_c_super(&parse_and_opt(src), opts);
    compile_and_run_c_source(&c)
}

fn compile_and_run_emitted_c_sanitized(src: &str, opts: CodegenOpts) -> String {
    let c = emit_c(&parse_and_opt(src), opts);
    compile_and_run_c_source_sanitized(&c)
}

fn compile_and_run_emitted_c_super_sanitized(src: &str, opts: CodegenOpts) -> String {
    let c = emit_c_super(&parse_and_opt(src), opts);
    compile_and_run_c_source_sanitized(&c)
}

fn assert_super_c_matches_plain_c(src: &str, opts: CodegenOpts) {
    let plain = compile_and_run_emitted_c(src, opts);
    let super_out = compile_and_run_emitted_c_super(src, opts);
    let super_payload = super_out
        .lines()
        .take_while(|line| !line.starts_with("memo hits:"))
        .collect::<Vec<_>>()
        .join("\n");
    let plain_trimmed = plain.trim_end();
    let super_trimmed = super_payload.trim_end();
    assert_eq!(
        super_trimmed, plain_trimmed,
        "super C output mismatch for {src}\nplain:\n{plain}\nsuper:\n{super_out}"
    );
}

fn memo_hits(output: &str) -> u64 {
    memo_stat(output, "memo hits:")
}

fn memo_stat(output: &str, prefix: &str) -> u64 {
    output
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .map(|rest| rest.trim().parse::<u64>().unwrap())
        .unwrap_or_else(|| panic!("missing memo stat line: {prefix}"))
}

fn assert_super_c_matches_plain_c_and_has_memo_hits(src: &str, opts: CodegenOpts, min_hits: u64) {
    let plain = compile_and_run_emitted_c(src, opts);
    let super_out = compile_and_run_emitted_c_super(src, opts);
    let super_payload = super_out
        .lines()
        .take_while(|line| !line.starts_with("memo hits:"))
        .collect::<Vec<_>>()
        .join("\n");
    let plain_trimmed = plain.trim_end();
    let super_trimmed = super_payload.trim_end();
    assert_eq!(
        super_trimmed, plain_trimmed,
        "super C output mismatch for {src}\nplain:\n{plain}\nsuper:\n{super_out}"
    );
    let hits = memo_hits(&super_out);
    assert!(
        hits >= min_hits,
        "expected at least {min_hits} memo hits for {src}, got {hits}\n{super_out}"
    );
}

fn compile_and_run_c_source(c: &str) -> String {
    compile_and_run_c_source_with_args(c, &[])
}

fn compile_and_run_c_template(path: &str) -> String {
    compile_and_run_c_source(&fs::read_to_string(path).unwrap())
}

fn compile_and_run_c_source_sanitized(c: &str) -> String {
    compile_and_run_c_source_with_args(
        c,
        &[
            "-g3",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined",
        ],
    )
}

fn compile_and_run_c_source_with_args(c: &str, extra_cc_args: &[&str]) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = BF_TEST_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!(
        "a_simple_life_bf_{}_{}_{}",
        std::process::id(),
        timestamp,
        counter
    ));
    fs::create_dir_all(&base).unwrap();
    let source = base.join("program.c");
    let binary = base.join("program.bin");
    fs::write(&source, c).unwrap();

    let mut compile_cmd = Command::new("cc");
    compile_cmd
        .arg("-std=c23")
        .arg("-O0")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wpedantic")
        .arg("-Werror");
    compile_cmd.args(extra_cc_args);
    compile_cmd.arg(&source).arg("-o").arg(&binary);
    let compile = compile_cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "cc failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let output = Command::new(&binary)
        .stdin(Stdio::null())
        .env(
            "ASAN_OPTIONS",
            "detect_leaks=0:halt_on_error=1:abort_on_error=1",
        )
        .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "program failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn parser_accepts_inline_source() {
    assert_eq!(
        parse_only("+++"),
        vec![BfIr::Add(1), BfIr::Add(1), BfIr::Add(1)]
    );
}

#[test]
fn parser_ignores_non_bf_cli_spacing() {
    assert_eq!(
        parse_only("+ + +"),
        vec![BfIr::Add(1), BfIr::Add(1), BfIr::Add(1)]
    );
}

#[test]
fn format_inline_source_ir() {
    assert_eq!(
        format_ir(&parse_only("+++[->+<]")),
        "Add(1)\nAdd(1)\nAdd(1)\nLoop {\n  Add(-1)\n  MovePtr(1)\n  Add(1)\n  MovePtr(-1)\n}\n"
    );
}

#[test]
fn unmatched_right_bracket_errors() {
    assert!(
        Parser::new("]")
            .parse()
            .unwrap_err()
            .contains("unmatched ']'")
    );
}

#[test]
fn unmatched_left_bracket_errors() {
    assert!(
        Parser::new("[")
            .parse()
            .unwrap_err()
            .contains("unmatched '['")
    );
}

#[test]
fn plain_c_template_compiles_standalone_under_strict_c23() {
    let output = compile_and_run_c_template("src/bf/bf.c.in");
    assert!(
        output.is_empty(),
        "unexpected standalone template output: {output}"
    );
}

#[test]
fn super_c_template_compiles_standalone_under_strict_c23() {
    let output = compile_and_run_c_template("src/bf/bf_super.c.in");
    assert!(
        output.contains("memo hits:"),
        "unexpected standalone super template output: {output}"
    );
}

#[test]
fn emit_c_signed_runtime_outputs_negative_value() {
    let stdout = compile_and_run_emitted_c(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
    );
    assert_eq!(stdout, "-1\n");
}

#[test]
fn emit_c_char_only_program_omits_numeric_format_constants() {
    let c = parse_opt_and_emit_c("+++.");
    assert!(!c.contains("BF_SCANF_FMT"));
    assert!(!c.contains("BF_PRINTF_FMT"));
}

#[test]
fn emit_c_wraps_tape_pointer_moves() {
    let c = emit_c(&parse_and_opt(">>"), default_c_opts());
    assert!(
        c.contains("static ptrdiff_t bf_wrap_ptr(ptrdiff_t ptr, ptrdiff_t delta, ptrdiff_t len) {")
    );
    assert!(c.contains("#define BF_TEMPLATE_TAPE_LEN 30000"));
    assert!(c.contains("ptr = bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN);"));
}

#[test]
fn emit_c_unsigned_runtime_executes_distribute_loop_correctly() {
    let stdout = compile_and_run_emitted_c(
        "+++[->++<]>.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert_eq!(stdout, "6\n");
}

#[test]
fn emit_c_unsigned_runtime_wraps_underflow_correctly() {
    let stdout = compile_and_run_emitted_c(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert_eq!(stdout, "255\n");
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
    assert!(c.contains("#define BF_TEMPLATE_SIGNED_CELLS 0"));
    assert!(c.contains("bf_wrap_add_i64_unsigned"));
    assert!(c.contains("bf_wrap_sub_i64_unsigned"));
    assert!(c.contains("bf_wrap_mul_i64_unsigned"));
    assert!(c.contains("bf_wrap_from_u64_unsigned"));
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
    assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {};"));
    assert!(c.contains("#define BF_TEMPLATE_CELL_BITS 63"));
    assert!(!c.contains("BF_CELL_MASK = "));
    assert!(c.contains("ptrdiff_t ptr = 0;"));
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
    assert!(c.contains("#define BF_TEMPLATE_INPUT_MASK UINT64_C(31)"));
    assert!(c.contains("#define BF_TEMPLATE_OUTPUT_MASK UINT64_C(63)"));
    assert!(c.contains("BF_SIGNED_CELLS ? bf_wrap_from_u64_signed(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS) : bf_wrap_from_u64_unsigned(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS)"));
    assert!(c.contains("putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"));
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
    assert!(c.contains("#define BF_TEMPLATE_INPUT_MASK UINT64_C(7)"));
    assert!(c.contains("#define BF_TEMPLATE_OUTPUT_MASK UINT64_C(15)"));
    assert!(c.contains("{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"));
    assert!(c.contains("bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS)"));
    assert!(c.contains("printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"));
}

#[test]
fn emit_c_runtime_wraps_pointer_moves_correctly() {
    let stdout = compile_and_run_emitted_c(
        "<+.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert_eq!(stdout, "1\n");
}

#[test]
fn emit_c_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_c_sanitized(
        "+++[->++<]>.<.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    );
    assert_eq!(stdout, "6\n0\n");
}

#[test]
fn emit_c_signed_63bit_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_c_sanitized(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 63,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
    );
    assert_eq!(stdout, "-1\n");
}

#[test]
fn emit_c_for_empty_loop_stays_guarded() {
    let c = parse_opt_and_emit_c("[]");
    assert!(c.contains("while (tape[ptr] != 0) {"));
    assert!(c.contains("bf_diverge_forever();"));
}

#[test]
fn parse_opts_rejects_64_bit_io_masks() {
    let args = vec!["--input-bits".to_string(), "64".to_string()];
    assert!(parse_opts(&args).unwrap_err().contains("expected 0..=63"));
    let args = vec!["--output-bits".to_string(), "64".to_string()];
    assert!(parse_opts(&args).unwrap_err().contains("expected 0..=63"));
}

#[test]
fn read_input_accepts_emit_ir_alias_without_treating_it_as_source() {
    let args = vec![
        "--emit-ir".to_string(),
        "--".to_string(),
        "+++[->++<]>.<.".to_string(),
    ];
    let (mode, opts, src) = read_input(&args).unwrap();
    assert_eq!(mode, super::cli::OutputMode::EmitIr);
    assert_eq!(opts, life_opts());
    assert_eq!(src, "+++[->++<]>.<.");
}

#[test]
fn read_input_accepts_emit_life_hashlife_mode() {
    let args = vec![
        "--emit-life-hashlife".to_string(),
        "--signed-cells".to_string(),
        "false".to_string(),
        "--".to_string(),
        "+.".to_string(),
    ];
    let (mode, opts, src) = read_input(&args).unwrap();
    assert!(matches!(mode, super::cli::OutputMode::EmitLifeHashLife));
    assert_eq!(opts.cell_sign, CellSign::Unsigned);
    assert_eq!(src, "+.");
}

#[test]
fn emit_ir_alias_matches_dump_ir_for_distribution_program() {
    let parsed = parse_only("+++[->++<]>.<.");
    let optimized = optimize(parsed.clone());
    let expected = "=== Parsed IR ===\n".to_string()
        + &format_ir(&parsed)
        + "=== Optimized IR ===\n"
        + &format_ir(&optimized);
    assert_eq!(
        expected,
        "=== Parsed IR ===\nAdd(1)\nAdd(1)\nAdd(1)\nLoop {\n  Add(-1)\n  MovePtr(1)\n  Add(1)\n  Add(1)\n  MovePtr(-1)\n}\nMovePtr(1)\nOutput\nMovePtr(-1)\nOutput\n=== Optimized IR ===\nAdd(3)\nDistribute { targets: [(1, 2)] }\nMovePtr(1)\nOutput\nMovePtr(-1)\nOutput\n"
    );
}
