use super::c_backend::{
    BfEvalError, emit_c, format_ir, interpret_for_tests, interpret_unsigned_for_tests,
};
use super::c_super_backend::emit_c_super;
use super::cli::{parse_opts, read_input};
use super::ir::{BfIr, Parser, ShiftDir, validate_canonical_ir};
use super::ir_report::{
    IrOutputFormat, IrRenderOpts, IrSectionSelection, build_ir_report, render_ir_json,
    render_ir_text,
};
use super::optimizer::{CellSign, CodegenOpts, IoMode, optimize_with_opts};
use super::summary::{
    LoopId, LoopSummary, OffsetOp, SummaryEffect, SummaryProvenance, normalize_offset_body,
};
use crate::{RequiredErrorExt, RequiredExt};
use std::fs;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

mod c_super_backend;
mod c_super_runtime;
mod ir_report;
mod life;
mod life_rich;
mod optimizer;
mod payload;

static BF_TEST_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const BF_PLAIN_STATS_SENTINEL: &str = "=== BF PLAIN RUNTIME STATS ===";
const BF_SUPER_STATS_SENTINEL: &str = "=== BF SUPER RUNTIME STATS ===";
const BF_TEST_PROGRAM_TIMEOUT: Duration = Duration::from_secs(60);
const BF_TEST_DEFAULT_WORK_LIMIT: &str = "-DBF_TEST_WORK_LIMIT=100000000";

#[derive(Clone, Copy, Debug)]
enum TestCBackend {
    Plain,
    Super,
}

#[derive(Debug, PartialEq, Eq)]
enum GeneratedCOutcome {
    Terminated {
        stdout: Vec<u8>,
    },
    WorkLimitReached {
        observable_prefix: Vec<u8>,
    },
    SemanticError {
        exit_code: Option<i32>,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
}

fn generated_c_outcome(output: std::process::Output) -> GeneratedCOutcome {
    if output.status.success() {
        return GeneratedCOutcome::Terminated {
            stdout: output.stdout,
        };
    }
    if output.status.code() == Some(124)
        && (String::from_utf8_lossy(&output.stderr).contains("BF_TEST_WORK_LIMIT_REACHED")
            || String::from_utf8_lossy(&output.stderr).contains("BF_TEST_SEMANTIC_FUEL_REACHED"))
    {
        return GeneratedCOutcome::WorkLimitReached {
            observable_prefix: output.stdout,
        };
    }
    GeneratedCOutcome::SemanticError {
        exit_code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    }
}

fn assert_panics_quietly<F>(f: F)
where
    F: FnOnce() + std::panic::UnwindSafe,
{
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    let _guard = PANIC_HOOK_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .or_invariant("required value");
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    assert!(
        result.is_err(),
        "expected panic, but code completed successfully"
    );
}

fn parse_and_opt(src: &str) -> Vec<BfIr> {
    optimize_with_opts(
        Parser::new(src).parse().or_invariant("required value"),
        life_opts(),
    )
}

fn parse_and_opt_with_opts(src: &str, opts: CodegenOpts) -> Vec<BfIr> {
    optimize_with_opts(
        Parser::new(src).parse().or_invariant("required value"),
        opts,
    )
}

fn parse_only(src: &str) -> Vec<BfIr> {
    Parser::new(src).parse().or_invariant("required value")
}

#[test]
fn shared_offset_normalizer_combines_pointer_relative_adds() {
    let body = parse_only(">+>++<<-");
    let normalized = normalize_offset_body(&body);
    assert_eq!(normalized.net_pointer_delta, 0);
    assert_eq!(
        normalized.ops,
        vec![
            OffsetOp::AddAt {
                offset: 0,
                delta: -1,
            },
            OffsetOp::AddAt {
                offset: 1,
                delta: 1,
            },
            OffsetOp::AddAt {
                offset: 2,
                delta: 2,
            },
        ]
    );
    assert_eq!(normalized.touched_offsets, vec![0, 1, 2]);
}

#[test]
fn shared_offset_normalizer_fails_closed_on_pointer_overflow() {
    let normalized = normalize_offset_body(&[
        BfIr::MovePtr(crate::bf::BfOffset::MAX),
        BfIr::MovePtr(1),
        BfIr::Add(1),
    ]);

    assert_eq!(normalized.ops, vec![OffsetOp::OpaqueLoop]);
    assert_eq!(normalized.net_pointer_delta, 0);
}

#[test]
fn shared_loop_summary_roundtrips_affine_and_product_effects() {
    let nodes = vec![
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 3,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 2,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        },
    ];
    let summary = LoopSummary::from_ir_nodes(LoopId(17), SummaryProvenance::Static, &nodes)
        .or_invariant("required value");
    assert_eq!(summary.touched_offsets, vec![0, 1, 2]);
    assert!(matches!(
        summary.effects[0],
        SummaryEffect::AddScaled { .. }
    ));
    assert!(matches!(
        summary.effects[1],
        SummaryEffect::AddProduct { .. }
    ));
    assert!(summary.validate_runtime().is_ok());
    assert_eq!(summary.lower_to_ir().or_invariant("required value"), nodes);
}

#[test]
fn shared_loop_summary_rebase_updates_effects_and_metadata() {
    let original = LoopSummary::from_ir_nodes(
        LoopId(18),
        SummaryProvenance::Static,
        &[BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 2,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        }],
    )
    .or_invariant("required value");
    let mut rebased = original.clone();
    rebased.rebase(7).or_invariant("required value");
    assert_eq!(rebased.touched_offsets, vec![7, 8, 9]);
    assert!(matches!(
        rebased.effects.as_slice(),
        [SummaryEffect::AddProduct {
            lhs: 7,
            rhs: 8,
            dst: 9,
            ..
        }]
    ));

    let mut composed = original;
    composed.rebase(3).or_invariant("required value");
    composed.rebase(4).or_invariant("required value");
    assert_eq!(composed, rebased);
}

#[test]
fn static_summary_validation_is_not_limited_by_runtime_window_capacity() {
    let summary = LoopSummary::from_ir_nodes(
        LoopId(3),
        SummaryProvenance::Static,
        &[BfIr::Affine {
            src: 0,
            dst: 12,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        }],
    )
    .or_invariant("required value");
    assert!(summary.validate().is_ok());
    assert!(
        summary
            .validate_runtime()
            .error_or_invariant("expected error")
            .contains("runtime maximum")
    );
}

#[test]
fn shared_summary_validation_rejects_understated_effect_metadata() {
    let mut summary = LoopSummary::from_ir_nodes(
        LoopId(4),
        SummaryProvenance::Static,
        &[BfIr::Square {
            src: 1,
            dst: 2,
            preserve_src: true,
            set_dst: true,
        }],
    )
    .or_invariant("required value");
    summary.read_offsets.clear();
    summary.touched_offsets = vec![2];

    assert_eq!(
        summary.validate().error_or_invariant("expected error"),
        "summary offset metadata does not match its effects"
    );
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
    let optimized = optimize_with_opts(parsed.clone(), life_opts());
    let parsed_result = interpret_unsigned_for_tests(&parsed, life_opts().cell_bits);
    let optimized_result = interpret_unsigned_for_tests(&optimized, life_opts().cell_bits);
    match (optimized_result, parsed_result) {
        (Ok((optimized_tape, optimized_ptr)), Ok((parsed_tape, parsed_ptr))) => {
            assert_eq!(optimized_ptr, parsed_ptr, "pointer mismatch for {src}");
            assert_eq!(optimized_tape, parsed_tape, "tape mismatch for {src}");
        }
        (Err(optimized), Err(parsed)) => {
            crate::invariant_failure!(
                "semantic equivalence test for {src} did not produce observable states: optimized={optimized:?} parsed={parsed:?}"
            );
        }
        (optimized, parsed) => {
            crate::invariant_failure!(
                "semantic mismatch for {src}: optimized={optimized:?} parsed={parsed:?}"
            );
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
    let optimized = optimize_with_opts(parsed.clone(), opts);
    let parsed_result = interpret_for_tests(&parsed, opts);
    let optimized_result = interpret_for_tests(&optimized, opts);
    match (optimized_result, parsed_result) {
        (Ok((optimized_tape, optimized_ptr)), Ok((parsed_tape, parsed_ptr))) => {
            assert_eq!(optimized_ptr, parsed_ptr, "pointer mismatch for {src}");
            assert_eq!(optimized_tape, parsed_tape, "tape mismatch for {src}");
        }
        (Err(optimized), Err(parsed)) => {
            crate::invariant_failure!(
                "semantic equivalence test for {src} did not produce observable states: optimized={optimized:?} parsed={parsed:?}"
            );
        }
        (optimized, parsed) => {
            crate::invariant_failure!(
                "semantic mismatch for {src}: optimized={optimized:?} parsed={parsed:?}"
            );
        }
    }
}

fn emit_backend_ir(program: &[BfIr], opts: CodegenOpts, backend: TestCBackend) -> String {
    match backend {
        TestCBackend::Plain => emit_c(program, opts),
        TestCBackend::Super => emit_c_super(program, opts),
    }
}

fn compile_and_run_emitted_backend(
    src: &str,
    opts: CodegenOpts,
    backend: TestCBackend,
    sanitized: bool,
) -> String {
    let c = emit_backend_ir(&parse_and_opt_with_opts(src, opts), opts, backend);
    if sanitized {
        compile_and_run_c_source_sanitized(&c)
    } else {
        compile_and_run_c_source(&c)
    }
}

fn compile_and_run_ir_backend(
    program: &[BfIr],
    opts: CodegenOpts,
    backend: TestCBackend,
) -> String {
    compile_and_run_c_source(&emit_backend_ir(program, opts, backend))
}

fn assert_super_payload_matches_plain(label: &str, plain: &str, super_out: &str) {
    assert_eq!(
        split_super_c_payload(super_out).trim_end(),
        split_plain_c_payload(plain).trim_end(),
        "super C output mismatch for {label}\nplain:\n{plain}\nsuper:\n{super_out}"
    );
}

fn assert_super_c_matches_plain_c_ir(label: &str, program: &[BfIr], opts: CodegenOpts) {
    let plain = compile_and_run_ir_backend(program, opts, TestCBackend::Plain);
    let super_out = compile_and_run_ir_backend(program, opts, TestCBackend::Super);
    assert_super_payload_matches_plain(label, &plain, &super_out);
}

fn assert_super_c_matches_plain_c(src: &str, opts: CodegenOpts) {
    let plain = compile_and_run_emitted_backend(src, opts, TestCBackend::Plain, false);
    let super_out = compile_and_run_emitted_backend(src, opts, TestCBackend::Super, false);
    assert_super_payload_matches_plain(src, &plain, &super_out);
}

fn memo_hits(output: &str) -> u64 {
    memo_stat(output, "memo hits:")
}

fn split_plain_c_payload(output: &str) -> &str {
    output
        .split_once(BF_PLAIN_STATS_SENTINEL)
        .map(|(payload, _)| payload)
        .unwrap_or(output)
}

fn split_plain_c_stats(output: &str) -> &str {
    output
        .split_once(BF_PLAIN_STATS_SENTINEL)
        .map(|(_, stats)| stats)
        .unwrap_or("")
}

fn split_super_c_payload(output: &str) -> &str {
    output
        .split_once(BF_SUPER_STATS_SENTINEL)
        .map(|(payload, _)| payload)
        .unwrap_or(output)
}

fn split_super_c_stats(output: &str) -> &str {
    output
        .split_once(BF_SUPER_STATS_SENTINEL)
        .map(|(_, stats)| stats)
        .unwrap_or("")
}

fn plain_stat(output: &str, prefix: &str) -> u64 {
    split_plain_c_stats(output)
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .map(|rest| rest.trim().parse::<u64>().or_invariant("required value"))
        .unwrap_or_else(|| crate::invariant_failure!("missing plain stat line: {prefix}"))
}

fn memo_stat(output: &str, prefix: &str) -> u64 {
    split_super_c_stats(output)
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .map(|rest| rest.trim().parse::<u64>().or_invariant("required value"))
        .unwrap_or_else(|| crate::invariant_failure!("missing memo stat line: {prefix}"))
}

fn super_work_stat(output: &str, prefix: &str) -> u64 {
    memo_stat(output, prefix)
}

fn assert_super_c_does_less_work_than_plain_c(src: &str, opts: CodegenOpts, work_prefix: &str) {
    let plain = compile_and_run_emitted_backend_optimized(src, opts, TestCBackend::Plain);
    let super_out = compile_and_run_emitted_backend_optimized(src, opts, TestCBackend::Super);
    assert_super_payload_matches_plain(src, &plain, &super_out);
    let plain_work = plain_stat(&plain, work_prefix);
    let super_work = super_work_stat(&super_out, work_prefix);
    assert!(
        super_work < plain_work,
        "expected super C to reduce {work_prefix} for {src}, plain={plain_work} super={super_work}\nplain:\n{plain}\nsuper:\n{super_out}"
    );
}

fn compile_and_run_emitted_backend_optimized(
    src: &str,
    opts: CodegenOpts,
    backend: TestCBackend,
) -> String {
    let c = emit_backend_ir(&parse_and_opt_with_opts(src, opts), opts, backend);
    compile_and_run_c_source_with_args(&c, &["-O3"])
}

fn assert_super_c_matches_plain_c_and_has_memo_hits(src: &str, opts: CodegenOpts, min_hits: u64) {
    let plain = compile_and_run_emitted_backend(src, opts, TestCBackend::Plain, false);
    let super_out = compile_and_run_emitted_backend(src, opts, TestCBackend::Super, false);
    assert_super_payload_matches_plain(src, &plain, &super_out);
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
    compile_and_run_c_source(&fs::read_to_string(path).or_invariant("required value"))
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
    let output = compile_and_run_c_source_capture(c, extra_cc_args);
    assert!(
        output.status.success(),
        "program failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).or_invariant("required value")
}

fn compile_and_run_c_source_capture(c: &str, extra_cc_args: &[&str]) -> std::process::Output {
    static C_EXECUTION_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = C_EXECUTION_LOCK.get_or_init(|| Mutex::new(()));
    let _execution_guard = match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .or_invariant("required value")
        .as_nanos();
    let counter = BF_TEST_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!(
        "a_simple_life_bf_{}_{}_{}",
        std::process::id(),
        timestamp,
        counter
    ));
    fs::create_dir_all(&base).or_invariant("required value");
    let source = base.join("program.c");
    let binary = base.join("program.bin");
    fs::write(&source, c).or_invariant("required value");

    let mut compile_cmd = Command::new("cc");
    compile_cmd
        .arg("-std=c2x")
        .arg("-O0")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wpedantic")
        .arg("-Werror");
    if !extra_cc_args
        .iter()
        .any(|arg| arg.starts_with("-DBF_TEST_WORK_LIMIT="))
    {
        compile_cmd.arg(BF_TEST_DEFAULT_WORK_LIMIT);
    }
    compile_cmd.args(extra_cc_args);
    compile_cmd.arg(&source).arg("-o").arg(&binary);
    let compile = compile_cmd.output().or_invariant("required value");
    assert!(
        compile.status.success(),
        "cc failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let mut child = Command::new(&binary)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env(
            "ASAN_OPTIONS",
            "detect_leaks=0:halt_on_error=1:abort_on_error=1",
        )
        .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
        .spawn()
        .or_invariant("required value");
    let deadline = Instant::now() + BF_TEST_PROGRAM_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().or_invariant("required value") {
            break child.wait_with_output().unwrap_or_else(|err| {
                crate::invariant_failure!(
                    "failed to collect program output after exit ({status}): {err}"
                )
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap_or_else(|err| {
                crate::invariant_failure!("failed to collect timed-out program output: {err}")
            });
            crate::invariant_failure!(
                "program timed out after {:?}: stdout={}\nstderr={}",
                BF_TEST_PROGRAM_TIMEOUT,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }
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
fn format_richer_summary_ir() {
    let program = vec![
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 3,
            preserve_src: false,
            set_dst: true,
        },
        BfIr::Shift {
            src: 1,
            dst: 2,
            amount: 2,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::Shift {
            src: 2,
            dst: 3,
            amount: 1,
            dir: ShiftDir::Right,
            preserve_src: false,
            set_dst: true,
        },
        BfIr::Square {
            src: 3,
            dst: 4,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::MulAdd {
            lhs: 1,
            rhs: 4,
            dst: 5,
            preserve_lhs: true,
            preserve_rhs: false,
            set_dst: true,
        },
    ];
    assert_eq!(
        format_ir(&program),
        "Affine { src: 0, dst: 1, coeff: 3, preserve_src: false, set_dst: true }\nShift { src: 1, dst: 2, amount: 2, dir: Left, preserve_src: true, set_dst: false }\nShift { src: 2, dst: 3, amount: 1, dir: Right, preserve_src: false, set_dst: true }\nSquare { src: 3, dst: 4, preserve_src: true, set_dst: false }\nMulAdd { lhs: 1, rhs: 4, dst: 5, preserve_lhs: true, preserve_rhs: false, set_dst: true }\n"
    );
}

#[test]
fn format_optimized_alias_heavy_ir_is_canonical() {
    let optimized = optimize_with_opts(
        vec![BfIr::MulAdd {
            lhs: 1,
            rhs: 1,
            dst: 4,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        }],
        life_opts(),
    );
    assert_eq!(
        format_ir(&optimized),
        "Square { src: 1, dst: 4, preserve_src: true, set_dst: false }\n"
    );
}

#[test]
fn validate_canonical_ir_rejects_noncanonical_muladd_alias() {
    let err = validate_canonical_ir(&[BfIr::MulAdd {
        lhs: 2,
        rhs: 2,
        dst: 3,
        preserve_lhs: true,
        preserve_rhs: false,
        set_dst: true,
    }])
    .error_or_invariant("expected error");
    assert!(err.contains("MulAdd"));
    assert!(err.contains("Square"));
}

#[test]
fn unmatched_right_bracket_errors() {
    assert!(
        Parser::new("]")
            .parse()
            .error_or_invariant("expected error")
            .contains("unmatched ']'")
    );
}

#[test]
fn unmatched_left_bracket_errors() {
    assert!(
        Parser::new("[")
            .parse()
            .error_or_invariant("expected error")
            .contains("unmatched '['")
    );
}

#[test]
fn plain_c_template_compiles_standalone_under_strict_c23_compatibility_mode() {
    let output = compile_and_run_c_template("src/bf/bf.c.in");
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
    let output = compile_and_run_c_template("src/bf/bf_super.c.in");
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
    assert!(!c.contains("BF_SCANF_FMT"));
    assert!(!c.contains("BF_PRINTF_FMT"));
}

#[test]
fn emit_c_wraps_tape_pointer_moves() {
    let c = emit_c(&parse_and_opt(">>"), default_c_opts());
    assert!(c.contains("ptrdiff_t bf_wrap_ptr(ptrdiff_t ptr, ptrdiff_t delta, ptrdiff_t len) {"));
    assert!(c.contains("#define BF_TEMPLATE_TAPE_LEN 30000"));
    assert!(c.contains("ptr = bf_wrap_ptr(ptr, 2, BF_TAPE_LEN);"));
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
    assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {0};"));
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
    assert!(c.contains("while (tape[ptr] != 0) {"));
    assert!(c.contains("bf_diverge_forever();"));
}

#[test]
fn parse_opts_rejects_64_bit_io_masks() {
    let args = vec!["--input-bits".to_string(), "64".to_string()];
    assert!(
        parse_opts(&args)
            .error_or_invariant("expected error")
            .contains("expected 0..=63")
    );
    let args = vec!["--output-bits".to_string(), "64".to_string()];
    assert!(
        parse_opts(&args)
            .error_or_invariant("expected error")
            .contains("expected 0..=63")
    );
}

#[test]
fn read_input_accepts_emit_ir_alias_without_treating_it_as_source() {
    let args = vec![
        "--emit-ir".to_string(),
        "--".to_string(),
        "+++[->++<]>.<.".to_string(),
    ];
    let (mode, opts, ir_opts, src) = read_input(&args).or_invariant("required value");
    assert_eq!(mode, super::cli::OutputMode::EmitIr);
    assert_eq!(opts, life_opts());
    assert_eq!(ir_opts, IrRenderOpts::default());
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
    let (mode, opts, _, src) = read_input(&args).or_invariant("required value");
    assert!(matches!(mode, super::cli::OutputMode::EmitLifeHashLife));
    assert_eq!(opts.cell_sign, CellSign::Unsigned);
    assert_eq!(src, "+.");
}

#[test]
fn read_input_parses_emit_ir_render_flags() {
    let args = vec![
        "--emit-ir".to_string(),
        "--emit-ir-format".to_string(),
        "json".to_string(),
        "--emit-ir-section".to_string(),
        "optimized".to_string(),
        "--emit-ir-max-lines".to_string(),
        "12".to_string(),
        "--emit-ir-max-depth".to_string(),
        "3".to_string(),
        "--".to_string(),
        "+.".to_string(),
    ];
    let (mode, _, ir_opts, src) = read_input(&args).or_invariant("required value");
    assert_eq!(mode, super::cli::OutputMode::EmitIr);
    assert_eq!(ir_opts.format, IrOutputFormat::Json);
    assert_eq!(ir_opts.section, IrSectionSelection::Optimized);
    assert_eq!(ir_opts.max_lines, 12);
    assert_eq!(ir_opts.max_depth, 3);
    assert_eq!(src, "+.");
}
