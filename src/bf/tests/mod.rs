use super::c_backend::{BfEvalError, emit_c, interpret_for_tests, interpret_unsigned_for_tests};
use super::c_super_backend::emit_c_super;
use super::cli::{parse_opts, read_input};
use super::format_ir;
use super::ir::{BfIr, Parser, ShiftDir, validate_canonical_ir};
use super::ir_report::{
    IrOutputFormat, IrRenderOpts, IrSectionSelection, build_ir_report, render_ir_json,
    render_ir_text,
};
use super::optimizer::{CellSign, CodegenOpts, IoMode, optimize_with_opts};
use super::summary::{
    LoopId, LoopSummary, OffsetOp, SummaryEffect, SummaryProvenance, normalize_offset_body,
};
use crate::test_support::c::CSource;
use crate::{RequiredErrorExt, RequiredExt};
use std::sync::{Mutex, OnceLock};

mod c_shape;
mod c_super_backend;
mod c_super_runtime;
mod ir_report;
mod life;
mod life_rich;
mod optimizer;
mod payload;
mod planner_shape;
mod polynomial_backend;

const BF_PLAIN_STATS_SENTINEL: &str = "=== BF PLAIN RUNTIME STATS ===";
const BF_SUPER_STATS_SENTINEL: &str = "=== BF SUPER RUNTIME STATS ===";

#[derive(Clone, Copy, Debug)]
enum TestCBackend {
    Plain,
    Super,
}

impl TestCBackend {
    fn runtime_template(self) -> &'static str {
        match self {
            Self::Plain => crate::bf::c_support::PLAIN_RUNTIME_TEMPLATE,
            Self::Super => crate::bf::c_support::SUPER_RUNTIME_TEMPLATE,
        }
    }

    fn stats_function(self) -> &'static str {
        match self {
            Self::Plain => "print_work_stats",
            Self::Super => "print_memo_stats",
        }
    }

    fn stats_call(self) -> String {
        format!("{}();", self.stats_function())
    }

    fn stats_sentinel(self) -> &'static str {
        match self {
            Self::Plain => BF_PLAIN_STATS_SENTINEL,
            Self::Super => BF_SUPER_STATS_SENTINEL,
        }
    }
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

fn compile_and_run_c_template(backend: TestCBackend) -> String {
    compile_and_run_c_source(backend.runtime_template())
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
        "program failed: status={} code={:?} flags={extra_cc_args:?}\nstdout={}\nstderr={}",
        output.status,
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).or_invariant("required value")
}

fn compile_and_run_c_source_capture(c: &str, extra_cc_args: &[&str]) -> std::process::Output {
    let c = super::c_support::expand_runtime_fragments(c);
    crate::test_support::compiled_c::compile_and_run(&c, extra_cc_args)
        .unwrap_or_else(|error| crate::invariant_failure!("generated-C harness failed: {error}"))
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
