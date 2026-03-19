mod cli;
mod codegen;
mod ir;
mod optimizer;

pub use cli::run;
pub use codegen::{
    BfLifeEmitError, compile_to_life_grid, emit_c, format_ir, serialize_legacy_life_grid,
    serialize_life_grid,
};
pub use ir::{BfIr, Parser};
pub use optimizer::{CellSign, CodegenOpts, IoMode, optimize};

#[cfg(test)]
mod tests {
    use super::cli::parse_opts;
    use super::codegen::{
        compile_to_life_grid, emit_c, format_ir, interpret_for_tests, interpret_unsigned_for_tests,
    };
    use super::ir::{BfIr, Parser};
    use super::optimizer::{CellSign, CodegenOpts, IoMode, canonicalize_loop, optimize};
    use std::fs;
    use std::process::{Command, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!("a_simple_life_bf_{unique}"));
        fs::create_dir_all(&base).unwrap();
        let source = base.join("program.c");
        let binary = base.join("program.bin");
        fs::write(&source, c).unwrap();

        let compile = Command::new("cc")
            .arg("-std=c11")
            .arg("-O0")
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "cc failed: stdout={}\nstderr={}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr)
        );

        let output = Command::new(&binary).stdin(Stdio::null()).output().unwrap();
        assert!(
            output.status.success(),
            "program failed: stdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }

    // --- parser ---

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

    // --- optimizer ---

    #[test]
    fn optimizer_combines_parsed_runs() {
        assert_eq!(parse_and_opt("+++"), vec![BfIr::Add(3)]);
    }

    #[test]
    fn merges_adds_and_moves() {
        assert_eq!(
            parse_and_opt("++++-->>><<"),
            vec![BfIr::Add(2), BfIr::MovePtr(1)]
        );
    }

    #[test]
    fn clear_minus_loop() {
        assert_eq!(parse_and_opt("[-]"), vec![BfIr::Clear]);
    }

    #[test]
    fn clear_plus_loop() {
        assert_eq!(parse_and_opt("[+]"), vec![BfIr::Clear]);
    }

    #[test]
    fn add_then_clear_collapses_to_clear() {
        assert_eq!(parse_and_opt("+++[-]"), vec![BfIr::Clear]);
    }

    #[test]
    fn clear_then_add_becomes_add() {
        assert_eq!(parse_and_opt("[-]+++"), vec![BfIr::Add(3)]);
    }

    #[test]
    fn repeated_clear_collapses() {
        assert_eq!(parse_and_opt("[-][-]"), vec![BfIr::Clear]);
    }

    #[test]
    fn distribute_single_target_add() {
        assert_eq!(
            parse_and_opt("[->+<]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn distribute_single_target_mul() {
        assert_eq!(
            parse_and_opt("[->+++<]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 3)]
            }]
        );
    }

    #[test]
    fn distribute_single_target_sub() {
        assert_eq!(
            parse_and_opt("[->-<]"),
            vec![BfIr::Distribute {
                targets: vec![(1, -1)]
            }]
        );
    }

    #[test]
    fn distribute_multiple_targets() {
        assert_eq!(
            parse_and_opt("[->+>++<<]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1), (2, 2)]
            }]
        );
    }

    #[test]
    fn distribute_to_left() {
        assert_eq!(
            parse_and_opt("[<+>-]"),
            vec![BfIr::Distribute {
                targets: vec![(-1, 1)]
            }]
        );
    }

    #[test]
    fn distribute_then_clear_keeps_distribute() {
        assert_eq!(
            parse_and_opt("[->+<][-]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn clear_then_distribute_keeps_clear() {
        assert_eq!(parse_and_opt("[-][->+<]"), vec![BfIr::Clear]);
    }

    #[test]
    fn repeated_distribute_keeps_first_overwrite() {
        assert_eq!(
            parse_and_opt("[->+<][->++<]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn affine_loop_with_extra_balanced_motion_still_becomes_distribute() {
        assert_eq!(
            parse_and_opt("[->+>+<+<]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 2), (2, 1)]
            }]
        );
    }

    #[test]
    fn affine_loop_with_net_nonzero_pointer_is_not_summarized() {
        assert_eq!(
            parse_and_opt("[->+>+<]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ])]
        );
    }

    #[test]
    fn affine_loop_with_non_unit_source_delta_is_not_summarized() {
        assert_eq!(
            parse_and_opt("[--->+<]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(-3),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ])]
        );
    }

    #[test]
    fn keeps_general_loops() {
        assert_eq!(
            parse_and_opt("[->+<+]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
                BfIr::Add(1),
            ])]
        );
    }

    #[test]
    fn nested_optimization() {
        assert_eq!(parse_and_opt("[[-]]"), vec![BfIr::Clear]);
    }

    #[test]
    fn empty_loop_stays_guarded_loop() {
        assert_eq!(parse_and_opt("[]"), vec![BfIr::Loop(vec![BfIr::Diverge])]);
    }

    #[test]
    fn nested_empty_loops_canonicalize_to_the_same_guarded_diverge_loop() {
        for src in ["[[]]", "[[[]]]", "[[[[]]]]"] {
            assert_eq!(
                parse_and_opt(src),
                vec![BfIr::Loop(vec![BfIr::Diverge])],
                "failed for {src}"
            );
        }
    }

    #[test]
    fn outer_guarded_distribute_loop_summarizes_to_distribute() {
        assert_eq!(
            parse_and_opt("[[->+<]]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn empty_loop_does_not_make_suffix_dead() {
        assert_eq!(
            parse_and_opt("[]+++"),
            vec![BfIr::Loop(vec![BfIr::Diverge]), BfIr::Add(3)]
        );
    }

    #[test]
    fn outer_loop_with_inner_clear_stays_loop() {
        assert_eq!(
            parse_and_opt("[>[-]]"),
            vec![BfIr::Loop(vec![BfIr::MovePtr(1), BfIr::Clear,])]
        );
    }

    #[test]
    fn outer_loop_with_balanced_inner_clear_stays_loop() {
        assert_eq!(
            parse_and_opt("[>[-]<]"),
            vec![BfIr::Loop(vec![
                BfIr::MovePtr(1),
                BfIr::Clear,
                BfIr::MovePtr(-1),
            ])]
        );
    }

    #[test]
    fn outer_loop_with_prefix_and_inner_clear_stays_loop() {
        assert_eq!(
            parse_and_opt("[>+<[-]]"),
            vec![BfIr::Loop(vec![
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
                BfIr::Clear,
            ])]
        );
    }

    #[test]
    fn outer_loop_with_leading_clear_and_suffix_stays_loop() {
        assert_eq!(
            parse_and_opt("[[-]>+<]"),
            vec![BfIr::Loop(vec![
                BfIr::Clear,
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ])]
        );
    }

    #[test]
    fn outer_loop_with_diverging_inner_stays_loop() {
        assert_eq!(
            parse_and_opt("[+[]]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(1),
                BfIr::Loop(vec![BfIr::Diverge]),
            ])]
        );
    }

    #[test]
    fn outer_guarded_clear_loop_summarizes_to_clear() {
        assert_eq!(parse_and_opt("[[[-]]]"), vec![BfIr::Clear]);
    }

    #[test]
    fn outer_guarded_nested_distribute_loop_summarizes_to_distribute() {
        assert_eq!(
            parse_and_opt("[[[->+<]]]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn outer_loop_with_inner_clear_and_decrement_stays_loop() {
        assert_eq!(
            parse_and_opt("[>[-]<-]"),
            vec![BfIr::Loop(vec![
                BfIr::MovePtr(1),
                BfIr::Clear,
                BfIr::MovePtr(-1),
                BfIr::Add(-1),
            ])]
        );
    }

    #[test]
    fn outer_loop_with_distribute_then_add_stays_loop() {
        assert_eq!(
            parse_and_opt("[[->+<]-]"),
            vec![BfIr::Loop(vec![
                BfIr::Distribute {
                    targets: vec![(1, 1)]
                },
                BfIr::Add(-1),
            ])]
        );
    }

    #[test]
    fn outer_loop_with_add_and_inner_clear_stays_loop() {
        assert_eq!(parse_and_opt("[+[-]]"), vec![BfIr::Clear]);
    }

    #[test]
    fn outer_loop_with_prefix_and_inner_distribute_stays_loop() {
        assert_eq!(
            parse_and_opt("[>+<[->+<]]"),
            vec![BfIr::Loop(vec![
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
                BfIr::Distribute {
                    targets: vec![(1, 1)]
                },
            ])]
        );
    }

    #[test]
    fn increment_countdown_loop_becomes_distribute_with_negated_coeffs() {
        assert_eq!(
            parse_and_opt("[>-<+]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn increment_loop_with_multiple_targets_negates_all_coeffs() {
        assert_eq!(
            parse_and_opt("[>->>--<<<+]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1), (3, 2)]
            }]
        );
    }

    #[test]
    fn inner_clear_loop_summarizes_through_outer_guard() {
        assert_eq!(parse_and_opt("[[-]]"), vec![BfIr::Clear]);
    }

    #[test]
    fn double_nested_distribute_loop_summarizes_to_distribute() {
        assert_eq!(
            parse_and_opt("[[[->+<]]]"),
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    // non-unit source delta — unsound to summarize under wrapping semantics:
    // termination is not guaranteed for all initial cell values

    #[test]
    fn affine_loop_non_unit_delta_stays_loop() {
        // [-->>+<<] : src_delta=-2, not unit — stays loop
        assert_eq!(
            parse_and_opt("[-->>+<<]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(-2),
                BfIr::MovePtr(2),
                BfIr::Add(1),
                BfIr::MovePtr(-2),
            ])]
        );
    }

    #[test]
    fn affine_loop_decrement_by_3_stays_loop_even_when_coefficients_divide() {
        // [--->+++<] : src_delta=-3, divisible but not unit — stays loop
        // cell[0]=1 → 1-3 wraps → non-terminating under 8-bit unsigned
        assert_eq!(
            parse_and_opt("[--->+++<]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(-3),
                BfIr::MovePtr(1),
                BfIr::Add(3),
                BfIr::MovePtr(-1),
            ])]
        );
    }

    // diverge-in-nested-loop: outer loop body contains Diverge, stays loop

    #[test]
    fn outer_loop_with_add_and_nested_diverge_stays_loop() {
        assert_eq!(
            parse_and_opt("[+[[]]]"),
            vec![BfIr::Loop(vec![
                BfIr::Add(1),
                BfIr::Loop(vec![BfIr::Diverge]),
            ])]
        );
    }

    #[test]
    fn outer_loop_with_prefix_and_nested_diverge_stays_loop() {
        assert_eq!(
            parse_and_opt("[>+<[[]]]"),
            vec![BfIr::Loop(vec![
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
                BfIr::Loop(vec![BfIr::Diverge]),
            ])]
        );
    }

    // --- canonicalize_loop direct: inline_summarizable_inner_loops ---
    // These call canonicalize_loop directly with bodies containing unprocessed
    // Loop nodes, exercising the inline pass that optimize_sequence already
    // handles implicitly via its bottom-up stack traversal.

    #[test]
    fn canonicalize_loop_summarizes_inner_clear_body_to_clear() {
        // Body: [Loop([Add(-1)])] — inner [-] not yet summarized.
        // inline pass: Loop([Add(-1)]) -> Clear. Body becomes [Clear].
        // summarize_loop([Clear]) -> Some(Clear).
        let body = vec![BfIr::Loop(vec![BfIr::Add(-1)])];
        assert_eq!(canonicalize_loop(body), BfIr::Clear);
    }

    #[test]
    fn canonicalize_loop_summarizes_inner_distribute_body_to_distribute() {
        // Body: [Loop([Add(-1), MovePtr(1), Add(1), MovePtr(-1)])] — inner [->+<] not yet summarized.
        // inline pass: inner -> Distribute{(1,1)}. Body becomes [Distribute{(1,1)}].
        // summarize_loop([Distribute{..}]) -> Some(Distribute{(1,1)}).
        let body = vec![BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::MovePtr(1),
            BfIr::Add(1),
            BfIr::MovePtr(-1),
        ])];
        assert_eq!(
            canonicalize_loop(body),
            BfIr::Distribute {
                targets: vec![(1, 1)]
            }
        );
    }

    #[test]
    fn canonicalize_loop_normalizes_after_inner_loop_summarization() {
        // Body: [Add(1), Loop([Add(-1)]), Add(-1)] — inner [-] not yet summarized.
        // inline pass: Loop([Add(-1)]) -> Clear. Body: [Add(1), Clear, Add(-1)].
        // stronger peepholes reduce that to [Add(-1)], which is then a clear-like loop.
        let body = vec![BfIr::Add(1), BfIr::Loop(vec![BfIr::Add(-1)]), BfIr::Add(-1)];
        assert_eq!(canonicalize_loop(body), BfIr::Clear);
    }

    #[test]
    fn canonicalize_loop_keeps_unsummarizable_inner_loop_guarded() {
        // Body: [Loop([Add(-1), MovePtr(1), Add(1)])] — inner has net nonzero ptr, not summarizable.
        // inline pass: summarize_loop returns None -> no substitution.
        // Outer body still contains Loop -> not summarizable -> stays Loop.
        let body = vec![BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::MovePtr(1),
            BfIr::Add(1),
        ])];
        assert!(matches!(canonicalize_loop(body), BfIr::Loop(_)));
    }

    #[test]
    fn optimize_normalizes_zero_target_distribute_to_clear() {
        assert_eq!(
            optimize(vec![BfIr::Distribute {
                targets: vec![(1, 0), (0, 7), (2, 0)],
            }]),
            vec![BfIr::Clear]
        );
    }

    #[test]
    fn optimize_deduplicates_distribute_targets() {
        assert_eq!(
            optimize(vec![BfIr::Distribute {
                targets: vec![(1, 1), (1, 2), (2, 0), (1, -1)],
            }]),
            vec![BfIr::Distribute {
                targets: vec![(1, 2)]
            }]
        );
    }

    #[test]
    fn optimized_ir_symbol_count_drops_for_current_cell_overwrites() {
        let optimized = parse_and_opt("+++[-][->+<][-]");
        assert_eq!(optimized, vec![BfIr::Clear]);
        assert_eq!(optimized.len(), 1);
    }

    #[test]
    fn optimized_ir_matches_unoptimized_for_guarded_diverge_shapes() {
        for src in ["[]", "[[]]", "[[[]]]", "+[[]]", "[+[[]]]"] {
            assert_optimized_matches_unoptimized_unsigned(src);
        }
    }

    #[test]
    fn optimized_ir_matches_unoptimized_for_summarized_affine_loops() {
        for src in ["[-]", "[+]", "[->+<]", "[->+>++<<]"] {
            assert_optimized_matches_unoptimized_unsigned(src);
        }
    }

    #[test]
    fn optimized_ir_matches_unoptimized_for_unsummarized_loop_regressions() {
        for src in [
            "[-->>+<<]",
            "[--->+++<]",
            "[->+>+<]",
            "[>[-]<-]",
            "[[->+<]-]",
            "[>+<[->+<]]",
        ] {
            assert_optimized_matches_unoptimized_unsigned(src);
        }
    }

    #[test]
    fn optimized_ir_matches_unoptimized_for_signed_semantics() {
        for src in [
            "+++[-]",
            "+++[->+<]",
            "+++[->+>++<<]",
            "[-]+++[-]",
            "[-->>+<<]",
            "[>[-]<-]",
        ] {
            assert_optimized_matches_unoptimized_with_opts(src, signed_test_opts());
        }
    }

    // --- emit_c ---

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
        assert!(c.contains(
            "static ptrdiff_t bf_wrap_ptr(ptrdiff_t ptr, ptrdiff_t delta, ptrdiff_t len) {"
        ));
        assert!(c.contains("BF_TAPE_LEN = 30000"));
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
        assert!(c.contains("BF_SIGNED_CELLS = 0"));
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
        assert!(c.contains("BF_CELL_BITS = 63"));
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
        assert!(c.contains("BF_INPUT_MASK = UINT64_C(31)"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(63)"));
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
        assert!(c.contains("BF_INPUT_MASK = UINT64_C(7)"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(15)"));
        assert!(c.contains("{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"));
        assert!(
            c.contains("bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS)")
        );
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
    fn emit_c_for_empty_loop_stays_guarded() {
        let c = parse_opt_and_emit_c("[]");
        assert!(c.contains("while (tape[ptr] != 0) {"));
        assert!(c.contains("bf_diverge_forever();"));
    }

    // --- parse_opts ---

    #[test]
    fn parse_opts_rejects_64_bit_io_masks() {
        let args = vec!["--input-bits".to_string(), "64".to_string()];
        assert!(parse_opts(&args).unwrap_err().contains("expected 0..=63"));
        let args = vec!["--output-bits".to_string(), "64".to_string()];
        assert!(parse_opts(&args).unwrap_err().contains("expected 0..=63"));
    }

    // --- compile_to_life_grid ---

    #[test]
    fn life_grid_empty_program_produces_empty_grid() {
        let grid = compile_to_life_grid(&parse_and_opt(""), life_opts()).unwrap();
        assert_eq!(grid.live_cells(), vec![]);
    }

    #[test]
    fn life_grid_add_encodes_cells_at_tape_zero() {
        let grid = compile_to_life_grid(&parse_and_opt("+++"), life_opts()).unwrap();
        let mut cells = grid.live_cells();
        cells.sort_unstable();
        assert_eq!(cells, vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn life_grid_distribute_encodes_second_tape_cell() {
        let grid = compile_to_life_grid(&parse_and_opt("+++[->+<]"), life_opts()).unwrap();
        let mut cells = grid.live_cells();
        cells.sort_unstable();
        assert_eq!(cells, vec![(257, 0), (258, 0), (259, 0)]);
    }

    #[test]
    fn life_grid_cell_bits_controls_stride() {
        let opts = CodegenOpts {
            cell_bits: 4,
            ..life_opts()
        };
        let grid = compile_to_life_grid(&parse_and_opt("+++[->+++++<]"), opts).unwrap();
        let mut cells = grid.live_cells();
        cells.sort_unstable();
        assert_eq!(cells.len(), 15);
        assert_eq!(cells[0], (17, 0));
        assert_eq!(cells[14], (31, 0));
    }

    #[test]
    fn life_grid_roundtrips_through_format() {
        let ir = parse_and_opt("+++++[->++<]");
        let grid = compile_to_life_grid(&ir, life_opts()).unwrap();
        let s = crate::persistence::serialize_life_grid(&grid);
        let loaded = crate::persistence::deserialize_life_grid(&s).unwrap();
        assert_eq!(loaded, grid);
    }

    #[test]
    fn default_life_emitter_outputs_hashlife_snapshot() {
        let ir = parse_and_opt("+++++[->++<]");
        let grid = compile_to_life_grid(&ir, life_opts()).unwrap();
        let snapshot = super::serialize_life_grid(&ir, life_opts()).unwrap();
        let loaded = crate::hashlife::deserialize_snapshot_to_grid(&snapshot).unwrap();
        assert_eq!(loaded, grid);
    }

    #[test]
    fn legacy_life_emitter_preserves_cell_list_format() {
        let ir = parse_and_opt("+++++[->++<]");
        let grid = compile_to_life_grid(&ir, life_opts()).unwrap();
        let serialized = super::serialize_legacy_life_grid(&ir, life_opts()).unwrap();
        let loaded = crate::persistence::deserialize_life_grid(&serialized).unwrap();
        assert_eq!(loaded, grid);
    }

    #[test]
    fn life_emit_rejects_signed_cells() {
        let err = super::serialize_life_grid(
            &parse_and_opt("+++"),
            CodegenOpts {
                cell_sign: CellSign::Signed,
                ..life_opts()
            },
        )
        .unwrap_err();
        assert_eq!(err, super::BfLifeEmitError::SignedCellsUnsupported);
    }

    #[test]
    fn life_emit_rejects_diverging_programs() {
        let err = super::serialize_life_grid(&parse_and_opt("+[]"), life_opts()).unwrap_err();
        assert_eq!(err, super::BfLifeEmitError::DivergenceDetected);
        let legacy_err =
            super::serialize_legacy_life_grid(&parse_and_opt("+[]"), life_opts()).unwrap_err();
        assert_eq!(legacy_err, super::BfLifeEmitError::DivergenceDetected);
    }

    #[test]
    fn life_emit_rejects_step_budget_exceeded_programs() {
        let program = vec![BfIr::Add(1); 10_000_001];
        let err = compile_to_life_grid(&program, life_opts()).unwrap_err();
        assert_eq!(err, super::BfLifeEmitError::StepBudgetExceeded);
    }
}
