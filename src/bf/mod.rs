mod cli;
mod codegen;
mod ir;
mod optimizer;

pub use cli::run;
pub use codegen::{compile_to_life_grid, emit_c, format_ir, serialize_life_grid};
pub use ir::{BfIr, Parser};
pub use optimizer::{CellSign, CodegenOpts, IoMode, optimize};

#[cfg(test)]
mod tests {
    use super::cli::parse_opts;
    use super::codegen::{compile_to_life_grid, emit_c, format_ir};
    use super::ir::{BfIr, Parser};
    use super::optimizer::CanonResult;
    use super::optimizer::{CellSign, CodegenOpts, IoMode, canonicalize_loop, optimize};

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
        assert_eq!(
            parse_and_opt("[+[-]]"),
            vec![BfIr::Loop(vec![BfIr::Add(1), BfIr::Clear,])]
        );
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
        assert!(matches!(
            canonicalize_loop(body),
            CanonResult::Summary(BfIr::Clear)
        ));
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
        assert!(matches!(
            canonicalize_loop(body),
            CanonResult::Summary(BfIr::Distribute { targets }) if targets == vec![(1, 1)]
        ));
    }

    #[test]
    fn canonicalize_loop_normalizes_after_inner_loop_summarization() {
        // Body: [Add(1), Loop([Add(-1)]), Add(-1)] — inner [-] not yet summarized.
        // inline pass: Loop([Add(-1)]) -> Clear. Body: [Add(1), Clear, Add(-1)].
        // normalize: merge_adjacent doesn't merge across Clear -> [Add(1), Clear, Add(-1)].
        // summarize_loop: not affine (contains Clear). Stays Loop.
        let body = vec![BfIr::Add(1), BfIr::Loop(vec![BfIr::Add(-1)]), BfIr::Add(-1)];
        assert!(matches!(canonicalize_loop(body), CanonResult::Loop(_)));
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
        assert!(matches!(canonicalize_loop(body), CanonResult::Loop(_)));
    }

    // --- emit_c ---

    #[test]
    fn emit_c_for_simple_add_and_output() {
        let c = parse_opt_and_emit_c("+++.");
        assert!(c.contains("#include <stdckdint.h>"));
        assert!(c.contains("ckd_add(&out, ua, ub)"));
        assert!(c.contains("ckd_sub(&out, ua, ub)"));
        assert!(c.contains("ckd_mul(&out, ua, ub)"));
        assert!(c.contains("tape[ptr] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[ptr], INT64_C(3), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[ptr], INT64_C(3), BF_CELL_BITS);"));
        assert!(c.contains("putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"));
        assert!(c.contains("fflush(stdout);"));
        assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {};"));
        assert!(c.contains("BF_TAPE_LEN = 30000"));
        assert!(c.contains("BF_CELL_BITS = 8"));
        assert!(c.contains("BF_SIGNED_CELLS = 1"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(255)"));
        assert!(!c.contains("BF_INPUT_MASK = "));
        assert!(!c.contains("BF_CELL_MASK = "));
        assert!(!c.contains("static ptrdiff_t bf_wrap_ptr("));
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
    fn emit_c_for_distribute_loop() {
        let c = parse_opt_and_emit_c("[->+>++<<]");
        assert!(c.contains("int64_t v = tape[ptr];"));
        assert!(c.contains("tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)], bf_wrap_mul_i64_signed(v, INT64_C(1), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)], bf_wrap_mul_i64_unsigned(v, INT64_C(1), BF_CELL_BITS), BF_CELL_BITS);"));
        assert!(c.contains("tape[bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN)] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN)], bf_wrap_mul_i64_signed(v, INT64_C(2), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN)], bf_wrap_mul_i64_unsigned(v, INT64_C(2), BF_CELL_BITS), BF_CELL_BITS);"));
        assert!(c.contains("tape[ptr] = 0;"));
    }

    #[test]
    fn emit_c_numeric_io_mode() {
        let c = emit_c(
            &parse_and_opt(",."),
            CodegenOpts {
                io_mode: IoMode::Number,
                cell_bits: 32,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Signed,
            },
        );
        assert!(c.contains("{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"));
        assert!(c.contains("printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"));
        assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {};"));
        assert!(c.contains("BF_CELL_BITS = 32"));
        assert!(c.contains("BF_CELL_MASK = UINT64_C(4294967295)"));
        assert!(c.contains("BF_INPUT_MASK = UINT64_C(4294967295)"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(4294967295)"));
        assert!(!c.contains("static ptrdiff_t bf_wrap_ptr("));
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
    fn emit_c_uses_unsigned_numeric_io_when_requested() {
        let c = emit_c(
            &parse_and_opt(",."),
            CodegenOpts {
                io_mode: IoMode::Number,
                cell_sign: CellSign::Unsigned,
                ..default_c_opts()
            },
        );
        assert!(c.contains("BF_SIGNED_CELLS = 0"));
        assert!(c.contains("{ uint64_t tmp = 0; if (scanf(\"%\" SCNu64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_unsigned(tmp & BF_INPUT_MASK, BF_CELL_BITS); }"));
        assert!(c.contains(
            "printf(\"%\" PRIu64 \"\\n\", (uint64_t)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"
        ));
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
        let grid = compile_to_life_grid(&parse_and_opt(""), life_opts());
        assert_eq!(grid.live_cells(), vec![]);
    }

    #[test]
    fn life_grid_add_encodes_cells_at_tape_zero() {
        let grid = compile_to_life_grid(&parse_and_opt("+++"), life_opts());
        let mut cells = grid.live_cells();
        cells.sort_unstable();
        assert_eq!(cells, vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn life_grid_distribute_encodes_second_tape_cell() {
        let grid = compile_to_life_grid(&parse_and_opt("+++[->+<]"), life_opts());
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
        let grid = compile_to_life_grid(&parse_and_opt("+++[->+++++<]"), opts);
        let mut cells = grid.live_cells();
        cells.sort_unstable();
        assert_eq!(cells.len(), 15);
        assert_eq!(cells[0], (17, 0));
        assert_eq!(cells[14], (31, 0));
    }

    #[test]
    fn life_grid_roundtrips_through_format() {
        let ir = parse_and_opt("+++++[->++<]");
        let grid = compile_to_life_grid(&ir, life_opts());
        let s = crate::life_grid_format::serialize(&grid);
        let loaded = crate::life_grid_format::deserialize(&s).unwrap();
        assert_eq!(loaded, grid);
    }
}
