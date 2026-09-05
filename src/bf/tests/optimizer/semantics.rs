use super::*;

#[test]
fn overflowing_scale_coefficients_fold_using_shared_modular_arithmetic() {
    for cell_sign in [CellSign::Unsigned, CellSign::Signed] {
        let opts = CodegenOpts {
            cell_sign,
            ..life_opts()
        };
        let scale = BfIr::Affine {
            src: 0,
            dst: 0,
            coeff: 2_000_000_003,
            preserve_src: true,
            set_dst: true,
        };
        let original = vec![BfIr::Add(7), scale.clone(), scale];
        let optimized = optimize_with_opts(original.clone(), opts);
        assert_eq!(
            optimized,
            vec![
                BfIr::Add(7),
                BfIr::Affine {
                    src: 0,
                    dst: 0,
                    coeff: 9,
                    preserve_src: true,
                    set_dst: true
                }
            ],
            "cell_sign={cell_sign:?}: wrapped coefficient product must not be rejected at the i32 boundary"
        );
        assert_eq!(
            interpret_for_tests(&optimized, opts).or_invariant("optimized scaling terminates"),
            interpret_for_tests(&original, opts).or_invariant("original scaling terminates"),
            "cell_sign={cell_sign:?}: modular folding must preserve every cell and pointer"
        );
    }
}

#[test]
fn optimize_canonicalizes_muladd_with_aliased_sources_and_destination_to_square() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::MulAdd {
                lhs: 3,
                rhs: 3,
                dst: 3,
                preserve_lhs: false,
                preserve_rhs: false,
                set_dst: false,
            }],
            life_opts()
        ),
        vec![BfIr::Square {
            src: 3,
            dst: 3,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_rewrites_in_place_power_of_two_affine_to_shift() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Affine {
                src: 0,
                dst: 0,
                coeff: 3,
                preserve_src: true,
                set_dst: false,
            }],
            life_opts(),
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 0,
            amount: 2,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        }]
    );
}

#[test]
fn optimize_combines_in_place_affine_multiply_chain() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 0,
                    coeff: 2,
                    preserve_src: true,
                    set_dst: true,
                },
                BfIr::Affine {
                    src: 0,
                    dst: 0,
                    coeff: 4,
                    preserve_src: true,
                    set_dst: true,
                },
            ],
            life_opts(),
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 0,
            amount: 3,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        }]
    );
}

#[test]
fn optimize_combines_same_direction_in_place_shifts() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Shift {
                    src: 0,
                    dst: 0,
                    amount: 1,
                    dir: ShiftDir::Left,
                    preserve_src: true,
                    set_dst: true,
                },
                BfIr::Shift {
                    src: 0,
                    dst: 0,
                    amount: 2,
                    dir: ShiftDir::Left,
                    preserve_src: true,
                    set_dst: true,
                },
            ],
            life_opts(),
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 0,
            amount: 3,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
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
fn optimizer_recognizes_preserving_copy_loop_as_affine() {
    assert_eq!(
        parse_and_opt("[->+>+<<]>>[-<<+>>]<<"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimizer_recognizes_preserving_power_of_two_copy_loop_as_shift_left() {
    assert_eq!(
        parse_and_opt("[->++>+<<]>>[-<<+>>]<<"),
        vec![BfIr::Shift {
            src: 0,
            dst: 1,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimizer_recognizes_preserving_multitarget_copy_loop_as_distribute() {
    assert_eq!(
        parse_and_opt("[->+>++>+<<<]>>>[-<<<+>>>]<<<"),
        vec![BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: true,
        }]
    );
}

#[test]
fn optimizer_recognizes_nested_multiply_loop_as_muladd() {
    assert_eq!(
        parse_and_opt("[->[->+>+<<]>>[-<<+>>]<<<]"),
        vec![BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 2,
            preserve_lhs: false,
            preserve_rhs: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimizer_keeps_multitarget_distribute_when_no_single_rich_summary_exists() {
    assert_eq!(
        parse_and_opt("[->+>+++<<]"),
        vec![BfIr::Distribute {
            targets: vec![(1, 1), (2, 3)],
            preserve_src: false,
        }]
    );
}

#[test]
fn optimizer_recognizes_square_via_preserved_copy_then_temp_clear() {
    assert_eq!(
        parse_and_opt("[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<"),
        vec![BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn optimizer_keeps_near_match_square_cleanup_partial_when_temp_accounting_is_incomplete() {
    let optimized = parse_and_opt("[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]");
    assert_eq!(
        optimized,
        vec![
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::Square {
                src: 0,
                dst: 2,
                preserve_src: false,
                set_dst: false,
            },
            BfIr::MovePtr(1),
            BfIr::Clear,
        ]
    );
    assert!(
        !optimized
            .iter()
            .any(|node| matches!(node, BfIr::MulAdd { .. })),
        "near-match should not be rewritten into exponentiation-style square+muladd"
    );
}

#[test]
fn optimizer_recognizes_exponentiation_style_nest_as_square_then_muladd() {
    assert_eq!(
        parse_and_opt(
            "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[->[->+>+<<]>>[-<<+>>]<<<]<"
        ),
        vec![
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::Square {
                src: 0,
                dst: 2,
                preserve_src: false,
                set_dst: false,
            },
            BfIr::MovePtr(1),
            BfIr::MulAdd {
                lhs: 0,
                rhs: 1,
                dst: 2,
                preserve_lhs: false,
                preserve_rhs: true,
                set_dst: false,
            },
            BfIr::MovePtr(-1),
        ]
    );
}

#[test]
fn optimizer_does_not_collapse_square_when_preserved_copy_remains_live() {
    assert_eq!(
        parse_and_opt("[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]"),
        vec![
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::Square {
                src: 0,
                dst: 2,
                preserve_src: false,
                set_dst: false,
            },
        ]
    );
}

#[test]
fn optimized_ir_matches_unoptimized_for_guarded_diverge_shapes() {
    for src in ["[]", "[[]]", "[[[]]]", "+[[]]", "[+[[]]]", "+[[-]+]"] {
        let parsed = parse_only(src);
        let optimized = optimize_with_opts(parsed.clone(), life_opts());
        let parsed_result = interpret_unsigned_for_tests(&parsed, life_opts().cell_bits);
        let optimized_result = interpret_unsigned_for_tests(&optimized, life_opts().cell_bits);
        match (optimized_result, parsed_result) {
            (Ok(optimized), Ok(parsed)) => assert_eq!(
                optimized, parsed,
                "zero-entry guarded divergence mismatch for {src}"
            ),
            (Err(BfEvalError::DivergenceDetected), Err(BfEvalError::StepBudgetExceeded)) => {}
            (Err(BfEvalError::StepBudgetExceeded), Err(BfEvalError::StepBudgetExceeded)) => {}
            (optimized, parsed) => crate::invariant_failure!(
                "guarded divergence classification mismatch for {src}: optimized={optimized:?} parsed={parsed:?}"
            ),
        }
    }
}

#[test]
fn optimized_ir_matches_unoptimized_for_summarized_arithmetic_sources() {
    for src in [
        "[-]",
        "[+]",
        "[->+<]",
        "[->+>++<<]",
        "[->+>++>+<<<]>>>[-<<<+>>>]<<<",
        "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<",
        "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[->[->+>+<<]>>[-<<+>>]<<<]<",
    ] {
        assert_optimized_matches_unoptimized_unsigned(src);
    }
}

#[test]
fn optimized_ir_matches_unoptimized_for_rich_sequence_canonicalization() {
    let program = vec![
        BfIr::Add(5),
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::Square {
            src: 1,
            dst: 2,
            preserve_src: false,
            set_dst: false,
        },
        BfIr::MovePtr(2),
        BfIr::Output,
    ];
    let optimized = optimize_with_opts(program.clone(), life_opts());
    let parsed_result = interpret_unsigned_for_tests(&program, life_opts().cell_bits);
    let optimized_result = interpret_unsigned_for_tests(&optimized, life_opts().cell_bits);
    assert_eq!(optimized_result, parsed_result);
    assert_eq!(
        optimized,
        vec![
            BfIr::Add(5),
            BfIr::Square {
                src: 0,
                dst: 2,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::MovePtr(2),
            BfIr::Output,
        ]
    );
}

#[test]
fn optimized_ir_matches_unoptimized_for_temp_consuming_polynomial_cleanup() {
    let program = vec![
        BfIr::Add(5),
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::Square {
            src: 1,
            dst: 2,
            preserve_src: false,
            set_dst: false,
        },
        BfIr::MovePtr(1),
        BfIr::Clear,
        BfIr::MovePtr(-1),
        BfIr::MovePtr(2),
        BfIr::Output,
    ];
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
    let parsed_result = interpret_unsigned_for_tests(&program, life_opts().cell_bits);
    let optimized_result = interpret_unsigned_for_tests(&optimized, life_opts().cell_bits);
    assert_eq!(optimized_result, parsed_result);
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

#[test]
fn recognized_nested_arithmetic_patterns_do_not_remain_as_loop_chains() {
    for src in [
        "[->+<]",
        "[->+>++>+<<<]>>>[-<<<+>>>]<<<",
        "[->[->+>+<<]>>[-<<+>>]<<<]",
        "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<",
        "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[->[->+>+<<]>>[-<<+>>]<<<]<",
    ] {
        let optimized = parse_and_opt(src);
        assert!(
            optimized.iter().all(|node| !matches!(node, BfIr::Loop(_))),
            "recognized arithmetic source should not remain as loop chain: {src} -> {optimized:?}"
        );
    }
}

#[test]
fn signed_optimizer_emits_proven_rich_summaries() {
    let opts = signed_test_opts();
    assert_eq!(
        parse_and_opt_with_opts("[->+<]", opts),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
    assert_eq!(
        parse_and_opt_with_opts("[->++<]", opts),
        vec![BfIr::Shift {
            src: 0,
            dst: 1,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: false,
            set_dst: false,
        }]
    );
    assert_eq!(
        parse_and_opt_with_opts("[->+>++>+<<<]>>>[-<<<+>>>]<<<", opts),
        vec![BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: true,
        }]
    );
    assert_eq!(
        parse_and_opt_with_opts("[->[->+>+<<]>>[-<<+>>]<<<]", opts),
        vec![BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 2,
            preserve_lhs: false,
            preserve_rhs: true,
            set_dst: false,
        }]
    );
    assert_eq!(
        parse_and_opt_with_opts("[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<", opts),
        vec![BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn signed_and_unsigned_optimizers_match_for_proven_degree_two_patterns() {
    for src in [
        "[->+<]",
        "[->++<]",
        "[->+>++>+<<<]>>>[-<<<+>>>]<<<",
        "[->[->+>+<<]>>[-<<+>>]<<<]",
        "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[-]<",
        "[->+>+<<]>>[-<<+>>]<<[->[->+>+<<]>>[-<<+>>]<<<]>[->[->+>+<<]>>[-<<+>>]<<<]<",
    ] {
        assert_eq!(
            parse_and_opt_with_opts(src, life_opts()),
            parse_and_opt_with_opts(src, signed_test_opts()),
            "optimizer shape mismatch between unsigned and signed semantics for {src}"
        );
    }
}

#[test]
fn signed_optimizer_rewrites_only_proven_shift_and_affine_forms() {
    let opts = signed_test_opts();
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Affine {
                src: 0,
                dst: 0,
                coeff: 3,
                preserve_src: true,
                set_dst: false,
            }],
            opts,
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 0,
            amount: 2,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        }]
    );
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Affine {
                src: 0,
                dst: 0,
                coeff: -2,
                preserve_src: true,
                set_dst: false,
            }],
            opts,
        ),
        vec![BfIr::Affine {
            src: 0,
            dst: 0,
            coeff: -2,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn signed_optimizer_summarizes_odd_non_unit_source_delta_loops() {
    assert_eq!(
        parse_and_opt_with_opts(
            "[--->+++<]",
            CodegenOpts {
                cell_sign: CellSign::Signed,
                ..life_opts()
            },
        ),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}
