use super::*;
use crate::bf::optimizer::canonicalize_loop;

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

#[test]
fn affine_loop_non_unit_delta_stays_loop() {
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

#[test]
fn canonicalize_loop_summarizes_inner_clear_body_to_clear() {
    let body = vec![BfIr::Loop(vec![BfIr::Add(-1)])];
    assert_eq!(canonicalize_loop(body), BfIr::Clear);
}

#[test]
fn canonicalize_loop_summarizes_inner_distribute_body_to_distribute() {
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
    let body = vec![BfIr::Add(1), BfIr::Loop(vec![BfIr::Add(-1)]), BfIr::Add(-1)];
    assert_eq!(canonicalize_loop(body), BfIr::Clear);
}

#[test]
fn canonicalize_loop_keeps_unsummarizable_inner_loop_guarded() {
    let body = vec![BfIr::Loop(vec![
        BfIr::Add(-1),
        BfIr::MovePtr(1),
        BfIr::Add(1),
    ])];
    assert!(matches!(canonicalize_loop(body), BfIr::Loop(_)));
}

#[test]
fn optimizer_keeps_multi_decrement_loop_guarded() {
    assert_eq!(parse_and_opt("[--]"), vec![BfIr::Loop(vec![BfIr::Add(-2)])]);
}

#[test]
fn optimizer_keeps_non_affine_source_dependent_loop_guarded() {
    assert_eq!(
        parse_and_opt("[->+<+>]"),
        vec![BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::MovePtr(1),
            BfIr::Add(1),
            BfIr::MovePtr(-1),
            BfIr::Add(1),
            BfIr::MovePtr(1),
        ])]
    );
}

#[test]
fn optimizer_keeps_io_loops_guarded() {
    assert_eq!(parse_and_opt("[,.]"), vec![BfIr::Loop(vec![BfIr::Input, BfIr::Output])]);
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
