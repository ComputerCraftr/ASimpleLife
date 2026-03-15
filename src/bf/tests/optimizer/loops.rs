use super::*;
use crate::RequiredExt;

#[test]
fn pointer_only_loops_summarize_to_directional_scans() {
    assert_eq!(parse_and_opt("[>>>]"), vec![BfIr::Scan { stride: 3 }]);
    assert_eq!(parse_and_opt("[<<]"), vec![BfIr::Scan { stride: -2 }]);
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
fn outer_guarded_nested_single_target_distribute_loop_summarizes_to_affine() {
    assert_eq!(
        parse_and_opt("[[[->+<]]]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
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
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: false,
                set_dst: false,
            },
            BfIr::Add(-1),
        ])]
    );
}

#[test]
fn outer_loop_with_add_and_inner_clear_summarizes_to_clear() {
    assert_eq!(parse_and_opt("[+[-]]"), vec![BfIr::Clear]);
}

#[test]
fn outer_loop_with_inner_clear_then_nonzero_add_stays_guarded() {
    assert_eq!(
        parse_and_opt("[[-]+]"),
        vec![BfIr::Loop(vec![BfIr::Clear, BfIr::Add(1)])]
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
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: false,
                set_dst: false,
            },
        ])]
    );
}

#[test]
fn increment_countdown_loop_becomes_distribute_with_negated_coeffs() {
    assert_eq!(
        parse_and_opt("[>-<+]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn increment_loop_with_multiple_targets_negates_all_coeffs() {
    assert_eq!(
        parse_and_opt("[>->>--<<<+]"),
        vec![BfIr::Distribute {
            targets: vec![(1, 1), (3, 2)],
            preserve_src: false,
        }]
    );
}

#[test]
fn inner_clear_loop_summarizes_through_outer_guard() {
    assert_eq!(parse_and_opt("[[-]]"), vec![BfIr::Clear]);
}

#[test]
fn double_nested_single_target_distribute_loop_summarizes_to_affine() {
    assert_eq!(
        parse_and_opt("[[[->+<]]]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn affine_loop_even_non_unit_delta_stays_loop() {
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
fn affine_loop_decrement_by_3_summarizes_via_ssa() {
    assert_eq!(
        parse_and_opt("[--->+++<]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn affine_loop_decrement_by_3_summarizes_to_shift_when_scaled_factor_is_power_of_two() {
    assert_eq!(
        parse_and_opt("[--->++++++<]"),
        vec![BfIr::Shift {
            src: 0,
            dst: 1,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn affine_loop_decrement_by_3_summarizes_multitarget_distribute_via_ssa() {
    assert_eq!(
        parse_and_opt("[--->+++>++++++<<]"),
        vec![BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: false,
        }]
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
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }
    );
}

#[test]
fn canonicalize_loop_summarizes_single_affine_body() {
    let body = vec![BfIr::Affine {
        src: 0,
        dst: 1,
        coeff: 1,
        preserve_src: false,
        set_dst: false,
    }];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }
    );
}

#[test]
fn canonicalize_loop_summarizes_single_distribute_body() {
    let body = vec![BfIr::Distribute {
        targets: vec![(1, 1), (2, 2)],
        preserve_src: false,
    }];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: false,
        }
    );
}

#[test]
fn canonicalize_loop_summarizes_preserving_distribute_from_invariant_source() {
    let body = vec![
        BfIr::Add(-1),
        BfIr::MovePtr(1),
        BfIr::Distribute {
            targets: vec![(1, 1)],
            preserve_src: true,
        },
        BfIr::MovePtr(-1),
    ];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::MulAdd {
            lhs: 0,
            rhs: 1,
            dst: 2,
            preserve_lhs: false,
            preserve_rhs: true,
            set_dst: false,
        }
    );
}

#[test]
fn canonicalize_loop_canonicalizes_guard_writing_distribute_to_guarded_affine() {
    let body = vec![
        BfIr::Add(-1),
        BfIr::MovePtr(1),
        BfIr::Distribute {
            targets: vec![(-1, 1)],
            preserve_src: true,
        },
        BfIr::MovePtr(-1),
    ];

    assert_eq!(
        canonicalize_loop(body),
        BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::MovePtr(1),
            BfIr::Affine {
                src: 0,
                dst: -1,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::MovePtr(-1),
        ])
    );
}

#[test]
fn canonicalize_loop_keeps_c_tape_wrapped_source_alias_guarded() {
    let body = vec![
        BfIr::Add(-1),
        BfIr::MovePtr(30_000),
        BfIr::Distribute {
            targets: vec![(1, 1)],
            preserve_src: true,
        },
        BfIr::MovePtr(-30_000),
    ];

    assert!(matches!(canonicalize_loop(body), BfIr::Loop(_)));
}

#[test]
fn canonicalize_loop_keeps_life_tape_wrapped_source_alias_guarded() {
    let body = vec![
        BfIr::Add(-1),
        BfIr::MovePtr(64),
        BfIr::Distribute {
            targets: vec![(1, 1)],
            preserve_src: true,
        },
        BfIr::MovePtr(-64),
    ];

    assert!(matches!(canonicalize_loop(body), BfIr::Loop(_)));
}

#[test]
fn wrapped_c_tape_alias_optimized_runtime_matches_guarded_loop() {
    let body = vec![
        BfIr::Add(-1),
        BfIr::MovePtr(30_000),
        BfIr::Distribute {
            targets: vec![(1, 1)],
            preserve_src: true,
        },
        BfIr::MovePtr(-30_000),
    ];
    let original = vec![BfIr::Add(3), BfIr::Loop(body)];
    let optimized = optimize_with_opts(original.clone(), life_opts());

    let expected = interpret_unsigned_for_tests(&original, life_opts().cell_bits)
        .or_invariant("guarded wrapped-tape loop should terminate");
    let actual = interpret_unsigned_for_tests(&optimized, life_opts().cell_bits)
        .or_invariant("optimized wrapped-tape loop should terminate");

    assert_eq!(
        actual, expected,
        "wrapped-tape alias changed loop semantics"
    );
    assert_eq!(
        actual.0[1], 3,
        "fixture should expose the former x*x miscompile"
    );
}
