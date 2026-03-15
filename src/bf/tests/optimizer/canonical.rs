use super::*;

#[test]
fn nonzero_self_affine_clear_preserves_destination_offset() {
    let optimized = optimize_with_opts(
        vec![BfIr::Affine {
            src: 1,
            dst: 1,
            coeff: 0,
            preserve_src: false,
            set_dst: true,
        }],
        life_opts(),
    );
    assert_eq!(optimized, vec![BfIr::ClearAt { offset: 1 }]);

    let executed = vec![
        BfIr::Add(7),
        BfIr::MovePtr(1),
        BfIr::Add(9),
        BfIr::MovePtr(-1),
        optimized[0].clone(),
    ];
    let (tape, ptr) = interpret_unsigned_for_tests(&executed, 8).or_invariant("required value");
    assert_eq!(ptr, 0);
    assert_eq!(tape[0], 7);
    assert_eq!(tape[1], 0);
}

#[test]
fn canonicalize_loop_keeps_nested_multitarget_distribute_restore_body_guarded() {
    let body = vec![
        BfIr::Distribute {
            targets: vec![(1, 1), (2, 2), (3, 1)],
            preserve_src: false,
        },
        BfIr::MovePtr(3),
        BfIr::Affine {
            src: 0,
            dst: -3,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        },
        BfIr::MovePtr(-3),
    ];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::Loop(vec![BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: true,
        }])
    );
}

#[test]
fn canonicalize_loop_keeps_wrapped_nested_temp_restore_body_guarded() {
    let body = vec![
        BfIr::Distribute {
            targets: vec![(1, 2), (64, 1)],
            preserve_src: false,
        },
        BfIr::MovePtr(64),
        BfIr::Affine {
            src: 0,
            dst: -64,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        },
        BfIr::MovePtr(-64),
    ];

    assert_eq!(canonicalize_loop(body.clone()), BfIr::Loop(body));
}

#[test]
fn canonicalize_loop_summarizes_single_shift_body() {
    let body = vec![BfIr::Shift {
        src: 0,
        dst: 1,
        amount: 1,
        dir: ShiftDir::Left,
        preserve_src: false,
        set_dst: false,
    }];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::Shift {
            src: 0,
            dst: 1,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: false,
            set_dst: false,
        }
    );
}

#[test]
fn canonicalize_loop_summarizes_single_square_body() {
    let body = vec![BfIr::Square {
        src: 0,
        dst: 2,
        preserve_src: false,
        set_dst: false,
    }];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: false,
            set_dst: false,
        }
    );
}

#[test]
fn canonicalize_loop_keeps_noncanonical_rich_body_guarded() {
    let body = vec![BfIr::Square {
        src: 0,
        dst: 2,
        preserve_src: true,
        set_dst: false,
    }];
    assert_eq!(canonicalize_loop(body.clone()), BfIr::Loop(body),);
}

#[test]
fn canonicalize_loop_summarizes_single_muladd_body() {
    let body = vec![BfIr::MulAdd {
        lhs: 0,
        rhs: 1,
        dst: 2,
        preserve_lhs: false,
        preserve_rhs: true,
        set_dst: false,
    }];
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
fn canonicalize_loop_keeps_nonzero_suffix_after_inner_clear_guarded() {
    let body = vec![BfIr::Add(1), BfIr::Loop(vec![BfIr::Add(-1)]), BfIr::Add(-1)];
    assert_eq!(
        canonicalize_loop(body),
        BfIr::Loop(vec![BfIr::Clear, BfIr::Add(-1)])
    );
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
    assert_eq!(
        parse_and_opt("[,.]"),
        vec![BfIr::Loop(vec![BfIr::Input, BfIr::Output])]
    );
}

#[test]
fn optimize_normalizes_zero_target_distribute_to_clear() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Distribute {
                targets: vec![(1, 0), (0, 7), (2, 0)],
                preserve_src: false,
            }],
            life_opts()
        ),
        vec![BfIr::Clear]
    );
}

#[test]
fn optimize_deduplicates_distribute_targets() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Distribute {
                targets: vec![(1, 1), (1, 2), (2, 0), (1, -1)],
                preserve_src: false,
            }],
            life_opts()
        ),
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
fn optimize_normalizes_zero_shift_left_to_affine() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Shift {
                src: 0,
                dst: 1,
                amount: 0,
                dir: ShiftDir::Left,
                preserve_src: false,
                set_dst: true,
            }],
            life_opts()
        ),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: true,
        }]
    );
}

#[test]
fn optimize_normalizes_zero_shift_right_to_affine() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Shift {
                src: 0,
                dst: 1,
                amount: 0,
                dir: ShiftDir::Right,
                preserve_src: false,
                set_dst: false,
            }],
            life_opts()
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

#[test]
fn optimize_forces_aliasing_summary_sources_to_be_preserved() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 0,
                    coeff: 2,
                    preserve_src: false,
                    set_dst: false,
                },
                BfIr::Shift {
                    src: 1,
                    dst: 1,
                    amount: 3,
                    dir: ShiftDir::Left,
                    preserve_src: false,
                    set_dst: true,
                },
                BfIr::Square {
                    src: 2,
                    dst: 2,
                    preserve_src: false,
                    set_dst: false,
                },
                BfIr::MulAdd {
                    lhs: 3,
                    rhs: 4,
                    dst: 4,
                    preserve_lhs: false,
                    preserve_rhs: false,
                    set_dst: true,
                },
            ],
            life_opts()
        ),
        vec![
            BfIr::Affine {
                src: 0,
                dst: 0,
                coeff: 2,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::Shift {
                src: 1,
                dst: 1,
                amount: 3,
                dir: ShiftDir::Left,
                preserve_src: true,
                set_dst: true,
            },
            BfIr::Square {
                src: 2,
                dst: 2,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::MulAdd {
                lhs: 3,
                rhs: 4,
                dst: 4,
                preserve_lhs: false,
                preserve_rhs: true,
                set_dst: true,
            },
        ]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_affine_on_temp() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::Affine {
                    src: 1,
                    dst: 2,
                    coeff: 3,
                    preserve_src: false,
                    set_dst: false,
                },
            ],
            life_opts()
        ),
        vec![BfIr::Affine {
            src: 0,
            dst: 2,
            coeff: 3,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_shift_on_temp() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::Shift {
                    src: 1,
                    dst: 2,
                    amount: 2,
                    dir: ShiftDir::Left,
                    preserve_src: false,
                    set_dst: false,
                },
            ],
            life_opts()
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 2,
            amount: 2,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_square_on_temp() {
    assert_eq!(
        optimize_with_opts(
            vec![
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
            ],
            life_opts()
        ),
        vec![BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_shift_then_temp_clear() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::Shift {
                    src: 0,
                    dst: 2,
                    amount: 1,
                    dir: ShiftDir::Left,
                    preserve_src: false,
                    set_dst: false,
                },
                BfIr::MovePtr(1),
                BfIr::Clear,
                BfIr::MovePtr(-1),
            ],
            life_opts()
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 2,
            amount: 1,
            dir: ShiftDir::Left,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_square_then_temp_clear() {
    assert_eq!(
        optimize_with_opts(
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
                BfIr::MovePtr(-1),
            ],
            life_opts()
        ),
        vec![BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_affine_on_temp_then_temp_clear() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::Affine {
                    src: 1,
                    dst: 2,
                    coeff: 3,
                    preserve_src: false,
                    set_dst: false,
                },
                BfIr::MovePtr(1),
                BfIr::Clear,
                BfIr::MovePtr(-1),
            ],
            life_opts()
        ),
        vec![BfIr::Affine {
            src: 0,
            dst: 2,
            coeff: 3,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_shift_on_temp_then_temp_clear() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::Shift {
                    src: 1,
                    dst: 2,
                    amount: 2,
                    dir: ShiftDir::Left,
                    preserve_src: false,
                    set_dst: false,
                },
                BfIr::MovePtr(1),
                BfIr::Clear,
                BfIr::MovePtr(-1),
            ],
            life_opts()
        ),
        vec![BfIr::Shift {
            src: 0,
            dst: 2,
            amount: 2,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_square_on_temp_then_temp_clear() {
    assert_eq!(
        optimize_with_opts(
            vec![
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
            ],
            life_opts()
        ),
        vec![BfIr::Square {
            src: 0,
            dst: 2,
            preserve_src: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_canonicalizes_muladd_with_aliased_sources_to_square() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::MulAdd {
                lhs: 2,
                rhs: 2,
                dst: 5,
                preserve_lhs: true,
                preserve_rhs: false,
                set_dst: true,
            }],
            life_opts()
        ),
        vec![BfIr::Square {
            src: 2,
            dst: 5,
            preserve_src: false,
            set_dst: true,
        }]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_muladd_on_temp() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::MulAdd {
                    lhs: 1,
                    rhs: 2,
                    dst: 3,
                    preserve_lhs: false,
                    preserve_rhs: true,
                    set_dst: false,
                },
            ],
            life_opts(),
        ),
        vec![
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 1,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::MulAdd {
                lhs: 0,
                rhs: 2,
                dst: 3,
                preserve_lhs: true,
                preserve_rhs: true,
                set_dst: false,
            },
        ]
    );
}

#[test]
fn optimize_collapses_preserved_copy_plus_muladd_on_temp_then_temp_clear() {
    assert_eq!(
        optimize_with_opts(
            vec![
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::MulAdd {
                    lhs: 1,
                    rhs: 2,
                    dst: 3,
                    preserve_lhs: false,
                    preserve_rhs: true,
                    set_dst: false,
                },
                BfIr::MovePtr(1),
                BfIr::Clear,
                BfIr::MovePtr(-1),
            ],
            life_opts(),
        ),
        vec![BfIr::MulAdd {
            lhs: 0,
            rhs: 2,
            dst: 3,
            preserve_lhs: true,
            preserve_rhs: true,
            set_dst: false,
        }]
    );
}

#[test]
fn optimize_chooses_shift_for_combined_in_place_scaling_chain() {
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
                    coeff: 2,
                    preserve_src: true,
                    set_dst: true,
                },
            ],
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
fn optimize_ssa_summarizes_direct_product_loop_to_muladd() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::MovePtr(-1),
            ])],
            life_opts(),
        ),
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
fn optimize_ssa_summarizes_mixed_linear_and_product_loop_to_short_sequence() {
    assert_eq!(
        optimize_with_opts(
            vec![BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Affine {
                    src: 0,
                    dst: 1,
                    coeff: 1,
                    preserve_src: true,
                    set_dst: false,
                },
                BfIr::MovePtr(-1),
                BfIr::MovePtr(2),
                BfIr::Add(2),
                BfIr::MovePtr(-2),
            ])],
            life_opts(),
        ),
        vec![
            BfIr::Shift {
                src: 0,
                dst: 2,
                amount: 1,
                dir: ShiftDir::Left,
                preserve_src: true,
                set_dst: false,
            },
            BfIr::MulAdd {
                lhs: 0,
                rhs: 1,
                dst: 2,
                preserve_lhs: false,
                preserve_rhs: true,
                set_dst: false,
            },
        ]
    );
}
