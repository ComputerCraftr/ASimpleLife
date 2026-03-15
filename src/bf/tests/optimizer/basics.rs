use super::*;

#[test]
fn optimizer_combines_parsed_runs() {
    assert_eq!(parse_and_opt("+++"), vec![BfIr::Add(3)]);
}

#[test]
fn in_place_identity_affine_normalization_terminates_and_is_idempotent() {
    let program = vec![BfIr::Affine {
        src: 0,
        dst: 0,
        coeff: 1,
        preserve_src: true,
        set_dst: true,
    }];
    let once = optimize_with_opts(program, default_c_opts());
    let twice = optimize_with_opts(once.clone(), default_c_opts());

    assert_eq!(
        twice, once,
        "identity normalization must reach a fixed point"
    );
}

#[test]
fn pointer_move_combining_keeps_overflowing_moves_separate() {
    let optimized = optimize_with_opts(
        vec![BfIr::MovePtr(crate::bf::BfOffset::MAX), BfIr::MovePtr(1)],
        default_c_opts(),
    );

    assert_eq!(
        optimized,
        vec![BfIr::MovePtr(crate::bf::BfOffset::MAX), BfIr::MovePtr(1)]
    );
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
fn clear_then_add_preserves_overwrite_before_delta() {
    assert_eq!(parse_and_opt("[-]+++"), vec![BfIr::Clear, BfIr::Add(3)]);
}

#[test]
fn repeated_clear_collapses() {
    assert_eq!(parse_and_opt("[-][-]"), vec![BfIr::Clear]);
}

#[test]
fn distribute_single_target_add() {
    assert_eq!(
        parse_and_opt("[->+<]"),
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
fn distribute_single_target_mul() {
    assert_eq!(
        parse_and_opt("[->+++<]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 3,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn distribute_single_target_sub() {
    assert_eq!(
        parse_and_opt("[->-<]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: -1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn distribute_single_target_power_of_two_becomes_shift_left() {
    assert_eq!(
        parse_and_opt("[->++<]"),
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
fn distribute_multiple_targets() {
    assert_eq!(
        parse_and_opt("[->+>++<<]"),
        vec![BfIr::Distribute {
            targets: vec![(1, 1), (2, 2)],
            preserve_src: false,
        }]
    );
}

#[test]
fn distribute_to_left() {
    assert_eq!(
        parse_and_opt("[<+>-]"),
        vec![BfIr::Affine {
            src: 0,
            dst: -1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}

#[test]
fn distribute_then_clear_keeps_distribute() {
    assert_eq!(
        parse_and_opt("[->+<][-]"),
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
fn clear_then_distribute_keeps_clear() {
    assert_eq!(parse_and_opt("[-][->+<]"), vec![BfIr::Clear]);
}

#[test]
fn repeated_distribute_keeps_first_overwrite() {
    assert_eq!(
        parse_and_opt("[->+<][->++<]"),
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
fn affine_loop_with_extra_balanced_motion_still_becomes_distribute() {
    assert_eq!(
        parse_and_opt("[->+>+<+<]"),
        vec![BfIr::Distribute {
            targets: vec![(1, 2), (2, 1)],
            preserve_src: false,
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
fn affine_loop_with_non_unit_source_delta_summarizes_to_modular_affine() {
    assert_eq!(
        parse_and_opt("[--->+<]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: -85,
            preserve_src: false,
            set_dst: false,
        }]
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
fn outer_guarded_single_target_distribute_loop_summarizes_to_affine() {
    assert_eq!(
        parse_and_opt("[[->+<]]"),
        vec![BfIr::Affine {
            src: 0,
            dst: 1,
            coeff: 1,
            preserve_src: false,
            set_dst: false,
        }]
    );
}
