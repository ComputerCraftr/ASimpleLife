use super::*;

fn required<T>(value: Option<T>) -> T {
    value.or_invariant("required symbolic test value")
}

fn power(offset: Offset, degree: u8) -> SymbolicPolynomial {
    let input = SymbolicPolynomial::input(offset);
    let mut result = SymbolicPolynomial::constant(1);
    for _ in 0..degree {
        result = required(result.multiplied(&input));
    }
    result
}

fn add(mut lhs: SymbolicPolynomial, rhs: &SymbolicPolynomial) -> SymbolicPolynomial {
    lhs.add_assign(rhs).or_invariant("required symbolic sum");
    lhs
}

#[test]
fn degree_two_substitution_preserves_compatibility_variants() {
    let values = BTreeMap::from([(0, required(SymbolicPolynomial::input(0).add_constant(-1)))]);
    let composed = required(SymbolicPolynomial::product(0, 1).substitute(&values));

    assert_eq!(composed.degree(), 2);
    assert_eq!(
        composed.terms,
        BTreeMap::from([
            (SymbolicMonomial::Linear(1), -1),
            (SymbolicMonomial::Product(0, 1), 1),
        ])
    );
}

#[test]
fn constructs_degrees_three_through_eight_with_canonical_exponents() {
    for degree in 3..=SYMBOLIC_DEGREE_MAX {
        let polynomial = power(7, degree);
        assert_eq!(polynomial.degree(), degree);
        assert_eq!(
            polynomial.terms,
            BTreeMap::from([(SymbolicMonomial::Powers(Box::new([(7, degree)])), 1,)])
        );
    }
}

#[test]
fn factors_repeat_canonical_exponents_for_emitters() {
    let polynomial = required(SymbolicPolynomial::product(4, 2).multiplied(&power(2, 3)));
    let monomial = polynomial.terms.keys().next().or_invariant("required term");

    assert_eq!(monomial.factors().collect::<Vec<_>>(), vec![2, 2, 2, 2, 4]);
}

#[test]
fn canonical_terms_merge_commuted_factors_and_cancel() {
    let lhs = required(SymbolicPolynomial::product(2, 1).multiplied(&SymbolicPolynomial::input(1)));
    let rhs = required(SymbolicPolynomial::product(1, 1).multiplied(&SymbolicPolynomial::input(2)));
    let combined = add(lhs, &required(rhs.scaled(-1)));

    assert!(combined.is_zero());
    assert_eq!(combined.cost(), 1);
}

#[test]
fn degree_growth_beyond_eight_fails_closed() {
    assert!(
        power(0, SYMBOLIC_DEGREE_MAX)
            .multiplied(&SymbolicPolynomial::input(1))
            .is_none()
    );
    assert!(
        SymbolicPolynomial::product(0, 0)
            .substitute(&BTreeMap::from([(0, power(1, 5))]))
            .is_none()
    );
}

#[test]
fn degree_sixty_four_intermediates_cancel_before_final_degree_check() {
    let source = add(power(0, 8), &required(power(1, 8).scaled(-1)));
    let replacement = power(2, 8);
    let values = BTreeMap::from([(0, replacement.clone()), (1, replacement)]);

    assert!(required(source.substitute(&values)).is_zero());
}

#[test]
fn cancellation_occurs_before_final_term_limit() {
    let mut first = SymbolicPolynomial::zero();
    for offset in 1..=64 {
        first = required(first.add_scaled_input(offset, 1));
    }
    let mut second = SymbolicPolynomial::input(-100);
    for offset in 1..=63 {
        second = required(second.add_scaled_input(offset, -1));
    }
    let source = add(
        SymbolicPolynomial::input(0),
        &SymbolicPolynomial::input(100),
    );

    let result = required(source.substitute(&BTreeMap::from([(0, first), (100, second)])));
    assert_eq!(
        result,
        add(
            SymbolicPolynomial::input(-100),
            &SymbolicPolynomial::input(64)
        )
    );
}

#[test]
fn shared_substitution_budget_accumulates_across_targets() {
    let mut wide = SymbolicPolynomial::zero();
    for offset in 1..=64 {
        wide = required(wide.add_scaled_input(offset, 1));
    }
    let source = SymbolicPolynomial::input(0);
    let values = BTreeMap::from([(0, wide)]);
    let semantics = required(PolynomialSemantics::new(
        8,
        super::super::CellSign::Unsigned,
        30000,
    ));
    let mut budget = SubstitutionBudget::new();

    let mut admitted = 0;
    for _ in 0..256 {
        if source
            .substitute_with(&values, &semantics, &mut budget)
            .is_some()
        {
            admitted += 1;
        }
    }
    assert_eq!(admitted, 64);
    assert_eq!(budget.products(), SYMBOLIC_PRODUCT_BUDGET);
    assert_eq!(budget.scratch_lanes(), 64);
    assert!(
        source
            .substitute_with(&values, &semantics, &mut budget)
            .is_none()
    );

    let kernels = budget.kernel_accounting();
    assert_eq!(
        kernels.native_avx2_lanes + kernels.native_neon_lanes + kernels.scalar_lanes,
        SYMBOLIC_PRODUCT_BUDGET
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(kernels.native_neon_lanes, SYMBOLIC_PRODUCT_BUDGET);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        assert_eq!(kernels.native_avx2_lanes, SYMBOLIC_PRODUCT_BUDGET);
    }
}

#[test]
fn modular_coefficients_are_centered_for_all_cell_modes_and_widths() {
    for bits in [1, 8, 16, 32, 33, 63] {
        for sign in [
            super::super::CellSign::Unsigned,
            super::super::CellSign::Signed,
        ] {
            let semantics = required(PolynomialSemantics::new(bits, sign, 30000));
            assert_eq!(semantics.normalize_coefficient(-1), Some(-1));
            let residue =
                i64::try_from((1_i128 << bits) - 1).or_invariant("supported modulus fits i64");
            assert_eq!(semantics.normalize_coefficient(residue), Some(-1));
            assert_eq!(
                required(SymbolicPolynomial::input(0).scaled_with(residue, &semantics))
                    .terms
                    .get(&SymbolicMonomial::Linear(0)),
                Some(&-1)
            );
        }
    }
}

#[test]
fn modular_normalization_cancels_tape_aliases() {
    let semantics = required(PolynomialSemantics::new(
        8,
        super::super::CellSign::Unsigned,
        16,
    ));
    let polynomial = add(
        SymbolicPolynomial::input(-1),
        &required(SymbolicPolynomial::input(15).scaled(255)),
    );

    assert!(required(polynomial.normalized(&semantics)).is_zero());
}

#[test]
fn modular_substitution_wraps_products_and_preserves_planner_delta() {
    let semantics = required(PolynomialSemantics::new(
        8,
        super::super::CellSign::Unsigned,
        64,
    ));
    let source = required(SymbolicPolynomial::input(0).add_constant(255));
    let mut budget = SubstitutionBudget::new();
    let result = required(source.substitute_with(&BTreeMap::new(), &semantics, &mut budget));
    assert_eq!(result.additive_delta_for(0), Some(-1));

    let square = SymbolicPolynomial::product(0, 0);
    let replacement = required(SymbolicPolynomial::input(1).add_constant(1));
    let expanded = required(square.substitute_with(
        &BTreeMap::from([(0, replacement)]),
        &semantics,
        &mut budget,
    ));
    assert_eq!(expanded.constant, 1);
    assert_eq!(expanded.terms.get(&SymbolicMonomial::Linear(1)), Some(&2));
    assert_eq!(
        expanded.terms.get(&SymbolicMonomial::Product(1, 1)),
        Some(&1)
    );
}

#[test]
fn shifted_with_wraps_sources_to_tape() {
    let semantics = required(PolynomialSemantics::new(
        16,
        super::super::CellSign::Signed,
        8,
    ));
    assert_eq!(
        required(SymbolicPolynomial::input(7).shifted_with(2, &semantics)),
        SymbolicPolynomial::input(1)
    );
}

#[test]
fn shifted_with_normalizes_extreme_offsets_before_narrowing() {
    let semantics = required(PolynomialSemantics::new(
        8,
        super::super::CellSign::Unsigned,
        30000,
    ));
    let shifted =
        required(SymbolicPolynomial::input(Offset::MAX).shifted_with(Offset::MAX, &semantics));
    let expected = required(
        semantics.normalize_offset_wide(i128::from(Offset::MAX) + i128::from(Offset::MAX)),
    );

    assert_eq!(shifted, SymbolicPolynomial::input(expected));
}

#[test]
fn shifted_with_merges_wrapped_aliases_before_checked_coefficient_overflow() {
    let semantics = required(PolynomialSemantics::new(
        63,
        super::super::CellSign::Unsigned,
        30000,
    ));
    let polynomial = required(
        required(
            required(SymbolicPolynomial::zero().add_scaled_input(0, i64::MAX))
                .add_scaled_input(30000, i64::MAX),
        )
        .add_scaled_input(60000, i64::MAX),
    );

    let shifted = required(polynomial.shifted_with(0, &semantics));

    assert_eq!(shifted.terms.get(&SymbolicMonomial::Linear(0)), Some(&-3));
}
