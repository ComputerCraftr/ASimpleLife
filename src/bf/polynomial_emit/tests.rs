use super::*;
use crate::bf::symbolic::{SymbolicMonomial, SymbolicPolynomial};
use std::collections::{BTreeMap, BTreeSet};

struct TestBackend;

impl PolynomialCBackend for TestBackend {
    fn source(&self, offset: BfOffset) -> String {
        format!("cell({offset})")
    }

    fn target(&self, offset: BfOffset) -> String {
        format!("cell({offset})")
    }

    fn wrap_add(&self, lhs: &str, rhs: &str) -> String {
        format!("add({lhs}, {rhs})")
    }

    fn wrap_mul(&self, lhs: &str, rhs: &str) -> String {
        format!("mul({lhs}, {rhs})")
    }

    fn zero_region(&self, start: BfOffset, len: BfOffset) -> String {
        format!("zero({start}, {len});")
    }
}

fn transfer(
    effects: BTreeMap<BfOffset, SymbolicPolynomial>,
    reads: BTreeSet<BfOffset>,
) -> SymbolicTransfer {
    SymbolicTransfer {
        ptr_delta: 0,
        effects,
        reads,
        may_input: false,
        may_output: false,
        may_diverge: false,
        unknown: false,
    }
}

fn emit(transfer: &SymbolicTransfer) -> Option<Vec<String>> {
    emit_symbolic_transfer(
        transfer,
        &TestBackend,
        &mut PolynomialEmissionBudget::default(),
        0,
    )
}

#[test]
fn exact_emission_snapshots_sources_and_reuses_a_square() {
    let square = SymbolicMonomial::Product(1, 1);
    let transfer = transfer(
        BTreeMap::from([
            (
                1,
                SymbolicPolynomial {
                    constant: 3,
                    terms: BTreeMap::from([(square.clone(), 2)]).into(),
                },
            ),
            (
                2,
                SymbolicPolynomial {
                    constant: 0,
                    terms: BTreeMap::from([(square, 1)]).into(),
                },
            ),
        ]),
        BTreeSet::from([1]),
    );

    assert_eq!(
        emit(&transfer).or_invariant("small square transfer fits emission limits"),
        [
            "{",
            "int64_t bf_poly_src_0 = cell(1);",
            "int64_t bf_poly_product_0 = mul(bf_poly_src_0, bf_poly_src_0);",
            "int64_t bf_poly_value_0 = add(INT64_C(0), INT64_C(3));",
            "bf_poly_value_0 = add(bf_poly_value_0, mul(bf_poly_product_0, INT64_C(2)));",
            "cell(1) = bf_poly_value_0;",
            "int64_t bf_poly_value_1 = bf_poly_product_0;",
            "cell(2) = bf_poly_value_1;",
            "}",
        ]
    );
}

#[test]
fn exact_emission_coalesces_adjacent_zero_effects() {
    let transfer = transfer(
        BTreeMap::from([
            (-2, SymbolicPolynomial::zero()),
            (-1, SymbolicPolynomial::zero()),
            (1, SymbolicPolynomial::zero()),
        ]),
        BTreeSet::new(),
    );

    assert_eq!(
        emit(&transfer).or_invariant("three clears fit emission limits"),
        ["{", "zero(-2, 2);", "zero(1, 1);", "}"]
    );
}

#[test]
fn centered_unsigned_constant_is_wrapped_before_write() {
    let transfer = transfer(
        BTreeMap::from([(0, SymbolicPolynomial::constant(-1))]),
        BTreeSet::new(),
    );

    assert_eq!(
        emit(&transfer).or_invariant("constant transfer fits emission limits"),
        [
            "{",
            "int64_t bf_poly_value_0 = add(INT64_C(0), INT64_C(-1));",
            "cell(0) = bf_poly_value_0;",
            "}",
        ]
    );
}

#[test]
fn eighth_power_uses_three_repeated_squares() {
    let transfer = transfer(
        BTreeMap::from([(
            1,
            SymbolicPolynomial {
                constant: 0,
                terms: BTreeMap::from([(
                    SymbolicMonomial::Powers(vec![(0, 8)].into_boxed_slice()),
                    1,
                )])
                .into(),
            },
        )]),
        BTreeSet::from([0]),
    );

    assert_eq!(evaluation_cost_breakdown(&transfer).multiplications, 3);
    assert_eq!(
        emit(&transfer).or_invariant("eighth power fits emission limits"),
        [
            "{",
            "int64_t bf_poly_src_0 = cell(0);",
            "int64_t bf_poly_product_0 = mul(bf_poly_src_0, bf_poly_src_0);",
            "int64_t bf_poly_product_1 = mul(bf_poly_product_0, bf_poly_product_0);",
            "int64_t bf_poly_product_2 = mul(bf_poly_product_1, bf_poly_product_1);",
            "int64_t bf_poly_value_0 = bf_poly_product_2;",
            "cell(1) = bf_poly_value_0;",
            "}",
        ]
    );
}

#[test]
fn four_independent_squares_use_one_depth_batch() {
    let effects = (0..4)
        .map(|offset| {
            (
                offset + 4,
                SymbolicPolynomial {
                    constant: 0,
                    terms: BTreeMap::from([(SymbolicMonomial::Product(offset, offset), 1)]).into(),
                },
            )
        })
        .collect();
    let transfer = transfer(effects, BTreeSet::from([0, 1, 2, 3]));
    let emitted = emit(&transfer).or_invariant("four squares fit emission limits");

    assert_eq!(
        &emitted[5..9],
        [
            "const int64_t bf_poly_lhs_1[4] = { bf_poly_src_0, bf_poly_src_1, bf_poly_src_2, bf_poly_src_3 };",
            "const int64_t bf_poly_rhs_1[4] = { bf_poly_src_0, bf_poly_src_1, bf_poly_src_2, bf_poly_src_3 };",
            "int64_t bf_poly_batch_1[4];",
            "bf_polynomial_mul_batch(bf_poly_lhs_1, bf_poly_rhs_1, bf_poly_batch_1, 4U, BF_CELL_BITS, BF_SIGNED_CELLS);",
        ]
    );
    assert!(emitted.contains(&"int64_t bf_poly_value_3 = bf_poly_batch_1[3];".to_string()));
}

#[test]
fn exact_emission_factors_common_source_only_when_strictly_cheaper() {
    let transfer = transfer(
        BTreeMap::from([(
            3,
            SymbolicPolynomial {
                constant: 0,
                terms: BTreeMap::from([
                    (SymbolicMonomial::Product(0, 1), 1),
                    (SymbolicMonomial::Product(0, 2), 1),
                ])
                .into(),
            },
        )]),
        BTreeSet::from([0, 1, 2]),
    );

    assert_eq!(evaluation_cost_breakdown(&transfer).multiplications, 1);
    assert_eq!(
        emit(&transfer).or_invariant("factored transfer fits emission limits"),
        [
            "{",
            "int64_t bf_poly_src_0 = cell(0);",
            "int64_t bf_poly_src_1 = cell(1);",
            "int64_t bf_poly_src_2 = cell(2);",
            "int64_t bf_poly_factor_value_0 = bf_poly_src_1;",
            "bf_poly_factor_value_0 = add(bf_poly_factor_value_0, bf_poly_src_2);",
            "int64_t bf_poly_factored_0 = mul(bf_poly_src_0, bf_poly_factor_value_0);",
            "int64_t bf_poly_value_0 = bf_poly_factored_0;",
            "cell(3) = bf_poly_value_0;",
            "}",
        ]
    );
}

#[test]
fn oversized_polynomial_is_rejected_by_preflight() {
    let terms = (0..=super::super::symbolic::SYMBOLIC_TERM_MAX)
        .map(|offset| {
            (
                SymbolicMonomial::Linear(
                    BfOffset::try_from(offset).or_invariant("test offset fits BF address"),
                ),
                1,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let transfer = transfer(
        BTreeMap::from([(
            0,
            SymbolicPolynomial {
                constant: 0,
                terms: terms.into(),
            },
        )]),
        (0..=super::super::symbolic::SYMBOLIC_TERM_MAX)
            .map(|offset| BfOffset::try_from(offset).or_invariant("test offset fits BF address"))
            .collect(),
    );

    assert!(!can_emit_transfer(&transfer));
    assert!(emit(&transfer).is_none());
}

#[test]
fn oversized_snapshot_is_rejected_before_emission() {
    let reads = (0..=EMIT_SNAPSHOT_CELL_MAX)
        .map(|offset| BfOffset::try_from(offset).or_invariant("test offset fits BF address"))
        .collect();
    let transfer = transfer(BTreeMap::new(), reads);

    assert!(!can_emit_transfer(&transfer));
    assert!(emit(&transfer).is_none());
}

struct HugeBackend;

impl PolynomialCBackend for HugeBackend {
    fn source(&self, _offset: BfOffset) -> String {
        "x".repeat(EMIT_SOURCE_BYTE_MAX)
    }

    fn target(&self, offset: BfOffset) -> String {
        TestBackend.target(offset)
    }

    fn wrap_add(&self, lhs: &str, rhs: &str) -> String {
        TestBackend.wrap_add(lhs, rhs)
    }

    fn wrap_mul(&self, lhs: &str, rhs: &str) -> String {
        TestBackend.wrap_mul(lhs, rhs)
    }

    fn zero_region(&self, start: BfOffset, len: BfOffset) -> String {
        TestBackend.zero_region(start, len)
    }
}

struct BudgetBackend;

impl PolynomialCBackend for BudgetBackend {
    fn source(&self, _offset: BfOffset) -> String {
        "x".repeat(EMIT_SOURCE_BYTE_MAX / 2)
    }

    fn target(&self, offset: BfOffset) -> String {
        TestBackend.target(offset)
    }

    fn wrap_add(&self, lhs: &str, rhs: &str) -> String {
        TestBackend.wrap_add(lhs, rhs)
    }

    fn wrap_mul(&self, lhs: &str, rhs: &str) -> String {
        TestBackend.wrap_mul(lhs, rhs)
    }

    fn zero_region(&self, start: BfOffset, len: BfOffset) -> String {
        TestBackend.zero_region(start, len)
    }
}

#[test]
fn exact_source_byte_check_rejects_before_publication() {
    let transfer = transfer(
        BTreeMap::from([(1, SymbolicPolynomial::input(0))]),
        BTreeSet::from([0]),
    );

    assert!(can_emit_transfer(&transfer));
    assert!(
        emit_symbolic_transfer(
            &transfer,
            &HugeBackend,
            &mut PolynomialEmissionBudget::default(),
            0,
        )
        .is_none()
    );
}

#[test]
fn malformed_zero_exponent_is_rejected_without_dag_expansion() {
    let transfer = transfer(
        BTreeMap::from([(
            1,
            SymbolicPolynomial {
                constant: 0,
                terms: BTreeMap::from([(
                    SymbolicMonomial::Powers(vec![(0, 0)].into_boxed_slice()),
                    1,
                )])
                .into(),
            },
        )]),
        BTreeSet::from([0]),
    );

    assert!(!can_emit_transfer(&transfer));
    assert!(emit(&transfer).is_none());
}

#[test]
fn compilation_source_budget_rejects_whole_later_region() {
    let transfer = transfer(
        BTreeMap::from([(1, SymbolicPolynomial::input(0))]),
        BTreeSet::from([0]),
    );
    let mut budget = PolynomialEmissionBudget::default();
    let mut admitted = 0;
    while emit_symbolic_transfer(&transfer, &BudgetBackend, &mut budget, 0).is_some() {
        admitted += 1;
    }

    assert!(admitted > 0);
    assert!(admitted < 64);
    assert!(emit_symbolic_transfer(&transfer, &BudgetBackend, &mut budget, 0).is_none());
}
