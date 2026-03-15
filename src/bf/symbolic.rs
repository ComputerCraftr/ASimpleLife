#[cfg(test)]
use crate::RequiredExt;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const SYMBOLIC_TERM_MAX: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SymbolicMonomial {
    Linear(crate::bf::BfOffset),
    Product(crate::bf::BfOffset, crate::bf::BfOffset),
}

impl SymbolicMonomial {
    fn product(lhs: crate::bf::BfOffset, rhs: crate::bf::BfOffset) -> Self {
        if lhs <= rhs {
            Self::Product(lhs, rhs)
        } else {
            Self::Product(rhs, lhs)
        }
    }

    fn shifted(self, delta: crate::bf::BfOffset) -> Option<Self> {
        Some(match self {
            Self::Linear(offset) => Self::Linear(offset.checked_add(delta)?),
            Self::Product(lhs, rhs) => {
                Self::product(lhs.checked_add(delta)?, rhs.checked_add(delta)?)
            }
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct SymbolicPolynomial {
    pub(crate) constant: i64,
    pub(crate) terms: BTreeMap<SymbolicMonomial, i64>,
}

impl SymbolicPolynomial {
    pub(crate) fn zero() -> Self {
        Self::default()
    }

    pub(crate) fn input(offset: crate::bf::BfOffset) -> Self {
        Self {
            constant: 0,
            terms: BTreeMap::from([(SymbolicMonomial::Linear(offset), 1)]),
        }
    }

    pub(crate) fn constant(value: i64) -> Self {
        Self {
            constant: value,
            terms: BTreeMap::new(),
        }
    }

    pub(crate) fn add_constant(mut self, value: i64) -> Option<Self> {
        self.constant = self.constant.checked_add(value)?;
        Some(self)
    }

    pub(crate) fn add_scaled_input(
        mut self,
        offset: crate::bf::BfOffset,
        coeff: i64,
    ) -> Option<Self> {
        self.add_term(SymbolicMonomial::Linear(offset), coeff)?;
        Some(self)
    }

    pub(crate) fn product(lhs: crate::bf::BfOffset, rhs: crate::bf::BfOffset) -> Self {
        Self {
            constant: 0,
            terms: BTreeMap::from([(SymbolicMonomial::product(lhs, rhs), 1)]),
        }
    }

    fn add_term(&mut self, term: SymbolicMonomial, coeff: i64) -> Option<()> {
        if coeff == 0 {
            return Some(());
        }
        let next = self
            .terms
            .get(&term)
            .copied()
            .unwrap_or(0)
            .checked_add(coeff)?;
        if next == 0 {
            self.terms.remove(&term);
        } else {
            self.terms.insert(term, next);
        }
        (self.terms.len() <= SYMBOLIC_TERM_MAX).then_some(())
    }

    pub(crate) fn add_assign(&mut self, other: &Self) -> Option<()> {
        self.constant = self.constant.checked_add(other.constant)?;
        for (&term, &coeff) in &other.terms {
            self.add_term(term, coeff)?;
        }
        Some(())
    }

    pub(crate) fn scaled(&self, coeff: i64) -> Option<Self> {
        let mut result = Self::constant(self.constant.checked_mul(coeff)?);
        for (&term, &value) in &self.terms {
            result.add_term(term, value.checked_mul(coeff)?)?;
        }
        Some(result)
    }

    pub(crate) fn multiplied(&self, other: &Self) -> Option<Self> {
        let mut result = Self::constant(self.constant.checked_mul(other.constant)?);
        for (&term, &coeff) in &self.terms {
            result.add_term(term, coeff.checked_mul(other.constant)?)?;
        }
        for (&term, &coeff) in &other.terms {
            result.add_term(term, coeff.checked_mul(self.constant)?)?;
        }
        for (&left, &left_coeff) in &self.terms {
            for (&right, &right_coeff) in &other.terms {
                let (SymbolicMonomial::Linear(lhs), SymbolicMonomial::Linear(rhs)) = (left, right)
                else {
                    return None;
                };
                result.add_term(
                    SymbolicMonomial::product(lhs, rhs),
                    left_coeff.checked_mul(right_coeff)?,
                )?;
            }
        }
        Some(result)
    }

    pub(crate) fn shifted(&self, delta: crate::bf::BfOffset) -> Option<Self> {
        let mut terms = BTreeMap::new();
        for (&term, &coeff) in &self.terms {
            terms.insert(term.shifted(delta)?, coeff);
        }
        Some(Self {
            constant: self.constant,
            terms,
        })
    }

    pub(crate) fn substitute(
        &self,
        values: &BTreeMap<crate::bf::BfOffset, SymbolicPolynomial>,
    ) -> Option<Self> {
        let value = |offset| {
            values
                .get(&offset)
                .cloned()
                .unwrap_or_else(|| Self::input(offset))
        };
        let mut result = Self::constant(self.constant);
        for (&term, &coeff) in &self.terms {
            let expanded = match term {
                SymbolicMonomial::Linear(offset) => value(offset),
                SymbolicMonomial::Product(lhs, rhs) => value(lhs).multiplied(&value(rhs))?,
            };
            result.add_assign(&expanded.scaled(coeff)?)?;
        }
        Some(result)
    }

    pub(crate) fn sources(&self) -> BTreeSet<crate::bf::BfOffset> {
        let mut sources = BTreeSet::new();
        for term in self.terms.keys() {
            match *term {
                SymbolicMonomial::Linear(offset) => {
                    sources.insert(offset);
                }
                SymbolicMonomial::Product(lhs, rhs) => {
                    sources.extend([lhs, rhs]);
                }
            }
        }
        sources
    }

    pub(crate) fn degree(&self) -> u8 {
        if self
            .terms
            .keys()
            .any(|term| matches!(term, SymbolicMonomial::Product(_, _)))
        {
            2
        } else if self.terms.is_empty() {
            0
        } else {
            1
        }
    }

    pub(crate) fn additive_delta_for(&self, offset: crate::bf::BfOffset) -> Option<i64> {
        (self.terms.len() == 1 && self.terms.get(&SymbolicMonomial::Linear(offset)) == Some(&1))
            .then_some(self.constant)
    }

    pub(crate) fn cost(&self) -> usize {
        1 + self.terms.len() + usize::from(self.degree())
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.constant == 0 && self.terms.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_two_substitution_composes_invariant_products() {
        let product = SymbolicPolynomial::product(0, 1);
        let values = BTreeMap::from([
            (
                0,
                SymbolicPolynomial::input(0)
                    .add_constant(-1)
                    .or_invariant("required value"),
            ),
            (1, SymbolicPolynomial::input(1)),
        ]);

        let composed = product.substitute(&values).or_invariant("required value");

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
    fn degree_growth_beyond_two_fails_closed() {
        let square = SymbolicPolynomial::product(0, 0);
        let values = BTreeMap::from([(0, SymbolicPolynomial::product(1, 2))]);

        assert_eq!(square.substitute(&values), None);
    }

    #[test]
    fn canonical_terms_merge_commuted_products_and_cancel_coefficients() {
        let mut polynomial = SymbolicPolynomial::product(2, 1);
        polynomial
            .add_assign(
                &SymbolicPolynomial::product(1, 2)
                    .scaled(-1)
                    .or_invariant("required value"),
            )
            .or_invariant("required value");

        assert!(polynomial.is_zero());
        assert_eq!(polynomial.cost(), 1);
    }
}
