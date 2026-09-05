//! Bounded evaluation algebra over raw cell bits. Equality of normalized terms
//! is sufficient for reuse, not a complete test of functional equivalence: e.g.
//! `x*x` and `x` agree modulo two without having identical monomials. Unequal
//! polynomials never prove different runtime states or authorize a branch.

#[cfg(test)]
use crate::RequiredExt;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const SYMBOLIC_DEGREE_MAX: u8 = 8;
pub(crate) const SYMBOLIC_TERM_MAX: usize = 64;
pub(crate) const SYMBOLIC_PRODUCT_BUDGET: usize = 4096;

type Offset = crate::bf::BfOffset;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SymbolicMonomial {
    Linear(Offset),
    Product(Offset, Offset),
    Powers(Box<[(Offset, u8)]>),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct SymbolicTerms(Vec<(SymbolicMonomial, i64)>);

impl SymbolicTerms {
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&SymbolicMonomial, &i64)> {
        self.0.iter().map(|(term, coefficient)| (term, coefficient))
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &SymbolicMonomial> {
        self.0.iter().map(|(term, _)| term)
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &i64> {
        self.0.iter().map(|(_, coefficient)| coefficient)
    }

    pub(crate) fn get(&self, term: &SymbolicMonomial) -> Option<&i64> {
        self.position(term).ok().map(|index| &self.0[index].1)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn position(&self, term: &SymbolicMonomial) -> Result<usize, usize> {
        self.0
            .binary_search_by(|(candidate, _)| candidate.cmp(term))
    }

    fn insert(&mut self, term: SymbolicMonomial, coefficient: i64) {
        match self.position(&term) {
            Ok(index) => self.0[index].1 = coefficient,
            Err(index) => self.0.insert(index, (term, coefficient)),
        }
    }

    fn remove(&mut self, term: &SymbolicMonomial) {
        if let Ok(index) = self.position(term) {
            self.0.remove(index);
        }
    }

    fn retain(&mut self, mut keep: impl FnMut(&SymbolicMonomial, &mut i64) -> bool) {
        self.0
            .retain_mut(|(term, coefficient)| keep(term, coefficient));
    }
}

impl From<BTreeMap<SymbolicMonomial, i64>> for SymbolicTerms {
    fn from(terms: BTreeMap<SymbolicMonomial, i64>) -> Self {
        Self(terms.into_iter().collect())
    }
}

impl PartialEq<BTreeMap<SymbolicMonomial, i64>> for SymbolicTerms {
    fn eq(&self, other: &BTreeMap<SymbolicMonomial, i64>) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a> IntoIterator for &'a SymbolicTerms {
    type Item = (&'a SymbolicMonomial, &'a i64);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (SymbolicMonomial, i64)>,
        fn(&(SymbolicMonomial, i64)) -> (&SymbolicMonomial, &i64),
    >;

    fn into_iter(self) -> Self::IntoIter {
        fn references(pair: &(SymbolicMonomial, i64)) -> (&SymbolicMonomial, &i64) {
            (&pair.0, &pair.1)
        }
        self.0.iter().map(references)
    }
}

pub(crate) struct MonomialFactors<'a> {
    powers: MonomialPowers<'a>,
    current: Option<(Offset, u8)>,
}

enum MonomialPowers<'a> {
    Linear(Option<Offset>),
    Product([Offset; 2], usize),
    Powers(std::slice::Iter<'a, (Offset, u8)>),
}

impl Iterator for MonomialFactors<'_> {
    type Item = Offset;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((offset, remaining)) = self.current {
            if remaining > 1 {
                self.current = Some((offset, remaining - 1));
            } else {
                self.current = None;
            }
            return Some(offset);
        }
        match &mut self.powers {
            MonomialPowers::Linear(offset) => offset.take(),
            MonomialPowers::Product(offsets, index) => {
                let offset = offsets.get(*index).copied();
                *index += usize::from(offset.is_some());
                offset
            }
            MonomialPowers::Powers(powers) => {
                let &(offset, exponent) = powers.next()?;
                if exponent > 1 {
                    self.current = Some((offset, exponent - 1));
                }
                Some(offset)
            }
        }
    }
}

impl SymbolicMonomial {
    fn product(lhs: Offset, rhs: Offset) -> Self {
        if lhs <= rhs {
            Self::Product(lhs, rhs)
        } else {
            Self::Product(rhs, lhs)
        }
    }

    fn from_sorted_factors(factors: &[Offset]) -> Option<Self> {
        let degree = u8::try_from(factors.len()).ok()?;
        if degree == 0 {
            return None;
        }
        match factors {
            [offset] => Some(Self::Linear(*offset)),
            [lhs, rhs] => Some(Self::product(*lhs, *rhs)),
            _ => {
                let mut powers = Vec::new();
                for &offset in factors {
                    match powers.last_mut() {
                        Some((last, exponent)) if *last == offset => {
                            *exponent = u8::checked_add(*exponent, 1)?;
                        }
                        _ => powers.push((offset, 1_u8)),
                    }
                }
                Some(Self::Powers(powers.into_boxed_slice()))
            }
        }
    }

    pub(crate) fn factors(&self) -> MonomialFactors<'_> {
        let powers = match self {
            Self::Linear(offset) => MonomialPowers::Linear(Some(*offset)),
            Self::Product(lhs, rhs) => MonomialPowers::Product([*lhs, *rhs], 0),
            Self::Powers(powers) => MonomialPowers::Powers(powers.iter()),
        };
        MonomialFactors {
            powers,
            current: None,
        }
    }

    fn multiplied(&self, other: &Self) -> Option<Self> {
        let mut factors = self.factors().chain(other.factors()).collect::<Vec<_>>();
        factors.sort_unstable();
        Self::from_sorted_factors(&factors)
    }

    fn mapped(&self, mut map: impl FnMut(Offset) -> Option<Offset>) -> Option<Self> {
        let mut factors = self.factors().map(&mut map).collect::<Option<Vec<_>>>()?;
        factors.sort_unstable();
        Self::from_sorted_factors(&factors)
    }

    pub(crate) fn degree(&self) -> u8 {
        match self {
            Self::Linear(_) => 1,
            Self::Product(_, _) => 2,
            Self::Powers(powers) => powers.iter().map(|&(_, exponent)| exponent).sum(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PolynomialSemantics {
    cell_bits: u32,
    cell_sign: super::CellSign,
    tape_len: Offset,
}

impl PolynomialSemantics {
    pub(crate) fn new(cell_bits: u32, cell_sign: super::CellSign, tape_len: usize) -> Option<Self> {
        let tape_len = Offset::try_from(tape_len).ok()?;
        (cell_bits > 0 && cell_bits <= super::MAX_CELL_BITS && tape_len > 0).then_some(Self {
            cell_bits,
            cell_sign,
            tape_len,
        })
    }

    pub(crate) fn normalize_offset(self, offset: Offset) -> Offset {
        let offset = offset.rem_euclid(self.tape_len);
        if offset > self.tape_len / 2 {
            offset - self.tape_len
        } else {
            offset
        }
    }

    #[doc = "source-policy: checked-narrowing-boundary"]
    pub(crate) fn normalize_offset_wide(self, offset: i128) -> Option<Offset> {
        let tape_len = i128::from(self.tape_len);
        let offset = offset.rem_euclid(tape_len);
        let centered = if offset > tape_len / 2 {
            offset - tape_len
        } else {
            offset
        };
        Offset::try_from(centered).ok()
    }

    fn modulus(self) -> i128 {
        1_i128 << self.cell_bits
    }

    #[doc = "source-policy: checked-narrowing-boundary"]
    pub(crate) fn normalize_wide(self, value: i128) -> Option<i64> {
        let modulus = self.modulus();
        let raw = value.rem_euclid(modulus);
        let value = if raw >= modulus / 2 {
            raw - modulus
        } else {
            raw
        };
        i64::try_from(value).ok()
    }

    pub(crate) fn normalize_coefficient(self, value: i64) -> Option<i64> {
        self.normalize_wide(i128::from(value))
    }
}

#[derive(Clone, Copy)]
enum Arithmetic {
    Exact,
    Modular(PolynomialSemantics),
}

impl Arithmetic {
    fn normalize(self, value: i64) -> Option<i64> {
        match self {
            Self::Exact => Some(value),
            Self::Modular(semantics) => semantics.normalize_coefficient(value),
        }
    }

    fn add(self, lhs: i64, rhs: i64) -> Option<i64> {
        match self {
            Self::Exact => lhs.checked_add(rhs),
            Self::Modular(semantics) => semantics.normalize_wide(i128::from(lhs) + i128::from(rhs)),
        }
    }

    fn multiply(self, lhs: i64, rhs: i64) -> Option<i64> {
        match self {
            Self::Exact => lhs.checked_mul(rhs),
            Self::Modular(semantics) => semantics.normalize_wide(i128::from(lhs) * i128::from(rhs)),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct SymbolicPolynomial {
    pub(crate) constant: i64,
    pub(crate) terms: SymbolicTerms,
}

impl SymbolicPolynomial {
    pub(crate) fn zero() -> Self {
        Self::default()
    }

    pub(crate) fn input(offset: Offset) -> Self {
        Self {
            constant: 0,
            terms: BTreeMap::from([(SymbolicMonomial::Linear(offset), 1)]).into(),
        }
    }

    pub(crate) fn constant(value: i64) -> Self {
        Self {
            constant: value,
            terms: SymbolicTerms::default(),
        }
    }

    pub(crate) fn add_constant(mut self, value: i64) -> Option<Self> {
        self.constant = self.constant.checked_add(value)?;
        Some(self)
    }

    pub(crate) fn add_scaled_input(mut self, offset: Offset, coefficient: i64) -> Option<Self> {
        add_term(
            &mut self.terms,
            SymbolicMonomial::Linear(offset),
            coefficient,
            Arithmetic::Exact,
        )?;
        self.finish()
    }

    pub(crate) fn product(lhs: Offset, rhs: Offset) -> Self {
        Self {
            constant: 0,
            terms: BTreeMap::from([(SymbolicMonomial::product(lhs, rhs), 1)]).into(),
        }
    }

    pub(crate) fn add_assign(&mut self, other: &Self) -> Option<()> {
        let constant = self.constant.checked_add(other.constant)?;
        let mut terms = self.terms.clone();
        for (term, &coefficient) in &other.terms {
            add_term(&mut terms, term.clone(), coefficient, Arithmetic::Exact)?;
        }
        if terms.len() > SYMBOLIC_TERM_MAX {
            return None;
        }
        self.constant = constant;
        self.terms = terms;
        Some(())
    }

    #[cfg(test)]
    pub(crate) fn scaled(&self, coefficient: i64) -> Option<Self> {
        let mut result = Self::constant(self.constant.checked_mul(coefficient)?);
        for (term, &value) in &self.terms {
            add_term(
                &mut result.terms,
                term.clone(),
                value.checked_mul(coefficient)?,
                Arithmetic::Exact,
            )?;
        }
        result.finish()
    }

    #[cfg(test)]
    pub(crate) fn scaled_with(
        &self,
        coefficient: i64,
        semantics: &PolynomialSemantics,
    ) -> Option<Self> {
        let arithmetic = Arithmetic::Modular(*semantics);
        let normalized = self.normalized(semantics)?;
        let mut result = Self::constant(arithmetic.multiply(normalized.constant, coefficient)?);
        for (term, &value) in &normalized.terms {
            add_term(
                &mut result.terms,
                term.clone(),
                arithmetic.multiply(value, coefficient)?,
                arithmetic,
            )?;
        }
        result.finish()
    }

    #[cfg(test)]
    pub(crate) fn multiplied(&self, other: &Self) -> Option<Self> {
        let mut budget = SubstitutionBudget::new();
        multiply(self, other, Arithmetic::Exact, &mut budget)?.finish()
    }

    pub(crate) fn shifted(&self, delta: Offset) -> Option<Self> {
        self.map_offsets(|offset| offset.checked_add(delta))
    }

    pub(crate) fn normalized(&self, semantics: &PolynomialSemantics) -> Option<Self> {
        let arithmetic = Arithmetic::Modular(*semantics);
        let mut result = Self::constant(arithmetic.normalize(self.constant)?);
        for (term, &coefficient) in &self.terms {
            add_term(
                &mut result.terms,
                term.mapped(|offset| Some(semantics.normalize_offset(offset)))?,
                coefficient,
                arithmetic,
            )?;
        }
        result.finish()
    }

    pub(crate) fn shifted_with(
        &self,
        delta: Offset,
        semantics: &PolynomialSemantics,
    ) -> Option<Self> {
        self.map_offsets_with(
            |offset| semantics.normalize_offset_wide(i128::from(offset) + i128::from(delta)),
            Arithmetic::Modular(*semantics),
        )
    }

    fn map_offsets(&self, mut map: impl FnMut(Offset) -> Option<Offset>) -> Option<Self> {
        self.map_offsets_with(&mut map, Arithmetic::Exact)
    }

    fn map_offsets_with(
        &self,
        mut map: impl FnMut(Offset) -> Option<Offset>,
        arithmetic: Arithmetic,
    ) -> Option<Self> {
        let mut terms = SymbolicTerms::default();
        for (term, &coefficient) in &self.terms {
            add_term(&mut terms, term.mapped(&mut map)?, coefficient, arithmetic)?;
        }
        Some(Self {
            constant: arithmetic.normalize(self.constant)?,
            terms,
        })
    }

    pub(crate) fn substitute(&self, values: &BTreeMap<Offset, Self>) -> Option<Self> {
        let mut budget = SubstitutionBudget::new();
        self.substitute_arithmetic(values, Arithmetic::Exact, &mut budget)
    }

    pub(crate) fn substitute_with(
        &self,
        values: &BTreeMap<Offset, Self>,
        semantics: &PolynomialSemantics,
        budget: &mut SubstitutionBudget,
    ) -> Option<Self> {
        self.normalized(semantics)?.substitute_arithmetic(
            values,
            Arithmetic::Modular(*semantics),
            budget,
        )
    }

    fn substitute_arithmetic(
        &self,
        values: &BTreeMap<Offset, Self>,
        arithmetic: Arithmetic,
        budget: &mut SubstitutionBudget,
    ) -> Option<Self> {
        let mut result = Self::constant(arithmetic.normalize(self.constant)?);
        for (monomial, &coefficient) in &self.terms {
            if monomial.degree() == 1 {
                let offset = monomial.factors().next()?;
                match values.get(&offset) {
                    Some(value) => {
                        add_scaled(&mut result, value, coefficient, arithmetic, budget)?;
                    }
                    None => {
                        budget.charge(1)?;
                        add_term(
                            &mut result.terms,
                            SymbolicMonomial::Linear(offset),
                            coefficient,
                            arithmetic,
                        )?;
                    }
                }
                continue;
            }
            let mut expanded = Self::constant(1);
            for offset in monomial.factors() {
                expanded = match values.get(&offset) {
                    Some(value) => multiply(&expanded, value, arithmetic, budget)?,
                    None => multiply_by_input(&expanded, offset, arithmetic, budget)?,
                };
            }
            add_scaled(&mut result, &expanded, coefficient, arithmetic, budget)?;
        }
        result.finish()
    }

    pub(crate) fn sources(&self) -> BTreeSet<Offset> {
        self.terms
            .keys()
            .flat_map(SymbolicMonomial::factors)
            .collect()
    }

    pub(crate) fn degree(&self) -> u8 {
        self.terms
            .keys()
            .map(SymbolicMonomial::degree)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn additive_delta_for(&self, offset: Offset) -> Option<i64> {
        (self.terms.len() == 1 && self.terms.get(&SymbolicMonomial::Linear(offset)) == Some(&1))
            .then_some(self.constant)
    }

    #[cfg(test)]
    pub(crate) fn cost(&self) -> usize {
        1 + self.terms.len() + usize::from(self.degree())
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.constant == 0 && self.terms.is_empty()
    }

    fn finish(mut self) -> Option<Self> {
        self.terms.retain(|_, coefficient| *coefficient != 0);
        (self.degree() <= SYMBOLIC_DEGREE_MAX && self.terms.len() <= SYMBOLIC_TERM_MAX)
            .then_some(self)
    }
}

const KERNEL_SCRATCH_LANES: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubstitutionBudget {
    products: usize,
    scratch_lhs: [u64; KERNEL_SCRATCH_LANES],
    scratch_rhs: [u64; KERNEL_SCRATCH_LANES],
    scratch_output: [u64; KERNEL_SCRATCH_LANES],
    kernels: super::coefficient_kernels::CoefficientKernelAccounting,
}

impl Default for SubstitutionBudget {
    fn default() -> Self {
        Self {
            products: 0,
            scratch_lhs: [0; KERNEL_SCRATCH_LANES],
            scratch_rhs: [0; KERNEL_SCRATCH_LANES],
            scratch_output: [0; KERNEL_SCRATCH_LANES],
            kernels: super::coefficient_kernels::CoefficientKernelAccounting::default(),
        }
    }
}

impl SubstitutionBudget {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn products(&self) -> usize {
        self.products
    }

    pub(crate) fn kernel_accounting(
        &self,
    ) -> super::coefficient_kernels::CoefficientKernelAccounting {
        self.kernels
    }

    pub(crate) fn scratch_lanes(&self) -> usize {
        KERNEL_SCRATCH_LANES
    }

    fn charge(&mut self, count: usize) -> Option<()> {
        let products = self.products.checked_add(count)?;
        if products > SYMBOLIC_PRODUCT_BUDGET {
            return None;
        }
        self.products = products;
        Some(())
    }

    fn record_kernel(
        &mut self,
        accounting: super::coefficient_kernels::CoefficientKernelAccounting,
    ) -> Option<()> {
        self.kernels.native_avx2_lanes = self
            .kernels
            .native_avx2_lanes
            .checked_add(accounting.native_avx2_lanes)?;
        self.kernels.native_neon_lanes = self
            .kernels
            .native_neon_lanes
            .checked_add(accounting.native_neon_lanes)?;
        self.kernels.scalar_lanes = self
            .kernels
            .scalar_lanes
            .checked_add(accounting.scalar_lanes)?;
        Some(())
    }

    fn record_scalar(&mut self, lanes: usize) -> Option<()> {
        self.kernels.scalar_lanes = self.kernels.scalar_lanes.checked_add(lanes)?;
        Some(())
    }
}

fn add_term(
    terms: &mut SymbolicTerms,
    term: SymbolicMonomial,
    coefficient: i64,
    arithmetic: Arithmetic,
) -> Option<()> {
    let coefficient = arithmetic.normalize(coefficient)?;
    if coefficient == 0 {
        return Some(());
    }
    let next = arithmetic.add(terms.get(&term).copied().unwrap_or(0), coefficient)?;
    if next == 0 {
        terms.remove(&term);
    } else {
        terms.insert(term, next);
    }
    Some(())
}

fn component_count(polynomial: &SymbolicPolynomial) -> usize {
    polynomial.terms.len() + usize::from(polynomial.constant != 0)
}

fn add_scaled(
    result: &mut SymbolicPolynomial,
    value: &SymbolicPolynomial,
    scale: i64,
    arithmetic: Arithmetic,
    budget: &mut SubstitutionBudget,
) -> Option<()> {
    budget.charge(component_count(value))?;
    result.constant =
        arithmetic.add(result.constant, arithmetic.multiply(value.constant, scale)?)?;
    match arithmetic {
        Arithmetic::Exact => {
            for (term, &coefficient) in &value.terms {
                add_term(
                    &mut result.terms,
                    term.clone(),
                    arithmetic.multiply(coefficient, scale)?,
                    arithmetic,
                )?;
            }
        }
        Arithmetic::Modular(semantics) => {
            let mut start = 0;
            while start < value.terms.len() {
                let lanes = (value.terms.len() - start).min(KERNEL_SCRATCH_LANES);
                if lanes < 4 {
                    budget.record_scalar(lanes)?;
                    for (term, &coefficient) in value.terms.iter().skip(start).take(lanes) {
                        add_term(
                            &mut result.terms,
                            term.clone(),
                            arithmetic.multiply(coefficient, scale)?,
                            arithmetic,
                        )?;
                    }
                } else {
                    for lane in 0..lanes {
                        budget.scratch_lhs[lane] = i64_to_ring(value.terms.0[start + lane].1);
                    }
                    let accounting = super::coefficient_kernels::scale_low64_mod(
                        &mut budget.scratch_output[..lanes],
                        &budget.scratch_lhs[..lanes],
                        i64_to_ring(scale),
                        semantics.cell_bits,
                    )
                    .ok()?;
                    budget.record_kernel(accounting)?;
                    for lane in 0..lanes {
                        let coefficient =
                            semantics.normalize_wide(i128::from(budget.scratch_output[lane]))?;
                        add_term(
                            &mut result.terms,
                            value.terms.0[start + lane].0.clone(),
                            coefficient,
                            arithmetic,
                        )?;
                    }
                }
                start += lanes;
            }
        }
    }
    Some(())
}

fn multiply(
    lhs: &SymbolicPolynomial,
    rhs: &SymbolicPolynomial,
    arithmetic: Arithmetic,
    budget: &mut SubstitutionBudget,
) -> Option<SymbolicPolynomial> {
    budget.charge(component_count(lhs).checked_mul(component_count(rhs))?)?;
    let mut result = SymbolicPolynomial::constant(arithmetic.multiply(lhs.constant, rhs.constant)?);
    for (term, &coefficient) in &lhs.terms {
        add_term(
            &mut result.terms,
            term.clone(),
            arithmetic.multiply(coefficient, rhs.constant)?,
            arithmetic,
        )?;
    }
    for (term, &coefficient) in &rhs.terms {
        add_term(
            &mut result.terms,
            term.clone(),
            arithmetic.multiply(coefficient, lhs.constant)?,
            arithmetic,
        )?;
    }
    match arithmetic {
        Arithmetic::Exact => {
            for (left, &left_coefficient) in &lhs.terms {
                for (right, &right_coefficient) in &rhs.terms {
                    add_term(
                        &mut result.terms,
                        left.multiplied(right)?,
                        arithmetic.multiply(left_coefficient, right_coefficient)?,
                        arithmetic,
                    )?;
                }
            }
        }
        Arithmetic::Modular(semantics) => {
            let count = lhs.terms.len().checked_mul(rhs.terms.len())?;
            let mut start = 0;
            while start < count {
                let lanes = (count - start).min(KERNEL_SCRATCH_LANES);
                if lanes < 4 {
                    budget.record_scalar(lanes)?;
                    for index in start..start + lanes {
                        let left = &lhs.terms.0[index / rhs.terms.len()];
                        let right = &rhs.terms.0[index % rhs.terms.len()];
                        add_term(
                            &mut result.terms,
                            left.0.multiplied(&right.0)?,
                            arithmetic.multiply(left.1, right.1)?,
                            arithmetic,
                        )?;
                    }
                } else {
                    let mut pairs = [(&lhs.terms.0[0].0, &rhs.terms.0[0].0); KERNEL_SCRATCH_LANES];
                    for (lane, pair) in pairs[..lanes].iter_mut().enumerate() {
                        let index = start + lane;
                        let left = &lhs.terms.0[index / rhs.terms.len()];
                        let right = &rhs.terms.0[index % rhs.terms.len()];
                        *pair = (&left.0, &right.0);
                        budget.scratch_lhs[lane] = i64_to_ring(left.1);
                        budget.scratch_rhs[lane] = i64_to_ring(right.1);
                    }
                    let accounting = super::coefficient_kernels::multiply_low64_mod(
                        &mut budget.scratch_output[..lanes],
                        &budget.scratch_lhs[..lanes],
                        &budget.scratch_rhs[..lanes],
                        semantics.cell_bits,
                    )
                    .ok()?;
                    budget.record_kernel(accounting)?;
                    for (lane, (left, right)) in pairs[..lanes].iter().enumerate() {
                        let coefficient =
                            semantics.normalize_wide(i128::from(budget.scratch_output[lane]))?;
                        add_term(
                            &mut result.terms,
                            left.multiplied(right)?,
                            coefficient,
                            arithmetic,
                        )?;
                    }
                }
                start += lanes;
            }
        }
    }
    Some(result)
}

fn i64_to_ring(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

fn multiply_by_input(
    value: &SymbolicPolynomial,
    offset: Offset,
    arithmetic: Arithmetic,
    budget: &mut SubstitutionBudget,
) -> Option<SymbolicPolynomial> {
    budget.charge(component_count(value))?;
    let input = SymbolicMonomial::Linear(offset);
    let mut result = SymbolicPolynomial::zero();
    add_term(&mut result.terms, input.clone(), value.constant, arithmetic)?;
    for (term, &coefficient) in &value.terms {
        add_term(
            &mut result.terms,
            term.multiplied(&input)?,
            coefficient,
            arithmetic,
        )?;
    }
    Some(result)
}

#[cfg(test)]
mod tests;
