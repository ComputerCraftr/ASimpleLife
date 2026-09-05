use super::BfOffset;
use super::summary::SymbolicTransfer;
use super::symbolic::{SYMBOLIC_DEGREE_MAX, SymbolicMonomial, SymbolicPolynomial};
use crate::RequiredExt;
use std::collections::{BTreeMap, HashMap};

const EMIT_DAG_NODE_MAX: usize = 2_048;
const EMIT_C_TEMPORARY_MAX: usize = 1_024;
const EMIT_SNAPSHOT_CELL_MAX: usize = 256;
const EMIT_STACK_BYTE_MAX: usize = 8 * 1024;
const EMIT_SOURCE_BYTE_MAX: usize = 256 * 1024;
const EMIT_COMPILATION_DAG_NODE_MAX: usize = 32 * 1024;
const EMIT_COMPILATION_SOURCE_BYTE_MAX: usize = 4 * 1024 * 1024;
const C_CELL_BYTES: usize = size_of::<i64>();

#[derive(Clone, Default)]
pub(crate) struct PolynomialEmissionBudget {
    dag_nodes: usize,
    source_bytes: usize,
}

impl PolynomialEmissionBudget {
    fn reserve(&mut self, dag_nodes: usize, source_bytes: usize) -> Option<()> {
        let next_dag_nodes = self.dag_nodes.checked_add(dag_nodes)?;
        let next_source_bytes = self.source_bytes.checked_add(source_bytes)?;
        if next_dag_nodes > EMIT_COMPILATION_DAG_NODE_MAX
            || next_source_bytes > EMIT_COMPILATION_SOURCE_BYTE_MAX
        {
            return None;
        }
        self.dag_nodes = next_dag_nodes;
        self.source_bytes = next_source_bytes;
        Some(())
    }

    pub(crate) fn reserve_source_bytes(&mut self, source_bytes: usize) -> Option<()> {
        self.reserve(0, source_bytes)
    }
}

pub(crate) trait PolynomialCBackend {
    fn source(&self, offset: BfOffset) -> String;
    fn target(&self, offset: BfOffset) -> String;
    fn wrap_add(&self, lhs: &str, rhs: &str) -> String;
    fn wrap_mul(&self, lhs: &str, rhs: &str) -> String;
    fn zero_region(&self, start: BfOffset, len: BfOffset) -> String;
    fn mul_batch(&self, depth: usize, lhs: &[String], rhs: &[String]) -> Vec<String> {
        let count = lhs.len();
        vec![
            format!(
                "const int64_t bf_poly_lhs_{depth}[{count}] = {{ {} }};",
                lhs.join(", ")
            ),
            format!(
                "const int64_t bf_poly_rhs_{depth}[{count}] = {{ {} }};",
                rhs.join(", ")
            ),
            format!("int64_t bf_poly_batch_{depth}[{count}];"),
            format!(
                "bf_polynomial_mul_batch(bf_poly_lhs_{depth}, bf_poly_rhs_{depth}, bf_poly_batch_{depth}, {count}U, BF_CELL_BITS, BF_SIGNED_CELLS);"
            ),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ProductOperand {
    Source(usize),
    Product(usize),
}

#[derive(Clone, Copy, Debug)]
struct ProductNode {
    lhs: ProductOperand,
    rhs: ProductOperand,
    depth: usize,
}

#[derive(Clone, Copy)]
enum ProductStorage {
    Scalar,
    Batch { depth: usize, index: usize },
}

#[derive(Default)]
struct ExpressionDag {
    products: Vec<ProductNode>,
    product_ids: HashMap<(ProductOperand, ProductOperand), usize>,
    monomial_values: BTreeMap<SymbolicMonomial, ProductOperand>,
}

#[derive(Clone)]
enum ResidualTerm {
    Constant(i64),
    Monomial(SymbolicMonomial, i64),
}

#[derive(Clone)]
struct Factorization {
    common: SymbolicMonomial,
    residual: Vec<ResidualTerm>,
}

#[derive(Clone, Default)]
enum PolynomialForm {
    #[default]
    Expanded,
    Factored(Factorization),
}

struct EmissionAnalysis {
    forms: BTreeMap<BfOffset, PolynomialForm>,
    dag: ExpressionDag,
    arithmetic: usize,
    temporaries: usize,
}

impl ExpressionDag {
    fn build(
        transfer: &SymbolicTransfer,
        forms: &BTreeMap<BfOffset, PolynomialForm>,
        sources: &BTreeMap<BfOffset, usize>,
    ) -> Option<Self> {
        let mut dag = Self::default();
        for (&offset, polynomial) in &transfer.effects {
            match forms.get(&offset).unwrap_or(&PolynomialForm::Expanded) {
                PolynomialForm::Expanded => {
                    for monomial in polynomial.terms.keys() {
                        dag.insert_monomial(monomial, sources)?;
                    }
                }
                PolynomialForm::Factored(factorization) => {
                    dag.insert_monomial(&factorization.common, sources)?;
                    for residual in &factorization.residual {
                        if let ResidualTerm::Monomial(monomial, _) = residual {
                            dag.insert_monomial(monomial, sources)?;
                        }
                    }
                }
            }
        }
        Some(dag)
    }

    fn insert_monomial(
        &mut self,
        monomial: &SymbolicMonomial,
        sources: &BTreeMap<BfOffset, usize>,
    ) -> Option<()> {
        if self.monomial_values.contains_key(monomial) {
            return Some(());
        }
        let mut factors = monomial.factors().peekable();
        let mut value = None;
        while let Some(offset) = factors.next() {
            let mut exponent = 1_u8;
            while factors.next_if_eq(&offset).is_some() {
                exponent = exponent
                    .checked_add(1)
                    .or_invariant("symbolic monomial degree fits u8");
            }
            let source_index = sources.get(&offset).copied()?;
            let power = self.power(ProductOperand::Source(source_index), exponent)?;
            value = Some(match value {
                Some(lhs) => self.product(lhs, power)?,
                None => power,
            });
        }
        self.monomial_values.insert(monomial.clone(), value?);
        Some(())
    }

    fn power(&mut self, source: ProductOperand, mut exponent: u8) -> Option<ProductOperand> {
        let mut result = None;
        let mut squared = source;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = Some(match result {
                    Some(lhs) => self.product(lhs, squared)?,
                    None => squared,
                });
            }
            exponent >>= 1;
            if exponent != 0 {
                squared = self.product(squared, squared)?;
            }
        }
        result
    }

    fn product(&mut self, lhs: ProductOperand, rhs: ProductOperand) -> Option<ProductOperand> {
        let key = ordered_operands(lhs, rhs);
        if let Some(&index) = self.product_ids.get(&key) {
            return Some(ProductOperand::Product(index));
        }
        if self.products.len() >= EMIT_DAG_NODE_MAX {
            return None;
        }
        let depth = 1 + self.operand_depth(key.0).max(self.operand_depth(key.1));
        let index = self.products.len();
        self.products.push(ProductNode {
            lhs: key.0,
            rhs: key.1,
            depth,
        });
        self.product_ids.insert(key, index);
        Some(ProductOperand::Product(index))
    }

    fn operand_depth(&self, operand: ProductOperand) -> usize {
        match operand {
            ProductOperand::Source(_) => 0,
            ProductOperand::Product(index) => self.products[index].depth,
        }
    }
}

pub(crate) struct EvaluationCost {
    pub(crate) total: usize,
    pub(crate) multiplications: usize,
}

fn analyze_forms(
    transfer: &SymbolicTransfer,
    forms: BTreeMap<BfOffset, PolynomialForm>,
) -> Option<EmissionAnalysis> {
    let sources = source_indices(transfer);
    let dag = ExpressionDag::build(transfer, &forms, &sources)?;
    let mut coefficient_products = 0;
    let mut additions = 0;
    let mut factored = 0;
    for (&offset, polynomial) in &transfer.effects {
        match forms.get(&offset).unwrap_or(&PolynomialForm::Expanded) {
            PolynomialForm::Expanded => {
                additions += if polynomial.constant == 0 {
                    polynomial.terms.len().saturating_sub(1)
                } else {
                    polynomial.terms.len()
                };
                coefficient_products += polynomial
                    .terms
                    .values()
                    .filter(|&&coefficient| coefficient != 1)
                    .count();
            }
            PolynomialForm::Factored(factorization) => {
                factored += 1;
                additions += factorization.residual.len().saturating_sub(1)
                    + usize::from(polynomial.constant != 0);
                coefficient_products += factorization
                    .residual
                    .iter()
                    .filter(|term| {
                        matches!(term, ResidualTerm::Monomial(_, coefficient) if *coefficient != 1)
                    })
                    .count();
            }
        }
    }
    let multiplications = dag.products.len() + coefficient_products + factored;
    let arithmetic = additions + multiplications;
    let nonzero_effects = transfer
        .effects
        .values()
        .filter(|polynomial| !polynomial.is_zero())
        .count();
    let base_product_temps = product_temporary_count(&dag);
    let factored_temps = if factored >= 4 {
        factored * 4
    } else {
        factored * 2
    };
    Some(EmissionAnalysis {
        forms,
        dag,
        arithmetic,
        temporaries: transfer.reads.len() + nonzero_effects + base_product_temps + factored_temps,
    })
}

fn select_analysis(transfer: &SymbolicTransfer) -> Option<EmissionAnalysis> {
    can_emit_transfer(transfer).then_some(())?;
    let candidates = transfer
        .effects
        .iter()
        .filter_map(|(&offset, polynomial)| {
            common_factorization(polynomial).map(|factorization| (offset, factorization))
        })
        .collect::<Vec<_>>();

    let mut forms = BTreeMap::new();
    let mut best = analyze_forms(transfer, forms.clone())?;
    // First admit independently profitable factors in output order.
    for (offset, factorization) in &candidates {
        forms.insert(*offset, PolynomialForm::Factored(factorization.clone()));
        let candidate = analyze_forms(transfer, forms.clone())?;
        if analysis_score(&candidate) < analysis_score(&best) {
            best = candidate;
        } else {
            forms.remove(offset);
        }
    }
    if candidates.len() < 2 {
        return Some(best);
    }

    // Also consider factors jointly: shared DAG nodes can make a group profitable even
    // when every member ties the expanded form in isolation. Remove neutral or harmful
    // members deterministically, preferring the original form on ties.
    forms = candidates
        .iter()
        .map(|(offset, factorization)| (*offset, PolynomialForm::Factored(factorization.clone())))
        .collect();
    let mut joint = analyze_forms(transfer, forms.clone())?;
    for (offset, _) in &candidates {
        let retained = forms.remove(offset);
        let without = analyze_forms(transfer, forms.clone())?;
        if analysis_score(&without) <= analysis_score(&joint) {
            joint = without;
        } else if let Some(form) = retained {
            forms.insert(*offset, form);
        }
    }
    if analysis_score(&joint) < analysis_score(&best) {
        best = joint;
    }
    Some(best)
}

fn analysis_score(analysis: &EmissionAnalysis) -> (usize, usize) {
    (analysis.arithmetic, analysis.temporaries)
}

pub(crate) fn evaluation_cost_breakdown(transfer: &SymbolicTransfer) -> EvaluationCost {
    let Some(analysis) = select_analysis(transfer) else {
        return EvaluationCost {
            total: usize::MAX,
            multiplications: usize::MAX,
        };
    };
    EvaluationCost {
        total: transfer.reads.len() + transfer.effects.len() + analysis.arithmetic,
        multiplications: analysis.dag.products.len()
            + transfer
                .effects
                .iter()
                .map(|(&offset, polynomial)| match analysis.forms.get(&offset) {
                    Some(PolynomialForm::Factored(factorization)) => {
                        1 + factorization
                            .residual
                            .iter()
                            .filter(|term| {
                                matches!(term, ResidualTerm::Monomial(_, coefficient) if *coefficient != 1)
                            })
                            .count()
                    }
                    _ => polynomial
                        .terms
                        .values()
                        .filter(|&&coefficient| coefficient != 1)
                        .count(),
                })
                .sum::<usize>(),
    }
}

fn product_temporary_count(dag: &ExpressionDag) -> usize {
    let mut by_depth = BTreeMap::<usize, usize>::new();
    for product in &dag.products {
        *by_depth.entry(product.depth).or_default() += 1;
    }
    by_depth
        .values()
        .map(|&count| if count >= 4 { count * 3 } else { count })
        .sum()
}

pub(crate) fn evaluation_cost(transfer: &SymbolicTransfer) -> usize {
    evaluation_cost_breakdown(transfer).total
}

/// Cheap structural gate for callers that must reject before expression-DAG expansion.
pub(crate) fn can_emit_transfer(transfer: &SymbolicTransfer) -> bool {
    if !transfer.is_pure_windowed()
        || transfer.ptr_delta != 0
        || transfer.reads.len() > EMIT_SNAPSHOT_CELL_MAX
    {
        return false;
    }
    let mut dag_nodes = 0_usize;
    let mut terms = 0_usize;
    let mut nonzero_effects = 0_usize;
    for polynomial in transfer.effects.values() {
        if polynomial.terms.len() > super::symbolic::SYMBOLIC_TERM_MAX {
            return false;
        }
        nonzero_effects += usize::from(!polynomial.is_zero());
        terms = match terms.checked_add(polynomial.terms.len()) {
            Some(value) => value,
            None => return false,
        };
        for monomial in polynomial.terms.keys() {
            if matches!(monomial, SymbolicMonomial::Powers(powers) if powers.iter().any(|&(_, exponent)| exponent == 0))
            {
                return false;
            }
            let mut degree = 0_usize;
            for offset in monomial.factors() {
                degree += 1;
                if degree > usize::from(SYMBOLIC_DEGREE_MAX) || !transfer.reads.contains(&offset) {
                    return false;
                }
            }
            if degree == 0 {
                return false;
            }
            dag_nodes = match dag_nodes.checked_add(degree.saturating_sub(1)) {
                Some(value) if value <= EMIT_DAG_NODE_MAX => value,
                _ => return false,
            };
        }
    }
    let temporaries = transfer
        .reads
        .len()
        .checked_add(nonzero_effects)
        .and_then(|count| count.checked_add(dag_nodes.checked_mul(3)?))
        .and_then(|count| count.checked_add(nonzero_effects.checked_mul(4)?));
    let Some(temporaries) = temporaries else {
        return false;
    };
    let estimated_source = transfer
        .reads
        .len()
        .checked_mul(96)
        .and_then(|bytes| bytes.checked_add(transfer.effects.len().checked_mul(192)?))
        .and_then(|bytes| bytes.checked_add(terms.checked_mul(192)?))
        .and_then(|bytes| bytes.checked_add(dag_nodes.checked_mul(256)?));
    temporaries <= EMIT_C_TEMPORARY_MAX
        && temporaries
            .checked_mul(C_CELL_BYTES)
            .is_some_and(|bytes| bytes <= EMIT_STACK_BYTE_MAX)
        && estimated_source.is_some_and(|bytes| bytes <= EMIT_SOURCE_BYTE_MAX)
}

/// Emits a transfer as a block of C statements without nesting polynomial expressions.
///
/// Source cells are captured before the first target write. Multiplication nodes are
/// interned across every output polynomial, so repeated squares and product prefixes
/// are evaluated once per application.
pub(crate) fn emit_symbolic_transfer<B: PolynomialCBackend>(
    transfer: &SymbolicTransfer,
    backend: &B,
    budget: &mut PolynomialEmissionBudget,
    indent_bytes: usize,
) -> Option<Vec<String>> {
    can_emit_transfer(transfer).then_some(())?;
    let analysis = select_analysis(transfer)?;
    if analysis.dag.products.len() > EMIT_DAG_NODE_MAX
        || analysis.temporaries > EMIT_C_TEMPORARY_MAX
        || analysis
            .temporaries
            .checked_mul(C_CELL_BYTES)
            .is_none_or(|bytes| bytes > EMIT_STACK_BYTE_MAX)
    {
        return None;
    }
    let mut lines = vec!["{".to_string()];
    let source_indices = source_indices(transfer);
    for (&offset, &index) in &source_indices {
        lines.push(format!(
            "int64_t bf_poly_src_{index} = {};",
            backend.source(offset)
        ));
    }

    let dag = &analysis.dag;
    let mut products_by_depth = BTreeMap::<usize, Vec<usize>>::new();
    for (index, product) in dag.products.iter().enumerate() {
        products_by_depth
            .entry(product.depth)
            .or_default()
            .push(index);
    }
    let mut storage = vec![ProductStorage::Scalar; dag.products.len()];
    for (&depth, products) in &products_by_depth {
        if products.len() >= 4 {
            for (batch_index, &product_index) in products.iter().enumerate() {
                storage[product_index] = ProductStorage::Batch {
                    depth,
                    index: batch_index,
                };
            }
        }
    }
    for (&depth, products) in &products_by_depth {
        if products.len() >= 4 {
            let lhs = products
                .iter()
                .map(|&index| operand_name(dag.products[index].lhs, &storage))
                .collect::<Vec<_>>();
            let rhs = products
                .iter()
                .map(|&index| operand_name(dag.products[index].rhs, &storage))
                .collect::<Vec<_>>();
            lines.extend(backend.mul_batch(depth, &lhs, &rhs));
        } else {
            for &index in products {
                let product = dag.products[index];
                lines.push(format!(
                    "int64_t bf_poly_product_{index} = {};",
                    backend.wrap_mul(
                        &operand_name(product.lhs, &storage),
                        &operand_name(product.rhs, &storage),
                    )
                ));
            }
        }
    }

    let factored_offsets = analysis
        .forms
        .iter()
        .filter_map(|(&offset, form)| matches!(form, PolynomialForm::Factored(_)).then_some(offset))
        .collect::<Vec<_>>();
    for (factor_index, &offset) in factored_offsets.iter().enumerate() {
        let PolynomialForm::Factored(factorization) = &analysis.forms[&offset] else {
            crate::invariant_failure!("selected factored offset has factored form")
        };
        let value_name = format!("bf_poly_factor_value_{factor_index}");
        let mut residual = factorization.residual.iter();
        let first = residual
            .next()
            .or_invariant("factored polynomial has residual terms");
        lines.push(format!(
            "int64_t {value_name} = {};",
            residual_term_expr(first, dag, &storage, backend)
        ));
        for residual in residual {
            let term = residual_term_expr(residual, dag, &storage, backend);
            lines.push(format!(
                "{value_name} = {};",
                backend.wrap_add(&value_name, &term)
            ));
        }
    }
    let factored_batch_depth = dag
        .products
        .iter()
        .map(|product| product.depth)
        .max()
        .unwrap_or(0)
        + 1;
    if factored_offsets.len() >= 4 {
        let lhs = factored_offsets
            .iter()
            .map(|offset| {
                let PolynomialForm::Factored(factorization) = &analysis.forms[offset] else {
                    crate::invariant_failure!("selected factored offset has factored form")
                };
                analyzed_monomial_name(dag, &factorization.common, &storage)
            })
            .collect::<Vec<_>>();
        let rhs = (0..factored_offsets.len())
            .map(|index| format!("bf_poly_factor_value_{index}"))
            .collect::<Vec<_>>();
        lines.extend(backend.mul_batch(factored_batch_depth, &lhs, &rhs));
    } else {
        for (factor_index, &offset) in factored_offsets.iter().enumerate() {
            let PolynomialForm::Factored(factorization) = &analysis.forms[&offset] else {
                crate::invariant_failure!("selected factored offset has factored form")
            };
            let common = analyzed_monomial_name(dag, &factorization.common, &storage);
            lines.push(format!(
                "int64_t bf_poly_factored_{factor_index} = {};",
                backend.wrap_mul(&common, &format!("bf_poly_factor_value_{factor_index}"))
            ));
        }
    }

    let mut effects = transfer.effects.iter().peekable();
    let mut value_index = 0;
    while let Some((&offset, polynomial)) = effects.next() {
        if polynomial.is_zero() {
            let start = offset;
            let mut end = offset;
            while let Some((&next_offset, next_polynomial)) = effects.peek().copied() {
                if next_offset != end + 1 || !next_polynomial.is_zero() {
                    break;
                }
                effects.next();
                end = next_offset;
            }
            lines.push(backend.zero_region(start, end - start + 1));
            continue;
        }

        let value_name = format!("bf_poly_value_{value_index}");
        match analysis.forms.get(&offset) {
            Some(PolynomialForm::Factored(_)) => {
                let factor_index = factored_offsets
                    .binary_search(&offset)
                    .or_invariant("factored offsets contain selected effect");
                let term = if factored_offsets.len() >= 4 {
                    format!("bf_poly_batch_{factored_batch_depth}[{factor_index}]")
                } else {
                    format!("bf_poly_factored_{factor_index}")
                };
                if polynomial.constant == 0 {
                    lines.push(format!("int64_t {value_name} = {term};"));
                } else {
                    lines.push(format!(
                        "int64_t {value_name} = {};",
                        backend.wrap_add("INT64_C(0)", &c_i64_literal(polynomial.constant))
                    ));
                    lines.push(format!(
                        "{value_name} = {};",
                        backend.wrap_add(&value_name, &term)
                    ));
                }
            }
            _ => {
                let mut terms = polynomial.terms.iter();
                if polynomial.constant == 0 {
                    let (monomial, &coefficient) = terms
                        .next()
                        .or_invariant("nonzero polynomial has a term or constant");
                    lines.push(format!(
                        "int64_t {value_name} = {};",
                        scaled_monomial_expr(monomial, coefficient, dag, &storage, backend)
                    ));
                } else {
                    lines.push(format!(
                        "int64_t {value_name} = {};",
                        backend.wrap_add("INT64_C(0)", &c_i64_literal(polynomial.constant))
                    ));
                }
                for (monomial, &coefficient) in terms {
                    let term = scaled_monomial_expr(monomial, coefficient, dag, &storage, backend);
                    lines.push(format!(
                        "{value_name} = {};",
                        backend.wrap_add(&value_name, &term)
                    ));
                }
            }
        }
        lines.push(format!("{} = {value_name};", backend.target(offset)));
        value_index += 1;
    }
    lines.push("}".to_string());
    let source_bytes = lines.iter().try_fold(0_usize, |bytes, line| {
        bytes
            .checked_add(indent_bytes)?
            .checked_add(line.len())?
            .checked_add(1)
    })?;
    (source_bytes <= EMIT_SOURCE_BYTE_MAX).then_some(())?;
    budget.reserve(analysis.dag.products.len(), source_bytes)?;
    Some(lines)
}

fn analyzed_monomial_name(
    dag: &ExpressionDag,
    monomial: &SymbolicMonomial,
    storage: &[ProductStorage],
) -> String {
    operand_name(
        dag.monomial_values
            .get(monomial)
            .copied()
            .or_invariant("every emitted monomial has an analyzed DAG value"),
        storage,
    )
}

fn scaled_monomial_expr<B: PolynomialCBackend>(
    monomial: &SymbolicMonomial,
    coefficient: i64,
    dag: &ExpressionDag,
    storage: &[ProductStorage],
    backend: &B,
) -> String {
    let base = analyzed_monomial_name(dag, monomial, storage);
    if coefficient == 1 {
        base
    } else {
        backend.wrap_mul(&base, &c_i64_literal(coefficient))
    }
}

fn residual_term_expr<B: PolynomialCBackend>(
    residual: &ResidualTerm,
    dag: &ExpressionDag,
    storage: &[ProductStorage],
    backend: &B,
) -> String {
    match residual {
        ResidualTerm::Constant(coefficient) => {
            backend.wrap_add("INT64_C(0)", &c_i64_literal(*coefficient))
        }
        ResidualTerm::Monomial(monomial, coefficient) => {
            scaled_monomial_expr(monomial, *coefficient, dag, storage, backend)
        }
    }
}

fn common_factorization(polynomial: &SymbolicPolynomial) -> Option<Factorization> {
    if polynomial.terms.len() < 2 {
        return None;
    }
    const FACTOR_MAX: usize = SYMBOLIC_DEGREE_MAX as usize;
    let first = polynomial.terms.keys().next()?;
    let mut common = [0; FACTOR_MAX];
    let mut common_len = 0;
    for factor in first.factors() {
        common[common_len] = factor;
        common_len += 1;
    }
    for monomial in polynomial.terms.keys().skip(1) {
        let mut intersection = [0; FACTOR_MAX];
        let mut intersection_len = 0;
        let mut common_index = 0;
        for factor in monomial.factors() {
            while common_index < common_len && common[common_index] < factor {
                common_index += 1;
            }
            if common_index < common_len && common[common_index] == factor {
                intersection[intersection_len] = factor;
                intersection_len += 1;
                common_index += 1;
            }
        }
        common = intersection;
        common_len = intersection_len;
        if common_len == 0 {
            return None;
        }
    }

    let common_monomial = monomial_from_sorted_factors(&common[..common_len])?;
    let mut residual = Vec::with_capacity(polynomial.terms.len());
    for (monomial, &coefficient) in polynomial.terms.iter() {
        let mut factors = [0; FACTOR_MAX];
        let mut factor_len = 0;
        let mut common_index = 0;
        for factor in monomial.factors() {
            if common_index < common_len && factor == common[common_index] {
                common_index += 1;
            } else {
                factors[factor_len] = factor;
                factor_len += 1;
            }
        }
        residual.push(match monomial_from_sorted_factors(&factors[..factor_len]) {
            Some(term) => ResidualTerm::Monomial(term, coefficient),
            None => ResidualTerm::Constant(coefficient),
        });
    }
    Some(Factorization {
        common: common_monomial,
        residual,
    })
}

fn monomial_from_sorted_factors(factors: &[BfOffset]) -> Option<SymbolicMonomial> {
    match factors {
        [] => None,
        [offset] => Some(SymbolicMonomial::Linear(*offset)),
        [lhs, rhs] => Some(SymbolicMonomial::Product(*lhs, *rhs)),
        _ => {
            let mut powers = Vec::<(BfOffset, u8)>::new();
            for &offset in factors {
                match powers.last_mut() {
                    Some((previous, exponent)) if *previous == offset => {
                        *exponent = exponent
                            .checked_add(1)
                            .or_invariant("symbolic monomial exponent fits u8");
                    }
                    _ => powers.push((offset, 1)),
                }
            }
            Some(SymbolicMonomial::Powers(powers.into_boxed_slice()))
        }
    }
}

fn source_indices(transfer: &SymbolicTransfer) -> BTreeMap<BfOffset, usize> {
    transfer
        .reads
        .iter()
        .copied()
        .enumerate()
        .map(|(index, offset)| (offset, index))
        .collect()
}

fn ordered_operands(lhs: ProductOperand, rhs: ProductOperand) -> (ProductOperand, ProductOperand) {
    if operand_sort_key(lhs) <= operand_sort_key(rhs) {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

fn operand_sort_key(operand: ProductOperand) -> (u8, usize) {
    match operand {
        ProductOperand::Source(index) => (0, index),
        ProductOperand::Product(index) => (1, index),
    }
}

fn operand_name(operand: ProductOperand, storage: &[ProductStorage]) -> String {
    match operand {
        ProductOperand::Source(index) => format!("bf_poly_src_{index}"),
        ProductOperand::Product(index) => match storage[index] {
            ProductStorage::Scalar => format!("bf_poly_product_{index}"),
            ProductStorage::Batch { depth, index } => {
                format!("bf_poly_batch_{depth}[{index}]")
            }
        },
    }
}

fn c_i64_literal(value: i64) -> String {
    if value == i64::MIN {
        "(-INT64_C(9223372036854775807) - INT64_C(1))".to_string()
    } else {
        format!("INT64_C({value})")
    }
}

#[cfg(test)]
mod tests;
