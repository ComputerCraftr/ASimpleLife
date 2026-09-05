use super::*;
use crate::bf::polynomial_emit::{PolynomialCBackend, emit_symbolic_transfer};

struct SuperPolynomialBackend;

impl PolynomialCBackend for SuperPolynomialBackend {
    fn source(&self, offset: crate::bf::BfOffset) -> String {
        cell_expr(offset)
    }

    fn target(&self, offset: crate::bf::BfOffset) -> String {
        cell_expr(offset)
    }

    fn wrap_add(&self, lhs: &str, rhs: &str) -> String {
        format!("bf_wrap_add({lhs}, {rhs}, BF_CELL_BITS, BF_SIGNED_CELLS)")
    }

    fn wrap_mul(&self, lhs: &str, rhs: &str) -> String {
        format!("bf_wrap_mul({lhs}, {rhs}, BF_CELL_BITS, BF_SIGNED_CELLS)")
    }

    fn zero_region(&self, start: crate::bf::BfOffset, len: crate::bf::BfOffset) -> String {
        format!("bf_zero_region(tape, ptr, {start}, {len}U);")
    }
}

pub(super) fn emit_symbolic_transfer_apply(out: &mut String, lines: &[String], level: usize) {
    push_c_line(out, level, "bf_work_op();");
    for line in lines {
        push_c_line(out, level, line);
    }
}

pub(super) fn render_symbolic_transfer(
    transfer: &SymbolicTransfer,
    budget: &mut PolynomialEmissionBudget,
    level: usize,
) -> Option<Vec<String>> {
    emit_symbolic_transfer(transfer, &SuperPolynomialBackend, budget, level * 4)
}

fn emit_powered_loop_fallback(out: &mut String, body: NodeId, level: usize) {
    push_c_line(out, level, "recursion_fallbacks++;");
    push_c_line(out, level, "while (tape[ptr] != 0) {");
    push_c_line(out, level + 1, "bf_work_loop_iter();");
    push_c_line(
        out,
        level + 1,
        &format!("exec_node_{}(tape, &ptr);", body.0),
    );
    push_c_line(out, level, "}");
}

pub(super) fn emit_powered_loop_apply(
    out: &mut String,
    analysis: &PoweredLoopAnalysis,
    max_power: u8,
    level: usize,
    emission_budget: &mut PolynomialEmissionBudget,
) {
    const POWERED_SCAFFOLD_BYTES: usize = 4 * 1024;
    const POWER_BRANCH_BYTES: usize = 256;
    let Some(scaffold_bytes) = usize::from(max_power)
        .checked_add(1)
        .and_then(|count| count.checked_mul(POWER_BRANCH_BYTES))
        .and_then(|bytes| bytes.checked_add(POWERED_SCAFFOLD_BYTES))
    else {
        emit_powered_loop_fallback(out, analysis.body, level);
        return;
    };
    let mut trial_budget = emission_budget.clone();
    if max_power >= 64 || trial_budget.reserve_source_bytes(scaffold_bytes).is_none() {
        emit_powered_loop_fallback(out, analysis.body, level);
        return;
    }
    let power_emissions = if analysis.only_drains_guard() {
        Vec::new()
    } else {
        let Some(powers) = analysis.powers.get(..=usize::from(max_power)) else {
            emit_powered_loop_fallback(out, analysis.body, level);
            return;
        };
        let Some(emissions) = powers
            .iter()
            .map(|transfer| render_symbolic_transfer(transfer, &mut trial_budget, level + 2))
            .collect::<Option<Vec<_>>>()
        else {
            emit_powered_loop_fallback(out, analysis.body, level);
            return;
        };
        emissions
    };
    *emission_budget = trial_budget;
    push_c_line(
        out,
        level,
        &format!(
            "int64_t guard = tape[{}];",
            wrap_ptr_expr(analysis.guard_offset)
        ),
    );
    push_c_line(out, level, "uint64_t remaining_iters = 0;");
    push_c_line(out, level, "int powered_ok = 0;");
    push_c_line(out, level, "if (guard == 0) {");
    push_c_line(out, level + 1, "powered_ok = 1;");
    push_c_line(out, level, "} else if (BF_SIGNED_CELLS) {");
    if analysis.guard_delta < 0 {
        push_c_line(
            out,
            level + 1,
            &format!(
                "if (guard > 0 && ((uint64_t)guard % UINT64_C({})) == 0) {{",
                -analysis.guard_delta
            ),
        );
        push_c_line(
            out,
            level + 2,
            &format!(
                "remaining_iters = (uint64_t)guard / UINT64_C({});",
                -analysis.guard_delta
            ),
        );
    } else {
        push_c_line(
            out,
            level + 1,
            "uint64_t guard_magnitude = (uint64_t)(-(guard + INT64_C(1))) + UINT64_C(1);",
        );
        push_c_line(
            out,
            level + 1,
            &format!(
                "if (guard < 0 && (guard_magnitude % UINT64_C({})) == 0) {{",
                analysis.guard_delta
            ),
        );
        push_c_line(
            out,
            level + 2,
            &format!(
                "remaining_iters = guard_magnitude / UINT64_C({});",
                analysis.guard_delta
            ),
        );
    }
    push_c_line(out, level + 2, "powered_ok = 1;");
    push_c_line(out, level + 1, "}");
    push_c_line(out, level, "} else {");
    if analysis.guard_delta < 0 {
        push_c_line(
            out,
            level + 1,
            &format!(
                "if (guard > 0 && ((uint64_t)guard % UINT64_C({})) == 0) {{",
                -analysis.guard_delta
            ),
        );
        push_c_line(
            out,
            level + 2,
            &format!(
                "remaining_iters = (uint64_t)guard / UINT64_C({});",
                -analysis.guard_delta
            ),
        );
        push_c_line(out, level + 2, "powered_ok = 1;");
        push_c_line(out, level + 1, "}");
    }
    push_c_line(out, level, "}");
    push_c_line(out, level, "if (powered_ok) {");
    push_c_line(out, level + 1, "symbolic_power_builds++;");
    if analysis.only_drains_guard() {
        push_c_line(out, level + 1, "if (remaining_iters != 0) {");
        push_c_line(out, level + 2, "symbolic_power_hits++;");
        push_c_line(out, level + 2, "bf_work_op();");
        push_c_line(
            out,
            level + 2,
            &format!("tape[{}] = 0;", wrap_ptr_expr(analysis.guard_offset)),
        );
        push_c_line(out, level + 1, "}");
    } else {
        for power in (0..=max_power).rev() {
            let iterations = 1_u64 << power;
            let condition = if power == max_power { "while" } else { "if" };
            push_c_line(
                out,
                level + 1,
                &format!("{condition} (remaining_iters >= UINT64_C({iterations})) {{"),
            );
            push_c_line(out, level + 2, "symbolic_power_hits++;");
            emit_symbolic_transfer_apply(out, &power_emissions[usize::from(power)], level + 2);
            push_c_line(
                out,
                level + 2,
                &format!("remaining_iters -= UINT64_C({iterations});"),
            );
            push_c_line(out, level + 1, "}");
        }
    }
    push_c_line(out, level, "} else {");
    emit_powered_loop_fallback(out, analysis.body, level + 1);
    push_c_line(out, level, "}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::symbolic::SymbolicPolynomial;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    #[test]
    fn exhausted_emission_budget_uses_complete_powered_loop_fallback() {
        let power = SymbolicTransfer {
            ptr_delta: 0,
            effects: BTreeMap::from([(1, SymbolicPolynomial::input(0))]),
            reads: BTreeSet::from([0]),
            may_input: false,
            may_output: false,
            may_diverge: false,
            unknown: false,
        };
        let analysis = PoweredLoopAnalysis {
            body: NodeId(7),
            guard_offset: 0,
            guard_delta: -1,
            powers: Arc::from([power]),
        };
        let mut budget = PolynomialEmissionBudget::default();
        budget
            .reserve_source_bytes(4 * 1024 * 1024)
            .or_invariant("reserve the production symbolic source allowance");
        let mut out = String::new();

        emit_powered_loop_apply(&mut out, &analysis, 0, 0, &mut budget);

        assert!(out.contains("recursion_fallbacks++;"));
        assert!(out.contains("exec_node_7(tape, &ptr);"));
        assert!(!out.contains("int64_t guard ="));
        assert!(!out.contains("bf_poly_"));
    }
}
