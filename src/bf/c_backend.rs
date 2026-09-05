use super::BF_C_TAPE_LEN as C_TAPE_LEN;
use super::c_super_backend::{CompileWork, summarize_c_region_with_work};
use super::c_support::{
    expand_runtime_fragments, mask_literal, normalized_c_offset, push_c_line, signed_cells_flag,
    split_char_input_stmt, split_number_input_stmt, split_output_stmt, wrap_ptr_expr,
};
use super::ir::{BfIr, ShiftDir, validate_canonical_ir};
#[cfg(test)]
use super::optimizer::CellSign;
use super::optimizer::{CodegenOpts, IoMode};
use super::polynomial_emit::{
    PolynomialCBackend, PolynomialEmissionBudget, can_emit_transfer, emit_symbolic_transfer,
    evaluation_cost_breakdown,
};
use crate::RequiredExt;

mod batches;
use batches::{emit_add_batch, emit_clear_batch};

#[cfg(test)]
const BF_TEST_STEP_BUDGET: u64 = 10_000_000;

struct PlainPolynomialBackend;

impl PolynomialCBackend for PlainPolynomialBackend {
    fn source(&self, offset: crate::bf::BfOffset) -> String {
        format!("tape[{}]", wrap_ptr_expr(offset))
    }

    fn target(&self, offset: crate::bf::BfOffset) -> String {
        format!("tape[{}]", wrap_ptr_expr(offset))
    }

    fn wrap_add(&self, lhs: &str, rhs: &str) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_add_i64_signed({lhs}, {rhs}, BF_CELL_BITS) : bf_wrap_add_i64_unsigned({lhs}, {rhs}, BF_CELL_BITS)"
        )
    }

    fn wrap_mul(&self, lhs: &str, rhs: &str) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_mul_i64_signed({lhs}, {rhs}, BF_CELL_BITS) : bf_wrap_mul_i64_unsigned({lhs}, {rhs}, BF_CELL_BITS)"
        )
    }

    fn zero_region(&self, start: crate::bf::BfOffset, len: crate::bf::BfOffset) -> String {
        let len = usize::try_from(len).or_invariant("positive symbolic zero region fits usize");
        let assignments = (0..len)
            .map(|index| {
                let offset = start
                    .checked_add(
                        crate::bf::BfOffset::try_from(index)
                            .or_invariant("symbolic zero region index fits offset"),
                    )
                    .or_invariant("symbolic zero region offset remains in range");
                format!("tape[{}] = 0;", wrap_ptr_expr(offset))
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("{{ {assignments} }}")
    }
}

fn is_polynomial_rich_op(node: &BfIr) -> bool {
    matches!(
        node,
        BfIr::Affine { .. } | BfIr::Square { .. } | BfIr::MulAdd { .. }
    )
}

pub fn emit_c(program: &[BfIr], opts: CodegenOpts) -> String {
    if let Err(error) = opts.validate() {
        return format!("#error {error}\n");
    }
    validate_canonical_ir(program).or_invariant("plain C backend requires canonical richer IR");

    fn add_expr(d: u64) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[ptr], INT64_C({d}), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[ptr], INT64_C({d}), BF_CELL_BITS)"
        )
    }
    fn sub_expr(d: u64) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_sub_i64_signed(tape[ptr], INT64_C({d}), BF_CELL_BITS) : bf_wrap_sub_i64_unsigned(tape[ptr], INT64_C({d}), BF_CELL_BITS)"
        )
    }
    fn cell_expr(offset: crate::bf::BfOffset) -> String {
        format!("tape[{}]", wrap_ptr_expr(offset))
    }
    fn shift_left_expr(src_expr: &str, amount: u32) -> String {
        format!("bf_wrap_shift_left_i64({src_expr}, {amount}, BF_CELL_BITS, BF_SIGNED_CELLS)")
    }
    fn shift_right_expr(src_expr: &str, amount: u32) -> String {
        format!("bf_wrap_shift_right_i64({src_expr}, {amount}, BF_CELL_BITS, BF_SIGNED_CELLS)")
    }
    fn mul_expr(lhs_expr: &str, rhs_expr: &str) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_mul_i64_signed({lhs_expr}, {rhs_expr}, BF_CELL_BITS) : bf_wrap_mul_i64_unsigned({lhs_expr}, {rhs_expr}, BF_CELL_BITS)"
        )
    }
    fn add_dynamic_expr(base_expr: &str, delta_expr: &str) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_add_i64_signed({base_expr}, {delta_expr}, BF_CELL_BITS) : bf_wrap_add_i64_unsigned({base_expr}, {delta_expr}, BF_CELL_BITS)"
        )
    }

    enum EmitFrame<'a> {
        Seq {
            nodes: &'a [BfIr],
            index: usize,
            level: usize,
            allow_symbolic: bool,
        },
        Close {
            level: usize,
        },
    }

    fn emit_body(out: &mut String, program: &[BfIr], opts: CodegenOpts) {
        let mut compile_work = CompileWork::default();
        let mut emission_budget = PolynomialEmissionBudget::default();
        let mut stack = vec![EmitFrame::Seq {
            nodes: program,
            index: 0,
            level: 1,
            allow_symbolic: true,
        }];
        while let Some(frame) = stack.pop() {
            match frame {
                EmitFrame::Seq {
                    nodes,
                    mut index,
                    level,
                    allow_symbolic,
                } => {
                    if index >= nodes.len() {
                        continue;
                    }
                    if let Some(next_index) = emit_add_batch(out, nodes, index, level) {
                        stack.push(EmitFrame::Seq {
                            nodes,
                            index: next_index,
                            level,
                            allow_symbolic,
                        });
                        continue;
                    }
                    if let Some(next_index) = emit_clear_batch(out, nodes, index, level) {
                        stack.push(EmitFrame::Seq {
                            nodes,
                            index: next_index,
                            level,
                            allow_symbolic,
                        });
                        continue;
                    }
                    if allow_symbolic && is_polynomial_rich_op(&nodes[index]) {
                        const POLYNOMIAL_REGION_MAX: usize = 16;
                        let limit = nodes.len().min(index + POLYNOMIAL_REGION_MAX);
                        let mut end = index + 1;
                        while end < limit && is_polynomial_rich_op(&nodes[end]) {
                            end += 1;
                        }
                        if end - index > 1
                            && let Some(transfer) = summarize_c_region_with_work(
                                &nodes[index..end],
                                opts,
                                &mut compile_work,
                            )
                            && can_emit_transfer(&transfer)
                            && compile_work.admit_evaluation(&transfer)
                        {
                            let cost = evaluation_cost_breakdown(&transfer);
                            if cost.multiplications > 0
                                && cost.total <= (end - index) * 6
                                && let Some(lines) = emit_symbolic_transfer(
                                    &transfer,
                                    &PlainPolynomialBackend,
                                    &mut emission_budget,
                                    (level + 1) * 4,
                                )
                            {
                                stack.push(EmitFrame::Seq {
                                    nodes,
                                    index: end,
                                    level,
                                    allow_symbolic,
                                });
                                stack.push(EmitFrame::Close { level });
                                stack.push(EmitFrame::Seq {
                                    nodes: &nodes[index..end],
                                    index: 0,
                                    level: level + 1,
                                    allow_symbolic: false,
                                });
                                push_c_line(out, level, "if (!bf_semantic_fuel_enabled()) {");
                                push_c_line(out, level + 1, "bf_work_dispatch();");
                                push_c_line(out, level + 1, "bf_work_op();");
                                for line in lines {
                                    push_c_line(out, level + 1, &line);
                                }
                                push_c_line(out, level, "} else {");
                                continue;
                            }
                        }
                    }
                    let node = &nodes[index];
                    index += 1;
                    stack.push(EmitFrame::Seq {
                        nodes,
                        index,
                        level,
                        allow_symbolic,
                    });
                    match node {
                        BfIr::MovePtr(n) if *n != 0 => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(
                                out,
                                level,
                                &format!("bf_semantic_step(UINT64_C({}));", n.unsigned_abs()),
                            );
                            push_c_line(out, level, &format!("ptr = {};", wrap_ptr_expr(*n)))
                        }
                        BfIr::MovePtr(_) => {}
                        BfIr::Add(n) if *n > 0 => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(
                                out,
                                level,
                                &format!("bf_semantic_step(UINT64_C({}));", n.unsigned_abs()),
                            );
                            push_c_line(
                                out,
                                level,
                                &format!(
                                    "tape[ptr] = {};",
                                    add_expr(
                                        u64::try_from(*n).or_invariant("positive add fits u64")
                                    )
                                ),
                            )
                        }
                        BfIr::Add(n) if *n < 0 => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(
                                out,
                                level,
                                &format!("bf_semantic_step(UINT64_C({}));", n.unsigned_abs()),
                            );
                            push_c_line(
                                out,
                                level,
                                &format!("tape[ptr] = {};", sub_expr(u64::from(n.unsigned_abs()))),
                            )
                        }
                        BfIr::Add(_) => {}
                        BfIr::Input => match opts.io_mode {
                            IoMode::Char => {
                                push_c_line(out, level, "bf_work_dispatch();");
                                push_c_line(out, level, "bf_work_op();");
                                push_c_line(out, level, "bf_semantic_step(UINT64_C(1));");
                                push_c_line(out, level, split_char_input_stmt())
                            }
                            IoMode::Number => {
                                push_c_line(out, level, "bf_work_dispatch();");
                                push_c_line(out, level, "bf_work_op();");
                                push_c_line(out, level, "bf_semantic_step(UINT64_C(1));");
                                push_c_line(out, level, split_number_input_stmt(opts.cell_sign))
                            }
                        },
                        BfIr::Output => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_step(UINT64_C(1));");
                            push_c_line(out, level, split_output_stmt(opts));
                            push_c_line(out, level, "fflush(stdout);");
                        }
                        BfIr::Loop(body) => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "while (tape[ptr] != 0) {");
                            push_c_line(out, level + 1, "bf_work_loop_iter();");
                            push_c_line(out, level + 1, "bf_semantic_step(UINT64_C(1));");
                            stack.push(EmitFrame::Close { level });
                            stack.push(EmitFrame::Seq {
                                nodes: body,
                                index: 0,
                                level: level + 1,
                                allow_symbolic,
                            });
                        }
                        BfIr::Clear => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            push_c_line(out, level, "tape[ptr] = 0;");
                        }
                        BfIr::ClearAt { offset } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            push_c_line(
                                out,
                                level,
                                &format!("tape[{}] = 0;", wrap_ptr_expr(*offset)),
                            );
                        }
                        BfIr::Scan { stride } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            push_c_line(out, level, "{");
                            push_c_line(out, level + 1, "uint64_t bf_scan_steps = 0;");
                            push_c_line(
                                out,
                                level + 1,
                                &format!(
                                    "ptr = bf_scan_zero(tape, ptr, {}, BF_TAPE_LEN, &bf_scan_steps);",
                                    normalized_c_offset(*stride)
                                ),
                            );
                            push_c_line(out, level, "}");
                        }
                        BfIr::Distribute {
                            targets,
                            preserve_src,
                        } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            push_c_line(out, level, "{");
                            if targets.is_empty() {
                                if !preserve_src {
                                    push_c_line(out, level + 1, "tape[ptr] = 0;");
                                }
                            } else {
                                let offsets = targets
                                    .iter()
                                    .map(|(offset, _)| normalized_c_offset(*offset).to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                let coefficients = targets
                                    .iter()
                                    .map(|(_, coeff)| format!("INT64_C({coeff})"))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!("const ptrdiff_t offsets[] = {{{offsets}}};"),
                                );
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!("const int64_t coefficients[] = {{{coefficients}}};"),
                                );
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!(
                                        "bf_transfer_batch(tape, ptr, 0, offsets, coefficients, {}, {}, BF_TAPE_LEN, BF_CELL_BITS, BF_SIGNED_CELLS);",
                                        targets.len(),
                                        i32::from(*preserve_src)
                                    ),
                                );
                            }
                            push_c_line(out, level, "}");
                        }
                        BfIr::Affine {
                            src,
                            dst,
                            coeff,
                            preserve_src,
                            set_dst,
                        } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            let src_expr = cell_expr(*src);
                            let dst_expr = cell_expr(*dst);
                            let coeff_expr = format!("INT64_C({coeff})");
                            push_c_line(out, level, "{");
                            push_c_line(out, level + 1, &format!("int64_t src_v = {src_expr};"));
                            if !set_dst && *dst != *src {
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!("int64_t dst_v = {dst_expr};"),
                                );
                            }
                            let base = if *set_dst {
                                "0".to_string()
                            } else if *dst == *src {
                                "src_v".to_string()
                            } else {
                                "dst_v".to_string()
                            };
                            let prod = mul_expr("src_v", &coeff_expr);
                            push_c_line(
                                out,
                                level + 1,
                                &format!("int64_t dst_next = {};", add_dynamic_expr(&base, &prod)),
                            );
                            if !preserve_src && *src != *dst {
                                push_c_line(out, level + 1, &format!("{src_expr} = 0;"));
                            }
                            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
                            push_c_line(out, level, "}");
                        }
                        BfIr::Shift {
                            src,
                            dst,
                            amount,
                            dir,
                            preserve_src,
                            set_dst,
                        } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            let src_expr = cell_expr(*src);
                            let dst_expr = cell_expr(*dst);
                            push_c_line(out, level, "{");
                            push_c_line(out, level + 1, &format!("int64_t src_v = {src_expr};"));
                            if !set_dst && *dst != *src {
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!("int64_t dst_v = {dst_expr};"),
                                );
                            }
                            let shifted = match dir {
                                ShiftDir::Left => shift_left_expr("src_v", *amount),
                                ShiftDir::Right => shift_right_expr("src_v", *amount),
                            };
                            let rhs = if *set_dst {
                                shifted
                            } else {
                                add_dynamic_expr(
                                    if *dst == *src { "src_v" } else { "dst_v" },
                                    &shifted,
                                )
                            };
                            push_c_line(out, level + 1, &format!("int64_t dst_next = {rhs};"));
                            if !preserve_src && *src != *dst {
                                push_c_line(out, level + 1, &format!("{src_expr} = 0;"));
                            }
                            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
                            push_c_line(out, level, "}");
                        }
                        BfIr::Square {
                            src,
                            dst,
                            preserve_src,
                            set_dst,
                        } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            let src_expr = cell_expr(*src);
                            let dst_expr = cell_expr(*dst);
                            push_c_line(out, level, "{");
                            push_c_line(out, level + 1, &format!("int64_t src_v = {src_expr};"));
                            if !set_dst && *dst != *src {
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!("int64_t dst_v = {dst_expr};"),
                                );
                            }
                            let square = mul_expr("src_v", "src_v");
                            let rhs = if *set_dst {
                                square
                            } else {
                                add_dynamic_expr(
                                    if *dst == *src { "src_v" } else { "dst_v" },
                                    &square,
                                )
                            };
                            push_c_line(out, level + 1, &format!("int64_t dst_next = {rhs};"));
                            if !preserve_src && *src != *dst {
                                push_c_line(out, level + 1, &format!("{src_expr} = 0;"));
                            }
                            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
                            push_c_line(out, level, "}");
                        }
                        BfIr::MulAdd {
                            lhs,
                            rhs,
                            dst,
                            preserve_lhs,
                            preserve_rhs,
                            set_dst,
                        } => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_semantic_summary_unsupported();");
                            let lhs_expr = cell_expr(*lhs);
                            let rhs_expr = cell_expr(*rhs);
                            let dst_expr = cell_expr(*dst);
                            push_c_line(out, level, "{");
                            push_c_line(out, level + 1, &format!("int64_t lhs_v = {lhs_expr};"));
                            push_c_line(out, level + 1, &format!("int64_t rhs_v = {rhs_expr};"));
                            if !set_dst && *dst != *lhs && *dst != *rhs {
                                push_c_line(
                                    out,
                                    level + 1,
                                    &format!("int64_t dst_v = {dst_expr};"),
                                );
                            }
                            let prod = mul_expr("lhs_v", "rhs_v");
                            let base = if *set_dst {
                                "0".to_string()
                            } else if *dst == *lhs {
                                "lhs_v".to_string()
                            } else if *dst == *rhs {
                                "rhs_v".to_string()
                            } else {
                                "dst_v".to_string()
                            };
                            push_c_line(
                                out,
                                level + 1,
                                &format!("int64_t dst_next = {};", add_dynamic_expr(&base, &prod)),
                            );
                            if !preserve_lhs && *lhs != *dst {
                                push_c_line(out, level + 1, &format!("{lhs_expr} = 0;"));
                            }
                            if !preserve_rhs && *rhs != *dst && (*rhs != *lhs || *preserve_lhs) {
                                push_c_line(out, level + 1, &format!("{rhs_expr} = 0;"));
                            }
                            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
                            push_c_line(out, level, "}");
                        }
                        BfIr::Diverge => {
                            push_c_line(out, level, "bf_work_dispatch();");
                            push_c_line(out, level, "bf_work_op();");
                            push_c_line(out, level, "bf_diverge_forever();");
                        }
                    }
                }
                EmitFrame::Close { level } => push_c_line(out, level, "}"),
            }
        }
    }

    fn any_ir(nodes: &[BfIr], pred: &impl Fn(&BfIr) -> bool) -> bool {
        let mut stack: Vec<&[BfIr]> = vec![nodes];
        while let Some(seq) = stack.pop() {
            for node in seq {
                if pred(node) {
                    return true;
                }
                if let BfIr::Loop(body) = node {
                    stack.push(body);
                }
            }
        }
        false
    }

    fn strip_block(s: &mut String, begin: &str, end: &str) {
        if let (Some(a), Some(b)) = (s.find(begin), s.find(end)) {
            let end_pos = b + end.len();
            let end_pos = if s.as_bytes().get(end_pos) == Some(&b'\n') {
                end_pos + 1
            } else {
                end_pos
            };
            s.replace_range(a..end_pos, "");
        }
    }

    fn keep_block(s: &mut String, begin: &str, end: &str) {
        s.replace_range(
            s.find(begin).or_invariant("required value")
                ..s.find(begin).or_invariant("required value") + begin.len() + 1,
            "",
        );
        let end_pos = s.find(end).or_invariant("required value");
        let end_pos_after = end_pos + end.len();
        let end_pos_after = if s.as_bytes().get(end_pos_after) == Some(&b'\n') {
            end_pos_after + 1
        } else {
            end_pos_after
        };
        s.replace_range(end_pos..end_pos_after, "");
    }

    // The runtime-selected scan kernels share the checked pointer wrapper even
    // when a particular program has no explicit pointer movement.
    let needs_ptr_wrap = true;
    let needs_input = any_ir(program, &|n| matches!(n, BfIr::Input));
    let needs_cell_mask = matches!(opts.io_mode, IoMode::Number)
        && any_ir(program, &|n| matches!(n, BfIr::Input | BfIr::Output));

    let input_bits = opts.input_bits.unwrap_or(opts.cell_bits).min(63);
    let output_bits = opts.output_bits.unwrap_or(opts.cell_bits).min(63);
    let config = format!(
        "#define BF_TEMPLATE_TAPE_LEN {}\n#define BF_TEMPLATE_CELL_BITS {}\n#define BF_TEMPLATE_SIGNED_CELLS {}\n#define BF_TEMPLATE_CELL_MASK {}\n#define BF_TEMPLATE_INPUT_MASK {}\n#define BF_TEMPLATE_OUTPUT_MASK {}\n",
        C_TAPE_LEN,
        opts.cell_bits,
        signed_cells_flag(opts.cell_sign),
        mask_literal(opts.cell_bits),
        mask_literal(input_bits),
        mask_literal(output_bits),
    );

    let mut ir_body = String::new();
    emit_body(&mut ir_body, program, opts);

    let mut out = expand_runtime_fragments(super::c_support::PLAIN_RUNTIME_TEMPLATE);

    if needs_ptr_wrap {
        keep_block(
            &mut out,
            "/* @BF_WRAP_PTR_BEGIN */",
            "/* @BF_WRAP_PTR_END */",
        );
    } else {
        strip_block(
            &mut out,
            "/* @BF_WRAP_PTR_BEGIN */",
            "/* @BF_WRAP_PTR_END */",
        );
    }
    if needs_cell_mask {
        keep_block(
            &mut out,
            "/* @BF_CELL_MASK_BEGIN */",
            "/* @BF_CELL_MASK_END */",
        );
    } else {
        strip_block(
            &mut out,
            "/* @BF_CELL_MASK_BEGIN */",
            "/* @BF_CELL_MASK_END */",
        );
    }
    if needs_input {
        keep_block(
            &mut out,
            "/* @BF_INPUT_MASK_BEGIN */",
            "/* @BF_INPUT_MASK_END */",
        );
    } else {
        strip_block(
            &mut out,
            "/* @BF_INPUT_MASK_BEGIN */",
            "/* @BF_INPUT_MASK_END */",
        );
    }

    out.replace("/* @BF_CONFIG */", config.trim_end())
        .replace("    /* @BF_PROGRAM */", ir_body.trim_end())
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BfEvalError {
    DivergenceDetected,
    InvalidOptions,
    StepBudgetExceeded,
}

#[cfg(test)]
pub(crate) fn interpret_unsigned_for_tests(
    program: &[BfIr],
    cell_bits: u32,
) -> Result<(Vec<i64>, usize), BfEvalError> {
    interpret_for_tests(
        program,
        CodegenOpts {
            io_mode: IoMode::Char,
            cell_bits,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
    )
}

#[cfg(test)]
pub(crate) use interpreter::interpret_for_tests;

#[cfg(test)]
mod interpreter;
