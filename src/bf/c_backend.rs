use super::BF_C_TAPE_LEN as C_TAPE_LEN;
use super::c_support::{
    mask_literal, normalized_c_offset, push_c_line, signed_cells_flag, split_char_input_stmt,
    split_number_input_stmt, split_output_stmt, wrap_ptr_expr,
};
use super::ir::{BfIr, ShiftDir, validate_canonical_ir};
#[cfg(test)]
use super::optimizer::CellSign;
use super::optimizer::{CodegenOpts, IoMode};
use crate::RequiredExt;

mod batches;
use batches::{emit_add_batch, emit_clear_batch};

#[cfg(test)]
const BF_TEST_STEP_BUDGET: u64 = 10_000_000;

fn indent(n: usize) -> String {
    " ".repeat(n)
}

pub fn format_ir(program: &[BfIr]) -> String {
    enum Frame<'a> {
        Seq {
            nodes: &'a [BfIr],
            index: usize,
            indent: usize,
        },
        Close {
            indent: usize,
        },
    }

    let mut out = String::new();
    let mut stack = vec![Frame::Seq {
        nodes: program,
        index: 0,
        indent: 0,
    }];

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Seq {
                nodes,
                mut index,
                indent: ind,
            } => {
                if index >= nodes.len() {
                    continue;
                }
                let pad = indent(ind);
                let node = &nodes[index];
                index += 1;
                stack.push(Frame::Seq {
                    nodes,
                    index,
                    indent: ind,
                });
                match node {
                    BfIr::MovePtr(n) => out.push_str(&format!("{pad}MovePtr({n})\n")),
                    BfIr::Add(n) => out.push_str(&format!("{pad}Add({n})\n")),
                    BfIr::Input => out.push_str(&format!("{pad}Input\n")),
                    BfIr::Output => out.push_str(&format!("{pad}Output\n")),
                    BfIr::Clear => out.push_str(&format!("{pad}Clear\n")),
                    BfIr::ClearAt { offset } => {
                        out.push_str(&format!("{pad}ClearAt({offset})\n"));
                    }
                    BfIr::Scan { stride } => {
                        out.push_str(&format!("{pad}Scan {{ stride: {stride} }}\n"))
                    }
                    BfIr::Shift {
                        src,
                        dst,
                        amount,
                        dir,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Shift {{ src: {src}, dst: {dst}, amount: {amount}, dir: {dir:?}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::Affine {
                        src,
                        dst,
                        coeff,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Affine {{ src: {src}, dst: {dst}, coeff: {coeff}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::Square {
                        src,
                        dst,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Square {{ src: {src}, dst: {dst}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::MulAdd {
                        lhs,
                        rhs,
                        dst,
                        preserve_lhs,
                        preserve_rhs,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}MulAdd {{ lhs: {lhs}, rhs: {rhs}, dst: {dst}, preserve_lhs: {preserve_lhs}, preserve_rhs: {preserve_rhs}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::Diverge => out.push_str(&format!("{pad}Diverge\n")),
                    BfIr::Distribute {
                        targets,
                        preserve_src,
                    } => out.push_str(&format!(
                        "{pad}Distribute {{ targets: {targets:?}, preserve_src: {preserve_src} }}\n"
                    )),
                    BfIr::Loop(body) => {
                        out.push_str(&format!("{pad}Loop {{\n"));
                        stack.push(Frame::Close { indent: ind });
                        stack.push(Frame::Seq {
                            nodes: body,
                            index: 0,
                            indent: ind + 2,
                        });
                    }
                }
            }
            Frame::Close { indent: ind } => out.push_str(&format!(
                "{}}}
",
                indent(ind)
            )),
        }
    }
    out
}

pub fn emit_c(program: &[BfIr], opts: CodegenOpts) -> String {
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
        },
        Close {
            level: usize,
        },
    }

    fn emit_body(out: &mut String, program: &[BfIr], opts: CodegenOpts) {
        let mut stack = vec![EmitFrame::Seq {
            nodes: program,
            index: 0,
            level: 1,
        }];
        while let Some(frame) = stack.pop() {
            match frame {
                EmitFrame::Seq {
                    nodes,
                    mut index,
                    level,
                } => {
                    if index >= nodes.len() {
                        continue;
                    }
                    if let Some(next_index) = emit_add_batch(out, nodes, index, level) {
                        stack.push(EmitFrame::Seq {
                            nodes,
                            index: next_index,
                            level,
                        });
                        continue;
                    }
                    if let Some(next_index) = emit_clear_batch(out, nodes, index, level) {
                        stack.push(EmitFrame::Seq {
                            nodes,
                            index: next_index,
                            level,
                        });
                        continue;
                    }
                    let node = &nodes[index];
                    index += 1;
                    stack.push(EmitFrame::Seq {
                        nodes,
                        index,
                        level,
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

    let mut out = include_str!("bf.c.in").to_owned();

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
pub(crate) fn interpret_for_tests(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<(Vec<i64>, usize), BfEvalError> {
    fn wrap_unsigned(v: i64, bits: u32) -> i64 {
        if bits == 0 {
            return 0;
        }
        let mask = (1_u64 << bits) - 1;
        (v.cast_unsigned() & mask).cast_signed()
    }

    fn wrap_signed(v: i64, bits: u32) -> i64 {
        if bits == 0 {
            return 0;
        }
        let mask = (1_u64 << bits) - 1;
        let raw = v.cast_unsigned() & mask;
        let sign_bit = 1_u64 << (bits - 1);
        if raw & sign_bit == 0 {
            raw.cast_signed()
        } else {
            i64::try_from(i128::from(raw) - (1_i128 << bits))
                .or_invariant("wrapped signed cell fits i64")
        }
    }

    fn wrap(v: i64, bits: u32, sign: CellSign) -> i64 {
        match sign {
            CellSign::Signed => wrap_signed(v, bits),
            CellSign::Unsigned => wrap_unsigned(v, bits),
        }
    }

    let cell_bits = opts.cell_bits.min(63);
    let mut tape = vec![0_i64; C_TAPE_LEN];
    let mut ptr = 0_usize;
    let mut stack: Vec<(&[BfIr], usize)> = vec![(program, 0)];
    let mut steps = 0_u64;

    while let Some((nodes, index)) = stack.last_mut() {
        if *index >= nodes.len() {
            stack.pop();
            continue;
        }
        if steps >= BF_TEST_STEP_BUDGET {
            return Err(BfEvalError::StepBudgetExceeded);
        }
        steps += 1;

        let node = &nodes[*index];
        *index += 1;

        match node {
            BfIr::MovePtr(n) => {
                ptr = super::tape::wrapped_index(ptr, *n, tape.len());
            }
            BfIr::Add(n) => {
                tape[ptr] = wrap(tape[ptr] + i64::from(*n), cell_bits, opts.cell_sign);
            }
            BfIr::Input | BfIr::Output => {}
            BfIr::Clear => tape[ptr] = 0,
            BfIr::ClearAt { offset } => {
                let target = super::tape::wrapped_index(ptr, *offset, tape.len());
                tape[target] = 0;
            }
            BfIr::Scan { stride } => {
                while tape[ptr] != 0 {
                    if steps >= BF_TEST_STEP_BUDGET {
                        return Err(BfEvalError::StepBudgetExceeded);
                    }
                    steps += 1;
                    ptr = super::tape::wrapped_index(ptr, *stride, tape.len());
                }
            }
            BfIr::Distribute {
                targets,
                preserve_src,
            } => {
                let v = tape[ptr];
                for &(offset, coeff) in targets {
                    let t = super::tape::wrapped_index(ptr, offset, tape.len());
                    tape[t] = wrap(tape[t] + v * i64::from(coeff), cell_bits, opts.cell_sign);
                }
                if !preserve_src {
                    tape[ptr] = 0;
                }
            }
            BfIr::Affine {
                src,
                dst,
                coeff,
                preserve_src,
                set_dst,
            } => {
                let s = super::tape::wrapped_index(ptr, *src, tape.len());
                let d = super::tape::wrapped_index(ptr, *dst, tape.len());
                let src_v = tape[s];
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(base + src_v * i64::from(*coeff), cell_bits, opts.cell_sign);
                if !preserve_src && s != d {
                    tape[s] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::Shift {
                src,
                dst,
                amount,
                dir,
                preserve_src,
                set_dst,
            } => {
                let s = super::tape::wrapped_index(ptr, *src, tape.len());
                let d = super::tape::wrapped_index(ptr, *dst, tape.len());
                let src_raw = tape[s].cast_unsigned() & ((1_u64 << cell_bits) - 1);
                let shifted_raw = match dir {
                    ShiftDir::Left => src_raw.checked_shl(*amount).unwrap_or(0),
                    ShiftDir::Right => src_raw.checked_shr(*amount).unwrap_or(0),
                };
                let shifted = match dir {
                    ShiftDir::Left => wrap(
                        wrap_signed(shifted_raw.cast_signed(), cell_bits),
                        cell_bits,
                        opts.cell_sign,
                    ),
                    ShiftDir::Right => match opts.cell_sign {
                        CellSign::Signed => wrap_signed(shifted_raw.cast_signed(), cell_bits),
                        CellSign::Unsigned => wrap_unsigned(shifted_raw.cast_signed(), cell_bits),
                    },
                };
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(base + shifted, cell_bits, opts.cell_sign);
                if !preserve_src && s != d {
                    tape[s] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::Square {
                src,
                dst,
                preserve_src,
                set_dst,
            } => {
                let s = super::tape::wrapped_index(ptr, *src, tape.len());
                let d = super::tape::wrapped_index(ptr, *dst, tape.len());
                let src_v = tape[s];
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(base + src_v * src_v, cell_bits, opts.cell_sign);
                if !preserve_src && s != d {
                    tape[s] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::MulAdd {
                lhs,
                rhs,
                dst,
                preserve_lhs,
                preserve_rhs,
                set_dst,
            } => {
                let l = super::tape::wrapped_index(ptr, *lhs, tape.len());
                let r = super::tape::wrapped_index(ptr, *rhs, tape.len());
                let d = super::tape::wrapped_index(ptr, *dst, tape.len());
                let lhs_v = tape[l];
                let rhs_v = tape[r];
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(base + lhs_v * rhs_v, cell_bits, opts.cell_sign);
                if !preserve_lhs && l != d {
                    tape[l] = 0;
                }
                if !preserve_rhs && r != d && (r != l || *preserve_lhs) {
                    tape[r] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::Diverge => return Err(BfEvalError::DivergenceDetected),
            BfIr::Loop(body) => {
                if tape[ptr] != 0 {
                    stack.last_mut().or_invariant("required value").1 -= 1;
                    stack.push((body, 0));
                }
            }
        }
    }
    Ok((tape, ptr))
}
