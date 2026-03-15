use super::*;
use crate::RequiredExt;
use crate::bf::c_support::{
    mask_literal, normalized_c_offset, push_c_line, signed_cells_flag, unified_input_stmt,
    unified_output_stmt, wrap_ptr_expr,
};
use crate::bf::ir::ShiftDir;
use crate::bf::optimizer::CodegenOpts;
use crate::bf::summary::{
    DEFAULT_DYNAMIC_LOOP_HOT_THRESHOLD, RUNTIME_SUMMARY_EFFECT_MAX, RUNTIME_SUMMARY_WINDOW_MAX,
    stable_summary_hash,
};
use std::collections::HashMap;

mod batches;
use batches::{emit_seq_add_batch, emit_seq_clear_batch};
mod reachable;
use reachable::executable_nodes;

type DynamicLoopOps = Vec<(u8, i32, i32, i32, u8)>;

fn lower_dynamic_loop_ops(engine: &EmitterEngine, body: NodeId) -> Option<DynamicLoopOps> {
    let mut ops = Vec::new();
    let mut stack = vec![body];
    while let Some(id) = stack.pop() {
        match engine.interner.get(id) {
            NodeKind::Add(delta) if *delta != 0 => ops.push((1, *delta, 0, 0, 0)),
            NodeKind::Add(_) | NodeKind::Move(0) => {}
            NodeKind::Move(delta) => ops.push((2, i32::try_from(*delta).ok()?, 0, 0, 0)),
            NodeKind::Affine {
                src,
                dst,
                coeff,
                preserve_src: true,
                set_dst: false,
            } => ops.push((
                3,
                i32::try_from(*src).ok()?,
                i32::try_from(*dst).ok()?,
                *coeff,
                1,
            )),
            NodeKind::Square {
                src,
                dst,
                preserve_src,
                set_dst,
            } => ops.push((
                4,
                i32::try_from(*src).ok()?,
                i32::try_from(*dst).ok()?,
                0,
                u8::from(*preserve_src) | (u8::from(*set_dst) << 1),
            )),
            NodeKind::MulAdd {
                lhs,
                rhs,
                dst,
                preserve_lhs,
                preserve_rhs,
                set_dst,
            } => {
                let flags = u8::from(*preserve_lhs)
                    | (u8::from(*preserve_rhs) << 1)
                    | (u8::from(*set_dst) << 2);
                ops.push((
                    5,
                    i32::try_from(*lhs).ok()?,
                    i32::try_from(*rhs).ok()?,
                    i32::try_from(*dst).ok()?,
                    flags,
                ));
            }
            NodeKind::Seq(children) => stack.extend(children.iter().rev().copied()),
            _ => return None,
        }
    }
    if ops.is_empty() {
        return None;
    }
    let mut pointer = 0i32;
    let mut min_offset = 0i32;
    let mut max_offset = 0i32;
    let mut guard_delta = 0i32;
    for &(kind, a, b, _, _) in &ops {
        if kind == 2 {
            pointer = pointer.checked_add(a)?;
            min_offset = min_offset.min(pointer);
            max_offset = max_offset.max(pointer);
        } else if pointer == 0 && kind == 1 {
            guard_delta = guard_delta.checked_add(a)?;
        } else if kind == 3 {
            let src = pointer.checked_add(a)?;
            let dst = pointer.checked_add(b)?;
            if src == 0 || dst == 0 {
                return None;
            }
        }
    }
    let span_minus_one = max_offset.checked_sub(min_offset)?;
    let window_max = i32::try_from(RUNTIME_SUMMARY_WINDOW_MAX).ok()?;
    (pointer == 0 && span_minus_one < window_max).then_some(())?;
    matches!(guard_delta, -1 | 1).then_some(ops)
}

fn stable_node_hash(engine: &EmitterEngine, body: NodeId) -> u64 {
    stable_summary_hash(&format!("{:?}", engine.interner.get(body)))
}

fn emit_dynamic_loop_spec(
    out: &mut String,
    engine: &EmitterEngine,
    loop_id: NodeId,
    body: NodeId,
    ops: &DynamicLoopOps,
    exact: ExactMemoSpec,
) {
    push_c_line(
        out,
        0,
        &format!("static const DynamicOp dynamic_ops_{}[] = {{", loop_id.0),
    );
    for (kind, a, b, c, flags) in ops {
        push_c_line(out, 1, &format!("{{ {kind}U, {a}, {b}, {c}, {flags}U }},"));
    }
    push_c_line(out, 0, "};");
    push_c_line(
        out,
        0,
        &format!(
            "static const DynamicLoopSpec dynamic_spec_{} = {{ {}U, UINT64_C({}), {}, {}U, {}U, dynamic_ops_{} }};",
            loop_id.0,
            loop_id.0,
            stable_node_hash(engine, body),
            exact.window.start,
            exact.window.len,
            ops.len(),
            loop_id.0
        ),
    );
    out.push('\n');
}

fn emit_dynamic_loop_fast_path(out: &mut String, id: NodeId, body: NodeId, level: usize) {
    push_c_line(
        out,
        level,
        &format!(
            "int dynamic_status = bf_semantic_fuel_enabled() ? 0 : bf_dynamic_loop_try_summary(&dynamic_spec_{}, tape, ptr);",
            id.0
        ),
    );
    push_c_line(out, level, "if (dynamic_status == 1) {");
    push_c_line(out, level + 1, "recursion_depth--;");
    push_c_line(out, level + 1, "*ptr_ref = ptr;");
    push_c_line(out, level + 1, "return;");
    push_c_line(out, level, "}");
    push_c_line(out, level, "if (dynamic_status == -2) {");
    push_c_line(
        out,
        level + 1,
        "int64_t transducer_input[BF_INLINE_MEMO_WINDOW] = {0};",
    );
    push_c_line(
        out,
        level + 1,
        &format!(
            "bf_capture_dynamic_window(&dynamic_spec_{}, tape, ptr, transducer_input);",
            id.0
        ),
    );
    push_c_line(out, level + 1, "dynamic_fallback_entries++;");
    push_c_line(out, level + 1, "while (tape[ptr] != 0) {");
    push_c_line(out, level + 2, "dynamic_fallback_iterations++;");
    push_c_line(out, level + 2, "bf_work_loop_iter();");
    push_c_line(
        out,
        level + 2,
        &format!("exec_node_{}(tape, &ptr);", body.0),
    );
    push_c_line(out, level + 1, "}");
    push_c_line(
        out,
        level + 1,
        &format!(
            "bf_record_runtime_transducer(&dynamic_spec_{}, transducer_input, tape, ptr);",
            id.0
        ),
    );
    push_c_line(out, level + 1, "recursion_depth--;");
    push_c_line(out, level + 1, "*ptr_ref = ptr;");
    push_c_line(out, level + 1, "return;");
    push_c_line(out, level, "}");
    push_c_line(out, level, "if (dynamic_status == -1) {");
    push_c_line(out, level + 1, "dynamic_memo_after_rejection++;");
    push_c_line(out, level, "}");
}

fn wrap_add_expr(delta: i32) -> String {
    format!("bf_wrap_add(tape[ptr], INT64_C({delta}), BF_CELL_BITS, BF_SIGNED_CELLS)")
}

fn wrap_sub_expr(delta: i64) -> String {
    format!("bf_wrap_sub(tape[ptr], INT64_C({delta}), BF_CELL_BITS, BF_SIGNED_CELLS)")
}

fn cell_expr(offset: crate::bf::BfOffset) -> String {
    format!("tape[{}]", wrap_ptr_expr(offset))
}

fn wrap_shift_left_expr(src_expr: &str, amount: u32) -> String {
    format!("bf_wrap_shift_left({src_expr}, {amount}, BF_CELL_BITS, BF_SIGNED_CELLS)")
}

fn wrap_shift_right_expr(src_expr: &str, amount: u32) -> String {
    format!("bf_wrap_shift_right({src_expr}, {amount}, BF_CELL_BITS, BF_SIGNED_CELLS)")
}

fn cell_var_name(offset: crate::bf::BfOffset) -> String {
    if offset < 0 {
        format!("cell_neg_{}", -offset)
    } else {
        format!("cell_{offset}")
    }
}

fn emit_symbolic_transfer_apply(out: &mut String, transfer: &SymbolicTransfer, level: usize) {
    push_c_line(out, level, "bf_work_op();");
    for &offset in &transfer.reads {
        push_c_line(
            out,
            level,
            &format!(
                "int64_t {} = tape[{}];",
                cell_var_name(offset),
                wrap_ptr_expr(offset)
            ),
        );
    }
    let effects = transfer.effects.iter().collect::<Vec<_>>();
    let mut index = 0;
    while index < effects.len() {
        let (&offset, polynomial) = effects[index];
        if polynomial.is_zero() {
            let start = offset;
            let mut end = offset;
            while index + 1 < effects.len()
                && effects[index + 1].0 == &(end + 1)
                && effects[index + 1].1.is_zero()
            {
                index += 1;
                end += 1;
            }
            push_c_line(
                out,
                level,
                &format!("bf_zero_region(tape, ptr, {start}, {}U);", end - start + 1),
            );
            index += 1;
            continue;
        }
        let target = wrap_ptr_expr(offset);
        let line = format!("tape[{target}] = {};", symbolic_polynomial_expr(polynomial));
        push_c_line(out, level, &line);
        index += 1;
    }
}

fn symbolic_polynomial_expr(polynomial: &SymbolicPolynomial) -> String {
    let mut expression = c_i64_literal(polynomial.constant);
    for (&term, &coeff) in &polynomial.terms {
        let base = match term {
            SymbolicMonomial::Linear(offset) => cell_var_name(offset),
            SymbolicMonomial::Product(lhs, rhs) => format!(
                "bf_wrap_mul({}, {}, BF_CELL_BITS, BF_SIGNED_CELLS)",
                cell_var_name(lhs),
                cell_var_name(rhs)
            ),
        };
        let value = if coeff == 1 {
            base
        } else {
            format!(
                "bf_wrap_mul({base}, {}, BF_CELL_BITS, BF_SIGNED_CELLS)",
                c_i64_literal(coeff)
            )
        };
        expression = format!("bf_wrap_add({expression}, {value}, BF_CELL_BITS, BF_SIGNED_CELLS)");
    }
    expression
}

fn c_i64_literal(value: i64) -> String {
    if value == i64::MIN {
        "(-INT64_C(9223372036854775807) - INT64_C(1))".to_string()
    } else {
        format!("INT64_C({value})")
    }
}

fn emit_powered_loop_apply(
    out: &mut String,
    analysis: &PoweredLoopAnalysis,
    max_power: u8,
    level: usize,
) {
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
    for power in (0..=max_power).rev() {
        let iterations = 1_u64 << power;
        let condition = if power == max_power { "while" } else { "if" };
        push_c_line(
            out,
            level + 1,
            &format!("{condition} (remaining_iters >= UINT64_C({iterations})) {{"),
        );
        push_c_line(out, level + 2, "symbolic_power_hits++;");
        emit_symbolic_transfer_apply(out, &analysis.powers[usize::from(power)], level + 2);
        push_c_line(
            out,
            level + 2,
            &format!("remaining_iters -= UINT64_C({iterations});"),
        );
        push_c_line(out, level + 1, "}");
    }
    push_c_line(out, level, "} else {");
    push_c_line(out, level + 1, "recursion_fallbacks++;");
    push_c_line(out, level + 1, "while (tape[ptr] != 0) {");
    push_c_line(out, level + 2, "bf_work_loop_iter();");
    push_c_line(
        out,
        level + 2,
        &format!("exec_node_{}(tape, &ptr);", analysis.body.0),
    );
    push_c_line(out, level + 1, "}");
    push_c_line(out, level, "}");
}

fn emit_exact_memo_prologue(
    out: &mut String,
    id: NodeId,
    exact: ExactMemoSpec,
    level: usize,
    body_level: usize,
) {
    push_c_line(out, level, "{");
    push_c_line(out, level + 1, "ptrdiff_t memo_base_ptr = ptr;");
    push_c_line(out, level + 1, "MemoKey key = {0};");
    push_c_line(out, level + 1, "MemoVal value = {0};");
    push_c_line(out, level + 1, &format!("key.node_id = {}U;", id.0));
    push_c_line(
        out,
        level + 1,
        &format!("key.window_start = {};", exact.window.start),
    );
    push_c_line(
        out,
        level + 1,
        &format!("key.window_len = {}U;", exact.window.len),
    );
    push_c_line(
        out,
        level + 1,
        "for (uint8_t i = 0; i < key.window_len; ++i) {",
    );
    push_c_line(
        out,
        level + 2,
        "key.window[i] = tape[bf_wrap_ptr(memo_base_ptr, (ptrdiff_t)key.window_start + i, BF_TAPE_LEN)];",
    );
    push_c_line(out, level + 1, "}");
    push_c_line(
        out,
        level + 1,
        "if (!bf_semantic_fuel_enabled() && bf_memo_lookup(&key, &value)) {",
    );
    push_c_line(
        out,
        level + 2,
        "for (uint8_t i = 0; i < value.window_len; ++i) {",
    );
    push_c_line(
        out,
        level + 3,
        "tape[bf_wrap_ptr(memo_base_ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)] = value.window[i];",
    );
    push_c_line(out, level + 2, "}");
    push_c_line(
        out,
        level + 2,
        "ptr = bf_wrap_ptr(memo_base_ptr, value.ptr_delta, BF_TAPE_LEN);",
    );
    push_c_line(out, level + 2, "recursion_depth--;");
    push_c_line(out, level + 2, "*ptr_ref = ptr;");
    push_c_line(out, level + 2, "return;");
    push_c_line(out, level + 1, "}");
    let _ = body_level;
}

fn emit_exact_memo_epilogue(out: &mut String, exact: ExactMemoSpec, level: usize) {
    push_c_line(out, level + 1, "value.node_id = key.node_id;");
    push_c_line(out, level + 1, "value.window_start = key.window_start;");
    push_c_line(out, level + 1, "value.window_len = key.window_len;");
    push_c_line(
        out,
        level + 1,
        &format!("value.ptr_delta = {};", exact.ptr_delta),
    );
    push_c_line(
        out,
        level + 1,
        "for (uint8_t i = 0; i < value.window_len; ++i) {",
    );
    push_c_line(
        out,
        level + 2,
        "value.window[i] = tape[bf_wrap_ptr(memo_base_ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)];",
    );
    push_c_line(out, level + 1, "}");
    push_c_line(out, level + 1, "bf_memo_store(&key, &value);");
    push_c_line(out, level, "}");
}

fn emit_raw_node(
    out: &mut String,
    engine: &mut EmitterEngine,
    id: NodeId,
    opts: CodegenOpts,
    level: usize,
) {
    match engine.interner.get(id) {
        NodeKind::Add(delta) if *delta > 0 => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(
                out,
                level,
                &format!("bf_semantic_step(UINT64_C({}));", delta.unsigned_abs()),
            );
            push_c_line(
                out,
                level,
                &format!("tape[ptr] = {};", wrap_add_expr(*delta)),
            );
        }
        NodeKind::Add(delta) if *delta < 0 => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(
                out,
                level,
                &format!("bf_semantic_step(UINT64_C({}));", delta.unsigned_abs()),
            );
            push_c_line(
                out,
                level,
                &format!("tape[ptr] = {};", wrap_sub_expr(-i64::from(*delta))),
            );
        }
        NodeKind::Add(_) => {}
        NodeKind::Move(delta) if *delta != 0 => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(
                out,
                level,
                &format!("bf_semantic_step(UINT64_C({}));", delta.unsigned_abs()),
            );
            push_c_line(out, level, &format!("ptr = {};", wrap_ptr_expr(*delta)));
        }
        NodeKind::Move(_) => {}
        NodeKind::Input => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_step(UINT64_C(1));");
            push_c_line(out, level, unified_input_stmt(opts));
        }
        NodeKind::Output => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_step(UINT64_C(1));");
            push_c_line(out, level, unified_output_stmt(opts));
        }
        NodeKind::Clear => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            push_c_line(out, level, "tape[ptr] = 0;");
        }
        NodeKind::ClearAt(offset) => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            push_c_line(
                out,
                level,
                &format!("tape[{}] = 0;", wrap_ptr_expr(*offset)),
            );
        }
        NodeKind::Scan { stride } => {
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
        NodeKind::Distribute {
            targets,
            preserve_src,
        } => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            push_c_line(out, level, "{");
            if targets.is_empty() {
                if !*preserve_src {
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
        NodeKind::Affine {
            src,
            dst,
            coeff,
            preserve_src,
            set_dst,
        } => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            let src_expr = cell_expr(*src);
            let dst_expr = cell_expr(*dst);
            push_c_line(out, level, "{");
            push_c_line(out, level + 1, &format!("int64_t src_v = {src_expr};"));
            if !*set_dst && dst != src {
                push_c_line(out, level + 1, &format!("int64_t dst_v = {dst_expr};"));
            }
            let base = if *set_dst {
                "INT64_C(0)".to_string()
            } else if dst == src {
                "src_v".to_string()
            } else {
                "dst_v".to_string()
            };
            push_c_line(
                out,
                level + 1,
                &format!(
                    "int64_t dst_next = bf_wrap_add({}, bf_wrap_mul(src_v, INT64_C({coeff}), BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);",
                    base
                ),
            );
            if !*preserve_src && src != dst {
                push_c_line(out, level + 1, &format!("{src_expr} = 0;"));
            }
            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
            push_c_line(out, level, "}");
        }
        NodeKind::Shift {
            src,
            dst,
            amount,
            dir,
            preserve_src,
            set_dst,
        } => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            let src_expr = cell_expr(*src);
            let dst_expr = cell_expr(*dst);
            push_c_line(out, level, "{");
            push_c_line(out, level + 1, &format!("int64_t src_v = {src_expr};"));
            if !*set_dst && dst != src {
                push_c_line(out, level + 1, &format!("int64_t dst_v = {dst_expr};"));
            }
            let base = if *set_dst {
                "INT64_C(0)".to_string()
            } else if dst == src {
                "src_v".to_string()
            } else {
                "dst_v".to_string()
            };
            push_c_line(
                out,
                level + 1,
                &format!(
                    "int64_t dst_next = bf_wrap_add({}, {}, BF_CELL_BITS, BF_SIGNED_CELLS);",
                    base,
                    match dir {
                        ShiftDir::Left => wrap_shift_left_expr("src_v", *amount),
                        ShiftDir::Right => wrap_shift_right_expr("src_v", *amount),
                    }
                ),
            );
            if !*preserve_src && src != dst {
                push_c_line(out, level + 1, &format!("{src_expr} = 0;"));
            }
            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
            push_c_line(out, level, "}");
        }
        NodeKind::Square {
            src,
            dst,
            preserve_src,
            set_dst,
        } => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            let src_expr = cell_expr(*src);
            let dst_expr = cell_expr(*dst);
            push_c_line(out, level, "{");
            push_c_line(out, level + 1, &format!("int64_t src_v = {src_expr};"));
            if !*set_dst && dst != src {
                push_c_line(out, level + 1, &format!("int64_t dst_v = {dst_expr};"));
            }
            let base = if *set_dst {
                "INT64_C(0)".to_string()
            } else if dst == src {
                "src_v".to_string()
            } else {
                "dst_v".to_string()
            };
            push_c_line(
                out,
                level + 1,
                &format!(
                    "int64_t dst_next = bf_wrap_add({}, bf_wrap_mul(src_v, src_v, BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);",
                    base
                ),
            );
            if !*preserve_src && src != dst {
                push_c_line(out, level + 1, &format!("{src_expr} = 0;"));
            }
            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
            push_c_line(out, level, "}");
        }
        NodeKind::MulAdd {
            lhs,
            rhs,
            dst,
            preserve_lhs,
            preserve_rhs,
            set_dst,
        } => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_semantic_summary_unsupported();");
            let lhs_expr = cell_expr(*lhs);
            let rhs_expr = cell_expr(*rhs);
            let dst_expr = cell_expr(*dst);
            push_c_line(out, level, "{");
            push_c_line(out, level + 1, &format!("int64_t lhs_v = {lhs_expr};"));
            push_c_line(out, level + 1, &format!("int64_t rhs_v = {rhs_expr};"));
            if !*set_dst && dst != lhs && dst != rhs {
                push_c_line(out, level + 1, &format!("int64_t dst_v = {dst_expr};"));
            }
            let base = if *set_dst {
                "INT64_C(0)".to_string()
            } else if dst == lhs {
                "lhs_v".to_string()
            } else if dst == rhs {
                "rhs_v".to_string()
            } else {
                "dst_v".to_string()
            };
            push_c_line(
                out,
                level + 1,
                &format!(
                    "int64_t dst_next = bf_wrap_add({}, bf_wrap_mul(lhs_v, rhs_v, BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);",
                    base
                ),
            );
            if !*preserve_lhs && lhs != dst {
                push_c_line(out, level + 1, &format!("{lhs_expr} = 0;"));
            }
            if !*preserve_rhs && rhs != dst && (rhs != lhs || *preserve_lhs) {
                push_c_line(out, level + 1, &format!("{rhs_expr} = 0;"));
            }
            push_c_line(out, level + 1, &format!("{dst_expr} = dst_next;"));
            push_c_line(out, level, "}");
        }
        NodeKind::Diverge => {
            push_c_line(out, level, "bf_work_op();");
            push_c_line(out, level, "bf_diverge_forever();");
        }
        NodeKind::Seq(children) => {
            let mut index = 0;
            while index < children.len() {
                if let Some(next) = emit_seq_add_batch(out, engine, children, index, level) {
                    index = next;
                    continue;
                }
                if let Some(next) = emit_seq_clear_batch(out, engine, children, index, level) {
                    index = next;
                    continue;
                }
                push_c_line(
                    out,
                    level,
                    &format!("exec_node_{}(tape, &ptr);", children[index].0),
                );
                index += 1;
            }
        }
        NodeKind::Loop(body) => {
            push_c_line(out, level, "while (tape[ptr] != 0) {");
            push_c_line(out, level + 1, "bf_work_loop_iter();");
            push_c_line(out, level + 1, "bf_semantic_step(UINT64_C(1));");
            push_c_line(
                out,
                level + 1,
                &format!("exec_node_{}(tape, &ptr);", body.0),
            );
            push_c_line(out, level, "}");
        }
    }
}

fn emit_exec_function(
    out: &mut String,
    engine: &mut EmitterEngine,
    dynamic_loops: &HashMap<NodeId, (NodeId, DynamicLoopOps, ExactMemoSpec)>,
    id: NodeId,
    opts: CodegenOpts,
) {
    let plan = engine.plan_node(id);
    let dynamic_body = dynamic_loops.get(&id).map(|(body, _, _)| *body);
    push_c_line(
        out,
        0,
        &format!(
            "static void exec_node_{}(int64_t tape[BF_TAPE_LEN], ptrdiff_t *ptr_ref) {{",
            id.0
        ),
    );
    push_c_line(out, 1, "(void)tape;");
    push_c_line(out, 1, "ptrdiff_t ptr = *ptr_ref;");
    push_c_line(out, 1, "bf_work_dispatch();");
    push_c_line(out, 1, "recursion_depth++;");
    push_c_line(out, 1, "if (recursion_depth > recursion_depth_max) {");
    push_c_line(out, 2, "recursion_depth_max = recursion_depth;");
    push_c_line(out, 1, "}");
    match plan {
        ExecPlan::ExactMemo(exact) => {
            emit_exact_memo_prologue(out, id, exact, 1, 2);
            emit_raw_node(out, engine, id, opts, 2);
            emit_exact_memo_epilogue(out, exact, 1);
        }
        ExecPlan::ExactLoopMemo { body, exact } => {
            if let Some(dynamic_body) = dynamic_body {
                emit_dynamic_loop_fast_path(out, id, dynamic_body, 1);
            }
            emit_exact_memo_prologue(out, id, exact, 1, 2);
            push_c_line(out, 2, "recursion_fallbacks++;");
            if dynamic_body.is_some() {
                push_c_line(out, 2, "dynamic_fallback_entries++;");
            }
            push_c_line(out, 2, "while (tape[ptr] != 0) {");
            push_c_line(out, 3, "bf_work_loop_iter();");
            push_c_line(out, 3, "bf_semantic_step(UINT64_C(1));");
            if dynamic_body.is_some() {
                push_c_line(out, 3, "dynamic_fallback_iterations++;");
            }
            push_c_line(out, 3, &format!("exec_node_{}(tape, &ptr);", body.0));
            push_c_line(out, 2, "}");
            emit_exact_memo_epilogue(out, exact, 1);
        }
        ExecPlan::ExactPoweredLoopMemo {
            body,
            exact,
            max_power,
        } => {
            let analysis = engine
                .loop_analysis(id)
                .or_invariant("powered symbolic loop plans require cached analysis");
            let powered = match analysis {
                LoopAnalysis::ExactMemoPlusSymbolicPower { powered, .. } => powered,
                _ => crate::invariant_failure!(
                    "powered symbolic loop plans require powered loop analysis"
                ),
            };
            emit_exact_memo_prologue(out, id, exact, 1, 2);
            push_c_line(out, 2, "if (bf_semantic_fuel_enabled()) {");
            push_c_line(out, 3, "while (tape[ptr] != 0) {");
            push_c_line(out, 4, "bf_work_loop_iter();");
            push_c_line(out, 4, "bf_semantic_step(UINT64_C(1));");
            push_c_line(out, 4, &format!("exec_node_{}(tape, &ptr);", body.0));
            push_c_line(out, 3, "}");
            push_c_line(out, 2, "} else {");
            emit_powered_loop_apply(out, powered, max_power, 3);
            push_c_line(out, 2, "}");
            emit_exact_memo_epilogue(out, exact, 1);
        }
        _ => emit_raw_node(out, engine, id, opts, 1),
    }
    push_c_line(out, 1, "recursion_depth--;");
    push_c_line(out, 1, "*ptr_ref = ptr;");
    push_c_line(out, 0, "}");
    out.push('\n');
}

pub fn emit_c_super(program: &[BfIr], opts: CodegenOpts) -> String {
    let mut engine = EmitterEngine::new();
    let root = engine.build_program(program);
    for index in 0..engine.interner.len() {
        let node_id = NodeId(u32::try_from(index).or_invariant("super-C node count exceeded u32"));
        let _ = engine.plan_node(node_id);
    }
    let executable = executable_nodes(&mut engine, root);

    let mut dynamic_loops = HashMap::new();
    for (index, is_executable) in executable.iter().copied().enumerate() {
        if !is_executable {
            continue;
        }
        let node_id = NodeId(u32::try_from(index).or_invariant("super-C node count exceeded u32"));
        if let ExecPlan::ExactLoopMemo { body, exact } = engine.plan_node(node_id)
            && let Some(ops) = lower_dynamic_loop_ops(&engine, body)
        {
            dynamic_loops.insert(node_id, (body, ops, exact));
        }
    }

    let mut functions = String::new();
    for (index, is_executable) in executable.iter().copied().enumerate() {
        if !is_executable {
            continue;
        }
        let node_id = NodeId(u32::try_from(index).or_invariant("super-C node count exceeded u32"));
        if let Some((body, ops, exact)) = dynamic_loops.get(&node_id) {
            emit_dynamic_loop_spec(&mut functions, &engine, node_id, *body, ops, *exact);
        }
    }
    for (index, is_executable) in executable.iter().copied().enumerate() {
        if !is_executable {
            continue;
        }
        let node_id = NodeId(u32::try_from(index).or_invariant("super-C node count exceeded u32"));
        let node_debug = format!("{:?}", engine.interner.get(node_id));
        let decision = engine.plan_decision(node_id);
        push_c_line(
            &mut functions,
            0,
            &format!(
                "/* node {} {} plan={:?} cost={} */",
                index, node_debug, decision.plan, decision.estimated_cost
            ),
        );
        emit_exec_function(&mut functions, &mut engine, &dynamic_loops, node_id, opts);
    }
    let config = format!(
        "#define BF_TEMPLATE_TAPE_LEN {}\n#define BF_TEMPLATE_CELL_BITS {}\n#define BF_TEMPLATE_SIGNED_CELLS {}\n#define BF_TEMPLATE_MEMO_CAPACITY {}\n#define BF_TEMPLATE_MEMO_WINDOW_MAX {}\n#define BF_TEMPLATE_MAX_NODES {}\n#define BF_TEMPLATE_INPUT_MASK {}\n#define BF_TEMPLATE_OUTPUT_MASK {}\n#define BF_TEMPLATE_ROOT_NODE {}\n#define BF_TEMPLATE_DYNAMIC_HOT_THRESHOLD {}\n#define BF_TEMPLATE_RUNTIME_SUMMARY_EFFECT_MAX {}\n",
        SUPER_C_TAPE_LEN,
        opts.cell_bits.min(63),
        signed_cells_flag(opts.cell_sign),
        SUPER_MEMO_CAPACITY,
        SUPER_MEMO_WINDOW_MAX,
        engine.interner.len(),
        mask_literal(opts.input_bits.unwrap_or(opts.cell_bits).min(63)),
        mask_literal(opts.output_bits.unwrap_or(opts.cell_bits).min(63)),
        root.0,
        DEFAULT_DYNAMIC_LOOP_HOT_THRESHOLD,
        RUNTIME_SUMMARY_EFFECT_MAX,
    );
    let functions = format!("#define BF_TEMPLATE_HAS_FUNCTIONS 1\n{functions}");
    include_str!("../bf_super.c.in")
        .replace("/* @BF_CONFIG */", config.trim_end())
        .replace("/* @BF_FUNCTIONS */", functions.trim_end())
}
