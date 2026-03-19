use super::*;
use crate::bf::c_support::{
    mask_literal, push_c_line, signed_cells_flag, unified_input_stmt, unified_output_stmt,
    wrap_ptr_expr,
};
use crate::bf::optimizer::CodegenOpts;

fn wrap_add_expr(delta: i32) -> String {
    format!("bf_wrap_add(tape[ptr], INT64_C({delta}), BF_CELL_BITS, BF_SIGNED_CELLS)")
}

fn wrap_sub_expr(delta: i64) -> String {
    format!("bf_wrap_sub(tape[ptr], INT64_C({delta}), BF_CELL_BITS, BF_SIGNED_CELLS)")
}

fn dist_expr(offset: isize, coeff: i32) -> String {
    let target = wrap_ptr_expr(offset);
    if coeff >= 0 {
        format!(
            "tape[{target}] = bf_wrap_add(tape[{target}], bf_wrap_mul(v, INT64_C({coeff}), BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);"
        )
    } else {
        format!(
            "tape[{target}] = bf_wrap_sub(tape[{target}], bf_wrap_mul(v, INT64_C({}), BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);",
            -(coeff as i64)
        )
    }
}

fn cell_var_name(offset: isize) -> String {
    if offset < 0 {
        format!("cell_neg_{}", -offset)
    } else {
        format!("cell_{offset}")
    }
}

fn emit_symbolic_transfer_apply(out: &mut String, transfer: &SymbolicTransfer, level: usize) {
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
    for (&offset, effect) in &transfer.effects {
        let target = wrap_ptr_expr(offset);
        let line = match effect {
            SymbolicCellEffect::AddConst(value) if *value > 0 => format!(
                "tape[{target}] = bf_wrap_add({}, INT64_C({}), BF_CELL_BITS, BF_SIGNED_CELLS);",
                cell_var_name(offset),
                value
            ),
            SymbolicCellEffect::AddConst(value) if *value < 0 => format!(
                "tape[{target}] = bf_wrap_sub({}, INT64_C({}), BF_CELL_BITS, BF_SIGNED_CELLS);",
                cell_var_name(offset),
                -value
            ),
            SymbolicCellEffect::AddConst(_) | SymbolicCellEffect::Clear => {
                format!("tape[{target}] = 0;")
            }
            SymbolicCellEffect::AddScaledSource {
                source_offset,
                coeff,
            } if *coeff >= 0 => format!(
                "tape[{target}] = bf_wrap_add({}, bf_wrap_mul({}, INT64_C({}), BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);",
                cell_var_name(offset),
                cell_var_name(*source_offset),
                coeff
            ),
            SymbolicCellEffect::AddScaledSource {
                source_offset,
                coeff,
            } => format!(
                "tape[{target}] = bf_wrap_sub({}, bf_wrap_mul({}, INT64_C({}), BF_CELL_BITS, BF_SIGNED_CELLS), BF_CELL_BITS, BF_SIGNED_CELLS);",
                cell_var_name(offset),
                cell_var_name(*source_offset),
                -(*coeff as i64)
            ),
            SymbolicCellEffect::Opaque => unreachable!("opaque effects are not symbolically emitted"),
        };
        push_c_line(out, level, &line);
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
        &format!("int64_t guard = tape[{}];", wrap_ptr_expr(analysis.guard_offset)),
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
            &format!(
                "if (guard < 0 && ((uint64_t)(-guard) % UINT64_C({})) == 0) {{",
                analysis.guard_delta
            ),
        );
        push_c_line(
            out,
            level + 2,
            &format!(
                "remaining_iters = (uint64_t)(-guard) / UINT64_C({});",
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
    for power in (0..=max_power).rev() {
        let iterations = 1_u64 << power;
        push_c_line(
            out,
            level + 1,
            &format!("if (remaining_iters >= UINT64_C({iterations})) {{"),
        );
        emit_symbolic_transfer_apply(out, &analysis.powers[usize::from(power)], level + 2);
        push_c_line(
            out,
            level + 2,
            &format!("remaining_iters -= UINT64_C({iterations});"),
        );
        push_c_line(out, level + 1, "}");
    }
    push_c_line(out, level, "} else {");
    push_c_line(out, level + 1, "while (tape[ptr] != 0) {");
    push_c_line(
        out,
        level + 2,
        &format!("exec_node_{}(tape, &ptr);", analysis.body.0),
    );
    push_c_line(out, level + 1, "}");
    push_c_line(out, level, "}");
}

fn emit_raw_node(
    out: &mut String,
    engine: &mut EmitterEngine,
    id: NodeId,
    opts: CodegenOpts,
    level: usize,
) {
    match engine.interner.get(id).clone() {
        NodeKind::Add(delta) if delta > 0 => {
            push_c_line(out, level, &format!("tape[ptr] = {};", wrap_add_expr(delta)));
        }
        NodeKind::Add(delta) if delta < 0 => {
            push_c_line(out, level, &format!("tape[ptr] = {};", wrap_sub_expr(-(delta as i64))));
        }
        NodeKind::Add(_) => {}
        NodeKind::Move(delta) if delta != 0 => {
            push_c_line(out, level, &format!("ptr = {};", wrap_ptr_expr(delta)));
        }
        NodeKind::Move(_) => {}
        NodeKind::Input => push_c_line(out, level, unified_input_stmt(opts)),
        NodeKind::Output => push_c_line(out, level, unified_output_stmt(opts)),
        NodeKind::Clear => push_c_line(out, level, "tape[ptr] = 0;"),
        NodeKind::Distribute(targets) => {
            push_c_line(out, level, "{");
            push_c_line(out, level + 1, "int64_t v = tape[ptr];");
            for (offset, coeff) in targets {
                push_c_line(out, level + 1, &dist_expr(offset, coeff));
            }
            push_c_line(out, level + 1, "tape[ptr] = 0;");
            push_c_line(out, level, "}");
        }
        NodeKind::Diverge => push_c_line(out, level, "bf_diverge_forever();"),
        NodeKind::Seq(children) => {
            for child in children {
                push_c_line(out, level, &format!("exec_node_{}(tape, &ptr);", child.0));
            }
        }
        NodeKind::Loop(body) => {
            push_c_line(out, level, "while (tape[ptr] != 0) {");
            push_c_line(out, level + 1, &format!("exec_node_{}(tape, &ptr);", body.0));
            push_c_line(out, level, "}");
        }
    }
}

fn emit_exec_function(out: &mut String, engine: &mut EmitterEngine, id: NodeId, opts: CodegenOpts) {
    let plan = engine.plan_node(id);
    let transfer = matches!(plan, ExecPlan::SymbolicMemo(_)).then(|| engine.transfer(id));
    let powered = matches!(plan, ExecPlan::PoweredSymbolicLoop { .. }).then(|| {
        match engine
            .loop_analysis(id)
            .expect("powered symbolic loop plans require loop analysis")
        {
            LoopAnalysis::Powered(powered) => powered,
            _ => unreachable!("powered symbolic loop plans require powered loop analysis"),
        }
    });
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
    match plan {
        ExecPlan::SymbolicMemo(window) => {
            push_c_line(out, 1, "{");
            push_c_line(out, 2, "MemoKey key = {0};");
            push_c_line(out, 2, "MemoVal value = {0};");
            push_c_line(out, 2, &format!("key.node_id = {}u;", id.0));
            push_c_line(out, 2, &format!("key.window_start = {};", window.start));
            push_c_line(out, 2, &format!("key.window_len = {}u;", window.len));
            push_c_line(
                out,
                2,
                "for (uint8_t i = 0; i < key.window_len; ++i) { key.window[i] = tape[bf_wrap_ptr(ptr, (ptrdiff_t)key.window_start + i, BF_TAPE_LEN)]; }",
            );
            push_c_line(out, 2, "if (bf_memo_lookup(&key, &value)) {");
            push_c_line(
                out,
                3,
                "for (uint8_t i = 0; i < value.window_len; ++i) { tape[bf_wrap_ptr(ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)] = value.window[i]; }",
            );
            push_c_line(out, 3, "*ptr_ref = ptr;");
            push_c_line(out, 3, "return;");
            push_c_line(out, 2, "}");
            emit_symbolic_transfer_apply(
                out,
                transfer.as_ref().expect("symbolic memo plans require a transfer"),
                2,
            );
            push_c_line(out, 2, "value.node_id = key.node_id;");
            push_c_line(out, 2, "value.window_start = key.window_start;");
            push_c_line(out, 2, "value.window_len = key.window_len;");
            push_c_line(
                out,
                2,
                "for (uint8_t i = 0; i < value.window_len; ++i) { value.window[i] = tape[bf_wrap_ptr(ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)]; }",
            );
            push_c_line(out, 2, "bf_memo_store(&key, &value);");
            push_c_line(out, 1, "}");
        }
        ExecPlan::SymbolicLoop { body, window } => {
            push_c_line(out, 1, "{");
            push_c_line(out, 2, "MemoKey key = {0};");
            push_c_line(out, 2, "MemoVal value = {0};");
            push_c_line(out, 2, &format!("key.node_id = {}u;", id.0));
            push_c_line(out, 2, &format!("key.window_start = {};", window.start));
            push_c_line(out, 2, &format!("key.window_len = {}u;", window.len));
            push_c_line(
                out,
                2,
                "for (uint8_t i = 0; i < key.window_len; ++i) { key.window[i] = tape[bf_wrap_ptr(ptr, (ptrdiff_t)key.window_start + i, BF_TAPE_LEN)]; }",
            );
            push_c_line(out, 2, "if (bf_memo_lookup(&key, &value)) {");
            push_c_line(
                out,
                3,
                "for (uint8_t i = 0; i < value.window_len; ++i) { tape[bf_wrap_ptr(ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)] = value.window[i]; }",
            );
            push_c_line(out, 3, "*ptr_ref = ptr;");
            push_c_line(out, 3, "return;");
            push_c_line(out, 2, "}");
            push_c_line(out, 2, "while (tape[ptr] != 0) {");
            push_c_line(out, 3, &format!("exec_node_{}(tape, &ptr);", body.0));
            push_c_line(out, 2, "}");
            push_c_line(out, 2, "value.node_id = key.node_id;");
            push_c_line(out, 2, "value.window_start = key.window_start;");
            push_c_line(out, 2, "value.window_len = key.window_len;");
            push_c_line(
                out,
                2,
                "for (uint8_t i = 0; i < value.window_len; ++i) { value.window[i] = tape[bf_wrap_ptr(ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)]; }",
            );
            push_c_line(out, 2, "bf_memo_store(&key, &value);");
            push_c_line(out, 1, "}");
        }
        ExecPlan::PoweredSymbolicLoop {
            body: _,
            window,
            max_power,
        } => {
            let analysis = powered
                .as_ref()
                .expect("powered symbolic loop plans require cached analysis");
            push_c_line(out, 1, "{");
            push_c_line(out, 2, "MemoKey key = {0};");
            push_c_line(out, 2, "MemoVal value = {0};");
            push_c_line(out, 2, &format!("key.node_id = {}u;", id.0));
            push_c_line(out, 2, &format!("key.window_start = {};", window.start));
            push_c_line(out, 2, &format!("key.window_len = {}u;", window.len));
            push_c_line(
                out,
                2,
                "for (uint8_t i = 0; i < key.window_len; ++i) { key.window[i] = tape[bf_wrap_ptr(ptr, (ptrdiff_t)key.window_start + i, BF_TAPE_LEN)]; }",
            );
            push_c_line(out, 2, "if (bf_memo_lookup(&key, &value)) {");
            push_c_line(
                out,
                3,
                "for (uint8_t i = 0; i < value.window_len; ++i) { tape[bf_wrap_ptr(ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)] = value.window[i]; }",
            );
            push_c_line(out, 3, "*ptr_ref = ptr;");
            push_c_line(out, 3, "return;");
            push_c_line(out, 2, "}");
            emit_powered_loop_apply(out, analysis, max_power, 2);
            push_c_line(out, 2, "value.node_id = key.node_id;");
            push_c_line(out, 2, "value.window_start = key.window_start;");
            push_c_line(out, 2, "value.window_len = key.window_len;");
            push_c_line(
                out,
                2,
                "for (uint8_t i = 0; i < value.window_len; ++i) { value.window[i] = tape[bf_wrap_ptr(ptr, (ptrdiff_t)value.window_start + i, BF_TAPE_LEN)]; }",
            );
            push_c_line(out, 2, "bf_memo_store(&key, &value);");
            push_c_line(out, 1, "}");
        }
        _ => emit_raw_node(out, engine, id, opts, 1),
    }
    push_c_line(out, 1, "*ptr_ref = ptr;");
    push_c_line(out, 0, "}");
    out.push('\n');
}

pub fn emit_c_super(program: &[BfIr], opts: CodegenOpts) -> String {
    let mut engine = EmitterEngine::new();
    let root = engine.build_program(program);
    for index in 0..engine.interner.len() {
        let _ = engine.plan_node(NodeId(index as u32));
    }

    let mut functions = String::new();
    for index in 0..engine.interner.len() {
        let node_id = NodeId(index as u32);
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
        emit_exec_function(&mut functions, &mut engine, node_id, opts);
    }
    let config = format!(
        "#define BF_TEMPLATE_TAPE_LEN {}\n#define BF_TEMPLATE_CELL_BITS {}\n#define BF_TEMPLATE_SIGNED_CELLS {}\n#define BF_TEMPLATE_MEMO_CAPACITY {}\n#define BF_TEMPLATE_MEMO_WINDOW_MAX {}\n#define BF_TEMPLATE_MAX_NODES {}\n#define BF_TEMPLATE_INPUT_MASK {}\n#define BF_TEMPLATE_OUTPUT_MASK {}\n#define BF_TEMPLATE_ROOT_NODE {}\n",
        SUPER_C_TAPE_LEN,
        opts.cell_bits.min(63),
        signed_cells_flag(opts.cell_sign),
        SUPER_MEMO_CAPACITY,
        SUPER_MEMO_WINDOW_MAX,
        engine.interner.len(),
        mask_literal(opts.input_bits.unwrap_or(opts.cell_bits).min(63)),
        mask_literal(opts.output_bits.unwrap_or(opts.cell_bits).min(63)),
        root.0,
    );
    let functions = format!("#define BF_TEMPLATE_HAS_FUNCTIONS 1\n{functions}");
    include_str!("../bf_super.c.in")
        .replace("/* @BF_CONFIG */", &config)
        .replace("/* @BF_FUNCTIONS */", &functions)
}
