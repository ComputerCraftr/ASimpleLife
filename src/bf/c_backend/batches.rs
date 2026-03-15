use super::{BfIr, normalized_c_offset, push_c_line};

pub(super) fn emit_add_batch(
    out: &mut String,
    nodes: &[BfIr],
    start: usize,
    level: usize,
) -> Option<usize> {
    let mut index = start;
    let mut pointer_delta = 0;
    let mut offsets = Vec::new();
    let mut deltas = Vec::new();
    let mut source_steps = Vec::new();
    while let [BfIr::MovePtr(move_delta), BfIr::Add(add_delta), ..] = &nodes[index..] {
        if *move_delta == 0 || *add_delta == 0 || offsets.len() == 8 {
            break;
        }
        pointer_delta = normalized_c_offset(pointer_delta + normalized_c_offset(*move_delta));
        offsets.push(pointer_delta);
        deltas.push(*add_delta);
        source_steps.push((move_delta.unsigned_abs(), add_delta.unsigned_abs()));
        index += 2;
    }
    if offsets.len() < 4 {
        return None;
    }
    let count = offsets.len();
    for (move_steps, add_steps) in source_steps {
        push_c_line(out, level, "bf_work_dispatch();");
        push_c_line(out, level, "bf_work_op();");
        push_c_line(
            out,
            level,
            &format!("bf_semantic_step(UINT64_C({move_steps}));"),
        );
        push_c_line(out, level, "bf_work_dispatch();");
        push_c_line(out, level, "bf_work_op();");
        push_c_line(
            out,
            level,
            &format!("bf_semantic_step(UINT64_C({add_steps}));"),
        );
    }
    let offsets = offsets
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let deltas = deltas
        .iter()
        .map(|delta| format!("INT64_C({delta})"))
        .collect::<Vec<_>>()
        .join(", ");
    push_c_line(out, level, "{");
    push_c_line(
        out,
        level + 1,
        &format!("const ptrdiff_t offsets[] = {{{offsets}}};"),
    );
    push_c_line(
        out,
        level + 1,
        &format!("const int64_t deltas[] = {{{deltas}}};"),
    );
    push_c_line(
        out,
        level + 1,
        &format!(
            "bf_add_at_batch(tape, ptr, offsets, deltas, {count}, BF_TAPE_LEN, BF_CELL_BITS, BF_SIGNED_CELLS);"
        ),
    );
    push_c_line(
        out,
        level + 1,
        &format!("ptr = bf_wrap_ptr(ptr, {pointer_delta}, BF_TAPE_LEN);"),
    );
    push_c_line(out, level, "}");
    Some(index)
}

pub(super) fn emit_clear_batch(
    out: &mut String,
    nodes: &[BfIr],
    start: usize,
    level: usize,
) -> Option<usize> {
    let mut index = start;
    let mut offsets = Vec::new();
    while index < nodes.len() && offsets.len() < 8 {
        match nodes[index] {
            BfIr::Clear => offsets.push(0),
            BfIr::ClearAt { offset } => offsets.push(normalized_c_offset(offset)),
            _ => break,
        }
        index += 1;
    }
    if offsets.len() < 4 {
        return None;
    }
    for _ in &offsets {
        push_c_line(out, level, "bf_work_dispatch();");
        push_c_line(out, level, "bf_work_op();");
        push_c_line(out, level, "bf_semantic_summary_unsupported();");
    }
    let count = offsets.len();
    let offsets = offsets
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    push_c_line(out, level, "{");
    push_c_line(
        out,
        level + 1,
        &format!("const ptrdiff_t offsets[] = {{{offsets}}};"),
    );
    push_c_line(
        out,
        level + 1,
        &format!("const int64_t values[{count}] = {{0}};"),
    );
    push_c_line(
        out,
        level + 1,
        &format!("bf_clear_set_batch(tape, ptr, offsets, values, {count}, BF_TAPE_LEN);"),
    );
    push_c_line(out, level, "}");
    Some(index)
}
