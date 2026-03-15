use super::*;

pub(super) fn seq_add_batch_end(
    engine: &EmitterEngine,
    children: &[NodeId],
    start: usize,
) -> Option<usize> {
    let mut index = start;
    let mut count = 0;
    while index + 1 < children.len() && count < 8 {
        match (
            engine.interner.get(children[index]),
            engine.interner.get(children[index + 1]),
        ) {
            (NodeKind::Move(delta), NodeKind::Add(add)) if *delta != 0 && *add != 0 => {
                count += 1;
                index += 2;
            }
            _ => break,
        }
    }
    (count >= 4).then_some(index)
}

pub(super) fn seq_clear_batch_end(
    engine: &EmitterEngine,
    children: &[NodeId],
    start: usize,
) -> Option<usize> {
    let mut index = start;
    let mut count = 0;
    while index < children.len() && count < 8 {
        if !matches!(
            engine.interner.get(children[index]),
            NodeKind::Clear | NodeKind::ClearAt(_)
        ) {
            break;
        }
        count += 1;
        index += 1;
    }
    (count >= 4).then_some(index)
}

pub(super) fn emit_seq_add_batch(
    out: &mut String,
    engine: &EmitterEngine,
    children: &[NodeId],
    start: usize,
    level: usize,
) -> Option<usize> {
    let mut index = start;
    let mut pointer_delta = 0;
    let mut offsets = Vec::new();
    let mut deltas = Vec::new();
    let mut source_steps = Vec::new();
    while index + 1 < children.len() && offsets.len() < 8 {
        let (move_delta, add_delta) = match (
            engine.interner.get(children[index]),
            engine.interner.get(children[index + 1]),
        ) {
            (NodeKind::Move(move_delta), NodeKind::Add(add_delta))
                if *move_delta != 0 && *add_delta != 0 =>
            {
                (*move_delta, *add_delta)
            }
            _ => break,
        };
        pointer_delta = normalized_c_offset(pointer_delta + normalized_c_offset(move_delta));
        offsets.push(pointer_delta);
        deltas.push(add_delta);
        source_steps.push((move_delta.unsigned_abs(), add_delta.unsigned_abs()));
        index += 2;
    }
    if offsets.len() < 4 {
        return None;
    }
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
    let count = offsets.len();
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

pub(super) fn emit_seq_clear_batch(
    out: &mut String,
    engine: &EmitterEngine,
    children: &[NodeId],
    start: usize,
    level: usize,
) -> Option<usize> {
    let mut index = start;
    let mut offsets = Vec::new();
    while index < children.len() && offsets.len() < 8 {
        match engine.interner.get(children[index]) {
            NodeKind::Clear => offsets.push(0),
            NodeKind::ClearAt(offset) => offsets.push(normalized_c_offset(*offset)),
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
