use super::c_support::{
    mask_literal, push_c_line, signed_cells_flag, split_char_input_stmt, split_number_input_stmt,
    split_output_stmt, wrap_ptr_expr,
};
use super::ir::BfIr;
#[cfg(test)]
use super::optimizer::CellSign;
use super::optimizer::{CodegenOpts, IoMode};

pub(super) const C_TAPE_LEN: usize = 30_000;
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
                    BfIr::Diverge => out.push_str(&format!("{pad}Diverge\n")),
                    BfIr::Distribute { targets } => {
                        out.push_str(&format!("{pad}Distribute {{ targets: {targets:?} }}\n"))
                    }
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
    fn dist_add(offset: isize, coeff: u64) -> String {
        let t = wrap_ptr_expr(offset);
        format!(
            "tape[{t}] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[{t}], bf_wrap_mul_i64_signed(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[{t}], bf_wrap_mul_i64_unsigned(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS)"
        )
    }
    fn dist_sub(offset: isize, coeff: u64) -> String {
        let t = wrap_ptr_expr(offset);
        format!(
            "tape[{t}] = BF_SIGNED_CELLS ? bf_wrap_sub_i64_signed(tape[{t}], bf_wrap_mul_i64_signed(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_sub_i64_unsigned(tape[{t}], bf_wrap_mul_i64_unsigned(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS)"
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
                    let node = &nodes[index];
                    index += 1;
                    stack.push(EmitFrame::Seq {
                        nodes,
                        index,
                        level,
                    });
                    match node {
                        BfIr::MovePtr(n) if *n != 0 => {
                            push_c_line(out, level, &format!("ptr = {};", wrap_ptr_expr(*n)))
                        }
                        BfIr::MovePtr(_) => {}
                        BfIr::Add(n) if *n > 0 => {
                            push_c_line(out, level, &format!("tape[ptr] = {};", add_expr(*n as u64)))
                        }
                        BfIr::Add(n) if *n < 0 => push_c_line(
                            out,
                            level,
                            &format!("tape[ptr] = {};", sub_expr((-(*n as i64)) as u64)),
                        ),
                        BfIr::Add(_) => {}
                        BfIr::Input => match opts.io_mode {
                            IoMode::Char => push_c_line(out, level, split_char_input_stmt()),
                            IoMode::Number => {
                                push_c_line(out, level, split_number_input_stmt(opts.cell_sign))
                            }
                        },
                        BfIr::Output => {
                            push_c_line(out, level, split_output_stmt(opts));
                            push_c_line(out, level, "fflush(stdout);");
                        }
                        BfIr::Loop(body) => {
                            push_c_line(out, level, "while (tape[ptr] != 0) {");
                            stack.push(EmitFrame::Close { level });
                            stack.push(EmitFrame::Seq {
                                nodes: body,
                                index: 0,
                                level: level + 1,
                            });
                        }
                        BfIr::Clear => push_c_line(out, level, "tape[ptr] = 0;"),
                        BfIr::Distribute { targets } => {
                            push_c_line(out, level, "{");
                            push_c_line(out, level + 1, "int64_t v = tape[ptr];");
                            for &(offset, coeff) in targets {
                                if coeff > 0 {
                                    push_c_line(
                                        out,
                                        level + 1,
                                        &format!("{};", dist_add(offset, coeff as u64)),
                                    );
                                } else if coeff < 0 {
                                    push_c_line(
                                        out,
                                        level + 1,
                                        &format!("{};", dist_sub(offset, (-(coeff as i64)) as u64)),
                                    );
                                }
                            }
                            push_c_line(out, level + 1, "tape[ptr] = 0;");
                            push_c_line(out, level, "}");
                        }
                        BfIr::Diverge => push_c_line(out, level, "bf_diverge_forever();"),
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
            s.find(begin).unwrap()..s.find(begin).unwrap() + begin.len() + 1,
            "",
        );
        let end_pos = s.find(end).unwrap();
        let end_pos_after = end_pos + end.len();
        let end_pos_after = if s.as_bytes().get(end_pos_after) == Some(&b'\n') {
            end_pos_after + 1
        } else {
            end_pos_after
        };
        s.replace_range(end_pos..end_pos_after, "");
    }

    let needs_ptr_wrap = any_ir(program, &|n| match n {
        BfIr::MovePtr(v) => *v != 0,
        BfIr::Distribute { targets } => targets.iter().any(|(off, _)| *off != 0),
        _ => false,
    });
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

    out.replace("/* @BF_CONFIG */", &config)
        .replace("/* @BF_PROGRAM */", &ir_body)
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
        ((v as u64) & mask) as i64
    }

    fn wrap_signed(v: i64, bits: u32) -> i64 {
        if bits == 0 {
            return 0;
        }
        let mask = (1_u64 << bits) - 1;
        let raw = (v as u64) & mask;
        let sign_bit = 1_u64 << (bits - 1);
        if raw & sign_bit == 0 {
            raw as i64
        } else {
            (raw as i128 - (1_i128 << bits)) as i64
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
                ptr = (ptr as i64 + *n as i64).rem_euclid(tape.len() as i64) as usize;
            }
            BfIr::Add(n) => {
                tape[ptr] = wrap(tape[ptr] + *n as i64, cell_bits, opts.cell_sign);
            }
            BfIr::Input | BfIr::Output => {}
            BfIr::Clear => tape[ptr] = 0,
            BfIr::Distribute { targets } => {
                let v = tape[ptr];
                for &(offset, coeff) in targets {
                    let t = (ptr as i64 + offset as i64).rem_euclid(tape.len() as i64) as usize;
                    tape[t] = wrap(tape[t] + v * coeff as i64, cell_bits, opts.cell_sign);
                }
                tape[ptr] = 0;
            }
            BfIr::Diverge => return Err(BfEvalError::DivergenceDetected),
            BfIr::Loop(body) => {
                if tape[ptr] != 0 {
                    stack.last_mut().unwrap().1 -= 1;
                    stack.push((body, 0));
                }
            }
        }
    }
    Ok((tape, ptr))
}
