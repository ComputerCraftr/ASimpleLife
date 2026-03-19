use crate::bitgrid::BitGrid;
use crate::hashlife;
use crate::persistence;
use std::error::Error;
use std::fmt;

use super::ir::BfIr;
use super::optimizer::{CellSign, CodegenOpts, IoMode};

pub(super) const C_TAPE_LEN: usize = 30_000;
const BF_LIFE_STEP_BUDGET: u64 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BfLifeEmitError {
    SignedCellsUnsupported,
    DivergenceDetected,
    StepBudgetExceeded,
}

impl fmt::Display for BfLifeEmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::SignedCellsUnsupported => {
                "Life emitters require --unsigned-cells; signed-cell export is unsupported"
            }
            Self::DivergenceDetected => {
                "Life emitters rejected a diverging program instead of emitting unsound output"
            }
            Self::StepBudgetExceeded => {
                "Life emitters rejected a program that exceeded the execution step budget"
            }
        };
        f.write_str(message)
    }
}

impl Error for BfLifeEmitError {}

fn indent(n: usize) -> String {
    " ".repeat(n)
}
fn indent4(level: usize) -> String {
    indent(level * 4)
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
    fn push_line(out: &mut String, level: usize, line: &str) {
        out.push_str(&indent4(level));
        out.push_str(line);
        out.push('\n');
    }

    fn wrap_ptr(offset: isize) -> String {
        format!("bf_wrap_ptr(ptr, (ptrdiff_t)({}), BF_TAPE_LEN)", offset)
    }

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
        let t = wrap_ptr(offset);
        format!(
            "tape[{t}] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[{t}], bf_wrap_mul_i64_signed(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[{t}], bf_wrap_mul_i64_unsigned(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS)"
        )
    }
    fn dist_sub(offset: isize, coeff: u64) -> String {
        let t = wrap_ptr(offset);
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
                            push_line(out, level, &format!("ptr = {};", wrap_ptr(*n)))
                        }
                        BfIr::MovePtr(_) => {}
                        BfIr::Add(n) if *n > 0 => {
                            push_line(out, level, &format!("tape[ptr] = {};", add_expr(*n as u64)))
                        }
                        BfIr::Add(n) if *n < 0 => push_line(
                            out,
                            level,
                            &format!("tape[ptr] = {};", sub_expr((-(*n as i64)) as u64)),
                        ),
                        BfIr::Add(_) => {}
                        BfIr::Input => match opts.io_mode {
                            IoMode::Char => push_line(
                                out,
                                level,
                                "{ int ch = getchar(); tape[ptr] = (ch == EOF) ? 0 : (BF_SIGNED_CELLS ? bf_wrap_from_u64_signed(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS) : bf_wrap_from_u64_unsigned(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS)); }",
                            ),
                            IoMode::Number => match opts.cell_sign {
                                CellSign::Signed => push_line(
                                    out,
                                    level,
                                    "{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }",
                                ),
                                CellSign::Unsigned => push_line(
                                    out,
                                    level,
                                    "{ uint64_t tmp = 0; if (scanf(\"%\" SCNu64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_unsigned(tmp & BF_INPUT_MASK, BF_CELL_BITS); }",
                                ),
                            },
                        },
                        BfIr::Output => match opts.io_mode {
                            IoMode::Char => {
                                push_line(
                                    out,
                                    level,
                                    "putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));",
                                );
                                push_line(out, level, "fflush(stdout);");
                            }
                            IoMode::Number => {
                                let line = match opts.cell_sign {
                                    CellSign::Signed => {
                                        "printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"
                                    }
                                    CellSign::Unsigned => {
                                        "printf(\"%\" PRIu64 \"\\n\", (uint64_t)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"
                                    }
                                };
                                push_line(out, level, line);
                                push_line(out, level, "fflush(stdout);");
                            }
                        },
                        BfIr::Loop(body) => {
                            push_line(out, level, "while (tape[ptr] != 0) {");
                            stack.push(EmitFrame::Close { level });
                            stack.push(EmitFrame::Seq {
                                nodes: body,
                                index: 0,
                                level: level + 1,
                            });
                        }
                        BfIr::Clear => push_line(out, level, "tape[ptr] = 0;"),
                        BfIr::Distribute { targets } => {
                            push_line(out, level, "{");
                            push_line(out, level + 1, "int64_t v = tape[ptr];");
                            for &(offset, coeff) in targets {
                                if coeff > 0 {
                                    push_line(
                                        out,
                                        level + 1,
                                        &format!("{};", dist_add(offset, coeff as u64)),
                                    );
                                } else if coeff < 0 {
                                    push_line(
                                        out,
                                        level + 1,
                                        &format!("{};", dist_sub(offset, (-(coeff as i64)) as u64)),
                                    );
                                }
                            }
                            push_line(out, level + 1, "tape[ptr] = 0;");
                            push_line(out, level, "}");
                        }
                        BfIr::Diverge => push_line(out, level, "bf_diverge_forever();"),
                    }
                }
                EmitFrame::Close { level } => push_line(out, level, "}"),
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

    fn mask_str(bits: u32) -> String {
        if bits == 0 {
            "UINT64_C(0)".to_owned()
        } else {
            format!("UINT64_C({})", (1u64 << bits) - 1)
        }
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

    out.replace("/* @BF_TAPE_LEN */", &C_TAPE_LEN.to_string())
        .replace("/* @BF_CELL_BITS */", &opts.cell_bits.to_string())
        .replace(
            "/* @BF_SIGNED_CELLS */",
            if matches!(opts.cell_sign, CellSign::Signed) {
                "1"
            } else {
                "0"
            },
        )
        .replace("/* @BF_CELL_MASK */", &mask_str(opts.cell_bits))
        .replace("/* @BF_INPUT_MASK */", &mask_str(input_bits))
        .replace("/* @BF_OUTPUT_MASK */", &mask_str(output_bits))
        .replace("/* @BF_PROGRAM */", &ir_body)
}

/// Interpret BF IR under the unsigned modular tape model used for Life export.
///
/// Each tape cell `i` with value `v` maps to `v` live cells at `y=0`,
/// `x = i * stride .. i * stride + v - 1`, where `stride = cell_max + 2`.
pub fn compile_to_life_grid(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<BitGrid, BfLifeEmitError> {
    const MAX_TAPE: usize = 30_000;
    if opts.cell_sign != CellSign::Unsigned {
        return Err(BfLifeEmitError::SignedCellsUnsupported);
    }
    let cell_bits = opts.cell_bits.min(63);
    let cell_max: i64 = if cell_bits == 0 {
        0
    } else {
        (1_i64 << cell_bits) - 1
    };
    let stride = cell_max + 2;

    let mut tape = vec![0_i64; MAX_TAPE];
    let mut ptr = 0_usize;
    interpret_unsigned_for_life_export(program, &mut tape, &mut ptr, cell_max)?;

    let mut cells = Vec::new();
    for (i, &v) in tape.iter().enumerate() {
        let count = v.max(0).min(cell_max);
        let base_x = i as i64 * stride;
        for dx in 0..count {
            cells.push((base_x + dx, 0_i64));
        }
    }
    Ok(BitGrid::from_cells(&cells))
}

/// Iteratively interpret BF IR for unsigned Life export only.
/// This is not a general BF semantics oracle and must fail closed on
/// unsupported or indeterminate execution rather than emitting approximations.
fn interpret_unsigned_for_life_export(
    program: &[BfIr],
    tape: &mut Vec<i64>,
    ptr: &mut usize,
    cell_mask: i64,
) -> Result<(), BfLifeEmitError> {
    fn wrap(v: i64, mask: i64) -> i64 {
        v & mask
    }

    let mut stack: Vec<(&[BfIr], usize)> = vec![(program, 0)];
    let mut steps = 0_u64;

    while let Some((nodes, index)) = stack.last_mut() {
        if *index >= nodes.len() {
            stack.pop();
            continue;
        }
        if steps >= BF_LIFE_STEP_BUDGET {
            return Err(BfLifeEmitError::StepBudgetExceeded);
        }
        steps += 1;

        let node = &nodes[*index];
        *index += 1;

        match node {
            BfIr::MovePtr(n) => {
                *ptr = (*ptr as i64 + *n as i64).rem_euclid(tape.len() as i64) as usize;
            }
            BfIr::Add(n) => {
                tape[*ptr] = wrap(tape[*ptr] + *n as i64, cell_mask);
            }
            BfIr::Input | BfIr::Output => {}
            BfIr::Clear => {
                tape[*ptr] = 0;
            }
            BfIr::Distribute { targets } => {
                let v = tape[*ptr];
                for &(offset, coeff) in targets {
                    let t = (*ptr as i64 + offset as i64).rem_euclid(tape.len() as i64) as usize;
                    tape[t] = wrap(tape[t] + v * coeff as i64, cell_mask);
                }
                tape[*ptr] = 0;
            }
            BfIr::Diverge => return Err(BfLifeEmitError::DivergenceDetected),
            BfIr::Loop(body) => {
                if tape[*ptr] != 0 {
                    stack.last_mut().unwrap().1 -= 1;
                    stack.push((body, 0));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn interpret_unsigned_for_tests(
    program: &[BfIr],
    cell_bits: u32,
) -> Result<(Vec<i64>, usize), BfLifeEmitError> {
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
) -> Result<(Vec<i64>, usize), BfLifeEmitError> {
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
        if steps >= BF_LIFE_STEP_BUDGET {
            return Err(BfLifeEmitError::StepBudgetExceeded);
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
            BfIr::Diverge => return Err(BfLifeEmitError::DivergenceDetected),
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

pub fn serialize_legacy_life_grid(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<String, BfLifeEmitError> {
    Ok(persistence::serialize_life_grid(&compile_to_life_grid(
        program, opts,
    )?))
}

pub fn serialize_life_grid(program: &[BfIr], opts: CodegenOpts) -> Result<String, BfLifeEmitError> {
    Ok(hashlife::serialize_grid_snapshot(&compile_to_life_grid(
        program, opts,
    )?))
}
