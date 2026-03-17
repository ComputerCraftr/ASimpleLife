use std::collections::BTreeMap;
use std::env;
use std::io::{self, Read};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BfIr {
    MovePtr(isize),
    Add(i32),
    Input,
    Output,
    Loop(Vec<BfIr>),

    Clear,

    /// Consume the current cell and add signed multiples of its original value
    /// into target cells relative to the current pointer, then clear source.
    ///
    /// Example:
    ///   Distribute { targets: vec![(1, 1), (2, 2)] }
    ///
    /// means:
    ///   cell[ptr + 1] += x
    ///   cell[ptr + 2] += 2*x
    ///   cell[ptr] = 0
    ///
    /// where x is the original value of cell[ptr].
    Distribute {
        targets: Vec<(isize, i32)>,
    },

    Diverge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    DumpIr,
    EmitC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IoMode {
    Char,
    Number,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellSign {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodegenOpts {
    io_mode: IoMode,
    cell_bits: u32,
    input_bits: Option<u32>,
    output_bits: Option<u32>,
    cell_sign: CellSign,
}

const C_TAPE_LEN: usize = 30_000;

fn indent_spaces(n: usize) -> String {
    " ".repeat(n)
}

fn indent_levels(level: usize) -> String {
    indent_spaces(level * 4)
}

pub struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    pub fn new(src: &str) -> Self {
        let chars = src
            .chars()
            .filter(|c| matches!(c, '>' | '<' | '+' | '-' | '.' | ',' | '[' | ']'))
            .collect();
        Self { chars, pos: 0 }
    }

    pub fn parse(mut self) -> Result<Vec<BfIr>, String> {
        let mut stack: Vec<Vec<BfIr>> = vec![Vec::new()];

        while let Some(&ch) = self.chars.get(self.pos) {
            match ch {
                '+' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .expect("parser stack must contain a current block")
                        .push(BfIr::Add(1));
                }
                '-' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .expect("parser stack must contain a current block")
                        .push(BfIr::Add(-1));
                }
                '>' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .expect("parser stack must contain a current block")
                        .push(BfIr::MovePtr(1));
                }
                '<' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .expect("parser stack must contain a current block")
                        .push(BfIr::MovePtr(-1));
                }
                '.' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .expect("parser stack must contain a current block")
                        .push(BfIr::Output);
                }
                ',' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .expect("parser stack must contain a current block")
                        .push(BfIr::Input);
                }
                '[' => {
                    self.pos += 1;
                    stack.push(Vec::new());
                }
                ']' => {
                    if stack.len() == 1 {
                        return Err(format!(
                            "unmatched ']' at filtered token index {}",
                            self.pos
                        ));
                    }
                    self.pos += 1;
                    let body = stack
                        .pop()
                        .expect("parser stack must contain a loop body to close");
                    stack
                        .last_mut()
                        .expect("parser stack must contain a parent block")
                        .push(BfIr::Loop(body));
                }
                _ => unreachable!(),
            }
        }

        if stack.len() != 1 {
            Err("unmatched '['".to_string())
        } else {
            Ok(stack
                .pop()
                .expect("parser stack must contain the top-level block"))
        }
    }
}

fn merge_adjacent(nodes: &mut Vec<BfIr>) {
    let mut merged = Vec::with_capacity(nodes.len());

    for node in nodes.drain(..) {
        match (merged.last_mut(), node) {
            (Some(BfIr::Add(a)), BfIr::Add(b)) => {
                *a += b;
                if *a == 0 {
                    merged.pop();
                }
            }
            (Some(BfIr::MovePtr(a)), BfIr::MovePtr(b)) => {
                *a += b;
                if *a == 0 {
                    merged.pop();
                }
            }
            (_, BfIr::Add(0)) | (_, BfIr::MovePtr(0)) => {}
            (_, n) => merged.push(n),
        }
    }

    *nodes = merged;
}

fn can_peel_one_shot_wrapper(body: &[BfIr]) -> bool {
    let mut ptr = 0isize;
    let mut known_zero = BTreeMap::<isize, ()>::new();

    for op in body {
        match op {
            BfIr::MovePtr(n) => {
                ptr += *n;
            }
            BfIr::Add(_) | BfIr::Input => {
                known_zero.remove(&ptr);
            }
            BfIr::Output => {}
            BfIr::Clear => {
                known_zero.insert(ptr, ());
            }
            BfIr::Distribute { targets } => {
                for (off, coeff) in targets {
                    if *coeff != 0 {
                        known_zero.remove(&(ptr + *off));
                    }
                }
                known_zero.insert(ptr, ());
            }
            BfIr::Diverge => {
                return true;
            }
            BfIr::Loop(_) => {
                return false;
            }
        }
    }

    known_zero.contains_key(&ptr)
}

fn summarize_loop(body: &[BfIr]) -> Option<BfIr> {
    match body {
        [] => Some(BfIr::Diverge),
        [BfIr::Diverge] => Some(BfIr::Diverge),
        [BfIr::Clear] => Some(BfIr::Clear),
        [BfIr::Distribute { targets }] => Some(BfIr::Distribute {
            targets: targets.clone(),
        }),
        [BfIr::Add(delta)] if *delta == -1 || *delta == 1 => Some(BfIr::Clear),
        _ => recognize_affine_loop(body),
    }
}

fn truncate_after_diverge(nodes: &mut Vec<BfIr>) {
    if let Some(pos) = nodes.iter().position(|n| matches!(n, BfIr::Diverge)) {
        nodes.truncate(pos + 1);
    }
}

fn normalize_sequence(nodes: &mut Vec<BfIr>) {
    merge_adjacent(nodes);
    truncate_after_diverge(nodes);
}

enum OptimizeFrame {
    Seq {
        input: std::vec::IntoIter<BfIr>,
        out: Vec<BfIr>,
    },
    LoopFinalize,
}

// Bottom-up optimizer pass: optimize child loop bodies first, peel any one-shot
// loop wrappers until stable, then summarize the normalized loop body if a
// closed-form structured equivalent exists.
fn optimize_sequence(program: Vec<BfIr>) -> Vec<BfIr> {
    let mut stack = vec![OptimizeFrame::Seq {
        input: program.into_iter(),
        out: Vec::new(),
    }];
    let mut completed: Option<Vec<BfIr>> = None;

    while let Some(frame) = stack.last_mut() {
        match frame {
            OptimizeFrame::Seq { input, out } => {
                let Some(node) = input.next() else {
                    let mut result = std::mem::take(out);
                    normalize_sequence(&mut result);
                    stack.pop();
                    completed = Some(result);
                    continue;
                };

                match node {
                    BfIr::Loop(body) => {
                        stack.push(OptimizeFrame::LoopFinalize);
                        stack.push(OptimizeFrame::Seq {
                            input: body.into_iter(),
                            out: Vec::new(),
                        });
                    }
                    BfIr::Add(0) | BfIr::MovePtr(0) => {}
                    other => out.push(other),
                }
            }
            OptimizeFrame::LoopFinalize => {
                let body = completed
                    .take()
                    .expect("loop finalization requires an optimized child body");
                stack.pop();

                let OptimizeFrame::Seq { out, .. } = stack
                    .last_mut()
                    .expect("loop finalization requires a parent sequence frame")
                else {
                    unreachable!("parent frame must be a sequence");
                };

                if let Some(summary) = summarize_loop(&body) {
                    out.push(summary);
                } else if can_peel_one_shot_wrapper(&body) {
                    out.extend(body);
                } else {
                    out.push(BfIr::Loop(body));
                }
            }
        }
    }

    completed.unwrap_or_default()
}

#[derive(Debug)]
struct LoopSummary {
    ptr_end: isize,
    deltas: BTreeMap<isize, i32>,
}

fn summarize_simple_loop(body: &[BfIr]) -> Option<LoopSummary> {
    let mut ptr = 0isize;
    let mut deltas = BTreeMap::<isize, i32>::new();

    for op in body {
        match *op {
            BfIr::Add(n) => {
                *deltas.entry(ptr).or_insert(0) += n;
            }
            BfIr::MovePtr(n) => {
                ptr += n;
            }
            _ => return None,
        }
    }

    // Remove explicit zero entries to keep targets cleaner.
    deltas.retain(|_, v| *v != 0);

    Some(LoopSummary {
        ptr_end: ptr,
        deltas,
    })
}

fn recognize_affine_loop(body: &[BfIr]) -> Option<BfIr> {
    let summary = summarize_simple_loop(body)?;

    // Must return to source cell.
    if summary.ptr_end != 0 {
        return None;
    }

    let src_delta = *summary.deltas.get(&0).unwrap_or(&0);

    // We only summarize loops that monotonically consume the source cell.
    if src_delta >= 0 {
        return None;
    }

    // To preserve exact BF semantics with closed-form replacement, the loop must
    // consume the source cell in a way that reaches zero exactly for all starting
    // values under the current abstract machine model. With integer cells, that is
    // only guaranteed for a unit decrement countdown loop.
    if src_delta != -1 {
        return None;
    }

    let mut targets = Vec::new();
    for (&offset, &coeff) in &summary.deltas {
        if offset == 0 {
            continue;
        }
        if coeff != 0 {
            targets.push((offset, coeff));
        }
    }

    if targets.is_empty() {
        return Some(BfIr::Clear);
    }

    Some(BfIr::Distribute { targets })
}

pub fn format_ir(program: &[BfIr]) -> String {
    enum FormatFrame<'a> {
        Seq {
            nodes: &'a [BfIr],
            index: usize,
            indent: usize,
        },
        CloseLoop {
            indent: usize,
        },
    }

    let mut out = String::new();
    let mut stack = vec![FormatFrame::Seq {
        nodes: program,
        index: 0,
        indent: 0,
    }];

    while let Some(frame) = stack.pop() {
        match frame {
            FormatFrame::Seq {
                nodes,
                mut index,
                indent,
            } => {
                if index >= nodes.len() {
                    continue;
                }

                let pad = indent_spaces(indent);
                let node = &nodes[index];
                index += 1;

                stack.push(FormatFrame::Seq {
                    nodes,
                    index,
                    indent,
                });

                match node {
                    BfIr::MovePtr(n) => {
                        out.push_str(&format!("{pad}MovePtr({n})\n"));
                    }
                    BfIr::Add(n) => {
                        out.push_str(&format!("{pad}Add({n})\n"));
                    }
                    BfIr::Input => {
                        out.push_str(&format!("{pad}Input\n"));
                    }
                    BfIr::Output => {
                        out.push_str(&format!("{pad}Output\n"));
                    }
                    BfIr::Clear => {
                        out.push_str(&format!("{pad}Clear\n"));
                    }
                    BfIr::Distribute { targets } => {
                        out.push_str(&format!("{pad}Distribute {{ targets: {:?} }}\n", targets));
                    }
                    BfIr::Diverge => {
                        out.push_str(&format!("{pad}Diverge\n"));
                    }
                    BfIr::Loop(body) => {
                        out.push_str(&format!("{pad}Loop {{\n"));
                        stack.push(FormatFrame::CloseLoop { indent });
                        stack.push(FormatFrame::Seq {
                            nodes: body,
                            index: 0,
                            indent: indent + 2,
                        });
                    }
                }
            }
            FormatFrame::CloseLoop { indent } => {
                let pad = indent_spaces(indent);
                out.push_str(&format!("{pad}}}\n"));
            }
        }
    }

    out
}

pub fn emit_c(program: &[BfIr], opts: CodegenOpts) -> String {
    fn push_line(out: &mut String, level: usize, line: &str) {
        out.push_str(&indent_levels(level));
        out.push_str(line);
        out.push('\n');
    }

    fn wrap_ptr_expr(offset: isize) -> String {
        format!("bf_wrap_ptr(ptr, (ptrdiff_t)({}), BF_TAPE_LEN)", offset)
    }

    fn add_expr(delta: u64) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[ptr], INT64_C({}), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[ptr], INT64_C({}), BF_CELL_BITS)",
            delta, delta
        )
    }

    fn sub_expr(delta: u64) -> String {
        format!(
            "BF_SIGNED_CELLS ? bf_wrap_sub_i64_signed(tape[ptr], INT64_C({}), BF_CELL_BITS) : bf_wrap_sub_i64_unsigned(tape[ptr], INT64_C({}), BF_CELL_BITS)",
            delta, delta
        )
    }

    fn distribute_add_expr(offset: isize, coeff: u64) -> String {
        let target = wrap_ptr_expr(offset);
        format!(
            "tape[{target}] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[{target}], bf_wrap_mul_i64_signed(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[{target}], bf_wrap_mul_i64_unsigned(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS)"
        )
    }

    fn distribute_sub_expr(offset: isize, coeff: u64) -> String {
        let target = wrap_ptr_expr(offset);
        format!(
            "tape[{target}] = BF_SIGNED_CELLS ? bf_wrap_sub_i64_signed(tape[{target}], bf_wrap_mul_i64_signed(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_sub_i64_unsigned(tape[{target}], bf_wrap_mul_i64_unsigned(v, INT64_C({coeff}), BF_CELL_BITS), BF_CELL_BITS)"
        )
    }

    enum EmitFrame<'a> {
        Seq {
            nodes: &'a [BfIr],
            index: usize,
            level: usize,
        },
        CloseLoop {
            level: usize,
        },
    }

    fn emit_ir_stack(out: &mut String, program: &[BfIr], opts: CodegenOpts) {
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
                        BfIr::MovePtr(n) => {
                            if *n != 0 {
                                push_line(out, level, &format!("ptr = {};", wrap_ptr_expr(*n)));
                            }
                        }
                        BfIr::Add(n) => {
                            if *n > 0 {
                                push_line(
                                    out,
                                    level,
                                    &format!("tape[ptr] = {};", add_expr(*n as u64)),
                                );
                            } else if *n < 0 {
                                let amount = (-(*n as i64)) as u64;
                                push_line(
                                    out,
                                    level,
                                    &format!("tape[ptr] = {};", sub_expr(amount)),
                                );
                            }
                        }
                        BfIr::Input => match opts.io_mode {
                            IoMode::Char => {
                                push_line(
                                    out,
                                    level,
                                    "{ int ch = getchar(); tape[ptr] = (ch == EOF) ? 0 : (BF_SIGNED_CELLS ? bf_wrap_from_u64_signed(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS) : bf_wrap_from_u64_unsigned(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS)); }",
                                );
                            }
                            IoMode::Number => match opts.cell_sign {
                                CellSign::Signed => {
                                    push_line(
                                        out,
                                        level,
                                        "{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }",
                                    );
                                }
                                CellSign::Unsigned => {
                                    push_line(
                                        out,
                                        level,
                                        "{ uint64_t tmp = 0; if (scanf(\"%\" SCNu64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_unsigned(tmp & BF_INPUT_MASK, BF_CELL_BITS); }",
                                    );
                                }
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
                            stack.push(EmitFrame::CloseLoop { level });
                            stack.push(EmitFrame::Seq {
                                nodes: body,
                                index: 0,
                                level: level + 1,
                            });
                        }
                        BfIr::Clear => {
                            push_line(out, level, "tape[ptr] = 0;");
                        }
                        BfIr::Distribute { targets } => {
                            push_line(out, level, "{");
                            push_line(out, level + 1, "int64_t v = tape[ptr];");
                            for (offset, coeff) in targets {
                                if *coeff > 0 {
                                    push_line(
                                        out,
                                        level + 1,
                                        &format!(
                                            "{};",
                                            distribute_add_expr(*offset, *coeff as u64)
                                        ),
                                    );
                                } else if *coeff < 0 {
                                    let amount = (-(*coeff as i64)) as u64;
                                    push_line(
                                        out,
                                        level + 1,
                                        &format!("{};", distribute_sub_expr(*offset, amount)),
                                    );
                                }
                            }
                            push_line(out, level + 1, "tape[ptr] = 0;");
                            push_line(out, level, "}");
                        }
                        BfIr::Diverge => {
                            push_line(out, level, "bf_diverge_forever();");
                        }
                    }
                }
                EmitFrame::CloseLoop { level } => {
                    push_line(out, level, "}");
                }
            }
        }
    }

    // --- Compute usage booleans with a generic walker ---
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

    let needs_ptr_wrap = any_ir(program, &|node| match node {
        BfIr::MovePtr(n) => *n != 0,
        BfIr::Distribute { targets } => targets.iter().any(|(off, _)| *off != 0),
        _ => false,
    });
    let needs_input = any_ir(program, &|node| matches!(node, BfIr::Input));
    let needs_cell_mask = matches!(opts.io_mode, IoMode::Number)
        && any_ir(program, &|node| matches!(node, BfIr::Input | BfIr::Output));

    let mut out = String::new();
    out.push_str(
        "#include <stdio.h>\n\
#include <stdint.h>\n\
#include <stddef.h>\n\
#include <inttypes.h>\n\
#include <stdckdint.h>\n\
\n\
static uint64_t bf_bits_mask(unsigned bits) {\n\
    return bits == 0 ? UINT64_C(0) : ((UINT64_C(1) << bits) - 1);\n\
}\n\
static int64_t bf_wrap_from_u64_signed(uint64_t raw, unsigned bits) {\n\
    if (bits == 0) return 0;\n\
    raw &= bf_bits_mask(bits);\n\
    if (bits == 63) {\n\
        uint64_t sign = UINT64_C(1) << 62;\n\
        if ((raw & sign) == 0) {\n\
            return (int64_t)raw;\n\
        }\n\
        uint64_t mag = raw - sign;\n\
        return (int64_t)mag - (int64_t)sign;\n\
    }\n\
    uint64_t sign = UINT64_C(1) << (bits - 1);\n\
    if ((raw & sign) == 0) {\n\
        return (int64_t)raw;\n\
    }\n\
    return (int64_t)raw - (int64_t)(UINT64_C(1) << bits);\n\
}\n\
static int64_t bf_wrap_from_u64_unsigned(uint64_t raw, unsigned bits) {\n\
    if (bits == 0) return 0;\n\
    return (int64_t)(raw & bf_bits_mask(bits));\n\
}\n\
static uint64_t bf_wrap_to_u64(int64_t value, unsigned bits) {\n\
    if (bits == 0) return UINT64_C(0);\n\
    return ((uint64_t)value) & bf_bits_mask(bits);\n\
}\n\
static int64_t bf_wrap_add_i64_signed(int64_t a, int64_t b, unsigned bits) {\n\
    uint64_t ua = bf_wrap_to_u64(a, bits);\n\
    uint64_t ub = bf_wrap_to_u64(b, bits);\n\
    uint64_t out = 0;\n\
    if (ckd_add(&out, ua, ub)) {\n\
        out = ua + ub;\n\
    }\n\
    return bf_wrap_from_u64_signed(out, bits);\n\
}\n\
static int64_t bf_wrap_sub_i64_signed(int64_t a, int64_t b, unsigned bits) {\n\
    uint64_t ua = bf_wrap_to_u64(a, bits);\n\
    uint64_t ub = bf_wrap_to_u64(b, bits);\n\
    uint64_t out = 0;\n\
    if (ckd_sub(&out, ua, ub)) {\n\
        out = ua - ub;\n\
    }\n\
    return bf_wrap_from_u64_signed(out, bits);\n\
}\n\
static int64_t bf_wrap_mul_i64_signed(int64_t a, int64_t b, unsigned bits) {\n\
    uint64_t ua = bf_wrap_to_u64(a, bits);\n\
    uint64_t ub = bf_wrap_to_u64(b, bits);\n\
    uint64_t out = 0;\n\
    if (ckd_mul(&out, ua, ub)) {\n\
        out = ua * ub;\n\
    }\n\
    return bf_wrap_from_u64_signed(out, bits);\n\
}\n\
static int64_t bf_wrap_add_i64_unsigned(int64_t a, int64_t b, unsigned bits) {\n\
    uint64_t ua = bf_wrap_to_u64(a, bits);\n\
    uint64_t ub = bf_wrap_to_u64(b, bits);\n\
    uint64_t out = 0;\n\
    if (ckd_add(&out, ua, ub)) {\n\
        out = ua + ub;\n\
    }\n\
    return bf_wrap_from_u64_unsigned(out, bits);\n\
}\n\
static int64_t bf_wrap_sub_i64_unsigned(int64_t a, int64_t b, unsigned bits) {\n\
    uint64_t ua = bf_wrap_to_u64(a, bits);\n\
    uint64_t ub = bf_wrap_to_u64(b, bits);\n\
    uint64_t out = 0;\n\
    if (ckd_sub(&out, ua, ub)) {\n\
        out = ua - ub;\n\
    }\n\
    return bf_wrap_from_u64_unsigned(out, bits);\n\
}\n\
static int64_t bf_wrap_mul_i64_unsigned(int64_t a, int64_t b, unsigned bits) {\n\
    uint64_t ua = bf_wrap_to_u64(a, bits);\n\
    uint64_t ub = bf_wrap_to_u64(b, bits);\n\
    uint64_t out = 0;\n\
    if (ckd_mul(&out, ua, ub)) {\n\
        out = ua * ub;\n\
    }\n\
    return bf_wrap_from_u64_unsigned(out, bits);\n\
}\n\
static _Noreturn void bf_diverge_forever(void) {\n\
    static volatile int bf_diverge_sink = 0;\n\
    for (;;) {\n\
        bf_diverge_sink = 1;\n\
    }\n\
}\n\
",
    );

    if needs_ptr_wrap {
        out.push_str(
            "static ptrdiff_t bf_wrap_ptr(ptrdiff_t ptr, ptrdiff_t delta, ptrdiff_t len) {\n\
    if (len <= 0) return 0;\n\
    ptr %= len;\n\
    if (ptr < 0) ptr += len;\n\
    delta %= len;\n\
    if (delta < 0) delta += len;\n\
    return (ptr + delta) % len;\n\
}\n\
",
        );
    }
    out.push_str("int main(void) {\n");

    let cell_ty = "int64_t";

    let input_bits = opts.input_bits.unwrap_or(opts.cell_bits).min(63);
    let output_bits = opts.output_bits.unwrap_or(opts.cell_bits).min(63);

    let input_mask = match input_bits {
        0 => "UINT64_C(0)",
        n => {
            let mask = (1u64 << n) - 1;
            Box::leak(format!("UINT64_C({})", mask).into_boxed_str())
        }
    };

    let output_mask = match output_bits {
        0 => "UINT64_C(0)",
        n => {
            let mask = (1u64 << n) - 1;
            Box::leak(format!("UINT64_C({})", mask).into_boxed_str())
        }
    };

    let cell_mask = match opts.cell_bits {
        0 => "UINT64_C(0)",
        n => {
            let mask = (1u64 << n) - 1;
            Box::leak(format!("UINT64_C({})", mask).into_boxed_str())
        }
    };

    push_line(&mut out, 1, "enum {");
    push_line(&mut out, 2, &format!("BF_TAPE_LEN = {},", C_TAPE_LEN));
    push_line(&mut out, 2, &format!("BF_CELL_BITS = {},", opts.cell_bits));
    push_line(
        &mut out,
        2,
        &format!(
            "BF_SIGNED_CELLS = {},",
            if matches!(opts.cell_sign, CellSign::Signed) {
                1
            } else {
                0
            }
        ),
    );
    if needs_cell_mask {
        push_line(&mut out, 2, &format!("BF_CELL_MASK = {},", cell_mask));
    }
    if needs_input {
        push_line(&mut out, 2, &format!("BF_INPUT_MASK = {},", input_mask));
    }
    push_line(&mut out, 2, &format!("BF_OUTPUT_MASK = {}", output_mask));
    push_line(&mut out, 1, "};");
    push_line(
        &mut out,
        1,
        &format!("{} tape[BF_TAPE_LEN] = {{0}};", cell_ty),
    );
    push_line(&mut out, 1, "ptrdiff_t ptr = 0;");
    out.push('\n');
    emit_ir_stack(&mut out, program, opts);
    out.push_str("\n    return 0;\n");
    out.push_str("}\n");
    out
}

fn parse_codegen_opts(args: &[String]) -> Result<(CodegenOpts, &[String]), String> {
    let mut opts = CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    };
    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "--io" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --io".to_string())?;
                opts.io_mode = match value.as_str() {
                    "char" => IoMode::Char,
                    "number" => IoMode::Number,
                    other => {
                        return Err(format!(
                            "unsupported --io value '{}'; expected 'char' or 'number'",
                            other
                        ));
                    }
                };
                i += 2;
            }
            "--cell-bits" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --cell-bits".to_string())?;
                let bits: u32 = value
                    .parse()
                    .map_err(|_| format!("invalid --cell-bits value '{}'", value))?;
                match bits {
                    0..=63 => opts.cell_bits = bits,
                    _ => {
                        return Err(format!(
                            "unsupported --cell-bits value '{}'; expected 0..=63",
                            bits
                        ));
                    }
                }
                i += 2;
            }
            "--input-bits" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --input-bits".to_string())?;
                let bits: u32 = value
                    .parse()
                    .map_err(|_| format!("invalid --input-bits value '{}'", value))?;
                match bits {
                    0..=63 => opts.input_bits = Some(bits),
                    _ => {
                        return Err(format!(
                            "unsupported --input-bits value '{}'; expected 0..=63",
                            bits
                        ));
                    }
                }
                i += 2;
            }
            "--output-bits" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --output-bits".to_string())?;
                let bits: u32 = value
                    .parse()
                    .map_err(|_| format!("invalid --output-bits value '{}'", value))?;
                match bits {
                    0..=63 => opts.output_bits = Some(bits),
                    _ => {
                        return Err(format!(
                            "unsupported --output-bits value '{}'; expected 0..=63",
                            bits
                        ));
                    }
                }
                i += 2;
            }
            "--signed-cells" => {
                opts.cell_sign = CellSign::Signed;
                i += 1;
            }
            "--unsigned-cells" => {
                opts.cell_sign = CellSign::Unsigned;
                i += 1;
            }
            _ => break,
        }
    }

    Ok((opts, &args[i..]))
}

fn read_input() -> Result<(OutputMode, CodegenOpts, String), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut mode = OutputMode::DumpIr;
    let mut rest: &[String] = &args;

    if let Some(first) = rest.first() {
        match first.as_str() {
            "--emit-c" => {
                mode = OutputMode::EmitC;
                rest = &rest[1..];
            }
            "--dump-ir" => {
                mode = OutputMode::DumpIr;
                rest = &rest[1..];
            }
            _ => {}
        }
    }

    let (codegen_opts, remaining) = parse_codegen_opts(rest)?;
    rest = remaining;

    if rest.is_empty() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {}", e))?;
        return Ok((mode, codegen_opts, buf));
    }

    if rest[0] == "--" {
        return Ok((mode, codegen_opts, rest[1..].join(" ")));
    }

    if rest.len() == 1 {
        let arg = &rest[0];
        if std::path::Path::new(arg).exists() {
            let src = std::fs::read_to_string(arg)
                .map_err(|e| format!("failed to read '{}': {}", arg, e))?;
            return Ok((mode, codegen_opts, src));
        }
        return Ok((mode, codegen_opts, arg.clone()));
    }

    Ok((mode, codegen_opts, rest.join(" ")))
}

pub fn main() {
    let (mode, codegen_opts, src) = match read_input() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let parsed = match Parser::new(&src).parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("parse error: {e}");
            std::process::exit(1);
        }
    };

    match mode {
        OutputMode::DumpIr => {
            let optimized = optimize_sequence(parsed.clone());

            println!("=== Parsed IR ===");
            print!("{}", format_ir(&parsed));

            println!("=== Optimized IR ===");
            print!("{}", format_ir(&optimized));
        }
        OutputMode::EmitC => {
            let optimized = optimize_sequence(parsed);
            print!("{}", emit_c(&optimized, codegen_opts));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_opt(src: &str) -> Vec<BfIr> {
        let parsed = Parser::new(src).parse().unwrap();
        optimize_sequence(parsed)
    }

    fn parse_only(src: &str) -> Vec<BfIr> {
        Parser::new(src).parse().unwrap()
    }

    fn parse_and_format(src: &str) -> String {
        format_ir(&parse_only(src))
    }

    fn parse_opt_and_emit_c(src: &str) -> String {
        let optimized = parse_and_opt(src);
        emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Char,
                cell_bits: 8,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Signed,
            },
        )
    }

    #[test]
    fn parser_accepts_inline_source() {
        let ir = parse_only("+++");
        assert_eq!(ir, vec![BfIr::Add(1), BfIr::Add(1), BfIr::Add(1)]);
    }

    #[test]
    fn parser_ignores_non_bf_cli_spacing() {
        let ir = parse_only("+ + +");
        assert_eq!(ir, vec![BfIr::Add(1), BfIr::Add(1), BfIr::Add(1)]);
    }

    #[test]
    fn format_inline_source_ir() {
        let formatted = parse_and_format("+++[->+<]");
        assert_eq!(
            formatted,
            "Add(1)\nAdd(1)\nAdd(1)\nLoop {\n  Add(-1)\n  MovePtr(1)\n  Add(1)\n  MovePtr(-1)\n}\n"
        );
    }

    #[test]
    fn optimizer_combines_parsed_runs() {
        let ir = parse_and_opt("+++");
        assert_eq!(ir, vec![BfIr::Add(3)]);
    }

    #[test]
    fn emit_c_for_simple_add_and_output() {
        let c = parse_opt_and_emit_c("+++.");
        assert!(c.contains("#include <stdckdint.h>"));
        assert!(c.contains("ckd_add(&out, ua, ub)"));
        assert!(c.contains("ckd_sub(&out, ua, ub)"));
        assert!(c.contains("ckd_mul(&out, ua, ub)"));
        assert!(c.contains("tape[ptr] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[ptr], INT64_C(3), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[ptr], INT64_C(3), BF_CELL_BITS);"));
        assert!(c.contains("putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"));
        assert!(c.contains("fflush(stdout);"));
        assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {0};"));
        assert!(c.contains("BF_TAPE_LEN = 30000"));
        assert!(c.contains("BF_CELL_BITS = 8"));
        assert!(c.contains("BF_SIGNED_CELLS = 1"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(255)"));
        assert!(!c.contains("BF_INPUT_MASK = "));
        assert!(!c.contains("BF_CELL_MASK = "));
        assert!(!c.contains("static ptrdiff_t bf_wrap_ptr("));
    }

    #[test]
    fn emit_c_char_only_program_omits_numeric_format_constants() {
        let c = parse_opt_and_emit_c("+++.");
        assert!(!c.contains("BF_SCANF_FMT"));
        assert!(!c.contains("BF_PRINTF_FMT"));
    }

    #[test]
    fn emit_c_wraps_tape_pointer_moves() {
        let optimized = parse_and_opt(">>");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Char,
                cell_bits: 8,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Signed,
            },
        );
        assert!(c.contains(
            "static ptrdiff_t bf_wrap_ptr(ptrdiff_t ptr, ptrdiff_t delta, ptrdiff_t len) {"
        ));
        assert!(c.contains("BF_TAPE_LEN = 30000"));
        assert!(c.contains("ptr = bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN);"));
    }

    #[test]
    fn emit_c_for_distribute_loop() {
        let c = parse_opt_and_emit_c("[->+>++<<]");
        assert!(c.contains("int64_t v = tape[ptr];"));
        assert!(c.contains("tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)], bf_wrap_mul_i64_signed(v, INT64_C(1), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(1), BF_TAPE_LEN)], bf_wrap_mul_i64_unsigned(v, INT64_C(1), BF_CELL_BITS), BF_CELL_BITS);"));
        assert!(c.contains("tape[bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN)] = BF_SIGNED_CELLS ? bf_wrap_add_i64_signed(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN)], bf_wrap_mul_i64_signed(v, INT64_C(2), BF_CELL_BITS), BF_CELL_BITS) : bf_wrap_add_i64_unsigned(tape[bf_wrap_ptr(ptr, (ptrdiff_t)(2), BF_TAPE_LEN)], bf_wrap_mul_i64_unsigned(v, INT64_C(2), BF_CELL_BITS), BF_CELL_BITS);"));
        assert!(c.contains("tape[ptr] = 0;"));
    }

    #[test]
    fn merges_adds_and_moves() {
        let ir = parse_and_opt("++++-->>><<");
        assert_eq!(ir, vec![BfIr::Add(2), BfIr::MovePtr(1)]);
    }

    #[test]
    fn clear_minus_loop() {
        let ir = parse_and_opt("[-]");
        assert_eq!(ir, vec![BfIr::Clear]);
    }

    #[test]
    fn clear_plus_loop() {
        let ir = parse_and_opt("[+]");
        assert_eq!(ir, vec![BfIr::Clear]);
    }

    #[test]
    fn distribute_single_target_add() {
        let ir = parse_and_opt("[->+<]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn distribute_single_target_mul() {
        let ir = parse_and_opt("[->+++<]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, 3)]
            }]
        );
    }

    #[test]
    fn distribute_single_target_sub() {
        let ir = parse_and_opt("[->-<]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, -1)]
            }]
        );
    }

    #[test]
    fn distribute_multiple_targets() {
        let ir = parse_and_opt("[->+>++<<]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, 1), (2, 2)]
            }]
        );
    }

    #[test]
    fn distribute_to_left() {
        let ir = parse_and_opt("[<+>-]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(-1, 1)]
            }]
        );
    }

    #[test]
    fn affine_loop_with_extra_balanced_motion_still_becomes_distribute() {
        let ir = parse_and_opt("[->+>+<+<]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, 2), (2, 1)]
            }]
        );
    }

    #[test]
    fn affine_loop_with_net_nonzero_pointer_is_not_summarized() {
        let ir = parse_and_opt("[->+>+<]");
        assert_eq!(
            ir,
            vec![BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ])]
        );
    }

    #[test]
    fn affine_loop_with_non_unit_source_delta_is_not_summarized() {
        let ir = parse_and_opt("[--->+<]");
        assert_eq!(
            ir,
            vec![BfIr::Loop(vec![
                BfIr::Add(-3),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ])]
        );
    }

    #[test]
    fn keeps_general_loops() {
        let ir = parse_and_opt("[->+<+]");
        assert_eq!(
            ir,
            vec![BfIr::Loop(vec![
                BfIr::Add(-1),
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
                BfIr::Add(1),
            ])]
        );
    }

    #[test]
    fn nested_optimization() {
        let ir = parse_and_opt("[[-]]");
        assert_eq!(ir, vec![BfIr::Clear]);
    }

    #[test]
    fn unmatched_right_bracket_errors() {
        let err = Parser::new("]").parse().unwrap_err();
        assert!(err.contains("unmatched ']'"));
    }

    #[test]
    fn unmatched_left_bracket_errors() {
        let err = Parser::new("[").parse().unwrap_err();
        assert!(err.contains("unmatched '['"));
    }

    #[test]
    fn emit_c_numeric_io_mode() {
        let optimized = parse_and_opt(",.");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Number,
                cell_bits: 32,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Signed,
            },
        );
        assert!(c.contains("{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"));
        assert!(c.contains("printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"));
        assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {0};"));
        assert!(c.contains("BF_CELL_BITS = 32"));
        assert!(c.contains("BF_CELL_MASK = UINT64_C(4294967295)"));
        assert!(c.contains("BF_INPUT_MASK = UINT64_C(4294967295)"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(4294967295)"));
        assert!(!c.contains("static ptrdiff_t bf_wrap_ptr("));
    }

    #[test]
    fn emit_c_uses_unsigned_wrapping_when_requested() {
        let optimized = parse_and_opt("++[->+<]");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Char,
                cell_bits: 8,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Unsigned,
            },
        );
        assert!(c.contains("BF_SIGNED_CELLS = 0"));
        assert!(c.contains("bf_wrap_add_i64_unsigned"));
        assert!(c.contains("bf_wrap_sub_i64_unsigned"));
        assert!(c.contains("bf_wrap_mul_i64_unsigned"));
        assert!(c.contains("bf_wrap_from_u64_unsigned"));
    }

    #[test]
    fn emit_c_uses_requested_cell_width() {
        let optimized = parse_and_opt("+");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Char,
                cell_bits: 63,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Signed,
            },
        );
        assert!(c.contains("int64_t tape[BF_TAPE_LEN] = {0};"));
        assert!(c.contains("BF_CELL_BITS = 63"));
        assert!(!c.contains("BF_CELL_MASK = "));
        assert!(c.contains("ptrdiff_t ptr = 0;"));
    }

    #[test]
    fn emit_c_applies_custom_char_masks() {
        let optimized = parse_and_opt(",.");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Char,
                cell_bits: 16,
                input_bits: Some(5),
                output_bits: Some(6),
                cell_sign: CellSign::Signed,
            },
        );
        assert!(c.contains("BF_INPUT_MASK = UINT64_C(31)"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(63)"));
        assert!(
            c.contains("BF_SIGNED_CELLS ? bf_wrap_from_u64_signed(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS) : bf_wrap_from_u64_unsigned(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS)")
        );
        assert!(c.contains("putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"));
    }

    #[test]
    fn emit_c_applies_custom_number_masks() {
        let optimized = parse_and_opt(",.");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Number,
                cell_bits: 32,
                input_bits: Some(3),
                output_bits: Some(4),
                cell_sign: CellSign::Signed,
            },
        );
        assert!(c.contains("BF_INPUT_MASK = UINT64_C(7)"));
        assert!(c.contains("BF_OUTPUT_MASK = UINT64_C(15)"));
        assert!(c.contains("{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"));
        assert!(
            c.contains("bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS)")
        );
        assert!(c.contains("printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"));
    }

    #[test]
    fn parse_codegen_opts_rejects_64_bit_io_masks() {
        let args = vec!["--input-bits".to_string(), "64".to_string()];
        let err = parse_codegen_opts(&args).unwrap_err();
        assert!(err.contains("expected 0..=63"));

        let args = vec!["--output-bits".to_string(), "64".to_string()];
        let err = parse_codegen_opts(&args).unwrap_err();
        assert!(err.contains("expected 0..=63"));
    }

    #[test]
    fn emit_c_uses_unsigned_numeric_io_when_requested() {
        let optimized = parse_and_opt(",.");
        let c = emit_c(
            &optimized,
            CodegenOpts {
                io_mode: IoMode::Number,
                cell_bits: 8,
                input_bits: None,
                output_bits: None,
                cell_sign: CellSign::Unsigned,
            },
        );
        assert!(c.contains("BF_SIGNED_CELLS = 0"));
        assert!(c.contains("{ uint64_t tmp = 0; if (scanf(\"%\" SCNu64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_unsigned(tmp & BF_INPUT_MASK, BF_CELL_BITS); }"));
        assert!(c.contains(
            "printf(\"%\" PRIu64 \"\\n\", (uint64_t)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"
        ));
    }

    #[test]
    fn emit_c_for_diverge() {
        let c = parse_opt_and_emit_c("[]");
        assert!(c.contains("static _Noreturn void bf_diverge_forever(void) {"));
        assert!(c.contains("volatile int bf_diverge_sink"));
        assert!(c.contains("bf_diverge_forever();"));
    }

    #[test]
    fn empty_loop_becomes_diverge() {
        let ir = parse_and_opt("[]");
        assert_eq!(ir, vec![BfIr::Diverge]);
    }

    #[test]
    fn nested_empty_loops_become_diverge_at_all_tested_depths() {
        for src in ["[[]]", "[[[]]]", "[[[[]]]]"] {
            let ir = parse_and_opt(src);
            assert_eq!(ir, vec![BfIr::Diverge], "failed for {src}");
        }
    }

    #[test]
    fn outer_loop_over_distribute_becomes_distribute() {
        let ir = parse_and_opt("[[->+<]]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }

    #[test]
    fn dead_suffix_after_diverge_is_removed() {
        let ir = parse_and_opt("[]+++");
        assert_eq!(ir, vec![BfIr::Diverge]);
    }

    #[test]
    fn one_shot_wrapper_with_final_clear_is_flattened() {
        let ir = parse_and_opt("[>[-]]");
        assert_eq!(ir, vec![BfIr::MovePtr(1), BfIr::Clear]);
    }

    #[test]
    fn one_shot_wrapper_with_prefix_and_final_clear_is_flattened() {
        let ir = parse_and_opt("[>+<[-]]");
        assert_eq!(
            ir,
            vec![
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
                BfIr::Clear,
            ]
        );
    }

    #[test]
    fn one_shot_wrapper_with_suffix_after_clear_is_flattened() {
        let ir = parse_and_opt("[[-]>+<]");
        assert_eq!(
            ir,
            vec![
                BfIr::Clear,
                BfIr::MovePtr(1),
                BfIr::Add(1),
                BfIr::MovePtr(-1),
            ]
        );
    }

    #[test]
    fn one_shot_wrapper_with_diverging_body_is_flattened() {
        let ir = parse_and_opt("[+[]]");
        assert_eq!(ir, vec![BfIr::Add(1), BfIr::Diverge]);
    }

    #[test]
    fn repeated_wrapper_canonicalization_reaches_clear_fixed_point() {
        let ir = parse_and_opt("[[[-]]]");
        assert_eq!(ir, vec![BfIr::Clear]);
    }

    #[test]
    fn repeated_wrapper_canonicalization_reaches_distribute_fixed_point() {
        let ir = parse_and_opt("[[[->+<]]]");
        assert_eq!(
            ir,
            vec![BfIr::Distribute {
                targets: vec![(1, 1)]
            }]
        );
    }
}
