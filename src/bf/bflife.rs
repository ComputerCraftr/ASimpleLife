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
        self.parse_block(false)
    }

    fn parse_block(&mut self, in_loop: bool) -> Result<Vec<BfIr>, String> {
        let mut out = Vec::new();

        while let Some(&ch) = self.chars.get(self.pos) {
            match ch {
                '+' => {
                    self.pos += 1;
                    out.push(BfIr::Add(1));
                }
                '-' => {
                    self.pos += 1;
                    out.push(BfIr::Add(-1));
                }
                '>' => {
                    self.pos += 1;
                    out.push(BfIr::MovePtr(1));
                }
                '<' => {
                    self.pos += 1;
                    out.push(BfIr::MovePtr(-1));
                }
                '.' => {
                    self.pos += 1;
                    out.push(BfIr::Output);
                }
                ',' => {
                    self.pos += 1;
                    out.push(BfIr::Input);
                }
                '[' => {
                    self.pos += 1; // consume '['
                    let body = self.parse_block(true)?;
                    out.push(BfIr::Loop(body));
                }
                ']' => {
                    if !in_loop {
                        return Err(format!(
                            "unmatched ']' at filtered token index {}",
                            self.pos
                        ));
                    }
                    self.pos += 1; // consume ']'
                    return Ok(out);
                }
                _ => unreachable!(),
            }
        }

        if in_loop {
            Err("unmatched '['".to_string())
        } else {
            Ok(out)
        }
    }
}

pub fn optimize(program: Vec<BfIr>) -> Vec<BfIr> {
    let mut out = Vec::new();

    for node in program {
        match node {
            BfIr::Loop(body) => {
                let body = optimize(body);

                if body.is_empty() {
                    // [] is an infinite loop if entered; keep it.
                    out.push(BfIr::Loop(body));
                    continue;
                }

                if let Some(replacement) = recognize_loop(&body) {
                    out.push(replacement);
                } else {
                    out.push(BfIr::Loop(body));
                }
            }
            BfIr::Add(0) | BfIr::MovePtr(0) => {}
            other => out.push(other),
        }
    }

    merge_adjacent(&mut out);
    out
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

fn recognize_loop(body: &[BfIr]) -> Option<BfIr> {
    // Common clear idioms for wrapping cell semantics.
    match body {
        [BfIr::Add(delta)] if *delta == -1 || *delta == 1 => {
            return Some(BfIr::Clear);
        }
        _ => {}
    }

    recognize_distribute(body)
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

fn recognize_distribute(body: &[BfIr]) -> Option<BfIr> {
    let summary = summarize_simple_loop(body)?;

    // Must return to source cell.
    if summary.ptr_end != 0 {
        return None;
    }

    // Must consume exactly one unit of the source per iteration.
    let src_delta = *summary.deltas.get(&0).unwrap_or(&0);
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

    // If there are no targets, that's just a clear loop.
    if targets.is_empty() {
        return Some(BfIr::Clear);
    }

    Some(BfIr::Distribute { targets })
}

pub fn format_ir(program: &[BfIr]) -> String {
    fn fmt_block(nodes: &[BfIr], indent: usize, out: &mut String) {
        let pad = " ".repeat(indent);
        for node in nodes {
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
                BfIr::Loop(body) => {
                    out.push_str(&format!("{pad}Loop {{\n"));
                    fmt_block(body, indent + 2, out);
                    out.push_str(&format!("{pad}}}\n"));
                }
            }
        }
    }

    let mut s = String::new();
    fmt_block(program, 0, &mut s);
    s
}

pub fn emit_c(program: &[BfIr], opts: CodegenOpts) -> String {
    fn indent(level: usize) -> String {
        "    ".repeat(level)
    }

    fn push_line(out: &mut String, level: usize, line: &str) {
        out.push_str(&indent(level));
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

    fn emit_block(nodes: &[BfIr], level: usize, out: &mut String, opts: CodegenOpts) {
        for node in nodes {
            match node {
                BfIr::MovePtr(n) => {
                    if *n != 0 {
                        push_line(out, level, &format!("ptr = {};", wrap_ptr_expr(*n)));
                    }
                }
                BfIr::Add(n) => {
                    if *n > 0 {
                        push_line(out, level, &format!("tape[ptr] = {};", add_expr(*n as u64)));
                    } else if *n < 0 {
                        let amount = (-(*n as i64)) as u64;
                        push_line(out, level, &format!("tape[ptr] = {};", sub_expr(amount)));
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
                    emit_block(body, level + 1, out, opts);
                    push_line(out, level, "}");
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
                                &format!("{};", distribute_add_expr(*offset, *coeff as u64)),
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
            }
        }
    }

    // --- Compute usage booleans directly ---
    fn block_uses_ptr_wrap(nodes: &[BfIr]) -> bool {
        for node in nodes {
            match node {
                BfIr::MovePtr(n) => {
                    if *n != 0 {
                        return true;
                    }
                }
                BfIr::Loop(body) => {
                    if block_uses_ptr_wrap(body) {
                        return true;
                    }
                }
                BfIr::Distribute { targets } => {
                    if targets.iter().any(|(off, _)| *off != 0) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn block_uses_input(nodes: &[BfIr]) -> bool {
        for node in nodes {
            match node {
                BfIr::Input => return true,
                BfIr::Loop(body) => {
                    if block_uses_input(body) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn block_uses_cell_mask(nodes: &[BfIr], opts: CodegenOpts) -> bool {
        if matches!(opts.io_mode, IoMode::Number) {
            for node in nodes {
                match node {
                    BfIr::Input | BfIr::Output => return true,
                    BfIr::Loop(body) => {
                        if block_uses_cell_mask(body, opts) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    let needs_ptr_wrap = block_uses_ptr_wrap(program);
    let needs_input = block_uses_input(program);
    let needs_cell_mask = block_uses_cell_mask(program, opts);

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
    emit_block(program, 1, &mut out, opts);
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

    let optimized = optimize(parsed.clone());

    match mode {
        OutputMode::DumpIr => {
            println!("=== Parsed IR ===");
            print!("{}", format_ir(&parsed));

            println!("=== Optimized IR ===");
            print!("{}", format_ir(&optimized));
        }
        OutputMode::EmitC => {
            print!("{}", emit_c(&optimized, codegen_opts));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_opt(src: &str) -> Vec<BfIr> {
        let parsed = Parser::new(src).parse().unwrap();
        optimize(parsed)
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
        assert_eq!(ir, vec![BfIr::Loop(vec![BfIr::Clear])]);
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
}
