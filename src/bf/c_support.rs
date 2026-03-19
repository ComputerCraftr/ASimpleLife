use super::optimizer::{CellSign, CodegenOpts, IoMode};

pub(super) fn indent4(level: usize) -> String {
    " ".repeat(level * 4)
}

pub(super) fn push_c_line(out: &mut String, level: usize, line: &str) {
    out.push_str(&indent4(level));
    out.push_str(line);
    out.push('\n');
}

pub(super) fn signed_cells_flag(sign: CellSign) -> &'static str {
    match sign {
        CellSign::Signed => "1",
        CellSign::Unsigned => "0",
    }
}

pub(super) fn mask_literal(bits: u32) -> String {
    if bits == 0 {
        "UINT64_C(0)".to_string()
    } else if bits >= 63 {
        "(UINT64_C(1) << 63) - 1".to_string()
    } else {
        format!("UINT64_C({})", (1u64 << bits) - 1)
    }
}

pub(super) fn wrap_ptr_expr(offset: isize) -> String {
    format!("bf_wrap_ptr(ptr, (ptrdiff_t)({offset}), BF_TAPE_LEN)")
}

pub(super) fn unified_input_stmt(opts: CodegenOpts) -> &'static str {
    match (opts.io_mode, opts.cell_sign) {
        (IoMode::Char, _) => {
            "{ int ch = getchar(); tape[ptr] = (ch == EOF) ? 0 : bf_wrap_from_u64(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS, BF_SIGNED_CELLS); }"
        }
        (IoMode::Number, CellSign::Signed) => {
            "{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS, BF_SIGNED_CELLS); }"
        }
        (IoMode::Number, CellSign::Unsigned) => {
            "{ uint64_t tmp = 0; if (scanf(\"%\" SCNu64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64(tmp & BF_INPUT_MASK, BF_CELL_BITS, BF_SIGNED_CELLS); }"
        }
    }
}

pub(super) fn unified_output_stmt(opts: CodegenOpts) -> &'static str {
    match (opts.io_mode, opts.cell_sign) {
        (IoMode::Char, _) => {
            "putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK)); fflush(stdout);"
        }
        (IoMode::Number, CellSign::Signed) => {
            "printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS, BF_SIGNED_CELLS)); fflush(stdout);"
        }
        (IoMode::Number, CellSign::Unsigned) => {
            "printf(\"%\" PRIu64 \"\\n\", ((uint64_t)tape[ptr]) & BF_OUTPUT_MASK); fflush(stdout);"
        }
    }
}

pub(super) fn split_char_input_stmt() -> &'static str {
    "{ int ch = getchar(); tape[ptr] = (ch == EOF) ? 0 : (BF_SIGNED_CELLS ? bf_wrap_from_u64_signed(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS) : bf_wrap_from_u64_unsigned(((uint64_t)(uint8_t)ch) & BF_INPUT_MASK, BF_CELL_BITS)); }"
}

pub(super) fn split_number_input_stmt(sign: CellSign) -> &'static str {
    match sign {
        CellSign::Signed => {
            "{ int64_t tmp = 0; if (scanf(\"%\" SCNd64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_signed(((uint64_t)tmp) & BF_INPUT_MASK, BF_CELL_BITS); }"
        }
        CellSign::Unsigned => {
            "{ uint64_t tmp = 0; if (scanf(\"%\" SCNu64, &tmp) != 1) tmp = 0; tape[ptr] = bf_wrap_from_u64_unsigned(tmp & BF_INPUT_MASK, BF_CELL_BITS); }"
        }
    }
}

pub(super) fn split_output_stmt(opts: CodegenOpts) -> &'static str {
    match (opts.io_mode, opts.cell_sign) {
        (IoMode::Char, _) => {
            "putchar((unsigned char)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"
        }
        (IoMode::Number, CellSign::Signed) => {
            "printf(\"%\" PRId64 \"\\n\", bf_wrap_from_u64_signed(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK, BF_CELL_BITS));"
        }
        (IoMode::Number, CellSign::Unsigned) => {
            "printf(\"%\" PRIu64 \"\\n\", (uint64_t)(((uint64_t)tape[ptr]) & BF_OUTPUT_MASK));"
        }
    }
}
