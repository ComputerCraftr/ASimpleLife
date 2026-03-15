use super::*;

#[test]
fn emit_c_super_signed_runtime_outputs_negative_value() {
    let stdout = compile_and_run_emitted_backend(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
        TestCBackend::Super,
        false,
    );
    assert!(stdout.starts_with("-1\n"));
    assert!(stdout.contains("memo hits:"));
    assert!(stdout.contains("memo misses:"));
}

#[test]
#[cfg(not(target_os = "windows"))]
fn emit_c_super_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_backend(
        "++++[--]++++[--]",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 8,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Unsigned,
        },
        TestCBackend::Super,
        true,
    );
    assert!(stdout.contains("memo hits:"));
    assert!(stdout.contains("memo evictions:"));
}

#[test]
#[cfg(not(target_os = "windows"))]
fn emit_c_super_signed_63bit_runtime_is_clean_under_asan_ubsan() {
    let stdout = compile_and_run_emitted_backend(
        "-.",
        CodegenOpts {
            io_mode: IoMode::Number,
            cell_bits: 63,
            input_bits: None,
            output_bits: None,
            cell_sign: CellSign::Signed,
        },
        TestCBackend::Super,
        true,
    );
    assert!(stdout.starts_with("-1\n"));
}
