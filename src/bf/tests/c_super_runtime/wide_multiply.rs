use super::*;

fn payload(output: &str, backend: TestCBackend) -> &str {
    match backend {
        TestCBackend::Plain => split_plain_c_payload(output),
        TestCBackend::Super => split_super_c_payload(output),
    }
}

fn native_simd_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

fn assert_wide_lane_accounting(
    output: &str,
    backend: TestCBackend,
    family: &str,
    total: u64,
    native_lanes: u64,
    native: bool,
) {
    let expected_native = if native && native_simd_available() {
        native_lanes
    } else {
        0
    };
    assert_eq!(
        backend_kernel_stat(output, backend, &format!("bf {family} kernel lanes:")),
        total,
        "backend={backend:?} family={family}\n{output}"
    );
    assert_eq!(
        backend_kernel_stat(
            output,
            backend,
            &format!("bf native {family} kernel lanes:")
        ),
        expected_native,
        "backend={backend:?} family={family}\n{output}"
    );
    assert_eq!(
        backend_kernel_stat(
            output,
            backend,
            &format!("bf scalar {family} kernel lanes:")
        ),
        total - expected_native,
        "backend={backend:?} family={family}\n{output}"
    );
    assert_kernel_lane_conservation(output, backend, family);
}

fn wide_polynomial_source(backend: TestCBackend, bits: u32, signed_cells: u32) -> String {
    let template = backend.runtime_template();
    let stats_call = backend.stats_call();
    format!(
        "#define BF_TEMPLATE_CELL_BITS {bits}\n\
         #define BF_TEMPLATE_SIGNED_CELLS {signed_cells}\n\
         #define main bf_template_main\n{template}\n#undef main\n\
         int main(void) {{\n\
             const int64_t lhs[] = {{INT64_C(1099511627779), -INT64_C(549755813893), INT64_C(4294967303), -INT64_C(9), INT64_C(2305843009213693963)}};\n\
             const int64_t rhs[] = {{-INT64_C(2199023255565), INT64_C(8589934609), -INT64_C(17), INT64_C(4611686018427387887), INT64_C(33)}};\n\
             int64_t out[5] = {{0}};\n\
             bf_kernel_init();\n\
             bf_polynomial_mul_batch(lhs, rhs, out, 3, {bits}, {signed_cells});\n\
             for (size_t i = 0; i < 3; ++i) printf(\"%\" PRId64 \" \", out[i]);\n\
             putchar('\\n');\n\
             bf_polynomial_mul_batch(lhs, rhs, out, 4, {bits}, {signed_cells});\n\
             for (size_t i = 0; i < 4; ++i) printf(\"%\" PRId64 \" \", out[i]);\n\
             putchar('\\n');\n\
             bf_polynomial_mul_batch(lhs, rhs, out, 5, {bits}, {signed_cells});\n\
             for (size_t i = 0; i < 5; ++i) printf(\"%\" PRId64 \" \", out[i]);\n\
             putchar('\\n'); {stats_call} return 0;\n\
         }}\n"
    )
}

fn wide_transfer_source(backend: TestCBackend, bits: u32, signed_cells: u32) -> String {
    let template = backend.runtime_template();
    let stats_call = backend.stats_call();
    format!(
        "#define BF_TEMPLATE_CELL_BITS {bits}\n\
         #define BF_TEMPLATE_SIGNED_CELLS {signed_cells}\n\
         #define main bf_template_main\n{template}\n#undef main\n\
         int main(void) {{\n\
             int64_t tape[BF_TEMPLATE_TAPE_LEN] = {{0}};\n\
             const ptrdiff_t offsets[] = {{1, 2, 3, 4, 5}};\n\
             const int64_t coefficients[] = {{INT64_C(1099511627779), -INT64_C(549755813893), INT64_C(4294967303), -INT64_C(9), INT64_C(2305843009213693963)}};\n\
             const int64_t initial[] = {{INT64_C(17), -INT64_C(19), INT64_C(4611686018427387887), INT64_C(23), -INT64_C(29)}};\n\
             tape[0] = -INT64_C(2199023255565);\n\
             for (size_t i = 0; i < 5; ++i) tape[i + 1] = initial[i];\n\
             bf_kernel_init();\n\
             bf_transfer_batch(tape, 0, 0, offsets, coefficients, 4, 1, BF_TEMPLATE_TAPE_LEN, {bits}, {signed_cells});\n\
             for (size_t i = 0; i < 5; ++i) tape[i + 1] = initial[i];\n\
             bf_transfer_batch(tape, 0, 0, offsets, coefficients, 5, 1, BF_TEMPLATE_TAPE_LEN, {bits}, {signed_cells});\n\
             for (size_t i = 1; i <= 5; ++i) printf(\"%\" PRId64 \" \", tape[i]);\n\
             putchar('\\n'); {stats_call} return 0;\n\
         }}\n"
    )
}

fn wide_summary_source(bits: u32, signed_cells: u32) -> String {
    let template = TestCBackend::Super.runtime_template();
    format!(
        "#define BF_TEMPLATE_CELL_BITS {bits}\n\
         #define BF_TEMPLATE_SIGNED_CELLS {signed_cells}\n\
         #define main bf_template_main\n{template}\n#undef main\n\
         int main(void) {{\n\
             int64_t tape[BF_TEMPLATE_TAPE_LEN] = {{0}};\n\
             const ptrdiff_t offsets[] = {{1, 2, 3, 4, 5}};\n\
             const int64_t lhs[] = {{INT64_C(1099511627779), -INT64_C(549755813893), INT64_C(4294967303), -INT64_C(9), INT64_C(2305843009213693963)}};\n\
             const int64_t rhs[] = {{-INT64_C(2199023255565), INT64_C(8589934609), -INT64_C(17), INT64_C(4611686018427387887), INT64_C(33)}};\n\
             const int64_t coefficients[] = {{INT64_C(34359738371), -INT64_C(65), INT64_C(1125899906842631), INT64_C(7), -INT64_C(2147483651)}};\n\
             const int64_t initial[] = {{INT64_C(17), -INT64_C(19), INT64_C(4611686018427387887), INT64_C(23), -INT64_C(29)}};\n\
             for (size_t i = 0; i < 5; ++i) tape[i + 1] = initial[i];\n\
             bf_kernel_init();\n\
             bf_summary_product_batch(tape, 0, offsets, lhs, rhs, coefficients, 4);\n\
             for (size_t i = 0; i < 5; ++i) tape[i + 1] = initial[i];\n\
             bf_summary_product_batch(tape, 0, offsets, lhs, rhs, coefficients, 5);\n\
             for (size_t i = 1; i <= 5; ++i) printf(\"%\" PRId64 \" \", tape[i]);\n\
             putchar('\\n'); print_memo_stats(); return 0;\n\
         }}\n"
    )
}

#[test]
fn unoptimized_multiply_families_match_scalar_without_vector_call_abi() {
    // The optimized tests inline the shared multiply helper opportunistically,
    // hiding MinGW by-value vector ABI failures in ordinary -O0 builds.
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        for bits in [8, 63] {
            let mut cases = vec![
                (wide_transfer_source(backend, bits, 1), "transfer", 9),
                (wide_polynomial_source(backend, bits, 1), "polynomial", 12),
            ];
            if matches!(backend, TestCBackend::Super) {
                cases.push((wide_summary_source(bits, 1), "summary", 9));
            }
            for (source, family, total) in cases {
                let native = compile_and_run_c_source_with_args(&source, &["-O0"]);
                let scalar = compile_and_run_c_source_with_args(
                    &source,
                    &["-O0", "-DBF_FORCE_SCALAR_KERNEL"],
                );
                assert_eq!(
                    payload(&native, backend),
                    payload(&scalar, backend),
                    "unoptimized backend={backend:?} family={family} bits={bits}"
                );
                assert_wide_lane_accounting(&native, backend, family, total, 8, true);
                assert_wide_lane_accounting(&scalar, backend, family, total, 8, false);
            }
        }
    }
}

#[test]
fn generated_c_wide_transfer_matches_forced_scalar_for_full_and_partial_lanes() {
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        for bits in [33, 63] {
            for signed_cells in [0, 1] {
                let source = wide_transfer_source(backend, bits, signed_cells);
                let native =
                    compile_and_run_c_source_with_args(&source, &["-O3", "-pedantic-errors"]);
                let scalar = compile_and_run_c_source_with_args(
                    &source,
                    &["-O3", "-pedantic-errors", "-DBF_FORCE_SCALAR_KERNEL"],
                );
                assert_eq!(
                    payload(&native, backend),
                    payload(&scalar, backend),
                    "template={backend:?} bits={bits} signed={signed_cells}"
                );
                assert_wide_lane_accounting(&native, backend, "transfer", 9, 8, true);
                assert_wide_lane_accounting(&scalar, backend, "transfer", 9, 8, false);
            }
        }
    }
}

#[test]
fn generated_super_c_wide_product_matches_forced_scalar_for_full_and_partial_lanes() {
    for bits in [33, 63] {
        for signed_cells in [0, 1] {
            let source = wide_summary_source(bits, signed_cells);
            let native = compile_and_run_c_source_with_args(&source, &["-O3", "-pedantic-errors"]);
            let scalar = compile_and_run_c_source_with_args(
                &source,
                &["-O3", "-pedantic-errors", "-DBF_FORCE_SCALAR_KERNEL"],
            );
            assert_eq!(
                payload(&native, TestCBackend::Super),
                payload(&scalar, TestCBackend::Super),
                "bits={bits} signed={signed_cells}"
            );
            assert_wide_lane_accounting(&native, TestCBackend::Super, "summary", 9, 8, true);
            assert_wide_lane_accounting(&scalar, TestCBackend::Super, "summary", 9, 8, false);
        }
    }
}

#[test]
fn generated_c_polynomial_batch_respects_native_dispatch_boundary() {
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        for bits in [33, 63] {
            for signed_cells in [0, 1] {
                let source = wide_polynomial_source(backend, bits, signed_cells);
                let native =
                    compile_and_run_c_source_with_args(&source, &["-O3", "-pedantic-errors"]);
                let scalar = compile_and_run_c_source_with_args(
                    &source,
                    &["-O3", "-pedantic-errors", "-DBF_FORCE_SCALAR_KERNEL"],
                );
                assert_eq!(
                    payload(&native, backend),
                    payload(&scalar, backend),
                    "template={backend:?} bits={bits} signed={signed_cells}"
                );
                assert_wide_lane_accounting(&native, backend, "polynomial", 12, 8, true);
                assert_wide_lane_accounting(&scalar, backend, "polynomial", 12, 8, false);
            }
        }
    }
}
