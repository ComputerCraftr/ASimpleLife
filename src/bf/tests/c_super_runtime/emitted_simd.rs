use super::*;

fn emitted_payload(output: &str, backend: TestCBackend) -> String {
    match backend {
        TestCBackend::Plain => split_plain_c_payload(output).to_owned(),
        TestCBackend::Super => split_super_c_payload(output).to_owned(),
    }
}

#[test]
fn emitted_move_add_pairs_reach_the_add_kernel() {
    let program = vec![
        BfIr::MovePtr(1),
        BfIr::Add(1),
        BfIr::MovePtr(1),
        BfIr::Add(2),
        BfIr::MovePtr(1),
        BfIr::Add(3),
        BfIr::MovePtr(1),
        BfIr::Add(4),
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let source = emit_backend_ir(&program, opts, backend);
        assert!(
            source.contains("bf_add_at_batch(tape, ptr"),
            "backend={backend:?} did not lower the IR batch through bf_add_at_batch"
        );
        let native = compile_and_run_c_source_with_args(&source, &["-O3"]);
        let scalar =
            compile_and_run_c_source_with_args(&source, &["-O3", "-DBF_FORCE_SCALAR_KERNEL"]);
        assert_eq!(
            emitted_payload(&native, backend),
            emitted_payload(&scalar, backend)
        );
        assert_eq!(emitted_payload(&native, backend).trim(), "4");
        assert_kernel_lane_conservation(&native, backend, "add");
        assert_eq!(backend_kernel_stat(&native, backend, "semantic steps:"), 15);
        assert_eq!(backend_kernel_stat(&scalar, backend, "semantic steps:"), 15);
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            assert_eq!(
                backend_kernel_stat(&native, backend, "bf native add kernel lanes:"),
                4,
                "backend={backend:?}\n{native}"
            );
            if matches!(backend, TestCBackend::Super) {
                assert!(
                    backend_kernel_stat(&native, backend, "bf native probe kernel lanes:") > 0,
                    "the emitted super-C program did not reach native memo probing:\n{native}"
                );
            }
        }
        assert_eq!(
            backend_kernel_stat(&scalar, backend, "bf native add kernel lanes:"),
            0,
            "backend={backend:?}\n{scalar}"
        );
    }
}

#[test]
fn emitted_clear_sequence_reaches_the_clear_set_kernel() {
    let program = vec![
        BfIr::Clear,
        BfIr::ClearAt { offset: 1 },
        BfIr::ClearAt { offset: 2 },
        BfIr::ClearAt { offset: 3 },
        BfIr::Output,
    ];
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let source = emit_backend_ir(&program, opts, backend);
        assert!(
            source.contains("bf_clear_set_batch(tape, ptr"),
            "backend={backend:?} did not lower the IR batch through bf_clear_set_batch"
        );
        let native = compile_and_run_c_source_with_args(&source, &["-O3"]);
        let scalar =
            compile_and_run_c_source_with_args(&source, &["-O3", "-DBF_FORCE_SCALAR_KERNEL"]);
        assert_eq!(
            emitted_payload(&native, backend),
            emitted_payload(&scalar, backend)
        );
        assert_eq!(emitted_payload(&native, backend).trim(), "0");
        assert_kernel_lane_conservation(&native, backend, "clear/set");
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            assert_eq!(
                backend_kernel_stat(&native, backend, "bf native clear/set kernel lanes:"),
                4,
                "backend={backend:?}\n{native}"
            );
        }
        assert_eq!(
            backend_kernel_stat(&scalar, backend, "bf native clear/set kernel lanes:"),
            0,
            "backend={backend:?}\n{scalar}"
        );
    }
}
