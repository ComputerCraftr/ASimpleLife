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
            CSource::parse(&source).has_call("bf_add_at_batch"),
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
            CSource::parse(&source).has_call("bf_clear_set_batch"),
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

#[test]
fn emitted_distribute_uses_transfer_kernel_and_wrapped_aliases_fall_back() {
    let opts = CodegenOpts {
        io_mode: IoMode::Number,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let disjoint = [
        BfIr::Add(3),
        BfIr::Distribute {
            targets: vec![(11, 1), (12, -2), (13, 3), (14, 4)],
            preserve_src: true,
        },
        BfIr::MovePtr(11),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    let aliased = [
        BfIr::Add(3),
        BfIr::Distribute {
            targets: vec![(11, 1), (30_011, 2), (12, 3), (13, 4)],
            preserve_src: true,
        },
        BfIr::MovePtr(11),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
        BfIr::MovePtr(1),
        BfIr::Output,
    ];
    for backend in [TestCBackend::Plain, TestCBackend::Super] {
        let output = compile_and_run_ir_backend(&disjoint, opts, backend);
        assert_eq!(
            emitted_payload(&output, backend).trim(),
            "3\n250\n9\n12",
            "backend={backend:?} disjoint payload\n{output}"
        );
        let stat = |prefix| backend_kernel_stat(&output, backend, prefix);
        assert_eq!(
            stat("bf transfer kernel lanes:"),
            4,
            "backend={backend:?} {output}"
        );
        assert_kernel_lane_conservation(&output, backend, "transfer");
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            assert_eq!(
                stat("bf native transfer kernel lanes:"),
                4,
                "backend={backend:?} {output}"
            );
        }

        let output = compile_and_run_ir_backend(&aliased, opts, backend);
        assert_eq!(
            emitted_payload(&output, backend).trim(),
            "9\n9\n12\n0",
            "backend={backend:?} aliased payload\n{output}"
        );
        let stat = |prefix| backend_kernel_stat(&output, backend, prefix);
        assert_eq!(
            stat("bf transfer kernel lanes:"),
            4,
            "backend={backend:?} {output}"
        );
        assert_kernel_lane_conservation(&output, backend, "transfer");
        assert_eq!(
            stat("bf native transfer kernel lanes:"),
            0,
            "wrapped aliases must not be attributed to native transfer: backend={backend:?} {output}"
        );
        assert!(
            stat("bf scalar kernel lanes:") >= 4,
            "wrapped duplicate destinations must force scalar transfer: backend={backend:?} {output}"
        );
    }
}
