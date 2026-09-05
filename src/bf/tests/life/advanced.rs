use super::*;
use crate::bf::{AssetRegistry, LifeAssetError};
use crate::bitgrid::BitGrid;
use crate::test_support::RustRepository;
use std::path::Path;

fn source_repository() -> RustRepository {
    RustRepository::discover(Path::new(env!("CARGO_MANIFEST_DIR")))
        .or_invariant("parsed repository Rust sources")
}

fn assert_release_compiler_is_independent_from_scaffold(repository: &RustRepository) {
    let scaffold = repository
        .source_containing_callable("compile_life_scaffold")
        .or_invariant("unique scaffold declaration");
    assert!(
        scaffold
            .function_is_test_only("compile_life_scaffold")
            .or_invariant("test Life scaffold declaration"),
        "the reference scaffold must remain explicitly test-only"
    );
    let references = repository
        .source_containing_callable("compile_to_life_circuit")
        .or_invariant("unique release compiler declaration")
        .function_path_references("compile_to_life_circuit")
        .or_invariant("release Life compiler declaration");
    assert!(
        !references.iter().any(|name| matches!(
            name.as_str(),
            "compile_life_scaffold" | "ReferenceLifeScaffold"
        )),
        "release Life compiler references test-only scaffold symbols: {:?}",
        references
    );
}

#[test]
fn life_release_api_fails_closed_for_primitive_ir() {
    let ir = vec![BfIr::Add(1), BfIr::MovePtr(1), BfIr::Add(-1)];
    let compile_error = crate::bf::compile_to_life_circuit(&ir, life_opts())
        .error_or_invariant("primitive Life compilation must fail closed");
    assert_eq!(
        compile_error,
        BfLifeCircuitError::ExecutableCircuitUnavailable
    );
}

#[test]
fn life_release_api_fails_closed_for_loops() {
    let ir = parse_and_opt("+++[->+<]");
    let compile_error = crate::bf::compile_to_life_circuit(&ir, life_opts())
        .error_or_invariant("loop Life compilation must fail closed");
    assert_eq!(
        compile_error,
        BfLifeCircuitError::ExecutableCircuitUnavailable
    );
}

#[test]
fn life_release_api_fails_closed_for_output() {
    let ir = vec![BfIr::Add(65), BfIr::Output];
    let compile_error = crate::bf::compile_to_life_circuit(&ir, life_opts())
        .error_or_invariant("output Life compilation must fail closed");
    assert_eq!(
        compile_error,
        BfLifeCircuitError::ExecutableCircuitUnavailable
    );
}

#[test]
fn life_release_api_fails_closed_for_rich_ir() {
    let ir = vec![
        BfIr::Add(3),
        BfIr::Shift {
            src: 0,
            dst: 1,
            amount: 2,
            dir: ShiftDir::Left,
            preserve_src: true,
            set_dst: true,
        },
        BfIr::Square {
            src: 1,
            dst: 2,
            preserve_src: true,
            set_dst: false,
        },
        BfIr::Output,
    ];
    let compile_error = crate::bf::compile_to_life_circuit(&ir, life_opts())
        .error_or_invariant("rich-IR Life compilation must fail closed");
    assert_eq!(
        compile_error,
        BfLifeCircuitError::ExecutableCircuitUnavailable
    );
}

#[test]
fn life_release_asset_manifest_declares_executable_backend_blocked() {
    let registry = AssetRegistry::load_repository().or_invariant("BF Life asset manifest");
    assert!(
        registry.manifest().components.is_empty(),
        "the blocked registry must not advertise unverified scaffold patterns as components"
    );
    assert!(
        matches!(
            registry.verify(),
            Err(LifeAssetError::ManifestBlocked { reason })
                if reason.contains("physical component")
        ),
        "BF-to-Life assets must remain fail-closed until independently verified components exist"
    );
}

#[test]
fn life_release_contract_rejects_unsupported_modes_before_asset_lookup() {
    let signed = CodegenOpts {
        cell_sign: CellSign::Signed,
        ..life_opts()
    };
    assert_eq!(
        crate::bf::compile_to_life_circuit(&[], signed)
            .error_or_invariant("signed Life compilation must be rejected"),
        BfLifeCircuitError::SignedCellsUnsupported
    );

    for requested in [0, 1, 7, 9, 16, 32, 63, 64] {
        let opts = CodegenOpts {
            cell_bits: requested,
            ..life_opts()
        };
        assert_eq!(
            crate::bf::compile_to_life_circuit(&[], opts)
                .error_or_invariant("non-8-bit Life compilation must be rejected"),
            BfLifeCircuitError::UnsupportedCellWidth { requested }
        );
    }

    assert_eq!(
        crate::bf::compile_to_life_circuit(&[BfIr::Input], life_opts())
            .error_or_invariant("Life input must be rejected"),
        BfLifeCircuitError::InputUnsupported
    );
    assert_eq!(
        crate::bf::compile_to_life_circuit(&[BfIr::Loop(vec![BfIr::Input])], life_opts())
            .error_or_invariant("nested Life input must be rejected"),
        BfLifeCircuitError::InputUnsupported
    );

    let oversized = vec![BfIr::Add(1); 4_097];
    assert_eq!(
        crate::bf::compile_to_life_circuit(&oversized, life_opts())
            .error_or_invariant("oversized physical program must be rejected"),
        BfLifeCircuitError::ProgramTooLarge {
            instructions: 4_097,
            maximum: 4_096,
        }
    );
}

#[test]
fn immutable_compiled_life_program_serialization_uses_only_life_cells() {
    let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)]);
    let program = crate::bf::life_backend::CompiledLifeProgram::test_fixture(grid.clone());

    assert_eq!(program.initial_grid(), &grid);
    assert_eq!(program.tape_cells(), 64);
    assert_eq!(program.cell_bits(), 8);
    assert_eq!(program.timing().generations_per_tick(), 4);
    assert_eq!(program.timing().output_frame_generations(), 36);
    assert_eq!(program.decoder().sync_pulses(), 1);
    assert_eq!(program.decoder().data_bits(), 8);
    assert!(program.decoder().least_significant_bit_first());
    assert_eq!(program.observation_ports().len(), 1);
    assert_eq!(program.observation_ports()[0].name(), "framed_output");
    assert_eq!(program.observation_ports()[0].minimum_clearance(), 8);
    let bounds = program.static_layout().machine_bounds();
    let port = program.observation_ports()[0].origin();
    assert!(port.0 > bounds.max_x() && port.1 > bounds.max_y());

    let life = crate::bf::serialize_life_circuit(&program).or_invariant("Life serialization");
    let snapshot =
        crate::bf::serialize_life_circuit_hashlife(&program).or_invariant("HashLife serialization");
    assert_eq!(
        deserialize_life_grid(&life).or_invariant("serialized Life grid"),
        grid
    );
    assert_eq!(
        deserialize_grid(&snapshot).or_invariant("serialized HashLife grid"),
        grid
    );
}

#[test]
fn life_release_readiness_cannot_be_inferred_from_test_scaffold() {
    let repository = source_repository();
    assert_release_compiler_is_independent_from_scaffold(&repository);
}

#[test]
fn life_cli_routes_emission_through_the_public_compiler() {
    let repository = source_repository();
    let cli = repository
        .source_containing_function_reference("run", "compile_to_life_circuit")
        .or_invariant("unique CLI run function calling the public Life compiler");
    let run_references = cli
        .function_path_references("run")
        .or_invariant("BF CLI run function");
    assert!(
        run_references
            .iter()
            .any(|name| name == "compile_to_life_circuit"),
        "Life CLI emission must execute the public compiler contract"
    );
    assert!(
        !run_references
            .iter()
            .any(|name| name == "ExecutableCircuitUnavailable"),
        "Life CLI emission must not mask typed compiler errors with a hard-coded blocker"
    );
}

#[test]
fn compiled_life_artifact_contains_no_host_runtime_state() {
    let repository = source_repository();
    let backend = repository
        .source_containing_struct("CompiledLifeProgram")
        .or_invariant("unique compiled Life artifact declaration");
    let artifact_fields = backend
        .struct_field_names("CompiledLifeProgram")
        .or_invariant("CompiledLifeProgram field declaration");
    for forbidden in [
        "expected",
        "output_latch",
        "outputs",
        "tape",
        "program_counter",
        "trajectory",
        "runtime_state",
    ] {
        assert!(
            !artifact_fields.iter().any(|field| field == forbidden),
            "CompiledLifeProgram retains forbidden host-derived field token {forbidden:?}: {artifact_fields:?}"
        );
    }

    for function in ["serialize_life_circuit", "serialize_life_circuit_hashlife"] {
        let serializer = repository
            .source_containing_callable(function)
            .or_invariant("unique production Life serializer source");
        let references = serializer
            .function_path_references(function)
            .or_invariant("production Life serializer function");
        for forbidden in [
            "ReferenceLifeScaffold",
            "reference_run_to_completion",
            "outputs",
            "output_bit_seed",
        ] {
            assert!(
                !references.iter().any(|name| name == forbidden),
                "production Life serializer {function} references forbidden host state symbol {forbidden:?}"
            );
        }
    }
}
