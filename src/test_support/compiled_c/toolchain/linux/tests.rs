use super::*;
use crate::RequiredExt;

#[test]
fn metadata_rejects_unresolved_inputs_and_search_mutating_scripts() {
    assert!(linker_search_dirs("SEARCH_DIR(\"relative\")").is_err());
    assert!(linker_search_dirs("SEARCH_DIR(/lib)").is_err());
    assert_eq!(
        linker_search_dirs("SEARCH_DIR(\"=/lib\")").or_invariant("GNU search metadata"),
        vec![PathBuf::from("/lib")]
    );
    assert!(loader_inputs("libc.so => not found").is_err());
    assert!(loader_inputs("").is_err());
    assert!(validation::simple_link_script("/* standard */ OUTPUT_FORMAT(\"elf64-littleaarch64\") GROUP ( /lib/libc.so.6 AS_NEEDED ( /lib/ld.so ) )").is_ok());
    for script in [
        "SEARCH_DIR(\"/custom\")",
        "INCLUDE other.ld",
        "VERSION { global: symbol; }",
        "/* unfinished",
    ] {
        assert!(
            validation::simple_link_script(script).is_err(),
            "unsafe script accepted: {script}"
        );
    }
}

#[test]
fn catalog_tracks_content_aliases_and_new_search_inputs() {
    use std::os::unix::fs::symlink;
    let root = crate::test_support::compiled_c::tests::TestRoot::new();
    fs::create_dir_all(&root.0).or_invariant("catalog root");
    let first = root.0.join("one.so");
    let second = root.0.join("two.so");
    let alias = root.0.join("library.so");
    fs::write(&first, b"one").or_invariant("first library");
    fs::write(&second, b"two").or_invariant("second library");
    symlink(&first, &alias).or_invariant("library alias");
    let seal = || catalog::seal(vec![root.0.clone()]).or_invariant("catalog seal");
    let original = seal();
    fs::write(&first, b"new").or_invariant("same-length library mutation");
    assert_ne!(
        original,
        seal(),
        "runtime byte changes reused a fingerprint"
    );
    let changed = seal();
    fs::remove_file(&alias).or_invariant("old alias");
    symlink(&second, &alias).or_invariant("rebound alias");
    assert_ne!(
        changed,
        seal(),
        "link resolution change reused a fingerprint"
    );
    let rebound = seal();
    fs::write(root.0.join("shadow.so"), b"shadow").or_invariant("new search input");
    assert_ne!(rebound, seal(), "new search input reused a fingerprint");
}

#[test]
fn debug_and_sanitizer_modes_reuse_only_their_own_compilation() {
    let root = crate::test_support::compiled_c::tests::TestRoot::new();
    for flags in [&["-g3"][..], &["-O3"], &["-fsanitize=undefined"]] {
        for warm in [false, true] {
            let mut counts = accounting::Request::new();
            let output = run_in(&root.0, "int main(void) { return 0; }", flags, &mut counts)
                .or_invariant("Linux mode execution");
            assert!(output.status.success(), "flags={flags:?}, warm={warm}");
            assert_eq!(
                counts.compiler_invocations,
                u64::from(!warm),
                "flags={flags:?}, warm={warm}"
            );
            assert_eq!(counts.execution_attempts, 1, "warm reuse skipped execution");
        }
    }
}

#[test]
fn unsealed_assembler_inputs_bypass_publication_and_headers_invalidate_reuse() {
    let root = crate::test_support::compiled_c::tests::TestRoot::new();
    fs::create_dir_all(&root.0).or_invariant("input root");
    let data = root.0.join("external.bin");
    let header = root.0.join("value.h");
    // GNU as records incbin dependencies, which preprocessing cannot see.
    let assembly = format!(
        "__asm__(\".pushsection .rodata\\n.incbin \\\"{}\\\"\\n.popsection\");\nint main(void) {{ return 0; }}",
        data.display()
    );
    for byte in [1, 2] {
        fs::write(&data, [byte]).or_invariant("external assembly input");
        let mut counts = accounting::Request::new();
        let output =
            run_in(&root.0, &assembly, &[], &mut counts).or_invariant("unsealed input execution");
        assert!(output.status.success());
        assert_eq!(
            counts.compiler_invocations, 1,
            "unsealed incbin input was cached"
        );
    }
    let source = format!(
        "#include \"{}\"\nint main(void) {{ return VALUE; }}",
        header.display()
    );
    for value in [0, 1] {
        fs::write(&header, format!("#define VALUE {value}\n")).or_invariant("header edit");
        for warm in [false, true] {
            let mut counts = accounting::Request::new();
            let output =
                run_in(&root.0, &source, &[], &mut counts).or_invariant("header execution");
            assert_eq!(
                output.status.code(),
                Some(value),
                "header change hidden by cache"
            );
            assert_eq!(counts.compiler_invocations, u64::from(!warm));
            assert_eq!(counts.execution_attempts, 1);
        }
    }
}
