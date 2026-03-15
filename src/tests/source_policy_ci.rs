use crate::RequiredExt;
use crate::test_support::{CfgPredicate, RustRepository};
use std::fs;
use std::path::Path;

#[test]
fn cargo_targets_force_static_crt_for_musl_and_msvc() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_path = repo_root.join(".cargo/config.toml");
    let config = fs::read_to_string(&config_path).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to read {}: {error}", config_path.display())
    });
    let msvc_section = "[target.'cfg(target_env = \"msvc\")']";
    let start = config.find(msvc_section).unwrap_or_else(|| {
        crate::invariant_failure!("missing static CRT Cargo section {msvc_section}")
    });
    let remainder = &config[start + msvc_section.len()..];
    let end = remainder.find("\n[").unwrap_or(remainder.len());
    assert!(
        remainder[..end].contains("target-feature=+crt-static"),
        "{msvc_section} must enable the static CRT explicitly"
    );

    let repository = RustRepository::discover(repo_root).or_invariant("parsed repository sources");
    let lib = repository
        .source_containing_macro("compile_error")
        .or_invariant("unique crate compile guard source");
    assert_eq!(
        lib.macro_cfg_predicate("compile_error"),
        Some(CfgPredicate::All(vec![
            CfgPredicate::Not(Box::new(CfgPredicate::Flag("doc".to_string()))),
            CfgPredicate::Any(vec![
                CfgPredicate::KeyValue {
                    key: "target_env".to_string(),
                    value: "musl".to_string(),
                },
                CfgPredicate::KeyValue {
                    key: "target_env".to_string(),
                    value: "msvc".to_string(),
                },
            ]),
            CfgPredicate::Not(Box::new(CfgPredicate::KeyValue {
                key: "target_feature".to_string(),
                value: "crt-static".to_string(),
            })),
        ])),
        "the crate must reject non-rustdoc musl or MSVC builds that disable crt-static"
    );

    let workflow = read_repo_file(repo_root, ".github/workflows/ci.yml");
    assert!(
        workflow.contains("find target/x86_64-unknown-linux-musl/release -maxdepth 1")
            && workflow.contains("readelf -d \"$executable\"")
            && workflow.contains("llvm-readobj --coff-imports")
            && workflow.contains("(vcruntime|msvcp|ucrtbase)")
            && workflow.contains(
                "RUSTDOCFLAGS: ${{ matrix.os == 'windows-latest' && '-C target-feature=+crt-static' || '' }}"
            ),
        "CI must inspect every linked executable and compile Windows rustdoc work with the static CRT"
    );
    let job_count = workflow.matches("runs-on:").count();
    assert_eq!(
        workflow.matches("actions/checkout@v6").count(),
        job_count,
        "every CI job must use the Node 24 checkout action"
    );
    assert_eq!(
        workflow
            .matches("uses: ./.github/actions/rust-setup")
            .count(),
        job_count,
        "every CI job must use the shared Rust setup action"
    );
    assert!(
        !workflow.contains("actions/checkout@v4") && !workflow.contains("actions/checkout@v5"),
        "deprecated checkout action lines must not return"
    );
    assert!(
        workflow.contains("rhysd/actionlint:1.7.12"),
        "CI must validate workflow changes with the current actionlint release"
    );

    let rust_setup = read_repo_file(repo_root, ".github/actions/rust-setup/action.yml");
    assert!(
        rust_setup.contains("dtolnay/rust-toolchain@stable")
            && rust_setup.contains("dtolnay/rust-toolchain@nightly")
            && rust_setup.contains("Swatinem/rust-cache@v2.9.2"),
        "the shared Rust setup must own current stable, nightly, and cache actions"
    );

    let dependabot = read_repo_file(repo_root, ".github/dependabot.yml");
    assert!(
        dependabot.contains("package-ecosystem: github-actions"),
        "Dependabot must keep GitHub actions current"
    );
}

fn read_repo_file(repo_root: &Path, relative: &str) -> String {
    let path = repo_root.join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to read {}: {error}", path.display())
    })
}
