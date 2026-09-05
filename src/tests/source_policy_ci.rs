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

    let workflow: serde_json::Value =
        yaml_serde::from_str(&read_repo_file(repo_root, ".github/workflows/ci.yml"))
            .or_invariant("structured CI workflow");
    let jobs = workflow["jobs"].as_object().or_invariant("CI jobs");
    for (name, job) in jobs {
        let steps = job["steps"].as_array().or_invariant("job steps");
        let actions: Vec<_> = steps
            .iter()
            .filter_map(|step| step["uses"].as_str())
            .collect();
        assert!(
            actions
                .iter()
                .any(|action| action.starts_with("actions/checkout@")),
            "job {name} lacks checkout"
        );
        assert!(
            actions.contains(&"./.github/actions/rust-setup"),
            "job {name} lacks shared Rust setup"
        );
    }
    let cross = &jobs["cross-platform"];
    assert_eq!(
        cross["strategy"]["matrix"]["os"],
        serde_json::json!(["macos-latest", "ubuntu-latest", "windows-latest"])
    );
    let steps = cross["steps"].as_array().or_invariant("platform steps");
    assert!(
        steps
            .iter()
            .any(|step| step["run"] == "${{ runner.os == 'Windows' && 'python' || 'python3' }} .github/scripts/verify_ci_binaries.py"),
        "ordinary binary artifact inventory must run on every platform"
    );
    assert!(
        steps
            .iter()
            .any(|step| step["run"]
                == "cargo test --workspace --all-features --release -- --nocapture"),
        "full release/doctest coverage must remain"
    );
    assert!(jobs.contains_key("static-crt-musl"));
    assert!(jobs.contains_key("native-kernels"));
    assert!(jobs.contains_key("scalar-fallback"));
    assert!(
        jobs.contains_key("generated-c"),
        "debug sanitizer configuration is not redundant with release coverage"
    );
    assert!(jobs.contains_key("miri"));
    assert_eq!(workflow["concurrency"]["cancel-in-progress"], true);

    let rust_setup = read_repo_file(repo_root, ".github/actions/rust-setup/action.yml");
    assert!(
        rust_setup.contains("dtolnay/rust-toolchain@stable")
            && rust_setup.contains("dtolnay/rust-toolchain@nightly")
            && rust_setup.contains("uses: Swatinem/rust-cache@"),
        "the shared Rust setup must own stable, nightly, and cache actions"
    );

    let dependabot = read_repo_file(repo_root, ".github/dependabot.yml");
    assert!(
        dependabot.contains("package-ecosystem: github-actions")
            && dependabot.contains("package-ecosystem: cargo"),
        "Dependabot must keep both GitHub actions and Cargo dependencies current"
    );
}

fn read_repo_file(repo_root: &Path, relative: &str) -> String {
    let path = repo_root.join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to read {}: {error}", path.display())
    })
}
