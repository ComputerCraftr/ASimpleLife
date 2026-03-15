use crate::RequiredExt;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "source_policy_ast.rs"]
mod source_policy_ast;
#[path = "source_policy_ci.rs"]
mod source_policy_ci;
#[path = "source_policy_mixers.rs"]
mod source_policy_mixers;
#[path = "source_policy_narrowing.rs"]
mod source_policy_narrowing;

use crate::test_support::RustRepository;
use source_policy_ast::{
    SourceFunction, SourceStruct, TestBlock, TestOnlyProductionMutator, c_source_functions,
    c_source_structs, rust_source_functions, rust_source_structs, rust_test_blocks,
    rust_test_only_production_mutators,
};
use source_policy_mixers::{c_inline_hash_mixers, rust_inline_hash_mixers};
use source_policy_narrowing::rust_wide_narrowing_violations;

const MAX_SOURCE_LINES_EXCLUSIVE: usize = 1000;
const MAX_STRUCT_FIELDS: usize = 24;
const SOURCE_POLICY_EXTENSIONS: &[&str] = &["rs", "c", "h"];
const EXCLUDED_SOURCE_DIRS: &[&str] = &[".git", "target", "vendor"];
const ALLOWED_IGNORED_TESTS: &[(&str, &str)] = &[
    (
        "exhaustive_all_5x5_patterns_reference_check",
        "exhaustive 5x5 sweep is expensive",
    ),
    (
        "hashlife_diagnostic_symmetry_gate_random_soup",
        "diagnostic symmetry-gate random benchmark",
    ),
    (
        "hashlife_diagnostic_symmetry_gate_structured_workload",
        "diagnostic symmetry-gate structured benchmark",
    ),
    (
        "hashlife_diagnostic_symmetry_gate_comparison_random_soup",
        "diagnostic symmetry-gate comparison benchmark",
    ),
    (
        "hashlife_diagnostic_symmetry_gate_comparison_structured",
        "diagnostic symmetry-gate comparison benchmark",
    ),
    (
        "hashlife_diagnostic_symmetry_gate_comparison_structured_light",
        "diagnostic symmetry-gate comparison benchmark",
    ),
    (
        "hashlife_diagnostic_symmetry_gate_promotion_candidate",
        "diagnostic symmetry-gate promotion benchmark",
    ),
    (
        "hashlife_diagnostic_medium_prime_jump_benchmark",
        "diagnostic benchmark",
    ),
    (
        "hashlife_gc_diagnostic_repeated_deep_runs",
        "diagnostic GC/runtime benchmark",
    ),
    (
        "hashlife_diagnostic_symmetric_mirror_reuse",
        "diagnostic symmetry benchmark",
    ),
    (
        "hashlife_diagnostic_step0_heavy_single_step",
        "diagnostic step0-heavy benchmark",
    ),
];

fn collect_source_files(root: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let mut entries = fs::read_dir(&dir)
            .unwrap_or_else(|err| {
                crate::invariant_failure!("failed to read {}: {err}", dir.display())
            })
            .map(|entry| entry.or_invariant("required value"))
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.path());

        for entry in entries.into_iter().rev() {
            let path = entry.path();
            if path.is_dir() {
                if !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| EXCLUDED_SOURCE_DIRS.contains(&name))
                {
                    dirs.push(path);
                }
            } else if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| extensions.contains(&ext))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn parse_source_functions(path: &Path, source: &str) -> Vec<SourceFunction> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("rs") => rust_source_functions(path, source),
        Some("c" | "h") => c_source_functions(path, source),
        extension => {
            crate::invariant_failure!("unsupported source policy extension: {extension:?}")
        }
    }
}

fn collect_source_functions(repo_root: &Path) -> Vec<SourceFunction> {
    let mut functions = Vec::new();
    for path in collect_source_files(repo_root, SOURCE_POLICY_EXTENSIONS) {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            crate::invariant_failure!("failed to read {}: {error}", path.display())
        });
        functions.extend(parse_source_functions(&path, &source));
    }
    functions
}

fn collect_source_structs(repo_root: &Path) -> Vec<SourceStruct> {
    let mut structs = Vec::new();
    for path in collect_source_files(repo_root, SOURCE_POLICY_EXTENSIONS) {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            crate::invariant_failure!("failed to read {}: {error}", path.display())
        });
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("rs") => structs.extend(rust_source_structs(&path, &source)),
            Some("c" | "h") => structs.extend(c_source_structs(&path, &source)),
            _ => {}
        }
    }
    structs
}

fn collect_test_only_production_mutators(repo_root: &Path) -> Vec<TestOnlyProductionMutator> {
    let mut mutators = Vec::new();
    for path in collect_source_files(repo_root, &["rs"]) {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            crate::invariant_failure!("failed to read {}: {error}", path.display())
        });
        mutators.extend(rust_test_only_production_mutators(&path, &source));
    }
    mutators
}

#[test]
fn source_policy_detects_wide_narrowing_without_comment_or_name_false_positives() {
    let fixture = r#"
        fn bad(bytes: u128, coordinate: i128, small: u8) {
            let a = bytes as usize;
            let b = i64::try_from(coordinate);
            let c = bytes.try_into();
            let d = small as u64;
            let harmless_name = "bytes as usize";
            // let ignored = coordinate as i64;
        }
    "#;
    let violations = rust_wide_narrowing_violations(Path::new("fixture.rs"), fixture);
    assert_eq!(
        violations
            .iter()
            .map(|violation| violation.target.as_str())
            .collect::<Vec<_>>(),
        ["usize", "i64", "untyped try_into", "u64"]
    );

    let boundary = r##"
        #[doc = "source-policy: checked-narrowing-boundary"]
        fn checked(bytes: u128) { let _ = usize::try_from(bytes); }
        fn unchecked(bytes: u128) { let _ = usize::try_from(bytes); }
    "##;
    let boundary_violations = rust_wide_narrowing_violations(Path::new("boundary.rs"), boundary);
    assert_eq!(
        boundary_violations
            .iter()
            .map(|violation| violation.function.as_str())
            .collect::<Vec<_>>(),
        ["unchecked"],
        "a checked-boundary marker must exempt only the annotated function"
    );
}

#[test]
fn source_policy_detects_inline_hash_mixers_without_comment_or_variable_false_positives() {
    let rust_source = r#"
        fn ad_hoc(mut value: u64) -> u64 {
            value = (value ^ (value >> 33)).wrapping_mul(0xff51afd7ed558ccd);
            value
        }
        fn xorshift_rng(mut state: u64) -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state
        }
        fn unrelated(mut value: u64) -> u64 {
            let wrapping_mul = "(value ^ (value >> 33)).wrapping_mul(7)";
            // value = (value ^ (value >> 33)).wrapping_mul(7);
            value = value.wrapping_mul(7);
            value ^ wrapping_mul.len() as u64
        }
    "#;
    let rust_mixers = rust_inline_hash_mixers(Path::new("fixture.rs"), rust_source);
    assert_eq!(
        rust_mixers
            .iter()
            .map(|mixer| mixer.function.as_str())
            .collect::<Vec<_>>(),
        ["ad_hoc"]
    );

    let c_source = r#"
        unsigned long ad_hoc(unsigned long value) {
            value = (value ^ (value >> 33)) * 0xff51afd7ed558ccdUL;
            return value;
        }
        unsigned long unrelated(unsigned long value) {
            /* value = (value ^ (value >> 33)) * 7; */
            return value * 7;
        }
    "#;
    let c_mixers = c_inline_hash_mixers(Path::new("fixture.c"), c_source);
    assert_eq!(
        c_mixers
            .iter()
            .map(|mixer| mixer.function.as_str())
            .collect::<Vec<_>>(),
        ["ad_hoc"]
    );
}

#[test]
fn source_policy_detects_rust_functions_methods_and_direct_recursion() {
    let source = r###"
        fn multiline_generic<T>(
            value: T,
        ) -> T where T: Clone {
            let note = r#"multiline_generic() and } are text"#;
            // multiline_generic(value);
            value
        }

        fn recurse(value: usize) -> usize {
            recurse(value - 1)
        }

        fn similarly_named() {
            recurse_later();
            module::similarly_named();
        }

        fn shadowed_local() {
            let shadowed_local = || {};
            shadowed_local();
        }

        fn shadowed_pattern() {
            let (shadowed_pattern,) = (|| {},);
            shadowed_pattern();
        }

        fn shadowed_parameter(shadowed_parameter: impl Fn()) {
            shadowed_parameter();
        }

        fn shadowed_closure_parameter() {
            let invoke = |shadowed_closure_parameter: fn()| shadowed_closure_parameter();
        }

        fn shadowed_match_arm(value: Option<fn()>) {
            match value {
                Some(shadowed_match_arm) => shadowed_match_arm(),
                None => {}
            }
        }

        fn shadowed_if_let(value: Option<fn()>) {
            if let Some(shadowed_if_let) = value {
                shadowed_if_let();
            }
        }

        fn shadowed_let_chain(value: Option<fn()>) {
            if let Some(shadowed_let_chain) = value
                && { shadowed_let_chain(); true }
            {}
        }

        fn shadowed_for_loop(values: Vec<fn()>) {
            for shadowed_for_loop in values {
                shadowed_for_loop();
            }
        }

        fn shadowed_nested_scope() {
            {
                let shadowed_nested_scope = || {};
                shadowed_nested_scope();
            }
        }

        fn call_before_shadowing() {
            call_before_shadowing();
            let call_before_shadowing = || {};
        }

        fn call_after_shadow_scope() {
            {
                let call_after_shadow_scope = || {};
                call_after_shadow_scope();
            }
            call_after_shadow_scope();
        }

        struct Walker;
        impl Walker {
            fn through_self<T>(
                &self,
            ) { self.through_self::<T>(); }
            fn through_self_type(&self) { Self::through_self_type(self); }
            fn through_concrete_type(&self) { Walker::through_concrete_type(self); }
            fn other_receivers(&self, other: &Walker) {
                other.other_receivers(self);
                self.other.other_receivers(self);
                module::other_receivers();
            }
            fn bare_free_function(&self) { bare_free_function(); }
        }

        trait Step {
            fn through_qself(&self);
            fn through_concrete_trait(&self);
            fn through_trait_path(&self);
        }
        impl Step for Walker {
            fn through_qself(&self) {
                <Self as Step>::through_qself(self);
            }
            fn through_concrete_trait(&self) {
                <Walker as Step>::through_concrete_trait(self);
            }
            fn through_trait_path(&self) {
                Step::through_trait_path(self);
            }
        }


        mod a { pub trait SameName { fn other_trait_path(&self); } }
        mod b { pub trait SameName { fn other_trait_path(&self); } }
        impl a::SameName for Walker {
            fn other_trait_path(&self) {
                b::SameName::other_trait_path(self);
            }
        }
    "###;
    let functions = rust_source_functions(Path::new("fixture.rs"), source);
    let recursive = functions
        .iter()
        .filter(|function| function.directly_recursive)
        .map(|function| function.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        recursive,
        [
            "recurse",
            "call_before_shadowing",
            "call_after_shadow_scope",
            "through_self",
            "through_self_type",
            "through_concrete_type",
            "through_qself",
            "through_concrete_trait",
            "through_trait_path"
        ]
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "multiline_generic")
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "similarly_named")
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "other_receivers")
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "bare_free_function" && !function.directly_recursive)
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "other_trait_path" && !function.directly_recursive)
    );
}

#[test]
fn source_policy_detects_c_functions_and_only_direct_symbol_calls() {
    let source = r#"
        int declared_only(int value);
        static int (*callback)(int);

        static int multiline(
            int value,
            int (*transform)(int)
        ) {
            const char *text = "multiline(value) and } are text";
            /* multiline(value); */
            callback(value);
            transform(value);
            multiline_later(value);
            object.multiline(value);
            return value;
        }

        static int recurse(int value) {
            return recurse(value - 1);
        }

        static int pointer_shadow(int value) {
            int (*pointer_shadow)(int) = callback;
            return pointer_shadow(value);
        }
    "#;
    let functions = c_source_functions(Path::new("fixture.h"), source);

    assert_eq!(functions.len(), 3);
    assert!(
        functions
            .iter()
            .any(|function| function.name == "multiline" && !function.directly_recursive)
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "recurse" && function.directly_recursive)
    );
    assert!(
        functions
            .iter()
            .any(|function| function.name == "pointer_shadow" && !function.directly_recursive)
    );
}

fn collect_test_blocks(repo_root: &Path) -> Vec<TestBlock> {
    let mut tests = Vec::new();
    for path in collect_source_files(repo_root, &["rs"]) {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            crate::invariant_failure!("failed to read {}: {error}", path.display())
        });
        tests.extend(rust_test_blocks(&path, &source));
    }
    tests
}

#[test]
fn source_policy_parses_ignored_test_reasons_from_attributes() {
    let tests = rust_test_blocks(
        Path::new("fixture.rs"),
        r#"
            #[test]
            #[ignore = "diagnostic benchmark"]
            fn diagnostic() {}

            #[test]
            #[ignore]
            fn missing_reason() {}

            #[test]
            fn active() {}
        "#,
    );
    assert_eq!(
        tests[0].ignored_reason.as_deref(),
        Some("diagnostic benchmark")
    );
    assert_eq!(tests[1].ignored_reason.as_deref(), Some(""));
    assert_eq!(tests[2].ignored_reason, None);
}

#[test]
fn ignored_tests_are_explicitly_allowlisted_as_non_release_diagnostics() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ignored = collect_test_blocks(repo_root)
        .into_iter()
        .filter_map(|test| {
            test.ignored_reason
                .map(|reason| (test.name, reason, test.path))
        })
        .collect::<Vec<_>>();

    let offenders = ignored
        .iter()
        .filter(|(name, reason, _)| {
            !ALLOWED_IGNORED_TESTS
                .iter()
                .any(|allowed| allowed.0 == name && allowed.1 == reason)
        })
        .map(|(name, reason, path)| {
            format!(
                "{}: `{name}` is ignored without a non-release allowlist entry (reason: {reason:?})",
                path.strip_prefix(repo_root)
                    .or_invariant("required value")
                    .display()
            )
        })
        .collect::<Vec<_>>();
    let missing = ALLOWED_IGNORED_TESTS
        .iter()
        .filter(|allowed| {
            !ignored
                .iter()
                .any(|(name, reason, _)| name == allowed.0 && reason == allowed.1)
        })
        .map(|allowed| format!("allowlisted ignored test `{}` no longer exists", allowed.0))
        .collect::<Vec<_>>();

    assert!(
        offenders.is_empty() && missing.is_empty(),
        "Ignored correctness, allocation, SIMD, Life, and differential tests are forbidden:\n{}\n{}",
        offenders.join("\n"),
        missing.join("\n")
    );
}

#[test]
fn source_policy_detects_struct_sizes_and_test_only_production_mutators() {
    let source = r#"
        struct ExactlyTwentyFour {
            f01: u8, f02: u8, f03: u8, f04: u8, f05: u8, f06: u8,
            f07: u8, f08: u8, f09: u8, f10: u8, f11: u8, f12: u8,
            f13: u8, f14: u8, f15: u8, f16: u8, f17: u8, f18: u8,
            f19: u8, f20: u8, f21: u8, f22: u8, f23: u8, f24: u8,
        }
        struct TwentyFive(
            u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
            u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
            u8, u8, u8, u8, u8,
        );
        struct Production;
        impl Production {
            #[cfg(test)]
            fn mutate_for_test(&mut self) { self.value = 1; }
            #[cfg(test)]
            fn inspect_for_test(&self) {}
            #[cfg(test)]
            fn local_only(&mut self) {
                let mutation_text = "self.value = 2;";
                // self.value = 3;
                consume(mutation_text);
            }
        }
        #[cfg(test)]
        impl Production {
            fn call_mutator_for_test(&mut self) { self.production_call(); }
        }
        #[cfg(test)]
        struct Fixture;
        #[cfg(test)]
        impl Fixture {
            fn mutate_fixture(&mut self) {}
        }
    "#;
    let path = Path::new("fixture.rs");
    let structs = rust_source_structs(path, source);
    let mutators = rust_test_only_production_mutators(path, source);

    assert_eq!(
        structs
            .iter()
            .find(|item| item.name == "ExactlyTwentyFour")
            .map(|item| item.field_count),
        Some(24)
    );
    assert_eq!(
        structs
            .iter()
            .find(|item| item.name == "TwentyFive")
            .map(|item| item.field_count),
        Some(25)
    );
    let mut methods = mutators
        .iter()
        .map(|mutator| mutator.method.as_str())
        .collect::<Vec<_>>();
    methods.sort_unstable();
    assert_eq!(methods, ["mutate_for_test"]);

    let c_source = r#"
        struct TwentyFourFields {
            int f01, f02, f03, f04, f05, f06;
            int f07, f08, f09, f10, f11, f12;
            int f13, f14, f15, f16, f17, f18;
            int f19, f20, f21, f22, f23, f24;
        };
        struct TwentyFiveFields {
            int f01; int f02; int f03; int f04; int f05;
            int f06; int f07; int f08; int f09; int f10;
            int f11; int f12; int f13; int f14; int f15;
            int f16; int f17; int f18; int f19; int f20;
            int f21; int f22; int f23; int f24; int f25;
        };
    "#;
    let c_structs = c_source_structs(Path::new("fixture.c"), c_source);
    assert_eq!(
        c_structs
            .iter()
            .find(|item| item.name == "TwentyFourFields")
            .map(|item| item.field_count),
        Some(24),
        "C policy must count every declarator, not only declaration statements"
    );
    assert_eq!(
        c_structs
            .iter()
            .find(|item| item.name == "TwentyFiveFields")
            .map(|item| item.field_count),
        Some(25)
    );
}

#[test]
fn source_files_stay_under_1000_lines() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sources = collect_source_files(repo_root, SOURCE_POLICY_EXTENSIONS);
    let mut offenders = Vec::new();

    for path in sources {
        let contents = fs::read_to_string(&path).unwrap_or_else(|err| {
            crate::invariant_failure!("failed to read {}: {err}", path.display())
        });
        let line_count = contents.lines().count();
        if line_count >= MAX_SOURCE_LINES_EXCLUSIVE {
            offenders.push(format!(
                "{} has {} lines",
                path.strip_prefix(repo_root)
                    .or_invariant("required value")
                    .display(),
                line_count
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "Rust and C source files must stay under {} lines:\n{}",
        MAX_SOURCE_LINES_EXCLUSIVE,
        offenders.join("\n")
    );
}

#[test]
fn hashlife_embedding_iterates_borrowed_chunks_without_materializing_live_cells() {
    let repository = RustRepository::discover(Path::new(env!("CARGO_MANIFEST_DIR")))
        .or_invariant("parsed repository Rust sources");
    let source = repository
        .source_containing_callable("try_embed_grid_state")
        .or_invariant("unique HashLife grid embedding source");
    assert_eq!(
        source.method_call_count("live_cells"),
        0,
        "HashLife embedding must borrow occupied storage instead of allocating a full live-cell copy"
    );
    assert!(
        source.method_call_count("occupied_chunks") > 0,
        "HashLife embedding no longer uses the borrowed occupied-chunk API"
    );
}

#[test]
fn source_files_do_not_use_direct_recursion() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let offenders = collect_source_functions(repo_root)
        .into_iter()
        .filter(|function| function.directly_recursive)
        .map(|function| {
            let display_path = function
                .path
                .strip_prefix(repo_root)
                .or_invariant("required value")
                .display();
            format!(
                "{display_path}: `{}` contains a direct self-call",
                function.name
            )
        })
        .collect::<Vec<_>>();

    assert!(
        offenders.is_empty(),
        "Source files must not use direct recursion:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn source_functions_do_not_define_inline_hash_mixers() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offenders = Vec::new();
    for path in collect_source_files(repo_root, SOURCE_POLICY_EXTENSIONS) {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            crate::invariant_failure!("failed to read {}: {error}", path.display())
        });
        let mixers = match path.extension().and_then(|extension| extension.to_str()) {
            Some("rs") => rust_inline_hash_mixers(&path, &source),
            Some("c" | "h") => c_inline_hash_mixers(&path, &source),
            _ => Vec::new(),
        };
        offenders.extend(mixers.into_iter().map(|mixer| {
            let display_path = mixer
                .path
                .strip_prefix(repo_root)
                .or_invariant("required value")
                .display();
            format!(
                "{display_path}: `{}` contains an inline hash mixer",
                mixer.function
            )
        }));
    }

    assert!(
        offenders.is_empty(),
        "Use crate::hashing::mix64 instead of defining XOR-shift/multiply finalizers:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn source_structs_have_at_most_24_fields() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let offenders = collect_source_structs(repo_root)
        .into_iter()
        .filter(|item| item.field_count > MAX_STRUCT_FIELDS)
        .map(|item| {
            let display_path = item
                .path
                .strip_prefix(repo_root)
                .or_invariant("required value")
                .display();
            format!(
                "{display_path}: `{}` has {} fields",
                item.name, item.field_count
            )
        })
        .collect::<Vec<_>>();

    assert!(
        offenders.is_empty(),
        "Source structs must have at most {MAX_STRUCT_FIELDS} fields; split by responsibility:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn production_types_do_not_expose_test_only_mutators() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let offenders = collect_test_only_production_mutators(repo_root)
        .into_iter()
        .map(|item| {
            let display_path = item
                .path
                .strip_prefix(repo_root)
                .or_invariant("required value")
                .display();
            format!(
                "{display_path}: `{}::{}` is a test-only mutable inherent method",
                item.owner, item.method
            )
        })
        .collect::<Vec<_>>();

    assert!(
        offenders.is_empty(),
        "Exercise production behavior or use a test-only fixture type instead of mutating production types:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn wide_values_narrow_only_in_checked_boundary_functions() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offenders = Vec::new();
    for path in collect_source_files(repo_root, &["rs"]) {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            crate::invariant_failure!("failed to read {}: {error}", path.display())
        });
        offenders.extend(
            rust_wide_narrowing_violations(&path, &source)
                .into_iter()
                .map(|violation| {
                    let display_path = violation
                        .path
                        .strip_prefix(repo_root)
                        .or_invariant("required value")
                        .display();
                    format!(
                        "{display_path}: `{}` uses a forbidden integer narrowing to {}",
                        violation.function, violation.target
                    )
                }),
        );
    }
    assert!(
        offenders.is_empty(),
        "Use explicit checked/infallible conversions, or a narrowly marked checked-boundary function:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn test_names_do_not_drift_from_high_risk_invariants() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let tests = collect_test_blocks(repo_root);
    let mut offenders = Vec::new();

    for test in tests {
        let display_path = test
            .path
            .strip_prefix(repo_root)
            .or_invariant("required value")
            .display();
        let facts = &test.facts;
        let name = test.name.as_str();

        if name.contains("not_summarized")
            && !facts.uses_bf_ir_variant("Loop")
            && ["Affine", "Shift", "Distribute", "Square", "MulAdd"]
                .iter()
                .any(|variant| facts.uses_bf_ir_variant(variant))
        {
            offenders.push(format!(
                "{display_path}: `{name}` claims not_summarized but expects non-loop richer IR"
            ));
        }

        if name.contains("stays_loop") && !facts.uses_bf_ir_variant("Loop") {
            offenders.push(format!(
                "{display_path}: `{name}` claims stays_loop but does not assert a guarded loop"
            ));
        }

        if name.contains("summarizes_to_distribute")
            && !facts.uses_bf_ir_variant("Distribute")
            && ["Affine", "Shift"]
                .iter()
                .any(|variant| facts.uses_bf_ir_variant(variant))
        {
            offenders.push(format!(
                "{display_path}: `{name}` claims summarize-to-distribute but asserts canonical single-target lowering"
            ));
        }

        if facts.has_bare_oracle_verify_assertion() {
            offenders.push(format!(
                "{display_path}: `{name}` uses a bare verify_* assertion without probe context"
            ));
        }

        if name.contains("render") && facts.calls_contains() && !facts.has_diagnostic_literal() {
            offenders.push(format!(
                "{display_path}: `{name}` checks rendered output without printing the frame/diff on failure"
            ));
        }

        if facts.has_normalize_assert_eq() {
            offenders.push(format!(
                "{display_path}: `{name}` compares normalized grids without shared context-rich messaging"
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "High-risk test name/invariant drift detected:\n{}",
        offenders.join("\n")
    );
}
