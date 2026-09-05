use super::*;
use crate::RequiredExt;

pub(super) struct TestRoot(pub(super) PathBuf);
impl TestRoot {
    pub(super) fn new() -> Self {
        Self(std::env::temp_dir().join(format!("c-harness-test-{}", storage::unique())))
    }
}
impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(windows)]
#[test]
fn windows_compiler_launch_path_preserves_helper_relocation() {
    use std::path::{Component, Prefix};
    let tool = toolchain::Toolchain::discover().or_invariant("Windows compiler");
    assert!(tool.compiler.is_absolute());
    assert!(
        !matches!(tool.compiler.components().next(), Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Verbatim(_) | Prefix::VerbatimDisk(_) | Prefix::VerbatimUNC(_, _))),
        "compiler argv[0] must not use extended-length syntax: {:?}",
        tool.compiler
    );
    let root = TestRoot::new();
    let output = run_in(
        &root.0,
        "#include <stdio.h>\nint main(void) { puts(\"frontend and linker work\"); return 0; }",
        &[],
        &mut accounting::Request::new(),
    )
    .or_invariant("Windows cc1, assembler, linker and execution");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "frontend and linker work"
    );
}

#[test]
fn concurrent_requests_preserve_execution_and_only_verified_compilation_reuse() {
    let root = TestRoot::new();
    let counts = std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..4 {
            let path = &root.0;
            threads.push(scope.spawn(move || {
                let mut counts = accounting::Request::new();
                let result = run_in(path, "int main(void) { return 0; }", &[], &mut counts)
                    .or_invariant("concurrent compile request");
                assert!(result.status.success());
                (counts.compiler_invocations, counts.execution_attempts)
            }));
        }
        threads
            .into_iter()
            .map(|thread| thread.join().or_invariant("request thread"))
            .collect::<Vec<_>>()
    });
    assert_eq!(
        counts.iter().map(|count| count.0).sum::<u64>(),
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            1
        } else {
            4
        },
        "verified reuse or conservative fallback violated: {counts:?}"
    );
    assert_eq!(
        counts.iter().map(|count| count.1).sum::<u64>(),
        4,
        "reuse skipped runtime execution: {counts:?}"
    );
}

#[cfg(unix)]
#[test]
fn killed_producer_releases_advisory_lock_for_a_new_builder() {
    let root = TestRoot::new();
    let work = storage::Workspace::create(&root.0).or_invariant("workspace");
    let lock_path = root.0.join("producer.lock");
    let ready = work.path.join("ready");
    std::thread::scope(|scope| {
        let process = scope.spawn(|| {
            let mut command = Command::new("python3");
            command.args(["-c", "import fcntl,sys,time; f=open(sys.argv[1],'w'); fcntl.flock(f,fcntl.LOCK_EX); open(sys.argv[2],'w').close(); time.sleep(30)"]).arg(&lock_path).arg(&ready);
            process::run(&mut command, &work.path, "producer", Duration::from_secs(2), 1024)
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while !ready.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            ready.exists(),
            "child did not acquire the lock before its deadline"
        );
        assert!(
            storage::Lock::try_exclusive(&lock_path)
                .or_invariant("lock probe")
                .is_none()
        );
        assert!(matches!(
            process.join().or_invariant("producer thread"),
            Err(process::ProcessFailure::Timeout { .. })
        ));
    });
    assert!(
        storage::Lock::try_exclusive(&lock_path)
            .or_invariant("lock after producer death")
            .is_some(),
        "crashed producer stranded the build key"
    );
}

#[test]
fn each_request_executes_and_only_verified_toolchains_reuse_compilation() {
    // Test exact execution bytes without depending on CRT text-mode newlines.
    let code =
        "#include <stdio.h>\nint main(void) { fputs(\"fresh execution\", stdout); return 0; }\n";
    let root = TestRoot::new();
    let mut first = accounting::Request::new();
    let mut second = accounting::Request::new();
    for counts in [&mut first, &mut second] {
        let output = run_in(&root.0, code, &[], counts).or_invariant("compile and execute");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"fresh execution");
        assert_eq!(
            counts.execution_attempts, 1,
            "a cache hit skipped execution"
        );
    }
    assert_eq!(first.compiler_invocations, 1);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    assert_eq!(
        second.compiler_invocations, 0,
        "supported sealed platform toolchain must reuse compilation"
    );
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    assert_eq!(
        second.compiler_invocations, 1,
        "a toolchain without a verified dependency provider must compile afresh"
    );
    let mut uncached = accounting::Request::new();
    let output = run_controlled(&root.0, code, &[], &mut uncached, false)
        .or_invariant("explicit cache bypass");
    assert!(output.status.success());
    assert_eq!(output.stdout, b"fresh execution");
    assert_eq!(uncached.compiler_invocations, 1);
    assert_eq!(uncached.execution_attempts, 1);
}

#[test]
fn compile_modes_and_fuel_defines_are_distinct_specs() {
    assert_ne!(
        digest(&serde_json::to_vec(&CompileSpec::strict(&[])).or_invariant("spec")),
        digest(&serde_json::to_vec(&CompileSpec::strict(&["-O3"])).or_invariant("spec"))
    );
    let spec = CompileSpec::strict(&["-DBF_TEST_WORK_LIMIT=10"]);
    assert_eq!(
        spec.arguments
            .iter()
            .filter(|arg| arg.starts_with("-DBF_TEST_WORK_LIMIT="))
            .count(),
        1
    );
}

#[test]
fn cleanup_handles_malformed_names_and_concurrent_workspace_retirement() {
    let root = TestRoot::new();
    fs::create_dir_all(&root.0).or_invariant("cache root");
    for name in ["entry-", "entry-x", "entry-\u{03bb}"] {
        fs::create_dir(root.0.join(name)).or_invariant("malformed entry");
    }
    {
        let _admission = storage::Lock::acquire(&root.0.join("admission.lock"), false)
            .or_invariant("cleanup coordination");
        storage::clean(&root.0).or_invariant("malformed cache cleanup");
    }
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..24 {
                    storage::Workspace::create(&root.0)
                        .or_invariant("concurrent workspace admission")
                        .finish()
                        .or_invariant("concurrent workspace retirement");
                }
            });
        }
    });
    assert!(
        !fs::read_dir(&root.0).or_invariant("cache").any(|entry| {
            entry
                .or_invariant("cache entry")
                .file_type()
                .or_invariant("entry type")
                .is_dir()
        }),
        "retirement left a recreated owner lock or abandoned directory"
    );
}

#[test]
fn impossible_admission_does_not_evict_useful_compilation_artifacts() {
    let root = TestRoot::new();
    fs::create_dir_all(&root.0).or_invariant("cache root");
    let entry = root.0.join(format!("entry-{}", digest(b"retained")));
    fs::create_dir(&entry).or_invariant("retained entry");
    fs::write(entry.join("artifact"), b"useful").or_invariant("retained artifact");
    let held = fs::File::create(root.0.join("non-evictable-capacity"))
        .or_invariant("sparse capacity fixture");
    held.set_len(storage::CACHE_LIMIT)
        .or_invariant("logical capacity");
    assert!(storage::collect(&root.0, 1).is_err());
    assert!(
        entry.exists(),
        "eviction destroyed useful work without making admission possible"
    );
}

#[test]
fn failed_compilation_cannot_publish_a_successful_artifact() {
    let root = TestRoot::new();
    for _ in 0..2 {
        let mut counts = accounting::Request::new();
        assert!(
            run_in(
                &root.0,
                "int main(void) { undefined_name; }",
                &[],
                &mut counts
            )
            .is_err()
        );
        assert_eq!(counts.execution_attempts, 0);
        assert_eq!(counts.outcome, Some("request_failure"));
    }
    assert!(!fs::read_dir(&root.0).or_invariant("cache").any(|entry| {
        entry
            .or_invariant("entry")
            .file_name()
            .to_string_lossy()
            .starts_with("entry-")
    }));
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn corrupted_artifacts_rebuild_and_lease_survives_eviction() {
    let root = TestRoot::new();
    let source = "int main(void) { return 0; }";
    run_in(&root.0, source, &[], &mut accounting::Request::new()).or_invariant("cold compilation");
    let entry = fs::read_dir(&root.0)
        .or_invariant("cache")
        .map(|entry| entry.or_invariant("entry"))
        .find(|entry| entry.file_name().to_string_lossy().starts_with("entry-"))
        .or_invariant("published entry")
        .path();
    let key = entry
        .file_name()
        .or_invariant("entry name")
        .to_str()
        .or_invariant("key")
        .strip_prefix("entry-")
        .or_invariant("prefix");
    let lease =
        storage::Lock::acquire(&storage::stripe(&root.0, "lease", key), true).or_invariant("lease");
    assert!(storage::collect(&root.0, storage::CACHE_LIMIT).is_err());
    assert!(entry.exists(), "eviction removed a leased artifact");
    drop(lease);
    fs::write(entry.join("program.bin"), b"corrupted").or_invariant("corrupt fixture");
    let mut counts = accounting::Request::new();
    let output = run_in(&root.0, source, &[], &mut counts).or_invariant("rebuild corruption");
    assert!(output.status.success());
    assert_eq!(counts.compiler_invocations, 1);
    assert_eq!(counts.execution_attempts, 1);
}

#[cfg(unix)]
#[test]
fn output_flood_and_descendant_timeout_are_failures_not_truncated_successes() {
    let root = TestRoot::new();
    let work = storage::Workspace::create(&root.0).or_invariant("workspace");
    let mut flood = Command::new("/bin/sh");
    flood.args(["-c", "while :; do printf '0123456789abcdef'; done"]);
    assert!(matches!(
        process::run(
            &mut flood,
            &work.path,
            "flood",
            Duration::from_secs(5),
            1024
        ),
        Err(process::ProcessFailure::OutputLimitExceeded { .. })
    ));
    let mut descendants = Command::new("/bin/sh");
    descendants.args(["-c", "sleep 30 & wait"]);
    let start = Instant::now();
    assert!(matches!(
        process::run(
            &mut descendants,
            &work.path,
            "timeout",
            Duration::from_millis(100),
            1024
        ),
        Err(process::ProcessFailure::Timeout { .. })
    ));
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "descendant kept output readers or cleanup blocked"
    );
    work.finish().or_invariant("cleanup after process failure");
}

#[test]
fn unusable_output_spools_are_rejected_before_process_creation() {
    for stream in ["stdout", "stderr"] {
        let root = TestRoot::new();
        let work = storage::Workspace::create(&root.0).or_invariant("workspace");
        let path = work.path.join(format!("setup.{stream}"));
        fs::create_dir(&path).or_invariant("failed spool fixture");
        let expected = fs::File::create(&path).err().or_invariant("spool error");
        assert_ne!(expected.kind(), std::io::ErrorKind::NotFound);
        // A spawn-first implementation returns NotFound instead. This checks
        // ordering directly, without a wall-clock assertion on a loaded runner.
        let mut command = Command::new(work.path.join("nonexistent-executable"));
        let result = process::run(
            &mut command,
            &work.path,
            "setup",
            Duration::from_secs(30),
            1024,
        );
        assert!(matches!(result, Err(process::ProcessFailure::Io(error))
            if error.kind() == expected.kind() && error.raw_os_error() == expected.raw_os_error()));
        work.finish().or_invariant("cleanup after setup failure");
    }
}

#[test]
fn incomplete_native_identity_disables_persistence_and_environment_is_controlled() {
    let root = TestRoot::new();
    let work = storage::Workspace::create(&root.0).or_invariant("workspace");
    let tool = toolchain::Toolchain::discover().or_invariant("compiler");
    for variable in [
        "CPATH",
        "LIBRARY_PATH",
        "GCC_EXEC_PREFIX",
        "COMPILER_PATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ] {
        assert!(
            !tool
                .environment
                .contains_key(&std::ffi::OsString::from(variable)),
            "ambient compilation override retained: {variable}"
        );
    }
    let identity = tool
        .fingerprint(&root.0, &work.path, &["-march=native".into()])
        .or_invariant("native identity");
    assert!(
        !identity.persistent,
        "unresolved effective native features authorized reuse"
    );
}
