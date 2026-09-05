//! Generated-C tests reuse compilation, never execution or assertions.
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

mod accounting;
mod cache;
mod process;
mod storage;
#[cfg(test)]
mod tests;
mod toolchain;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const SCHEMA: u32 = 1;

#[derive(Clone, Debug, Serialize)]
pub struct CompileSpec {
    pub arguments: Vec<String>,
}
impl CompileSpec {
    pub fn strict(extra: &[&str]) -> Self {
        let mut arguments: Vec<String> = [
            "-std=c2x",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
        ]
        .map(String::from)
        .into();
        if !extra
            .iter()
            .any(|arg| arg.starts_with("-DBF_TEST_WORK_LIMIT="))
        {
            arguments.push("-DBF_TEST_WORK_LIMIT=100000000".into());
        }
        arguments.extend(extra.iter().map(|s| (*s).to_string()));
        Self { arguments }
    }
}

fn digest(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("ASIMPLELIFE_C_TEST_CACHE_DIR") {
        return path.into();
    }
    // Locate Cargo's target tree from the actual test artifact, including custom
    // target directories and target triples; do not use a handwritten target path.
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent()?.parent()?.parent().map(Path::to_path_buf))
        .unwrap_or_else(std::env::temp_dir)
        .join("compiled-c-tests")
}

pub fn compile_and_run(source: &str, extra: &[&str]) -> Result<Output> {
    let root = cache_root();
    fs::create_dir_all(&root)?;
    if std::env::var_os("ASIMPLELIFE_C_TEST_CACHE_CLEAN").is_some() {
        static CLEANED: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<PathBuf>>> =
            std::sync::OnceLock::new();
        let mut cleaned = CLEANED
            .get_or_init(std::sync::Mutex::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cleaned.insert(root.clone()) {
            let _admission = storage::Lock::acquire(&root.join("admission.lock"), false)?;
            storage::clean(&root)?;
        }
    }
    let mut counts = accounting::Request::new();
    run_controlled(
        &root,
        source,
        extra,
        &mut counts,
        std::env::var_os("ASIMPLELIFE_C_TEST_CACHE_DISABLE").is_none(),
    )
}

fn run_in(
    root: &Path,
    source: &str,
    extra: &[&str],
    counts: &mut accounting::Request,
) -> Result<Output> {
    run_controlled(root, source, extra, counts, true)
}

fn run_controlled(
    root: &Path,
    source: &str,
    extra: &[&str],
    counts: &mut accounting::Request,
    reuse: bool,
) -> Result<Output> {
    let result = (|| {
        let work = storage::Workspace::create(root)?;
        let spec = CompileSpec::strict(extra);
        let toolchain = toolchain::Toolchain::discover()?;
        let started = Instant::now();
        let fingerprint = toolchain.fingerprint(root, &work.path, &spec.arguments)?;
        counts.time("fingerprint", started.elapsed());
        #[cfg(target_os = "linux")]
        let spec = {
            let mut spec = spec;
            if fingerprint.persistent {
                // Preserve __FILE__ and line markers without GCC's random CWD debug marker.
                spec.arguments.push("-fno-working-directory".into());
            }
            spec
        };
        fs::write(work.path.join("program.c"), source)?;
        let started = Instant::now();
        let preprocessed = {
            let _permit = storage::Permit::acquire(root, true, false)?;
            let mut command = toolchain.command(&toolchain.compiler);
            command
                .current_dir(&work.path)
                .args(&spec.arguments)
                .args(["-E", "program.c"]);
            process::run(
                &mut command,
                &work.path,
                "preprocess",
                Duration::from_secs(120),
                64 * 1024 * 1024,
            )?
        };
        counts.time("preprocess", started.elapsed());
        if !preprocessed.status.success() {
            return Err(format!(
                "preprocessing failed compiler={:?} arguments={:?}: {}",
                toolchain.compiler,
                spec.arguments,
                String::from_utf8_lossy(&preprocessed.stderr)
            )
            .into());
        }
        fs::write(work.path.join("program.i"), &preprocessed.stdout)?;
        let key = digest(&serde_json::to_vec(&(
            SCHEMA,
            &spec,
            &fingerprint,
            digest(source.as_bytes()),
            digest(&preprocessed.stdout),
        ))?);
        drop(preprocessed);
        let persistent = fingerprint.persistent && reuse;
        let executable = work.path.join(if cfg!(windows) {
            "execute.exe"
        } else {
            "execute.bin"
        });
        let artifact = cache::ArtifactRequest {
            root,
            key: &key,
            fingerprint: &fingerprint,
            destination: &executable,
        };
        let hit = if persistent {
            artifact.copy_hit()?
        } else {
            false
        };
        if hit {
            counts.outcome("direct_hit");
        } else {
            let _build = if persistent {
                Some(storage::Lock::acquire(
                    &storage::stripe(root, "build", &key),
                    false,
                )?)
            } else {
                None
            };
            if persistent && artifact.copy_hit()? {
                counts.outcome("coalesced_hit");
            } else {
                let started = Instant::now();
                #[cfg(target_os = "linux")]
                let mut instrumented = false;
                let result = {
                    let _permit = storage::Permit::acquire(root, true, false)?;
                    let mut command = toolchain.command(&toolchain.compiler);
                    command
                        .current_dir(&work.path)
                        .args(&spec.arguments)
                        .args(["-x", "cpp-output", "program.i", "-o"])
                        .arg(&executable);
                    #[cfg(target_os = "linux")]
                    if persistent {
                        instrumented = toolchain::linux::instrument(&mut command, &work.path)?;
                    }
                    counts.compiler_started();
                    process::run(
                        &mut command,
                        &work.path,
                        "compile",
                        Duration::from_secs(120),
                        16 * 1024 * 1024,
                    )
                };
                counts.time("codegen_link", started.elapsed());
                let compiled = match result {
                    Ok(output) if output.status.success() => {
                        counts.compiler_finished("success");
                        output
                    }
                    Ok(output) => {
                        counts.compiler_finished("failure");
                        return Err(format!(
                            "cc failed key={key} arguments={:?}: {}",
                            spec.arguments,
                            String::from_utf8_lossy(&output.stderr)
                        )
                        .into());
                    }
                    Err(error) => {
                        counts.compiler_finished("interrupted");
                        return Err(error.into());
                    }
                };
                work.check_size()?;
                cache::validate_executable(&executable, &fingerprint.target)?;
                // Optional cache publication cannot change the program result.
                #[cfg(target_os = "linux")]
                let publication_safe = if persistent && instrumented {
                    match toolchain::linux::audit(
                        &toolchain,
                        &fingerprint,
                        root,
                        &work.path,
                        &executable,
                        &compiled.stdout,
                    ) {
                        Ok(()) => true,
                        Err(error) => {
                            counts.note(&format!("Linux dependency audit bypass: {error}"));
                            false
                        }
                    }
                } else {
                    false
                };
                #[cfg(not(target_os = "linux"))]
                let publication_safe = persistent;
                drop(compiled);
                if publication_safe {
                    let verified = toolchain.fingerprint(root, &work.path, &spec.arguments)?;
                    if verified.identity == fingerprint.identity
                        && verified.persistent
                        && let Err(error) = artifact.publish(&executable)
                    {
                        counts.note(&format!("cache publication bypass: {error}"));
                    }
                }
                counts.outcome("compiled_success");
            }
        }
        // Build/lease locks are gone before execution-resource admission.
        let started = Instant::now();
        let sanitizer = spec
            .arguments
            .iter()
            .any(|arg| arg.starts_with("-fsanitize="));
        let output = {
            let _permit = storage::Permit::acquire(root, false, sanitizer)?;
            let mut command = toolchain.command(&executable);
            command
                .current_dir(&work.path)
                .env(
                    "ASAN_OPTIONS",
                    "detect_leaks=0:halt_on_error=1:abort_on_error=1",
                )
                .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1");
            counts.execution_started();
            process::run(
                &mut command,
                &work.path,
                "execute",
                Duration::from_secs(60),
                16 * 1024 * 1024,
            )?
        };
        counts.time("execution", started.elapsed());
        work.check_size()?;
        work.finish()?;
        Ok(output)
    })();
    if result.is_err() {
        counts.failed();
    }
    result
}
