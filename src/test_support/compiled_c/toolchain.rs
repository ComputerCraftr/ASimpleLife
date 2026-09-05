//! Conservative identities for the closed generated-C compile surface.
use super::*;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

#[cfg(target_os = "linux")]
pub(super) mod linux;

thread_local! {
    static INPUTS: std::cell::RefCell<Option<BTreeMap<PathBuf, String>>> = const { std::cell::RefCell::new(None) };
    static HASHED_INPUTS: std::cell::RefCell<std::collections::BTreeSet<PathBuf>> = const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
}

fn record(path: &Path) -> Result<()> {
    INPUTS.with(|inputs| -> Result<()> {
        if let Some(inputs) = &mut *inputs.borrow_mut() {
            inputs.insert(
                path.to_path_buf(),
                file_identity(&fs::symlink_metadata(path)?),
            );
        }
        Ok(())
    })
}

struct InputRecording;
impl Drop for InputRecording {
    fn drop(&mut self) {
        INPUTS.with(|inputs| *inputs.borrow_mut() = None);
        HASHED_INPUTS.with(|inputs| inputs.borrow_mut().clear());
    }
}

#[derive(Clone)]
struct Snapshot {
    fingerprint: ToolchainFingerprint,
    inputs: BTreeMap<PathBuf, String>,
}

fn known_arguments(args: &[String]) -> bool {
    args.iter().all(|arg| {
        matches!(
            arg.as_str(),
            "-std=c2x"
                | "-O0"
                | "-O3"
                | "-Wall"
                | "-Wextra"
                | "-Wpedantic"
                | "-Werror"
                | "-g3"
                | "-fno-working-directory"
                | "-fno-omit-frame-pointer"
                | "-pedantic-errors"
                | "-fno-sanitize-recover=all"
                | "-fsanitize=address,undefined"
                | "-fsanitize=undefined"
                | "-march=x86-64"
                | "-mno-avx2"
        ) || arg.starts_with("-DBF_")
    })
}

fn file_identity(metadata: &fs::Metadata) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}",
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mode(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec()
        )
    }
    #[cfg(not(unix))]
    {
        format!(
            "{}:{:?}:{:?}:{:?}",
            metadata.len(),
            metadata.modified(),
            metadata.created(),
            metadata.permissions()
        )
    }
}

fn settled_metadata(metadata: &fs::Metadata) -> bool {
    // Overlay filesystems may timestamp multiple writes in the same clock tick.
    // Never memoize a digest/discovery while its observed tick is still recent.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.ctime() < 0 || metadata.ctime_nsec() < 0 {
            return false;
        }
        let changed = std::time::UNIX_EPOCH
            .checked_add(Duration::from_secs(metadata.ctime().unsigned_abs()))
            .and_then(|time| {
                time.checked_add(Duration::from_nanos(metadata.ctime_nsec().unsigned_abs()))
            });
        let now = std::time::SystemTime::now();
        let (Some(changed), Ok(modified)) = (changed, metadata.modified()) else {
            return false;
        };
        [changed, modified].into_iter().all(|time| {
            now.duration_since(time)
                .is_ok_and(|age| age >= Duration::from_secs(2))
        })
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolchainFingerprint {
    pub compiler: PathBuf,
    pub target: String,
    pub identity: String,
    pub persistent: bool,
    #[serde(skip)]
    #[cfg(target_os = "linux")]
    pub verified_inputs: std::sync::Arc<std::collections::BTreeSet<PathBuf>>,
    pub loader: Option<PathBuf>,
}

#[derive(Clone)]
pub struct Toolchain {
    pub compiler: PathBuf,
    pub environment: BTreeMap<OsString, OsString>,
}

impl Toolchain {
    pub fn discover() -> Result<Self> {
        let compiler = resolve("cc")?;
        let mut environment = BTreeMap::new();
        // PATH is explicit and keyed: compiler helpers/linkers still need it.
        // Include/library overrides and loader injection variables are absent.
        for key in [
            "PATH",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "TMPDIR",
            "DEVELOPER_DIR",
            "SDKROOT",
            "MACOSX_DEPLOYMENT_TARGET",
        ] {
            if let Some(value) = std::env::var_os(key) {
                environment.insert(key.into(), value);
            }
        }
        environment.insert("LC_ALL".into(), "C".into());
        environment.insert("LANG".into(), "C".into());
        Ok(Self {
            compiler,
            environment,
        })
    }
    pub fn command(&self, executable: &Path) -> Command {
        let mut command = Command::new(executable);
        command.env_clear().envs(&self.environment);
        command
    }
    pub fn query(
        &self,
        root: &Path,
        work: &Path,
        executable: &Path,
        args: &[&str],
    ) -> Result<String> {
        let _permit = storage::Permit::acquire(root, true, false)?;
        let output = process::run(
            self.command(executable).args(args),
            work,
            "identity",
            Duration::from_secs(120),
            1024 * 1024,
        )?;
        if !output.status.success() {
            return Err(format!(
                "toolchain query {executable:?} {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(String::from_utf8(output.stdout)?)
    }
    pub fn fingerprint(
        &self,
        root: &Path,
        work: &Path,
        args: &[String],
    ) -> Result<ToolchainFingerprint> {
        static SNAPSHOTS: OnceLock<Mutex<BTreeMap<String, Snapshot>>> = OnceLock::new();
        // xcode-select can switch to another still-existing installation without
        // changing any file in the old SDK. Re-resolve selection before reuse.
        #[cfg(target_os = "macos")]
        let selection = self
            .query(root, work, Path::new("/usr/bin/xcode-select"), &["-p"])
            .ok();
        #[cfg(target_os = "linux")]
        let selection = Some("gnu-linux-catalog-v2".to_string());
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let selection: Option<String> = None;
        let mut snapshots = SNAPSHOTS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let supported = known_arguments(args) && selection.is_some();
        let snapshot_key = format!("{:?}:{:?}:{selection:?}", self.compiler, self.environment);
        let cached = snapshots.get(&snapshot_key);
        if let Some(snapshot) = cached.filter(|_| supported)
            && snapshot.inputs.iter().all(|(path, before)| {
                fs::symlink_metadata(path).is_ok_and(|metadata| {
                    settled_metadata(&metadata) && file_identity(&metadata) == *before
                })
            })
        {
            return Ok(snapshot.fingerprint.clone());
        }
        INPUTS.with(|inputs| *inputs.borrow_mut() = Some(BTreeMap::new()));
        let _recording = InputRecording;
        let version = self.query(root, work, &self.compiler, &["--version"])?;
        let target = self
            .query(root, work, &self.compiler, &["-dumpmachine"])?
            .trim()
            .to_string();
        let mut pieces = vec![
            digest_file(&self.compiler)?,
            version,
            target.clone(),
            format!("{:?}", self.environment),
            format!("{selection:?}"),
        ];
        // Unknown codegen/link flags may introduce extra dependencies. Such
        // invocations still compile, but never enter persistent storage.
        #[cfg(target_os = "linux")]
        let mut loader = None;
        #[cfg(not(target_os = "linux"))]
        let loader = None;
        #[cfg(target_os = "linux")]
        let discovery = if supported {
            linux::discover(self, root, work, &mut pieces).map(|path| loader = Some(path))
        } else {
            Err("unsupported compile arguments".into())
        };
        #[cfg(not(target_os = "linux"))]
        let discovery = if supported {
            self.platform_identity(root, work, &mut pieces)
        } else {
            Err("unsupported compile arguments".into())
        };
        let persistent = discovery.is_ok();
        if let Err(error) = discovery
            && std::env::var_os("ASIMPLELIFE_C_TEST_STATS").is_some()
        {
            eprintln!("C_TEST_CACHE_BYPASS {error}");
        }
        if !persistent {
            // No guessed SDK/runtime closure and no reuse under unknown native
            // selection. Each request receives a distinct key and compiles fresh.
            pieces.push(storage::unique());
        }
        let fingerprint = ToolchainFingerprint {
            compiler: self.compiler.clone(),
            target,
            identity: digest(&serde_json::to_vec(&pieces)?),
            persistent,
            #[cfg(target_os = "linux")]
            verified_inputs: std::sync::Arc::new(
                HASHED_INPUTS.with(|inputs| inputs.borrow().clone()),
            ),
            loader,
        };
        if persistent {
            let inputs = INPUTS
                .with(|inputs| inputs.borrow_mut().take())
                .ok_or("fingerprint recording unavailable")?;
            snapshots.insert(
                snapshot_key,
                Snapshot {
                    fingerprint: fingerprint.clone(),
                    inputs,
                },
            );
        }
        Ok(fingerprint)
    }

    #[cfg(target_os = "macos")]
    fn platform_identity(&self, root: &Path, work: &Path, pieces: &mut Vec<String>) -> Result<()> {
        if self.compiler != Path::new("/usr/bin/cc").canonicalize()? {
            return Err("non-system compiler needs its own dependency provider".into());
        }
        let xcrun = PathBuf::from("/usr/bin/xcrun");
        let sdk = self.query(root, work, &xcrun, &["--show-sdk-path"])?;
        let linker = self.query(root, work, &xcrun, &["--find", "ld"])?;
        let selected_compiler = self.query(root, work, &xcrun, &["--find", "clang"])?;
        // cc is Apple's driver shim; its selected compiler must be keyed too.
        pieces.push(digest_file(Path::new(selected_compiler.trim()))?);
        pieces.push(digest_file(Path::new(linker.trim()))?);
        pieces.push(self.query(root, work, &xcrun, &["--show-sdk-version"])?);
        let sdk = self
            .environment
            .get(&OsString::from("SDKROOT"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(sdk.trim()));
        pieces.push(tree_digest(&sdk.join("usr/lib"))?);
        pieces.push(digest_file(&sdk.join("SDKSettings.json"))?);
        let resource = self.query(root, work, &self.compiler, &["-print-resource-dir"])?;
        pieces.push(tree_digest(Path::new(resource.trim()))?);
        let tool_lib = Path::new(selected_compiler.trim())
            .parent()
            .and_then(Path::parent)
            .ok_or("compiler layout unavailable")?
            .join("lib");
        for name in ["libLTO.dylib", "libtapi.dylib"] {
            pieces.push(digest_file(&tool_lib.join(name))?);
        }
        // The signed/sealed OS build owns the libSystem/dyld shared runtime.
        // Do not persist on systems without a sealed system volume.
        let csr = self.query(
            root,
            work,
            &resolve("csrutil")?,
            &["authenticated-root", "status"],
        )?;
        if !csr.contains("enabled") {
            return Err("unsealed OS runtime identity unavailable".into());
        }
        pieces.push(digest_file(Path::new(
            "/System/Library/CoreServices/SystemVersion.plist",
        ))?);
        pieces.push(self.query(
            root,
            work,
            &resolve("sysctl")?,
            &["-n", "kern.osversion", "kern.osrelease", "hw.machine"],
        )?);
        Ok(())
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn platform_identity(
        &self,
        _root: &Path,
        _work: &Path,
        _pieces: &mut Vec<String>,
    ) -> Result<()> {
        // A relocatable GCC/Clang installation cannot be identified from --version
        // alone. Keep execution exact until a complete platform provider exists.
        Err("persistent runtime dependency provider unavailable for this host".into())
    }
}

fn resolve(name: &str) -> Result<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH absent")?) {
        for suffix in ["", ".exe"] {
            let candidate = directory.join(format!("{name}{suffix}"));
            if candidate.is_file() {
                // MinGW locates cc1 relative to argv[0]. canonicalize() returns
                // a Windows extended-length path, which its relocation logic
                // does not understand. Hashing still canonicalizes separately.
                #[cfg(windows)]
                return Ok(std::path::absolute(candidate)?);
                #[cfg(not(windows))]
                return Ok(candidate.canonicalize()?);
            }
        }
    }
    Err(format!("cannot resolve executable {name}").into())
}

pub fn digest_file(path: &Path) -> Result<String> {
    use sha2::Digest;
    use std::io::Read;
    // Metadata guards the memoized content digest, including Unix ctime so a
    // same-length rewrite with restored mtime still invalidates the identity.
    static HASHES: OnceLock<Mutex<BTreeMap<PathBuf, (String, String)>>> = OnceLock::new();
    record(path)?;
    let path = path.canonicalize()?;
    record(&path)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() {
        return Err("identity input is not a regular file".into());
    }
    let identity = file_identity(&metadata);
    if INPUTS.with(|inputs| inputs.borrow().is_some()) {
        HASHED_INPUTS.with(|inputs| inputs.borrow_mut().insert(path.clone()));
    }
    let hashes = HASHES.get_or_init(Mutex::default);
    if let Some((old, digest)) = hashes.lock().unwrap_or_else(|e| e.into_inner()).get(&path)
        && *old == identity
        && settled_metadata(&metadata)
    {
        return Ok(digest.clone());
    }
    let mut file = fs::File::open(&path)?;
    let mut hash = sha2::Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if file_identity(&fs::metadata(&path)?) != identity {
        return Err("toolchain input changed while hashing".into());
    }
    let digest: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    hashes
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(path, (identity, digest.clone()));
    Ok(digest)
}

#[cfg(target_os = "macos")]
fn tree_digest(root: &Path) -> Result<String> {
    let mut todo = vec![root.to_path_buf()];
    let mut files = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(path) = todo.pop() {
        record(&path)?;
        let canonical = path.canonicalize()?;
        record(&canonical)?;
        if !seen.insert(canonical) {
            continue;
        }
        if seen.len() > 65_536 {
            return Err("toolchain identity traversal bound exceeded".into());
        }
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            if entry.path().is_dir() {
                todo.push(entry.path());
            } else {
                files.insert(entry.path(), digest_file(&entry.path())?);
            }
        }
    }
    Ok(digest(&serde_json::to_vec(&files)?))
}
