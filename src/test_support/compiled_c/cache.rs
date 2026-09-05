use super::*;
use std::io::Read;

#[derive(Serialize, Deserialize)]
pub struct ArtifactManifest {
    schema: u32,
    key: String,
    toolchain: String,
    target: String,
    artifact: String,
}

pub struct ArtifactLease {
    _lock: storage::Lock,
}
pub struct ArtifactRequest<'a> {
    pub root: &'a Path,
    pub key: &'a str,
    pub fingerprint: &'a toolchain::ToolchainFingerprint,
    pub destination: &'a Path,
}

impl ArtifactRequest<'_> {
    fn directory(&self) -> PathBuf {
        self.root.join(format!("entry-{}", self.key))
    }
    fn lease(&self) -> Result<ArtifactLease> {
        Ok(ArtifactLease {
            _lock: storage::Lock::acquire(&storage::stripe(self.root, "lease", self.key), true)?,
        })
    }
    fn validate(&self) -> Result<ArtifactManifest> {
        let directory = self.directory();
        if !directory.symlink_metadata()?.is_dir() {
            return Err("cache entry is not a directory".into());
        }
        let manifest_path = directory.join("manifest.json");
        regular(&manifest_path, 65536)?;
        let manifest: ArtifactManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
        if manifest.schema != SCHEMA
            || manifest.key != self.key
            || manifest.target != self.fingerprint.target
            || manifest.toolchain != self.fingerprint.identity
        {
            return Err("cache manifest identity mismatch".into());
        }
        let binary = directory.join("program.bin");
        validate_executable(&binary, &manifest.target)?;
        if toolchain::digest_file(&binary)? != manifest.artifact {
            return Err("cache executable digest mismatch".into());
        }
        Ok(manifest)
    }
    pub fn copy_hit(&self) -> Result<bool> {
        let _lease = self.lease()?;
        let Ok(manifest) = self.validate() else {
            return Ok(false);
        };
        let directory = self.directory();
        fs::copy(directory.join("program.bin"), self.destination)?;
        validate_executable(self.destination, &manifest.target)?;
        if toolchain::digest_file(self.destination)? != manifest.artifact {
            return Err("private executable copy digest mismatch".into());
        }
        let _ = fs::write(directory.join("access"), []);
        Ok(true)
    }
    pub fn publish(&self, executable: &Path) -> Result<()> {
        let directory = self.directory();
        let _lease = storage::Lock::acquire(&storage::stripe(self.root, "lease", self.key), false)?;
        // Caller holds the build lock. An invalid prior entry is replaced only
        // under the exclusive artifact lease, never while a reader copies it.
        if self.validate().is_ok() {
            return Ok(());
        }
        if directory.exists() {
            fs::remove_dir_all(&directory)?;
        }
        let parent = executable.parent().ok_or("executable has no workspace")?;
        let staging = parent.join("publish");
        let candidate_bytes = fs::metadata(executable)?
            .len()
            .checked_add(65536)
            .ok_or("staging byte overflow")?;
        if storage::directory_size(parent)?
            .checked_add(candidate_bytes)
            .is_none_or(|bytes| bytes > storage::WORK_RESERVATION)
        {
            return Err("optional artifact staging exceeds reserved workspace capacity".into());
        }
        fs::create_dir(&staging)?;
        fs::copy(executable, staging.join("program.bin"))?;
        let manifest = ArtifactManifest {
            schema: SCHEMA,
            key: self.key.into(),
            target: self.fingerprint.target.clone(),
            toolchain: self.fingerprint.identity.clone(),
            artifact: toolchain::digest_file(executable)?,
        };
        fs::write(
            staging.join("manifest.partial"),
            serde_json::to_vec(&manifest)?,
        )?;
        fs::rename(
            staging.join("manifest.partial"),
            staging.join("manifest.json"),
        )?;
        fs::write(staging.join("access"), [])?;
        // Staging is covered by the active workspace reservation. Admission
        // accounts the additional retained copy before the rename publishes it.
        let _admission = storage::Lock::acquire(&self.root.join("admission.lock"), false)?;
        storage::collect(self.root, storage::directory_size(&staging)?)?;
        fs::rename(staging, directory)?;
        Ok(())
    }
}

fn regular(path: &Path, limit: u64) -> Result<()> {
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err("cache input is not a bounded regular file".into());
    }
    Ok(())
}

pub fn validate_executable(path: &Path, target: &str) -> Result<()> {
    regular(path, 128 * 1024 * 1024)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & 0o111 == 0 {
            return Err("cached file is not executable".into());
        }
    }
    let mut header = [0_u8; 4096];
    let count = fs::File::open(path)?.read(&mut header)?;
    if count < 20 {
        return Err("executable header truncated".into());
    }
    let arm = target.starts_with("aarch64") || target.starts_with("arm64");
    let x86 = target.starts_with("x86_64");
    let valid = if header[..4] == [0xcf, 0xfa, 0xed, 0xfe] {
        let cpu = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        (arm && cpu == 0x0100_000c) || (x86 && cpu == 0x0100_0007)
    } else if &header[..4] == b"\x7fELF" {
        let cpu = u16::from_le_bytes([header[18], header[19]]);
        header[4] == 2 && header[5] == 1 && ((arm && cpu == 183) || (x86 && cpu == 62))
    } else if &header[..2] == b"MZ" && count >= 64 {
        let start = usize::try_from(u32::from_le_bytes([
            header[60], header[61], header[62], header[63],
        ]))?;
        start.checked_add(6).is_some_and(|end| end <= count)
            && &header[start..start + 4] == b"PE\0\0"
            && ((arm && header[start + 4..start + 6] == [0x64, 0xaa])
                || (x86 && header[start + 4..start + 6] == [0x64, 0x86]))
    } else {
        false
    };
    if !valid {
        return Err(format!("executable format/target mismatch: {target}").into());
    }
    Ok(())
}
