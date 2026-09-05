use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const CACHE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
pub const WORK_RESERVATION: u64 = 256 * 1024 * 1024;
static UNIQUE: AtomicU64 = AtomicU64::new(0);

pub fn unique() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        UNIQUE.fetch_add(1, Ordering::Relaxed)
    )
}

pub fn lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
}

/// Fixed lock stripes are never unlinked: unlinking a held/waited-on lock file
/// can give two producers separate locks for the same key.
pub struct Lock(pub File);
impl Lock {
    pub fn acquire(path: &Path, shared: bool) -> io::Result<Self> {
        let file = lock_file(path)?;
        let start = Instant::now();
        loop {
            let result = if shared {
                file.try_lock_shared()
            } else {
                file.try_lock()
            };
            match result {
                Ok(()) => return Ok(Self(file)),
                Err(fs::TryLockError::Error(error)) => return Err(error),
                Err(fs::TryLockError::WouldBlock) => {
                    if start.elapsed() > Duration::from_secs(180) {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "cache lock deadline",
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }
    pub fn try_exclusive(path: &Path) -> io::Result<Option<Self>> {
        let file = lock_file(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self(file))),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(error)) => Err(error),
        }
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

pub fn stripe(root: &Path, category: &str, key: &str) -> PathBuf {
    // Directory names are untrusted, including truncated or non-ASCII keys.
    let stripe = if key.len() == 64 && key.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        key.get(..2).unwrap_or("invalid")
    } else {
        "invalid"
    };
    root.join(format!("{category}-{stripe}"))
}

pub struct Workspace {
    pub path: PathBuf,
    _owner: Lock,
}
impl Workspace {
    pub fn create(root: &Path) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        let started = Instant::now();
        loop {
            let admission = Lock::acquire(&root.join("admission.lock"), false)?;
            // Leave space for at least one maximum-size completed executable
            // to be published while all admitted workspaces remain alive.
            if collect(root, WORK_RESERVATION + 128 * 1024 * 1024).is_err() {
                drop(admission);
                if started.elapsed() >= Duration::from_secs(180) {
                    return Err(io::Error::other(
                        "test workspace admission capacity unavailable",
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            let path = root.join(format!("work-{}", unique()));
            fs::create_dir(&path)?;
            let owner = Lock::acquire(&path.join("owner.lock"), false)?;
            drop(admission);
            return Ok(Self {
                path,
                _owner: owner,
            });
        }
    }
    pub fn check_size(&self) -> io::Result<()> {
        if directory_size(&self.path)? > WORK_RESERVATION {
            return Err(io::Error::other("test workspace exceeded 256 MiB"));
        }
        Ok(())
    }
    pub fn finish(self) -> io::Result<()> {
        let path = self.path.clone();
        drop(self);
        if path.exists() {
            return Err(io::Error::other(format!(
                "test workspace cleanup failed: {}",
                path.display()
            )));
        }
        Ok(())
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        // Admission protects the interval between releasing owner.lock and
        // removing the directory. Otherwise a collector can recreate owner.lock
        // during remove_dir_all and strand a nonempty workspace.
        let admission = self
            .path
            .parent()
            .and_then(|root| Lock::acquire(&root.join("admission.lock"), false).ok());
        if admission.is_none() {
            return;
        }
        // Windows cannot unlink an open owner lock, so release it before removal.
        let _ = self._owner.0.unlock();
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub fn directory_size(root: &Path) -> io::Result<u64> {
    let mut todo = vec![root.to_path_buf()];
    let mut bytes = 0_u64;
    while let Some(path) = todo.pop() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.path().symlink_metadata()?;
            if metadata.is_dir() {
                todo.push(entry.path());
            } else {
                bytes = bytes
                    .checked_add(metadata.len())
                    .ok_or_else(|| io::Error::other("cache byte overflow"))?;
            }
        }
    }
    Ok(bytes)
}

/// Called under admission coordination. Never waits on a build or lease lock.
pub fn collect(root: &Path, reserve: u64) -> io::Result<()> {
    let mut retained = 0_u64;
    let mut candidates = Vec::new();
    for item in fs::read_dir(root)? {
        let item = item?;
        let name = item.file_name().to_string_lossy().into_owned();
        if !item.file_type()?.is_dir() {
            retained += item.metadata()?.len();
            continue;
        }
        if name.starts_with("work-") {
            if let Some(owner) = Lock::try_exclusive(&item.path().join("owner.lock"))? {
                drop(owner);
                fs::remove_dir_all(item.path())?;
            } else {
                retained += WORK_RESERVATION;
            }
        } else if let Some(key) = name.strip_prefix("entry-") {
            let size = directory_size(&item.path())?;
            retained += size;
            let age = fs::metadata(item.path().join("access"))
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            candidates.push((age, key.to_string(), item.path(), size));
        } else {
            retained += directory_size(&item.path())?;
        }
    }
    let evictable = candidates.iter().map(|candidate| candidate.3).sum::<u64>();
    if retained.saturating_sub(evictable).saturating_add(reserve) > CACHE_LIMIT {
        // Waiting for an active workspace is necessary. Evicting artifacts
        // cannot satisfy this request and would destroy useful warm reuse.
        return Err(io::Error::other(
            "active workspace reservations exhaust admission",
        ));
    }
    candidates.sort_by_key(|candidate| candidate.0);
    for (_, key, path, size) in candidates {
        if retained.saturating_add(reserve) <= CACHE_LIMIT {
            break;
        }
        let Some(_build) = Lock::try_exclusive(&stripe(root, "build", &key))? else {
            continue;
        };
        let Some(_lease) = Lock::try_exclusive(&stripe(root, "lease", &key))? else {
            continue;
        };
        fs::remove_dir_all(path)?;
        retained -= size;
    }
    if retained.saturating_add(reserve) > CACHE_LIMIT {
        return Err(io::Error::other("compiled test cache capacity unavailable"));
    }
    Ok(())
}

/// Caller holds admission coordination. Stable lock files are never removed.
pub fn clean(root: &Path) -> io::Result<()> {
    collect(root, 0)?;
    for item in fs::read_dir(root)? {
        let item = item?;
        let name = item.file_name().to_string_lossy().into_owned();
        if !item.file_type()?.is_dir() {
            continue;
        }
        if name.starts_with("work-") {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "cache has active workspaces",
            ));
        }
        if let Some(key) = name.strip_prefix("entry-") {
            let _build = Lock::try_exclusive(&stripe(root, "build", key))?.ok_or_else(|| {
                io::Error::new(io::ErrorKind::WouldBlock, "artifact is being built")
            })?;
            let _lease = Lock::try_exclusive(&stripe(root, "lease", key))?
                .ok_or_else(|| io::Error::new(io::ErrorKind::WouldBlock, "artifact is leased"))?;
            fs::remove_dir_all(item.path())?;
        }
    }
    Ok(())
}

pub struct Permit {
    _slots: Vec<Lock>,
}
impl Permit {
    pub fn acquire(root: &Path, compile: bool, sanitizer: bool) -> io::Result<Self> {
        let cpus = std::thread::available_parallelism().map_or(1, usize::from);
        let budget = cpus.clamp(2, 4);
        let ceiling = if compile { cpus.min(2) } else { cpus.min(4) };
        let weight = if compile || sanitizer { 2 } else { 1 };
        let class = if compile { "compiler" } else { "execution" };
        let started = Instant::now();
        loop {
            let admission = Lock::acquire(&root.join("permits.lock"), false)?;
            let mut held = Vec::new();
            for slot in 0..ceiling {
                if let Some(lock) = Lock::try_exclusive(&root.join(format!("{class}-{slot}.lock")))?
                {
                    held.push(lock);
                    break;
                }
            }
            let mut ready = !held.is_empty();
            if sanitizer {
                if let Some(lock) = Lock::try_exclusive(&root.join("sanitizer.lock"))? {
                    held.push(lock);
                } else {
                    ready = false;
                }
            }
            let base = held.len();
            if ready {
                for slot in 0..budget {
                    if let Some(lock) =
                        Lock::try_exclusive(&root.join(format!("weight-{slot}.lock")))?
                    {
                        held.push(lock);
                    }
                    if held.len() == base + weight {
                        break;
                    }
                }
                ready = held.len() == base + weight;
            }
            drop(admission);
            if ready {
                return Ok(Self { _slots: held });
            }
            drop(held);
            if started.elapsed() >= Duration::from_secs(180) {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "test resource permit deadline",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
