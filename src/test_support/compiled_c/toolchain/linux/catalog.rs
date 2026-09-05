use super::*;
use std::collections::{BTreeSet, VecDeque};

/// Watch every symlink hop and missing search-directory ancestor. A new shadow
/// input or an intermediate alternatives-link change must invalidate discovery.
pub(super) fn watch(path: &Path) -> Result<()> {
    let mut pending = VecDeque::from([path.to_path_buf()]);
    let mut seen = BTreeSet::new();
    while let Some(path) = pending.pop_front() {
        if !path.is_absolute() || seen.len() >= 256 {
            return Err("unbounded/relative input resolution".into());
        }
        if !seen.insert(path.clone()) {
            continue;
        }
        let mut prefix = PathBuf::new();
        for component in path.components() {
            prefix.push(component);
            match fs::symlink_metadata(&prefix) {
                Ok(metadata) => {
                    record(&prefix)?;
                    if metadata.is_symlink() {
                        let target = fs::read_link(&prefix)?;
                        pending.push_back(if target.is_absolute() {
                            target
                        } else {
                            prefix.parent().ok_or("symlink parent")?.join(target)
                        });
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

pub(super) fn executable(name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute() {
        watch(path)?;
        return Ok(path.canonicalize()?);
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH absent")?) {
        let path = directory.join(name);
        watch(&path)?;
        if path.is_file() {
            return Ok(path.canonicalize()?);
        }
    }
    Err(format!("unresolved tool {name}").into())
}

/// Seal direct search inputs and loader-admitted capability directories. Do not
/// traverse unrelated trees (for example SSL private keys beneath /usr/lib).
pub(super) fn seal(roots: Vec<PathBuf>) -> Result<String> {
    let mut pending = roots;
    let mut visited = BTreeSet::new();
    let mut files = BTreeMap::new();
    let mut bytes = 0_u64;
    while let Some(path) = pending.pop() {
        watch(&path)?;
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let canonical = path.canonicalize()?;
        watch(&canonical)?;
        if visited.len() >= 65536 {
            return Err("Linux catalog entry limit".into());
        }
        // Include alias resolution in identity even if content is already sealed.
        files
            .entry(path.clone())
            .or_insert_with(|| (canonical.clone(), String::new()));
        if !visited.insert(canonical.clone()) {
            continue;
        }
        if metadata.is_dir() {
            for entry in fs::read_dir(&canonical)? {
                let entry = entry?;
                if entry.path().is_file() {
                    pending.push(entry.path());
                }
            }
        } else if metadata.is_file() {
            bytes = bytes
                .checked_add(metadata.len())
                .ok_or("catalog bytes overflow")?;
            // This is a bounded streaming identity scan, not retained cache
            // storage. Multi-toolchain CI hosts can have several GiB of inputs.
            if bytes > 16 * 1024 * 1024 * 1024 {
                return Err("Linux catalog byte limit".into());
            }
            let hash = digest_file(&canonical)?;
            files.insert(canonical.clone(), (canonical, hash));
        } else {
            return Err("nonregular Linux toolchain input".into());
        }
    }
    Ok(digest(&serde_json::to_vec(&files)?))
}
