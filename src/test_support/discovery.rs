use std::fs;
use std::path::{Path, PathBuf};

/// Discover owned sources, including build scripts and integration tests. Never
/// follow symlinks into another repository or recurse through generated output.
pub(crate) fn source_files(root: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            let path = entry.path();
            if kind.is_dir() {
                if !matches!(
                    entry.file_name().to_str(),
                    Some(".git" | "target" | "vendor")
                ) {
                    pending.push(path);
                }
            } else if kind.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| extensions.contains(&ext))
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[test]
fn discovery_includes_arbitrary_module_layouts_and_excludes_generated_sources() {
    use crate::RequiredExt;
    let root = std::env::temp_dir().join(format!("source-discovery-{}", std::process::id()));
    for file in [
        "build.rs",
        "tests/nested/check.rs",
        "renamed/kernel.c",
        "renamed/kernel.h",
        "target/generated.rs",
        ".git/hidden.rs",
        "vendor/foreign.c",
    ] {
        let path = root.join(file);
        fs::create_dir_all(path.parent().or_invariant("fixture parent"))
            .or_invariant("fixture directory");
        fs::write(path, "").or_invariant("fixture file");
    }
    let paths = source_files(&root, &["rs", "c", "h"]).or_invariant("discover sources");
    let relative: Vec<_> = paths
        .iter()
        .map(|path| path.strip_prefix(&root).or_invariant("relative path"))
        .collect();
    assert_eq!(
        relative,
        [
            Path::new("build.rs"),
            Path::new("renamed/kernel.c"),
            Path::new("renamed/kernel.h"),
            Path::new("tests/nested/check.rs")
        ]
    );
    fs::remove_dir_all(root).or_invariant("remove discovery fixture");
}
