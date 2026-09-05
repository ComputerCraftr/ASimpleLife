use super::*;

pub(super) fn no_private_search(
    tool: &Toolchain,
    root: &Path,
    work: &Path,
    readelf: &Path,
    file: &Path,
) -> Result<bool> {
    let dynamic = output(
        tool,
        root,
        work,
        readelf,
        &["-dW", file.to_str().ok_or("ELF path encoding")?],
    )?;
    if dynamic.contains("(RPATH)")
        || dynamic.contains("(RUNPATH)")
        || dynamic.contains("(AUDIT)")
        || dynamic.contains("(DEPAUDIT)")
        || dynamic.contains("(FILTER)")
        || dynamic.contains("(AUXILIARY)")
    {
        return Err("private ELF dependency resolution is unsupported".into());
    }
    Ok(dynamic.contains("(NEEDED)"))
}

pub(super) fn capability_directories(roots: &[PathBuf], help: &str) -> Result<Vec<PathBuf>> {
    let mut names = std::collections::BTreeSet::from(["glibc-hwcaps"]);
    for line in help.lines().map(str::trim) {
        if (line.contains("supported") || line.contains("searched"))
            && let Some((name, _)) = line.split_once(" (")
            && !name.is_empty()
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
        {
            names.insert(name);
        }
    }
    let mut result = Vec::new();
    let mut pending: Vec<_> = roots
        .iter()
        .filter(|path| path.is_dir())
        .cloned()
        .map(|path| (path, 0))
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    while let Some((path, depth)) = pending.pop() {
        if !seen.insert(path.clone()) {
            continue;
        }
        if depth > 8 || seen.len() > 4096 {
            return Err("loader capability tree limit".into());
        }
        for name in &names {
            let child = path.join(name);
            catalog::watch(&child)?;
            if child.is_dir() {
                result.push(child.clone());
                pending.push((child, depth + 1));
            }
        }
    }
    Ok(result)
}

pub(super) fn link_input(path: &Path) -> Result<()> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    if magic.starts_with(b"\x7fELF") || magic == *b"!<arch>\n" {
        return Ok(());
    }
    // Thin archives resolve external members; unknown formats fail closed.
    if fs::metadata(path)?.len() > 65536 {
        return Err("unsupported linker script size".into());
    }
    simple_link_script(&fs::read_to_string(path)?)
}

/// GNU linker metadata, not C source: allow only input grouping. Commands that
/// add search paths or include other scripts require their own closure proof.
pub(super) fn simple_link_script(script: &str) -> Result<()> {
    let mut rest = script;
    let mut clean = String::new();
    while let Some((before, comment)) = rest.split_once("/*") {
        clean.push_str(before);
        rest = comment
            .split_once("*/")
            .ok_or("unterminated linker comment")?
            .1;
    }
    clean.push_str(rest);
    for word in clean
        .split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ',' | ';'))
        .filter(|word| !word.is_empty())
    {
        if matches!(word, "GROUP" | "INPUT" | "AS_NEEDED" | "OUTPUT_FORMAT")
            || word.starts_with('/')
            || word.starts_with("-l")
            || (word.starts_with("elf")
                && word
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_')))
            || (word.contains('.')
                && word
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-')))
            || (word.starts_with('"') && word.ends_with('"') && word.contains("elf"))
        {
            continue;
        }
        return Err(format!("unsupported GNU linker script token {word}").into());
    }
    Ok(())
}
