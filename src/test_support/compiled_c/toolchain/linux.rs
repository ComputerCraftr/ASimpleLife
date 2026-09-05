//! Native GNU GCC/ld + glibc provider. Discovery runs metadata queries only;
//! publication additionally verifies actual assembler, linker and loader inputs.
use super::*;
mod catalog;
#[cfg(test)]
mod tests;
mod validation;

fn output(
    tool: &Toolchain,
    root: &Path,
    work: &Path,
    program: &Path,
    args: &[&str],
) -> Result<String> {
    tool.query(root, work, program, args)
}

pub(super) fn discover(
    tool: &Toolchain,
    root: &Path,
    work: &Path,
    pieces: &mut Vec<String>,
) -> Result<PathBuf> {
    let gcc = &tool.compiler;
    let search = output(tool, root, work, gcc, &["-print-search-dirs"])?;
    let sysroot = output(tool, root, work, gcc, &["-print-sysroot"])?;
    if !matches!(sysroot.trim(), "" | "/") {
        return Err("non-native GCC sysroot needs a separate provider".into());
    }
    if output(tool, root, work, gcc, &["-print-file-name=specs"])?.trim() != "specs" {
        return Err("external GCC specs are unsupported".into());
    }
    let specs = output(tool, root, work, gcc, &["-dumpspecs"])?;
    if !specs.contains("*cross_compile:\n0\n") || !specs.contains("*link_command:") {
        return Err("not a native GNU GCC driver".into());
    }
    if specs.contains("-L")
        || specs.contains("-rpath")
        || !specs.contains("*post_link:\n\n")
        || !specs.contains("*self_spec:\n\n")
    {
        return Err("custom GCC search/driver specs are unsupported".into());
    }
    let mut roots = Vec::new();
    for kind in ["programs: =", "libraries: ="] {
        let paths = search
            .lines()
            .find_map(|line| line.strip_prefix(kind))
            .ok_or("GCC search paths absent")?;
        roots.extend(
            paths
                .split(':')
                .filter(|path| !path.is_empty())
                .map(PathBuf::from),
        );
    }
    let mut tools = vec![gcc.clone()];
    for name in ["cc1", "collect2", "as", "ld"] {
        let path = output(
            tool,
            root,
            work,
            gcc,
            &[&format!("-print-prog-name={name}")],
        )?;
        tools.push(catalog::executable(path.trim())?);
    }
    let linker = tools.last().ok_or("linker absent")?;
    let script = output(tool, root, work, linker, &["--verbose"])?;
    if !script.starts_with("GNU ld ") {
        return Err("non-GNU linker is unsupported".into());
    }
    roots.extend(linker_search_dirs(&script)?);
    let readelf = catalog::executable("readelf")?;
    let headers = output(
        tool,
        root,
        work,
        &readelf,
        &["-lW", gcc.to_str().ok_or("compiler path encoding")?],
    )?;
    let loader = headers
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("[Requesting program interpreter: ")
                .and_then(|line| line.strip_suffix(']'))
        })
        .ok_or("ELF interpreter absent")?;
    let loader = PathBuf::from(loader);
    let help = output(tool, root, work, &loader, &["--help"])?;
    let version = output(tool, root, work, &loader, &["--version"])?;
    if !version.contains("GLIBC") {
        return Err("non-glibc interpreter is unsupported".into());
    }
    pieces.push(version);
    for line in help.lines() {
        if let Some(directory) = line.trim().strip_suffix(" (system search path)") {
            roots.push(directory.into());
        }
    }
    let diagnostics = output(tool, root, work, &loader, &["--list-diagnostics"])?;
    let capabilities: Vec<_> = diagnostics
        .lines()
        .filter(|line| {
            line.starts_with("dl_hwcap")
                || line.starts_with("dl_platform=")
                || line.starts_with("x86.")
        })
        .collect();
    if capabilities.is_empty() {
        return Err("loader capability identity absent".into());
    }
    pieces.push(capabilities.join("\n"));
    let ldconfig = catalog::executable("ldconfig")?;
    let cache = output(tool, root, work, &ldconfig, &["-p"])?;
    for line in cache.lines() {
        if let Some((_, path)) = line.split_once(" => ") {
            let path = Path::new(path);
            if !path.is_absolute() {
                return Err("relative loader cache input".into());
            }
            roots.push(path.parent().ok_or("loader library parent")?.to_path_buf());
        }
    }
    for path in [
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/ld.so.preload",
    ] {
        roots.push(path.into());
    }
    if fs::read("/etc/ld.so.preload").is_ok_and(|bytes| !bytes.is_empty()) {
        return Err("global loader preloads disable persistence".into());
    }
    let plugin = output(
        tool,
        root,
        work,
        gcc,
        &["-print-file-name=liblto_plugin.so"],
    )?;
    tools.extend([readelf.clone(), catalog::executable(plugin.trim())?]);
    // ldconfig only enumerates the loader cache; it is not invoked by compile,
    // link or execution. Ubuntu ships it as a shell wrapper, not an ELF tool.
    // Seal its output, wrapper and ld.so.cache below. Actual link/loader inputs
    // still require independent coverage before publishing an artifact.
    // Private runtime search paths require a separate resolution provider.
    for program in &tools {
        if !validation::no_private_search(tool, root, work, &readelf, program)? {
            continue;
        }
        let dependencies = output(
            tool,
            root,
            work,
            &loader,
            &["--list", program.to_str().ok_or("tool path encoding")?],
        )?;
        for dependency in loader_inputs(&dependencies)? {
            validation::no_private_search(tool, root, work, &readelf, &dependency)?;
            roots.push(dependency);
        }
    }
    roots.extend(validation::capability_directories(&roots, &help)?);
    roots.extend(tools);
    roots.extend([loader.clone(), readelf, ldconfig]);
    pieces.extend([search, specs, script, cache, catalog::seal(roots)?]);
    Ok(loader)
}

fn linker_search_dirs(script: &str) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for part in script.split("SEARCH_DIR(").skip(1) {
        let part = part
            .strip_prefix('"')
            .ok_or("unsupported linker search quoting")?;
        let (path, rest) = part
            .split_once('"')
            .ok_or("unterminated linker search path")?;
        if !rest.starts_with(')') {
            return Err("malformed linker search directive".into());
        }
        let path = PathBuf::from(path.strip_prefix('=').unwrap_or(path));
        if !path.is_absolute() {
            return Err("relative linker search path".into());
        }
        result.push(path);
    }
    if result.is_empty() {
        return Err("linker search paths unavailable".into());
    }
    Ok(result)
}

fn loader_inputs(output: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for line in output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if line.starts_with("linux-vdso.so.1 (") {
            continue;
        }
        if line
            .strip_prefix("(0x")
            .and_then(|value| value.strip_suffix(')'))
            .is_some_and(|value| {
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            continue; // glibc's unnamed main executable, already sealed separately.
        }
        let path = line
            .split_once(" => ")
            .map_or(line, |(_, value)| value)
            .split_once(" (")
            .map(|(path, _)| path)
            .ok_or("unrecognized loader dependency")?;
        if !Path::new(path).is_absolute() {
            return Err("unresolved loader dependency".into());
        }
        paths.push(path.into());
    }
    if paths.is_empty() {
        return Err("empty loader dependency list".into());
    }
    Ok(paths)
}

pub(in crate::test_support::compiled_c) fn instrument(
    command: &mut Command,
    work: &Path,
) -> Result<bool> {
    let Some(path) = work.to_str().filter(|path| {
        !path
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, ',' | '\\'))
    }) else {
        return Ok(false);
    };
    fs::create_dir(work.join("compiler-tmp"))?;
    command
        .arg("-Wl,--trace")
        .arg(format!("-Wa,--MD,{path}/assembler.d"))
        .env("TMPDIR", work.join("compiler-tmp"));
    Ok(true)
}

fn covered(fingerprint: &ToolchainFingerprint, work: &Path, path: &Path) -> Result<()> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        work.join(path)
    };
    if path.starts_with(work)
        && !path
            .components()
            .any(|part| part == std::path::Component::ParentDir)
    {
        return Ok(()); // GCC removes its owned temporary assembler/object inputs.
    }
    let path = path.canonicalize()?;
    if !path.starts_with(work) && !fingerprint.verified_inputs.contains(&path) {
        return Err(format!("unsealed link/assembly input {}", path.display()).into());
    }
    Ok(())
}

pub(in crate::test_support::compiled_c) fn audit(
    tool: &Toolchain,
    fingerprint: &ToolchainFingerprint,
    root: &Path,
    work: &Path,
    executable: &Path,
    trace: &[u8],
) -> Result<()> {
    let trace = std::str::from_utf8(trace)?;
    if trace.trim().is_empty() {
        return Err("link input trace absent".into());
    }
    for path in trace.lines().map(str::trim).filter(|line| !line.is_empty()) {
        covered(fingerprint, work, Path::new(path))?;
        if !Path::new(path).starts_with(work) {
            validation::link_input(Path::new(path))?;
        }
    }
    let dependencies = fs::read_to_string(work.join("assembler.d"))?.replace("\\\n", "");
    let (_, inputs) = dependencies
        .split_once(':')
        .ok_or("assembler dependencies absent")?;
    let mut assembly_count = 0;
    for input in inputs.split_whitespace() {
        let path = Path::new(input);
        // GAS also reports the logical .file name. Both source forms are keyed
        // byte-for-byte, unlike arbitrary work-directory spools or incbin data.
        if ["program.c", "program.i"]
            .iter()
            .any(|name| path == Path::new(name) || path == work.join(name))
        {
            continue;
        }
        if input.contains('\\')
            || !path.starts_with(work.join("compiler-tmp"))
            || path.extension().is_none_or(|extension| extension != "s")
        {
            return Err("extra/unowned assembler dependencies disable persistence".into());
        }
        covered(fingerprint, work, path)?;
        assembly_count += 1;
    }
    if assembly_count != 1 {
        return Err("ambiguous compiler assembly input".into());
    }
    let loader = fingerprint
        .loader
        .as_ref()
        .ok_or("loader identity absent")?;
    let readelf = catalog::executable("readelf")?;
    validation::no_private_search(tool, root, work, &readelf, executable)?;
    let dependencies = output(
        tool,
        root,
        work,
        loader,
        &[
            "--list",
            executable.to_str().ok_or("executable path encoding")?,
        ],
    )?;
    for path in loader_inputs(&dependencies)? {
        covered(fingerprint, work, &path)?;
        validation::no_private_search(tool, root, work, &readelf, &path)?;
    }
    Ok(())
}
