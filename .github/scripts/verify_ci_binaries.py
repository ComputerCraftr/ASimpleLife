"""Validate actual Cargo artifacts, not assumptions about cargo test selection."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    if os.name == "nt":
        rustc = subprocess.check_output(["rustc", "-vV"], text=True)
        if "host: x86_64-pc-windows-msvc" not in rustc.splitlines():
            raise RuntimeError("merged MSVC gate requires the original x86_64 MSVC target")
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], text=True))
    members = set(metadata["workspace_members"])
    expected = {(package["id"], target["name"])
                for package in metadata["packages"] if package["id"] in members
                for target in package["targets"] if "bin" in target["kind"]}
    if not expected:
        raise RuntimeError("no ordinary binary targets discovered")
    artifacts = {}
    command = ["cargo", "build", "--workspace", "--bins", "--all-features",
               "--release", "--message-format=json"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as child:
        for line in child.stdout:
            message = json.loads(line)
            if message.get("reason") == "compiler-artifact" and message.get("executable"):
                artifacts[(message["package_id"], message["target"]["name"])] = message
        if child.wait() != 0:
            raise RuntimeError("ordinary release binary build failed")
    missing = expected - artifacts.keys()
    if missing:
        raise RuntimeError(f"missing compiled binary targets: {sorted(missing)}")
    for key in sorted(expected):
        artifact = artifacts[key]
        path = Path(artifact["executable"])
        if not path.is_file() or artifact["profile"]["test"]:
            raise RuntimeError(f"not an ordinary executable: {artifact}")
        print(json.dumps({"target": key[1], "executable": str(path),
                          "features": artifact["features"], "profile": artifact["profile"]}))
        if os.name == "nt":
            imports = subprocess.check_output(
                ["llvm-readobj", "--coff-imports", str(path)], text=True).lower()
            if any(name in imports for name in ("vcruntime", "msvcp", "ucrtbase")):
                raise RuntimeError(f"dynamic MSVC CRT dependency: {path}")
    if os.name == "nt":
        cfg = subprocess.check_output(["cargo", "rustc", "--lib", "--release",
                                       "--", "--print", "cfg"], text=True)
        if 'target_feature="crt-static"' not in cfg.splitlines():
            raise RuntimeError("MSVC static CRT not enabled")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"CI artifact inventory failed: {error}", file=sys.stderr)
        sys.exit(1)
