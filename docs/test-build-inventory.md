# Test Build Inventory

The optimization preserves configurations, not just matching test names.

| Retained Gate | Configuration and Coverage |
| --- | --- |
| Cross-platform | macOS, Linux, Windows; all-target/all-feature debug check; all-feature release unit, integration and doc tests |
| Binary artifacts | Cargo metadata discovers every workspace binary; Cargo JSON proves ordinary all-feature release compilation for each |
| MSVC inspection | Windows x86_64 MSVC host is checked explicitly; inspect the actual ordinary executable artifacts for dynamic CRT imports |
| musl | Separate x86_64 musl release build and ELF dependency inspection |
| Native kernels | AVX2 and NEON release configurations retain their distinct Rust flags and filtered tests |
| Scalar fallback | Release kernel and correctness tests retain forced scalar dispatch |
| Generated C | Debug-profile strict-template and sanitizer tests remain distinct from the cross-platform release suite |
| Quality | Formatting, structured CI policy checks, source policies and all-target/all-feature strict Clippy |
| Miri | Nightly scalar storage/allocation/kernel checks retain their exact filters |

Only the redundant standalone x86_64 MSVC artifact build was merged into the
Windows job. The three utility binaries without unit tests have `test = false`;
the ordinary-binary inventory gate prevents that from removing their compilation
coverage. Filtered library tests select `--lib`, avoiding irrelevant binary test
harnesses. Full Cargo test invocations still include integration tests/doctests.

Generated-C polynomial semantic fixtures run on every platform, including
Windows. ASan/UBSan variants run on Linux and macOS, matching the existing
sanitizer platform contract: the Windows MinGW toolchain does not ship those
runtime libraries. The Ubuntu generated-C gate also runs the polynomial suite
in the debug profile; sanitizer flags are never silently dropped. Shared AVX2
multiply helpers are forcibly inlined even at `-O0` to avoid MinGW's by-value
vector stack ABI ([GCC PR54412 regression coverage](https://gcc.gnu.org/pipermail/gcc-patches/2026-June/719571.html)).
Unoptimized native/scalar differential tests cover transfer,
summary, and polynomial callers separately from optimized execution.

The generated-C cache is local to a trusted workspace. It is not an execution
result cache. Changes to the compile surface or toolchain discovery must preserve
the conservative bypass for unrecognized link/native configuration.

## Cache Ownership

Build ownership, artifact leases, admission/eviction, and resource permits use
separate advisory locks. A producer takes build ownership before an exclusive
artifact lease, then admission for publication. A reader holds only a shared
artifact lease while validating and copying. Admission/eviction never waits for
build or lease ownership: it tries those locks and skips busy entries. Workspace
retirement holds admission while releasing its owner lock and deleting its files.
Stable lock files are never unlinked. All artifact/build locks are released before
waiting for an execution permit. Process death releases OS-owned locks.

The weighted permit pool and 2-GiB logical admission budget are shared across
harness processes using one cache root. Stream limits are enforced while draining
child pipes. Workspace capacity is checked at phase boundaries; this is not a
filesystem quota or a sandbox against arbitrary child file writes. Additional
publication staging is preflighted against its workspace reservation.

CI places executable artifacts in a job-local temporary directory, outside the
restored Cargo cache. Cross-run executable reuse is deliberately unavailable in
CI. Local reuse supports the closed Apple compiler/SDK provider and native GNU
GCC/GNU ld/glibc Linux. Linux discovery fingerprints search-directory inputs,
tool binaries/dependencies, GCC specs, CRT and sanitizer runtimes, loader cache
and hardware-capability selection. Publication checks assembler dependencies,
linker input traces, and resolved executable dependencies against that closure.
Private runtime search paths, external specs/sysroots, unrecognized linker
scripts, Linux Clang/musl, Windows, and unresolved native flags compile afresh. Windows
Job Object code requires native Windows lifecycle verification in addition to
cross-compilation. Complete provider discovery and exhaustive publication-crash
injection on those platforms remain separate follow-up coverage.

Compiler launch paths are separate from canonical hashing identities: on Windows,
retain ordinary absolute paths so MinGW can locate its compiler helpers. Linux
`ldconfig` is a metadata inspector and may be a distribution-provided shell
wrapper (Ubuntu 24.04); it is not an ELF compiler/linker dependency. Its output,
wrapper, and loader cache remain fingerprinted, and actual link inputs remain
audited. Cross-platform CI emits cache diagnostics to distinguish deliberate
bypasses from failed reuse.

## Linux Under Colima

The Linux checks can run against the current checkout without sharing macOS build
artifacts. From the repository root with Colima running:

```sh
docker run -d --init --name asimplelife-colima-tests --cpus=2 --memory=2560m \
  --mount "type=bind,source=$PWD,target=/workspace,readonly" \
  --mount type=volume,source=asimplelife-colima-target,target=/target \
  --mount type=volume,source=asimplelife-colima-registry,target=/usr/local/cargo/registry \
  -e CARGO_TARGET_DIR=/target -e CARGO_BUILD_JOBS=1 -e RUST_TEST_THREADS=2 \
  -w /workspace rust:1.98-bookworm sleep infinity
docker exec asimplelife-colima-tests rustup component add clippy rustfmt
docker exec asimplelife-colima-tests cargo test --locked --lib test_support::compiled_c
docker exec asimplelife-colima-tests cargo test --locked --lib bf::tests
docker exec asimplelife-colima-tests cargo test --locked --release --no-run
docker exec asimplelife-colima-tests cargo test --locked --release
docker exec asimplelife-colima-tests cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
docker exec asimplelife-colima-tests python3 .github/scripts/verify_ci_binaries.py
docker rm -f asimplelife-colima-tests
```

The image supplies GCC and Python for the generated-C and process-lifecycle
tests. The target and registry volumes remain available for subsequent runs.
Resource ceilings change build/test concurrency, not optimization settings or
test coverage. On Apple Silicon this validates AArch64 Linux/NEON, not x86 AVX2
or musl. Harness tests require single-flight compilation, zero warm codegen/link
invocations, fresh execution, corruption recovery, and distinct debug/sanitizer
variants on the supported image. For cold/warm BF diagnostics, run twice:

```sh
docker exec -e ASIMPLELIFE_C_TEST_STATS=1 asimplelife-colima-tests \
  cargo test --locked --release --lib bf::tests -- --nocapture
```

An unchanged cache-eligible warm run must report zero `compiler_invocations`
while retaining every `execution_attempts`. Preprocessing and toolchain validation
still run and are reported separately. Unverified configurations bypass reuse.
