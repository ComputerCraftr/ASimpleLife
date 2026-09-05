# ASimpleLife

Conway's Game of Life with a terminal UI, sparse cell stepping, HashLife
generation jumps, optional classification, and Brainfuck-to-C compilers.
Simulation, classification, and compiler performance measurements are separate
tools; a classifier prediction is not a substitute for evolving the universe.

## Build And Run

Use current stable Rust with Cargo (Rust 2024 edition). A C compiler is needed
for the Tree-sitter C grammar and generated-C tests; sanitizer tests additionally
need a compiler with ASan/UBSan support. Nightly is needed only for Miri.

```sh
cargo build --release --locked
cargo run --release --bin a_simple_life -- --pattern r_pentomino
cargo run --release --bin a_simple_life -- --pattern random --seed 42
```

Attached stdin and stdout with no explicit `--steps` start an unlimited TUI.
`--headless` selects finite mode, as does `--steps` in automatic mode. Explicit
`--tui` overrides automatic mode selection, including an explicit frame count.
Non-TTY automatic execution retains the 200-frame default. `--tui` and `--headless` are
mutually exclusive; `--classify` is a separate noninteractive operation.

```sh
cargo run --release --bin a_simple_life -- --headless --pattern block --steps 2 --delay-ms 0
cargo run --release --bin a_simple_life -- --headless --pattern r_pentomino --target-generation 100000000 --steps 1 --delay-ms 0
cargo run --release --bin a_simple_life -- --tui --load universe.hls
```

`--steps` counts displayed frames, not an absolute generation. The first frame
shows the initial or requested target generation; `--step-generations` controls
the interval between frames. `--target-generation` is an absolute startup target,
including when loading a saved generation. Targets before a loaded generation
are rejected. Finite rendering currently still emits terminal control sequences
for named seeds; it is not a machine-readable event stream.

Named seeds include block, blinker, pulsar, glider, diehard, acorn, r_pentomino,
Gosper glider gun, glider-producing switch engine, and blinker puffer. Use
`--help` on each binary for its complete flag list.

## Interactive Session

The simulation worker owns committed state. Exact controls use an ordered queue;
viewport requests and render frames use latest-value mailboxes so a slow terminal
can drop stale frames without dropping exact advance commands.

| Key | Action |
| --- | --- |
| `Space` | Pause or resume |
| `.` | Advance exactly one generation |
| `f` / `g` | Enter a relative delta / absolute generation target |
| `[` / `]` | Halve / double continuous generation quantum, capped at `2^63` |
| `-` / `+` | Slower / faster frame pacing |
| `a` | Toggle automatic viewport tracking |
| Arrows / Shift+Arrows | Pan one / ten display cells and enter manual mode |
| `Home` | Recenter once |
| `Tab` / `Shift+Tab` | Next / previous active group |
| `n` / `o` | Prepare a named seed / open a saved grid or snapshot |
| `b` | Enter BF source or a BF file; currently reports the unavailable Life backend |
| `c` | Choose whole-universe / isolated-region classification, or cancel active analysis |
| `Ctrl-S` | Save a HashLife snapshot |
| `?`, `PgUp`, `PgDn` | Help and status scrolling |
| `q` / `Ctrl-C` | Exit |
| `Esc` | Close a dialog/help or cancel pending preparation; otherwise exit |

Relative and absolute targets are interpreted at worker acceptance, not against
a stale UI frame. An already-passed absolute target reports an error rather than
rewinding. New sources are prepared separately; a failed load or BF compilation
does not replace the current universe. Generation display reflects committed
progress; execution is bounded by `u64` generations, geometry, and resources,
not by a preset interactive horizon.

The TUI restores raw mode, alternate screen, and cursor on normal shutdown and
Rust unwinding. Unix signal exits preserve their cause: SIGINT/Ctrl-C returns
130, SIGTERM 143, and SIGHUP 129. Normal `q` exits with 0. Windows uses the
`ctrlc` termination handler. Restoration is not possible after SIGKILL or abort.

## Viewport Controls

The interactive TUI automatically follows the largest discovered active spatial
group, ignoring still lifes. Population is smoothed across observations and a
sustained lead is required before switching, so oscillator phases do not cause
the camera to alternate between distant patterns.

- `Tab` / `Shift+Tab`: next / previous active group, wrapping around the list.
- In auto mode, navigation pins tracking to the selected group until it becomes
  inactive or disappears. Toggle `a` off and on to return to largest-group tracking.
- In manual mode, navigation moves once without enabling auto tracking.
- Arrows pan manually; Shift+Arrows pan farther. `Home` recenters once.
- `?` shows all controls; `PgUp` / `PgDn` scroll status and help.

Activity discovery is read-only and budgeted. On very large universes the status
reports incomplete discovery, and selection ranks the groups found so far rather
than claiming the globally largest group. A group is a connected region of
occupied or changing 8x8 tiles, not a classifier-assigned organism identity.

Resize retains the world-space center, and stale samples from an old viewport
revision are discarded. HashLife rendering uses clipped region sampling instead
of extracting the whole universe. Manual mode does not recenter each generation.

## HashLife And Persistence

HashLife retains a memoized DAG for large jumps, with exact D4 structural
canonicalization, orientation/automorphism caching, and epoch-aware handles.
Storage IDs and fingerprints are not structural ordering keys. Scalar fallbacks
remain authoritative where an exact SIMD decision is unavailable; native and
scalar work counters are reported separately.

The engine includes allocation accounting, pressure-driven GC, and fallible
session advancement. Successful segments commit exact progress; resource or
geometry failures must be handled using the reached generation, not the requested
delta. Population APIs distinguish `Exact(u128)` from a saturated lower bound.
Tracked allocation budgets are not process-RSS guarantees, and compressible
patterns do not imply a universal completion-time guarantee for arbitrary seeds.

Two versioned text formats are supported:

- `# life-grid v1` stores explicit live coordinates.
- `# hashlife-snapshot v1` stores generation, origin, root, and topological DAG
  records without expanding the complete state into cells.

TUI saves capture committed state, then write a sibling temporary file before
rename. Direct snapshot loads restore HashLife state; input validation checks
geometry, levels, references, and resource limits. Snapshots are simulation-only:
they do not preserve viewport settings, classifier reports, or BF decoding context.

## Optional Classification

```sh
cargo run --release --bin a_simple_life -- --pattern pulsar --classify --max-generations 512
cargo run --release --bin classify_bench -- --families smallbox --cases-per-family 2 --seed 42 --json
```

The report API separates exact outcomes (extinct, still life, oscillator,
spaceship), heuristic growth outcomes (emitter, puffer, expanding), and unresolved
cases. Evidence and observation horizon accompany the result. Heuristic evidence
does not prove an infinite future.

TUI classification captures an exact owned DAG at a committed generation. The
region option freezes the camera rectangle, not terminal pixels or historical
cells; its exterior starts dead and evolves without clipping or wrapping.
Results identify their scope and capture generation and remain historical
evidence while simulation continues. Analysis never controls the camera or clock.
Capture has a 16 MiB / 65,536-visit budget and an 8 ms cooperative deadline.
The subsystem has a 128 MiB aggregate logical budget; replacements wait for
cancelled work to release its storage. Cancellation is cooperative at HashLife
scheduler boundaries. Capture can briefly delay simulation, and independent
analysis still competes for CPU/memory. Bounded grid heuristics run only on the
independent state; resource exhaustion is reported rather than truncating input.
Normal benchmark mismatches fail; `--diagnostic` explicitly permits reporting
mismatches without making them a successful correctness claim.

## Brainfuck Compilers

`bf_life` parses structured BF IR and uses one semantics-aware optimizer. Despite
the binary name, the currently executable compiler targets are plain C and
super C, not physical Life circuits.

```sh
cargo run --release --bin bf_life -- --emit-ir -- '+++[->+<]>.'
cargo run --release --bin bf_life -- --emit-ir-format json --emit-ir-section optimized -- '+++[->+<]>.'
cargo run --release --bin bf_life -- --emit-c-super -- '++++++++[>++++++++<-]>+.' > /tmp/bf-program.c
cc -O3 /tmp/bf-program.c -o /tmp/bf-program
/tmp/bf-program
```

- Text IR is the default: summary, selected sections, and bounded elision
  (80 lines and depth 8 by default). `--emit-ir-no-elide` requests the full tree.
- JSON IR uses `bf-ir-report/v3`, structured nodes, section counts, and truncation
  metadata. Section selection and explicit line/depth limits also apply.
- Rich reductions include clear-at-offset, affine transfers, shifts, distribute,
  square, multiply-add, and scans. Unsupported proofs retain the original loop.
- Shared modular polynomial analysis supports degree eight, cancellation, common
  products, repeated squares, and bounded lazy powers. Each selected runtime power
  reads the state left by the preceding power; guard proofs remain mandatory.
- Compilation-wide symbolic work and emission limits bound optional analysis.
  Exceeding them preserves fallback execution instead of changing semantics.
- Super C combines proven symbolic regions with bounded memoization and guarded
  runtime reduction. Runtime tracing is not an unrestricted polynomial solver.
- Both C backends dispatch eligible independent operations to scalar, AVX2, or
  NEON kernels. Wrapped aliases and dependencies prohibit unsafe batching.
  Test fuel uses source-ordered fallback to preserve the exact observable prefix.

Cells default to unsigned 8-bit values on a circular 30,000-cell C tape.
`--cell-bits`, `--input-bits`, and `--output-bits` accept 0 through 63; larger widths
are rejected, not clamped. `--signed-cells true` changes signed interpretation,
not raw-bit wrapping arithmetic or logical right-shift behavior. `--io number`
selects numeric I/O. Generated executables append runtime work statistics after
an explicit `=== BF PLAIN RUNTIME STATS ===` or `=== BF SUPER RUNTIME STATS ===`
boundary; those diagnostics are separate from BF payload output.

### BF-to-Life Status

`--emit-life`, `--emit-life-hashlife`, application `--bf`, and the TUI BF dialog
fail closed with `ExecutableCircuitUnavailable` after supported-option checks.
No verified executable asset library is vendored. Shared analysis, immutable
program metadata, and component validation infrastructure are not a working
physical machine. See [the asset README](assets/bf_life/README.md) for blockers.

## Verification And Maintenance

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo test --workspace --all-features --release --locked
cargo +nightly miri test --lib bf::coefficient_kernels::tests -- --test-threads=1
cargo run --release --bin hashlife_perf_gate
```

The unified CI workflow covers current macOS/Linux/Windows runners, static
musl/MSVC CRT checks, native and forced-scalar kernels, generated-C sanitizers,
AST source policies, strict Clippy, and targeted nightly Miri. Release builds
unwind rather than abort. Cargo config enables MSVC static CRT; musl's target
default is static, and crate guards reject disabling either contract.

The versioned HashLife performance corpus checks work and state invariants.
Its five-second / 1-GiB gate applies only on the designated Apple M2 runner, not
all machines or arbitrary inputs. Empty pattern-filter matches fail. Native
coverage counters cannot be satisfied by scalar work.

Cargo.lock records reproducible resolutions. Dependabot checks Cargo and GitHub
Actions weekly, including major updates for review. Direct dependencies track
current stable releases, not prereleases; upstream transitive constraints may
retain older versions. Source tests do not freeze exact action patch versions.

Rust source contracts use Syn and C contracts use Tree-sitter, with shared
repository discovery rather than per-module source-path lists. Generated-C tests
query syntax and typed planner decisions; instrumentation edits a unique AST
node, not the first matching comment or string. Malformed C, ambiguous edits,
conditional template defaults, and `cfg_attr` scope have regression fixtures.
Template includes remain owned by the emitters; tests do not depend on the
process working directory to load them. These syntax checks do not replace
generated-C compilation or runtime differential tests.

### Generated-C Test Work

The BF harness caches validated compilation artifacts, never program outputs.
Each test executes a private copy with its original strict, optimized, sanitizer,
scalar/native, and fuel settings. Advisory build locks deduplicate identical
requests; artifact leases protect copying against eviction. Compiler and program
processes share weighted admission limits and own their descendant process trees.
Output limits and timeouts fail tests rather than truncating successful results.

The cache lives under Cargo's actual target directory and is capped at 2 GiB,
including reserved workspaces. It is shared between debug and release tests.
Current persistent discovery supports the closed generated-C argument surface on
sealed macOS installations with an identifiable Apple SDK/toolchain and native
GNU GCC/GNU ld/glibc Linux toolchains, including Linux under Colima. Linux seals
compiler helpers, search inputs, CRT and runtime libraries, then audits actual
assembler/linker/loader dependencies before publication. Other toolchains,
unknown flags (including unresolved `-march=native`), or incomplete dependency
discovery compile afresh. These conservative bypasses do not skip execution.

```sh
ASIMPLELIFE_C_TEST_STATS=1 cargo test --release --lib bf::tests -- --nocapture
ASIMPLELIFE_C_TEST_CACHE_DISABLE=1 cargo test --release --lib bf::tests
ASIMPLELIFE_C_TEST_CACHE_CLEAN=1 cargo test --release --lib bf::tests
```

`ASIMPLELIFE_C_TEST_CACHE_DIR` selects a separate cache directory. The clean
control removes unlocked entries once per process and refuses to destroy active
work. `C_TEST_WORK` records distinguish compilation outcomes and runtime attempts;
warm reuse still includes preprocessing, dependency validation, and execution.
Do not restore executable caches from untrusted CI runs or other users.
