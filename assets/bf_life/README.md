# BF-to-Life Physical Component Assets

## Current Status

No executable component library is currently vendored in this directory.

The existing Rust `life_macro_library` patterns are test scaffolding only. They
have not been independently verified as clocks, latches, logic gates, routing
elements, or output transducers and must not be used by production compilation.

Production BF-to-Life compilation therefore returns
`ExecutableCircuitUnavailable`. This is intentional fail-closed behavior, not a
completed backend.

This applies to `bf_life --emit-life`, `--emit-life-hashlife`, application `--bf`,
and the TUI BF-source dialog. Failed preparation leaves an existing interactive
universe unchanged. Plain C and super-C compilation remain available, including
their shared bounded polynomial analysis and native arithmetic kernels; those
host/compiler optimizations do not supply physical Life components.

`CompiledLifeProgram` separates immutable initial-grid and observation metadata
from reference execution. Having that type, a decoder contract, or passing
registry-validation tests is not proof of independently executing BF in Life.

## Registry Contract

`manifest.json` is the machine-readable production registry. The registry
loader validates repository-local paths and pattern SHA-256 digests, parses RLE
or coordinate patterns, and checks provenance, licensing, ports, bounds,
period, phase, isolation, and independent-verification evidence. It exposes
component lookup only after every required v1 kind passes those checks. The
manifest currently remains `blocked` with no components; no placeholder or
test-scaffold pattern is represented as executable.

The registry schema is `asimplelife/bf-life-assets/v1`. The current manifest
requires unsigned 8-bit cells, a circular 64-cell tape, and no input. Validation
and unsupported-option errors remain explicit; no reference-assisted output is
accepted as an executable implementation.

## Required Evidence

Before enabling compilation, each repository-local component must provide:

- its original source and unmodified pattern data;
- author, source URL, license, and redistribution terms;
- a cryptographic digest of the imported source;
- named input/output ports, orientation, phase, period, and bounding envelope;
- collision and isolation spacing requirements;
- independently evolved scalar-Life truth-table and long-horizon tests.

The required v1 library includes a clock, glider conduits and delays, splitters,
crossovers, eaters, dual-rail latches, increment/decrement logic, zero detection,
conditional routing, halt storage, and an eight-bit framed glider output path.

The manifest additionally names a head-token mover as a required component.
Its current output contract is one synchronization pulse followed by eight
LSB-first data pulses. These are requirements, not implemented assets.

## Remaining Integration Work

- Vendor compatible, provenance-checked patterns for every missing component.
- Prove component truth tables, routing isolation, clock timing, and long-term
  stability using independent Life evolution.
- Compose the physical tape, head, program counter, conditional branches,
  arithmetic, and halt/divergence behavior; verify complete programs against a
  separate BF oracle without using that oracle to construct the emitted grid.
- Resolve output observation during large HashLife jumps. Transient glider
  crossings can be missed by final-state sampling. A persistent physical log or
  another independently verified observation strategy is still required before
  arbitrary-jump BF execution can be advertised.

Generic `# hashlife-snapshot v1` files preserve a Life universe, not BF source,
physical decoder configuration, or output history. Do not treat a generic
snapshot load as resuming a BF workspace.

See the [project README](../../README.md) for working compiler commands,
simulation/TUI behavior, and verification gates.
