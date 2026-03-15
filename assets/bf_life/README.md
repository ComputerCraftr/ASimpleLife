# BF-to-Life Physical Component Assets

No executable component library is currently vendored in this directory.

The existing Rust `life_macro_library` patterns are test scaffolding only. They
have not been independently verified as clocks, latches, logic gates, routing
elements, or output transducers and must not be used by production compilation.

Production BF-to-Life compilation therefore returns
`ExecutableCircuitUnavailable`. This is intentional fail-closed behavior, not a
completed backend.

`manifest.json` is the machine-readable production registry. The registry
loader validates repository-local paths and pattern SHA-256 digests, parses RLE
or coordinate patterns, and checks provenance, licensing, ports, bounds,
period, phase, isolation, and independent-verification evidence. It exposes
component lookup only after every required v1 kind passes those checks. The
manifest currently remains `blocked` with no components; no placeholder or
test-scaffold pattern is represented as executable.

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
