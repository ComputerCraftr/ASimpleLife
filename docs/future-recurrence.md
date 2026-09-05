# Exact Future Interfaces and Recurrence

These are separate optimizations. Neither changes structural node identity,
snapshot semantics, or present-state inspection.

## Positive-Time Interfaces

The rule contract is synchronous binary Conway Life, B3/S23, on the radius-one
Moore neighborhood. For an 8 by 8 tile, the interface contains:

```
B2 = {(x,y) | min(x,y,7-x,7-y) < 2}
I6 = {(x,y) | 1 <= x,y <= 6}
Interface8(A) = current_A(B2) + next_A(I6)
```

The overlap is intentional. Under any identical exterior, equal `B2` gives
equal next states on the tile boundary and immediately adjacent exterior.
Equal `next(I6)` covers the rest of the tile. The complete universes are
therefore identical after one generation, and determinism preserves equality.

Smaller interfaces use exact structural identity. Larger interfaces contain
ordered child interfaces: replace children one at a time, using the preceding
common-exterior property at each replacement. This proves parent congruence
without assuming that matching sampled futures imply equivalence.

The cache answers only legal positive centered successors, after ordinary
structural/D4 caches miss. A jump exponent of zero means **one generation**.
Input/output orientation and centered output geometry remain part of the
successor contract. No interface can answer population, overlap construction,
cropping, serialization, or any other generation-zero query.

Derivation has a 4,096 child-slot work limit. Optional interface/cache storage
has a 32 MiB cap within HashLife accounting. Exact-key comparison, not hash
equality, establishes equivalence. Registry IDs are append-only until a
quiescent reset; reset discards every dependent mapping and result. Cached
results are weak references, not collection roots.

Nodes below 8 by 8 use the structural cache only: a second exact cache cannot
add reuse there. Unproductive future searches use deterministic cooldowns;
these bypasses are counted separately from proved misses and successful reuse.
Interface interior evolution reuses the packed Conway kernel, not per-cell
neighborhood reconstruction.

## Whole-Universe Recurrence

The shared tracker compares exact normalized observations from distinct
committed generations. A fingerprint only selects candidates for equality.
The anchor is `(minimum live x, minimum live y)`; empty states use `(0,0)`.

Cell witnesses carry shifted bits between chunk words and compare sorted
normalized chunks. HashLife witnesses reblock the DAG into a smallest enclosing
power-of-two square at local zero. They retain world-axis orientation, so D4
orbit equality alone cannot certify recurrence. Normalization does not extract
a cell list or full grid, and unavailable optional evidence is not simulation
failure.

Owned evidence is capped at 4,096 entries and 8 MiB. Weak DAG evidence records
its session and arena epoch and is retired when that identity domain changes.
It cannot keep an old universe alive through GC. A proved certificate instead
belongs to a source lineage; replacing the source invalidates it, while true
representation conversion preserves the lineage.

Clipped inspection cannot replace authoritative state. Only a complete,
validated representation conversion preserves both the universe and its
recurrence lineage.

If `U(g+p) = translate(d,U(g))`, determinism and translation equivariance give
the same recurrence at every later phase. For a current generation `h >= g+p`:

```
k = floor((target-h)/p)
U(h+k*p) = translate(k*d,U(h))
```

All products and coordinate calculations are checked. Translation and time
commit together only after the candidate is valid. Remaining generations run
normally. A certificate does not assert that its observed period is minimal.

Component-model discovery uses the equality verifier, but isolated component
proofs are not whole-universe skip authority. Existing component-composition
checks remain separate. Emitters, puffers, behavioral transducers, and general
contextual minimization are not implemented by this interface.
