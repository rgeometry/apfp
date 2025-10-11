# Adaptive Precision Floating-Point Arithmetic – Notes

## Context & Motivation
- Computational geometry algorithms lean on orientation, incircle, and insphere predicates whose sign determines combinatorial structure; naive IEEE double arithmetic lets roundoff flip these signs, yielding incoherent meshes or crashes.
- Shewchuk reframes robustness as an arithmetic problem: if predicate evaluations can be made exact (or provably accurate), the surrounding geometric algorithm can remain simple and reliable.
- Exact big-number libraries existed, but their performance penalties were too large for widespread use in geometry pipelines; the report targets software-level methods that are both exact and fast on IEEE hardware.

## Floating-Point Expansions
- A floating-point expansion expresses a value as an ordered sum of standard machine doubles whose magnitudes strictly increase; each component captures one “digit” of significance.
- Core kernels (FAST-TWO-SUM, TWO-SUM, TWO-PRODUCT, EXPANSION-SUM, FAST-EXPANSION-SUM, SCALE-EXPANSION) add or multiply expansions while exposing the roundoff of each hardware operation as an explicit component.
- Because components stay in native hardware format, conversions to custom big-number bases are avoided while still keeping error accounting explicit.
- Compression routines prune near-zero components and keep expansions short in practice (often fewer than six doubles) so the overhead stays low.

## Nonoverlapping IEEE Double Components
- Two IEEE doubles `x` and `y` are nonoverlapping if the least significant nonzero bit of the larger-magnitude operand has a magnitude strictly greater than the most significant bit of the smaller-magnitude operand.
- Example: in binary, `1.1000 × 2^k` and `1.001 × 2^(k-4)` are nonoverlapping; their significant bits occupy disjoint ranges, so adding them cannot create destructive interference or cancelation.
- Maintaining nonoverlap means the sign of an expansion equals the sign of its largest component, and crude magnitude bounds are obtained by looking at just a few leading terms.
- Algorithms like FAST-TWO-SUM guarantee that the “approximate sum” result and the residual error term are nonoverlapping when the machine obeys radix-2 round-to-even semantics, making the invariant easy to maintain.

## Adaptive Precision Workflow
- Each arithmetic kernel can be split: Line 1 produces a fast approximate result, while subsequent lines recover the precise residual when needed.
- Adaptive predicates evaluate determinants in stages (A, B, C, …): start with a hardware-precision approximation plus an error bound; if the interval excludes zero, stop; otherwise, reuse earlier partial expansions to refine the result.
- Error bounds come from forward error analysis of the expansion operations; they are cheap relative to full exact evaluation and dictate when to escalate precision.
- This staged approach often costs only a small multiple of a single hardware-evaluation when inputs are well separated, yet guarantees exact arithmetic in degenerate configurations without starting from scratch.

## Robust Geometric Predicates
- Orientation, incircle, and insphere predicates are implemented via determinant expansions (e.g., 3×3 and 4×4 matrices in 2D/3D) whose cofactors are evaluated using the expansion toolkit.
- Expression rewrites (translating coordinates, carefully ordering operations) control intermediate magnitude growth and reduce the number of expansion components.
- Detailed error tables specify how each expansion stage bounds its error in terms of input coordinate magnitudes, guiding the adaptive stopping tests.
- Benchmarks show substantial wins: the exact 2D orientation predicate runs about 13× faster than MPFUN-based implementations and only ~2× slower than Fortune & Van Wyk’s integer-restricted LN predicates while supporting arbitrary IEEE doubles.

## Performance & Practical Outcomes
- Adaptive predicates typically resolve >99% of random inputs at the first or second stage, so the average runtime is close to a filtered evaluation but worst-case is fully exact.
- Integrating these predicates into Delaunay triangulation yields robust behavior with modest overhead compared to non-robust versions; when degeneracies abound, the exact fallback prevents catastrophic failures.
- Released C code demonstrates that the approach is portable across IEEE-compliant platforms without relying on bit tricks or compiler intrinsics.

## Caveats
- Assumes binary radix with exact rounding; FAST-EXPANSION-SUM additionally needs round-to-even tie-breaking. Appendix B supplies a slower linear-time alternative for hardware lacking this rule.
- The method extends precision but not exponent range; supporting huge exponents would require multi-expansion structures or per-component exponents, which may erode performance.
- FFT-based multiplication is awkward because expansions are not in fixed-base digit form; large-precision workloads may still favor traditional arbitrary-precision libraries.
- Arithmetic alone does not cure all robustness issues—algorithm design must still avoid redundant or inconsistent predicates, and some problems (e.g., parsimonious arrangements) remain NP-hard even with exact arithmetic.
