# Adaptive Precision Floating-Point Arithmetic Library

This library offers adaptive-precision floating-point arithmetic routines for applications that demand robustness without sacrificing performance. It builds on the principles popularized by Shewchuk’s floating-point expansions, exposing fast kernels for exact addition, multiplication, scaling, and compression of IEEE double components. The result is a flexible toolkit that makes it practical to lift many algorithms to exact or tightly-bounded arithmetic while reusing standard hardware operations.

## Why Use This Library?
- **Robust geometric predicates**: Determine the orientation of 2D/3D point sets or test whether a point lies inside a circle or sphere without the usual roundoff pitfalls.
- **General-purpose arithmetic**: Unlike crates such as `robust`, these kernels are not limited to pre-built geometric predicates; you can assemble arbitrary polynomial expressions, maintain error bounds, or evaluate determinants of your own design.
- **Adaptive staging**: Each primitive returns both an approximate result and its residual error, so you can escalate precision only when a calculation is inconclusive—ideal for performance-critical code that occasionally encounters degeneracies.
- **Portable & efficient**: All operations rely on IEEE 754 binary floating-point with exact rounding (round-to-even preferred). No exotic hardware features or extended precision types are required.

## Getting Started
Add the crate to your `Cargo.toml`:

```toml
[dependencies]
apfp = { path = "." }
```

Then import the modules you need:

```rust
use apfp::expansion::{two_sum, fast_two_sum};
use apfp::predicates::{orient2d, incircle};
```

Each primitive returns a small expansion (ordered list of doubles); you can pass these expansions to downstream operations or compress them when you only need a bounded precision result.

## Example: Orientation Test

```rust
use apfp::predicates::orient2d;

let a = (0.0, 0.0);
let b = (1.0, 0.0);
let c = (0.9999999999999999, 1e-16);

let orientation = orient2d(a, b, c);
if orientation.is_positive() {
    println!("Counter-clockwise");
} else if orientation.is_negative() {
    println!("Clockwise");
} else {
    println!("Collinear");
}
```

`orient2d` escalates from a fast hardware evaluation to exact arithmetic only when the input configuration is near-degenerate, so most calls run at the speed of ordinary double arithmetic.

## Example: Building Your Own Expressions

```rust
use apfp::expansion::{two_sum, scale_expansion, compress};

// Compute (x + y) * alpha with explicit control over rounding error.
let (sum_hi, sum_lo) = two_sum(x, y);
let expansion = scale_expansion(&[sum_lo, sum_hi], alpha);
let normalized = compress(&expansion);
```

Because every step preserves nonoverlapping components, you can safely combine these results into more complex formulas or derive your own adaptive predicates.

## Requirements & Caveats
- Assumes IEEE 754 binary arithmetic with exact rounding; round-to-even tie-breaking is recommended for the fastest algorithms (alternatives are provided when unavailable).
- Expansions extend precision but not exponent range; exceptionally large or tiny magnitudes may require splitting numbers first.
- FFT-style multiplication is not provided; for very high precision workloads you may prefer a traditional big-number library.

## Learn More
- The accompanying `SHEWCHUK.md` file summarizes the foundational paper “Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates”.
- API documentation and detailed module guides are available through `cargo doc`.

## License
This project follows the license stated in `Cargo.toml`. Contributions and issues are welcome. Please include failing examples or benchmarks when reporting numerical robustness questions.
