# apfp - Adaptive Precision Floating-Point

Robust geometric predicates using adaptive precision arithmetic. The library provides exact sign computation for arithmetic expressions, automatically escalating precision only when needed.

## Features

- **Robust geometric predicates**: `orient2d`, `incircle`, `cmp_dist` that handle near-degenerate cases correctly
- **General-purpose sign computation**: The `apfp_signum!` macro computes exact signs for arbitrary arithmetic expressions
- **Adaptive staging**: Fast f64 evaluation with error bounds, falling back to double-double or exact arithmetic only when necessary
- **Allocation-free**: All operations use fixed stack buffers computed at compile time

## Quick Start

```toml
[dependencies]
apfp = "0.1"
```

### Using Pre-built Predicates

```rust
use apfp::geometry::f64::{Coord, orient2d, cmp_dist, Orientation};
use std::cmp::Ordering;

let a = Coord::new(0.0, 0.0);
let b = Coord::new(1.0, 0.0);
let c = Coord::new(0.5, 1e-16);

// Orientation test: is c left of, right of, or on line ab?
match orient2d(&a, &b, &c) {
    Orientation::CounterClockwise => println!("Counter-clockwise"),
    Orientation::Clockwise => println!("Clockwise"),
    Orientation::CoLinear => println!("Collinear"),
}

// Distance comparison: is p closer to origin than q?
let origin = Coord::new(0.0, 0.0);
let p = Coord::new(1.0, 0.0);
let q = Coord::new(0.0, 1.0);
match cmp_dist(&origin, &p, &q) {
    Ordering::Less => println!("p is closer"),
    Ordering::Greater => println!("q is closer"),
    Ordering::Equal => println!("equidistant"),
}
```

### Building Custom Expressions

Use `apfp_signum!` to compute the exact sign of any arithmetic expression:

```rust
use apfp::{apfp_signum, square};

// Compute sign of a determinant
let a = 1.0_f64;
let b = 2.0_f64;
let c = 3.0_f64;
let d = 4.0_f64;
let sign = apfp_signum!(a * d - b * c);
assert_eq!(sign, -1); // 1*4 - 2*3 = -2 < 0

// Use square() for squared terms (more efficient than x * x)
let x = 3.0_f64;
let y = 4.0_f64;
let z = 5.0_f64;
let sign = apfp_signum!(square(x) + square(y) - square(z));
assert_eq!(sign, 0); // 9 + 16 - 25 = 0
```

The macro supports `+`, `-`, `*`, and `square()` operations on `f64` values.

## How It Works

The library implements Shewchuk's adaptive precision arithmetic:

1. **Fast path**: Evaluate in f64 and compute an error bound. If the result magnitude exceeds the bound, return immediately.
2. **Double-double path**: Re-evaluate using double-double arithmetic (~106 bits). Check against tighter bounds.
3. **Exact path**: Compute the exact result using floating-point expansions.

Most calls complete in the fast path. The adaptive approach provides both correctness and performance.

## Performance

On random inputs, `orient2d` performs within 1.1x of the `geometry-predicates` crate (which uses the same underlying algorithm). The macro-based approach allows the compiler to inline and optimize the entire computation.

## Requirements

- IEEE 754 binary floating-point with round-to-nearest-even
- Rust 2024 edition

## License

This project is released under [The Unlicense](LICENSE).
