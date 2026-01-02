//! Adaptive precision floating-point arithmetic for robust geometric predicates.
//!
//! This crate provides exact sign computation for arithmetic expressions, using
//! adaptive precision that escalates only when needed. Most computations complete
//! using fast f64 arithmetic; near-degenerate cases automatically fall back to
//! higher precision.
//!
//! # Quick Start
//!
//! Use the type-specific modules for geometric predicates:
//!
//! ```rust
//! use apfp::geometry::f64::{Coord, orient2d, Orientation};
//!
//! let a = Coord::new(0.0, 0.0);
//! let b = Coord::new(1.0, 0.0);
//! let c = Coord::new(0.5, 1.0);
//!
//! match orient2d(&a, &b, &c) {
//!     Orientation::CounterClockwise => println!("c is left of ab"),
//!     Orientation::Clockwise => println!("c is right of ab"),
//!     Orientation::CoLinear => println!("c is on line ab"),
//! }
//! ```
//!
//! Or compute the sign of arbitrary expressions with [`apfp_signum!`]:
//!
//! ```rust
//! use apfp::{apfp_signum, square};
//!
//! let x = 3.0_f64;
//! let y = 4.0_f64;
//! let z = 5.0_f64;
//!
//! // Exact sign of x^2 + y^2 - z^2
//! let sign = apfp_signum!(square(x) + square(y) - square(z));
//! assert_eq!(sign, 0); // 9 + 16 - 25 = 0
//! ```
//!
//! # Integer Coordinates
//!
//! For integer coordinates (i8, i16, i32, i64), use the corresponding geometry modules:
//!
//! ```rust
//! use apfp::geometry::i32::{Coord, orient2d, Orientation};
//!
//! let a = Coord::new(0, 0);
//! let b = Coord::new(1, 0);
//! let c = Coord::new(0, 1);
//!
//! assert_eq!(orient2d(&a, &b, &c), Orientation::CounterClockwise);
//! ```
//!
//! Or use the [`int_signum!`] macro for arbitrary integer expressions:
//!
//! ```rust
//! use apfp::{int_signum, square};
//!
//! let x: i32 = 1000000;
//! let y: i32 = 999999;
//! // Exact sign even though x^2 would overflow i32
//! let sign = int_signum!(square(x) - square(y));
//! assert_eq!(sign, 1);
//! ```
//!
//! # Geometry Modules
//!
//! Each geometry module ([`geometry::f64`], [`geometry::i8`], [`geometry::i16`], [`geometry::i32`], [`geometry::i64`])
//! provides the same set of geometric predicates:
//!
//! - [`orient2d`](geometry::f64::orient2d) - Orientation of point relative to directed line
//! - [`orient2d_vec`](geometry::f64::orient2d_vec) - Orientation relative to line defined by point and direction
//! - [`orient2d_normal`](geometry::f64::orient2d_normal) - Orientation relative to line defined by point and normal
//! - [`cmp_dist`](geometry::f64::cmp_dist) - Compare squared distances from origin to two points
//! - [`incircle`](geometry::f64::incircle) - Test if point lies inside circumcircle of triangle
//!
//! # The `apfp_signum!` and `int_signum!` Macros
//!
//! These macros compute the exact sign (-1, 0, or 1) of arithmetic expressions.
//! They support:
//!
//! - Addition (`+`) and subtraction (`-`)
//! - Multiplication (`*`)
//! - [`square()`] for squared terms (more efficient than `x * x`)
//!
//! The macros analyze expressions at compile time to determine buffer sizes
//! and error bounds, ensuring allocation-free evaluation.

extern crate self as apfp;

#[doc(hidden)]
pub mod analysis;
pub(crate) mod expansion;
pub mod geometry;

pub use analysis::adaptive_signum::square;

// Re-export from geometry for convenience
pub use geometry::Orientation;
pub use geometry::f64::Coord;
