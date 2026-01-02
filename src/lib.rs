//! Adaptive precision floating-point arithmetic for robust geometric predicates.
//!
//! This crate provides exact sign computation for arithmetic expressions, using
//! adaptive precision that escalates only when needed. Most computations complete
//! using fast f64 arithmetic; near-degenerate cases automatically fall back to
//! higher precision.
//!
//! # Quick Start
//!
//! Use the pre-built geometric predicates:
//!
//! ```rust
//! use apfp::{orient2d, Coord, GeometryPredicateResult};
//!
//! let a = Coord::new(0.0, 0.0);
//! let b = Coord::new(1.0, 0.0);
//! let c = Coord::new(0.5, 1.0);
//!
//! match orient2d(&a, &b, &c) {
//!     GeometryPredicateResult::Positive => println!("c is left of ab"),
//!     GeometryPredicateResult::Negative => println!("c is right of ab"),
//!     GeometryPredicateResult::Zero => println!("c is on line ab"),
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
//! # Available Predicates
//!
//! - [`orient2d`]: Orientation of point relative to directed line
//! - [`incircle`]: Test if point lies inside circumcircle of triangle
//! - [`cmp_dist`]: Compare squared distances from origin to two points
//!
//! # The `apfp_signum!` Macro
//!
//! The [`apfp_signum!`] macro computes the exact sign (-1, 0, or 1) of an
//! arithmetic expression. It supports:
//!
//! - Addition (`+`) and subtraction (`-`)
//! - Multiplication (`*`)
//! - [`square()`] for squared terms (more efficient than `x * x`)
//!
//! The macro analyzes the expression at compile time to determine buffer sizes
//! and error bounds, ensuring allocation-free evaluation.

extern crate self as apfp;

#[doc(hidden)]
pub mod analysis;
pub(crate) mod expansion;
mod geometry;

pub use analysis::adaptive_signum::square;
pub use geometry::{
    Coord, GeometryPredicateResult, cmp_dist, incircle, orient2d, orient2d_normal, orient2d_vec,
};
