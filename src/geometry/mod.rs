//! Geometric predicates for various numeric types.
//!
//! Each submodule provides geometric predicates for a specific numeric type,
//! with consistent APIs across all types:
//!
//! - [`orient2d`] - Orientation of point relative to directed line
//! - [`orient2d_vec`] - Orientation relative to line defined by point and direction
//! - [`orient2d_normal`] - Orientation relative to line defined by point and normal
//! - [`cmp_dist`] - Compare squared distances from origin to two points
//! - [`incircle`] - Test if point lies inside circumcircle of triangle
//!
//! # Example
//!
//! ```rust
//! use apfp::geometry::{Orientation, i32::{Coord, orient2d}};
//!
//! let a = Coord::new(0, 0);
//! let b = Coord::new(1, 0);
//! let c = Coord::new(0, 1);
//!
//! assert_eq!(orient2d(&a, &b, &c), Orientation::CounterClockwise);
//! ```

/// Result of orientation predicates.
///
/// For `orient2d(a, b, c)`:
/// - `CounterClockwise`: point c is to the left of the directed line from a to b
/// - `Clockwise`: point c is to the right of the directed line from a to b
/// - `CoLinear`: points a, b, c are collinear
///
/// For `incircle(a, b, c, d)` (assuming a, b, c are counter-clockwise):
/// - `CounterClockwise`: point d is inside the circumcircle
/// - `Clockwise`: point d is outside the circumcircle
/// - `CoLinear`: point d is on the circumcircle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// Counter-clockwise orientation (positive determinant).
    /// For incircle: point is inside the circumcircle.
    CounterClockwise,
    /// Clockwise orientation (negative determinant).
    /// For incircle: point is outside the circumcircle.
    Clockwise,
    /// Collinear points (zero determinant).
    /// For incircle: point is on the circumcircle.
    CoLinear,
}

pub mod f64;
pub mod i16;
pub mod i32;
pub mod i64;
pub mod i8;
