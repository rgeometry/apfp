extern crate self as apfp;

#[doc(hidden)]
pub mod analysis;
pub(crate) mod expansion;
mod geometry;

pub use analysis::adaptive_signum::square;
pub use geometry::{Coord, GeometryPredicateResult, cmp_dist, incircle, orient2d};
