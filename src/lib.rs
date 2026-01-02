extern crate self as apfp;

pub mod analysis;
pub mod expansion;
pub mod geometry;

pub use analysis::adaptive_signum::square;
pub use geometry::{Coord, GeometryPredicateResult, cmp_dist, incircle, orient2d};
