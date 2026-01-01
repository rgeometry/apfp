extern crate self as apfp;

pub mod analysis;
pub mod expansion;
pub mod geometry;

pub use analysis::adaptive_signum::square;
pub use geometry::{cmp_dist, incircle, Coord, GeometryPredicateResult, orient2d};
