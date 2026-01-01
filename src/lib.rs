extern crate self as apfp;

pub mod analysis;
pub mod ap64;
pub mod ap64_fixed;
pub mod expansion;
pub mod geometry;

pub use analysis::orient2d_rational;
pub use ap64::Ap64;
pub use ap64_fixed::{Ap64Fixed, FixedExpansionOverflow};
pub use geometry::{
    Coord, GeometryPredicateResult, incircle, orient2d, orient2d_fixed, orient2d_inexact_baseline,
    orient2d_inexact_interval,
};
