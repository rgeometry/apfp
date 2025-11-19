mod coord;
mod predicates;

pub use coord::Coord;
pub use predicates::{
    GeometryPredicateResult, incircle, orient2d, orient2d_fixed, orient2d_inexact_baseline,
    orient2d_inexact_interval,
};
