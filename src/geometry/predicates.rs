use crate::{Coord, apfp_signum};
use std::cmp::Ordering;

/// Result of a geometric predicate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryPredicateResult {
    Positive,
    Negative,
    Zero,
}

/// Compute orientation of point c relative to line ab.
/// Returns Positive if c is to the left of ab (counter-clockwise),
/// Negative if to the right (clockwise), Zero if collinear.
pub fn orient2d(a: &Coord, b: &Coord, c: &Coord) -> GeometryPredicateResult {
    let sign = apfp_signum!((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x));
    match sign {
        1 => GeometryPredicateResult::Positive,
        -1 => GeometryPredicateResult::Negative,
        0 => GeometryPredicateResult::Zero,
        _ => unreachable!(),
    }
}

/// Compare squared distances from origin to p and q.
/// Returns Greater if dist(origin,p) > dist(origin,q), etc.
pub fn cmp_dist(origin: &Coord, p: &Coord, q: &Coord) -> Ordering {
    let sign = apfp_signum!(
        (square(p.x - origin.x) + square(p.y - origin.y))
            - (square(q.x - origin.x) + square(q.y - origin.y))
    );
    match sign {
        1 => Ordering::Greater,
        -1 => Ordering::Less,
        0 => Ordering::Equal,
        _ => unreachable!(),
    }
}

/// Compute whether point d lies inside the circumcircle of triangle abc.
/// Returns Positive if inside, Negative if outside, Zero if on the circle.
pub fn incircle(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> GeometryPredicateResult {
    let adx = a.x - d.x;
    let ady = a.y - d.y;
    let bdx = b.x - d.x;
    let bdy = b.y - d.y;
    let cdx = c.x - d.x;
    let cdy = c.y - d.y;

    let det = apfp_signum!(
        adx * (bdy * (square(cdx) + square(cdy)) - cdy * (square(bdx) + square(bdy)))
            + ady * (cdx * (square(bdx) + square(bdy)) - bdx * (square(cdx) + square(cdy)))
            + (square(adx) + square(ady)) * (bdx * cdy - cdx * bdy)
    );

    match det {
        1 => GeometryPredicateResult::Positive,
        -1 => GeometryPredicateResult::Negative,
        0 => GeometryPredicateResult::Zero,
        _ => unreachable!(),
    }
}
