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
#[inline]
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
#[inline]
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

/// Compute orientation of point `p` relative to a line through `a` in direction `v`.
///
/// This is equivalent to `orient2d(a, a + v, p)` but avoids precision loss
/// that would occur from computing `a + v` in floating-point arithmetic.
///
/// Returns Positive if p is to the left of the directed line (counter-clockwise),
/// Negative if to the right (clockwise), Zero if collinear.
///
/// # Mathematical derivation
/// Given `orient2d(a, b, c) = (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)`,
/// substituting `b = a + v` yields:
/// ```text
/// (a.x - p.x) * (a.y + v.y - p.y) - (a.y - p.y) * (a.x + v.x - p.x)
/// = (a.x - p.x) * v.y - (a.y - p.y) * v.x
/// ```
pub fn orient2d_vec(a: &Coord, v: &Coord, p: &Coord) -> GeometryPredicateResult {
    let sign = apfp_signum!((a.x - p.x) * v.y - (a.y - p.y) * v.x);
    match sign {
        1 => GeometryPredicateResult::Positive,
        -1 => GeometryPredicateResult::Negative,
        0 => GeometryPredicateResult::Zero,
        _ => unreachable!(),
    }
}

/// Compute orientation of point `p` relative to a line through `a` with normal `n`.
///
/// The line is defined by a point `a` and a normal vector `n` (perpendicular to the line).
/// This is equivalent to `orient2d_vec(a, (-n.y, n.x), p)` but expressed directly in
/// terms of the normal vector.
///
/// Returns Positive if p is on the side the normal points to,
/// Negative if on the opposite side, Zero if on the line.
///
/// # Mathematical derivation
/// If `n = (nx, ny)` is the normal, then the direction vector is `v = (-ny, nx)`.
/// Substituting into `orient2d_vec`:
/// ```text
/// (a.x - p.x) * v.y - (a.y - p.y) * v.x
/// = (a.x - p.x) * n.x - (a.y - p.y) * (-n.y)
/// = (a.x - p.x) * n.x + (a.y - p.y) * n.y
/// ```
/// This is the dot product of `(a - p)` with `n`.
#[inline]
pub fn orient2d_normal(a: &Coord, n: &Coord, p: &Coord) -> GeometryPredicateResult {
    let sign = apfp_signum!((a.x - p.x) * n.x + (a.y - p.y) * n.y);
    match sign {
        1 => GeometryPredicateResult::Positive,
        -1 => GeometryPredicateResult::Negative,
        0 => GeometryPredicateResult::Zero,
        _ => unreachable!(),
    }
}

/// Compute whether point d lies inside the circumcircle of triangle abc.
/// Returns Positive if inside, Negative if outside, Zero if on the circle.
#[inline]
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
