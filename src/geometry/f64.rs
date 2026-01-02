//! Geometric predicates for f64 coordinates.
//!
//! These predicates use adaptive precision arithmetic to provide exact results
//! for all inputs, while remaining fast for the common case.

pub use super::Orientation;
use crate::apfp_signum;
use std::cmp::Ordering;

/// 2D coordinate with f64 components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl Coord {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<(f64, f64)> for Coord {
    fn from(value: (f64, f64)) -> Self {
        Coord::new(value.0, value.1)
    }
}

/// Compute orientation of point c relative to line ab.
///
/// Returns CounterClockwise if c is to the left of ab,
/// Clockwise if to the right, CoLinear if collinear.
#[inline]
pub fn orient2d(a: &Coord, b: &Coord, c: &Coord) -> Orientation {
    let sign = apfp_signum!((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x));
    match sign {
        1 => Orientation::CounterClockwise,
        -1 => Orientation::Clockwise,
        0 => Orientation::CoLinear,
        _ => unreachable!(),
    }
}

/// Compute orientation of point `p` relative to a line through `a` in direction `v`.
///
/// This is equivalent to `orient2d(a, a + v, p)` but avoids precision loss
/// that would occur from computing `a + v` in floating-point arithmetic.
///
/// Returns CounterClockwise if p is to the left of the directed line,
/// Clockwise if to the right, CoLinear if collinear.
#[inline]
pub fn orient2d_vec(a: &Coord, v: &Coord, p: &Coord) -> Orientation {
    let sign = apfp_signum!((a.x - p.x) * v.y - (a.y - p.y) * v.x);
    match sign {
        1 => Orientation::CounterClockwise,
        -1 => Orientation::Clockwise,
        0 => Orientation::CoLinear,
        _ => unreachable!(),
    }
}

/// Compute orientation of point `p` relative to a line through `a` with normal `n`.
///
/// The line is defined by a point `a` and a normal vector `n` (perpendicular to the line).
/// This is equivalent to `orient2d_vec(a, (-n.y, n.x), p)` but expressed directly in
/// terms of the normal vector.
///
/// Returns CounterClockwise if p is on the side the normal points away from,
/// Clockwise if on the side the normal points to, CoLinear if on the line.
#[inline]
pub fn orient2d_normal(a: &Coord, n: &Coord, p: &Coord) -> Orientation {
    let sign = apfp_signum!((a.x - p.x) * n.x + (a.y - p.y) * n.y);
    match sign {
        1 => Orientation::CounterClockwise,
        -1 => Orientation::Clockwise,
        0 => Orientation::CoLinear,
        _ => unreachable!(),
    }
}

/// Compare squared distances from origin to p and q.
///
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

/// Compute whether point d lies inside the circumcircle of triangle abc.
///
/// Returns CounterClockwise if inside, Clockwise if outside, CoLinear if on the circle.
///
/// Note: This assumes the triangle abc is counter-clockwise. If clockwise,
/// the results are reversed.
#[inline]
pub fn incircle(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> Orientation {
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
        1 => Orientation::CounterClockwise,
        -1 => Orientation::Clockwise,
        0 => Orientation::CoLinear,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orient2d_ccw() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(1.0, 0.0);
        let c = Coord::new(0.0, 1.0);
        assert_eq!(orient2d(&a, &b, &c), Orientation::CounterClockwise);
    }

    #[test]
    fn orient2d_cw() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(0.0, 1.0);
        let c = Coord::new(1.0, 0.0);
        assert_eq!(orient2d(&a, &b, &c), Orientation::Clockwise);
    }

    #[test]
    fn orient2d_collinear() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(1.0, 1.0);
        let c = Coord::new(2.0, 2.0);
        assert_eq!(orient2d(&a, &b, &c), Orientation::CoLinear);
    }

    #[test]
    fn incircle_inside() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(1.0, 0.0);
        let c = Coord::new(0.0, 1.0);
        let d = Coord::new(0.25, 0.25);
        assert_eq!(incircle(&a, &b, &c, &d), Orientation::CounterClockwise);
    }

    #[test]
    fn incircle_outside() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(1.0, 0.0);
        let c = Coord::new(0.0, 1.0);
        let d = Coord::new(2.0, 2.0);
        assert_eq!(incircle(&a, &b, &c, &d), Orientation::Clockwise);
    }

    #[test]
    fn incircle_on_circle() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(1.0, 0.0);
        let c = Coord::new(0.0, 1.0);
        let d = Coord::new(1.0, 1.0);
        assert_eq!(incircle(&a, &b, &c, &d), Orientation::CoLinear);
    }
}
