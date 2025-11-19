//! Numerically-filtered predicates implemented in plain `f64` with a
//! statically-derived error bound and an exact fallback.

// Public modules
pub mod ast_static;

use crate::geometry::{Coord, GeometryPredicateResult};
use num::{BigRational, Zero};
use std::cmp::Ordering;

// We use the standard floating-point error model `fl(x op y) = (x op y) * (1 + δ)`
// with `|δ| <= u`, where `u = 0.5 * f64::EPSILON` for IEEE-754 double precision.
// Chaining `n` operations gives the classic `γ_n = (n * u) / (1 - n * u)` bound
// (Higham, *Accuracy and Stability of Numerical Algorithms*), so each predicate
// pre-computes the specific `γ_n` matching its operation count.
const U: f64 = 0.5 * f64::EPSILON;
const GAMMA5: f64 = (5.0 * U) / (1.0 - 5.0 * U);
const GAMMA7: f64 = (7.0 * U) / (1.0 - 7.0 * U);
const GAMMA11: f64 = (11.0 * U) / (1.0 - 11.0 * U);

/// Fast floating-point filter for orient2d that returns None when the result
/// is too close to zero to trust its sign.
pub fn orient2d_fast(a: Coord, b: Coord, c: Coord) -> Option<GeometryPredicateResult> {
    let ax = a.x;
    let ay = a.y;
    let bx = b.x;
    let by = b.y;
    let cx = c.x;
    let cy = c.y;

    let adx = ax - cx;
    let bdx = bx - cx;
    let ady = ay - cy;
    let bdy = by - cy;

    let det = adx * bdy - ady * bdx;
    let detsum = (adx * bdy).abs() + (ady * bdx).abs();

    let errbound = GAMMA7 * detsum;
    if det > errbound {
        return Some(GeometryPredicateResult::Positive);
    }
    if det < -errbound {
        return Some(GeometryPredicateResult::Negative);
    }

    None // Too close to call, need exact arithmetic
}

/// Fast `orient2d` filter that only falls back to exact arithmetic when the
/// floating-point result is too close to zero to trust its sign.
pub fn orient2d(a: Coord, b: Coord, c: Coord) -> GeometryPredicateResult {
    match orient2d_fast(a, b, c) {
        Some(result) => result,
        None => {
            let ax = a.x;
            let ay = a.y;
            let bx = b.x;
            let by = b.y;
            let cx = c.x;
            let cy = c.y;
            orient2d_exact(ax, ay, bx, by, cx, cy)
        }
    }
}

/// Orients `c` with respect to the directed segment that starts at `a` and
/// extends by `vector`. Analytically equivalent to `orient2d(a, a + vector, c)`
/// but avoids constructing `a + vector` to reduce rounding error.
pub fn orient_vec_2d(a: &Coord, vector: &Coord, c: &Coord) -> GeometryPredicateResult {
    let ax = a.x;
    let ay = a.y;
    let cx = c.x;
    let cy = c.y;
    let vx = vector.x;
    let vy = vector.y;

    let adx = ax - cx;
    let ady = ay - cy;

    let det = adx * vy - ady * vx;
    let detsum = (adx * vy).abs() + (ady * vx).abs();

    let errbound = GAMMA5 * detsum;
    if det > errbound {
        return GeometryPredicateResult::Positive;
    }
    if det < -errbound {
        return GeometryPredicateResult::Negative;
    }

    orient_vec_2d_exact(ax, ay, vx, vy, cx, cy)
}

/// Compare which of `p` or `q` lies closer to `origin` (squared distance) using
/// the same filtered/robust strategy.
pub fn cmp_dist(origin: &Coord, p: &Coord, q: &Coord) -> Ordering {
    match cmp_dist_fast(origin, p, q) {
        Some(ordering) => ordering,
        None => cmp_dist_exact(origin.x, origin.y, p.x, p.y, q.x, q.y),
    }
}

/// Compare which of `p` or `q` lies closer to `origin` (squared distance) using
/// the same filtered/robust strategy.
pub fn cmp_dist_fast(origin: &Coord, p: &Coord, q: &Coord) -> Option<Ordering> {
    let ox = origin.x;
    let oy = origin.y;

    let px = p.x;
    let py = p.y;
    let qx = q.x;
    let qy = q.y;

    let pdx = px - ox;
    let pdy = py - oy;
    let qdx = qx - ox;
    let qdy = qy - oy;

    let pdist = pdx * pdx + pdy * pdy;
    let qdist = qdx * qdx + qdy * qdy;

    let diff = pdist - qdist;
    let sum = pdist.abs() + qdist.abs();
    let errbound = GAMMA11 * sum;

    if diff > errbound {
        Some(Ordering::Greater)
    } else if diff < -errbound {
        Some(Ordering::Less)
    } else {
        None
    }
}

fn orient2d_exact(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> GeometryPredicateResult {
    fn to_rational(value: f64) -> BigRational {
        BigRational::from_float(value).expect("inputs must be finite numbers")
    }

    let axr = to_rational(ax);
    let ayr = to_rational(ay);
    let bxr = to_rational(bx);
    let byr = to_rational(by);
    let cxr = to_rational(cx);
    let cyr = to_rational(cy);

    let adx = &axr - &cxr;
    let bdx = &bxr - &cxr;
    let ady = &ayr - &cyr;
    let bdy = &byr - &cyr;

    let det = &adx * &bdy - &ady * &bdx;
    let zero = BigRational::zero();

    if det > zero {
        GeometryPredicateResult::Positive
    } else if det < zero {
        GeometryPredicateResult::Negative
    } else {
        GeometryPredicateResult::Zero
    }
}

fn orient_vec_2d_exact(
    ax: f64,
    ay: f64,
    vx: f64,
    vy: f64,
    cx: f64,
    cy: f64,
) -> GeometryPredicateResult {
    fn to_rational(value: f64) -> BigRational {
        BigRational::from_float(value).expect("inputs must be finite numbers")
    }

    let axr = to_rational(ax);
    let ayr = to_rational(ay);
    let vxr = to_rational(vx);
    let vyr = to_rational(vy);
    let cxr = to_rational(cx);
    let cyr = to_rational(cy);

    let adx = &axr - &cxr;
    let ady = &ayr - &cyr;

    let det = &adx * &vyr - &ady * &vxr;
    let zero = BigRational::zero();

    if det > zero {
        GeometryPredicateResult::Positive
    } else if det < zero {
        GeometryPredicateResult::Negative
    } else {
        GeometryPredicateResult::Zero
    }
}

fn cmp_dist_exact(ox: f64, oy: f64, px: f64, py: f64, qx: f64, qy: f64) -> Ordering {
    fn to_rational(value: f64) -> BigRational {
        BigRational::from_float(value).expect("inputs must be finite numbers")
    }

    let oxr = to_rational(ox);
    let oyr = to_rational(oy);
    let pxr = to_rational(px);
    let pyr = to_rational(py);
    let qxr = to_rational(qx);
    let qyr = to_rational(qy);

    let pdx = &pxr - &oxr;
    let pdy = &pyr - &oyr;
    let qdx = &qxr - &oxr;
    let qdy = &qyr - &oyr;

    let pdist = &pdx * &pdx + &pdy * &pdy;
    let qdist = &qdx * &qdx + &qdy * &qdy;

    pdist.cmp(&qdist)
}

#[cfg(test)]
mod tests {
    use super::{cmp_dist, orient_vec_2d, orient2d};
    use crate::geometry::{Coord, GeometryPredicateResult};
    use std::cmp::Ordering;

    #[test]
    fn agrees_with_exact_on_simple_cases() {
        let a = Coord::new(0.0, 0.0);
        let b = Coord::new(1.0, 0.0);
        let c_left = Coord::new(0.0, 1.0);
        let c_right = Coord::new(0.0, -1.0);
        let c_collinear = Coord::new(0.5, 0.0);

        assert_eq!(orient2d(a, b, c_left), GeometryPredicateResult::Positive);
        assert_eq!(orient2d(a, b, c_right), GeometryPredicateResult::Negative);
        assert_eq!(orient2d(a, b, c_collinear), GeometryPredicateResult::Zero);
    }

    #[test]
    fn orient_vec_matches_orient2d() {
        let a = Coord::new(2.0, -1.0);
        let c = Coord::new(-3.0, 4.0);
        let vector = Coord::new(1.5, 2.25);
        let b = Coord::new(a.x + vector.x, a.y + vector.y);

        assert_eq!(orient_vec_2d(&a, &vector, &c), orient2d(a, b, c));
    }

    #[test]
    fn cmp_dist_orders_correctly() {
        let origin = Coord::new(0.0, 0.0);
        let closer = Coord::new(1e-12, -1e-12);
        let farther = Coord::new(1.0, 1.0);

        assert_eq!(cmp_dist(&origin, &farther, &closer), Ordering::Greater);
        assert_eq!(cmp_dist(&origin, &closer, &farther), Ordering::Less);
    }

    #[test]
    fn cmp_dist_handles_ties() {
        let origin = Coord::new(-10.0, 5.0);
        let p = Coord::new(2.0, 3.0);
        let q = Coord::new(2.0, 3.0);

        assert_eq!(cmp_dist(&origin, &p, &q), Ordering::Equal);
    }
}
