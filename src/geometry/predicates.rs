use crate::{Ap64, Ap64Fixed};
use std::cmp::Ordering;

use super::Coord;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryPredicateResult {
    Positive,
    Negative,
    Zero,
}

impl GeometryPredicateResult {
    fn from_value(value: &Ap64) -> Self {
        #[cfg(feature = "short-circuit")]
        {
            let (lower, upper) = value.bounds();
            if lower > 0.0 {
                return GeometryPredicateResult::Positive;
            }
            if upper < 0.0 {
                return GeometryPredicateResult::Negative;
            }
        }

        if value.is_zero() {
            GeometryPredicateResult::Zero
        } else if value.approx() > 0.0 {
            GeometryPredicateResult::Positive
        } else {
            GeometryPredicateResult::Negative
        }
    }

    #[inline(always)]
    fn from_fixed<const N: usize>(value: &Ap64Fixed<N>) -> Self {
        if value.is_zero() {
            GeometryPredicateResult::Zero
        } else if value.approx() > 0.0 {
            GeometryPredicateResult::Positive
        } else {
            GeometryPredicateResult::Negative
        }
    }

    fn from_ordering(ordering: Ordering) -> Self {
        match ordering {
            Ordering::Greater => GeometryPredicateResult::Positive,
            Ordering::Less => GeometryPredicateResult::Negative,
            Ordering::Equal => GeometryPredicateResult::Zero,
        }
    }
}

pub fn orient2d(a: &Coord, b: &Coord, c: &Coord) -> GeometryPredicateResult {
    let ax = Ap64::from(a.x);
    let ay = Ap64::from(a.y);
    let bx = Ap64::from(b.x);
    let by = Ap64::from(b.y);
    let cx = Ap64::from(c.x);
    let cy = Ap64::from(c.y);

    let bax = &bx - &ax;
    let bay = &by - &ay;
    let cax = &cx - &ax;
    let cay = &cy - &ay;

    let left = &bax * &cay;
    let right = &bay * &cax;
    GeometryPredicateResult::from_ordering(left.compare(&right))
}

#[inline(always)]
fn fixed_from_f64(value: f64) -> Ap64Fixed<4> {
    let mut result = Ap64Fixed::<4>::zero();
    if value != 0.0 {
        result
            .push(value)
            .expect("Ap64Fixed<4> must have space for a single component");
    }
    result
}

pub fn orient2d_fixed(a: &Coord, b: &Coord, c: &Coord) -> GeometryPredicateResult {
    let ax = fixed_from_f64(a.x);
    let ay = fixed_from_f64(a.y);
    let bx = fixed_from_f64(b.x);
    let by = fixed_from_f64(b.y);
    let cx = fixed_from_f64(c.x);
    let cy = fixed_from_f64(c.y);

    let bax = &bx - &ax; // 2
    let bay = &by - &ay; // 2
    let cax = &cx - &ax; // 2
    let cay = &cy - &ay; // 2

    // (a + b) * (c + d) = (a + b) * c + (a + b) * d = ac + ad + bc + bd
    // (a + b) * (c + d) = ac + ad + bc + bd

    let det = (&bax * &cay) - &(&bay * &cax);
    GeometryPredicateResult::from_fixed(&det)
}

pub fn incircle(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> GeometryPredicateResult {
    let ax = Ap64::from(a.x);
    let ay = Ap64::from(a.y);
    let bx = Ap64::from(b.x);
    let by = Ap64::from(b.y);
    let cx = Ap64::from(c.x);
    let cy = Ap64::from(c.y);
    let dx = Ap64::from(d.x);
    let dy = Ap64::from(d.y);

    let adx = &ax - &dx;
    let ady = &ay - &dy;
    let bdx = &bx - &dx;
    let bdy = &by - &dy;
    let cdx = &cx - &dx;
    let cdy = &cy - &dy;

    let ad2 = (&adx * &adx) + &(&ady * &ady);
    let bd2 = (&bdx * &bdx) + &(&bdy * &bdy);
    let cd2 = (&cdx * &cdx) + &(&cdy * &cdy);

    let term1 = &adx * &((&bdy * &cd2) - &(&cdy * &bd2));
    let term2 = &ady * &((&cdx * &bd2) - &(&bdx * &cd2));
    let term3 = &ad2 * &((&bdx * &cdy) - &(&cdx * &bdy));

    let det = (&term1 + &term2) + &term3;
    GeometryPredicateResult::from_value(&det)
}
