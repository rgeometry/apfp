use crate::Ap64;

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

    let det = (&bax * &cay) - &(&bay * &cax);
    GeometryPredicateResult::from_value(&det)
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
