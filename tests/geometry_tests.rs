use apfp::{Coord, GeometryPredicateResult, incircle, orient2d};
use ntest::timeout;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use quickcheck::{QuickCheck, TestResult};

const MAG_LIMIT: f64 = 1.0e6;
const QC_TESTS: u64 = 300;
const QC_MAX_TESTS: u64 = 20_000;

#[test]
fn orient2d_detects_ccw() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    assert_eq!(orient2d(&a, &b, &c), GeometryPredicateResult::Positive);
}

#[test]
fn orient2d_detects_cw() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(0.0, 1.0);
    let c = Coord::new(1.0, 0.0);
    assert_eq!(orient2d(&a, &b, &c), GeometryPredicateResult::Negative);
}

#[test]
fn orient2d_detects_collinear() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 1.0);
    let c = Coord::new(2.0, 2.0);
    assert_eq!(orient2d(&a, &b, &c), GeometryPredicateResult::Zero);
}

#[test]
fn incircle_positive_for_point_inside() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(0.25, 0.25);
    assert_eq!(incircle(&a, &b, &c, &d), GeometryPredicateResult::Positive);
}

#[test]
fn incircle_negative_for_point_outside() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(2.0, 2.0);
    assert_eq!(incircle(&a, &b, &c, &d), GeometryPredicateResult::Negative);
}

#[test]
fn incircle_zero_on_circumference() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(1.0, 1.0);
    assert_eq!(incircle(&a, &b, &c, &d), GeometryPredicateResult::Zero);
}

fn to_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    BigRational::from_float(value)
}

fn orient2d_rational(a: &Coord, b: &Coord, c: &Coord) -> Option<GeometryPredicateResult> {
    let ax = to_rational(a.x)?;
    let ay = to_rational(a.y)?;
    let bx = to_rational(b.x)?;
    let by = to_rational(b.y)?;
    let cx = to_rational(c.x)?;
    let cy = to_rational(c.y)?;

    let bax = &bx - &ax;
    let bay = &by - &ay;
    let cax = &cx - &ax;
    let cay = &cy - &ay;

    let det = bax * &cay - bay * &cax;
    Some(result_from_rational(det))
}

fn incircle_rational(
    a: &Coord,
    b: &Coord,
    c: &Coord,
    d: &Coord,
) -> Option<GeometryPredicateResult> {
    let ax = to_rational(a.x)?;
    let ay = to_rational(a.y)?;
    let bx = to_rational(b.x)?;
    let by = to_rational(b.y)?;
    let cx = to_rational(c.x)?;
    let cy = to_rational(c.y)?;
    let dx = to_rational(d.x)?;
    let dy = to_rational(d.y)?;

    let adx = &ax - &dx;
    let ady = &ay - &dy;
    let bdx = &bx - &dx;
    let bdy = &by - &dy;
    let cdx = &cx - &dx;
    let cdy = &cy - &dy;

    let ad2 = (&adx * &adx) + (&ady * &ady);
    let bd2 = (&bdx * &bdx) + (&bdy * &bdy);
    let cd2 = (&cdx * &cdx) + (&cdy * &cdy);

    let det = (&adx * &(&bdy * &cd2 - &cdy * &bd2))
        + (&ady * &(&cdx * &bd2 - &bdx * &cd2))
        + (&ad2 * &(&bdx * &cdy - &cdx * &bdy));

    Some(result_from_rational(det))
}

fn result_from_rational(r: BigRational) -> GeometryPredicateResult {
    if r.is_zero() {
        GeometryPredicateResult::Zero
    } else if r.is_positive() {
        GeometryPredicateResult::Positive
    } else {
        GeometryPredicateResult::Negative
    }
}

fn result_from_f64(value: f64) -> GeometryPredicateResult {
    const EPS: f64 = 1.0e-12;
    if value.abs() <= EPS {
        GeometryPredicateResult::Zero
    } else if value > 0.0 {
        GeometryPredicateResult::Positive
    } else {
        GeometryPredicateResult::Negative
    }
}

fn within_limits(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= MAG_LIMIT)
}

fn run_qc(prop: fn(f64, f64, f64, f64, f64, f64) -> TestResult) {
    QuickCheck::new()
        .tests(QC_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(prop);
}

fn run_qc_incircle(prop: fn(f64, f64, f64, f64, f64, f64, f64, f64) -> TestResult) {
    QuickCheck::new()
        .tests(QC_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(prop);
}

fn property_orient2d_consistency(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> TestResult {
    if !within_limits(&[ax, ay, bx, by, cx, cy]) {
        return TestResult::discard();
    }

    let a = Coord::new(ax, ay);
    let b = Coord::new(bx, by);
    let c = Coord::new(cx, cy);

    let ours = orient2d(&a, &b, &c);

    let Some(rational_result) = orient2d_rational(&a, &b, &c) else {
        return TestResult::discard();
    };

    let robust_value = robust::orient2d(robust_coord(&a), robust_coord(&b), robust_coord(&c));
    let robust_result = result_from_f64(robust_value);

    TestResult::from_bool(ours == rational_result && ours == robust_result)
}

#[allow(clippy::too_many_arguments)]
fn property_incircle_consistency(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
    dx: f64,
    dy: f64,
) -> TestResult {
    if !within_limits(&[ax, ay, bx, by, cx, cy, dx, dy]) {
        return TestResult::discard();
    }

    let a = Coord::new(ax, ay);
    let b = Coord::new(bx, by);
    let c = Coord::new(cx, cy);
    let d = Coord::new(dx, dy);

    let ours = incircle(&a, &b, &c, &d);

    let Some(rational_result) = incircle_rational(&a, &b, &c, &d) else {
        return TestResult::discard();
    };

    let robust_value = robust::incircle(
        robust_coord(&a),
        robust_coord(&b),
        robust_coord(&c),
        robust_coord(&d),
    );
    let robust_result = result_from_f64(robust_value);

    TestResult::from_bool(ours == rational_result && ours == robust_result)
}

#[test]
#[timeout(5000)]
fn quickcheck_orient2d_consistency() {
    run_qc(property_orient2d_consistency);
}

#[test]
#[timeout(5000)]
fn quickcheck_incircle_consistency() {
    run_qc_incircle(property_incircle_consistency);
}

fn robust_coord(coord: &Coord) -> robust::Coord<f64> {
    robust::Coord {
        x: coord.x,
        y: coord.y,
    }
}
