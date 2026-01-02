use apfp::analysis::adaptive_signum::{
    Dd, Diff, Scalar, Signum, Square, Sum, dd_add, dd_from, dd_mul, dd_signum, dd_square, dd_sub,
    signum_exact,
};
use apfp::apfp_signum;
use apfp::geometry::Orientation;
use apfp::geometry::f64::{Coord, incircle, orient2d, orient2d_normal, orient2d_vec};
use geometry_predicates::orient2d as gp_orient2d;
use ntest::timeout;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use quickcheck::{QuickCheck, TestResult};

const MAG_LIMIT: f64 = 1.0e6;
const QC_TESTS: u64 = 300;
const QC_MAX_TESTS: u64 = 20_000;
const QC_STAGE_TESTS: u64 = 200;

const EPS_LIST: [f64; 12] = [
    1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14, 1.0e-16, 1.0e-18, 1.0e-20, 1.0e-24,
    1.0e-30,
];

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1;

type CmpDistExpr = Diff<
    Sum<Square<Diff<Scalar, Scalar>>, Square<Diff<Scalar, Scalar>>>,
    Sum<Square<Diff<Scalar, Scalar>>, Square<Diff<Scalar, Scalar>>>,
>;

type SquareDiffExpr = Diff<Square<Scalar>, Square<Scalar>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Fast,
    Dd,
    Exact,
}

#[test]
fn orient2d_detects_ccw() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    assert_eq!(orient2d(&a, &b, &c), Orientation::CounterClockwise);
}

#[test]
fn orient2d_detects_cw() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(0.0, 1.0);
    let c = Coord::new(1.0, 0.0);
    assert_eq!(orient2d(&a, &b, &c), Orientation::Clockwise);
}

#[test]
fn orient2d_detects_collinear() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 1.0);
    let c = Coord::new(2.0, 2.0);
    assert_eq!(orient2d(&a, &b, &c), Orientation::CoLinear);
}

#[test]
fn incircle_positive_for_point_inside() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(0.25, 0.25);
    assert_eq!(incircle(&a, &b, &c, &d), Orientation::CounterClockwise);
}

#[test]
fn incircle_negative_for_point_outside() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(2.0, 2.0);
    assert_eq!(incircle(&a, &b, &c, &d), Orientation::Clockwise);
}

#[test]
fn incircle_zero_on_circumference() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(1.0, 1.0);
    assert_eq!(incircle(&a, &b, &c, &d), Orientation::CoLinear);
}

fn to_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    BigRational::from_float(value)
}

fn orient2d_rational(a: &Coord, b: &Coord, c: &Coord) -> Option<Orientation> {
    let ax = to_rational(a.x)?;
    let ay = to_rational(a.y)?;
    let bx = to_rational(b.x)?;
    let by = to_rational(b.y)?;
    let cx = to_rational(c.x)?;
    let cy = to_rational(c.y)?;

    let adx = &ax - &cx;
    let bdx = &bx - &cx;
    let ady = &ay - &cy;
    let bdy = &by - &cy;

    let det = &adx * &bdy - &ady * &bdx;
    Some(result_from_rational(det))
}

fn incircle_rational(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> Option<Orientation> {
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

fn result_from_rational(r: BigRational) -> Orientation {
    if r.is_zero() {
        Orientation::CoLinear
    } else if r.is_positive() {
        Orientation::CounterClockwise
    } else {
        Orientation::Clockwise
    }
}

fn result_from_f64(value: f64) -> Orientation {
    const EPS: f64 = 1.0e-12;
    if value.abs() <= EPS {
        Orientation::CoLinear
    } else if value > 0.0 {
        Orientation::CounterClockwise
    } else {
        Orientation::Clockwise
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

fn run_qc_seed(prop: fn(u64) -> TestResult) {
    QuickCheck::new()
        .tests(QC_STAGE_TESTS)
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

#[test]
fn apfp_signum_square_regression() {
    let x = 1.0e6_f64 + 1.0e-3;
    let y = 1.0e6_f64 - 1.0e-3;
    let apfp = apfp_signum!(square(x) - square(y));
    let reference = square_diff_reference_sign(x, y);
    assert_eq!(apfp, reference);
}

fn sign_from_result(result: Orientation) -> i32 {
    match result {
        Orientation::CounterClockwise => 1,
        Orientation::Clockwise => -1,
        Orientation::CoLinear => 0,
    }
}

fn sign_from_f64(value: f64) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

fn square_diff_reference_sign(x: f64, y: f64) -> i32 {
    let expr = Diff(Square(Scalar(x)), Square(Scalar(y)));
    let mut buffer = [0.0_f64; <SquareDiffExpr as Signum>::STACK_LEN];
    signum_exact(expr, &mut buffer)
}

fn cmp_dist_reference_sign(origin: &Coord, p: &Coord, q: &Coord) -> i32 {
    let expr = Diff(
        Sum(
            Square(Diff(Scalar(p.x), Scalar(origin.x))),
            Square(Diff(Scalar(p.y), Scalar(origin.y))),
        ),
        Sum(
            Square(Diff(Scalar(q.x), Scalar(origin.x))),
            Square(Diff(Scalar(q.y), Scalar(origin.y))),
        ),
    );
    let mut buffer = [0.0_f64; <CmpDistExpr as Signum>::STACK_LEN];
    signum_exact(expr, &mut buffer)
}

fn gamma_for_ops(op_count: usize) -> f64 {
    let u = 0.5 * f64::EPSILON;
    let n = op_count as f64;
    (n * u) / (1.0 - n * u)
}

fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2.0) - 1.0
}

fn lcg_range(state: &mut u64, min: f64, max: f64) -> f64 {
    min + (max - min) * (lcg_next(state) * 0.5 + 0.5)
}

fn dd_orient2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> Dd {
    let adx = dd_sub(dd_from(ax), dd_from(cx));
    let bdx = dd_sub(dd_from(bx), dd_from(cx));
    let ady = dd_sub(dd_from(ay), dd_from(cy));
    let bdy = dd_sub(dd_from(by), dd_from(cy));
    let prod1 = dd_mul(adx, bdy);
    let prod2 = dd_mul(ady, bdx);
    dd_sub(prod1, prod2)
}

fn classify_orient2d(a: &Coord, b: &Coord, c: &Coord) -> Option<Stage> {
    let ax = a.x;
    let ay = a.y;
    let bx = b.x;
    let by = b.y;
    let cx = c.x;
    let cy = c.y;
    if !within_limits(&[ax, ay, bx, by, cx, cy]) {
        return None;
    }

    let adx = ax - cx;
    let bdx = bx - cx;
    let ady = ay - cy;
    let bdy = by - cy;
    let prod1 = adx * bdy;
    let prod2 = ady * bdx;
    let det = prod1 - prod2;
    let detsum = prod1.abs() + prod2.abs();
    let errbound = gamma_for_ops(7) * detsum;

    if det.abs() > errbound {
        return Some(Stage::Fast);
    }
    let dd_value = dd_orient2d(ax, ay, bx, by, cx, cy);
    if dd_signum(dd_value).is_some() {
        Some(Stage::Dd)
    } else {
        Some(Stage::Exact)
    }
}

fn dd_cmp_dist(ox: f64, oy: f64, px: f64, py: f64, qx: f64, qy: f64) -> Dd {
    let pdx = dd_sub(dd_from(px), dd_from(ox));
    let pdy = dd_sub(dd_from(py), dd_from(oy));
    let qdx = dd_sub(dd_from(qx), dd_from(ox));
    let qdy = dd_sub(dd_from(qy), dd_from(oy));
    let pdist = dd_add(dd_square(pdx), dd_square(pdy));
    let qdist = dd_add(dd_square(qdx), dd_square(qdy));
    dd_sub(pdist, qdist)
}

fn classify_cmp_dist(origin: &Coord, p: &Coord, q: &Coord) -> Option<Stage> {
    let ox = origin.x;
    let oy = origin.y;
    let px = p.x;
    let py = p.y;
    let qx = q.x;
    let qy = q.y;
    if !within_limits(&[ox, oy, px, py, qx, qy]) {
        return None;
    }

    let pdx = px - ox;
    let pdy = py - oy;
    let qdx = qx - ox;
    let qdy = qy - oy;
    let pdist = pdx * pdx + pdy * pdy;
    let qdist = qdx * qdx + qdy * qdy;
    let diff = pdist - qdist;
    let sum = pdist.abs() + qdist.abs();
    let errbound = gamma_for_ops(11) * sum;

    if diff.abs() > errbound {
        return Some(Stage::Fast);
    }
    let dd_value = dd_cmp_dist(ox, oy, px, py, qx, qy);
    if dd_signum(dd_value).is_some() {
        Some(Stage::Dd)
    } else {
        Some(Stage::Exact)
    }
}

fn find_orient2d_case(mut seed: u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    for _ in 0..200 {
        if stage == Stage::Fast {
            let ax = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let ay = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let bx = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let by = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let cx = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let cy = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let a = Coord::new(ax, ay);
            let b = Coord::new(bx, by);
            let c = Coord::new(cx, cy);
            if classify_orient2d(&a, &b, &c) == Some(stage) {
                return Some((a, b, c));
            }
            continue;
        }

        let dx = lcg_range(&mut seed, -1.0e3, 1.0e3);
        let dy = lcg_range(&mut seed, -1.0e3, 1.0e3);
        if dx.abs() + dy.abs() < 1.0e-12 {
            continue;
        }
        let t1 = lcg_range(&mut seed, -1.0e3, 1.0e3);
        let t2 = lcg_range(&mut seed, -1.0e3, 1.0e3);
        for &eps in &EPS_LIST {
            let a = Coord::new(t1 * dx, t1 * dy);
            let b = Coord::new(t2 * dx - eps * dy, t2 * dy + eps * dx);
            let c = Coord::new(0.0, 0.0);
            if classify_orient2d(&a, &b, &c) == Some(stage) {
                return Some((a, b, c));
            }
        }
    }
    None
}

fn find_cmp_dist_case(mut seed: u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    let origin = Coord::new(0.0, 0.0);
    for _ in 0..200 {
        if stage == Stage::Fast {
            let px = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let py = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let qx = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let qy = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let p = Coord::new(px, py);
            let q = Coord::new(qx, qy);
            if classify_cmp_dist(&origin, &p, &q) == Some(stage) {
                return Some((origin, p, q));
            }
            continue;
        }

        let r = lcg_range(&mut seed, 1.0e3, 1.0e6);
        for &eps in &EPS_LIST {
            let p = Coord::new(r, 0.0);
            let q = Coord::new(r + eps, 0.0);
            if classify_cmp_dist(&origin, &p, &q) == Some(stage) {
                return Some((origin, p, q));
            }
        }
    }
    None
}

fn property_apfp_signum_orient2d_stage(seed: u64, stage: Stage) -> TestResult {
    let Some((a, b, c)) = find_orient2d_case(seed, stage) else {
        return TestResult::discard();
    };
    if classify_orient2d(&a, &b, &c) != Some(stage) {
        return TestResult::discard();
    }

    let sign = apfp_signum!((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x));
    let Some(rational) = orient2d_rational(&a, &b, &c) else {
        return TestResult::discard();
    };
    let rational_sign = sign_from_result(rational);
    let gp_sign = sign_from_f64(gp_orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y]));

    TestResult::from_bool(sign == rational_sign && sign == gp_sign)
}

fn property_apfp_signum_cmp_dist_stage(seed: u64, stage: Stage) -> TestResult {
    let Some((origin, p, q)) = find_cmp_dist_case(seed, stage) else {
        return TestResult::discard();
    };
    if classify_cmp_dist(&origin, &p, &q) != Some(stage) {
        return TestResult::discard();
    }

    let sign = apfp_signum!(
        (square(p.x - origin.x) + square(p.y - origin.y))
            - (square(q.x - origin.x) + square(q.y - origin.y))
    );
    let reference_sign = cmp_dist_reference_sign(&origin, &p, &q);

    TestResult::from_bool(sign == reference_sign)
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_orient2d_fast() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_stage(seed, Stage::Fast));
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_orient2d_dd() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_stage(seed, Stage::Dd));
}

#[test]
#[timeout(12000)]
fn quickcheck_apfp_signum_orient2d_exact() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_stage(seed, Stage::Exact));
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_cmp_dist_fast() {
    run_qc_seed(|seed| property_apfp_signum_cmp_dist_stage(seed, Stage::Fast));
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_cmp_dist_dd() {
    run_qc_seed(|seed| property_apfp_signum_cmp_dist_stage(seed, Stage::Dd));
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_cmp_dist_exact() {
    run_qc_seed(|seed| property_apfp_signum_cmp_dist_stage(seed, Stage::Exact));
}

// ============================================================================
// orient2d_vec tests
// ============================================================================

fn orient2d_vec_rational(a: &Coord, v: &Coord, p: &Coord) -> Option<Orientation> {
    let ax = to_rational(a.x)?;
    let ay = to_rational(a.y)?;
    let vx = to_rational(v.x)?;
    let vy = to_rational(v.y)?;
    let px = to_rational(p.x)?;
    let py = to_rational(p.y)?;

    // orient2d_vec(a, v, p) = (a.x - p.x) * v.y - (a.y - p.y) * v.x
    let det = (&ax - &px) * &vy - (&ay - &py) * &vx;
    Some(result_from_rational(det))
}

#[test]
fn orient2d_vec_basic_ccw() {
    // Line from origin in direction (1, 0), point above
    let a = Coord::new(0.0, 0.0);
    let v = Coord::new(1.0, 0.0);
    let p = Coord::new(0.5, 1.0);
    assert_eq!(orient2d_vec(&a, &v, &p), Orientation::CounterClockwise);
}

#[test]
fn orient2d_vec_basic_cw() {
    // Line from origin in direction (1, 0), point below
    let a = Coord::new(0.0, 0.0);
    let v = Coord::new(1.0, 0.0);
    let p = Coord::new(0.5, -1.0);
    assert_eq!(orient2d_vec(&a, &v, &p), Orientation::Clockwise);
}

#[test]
fn orient2d_vec_collinear() {
    // Line from origin in direction (1, 1), point on line
    let a = Coord::new(0.0, 0.0);
    let v = Coord::new(1.0, 1.0);
    let p = Coord::new(2.0, 2.0);
    assert_eq!(orient2d_vec(&a, &v, &p), Orientation::CoLinear);
}

#[test]
fn orient2d_vec_matches_orient2d_simple() {
    // Test that orient2d_vec(a, v, p) == orient2d(a, a+v, p) for simple cases
    let a = Coord::new(1.0, 2.0);
    let v = Coord::new(3.0, 4.0);
    let p = Coord::new(5.0, 6.0);
    let b = Coord::new(a.x + v.x, a.y + v.y);
    assert_eq!(orient2d_vec(&a, &v, &p), orient2d(&a, &b, &p));
}

fn property_orient2d_vec_consistency(
    ax: f64,
    ay: f64,
    vx: f64,
    vy: f64,
    px: f64,
    py: f64,
) -> TestResult {
    if !within_limits(&[ax, ay, vx, vy, px, py]) {
        return TestResult::discard();
    }

    let a = Coord::new(ax, ay);
    let v = Coord::new(vx, vy);
    let p = Coord::new(px, py);

    let ours = orient2d_vec(&a, &v, &p);

    let Some(rational_result) = orient2d_vec_rational(&a, &v, &p) else {
        return TestResult::discard();
    };

    TestResult::from_bool(ours == rational_result)
}

#[test]
#[timeout(5000)]
fn quickcheck_orient2d_vec_consistency() {
    run_qc(property_orient2d_vec_consistency);
}

fn dd_orient2d_vec(ax: f64, ay: f64, vx: f64, vy: f64, px: f64, py: f64) -> Dd {
    // (a.x - p.x) * v.y - (a.y - p.y) * v.x
    let dx = dd_sub(dd_from(ax), dd_from(px));
    let dy = dd_sub(dd_from(ay), dd_from(py));
    let prod1 = dd_mul(dx, dd_from(vy));
    let prod2 = dd_mul(dy, dd_from(vx));
    dd_sub(prod1, prod2)
}

fn classify_orient2d_vec(a: &Coord, v: &Coord, p: &Coord) -> Option<Stage> {
    let ax = a.x;
    let ay = a.y;
    let vx = v.x;
    let vy = v.y;
    let px = p.x;
    let py = p.y;
    if !within_limits(&[ax, ay, vx, vy, px, py]) {
        return None;
    }

    let dx = ax - px;
    let dy = ay - py;
    let prod1 = dx * vy;
    let prod2 = dy * vx;
    let det = prod1 - prod2;
    let detsum = prod1.abs() + prod2.abs();
    let errbound = gamma_for_ops(5) * detsum; // 5 ops: 2 subs, 2 muls, 1 sub

    if det.abs() > errbound {
        return Some(Stage::Fast);
    }
    let dd_value = dd_orient2d_vec(ax, ay, vx, vy, px, py);
    if dd_signum(dd_value).is_some() {
        Some(Stage::Dd)
    } else {
        Some(Stage::Exact)
    }
}

fn find_orient2d_vec_case(mut seed: u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    for _ in 0..200 {
        if stage == Stage::Fast {
            let ax = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let ay = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let vx = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let vy = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let px = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let py = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let a = Coord::new(ax, ay);
            let v = Coord::new(vx, vy);
            let p = Coord::new(px, py);
            if classify_orient2d_vec(&a, &v, &p) == Some(stage) {
                return Some((a, v, p));
            }
            continue;
        }

        // For Dd and Exact stages, construct near-collinear cases
        let vx = lcg_range(&mut seed, -1.0e3, 1.0e3);
        let vy = lcg_range(&mut seed, -1.0e3, 1.0e3);
        if vx.abs() + vy.abs() < 1.0e-12 {
            continue;
        }
        let t = lcg_range(&mut seed, -1.0e3, 1.0e3);
        for &eps in &EPS_LIST {
            // Point nearly on the line: p = a + t*v + eps*perpendicular
            let a = Coord::new(0.0, 0.0);
            let v = Coord::new(vx, vy);
            let p = Coord::new(t * vx - eps * vy, t * vy + eps * vx);
            if classify_orient2d_vec(&a, &v, &p) == Some(stage) {
                return Some((a, v, p));
            }
        }
    }
    None
}

fn property_apfp_signum_orient2d_vec_stage(seed: u64, stage: Stage) -> TestResult {
    let Some((a, v, p)) = find_orient2d_vec_case(seed, stage) else {
        return TestResult::discard();
    };
    if classify_orient2d_vec(&a, &v, &p) != Some(stage) {
        return TestResult::discard();
    }

    let ours = orient2d_vec(&a, &v, &p);
    let Some(rational) = orient2d_vec_rational(&a, &v, &p) else {
        return TestResult::discard();
    };

    TestResult::from_bool(ours == rational)
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_orient2d_vec_fast() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_vec_stage(seed, Stage::Fast));
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_orient2d_vec_dd() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_vec_stage(seed, Stage::Dd));
}

#[test]
#[timeout(12000)]
fn quickcheck_apfp_signum_orient2d_vec_exact() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_vec_stage(seed, Stage::Exact));
}

// ============================================================================
// orient2d_normal tests
// ============================================================================

fn orient2d_normal_rational(a: &Coord, n: &Coord, p: &Coord) -> Option<Orientation> {
    let ax = to_rational(a.x)?;
    let ay = to_rational(a.y)?;
    let nx = to_rational(n.x)?;
    let ny = to_rational(n.y)?;
    let px = to_rational(p.x)?;
    let py = to_rational(p.y)?;

    // orient2d_normal(a, n, p) = (a.x - p.x) * n.x + (a.y - p.y) * n.y
    let det = (&ax - &px) * &nx + (&ay - &py) * &ny;
    Some(result_from_rational(det))
}

#[test]
fn orient2d_normal_basic_positive() {
    // Line through origin with normal pointing in +y direction (horizontal line)
    // Point above the line should be positive
    let a = Coord::new(0.0, 0.0);
    let n = Coord::new(0.0, 1.0);
    let p = Coord::new(0.0, -1.0); // Below origin, so (a-p) points up, dot with n is positive
    assert_eq!(orient2d_normal(&a, &n, &p), Orientation::CounterClockwise);
}

#[test]
fn orient2d_normal_basic_negative() {
    // Line through origin with normal pointing in +y direction (horizontal line)
    // Point below the line (in normal direction) should be negative
    let a = Coord::new(0.0, 0.0);
    let n = Coord::new(0.0, 1.0);
    let p = Coord::new(0.0, 1.0); // Above origin, so (a-p) points down, dot with n is negative
    assert_eq!(orient2d_normal(&a, &n, &p), Orientation::Clockwise);
}

#[test]
fn orient2d_normal_on_line() {
    // Line through origin with normal pointing in +y direction (horizontal line)
    // Point on the line should be zero
    let a = Coord::new(0.0, 0.0);
    let n = Coord::new(0.0, 1.0);
    let p = Coord::new(5.0, 0.0); // On the x-axis
    assert_eq!(orient2d_normal(&a, &n, &p), Orientation::CoLinear);
}

#[test]
fn orient2d_normal_matches_orient2d_vec() {
    // orient2d_normal(a, n, p) should equal orient2d_vec(a, (-n.y, n.x), p)
    let a = Coord::new(1.0, 2.0);
    let n = Coord::new(3.0, 4.0);
    let p = Coord::new(5.0, 6.0);
    let v = Coord::new(-n.y, n.x); // Perpendicular to normal
    assert_eq!(orient2d_normal(&a, &n, &p), orient2d_vec(&a, &v, &p));
}

fn property_orient2d_normal_consistency(
    ax: f64,
    ay: f64,
    nx: f64,
    ny: f64,
    px: f64,
    py: f64,
) -> TestResult {
    if !within_limits(&[ax, ay, nx, ny, px, py]) {
        return TestResult::discard();
    }

    let a = Coord::new(ax, ay);
    let n = Coord::new(nx, ny);
    let p = Coord::new(px, py);

    let ours = orient2d_normal(&a, &n, &p);

    let Some(rational_result) = orient2d_normal_rational(&a, &n, &p) else {
        return TestResult::discard();
    };

    TestResult::from_bool(ours == rational_result)
}

#[test]
#[timeout(5000)]
fn quickcheck_orient2d_normal_consistency() {
    run_qc(property_orient2d_normal_consistency);
}

fn dd_orient2d_normal(ax: f64, ay: f64, nx: f64, ny: f64, px: f64, py: f64) -> Dd {
    // (a.x - p.x) * n.x + (a.y - p.y) * n.y
    let dx = dd_sub(dd_from(ax), dd_from(px));
    let dy = dd_sub(dd_from(ay), dd_from(py));
    let prod1 = dd_mul(dx, dd_from(nx));
    let prod2 = dd_mul(dy, dd_from(ny));
    dd_add(prod1, prod2)
}

fn classify_orient2d_normal(a: &Coord, n: &Coord, p: &Coord) -> Option<Stage> {
    let ax = a.x;
    let ay = a.y;
    let nx = n.x;
    let ny = n.y;
    let px = p.x;
    let py = p.y;
    if !within_limits(&[ax, ay, nx, ny, px, py]) {
        return None;
    }

    let dx = ax - px;
    let dy = ay - py;
    let prod1 = dx * nx;
    let prod2 = dy * ny;
    let det = prod1 + prod2;
    let detsum = prod1.abs() + prod2.abs();
    let errbound = gamma_for_ops(5) * detsum; // 5 ops: 2 subs, 2 muls, 1 add

    if det.abs() > errbound {
        return Some(Stage::Fast);
    }
    let dd_value = dd_orient2d_normal(ax, ay, nx, ny, px, py);
    if dd_signum(dd_value).is_some() {
        Some(Stage::Dd)
    } else {
        Some(Stage::Exact)
    }
}

fn find_orient2d_normal_case(mut seed: u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    for _ in 0..200 {
        if stage == Stage::Fast {
            let ax = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let ay = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let nx = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let ny = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let px = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let py = lcg_range(&mut seed, -1.0e3, 1.0e3);
            let a = Coord::new(ax, ay);
            let n = Coord::new(nx, ny);
            let p = Coord::new(px, py);
            if classify_orient2d_normal(&a, &n, &p) == Some(stage) {
                return Some((a, n, p));
            }
            continue;
        }

        // For Dd and Exact stages, construct near-on-line cases
        // Normal n, direction v = (-n.y, n.x), point nearly on line: p = a + t*v + eps*n
        let nx = lcg_range(&mut seed, -1.0e3, 1.0e3);
        let ny = lcg_range(&mut seed, -1.0e3, 1.0e3);
        if nx.abs() + ny.abs() < 1.0e-12 {
            continue;
        }
        let t = lcg_range(&mut seed, -1.0e3, 1.0e3);
        for &eps in &EPS_LIST {
            let a = Coord::new(0.0, 0.0);
            let n = Coord::new(nx, ny);
            // p = t * (-n.y, n.x) + eps * (n.x, n.y) = (-t*n.y + eps*n.x, t*n.x + eps*n.y)
            let p = Coord::new(-t * ny + eps * nx, t * nx + eps * ny);
            if classify_orient2d_normal(&a, &n, &p) == Some(stage) {
                return Some((a, n, p));
            }
        }
    }
    None
}

fn property_apfp_signum_orient2d_normal_stage(seed: u64, stage: Stage) -> TestResult {
    let Some((a, n, p)) = find_orient2d_normal_case(seed, stage) else {
        return TestResult::discard();
    };
    if classify_orient2d_normal(&a, &n, &p) != Some(stage) {
        return TestResult::discard();
    }

    let ours = orient2d_normal(&a, &n, &p);
    let Some(rational) = orient2d_normal_rational(&a, &n, &p) else {
        return TestResult::discard();
    };

    TestResult::from_bool(ours == rational)
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_orient2d_normal_fast() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_normal_stage(seed, Stage::Fast));
}

#[test]
#[timeout(8000)]
fn quickcheck_apfp_signum_orient2d_normal_dd() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_normal_stage(seed, Stage::Dd));
}

#[test]
#[timeout(12000)]
fn quickcheck_apfp_signum_orient2d_normal_exact() {
    run_qc_seed(|seed| property_apfp_signum_orient2d_normal_stage(seed, Stage::Exact));
}

// ============================================================================
// Integer signum tests against num::BigInt
// ============================================================================

use apfp::geometry::i32 as i32_types;
use apfp::geometry::i64 as i64_types;
use apfp::int_signum;
use num_bigint::{BigInt as NumBigInt, Sign};

fn sign_to_i32(sign: Sign) -> i32 {
    match sign {
        Sign::Plus => 1,
        Sign::Minus => -1,
        Sign::NoSign => 0,
    }
}

fn orient2d_bigint_i32(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
    let ax = NumBigInt::from(ax);
    let ay = NumBigInt::from(ay);
    let bx = NumBigInt::from(bx);
    let by = NumBigInt::from(by);
    let cx = NumBigInt::from(cx);
    let cy = NumBigInt::from(cy);

    let adx = &ax - &cx;
    let ady = &ay - &cy;
    let bdx = &bx - &cx;
    let bdy = &by - &cy;

    let det = &adx * &bdy - &ady * &bdx;
    sign_to_i32(det.sign())
}

fn orient2d_bigint_i64(ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64) -> i32 {
    let ax = NumBigInt::from(ax);
    let ay = NumBigInt::from(ay);
    let bx = NumBigInt::from(bx);
    let by = NumBigInt::from(by);
    let cx = NumBigInt::from(cx);
    let cy = NumBigInt::from(cy);

    let adx = &ax - &cx;
    let ady = &ay - &cy;
    let bdx = &bx - &cx;
    let bdy = &by - &cy;

    let det = &adx * &bdy - &ady * &bdx;
    sign_to_i32(det.sign())
}

fn cmp_dist_bigint_i32(ox: i32, oy: i32, px: i32, py: i32, qx: i32, qy: i32) -> i32 {
    let ox = NumBigInt::from(ox);
    let oy = NumBigInt::from(oy);
    let px = NumBigInt::from(px);
    let py = NumBigInt::from(py);
    let qx = NumBigInt::from(qx);
    let qy = NumBigInt::from(qy);

    let pdx = &px - &ox;
    let pdy = &py - &oy;
    let qdx = &qx - &ox;
    let qdy = &qy - &oy;

    let pd2 = &pdx * &pdx + &pdy * &pdy;
    let qd2 = &qdx * &qdx + &qdy * &qdy;

    sign_to_i32((&pd2 - &qd2).sign())
}

#[allow(clippy::too_many_arguments)]
fn incircle_bigint_i32(
    ax: i32,
    ay: i32,
    bx: i32,
    by: i32,
    cx: i32,
    cy: i32,
    dx: i32,
    dy: i32,
) -> i32 {
    let ax = NumBigInt::from(ax);
    let ay = NumBigInt::from(ay);
    let bx = NumBigInt::from(bx);
    let by = NumBigInt::from(by);
    let cx = NumBigInt::from(cx);
    let cy = NumBigInt::from(cy);
    let dx = NumBigInt::from(dx);
    let dy = NumBigInt::from(dy);

    let adx = &ax - &dx;
    let ady = &ay - &dy;
    let bdx = &bx - &dx;
    let bdy = &by - &dy;
    let cdx = &cx - &dx;
    let cdy = &cy - &dy;

    let ad2 = &adx * &adx + &ady * &ady;
    let bd2 = &bdx * &bdx + &bdy * &bdy;
    let cd2 = &cdx * &cdx + &cdy * &cdy;

    let det = &adx * (&bdy * &cd2 - &cdy * &bd2)
        + &ady * (&cdx * &bd2 - &bdx * &cd2)
        + &ad2 * (&bdx * &cdy - &cdx * &bdy);

    sign_to_i32(det.sign())
}

#[test]
fn int_orient2d_i32_matches_bigint() {
    // Test various cases including edge cases
    let cases = [
        // CCW
        (0, 0, 10, 0, 5, 10),
        // CW
        (0, 0, 5, 10, 10, 0),
        // Collinear
        (0, 0, 5, 5, 10, 10),
        // Near overflow
        (
            i32::MAX / 2,
            i32::MAX / 2,
            -i32::MAX / 2,
            i32::MAX / 2,
            0,
            -i32::MAX / 2,
        ),
        // Large values
        (1000000000, 1000000000, -1000000000, 1000000000, 0, 0),
        // Negative values
        (-100, -200, 300, -400, 0, 0),
    ];

    for (ax, ay, bx, by, cx, cy) in cases {
        let a = i32_types::Coord::new(ax, ay);
        let b = i32_types::Coord::new(bx, by);
        let c = i32_types::Coord::new(cx, cy);

        let ours = match i32_types::orient2d(&a, &b, &c) {
            Orientation::CounterClockwise => 1,
            Orientation::Clockwise => -1,
            Orientation::CoLinear => 0,
        };
        let expected = orient2d_bigint_i32(ax, ay, bx, by, cx, cy);
        assert_eq!(
            ours, expected,
            "orient2d_i32({ax}, {ay}, {bx}, {by}, {cx}, {cy})"
        );
    }
}

#[test]
fn int_orient2d_i64_matches_bigint() {
    let cases: [(i64, i64, i64, i64, i64, i64); 6] = [
        // CCW
        (0, 0, 10, 0, 5, 10),
        // CW
        (0, 0, 5, 10, 10, 0),
        // Collinear
        (0, 0, 5, 5, 10, 10),
        // Large values that would overflow i64 * i64
        (
            i64::MAX / 4,
            i64::MAX / 4,
            -i64::MAX / 4,
            i64::MAX / 4,
            0,
            -i64::MAX / 4,
        ),
        // More large values
        (
            1_000_000_000_000i64,
            1_000_000_000_000,
            -1_000_000_000_000,
            1_000_000_000_000,
            0,
            0,
        ),
        // Negative values
        (-100, -200, 300, -400, 0, 0),
    ];

    for (ax, ay, bx, by, cx, cy) in cases {
        let a = i64_types::Coord::new(ax, ay);
        let b = i64_types::Coord::new(bx, by);
        let c = i64_types::Coord::new(cx, cy);

        let ours = match i64_types::orient2d(&a, &b, &c) {
            Orientation::CounterClockwise => 1,
            Orientation::Clockwise => -1,
            Orientation::CoLinear => 0,
        };
        let expected = orient2d_bigint_i64(ax, ay, bx, by, cx, cy);
        assert_eq!(
            ours, expected,
            "orient2d_i64({ax}, {ay}, {bx}, {by}, {cx}, {cy})"
        );
    }
}

fn property_int_orient2d_i32_consistency_bounded(seed: u64) -> TestResult {
    let mut state = seed;

    fn lcg_i32(state: &mut u64) -> i32 {
        *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        ((*state >> 33) as i32).wrapping_sub(i32::MAX / 2)
    }

    let ax = lcg_i32(&mut state);
    let ay = lcg_i32(&mut state);
    let bx = lcg_i32(&mut state);
    let by = lcg_i32(&mut state);
    let cx = lcg_i32(&mut state);
    let cy = lcg_i32(&mut state);

    let a = i32_types::Coord::new(ax, ay);
    let b = i32_types::Coord::new(bx, by);
    let c = i32_types::Coord::new(cx, cy);

    let ours = match i32_types::orient2d(&a, &b, &c) {
        Orientation::CounterClockwise => 1,
        Orientation::Clockwise => -1,
        Orientation::CoLinear => 0,
    };
    let expected = orient2d_bigint_i32(ax, ay, bx, by, cx, cy);

    TestResult::from_bool(ours == expected)
}

#[test]
#[timeout(5000)]
fn quickcheck_int_orient2d_i32() {
    QuickCheck::new()
        .tests(QC_STAGE_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(property_int_orient2d_i32_consistency_bounded as fn(u64) -> TestResult);
}

fn property_int_orient2d_i64_consistency_bounded(seed: u64) -> TestResult {
    let mut state = seed;

    fn lcg_i64(state: &mut u64, limit: i64) -> i64 {
        *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        let val = ((*state >> 1) as i64).wrapping_rem(limit * 2);
        val - limit
    }

    const LIMIT: i64 = 1_000_000_000;
    let ax = lcg_i64(&mut state, LIMIT);
    let ay = lcg_i64(&mut state, LIMIT);
    let bx = lcg_i64(&mut state, LIMIT);
    let by = lcg_i64(&mut state, LIMIT);
    let cx = lcg_i64(&mut state, LIMIT);
    let cy = lcg_i64(&mut state, LIMIT);

    let a = i64_types::Coord::new(ax, ay);
    let b = i64_types::Coord::new(bx, by);
    let c = i64_types::Coord::new(cx, cy);

    let ours = match i64_types::orient2d(&a, &b, &c) {
        Orientation::CounterClockwise => 1,
        Orientation::Clockwise => -1,
        Orientation::CoLinear => 0,
    };
    let expected = orient2d_bigint_i64(ax, ay, bx, by, cx, cy);

    TestResult::from_bool(ours == expected)
}

#[test]
#[timeout(8000)]
fn quickcheck_int_orient2d_i64() {
    QuickCheck::new()
        .tests(QC_STAGE_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(property_int_orient2d_i64_consistency_bounded as fn(u64) -> TestResult);
}

fn property_int_cmp_dist_i32_consistency_bounded(seed: u64) -> TestResult {
    let mut state = seed;

    fn lcg_i32(state: &mut u64) -> i32 {
        *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        ((*state >> 33) as i32).wrapping_sub(i32::MAX / 2)
    }

    let ox = lcg_i32(&mut state);
    let oy = lcg_i32(&mut state);
    let px = lcg_i32(&mut state);
    let py = lcg_i32(&mut state);
    let qx = lcg_i32(&mut state);
    let qy = lcg_i32(&mut state);

    let origin = i32_types::Coord::new(ox, oy);
    let p = i32_types::Coord::new(px, py);
    let q = i32_types::Coord::new(qx, qy);

    let ours = match i32_types::cmp_dist(&origin, &p, &q) {
        core::cmp::Ordering::Greater => 1,
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
    };
    let expected = cmp_dist_bigint_i32(ox, oy, px, py, qx, qy);

    TestResult::from_bool(ours == expected)
}

#[test]
#[timeout(5000)]
fn quickcheck_int_cmp_dist_i32() {
    QuickCheck::new()
        .tests(QC_STAGE_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(property_int_cmp_dist_i32_consistency_bounded as fn(u64) -> TestResult);
}

#[test]
fn int_incircle_i32_matches_bigint() {
    // Inside circle (CCW orientation assumed)
    let result = i32_types::incircle(
        &i32_types::Coord::new(0, 0),
        &i32_types::Coord::new(10, 0),
        &i32_types::Coord::new(0, 10),
        &i32_types::Coord::new(3, 3),
    );
    let expected = incircle_bigint_i32(0, 0, 10, 0, 0, 10, 3, 3);
    assert_eq!(
        match result {
            Orientation::CounterClockwise => 1,
            Orientation::Clockwise => -1,
            Orientation::CoLinear => 0,
        },
        expected,
        "incircle inside"
    );

    // Outside circle
    let result = i32_types::incircle(
        &i32_types::Coord::new(0, 0),
        &i32_types::Coord::new(10, 0),
        &i32_types::Coord::new(0, 10),
        &i32_types::Coord::new(100, 100),
    );
    let expected = incircle_bigint_i32(0, 0, 10, 0, 0, 10, 100, 100);
    assert_eq!(
        match result {
            Orientation::CounterClockwise => 1,
            Orientation::Clockwise => -1,
            Orientation::CoLinear => 0,
        },
        expected,
        "incircle outside"
    );

    // On circle
    let result = i32_types::incircle(
        &i32_types::Coord::new(0, 0),
        &i32_types::Coord::new(10, 0),
        &i32_types::Coord::new(0, 10),
        &i32_types::Coord::new(10, 10),
    );
    let expected = incircle_bigint_i32(0, 0, 10, 0, 0, 10, 10, 10);
    assert_eq!(
        match result {
            Orientation::CounterClockwise => 1,
            Orientation::Clockwise => -1,
            Orientation::CoLinear => 0,
        },
        expected,
        "incircle on circle"
    );
}

#[test]
fn int_signum_macro_matches_bigint() {
    // Test various expressions
    let a: i32 = 1000000;
    let b: i32 = 999999;

    // square(a) - square(b)
    let our_sign = int_signum!(square(a) - square(b));
    let a_big = NumBigInt::from(a);
    let b_big = NumBigInt::from(b);
    let expected = sign_to_i32((&a_big * &a_big - &b_big * &b_big).sign());
    assert_eq!(our_sign, expected, "square diff");

    // Complex expression
    let x: i32 = 12345;
    let y: i32 = 67890;
    let z: i32 = 11111;

    let our_sign = int_signum!((x - y) * (y - z) + (z - x) * x);
    let x_big = NumBigInt::from(x);
    let y_big = NumBigInt::from(y);
    let z_big = NumBigInt::from(z);
    let expected =
        sign_to_i32(((&x_big - &y_big) * (&y_big - &z_big) + (&z_big - &x_big) * &x_big).sign());
    assert_eq!(our_sign, expected, "complex expression");
}
