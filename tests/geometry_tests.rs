use apfp::analysis::adaptive_signum::{
    Dd, Diff, Scalar, Signum, Square, Sum, dd_add, dd_from, dd_mul, dd_signum, dd_square, dd_sub,
    signum_exact,
};
use apfp::apfp_signum;
use apfp::{Coord, GeometryPredicateResult, orient2d};
use geometry_predicates::orient2d as gp_orient2d;
use ntest::timeout;
use quickcheck::{QuickCheck, TestResult};

const MAG_LIMIT: f64 = 1.0e6;
const QC_TESTS: u64 = 300;
const QC_MAX_TESTS: u64 = 20_000;
const QC_STAGE_TESTS: u64 = 200;

const EPS_LIST: [f64; 12] = [
    1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14, 1.0e-16, 1.0e-18, 1.0e-20,
    1.0e-24, 1.0e-30,
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

fn orient2d_reference_sign(a: &Coord, b: &Coord, c: &Coord) -> Option<i32> {
    let gp_sign = sign_from_f64(gp_orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y]));
    let robust_value = robust::orient2d(robust_coord(a), robust_coord(b), robust_coord(c));
    let robust_sign = sign_from_f64(robust_value);
    if gp_sign == robust_sign {
        Some(gp_sign)
    } else {
        None
    }
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

fn square_diff_reference_sign(x: f64, y: f64) -> i32 {
    let expr = Diff(Square(Scalar(x)), Square(Scalar(y)));
    let mut buffer = [0.0_f64; <SquareDiffExpr as Signum>::STACK_LEN];
    signum_exact(expr, &mut buffer)
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
    let Some(reference_sign) = orient2d_reference_sign(&a, &b, &c) else {
        return TestResult::discard();
    };

    TestResult::from_bool(sign_from_result(ours) == reference_sign)
}

#[test]
#[timeout(5000)]
fn quickcheck_orient2d_consistency() {
    run_qc(property_orient2d_consistency);
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

fn sign_from_result(result: GeometryPredicateResult) -> i32 {
    match result {
        GeometryPredicateResult::Positive => 1,
        GeometryPredicateResult::Negative => -1,
        GeometryPredicateResult::Zero => 0,
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
    let Some(reference_sign) = orient2d_reference_sign(&a, &b, &c) else {
        return TestResult::discard();
    };

    TestResult::from_bool(sign == reference_sign)
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
#[timeout(8000)]
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
