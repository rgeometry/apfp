//! Benchmarks for incircle implementations.
//!
//! Measures per-call performance for:
//! - Naive f64 (baseline, non-robust)
//! - apfp (our adaptive implementation)
//! - robust crate (Shewchuk reference)
//! - Rational (exact but slow baseline)

use apfp::geometry::f64::{Coord, incircle};
use apfp::geometry::Orientation;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use num_rational::BigRational;
use num_traits::Zero;
use std::hint::black_box;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1;

/// Type alias for benchmark test cases: (a, b, c, d) coordinate quadruples.
type TestCases = Vec<(Coord, Coord, Coord, Coord)>;

/// Simple f64 incircle - no error checking, may give wrong results for near-cocircular points.
fn incircle_naive(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> i8 {
    let adx = a.x - d.x;
    let ady = a.y - d.y;
    let bdx = b.x - d.x;
    let bdy = b.y - d.y;
    let cdx = c.x - d.x;
    let cdy = c.y - d.y;

    let ad2 = adx * adx + ady * ady;
    let bd2 = bdx * bdx + bdy * bdy;
    let cd2 = cdx * cdx + cdy * cdy;

    let det = adx * (bdy * cd2 - cdy * bd2) + ady * (cdx * bd2 - bdx * cd2)
        + ad2 * (bdx * cdy - cdx * bdy);

    if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        0
    }
}

/// Exact incircle using arbitrary precision rationals - always correct but slow.
fn incircle_rational(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> i8 {
    let ax = BigRational::from_float(a.x).unwrap();
    let ay = BigRational::from_float(a.y).unwrap();
    let bx = BigRational::from_float(b.x).unwrap();
    let by = BigRational::from_float(b.y).unwrap();
    let cx = BigRational::from_float(c.x).unwrap();
    let cy = BigRational::from_float(c.y).unwrap();
    let dx = BigRational::from_float(d.x).unwrap();
    let dy = BigRational::from_float(d.y).unwrap();

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

    if det.is_zero() {
        0
    } else if det > BigRational::zero() {
        1
    } else {
        -1
    }
}

fn robust_incircle(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> i8 {
    let result = robust::incircle(
        robust::Coord { x: a.x, y: a.y },
        robust::Coord { x: b.x, y: b.y },
        robust::Coord { x: c.x, y: c.y },
        robust::Coord { x: d.x, y: d.y },
    );
    if result > 0.0 {
        1
    } else if result < 0.0 {
        -1
    } else {
        0
    }
}

fn apfp_incircle(a: &Coord, b: &Coord, c: &Coord, d: &Coord) -> i8 {
    match incircle(a, b, c, d) {
        Orientation::CounterClockwise => 1,
        Orientation::Clockwise => -1,
        Orientation::CoLinear => 0,
    }
}

/// Generate random test cases
fn generate_random_cases(count: usize) -> TestCases {
    let mut cases = Vec::with_capacity(count);
    let mut state = 0x1234_5678_9abc_def0u64;

    for _ in 0..count {
        let a = Coord::new(lcg(&mut state), lcg(&mut state));
        let b = Coord::new(lcg(&mut state), lcg(&mut state));
        let c = Coord::new(lcg(&mut state), lcg(&mut state));
        let d = Coord::new(lcg(&mut state), lcg(&mut state));
        cases.push((a, b, c, d));
    }

    cases
}

/// Generate near-cocircular test cases (points very close to being on the same circle)
fn generate_near_cocircular_cases(count: usize) -> TestCases {
    let mut cases = Vec::with_capacity(count);

    // Points on a unit circle, with d perturbed slightly
    for i in 0..count {
        let eps = 1e-14 * (i as f64 + 1.0);
        let a = Coord::new(1.0, 0.0);
        let b = Coord::new(0.0, 1.0);
        let c = Coord::new(-1.0, 0.0);
        // d is almost on the circle (at angle 3π/2 with small perturbation)
        let d = Coord::new(eps, -1.0 + eps);
        cases.push((a, b, c, d));
    }

    cases
}

fn bench_incircle(c: &mut Criterion) {
    let mut group = c.benchmark_group("incircle");
    let batch_size = BatchSize::SmallInput;

    let random_cases = generate_random_cases(100);
    let near_cocircular_cases = generate_near_cocircular_cases(100);

    // Setup functions that cycle through pre-generated samples
    let make_random_setup = || {
        let mut state = 0x1234_5678_9abc_def0u64;
        move || {
            let a = Coord::new(lcg(&mut state), lcg(&mut state));
            let b = Coord::new(lcg(&mut state), lcg(&mut state));
            let c = Coord::new(lcg(&mut state), lcg(&mut state));
            let d = Coord::new(lcg(&mut state), lcg(&mut state));
            (a, b, c, d)
        }
    };

    let make_stage_setup = |cases: Vec<(Coord, Coord, Coord, Coord)>| {
        let mut idx = 0usize;
        move || {
            let sample = cases[idx % cases.len()];
            idx = idx.wrapping_add(1);
            sample
        }
    };

    // === Random input benchmarks ===

    group.bench_function("naive_f64", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c, d)| black_box(incircle_naive(&a, &b, &c, &d)),
            batch_size,
        )
    });

    group.bench_function("apfp_random", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c, d)| black_box(apfp_incircle(&a, &b, &c, &d)),
            batch_size,
        )
    });

    group.bench_function("robust_random", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c, d)| black_box(robust_incircle(&a, &b, &c, &d)),
            batch_size,
        )
    });

    // === Near-cocircular benchmarks (stress test for adaptive paths) ===

    group.bench_function("apfp_near_cocircular", |bencher| {
        bencher.iter_batched(
            make_stage_setup(near_cocircular_cases.clone()),
            |(a, b, c, d)| black_box(apfp_incircle(&a, &b, &c, &d)),
            batch_size,
        )
    });

    group.bench_function("robust_near_cocircular", |bencher| {
        bencher.iter_batched(
            make_stage_setup(near_cocircular_cases.clone()),
            |(a, b, c, d)| black_box(robust_incircle(&a, &b, &c, &d)),
            batch_size,
        )
    });

    // === Reference: exact rational arithmetic ===

    group.bench_function("rational", |bencher| {
        bencher.iter_batched(
            make_stage_setup(random_cases.clone()),
            |(a, b, c, d)| black_box(incircle_rational(&a, &b, &c, &d)),
            batch_size,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_incircle);
criterion_main!(benches);

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}
