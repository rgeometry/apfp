use apfp::{
    Coord, GeometryPredicateResult, orient2d, orient2d_fixed, orient2d_inexact_baseline,
    orient2d_inexact_interval,
};
use criterion::{Criterion, criterion_group, criterion_main};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::hint::black_box;

const SAMPLE_COUNT: usize = 5_000;
const MAG_LIMIT: f64 = 1.0e6;

fn orient2d_apfp_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d(a, b, c));
    }
}

fn orient2d_inexact_baseline_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_inexact_baseline(a, b, c));
    }
}

fn orient2d_inexact_interval_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_inexact_interval(a, b, c));
    }
}

fn orient2d_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_rational(a, b, c));
    }
}

fn orient2d_fixed_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_fixed(a, b, c));
    }
}

fn orient2d_robust_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(robust_orient2d(a, b, c));
    }
}

fn bench_orient2d(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);

    c.bench_function("orient2d_apfp", |b| {
        b.iter(|| orient2d_apfp_batch(black_box(&samples)))
    });

    c.bench_function("orient2d_inexact_baseline", |b| {
        b.iter(|| orient2d_inexact_baseline_batch(black_box(&samples)))
    });

    c.bench_function("orient2d_inexact_interval", |b| {
        b.iter(|| orient2d_inexact_interval_batch(black_box(&samples)))
    });

    c.bench_function("orient2d_fixed", |b| {
        b.iter(|| orient2d_fixed_batch(black_box(&samples)))
    });

    c.bench_function("orient2d_rational", |b| {
        b.iter(|| orient2d_rational_batch(black_box(&samples)))
    });

    c.bench_function("orient2d_robust", |b| {
        b.iter(|| orient2d_robust_batch(black_box(&samples)))
    });
}

criterion_group!(benches, bench_orient2d);
criterion_main!(benches);

fn generate_samples(count: usize) -> Vec<(Coord, Coord, Coord)> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut samples = Vec::with_capacity(count);
    while samples.len() < count {
        let ax = lcg(&mut state);
        let ay = lcg(&mut state);
        let bx = lcg(&mut state);
        let by = lcg(&mut state);
        let cx = lcg(&mut state);
        let cy = lcg(&mut state);
        if !within_limits(&[ax, ay, bx, by, cx, cy]) {
            continue;
        }
        samples.push((Coord::new(ax, ay), Coord::new(bx, by), Coord::new(cx, cy)));
    }
    samples
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}

fn orient2d_rational(a: &Coord, b: &Coord, c: &Coord) -> GeometryPredicateResult {
    let Some(ax) = BigRational::from_float(a.x) else {
        return GeometryPredicateResult::Zero;
    };
    let Some(ay) = BigRational::from_float(a.y) else {
        return GeometryPredicateResult::Zero;
    };
    let Some(bx) = BigRational::from_float(b.x) else {
        return GeometryPredicateResult::Zero;
    };
    let Some(by) = BigRational::from_float(b.y) else {
        return GeometryPredicateResult::Zero;
    };
    let Some(cx) = BigRational::from_float(c.x) else {
        return GeometryPredicateResult::Zero;
    };
    let Some(cy) = BigRational::from_float(c.y) else {
        return GeometryPredicateResult::Zero;
    };

    let bax = &bx - &ax;
    let bay = &by - &ay;
    let cax = &cx - &ax;
    let cay = &cy - &ay;

    let det = bax * &cay - bay * &cax;
    result_from_rational(det)
}

fn robust_orient2d(a: &Coord, b: &Coord, c: &Coord) -> GeometryPredicateResult {
    let val = robust::orient2d(robust_coord(a), robust_coord(b), robust_coord(c));
    result_from_f64(val)
}

fn robust_coord(coord: &Coord) -> robust::Coord<f64> {
    robust::Coord {
        x: coord.x,
        y: coord.y,
    }
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
    if value == 0.0 {
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
