use apfp::{Coord, apfp_signum};
use criterion::{Criterion, criterion_group, criterion_main};
use num_rational::BigRational;
use num_traits::Signed;
use std::cmp::Ordering;
use std::hint::black_box;

const SAMPLE_COUNT: usize = 5_000;
const MAG_LIMIT: f64 = 1.0e6;

fn cmp_dist_fast(origin: Coord, p: Coord, q: Coord) -> Ordering {
    let pdx = p.x - origin.x;
    let pdy = p.y - origin.y;
    let qdx = q.x - origin.x;
    let qdy = q.y - origin.y;
    let pdist = pdx * pdx + pdy * pdy;
    let qdist = qdx * qdx + qdy * qdy;
    if pdist > qdist {
        Ordering::Greater
    } else if pdist < qdist {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

fn cmp_dist_macro(origin: Coord, p: Coord, q: Coord) -> Ordering {
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

fn to_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    BigRational::from_float(value)
}

fn cmp_dist_rational(origin: Coord, p: Coord, q: Coord) -> Option<Ordering> {
    let ox = to_rational(origin.x)?;
    let oy = to_rational(origin.y)?;
    let px = to_rational(p.x)?;
    let py = to_rational(p.y)?;
    let qx = to_rational(q.x)?;
    let qy = to_rational(q.y)?;

    let pdx = &px - &ox;
    let pdy = &py - &oy;
    let qdx = &qx - &ox;
    let qdy = &qy - &oy;

    let pdist = (&pdx * &pdx) + (&pdy * &pdy);
    let qdist = (&qdx * &qdx) + (&qdy * &qdy);
    let diff = pdist - qdist;

    if diff.is_positive() {
        Some(Ordering::Greater)
    } else if diff.is_negative() {
        Some(Ordering::Less)
    } else {
        Some(Ordering::Equal)
    }
}

fn cmp_dist_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        black_box(cmp_dist_fast(*origin, *p, *q));
    }
}

fn cmp_dist_macro_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        black_box(cmp_dist_macro(*origin, *p, *q));
    }
}

fn cmp_dist_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        black_box(cmp_dist_rational(*origin, *p, *q));
    }
}

fn bench_cmp_dist(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);
    let mut group = c.benchmark_group("cmp_dist_implementations");

    group.bench_function("cmp_dist_fast", |b| {
        b.iter(|| cmp_dist_fast_batch(black_box(&samples)))
    });

    group.bench_function("cmp_dist_macro", |b| {
        b.iter(|| cmp_dist_macro_batch(black_box(&samples)))
    });

    group.bench_function("cmp_dist_rational", |b| {
        b.iter(|| cmp_dist_rational_batch(black_box(&samples)))
    });

    group.finish();
}

criterion_group!(benches, bench_cmp_dist);
criterion_main!(benches);

fn generate_samples(count: usize) -> Vec<(Coord, Coord, Coord)> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut samples = Vec::with_capacity(count);
    while samples.len() < count {
        let ox = lcg(&mut state);
        let oy = lcg(&mut state);
        let px = lcg(&mut state);
        let py = lcg(&mut state);
        let qx = lcg(&mut state);
        let qy = lcg(&mut state);
        if !within_limits(&[ox, oy, px, py, qx, qy]) {
            continue;
        }
        samples.push((Coord::new(ox, oy), Coord::new(px, py), Coord::new(qx, qy)));
    }
    samples
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}

fn within_limits(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= MAG_LIMIT)
}
