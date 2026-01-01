use apfp::analysis::adaptive_signum::{
    Dd, dd_add, dd_from, dd_signum, dd_square, dd_sub, gamma_from_ops,
};
use apfp::{Coord, apfp_signum, cmp_dist, square};
use criterion::{Criterion, criterion_group, criterion_main};
use num_rational::BigRational;
use num_traits::Signed;
use std::cmp::Ordering;
use std::hint::black_box;

const SAMPLE_COUNT: usize = 5_000;
const STAGE_SAMPLE_COUNT: usize = 1_000;
const STAGE_MAX_ATTEMPTS: usize = 2_000_000;
const MAG_LIMIT: f64 = 1.0e6;

const EPS_LIST: [f64; 12] = [
    1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14, 1.0e-16, 1.0e-18, 1.0e-20, 1.0e-24,
    1.0e-30,
];

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Fast,
    Dd,
    Exact,
}

fn cmp_dist_fast(origin: &Coord, p: &Coord, q: &Coord) -> Ordering {
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

fn cmp_dist_rational(origin: &Coord, p: &Coord, q: &Coord) -> Ordering {
    let ox = BigRational::from_float(origin.x).expect("inputs must be finite");
    let oy = BigRational::from_float(origin.y).expect("inputs must be finite");
    let px = BigRational::from_float(p.x).expect("inputs must be finite");
    let py = BigRational::from_float(p.y).expect("inputs must be finite");
    let qx = BigRational::from_float(q.x).expect("inputs must be finite");
    let qy = BigRational::from_float(q.y).expect("inputs must be finite");

    let pdx = &px - &ox;
    let pdy = &py - &oy;
    let qdx = &qx - &ox;
    let qdy = &qy - &oy;

    let pdist = (&pdx * &pdx) + (&pdy * &pdy);
    let qdist = (&qdx * &qdx) + (&qdy * &qdy);
    let diff = pdist - qdist;

    if diff.is_positive() {
        Ordering::Greater
    } else if diff.is_negative() {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

fn cmp_dist_apfp_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        black_box(cmp_dist(origin, p, q));
    }
}

fn cmp_dist_apfp_signum_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        let sign = apfp_signum!(
            (square(p.x - origin.x) + square(p.y - origin.y))
                - (square(q.x - origin.x) + square(q.y - origin.y))
        );
        black_box(sign);
    }
}

fn cmp_dist_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        black_box(cmp_dist_fast(origin, p, q));
    }
}

fn cmp_dist_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (origin, p, q) in samples {
        black_box(cmp_dist_rational(origin, p, q));
    }
}

fn bench_cmp_dist(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);
    let stage_fast = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Fast);
    let stage_dd = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Dd);
    let stage_exact = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Exact);
    let mut group = c.benchmark_group("cmp_dist_implementations");

    group.bench_function("cmp_dist_fast", |b| {
        b.iter(|| cmp_dist_fast_batch(black_box(&samples)))
    });

    group.bench_function("cmp_dist_apfp", |b| {
        b.iter(|| cmp_dist_apfp_batch(black_box(&samples)))
    });

    group.bench_function("cmp_dist_apfp_fast_stage", |b| {
        b.iter(|| cmp_dist_apfp_signum_batch(black_box(&stage_fast)))
    });

    group.bench_function("cmp_dist_apfp_dd_stage", |b| {
        b.iter(|| cmp_dist_apfp_signum_batch(black_box(&stage_dd)))
    });

    group.bench_function("cmp_dist_apfp_exact_stage", |b| {
        b.iter(|| cmp_dist_apfp_signum_batch(black_box(&stage_exact)))
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

fn generate_stage_samples(count: usize, stage: Stage) -> Vec<(Coord, Coord, Coord)> {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ (stage as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let mut samples = Vec::with_capacity(count);
    let mut attempts = 0usize;
    while samples.len() < count && attempts < STAGE_MAX_ATTEMPTS {
        attempts += 1;
        if let Some(sample) = find_cmp_dist_case(&mut seed, stage) {
            samples.push(sample);
        }
    }
    samples
}

fn find_cmp_dist_case(seed: &mut u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    let origin = Coord::new(0.0, 0.0);
    for _ in 0..400 {
        if stage == Stage::Fast {
            let px = lcg_range(seed, -1.0e3, 1.0e3);
            let py = lcg_range(seed, -1.0e3, 1.0e3);
            let qx = lcg_range(seed, -1.0e3, 1.0e3);
            let qy = lcg_range(seed, -1.0e3, 1.0e3);
            let p = Coord::new(px, py);
            let q = Coord::new(qx, qy);
            if classify_cmp_dist(&origin, &p, &q) == Some(stage) {
                return Some((origin, p, q));
            }
            continue;
        }

        let r = lcg_range(seed, 1.0e3, 1.0e6);
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
    let errbound = gamma_from_ops(11) * sum;

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

fn dd_cmp_dist(ox: f64, oy: f64, px: f64, py: f64, qx: f64, qy: f64) -> Dd {
    let pdx = dd_sub(dd_from(px), dd_from(ox));
    let pdy = dd_sub(dd_from(py), dd_from(oy));
    let qdx = dd_sub(dd_from(qx), dd_from(ox));
    let qdy = dd_sub(dd_from(qy), dd_from(oy));
    let pdist = dd_add(dd_square(pdx), dd_square(pdy));
    let qdist = dd_add(dd_square(qdx), dd_square(qdy));
    dd_sub(pdist, qdist)
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}

fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2.0) - 1.0
}

fn lcg_range(state: &mut u64, min: f64, max: f64) -> f64 {
    min + (max - min) * (lcg_next(state) * 0.5 + 0.5)
}

fn within_limits(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= MAG_LIMIT)
}
