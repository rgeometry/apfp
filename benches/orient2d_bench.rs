use apfp::analysis::adaptive_signum::{Dd, dd_from, dd_mul, dd_signum, dd_sub, gamma_from_ops};
use apfp::{Coord, apfp_signum, orient2d};
use criterion::{Criterion, criterion_group, criterion_main};
use geometry_predicates::orient2d as gp_orient2d;
use num_rational::BigRational;
use num_traits::Zero;
use std::hint::black_box;
use std::time::Duration;

/// Number of random test cases to generate for benchmarking
const SAMPLE_COUNT: usize = 5_000;
/// Number of stage-specific cases to generate per branch
const STAGE_SAMPLE_COUNT: usize = 50;
/// Cap attempts while searching for stage-specific samples
const STAGE_MAX_ATTEMPTS: usize = 200_000;

/// Maximum absolute value for coordinate components to avoid overflow
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

fn orient2d_fast(a: &Coord, b: &Coord, c: &Coord) -> f64 {
    let adx = a.x - c.x;
    let bdx = b.x - c.x;
    let ady = a.y - c.y;
    let bdy = b.y - c.y;
    (adx * bdy) - (ady * bdx)
}

fn orient2d_rational(a: &Coord, b: &Coord, c: &Coord) -> BigRational {
    let ax = BigRational::from_float(a.x).expect("inputs must be finite");
    let ay = BigRational::from_float(a.y).expect("inputs must be finite");
    let bx = BigRational::from_float(b.x).expect("inputs must be finite");
    let by = BigRational::from_float(b.y).expect("inputs must be finite");
    let cx = BigRational::from_float(c.x).expect("inputs must be finite");
    let cy = BigRational::from_float(c.y).expect("inputs must be finite");

    let adx = &ax - &cx;
    let bdx = &bx - &cx;
    let ady = &ay - &cy;
    let bdy = &by - &cy;

    &adx * &bdy - &ady * &bdx
}

fn orient2d_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_fast(a, b, c));
    }
}

fn orient2d_apfp_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d(a, b, c));
    }
}

fn orient2d_apfp_signum_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        let sign = apfp_signum!((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x));
        black_box(sign);
    }
}

fn orient2d_geometry_predicates_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(gp_orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y]));
    }
}

fn orient2d_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        let det = orient2d_rational(a, b, c);
        black_box(det.is_zero());
    }
}

fn bench_orient2d(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);
    let stage_fast = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Fast);
    let stage_dd = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Dd);
    let stage_exact = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Exact);
    let mut group = c.benchmark_group("orient2d_implementations");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));

    group.bench_function("orient2d_fast", |b| {
        b.iter(|| orient2d_fast_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_apfp", |b| {
        b.iter(|| orient2d_apfp_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_apfp_fast_stage", |b| {
        b.iter(|| orient2d_apfp_signum_batch(black_box(&stage_fast)))
    });

    group.bench_function("orient2d_apfp_dd_stage", |b| {
        b.iter(|| orient2d_apfp_signum_batch(black_box(&stage_dd)))
    });

    group.bench_function("orient2d_apfp_exact_stage", |b| {
        b.iter(|| orient2d_apfp_signum_batch(black_box(&stage_exact)))
    });

    group.bench_function("orient2d_geometry_predicates", |b| {
        b.iter(|| orient2d_geometry_predicates_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_rational", |b| {
        b.iter(|| orient2d_rational_batch(black_box(&samples)))
    });

    group.finish();
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

fn generate_stage_samples(count: usize, stage: Stage) -> Vec<(Coord, Coord, Coord)> {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ (stage as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let mut attempts = 0usize;
    let mut sample = None;
    while sample.is_none() && attempts < STAGE_MAX_ATTEMPTS {
        attempts += 1;
        sample = find_orient2d_case(&mut seed, stage);
    }
    if sample.is_none() {
        sample = find_orient2d_case_deterministic(stage);
    }
    let Some(sample) = sample else {
        return Vec::new();
    };
    vec![sample; count]
}

fn find_orient2d_case(seed: &mut u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    for _ in 0..400 {
        if stage == Stage::Fast {
            let ax = lcg_range(seed, -1.0e3, 1.0e3);
            let ay = lcg_range(seed, -1.0e3, 1.0e3);
            let bx = lcg_range(seed, -1.0e3, 1.0e3);
            let by = lcg_range(seed, -1.0e3, 1.0e3);
            let cx = lcg_range(seed, -1.0e3, 1.0e3);
            let cy = lcg_range(seed, -1.0e3, 1.0e3);
            let a = Coord::new(ax, ay);
            let b = Coord::new(bx, by);
            let c = Coord::new(cx, cy);
            if classify_orient2d(&a, &b, &c) == Some(stage) {
                return Some((a, b, c));
            }
            continue;
        }

        let dx = lcg_range(seed, -1.0e3, 1.0e3);
        let dy = lcg_range(seed, -1.0e3, 1.0e3);
        if dx.abs() + dy.abs() < 1.0e-12 {
            continue;
        }
        let t1 = lcg_range(seed, -1.0e3, 1.0e3);
        let t2 = lcg_range(seed, -1.0e3, 1.0e3);
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

fn find_orient2d_case_deterministic(stage: Stage) -> Option<(Coord, Coord, Coord)> {
    let directions = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)];
    let scales = [1.0, 1.0e3, 1.0e6, 1.0e9];
    for &(dx, dy) in &directions {
        for &scale in &scales {
            let dx = dx * scale;
            let dy = dy * scale;
            for &t1 in &scales {
                for &t2 in &scales {
                    for &eps in &EPS_LIST {
                        let a = Coord::new(t1 * dx, t1 * dy);
                        let b = Coord::new(t2 * dx - eps * dy, t2 * dy + eps * dx);
                        let c = Coord::new(0.0, 0.0);
                        if classify_orient2d(&a, &b, &c) == Some(stage) {
                            return Some((a, b, c));
                        }
                    }
                }
            }
        }
    }
    None
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
    let errbound = gamma_from_ops(7) * detsum;

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

fn dd_orient2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> Dd {
    let adx = dd_sub(dd_from(ax), dd_from(cx));
    let bdx = dd_sub(dd_from(bx), dd_from(cx));
    let ady = dd_sub(dd_from(ay), dd_from(cy));
    let bdy = dd_sub(dd_from(by), dd_from(cy));
    let prod1 = dd_mul(adx, bdy);
    let prod2 = dd_mul(ady, bdx);
    dd_sub(prod1, prod2)
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
