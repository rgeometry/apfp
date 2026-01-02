use apfp::analysis::adaptive_signum::{
    Dd, dd_add, dd_from, dd_mul, dd_signum, dd_sub, gamma_from_ops,
};
use apfp::{Coord, apfp_signum, orient2d_normal, orient2d_vec};
use criterion::{Criterion, criterion_group, criterion_main};
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

/// Fast f64 implementation (lower performance bound)
fn orient2d_vec_fast(a: &Coord, v: &Coord, p: &Coord) -> f64 {
    let dx = a.x - p.x;
    let dy = a.y - p.y;
    dx * v.y - dy * v.x
}

/// BigRational implementation (upper performance bound - slowest but exact)
fn orient2d_vec_rational(a: &Coord, v: &Coord, p: &Coord) -> BigRational {
    let ax = BigRational::from_float(a.x).expect("inputs must be finite");
    let ay = BigRational::from_float(a.y).expect("inputs must be finite");
    let vx = BigRational::from_float(v.x).expect("inputs must be finite");
    let vy = BigRational::from_float(v.y).expect("inputs must be finite");
    let px = BigRational::from_float(p.x).expect("inputs must be finite");
    let py = BigRational::from_float(p.y).expect("inputs must be finite");

    let dx = &ax - &px;
    let dy = &ay - &py;
    &dx * &vy - &dy * &vx
}

fn orient2d_vec_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, v, p) in samples {
        black_box(orient2d_vec_fast(a, v, p));
    }
}

fn orient2d_vec_apfp_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, v, p) in samples {
        black_box(orient2d_vec(a, v, p));
    }
}

fn orient2d_vec_apfp_signum_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, v, p) in samples {
        let sign = apfp_signum!((a.x - p.x) * v.y - (a.y - p.y) * v.x);
        black_box(sign);
    }
}

fn orient2d_vec_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, v, p) in samples {
        let det = orient2d_vec_rational(a, v, p);
        black_box(det.is_zero());
    }
}

fn bench_orient2d_vec(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);
    let stage_fast = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Fast);
    let stage_dd = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Dd);
    let stage_exact = generate_stage_samples(STAGE_SAMPLE_COUNT, Stage::Exact);
    let mut group = c.benchmark_group("orient2d_vec_implementations");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));

    group.bench_function("orient2d_vec_fast", |b| {
        b.iter(|| orient2d_vec_fast_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_vec_apfp", |b| {
        b.iter(|| orient2d_vec_apfp_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_vec_apfp_fast_stage", |b| {
        b.iter(|| orient2d_vec_apfp_signum_batch(black_box(&stage_fast)))
    });

    group.bench_function("orient2d_vec_apfp_dd_stage", |b| {
        b.iter(|| orient2d_vec_apfp_signum_batch(black_box(&stage_dd)))
    });

    group.bench_function("orient2d_vec_apfp_exact_stage", |b| {
        b.iter(|| orient2d_vec_apfp_signum_batch(black_box(&stage_exact)))
    });

    group.bench_function("orient2d_vec_rational", |b| {
        b.iter(|| orient2d_vec_rational_batch(black_box(&samples)))
    });

    group.finish();
}

criterion_group!(benches, bench_orient2d_vec, bench_orient2d_normal);
criterion_main!(benches);

fn generate_samples(count: usize) -> Vec<(Coord, Coord, Coord)> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut samples = Vec::with_capacity(count);
    while samples.len() < count {
        let ax = lcg(&mut state);
        let ay = lcg(&mut state);
        let vx = lcg(&mut state);
        let vy = lcg(&mut state);
        let px = lcg(&mut state);
        let py = lcg(&mut state);
        if !within_limits(&[ax, ay, vx, vy, px, py]) {
            continue;
        }
        samples.push((Coord::new(ax, ay), Coord::new(vx, vy), Coord::new(px, py)));
    }
    samples
}

fn generate_stage_samples(count: usize, stage: Stage) -> Vec<(Coord, Coord, Coord)> {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ (stage as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let mut attempts = 0usize;
    let mut sample = None;
    while sample.is_none() && attempts < STAGE_MAX_ATTEMPTS {
        attempts += 1;
        sample = find_orient2d_vec_case(&mut seed, stage);
    }
    if sample.is_none() {
        sample = find_orient2d_vec_case_deterministic(stage);
    }
    let Some(sample) = sample else {
        return Vec::new();
    };
    vec![sample; count]
}

fn find_orient2d_vec_case(seed: &mut u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    for _ in 0..400 {
        if stage == Stage::Fast {
            let ax = lcg_range(seed, -1.0e3, 1.0e3);
            let ay = lcg_range(seed, -1.0e3, 1.0e3);
            let vx = lcg_range(seed, -1.0e3, 1.0e3);
            let vy = lcg_range(seed, -1.0e3, 1.0e3);
            let px = lcg_range(seed, -1.0e3, 1.0e3);
            let py = lcg_range(seed, -1.0e3, 1.0e3);
            let a = Coord::new(ax, ay);
            let v = Coord::new(vx, vy);
            let p = Coord::new(px, py);
            if classify_orient2d_vec(&a, &v, &p) == Some(stage) {
                return Some((a, v, p));
            }
            continue;
        }

        // For Dd and Exact stages, construct near-collinear cases
        let vx = lcg_range(seed, -1.0e3, 1.0e3);
        let vy = lcg_range(seed, -1.0e3, 1.0e3);
        if vx.abs() + vy.abs() < 1.0e-12 {
            continue;
        }
        let t = lcg_range(seed, -1.0e3, 1.0e3);
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

fn find_orient2d_vec_case_deterministic(stage: Stage) -> Option<(Coord, Coord, Coord)> {
    let directions = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)];
    let scales = [1.0, 1.0e3, 1.0e6, 1.0e9];
    for &(vx, vy) in &directions {
        for &scale in &scales {
            let vx = vx * scale;
            let vy = vy * scale;
            for &t in &scales {
                for &eps in &EPS_LIST {
                    let a = Coord::new(0.0, 0.0);
                    let v = Coord::new(vx, vy);
                    let p = Coord::new(t * vx - eps * vy, t * vy + eps * vx);
                    if classify_orient2d_vec(&a, &v, &p) == Some(stage) {
                        return Some((a, v, p));
                    }
                }
            }
        }
    }
    None
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
    let errbound = gamma_from_ops(5) * detsum; // 5 ops: 2 subs, 2 muls, 1 sub

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

fn dd_orient2d_vec(ax: f64, ay: f64, vx: f64, vy: f64, px: f64, py: f64) -> Dd {
    // (a.x - p.x) * v.y - (a.y - p.y) * v.x
    let dx = dd_sub(dd_from(ax), dd_from(px));
    let dy = dd_sub(dd_from(ay), dd_from(py));
    let prod1 = dd_mul(dx, dd_from(vy));
    let prod2 = dd_mul(dy, dd_from(vx));
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

// ============================================================================
// orient2d_normal benchmarks
// ============================================================================

/// Fast f64 implementation (lower performance bound)
fn orient2d_normal_fast(a: &Coord, n: &Coord, p: &Coord) -> f64 {
    let dx = a.x - p.x;
    let dy = a.y - p.y;
    dx * n.x + dy * n.y
}

/// BigRational implementation (upper performance bound - slowest but exact)
fn orient2d_normal_rational(a: &Coord, n: &Coord, p: &Coord) -> BigRational {
    let ax = BigRational::from_float(a.x).expect("inputs must be finite");
    let ay = BigRational::from_float(a.y).expect("inputs must be finite");
    let nx = BigRational::from_float(n.x).expect("inputs must be finite");
    let ny = BigRational::from_float(n.y).expect("inputs must be finite");
    let px = BigRational::from_float(p.x).expect("inputs must be finite");
    let py = BigRational::from_float(p.y).expect("inputs must be finite");

    let dx = &ax - &px;
    let dy = &ay - &py;
    &dx * &nx + &dy * &ny
}

fn orient2d_normal_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, n, p) in samples {
        black_box(orient2d_normal_fast(a, n, p));
    }
}

fn orient2d_normal_apfp_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, n, p) in samples {
        black_box(orient2d_normal(a, n, p));
    }
}

fn orient2d_normal_apfp_signum_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, n, p) in samples {
        let sign = apfp_signum!((a.x - p.x) * n.x + (a.y - p.y) * n.y);
        black_box(sign);
    }
}

fn orient2d_normal_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, n, p) in samples {
        let det = orient2d_normal_rational(a, n, p);
        black_box(det.is_zero());
    }
}

fn bench_orient2d_normal(c: &mut Criterion) {
    let samples = generate_normal_samples(SAMPLE_COUNT);
    let stage_fast = generate_normal_stage_samples(STAGE_SAMPLE_COUNT, Stage::Fast);
    let stage_dd = generate_normal_stage_samples(STAGE_SAMPLE_COUNT, Stage::Dd);
    let stage_exact = generate_normal_stage_samples(STAGE_SAMPLE_COUNT, Stage::Exact);
    let mut group = c.benchmark_group("orient2d_normal_implementations");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));

    group.bench_function("orient2d_normal_fast", |b| {
        b.iter(|| orient2d_normal_fast_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_normal_apfp", |b| {
        b.iter(|| orient2d_normal_apfp_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_normal_apfp_fast_stage", |b| {
        b.iter(|| orient2d_normal_apfp_signum_batch(black_box(&stage_fast)))
    });

    group.bench_function("orient2d_normal_apfp_dd_stage", |b| {
        b.iter(|| orient2d_normal_apfp_signum_batch(black_box(&stage_dd)))
    });

    group.bench_function("orient2d_normal_apfp_exact_stage", |b| {
        b.iter(|| orient2d_normal_apfp_signum_batch(black_box(&stage_exact)))
    });

    group.bench_function("orient2d_normal_rational", |b| {
        b.iter(|| orient2d_normal_rational_batch(black_box(&samples)))
    });

    group.finish();
}

fn generate_normal_samples(count: usize) -> Vec<(Coord, Coord, Coord)> {
    let mut state = 0xfedcba9876543210u64;
    let mut samples = Vec::with_capacity(count);
    while samples.len() < count {
        let ax = lcg(&mut state);
        let ay = lcg(&mut state);
        let nx = lcg(&mut state);
        let ny = lcg(&mut state);
        let px = lcg(&mut state);
        let py = lcg(&mut state);
        if !within_limits(&[ax, ay, nx, ny, px, py]) {
            continue;
        }
        samples.push((Coord::new(ax, ay), Coord::new(nx, ny), Coord::new(px, py)));
    }
    samples
}

fn generate_normal_stage_samples(count: usize, stage: Stage) -> Vec<(Coord, Coord, Coord)> {
    let mut seed = 0xfedcba9876543210u64 ^ (stage as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let mut attempts = 0usize;
    let mut sample = None;
    while sample.is_none() && attempts < STAGE_MAX_ATTEMPTS {
        attempts += 1;
        sample = find_orient2d_normal_case(&mut seed, stage);
    }
    if sample.is_none() {
        sample = find_orient2d_normal_case_deterministic(stage);
    }
    let Some(sample) = sample else {
        return Vec::new();
    };
    vec![sample; count]
}

fn find_orient2d_normal_case(seed: &mut u64, stage: Stage) -> Option<(Coord, Coord, Coord)> {
    for _ in 0..400 {
        if stage == Stage::Fast {
            let ax = lcg_range(seed, -1.0e3, 1.0e3);
            let ay = lcg_range(seed, -1.0e3, 1.0e3);
            let nx = lcg_range(seed, -1.0e3, 1.0e3);
            let ny = lcg_range(seed, -1.0e3, 1.0e3);
            let px = lcg_range(seed, -1.0e3, 1.0e3);
            let py = lcg_range(seed, -1.0e3, 1.0e3);
            let a = Coord::new(ax, ay);
            let n = Coord::new(nx, ny);
            let p = Coord::new(px, py);
            if classify_orient2d_normal(&a, &n, &p) == Some(stage) {
                return Some((a, n, p));
            }
            continue;
        }

        // For Dd and Exact stages, construct near-on-line cases
        let nx = lcg_range(seed, -1.0e3, 1.0e3);
        let ny = lcg_range(seed, -1.0e3, 1.0e3);
        if nx.abs() + ny.abs() < 1.0e-12 {
            continue;
        }
        let t = lcg_range(seed, -1.0e3, 1.0e3);
        for &eps in &EPS_LIST {
            let a = Coord::new(0.0, 0.0);
            let n = Coord::new(nx, ny);
            // p = t * (-n.y, n.x) + eps * (n.x, n.y)
            let p = Coord::new(-t * ny + eps * nx, t * nx + eps * ny);
            if classify_orient2d_normal(&a, &n, &p) == Some(stage) {
                return Some((a, n, p));
            }
        }
    }
    None
}

fn find_orient2d_normal_case_deterministic(stage: Stage) -> Option<(Coord, Coord, Coord)> {
    let directions = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)];
    let scales = [1.0, 1.0e3, 1.0e6, 1.0e9];
    for &(nx, ny) in &directions {
        for &scale in &scales {
            let nx = nx * scale;
            let ny = ny * scale;
            for &t in &scales {
                for &eps in &EPS_LIST {
                    let a = Coord::new(0.0, 0.0);
                    let n = Coord::new(nx, ny);
                    let p = Coord::new(-t * ny + eps * nx, t * nx + eps * ny);
                    if classify_orient2d_normal(&a, &n, &p) == Some(stage) {
                        return Some((a, n, p));
                    }
                }
            }
        }
    }
    None
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
    let errbound = gamma_from_ops(5) * detsum;

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

fn dd_orient2d_normal(ax: f64, ay: f64, nx: f64, ny: f64, px: f64, py: f64) -> Dd {
    let dx = dd_sub(dd_from(ax), dd_from(px));
    let dy = dd_sub(dd_from(ay), dd_from(py));
    let prod1 = dd_mul(dx, dd_from(nx));
    let prod2 = dd_mul(dy, dd_from(ny));
    dd_add(prod1, prod2)
}
