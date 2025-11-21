use apfp::analysis::ast_static::orient2d_exact;
use apfp::analysis::{orient2d_adaptive, orient2d_direct, orient2d_fast};
use apfp::{Coord, orient2d_rational};
use criterion::{Criterion, criterion_group, criterion_main};
use geometry_predicates::orient2d;
use std::hint::black_box;

/// Number of random test cases to generate for benchmarking
const SAMPLE_COUNT: usize = 5_000;

/// Maximum absolute value for coordinate components to avoid overflow
const MAG_LIMIT: f64 = 1.0e6;

/// Benchmark the fast floating-point filter for orient2d.
/// This is the fastest implementation but may return None for difficult cases.
fn orient2d_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_fast(*a, *b, *c));
    }
}

/// Benchmark the adaptive AST-based orient2d with fast-path filtering.
/// Uses fast floating-point filter followed by exact expansion arithmetic.
fn orient2d_adaptive_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_adaptive(*a, *b, *c));
    }
}

/// Benchmark the pure AST-based exact expansion arithmetic for orient2d.
/// Uses Shewchuk's expansion arithmetic without fast-path filtering.
fn orient2d_exact_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_exact(*a, *b, *c));
    }
}

/// Benchmark the direct expansion arithmetic for orient2d.
/// Direct computation without AST overhead.
fn orient2d_direct_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_direct(*a, *b, *c));
    }
}

/// Benchmark the AST-based rational arithmetic for orient2d.
/// Uses BigRational for mathematically exact computation.
fn orient2d_rational_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_rational(*a, *b, *c));
    }
}

/// Benchmark the geometry-predicates crate's orient2d implementation.
/// Uses adaptive precision arithmetic for exact computation.
fn orient2d_geometry_predicates_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y]));
    }
}

/// Benchmark suite comparing different orient2d implementations:
/// - orient2d_fast: Fast floating-point filter (may return None) ~4.6µs
/// - orient2d_geometry_predicates: geometry-predicates crate's adaptive precision ~5.3µs
/// - orient2d_adaptive: AST-based adaptive precision (fast filter + exact) ~10.3µs
/// - orient2d_exact: Pure AST-based exact expansion arithmetic ~441µs
/// - orient2d_direct: Direct expansion arithmetic without AST ~623µs
/// - orient2d_rational: AST-based BigRational arithmetic (reference) ~43ms
fn bench_orient2d(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);

    let mut group = c.benchmark_group("orient2d_implementations");

    group.bench_function("orient2d_fast", |b| {
        b.iter(|| orient2d_fast_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_adaptive", |b| {
        b.iter(|| orient2d_adaptive_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_exact", |b| {
        b.iter(|| orient2d_exact_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_direct", |b| {
        b.iter(|| orient2d_direct_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_rational", |b| {
        b.iter(|| orient2d_rational_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_geometry_predicates", |b| {
        b.iter(|| orient2d_geometry_predicates_batch(black_box(&samples)))
    });

    group.finish();
}

/// Benchmark suite for co-linear points - the most challenging case for geometric predicates.
/// Tests performance when all three points lie on the same line.
fn bench_orient2d_collinear(c: &mut Criterion) {
    let samples = generate_collinear_samples(SAMPLE_COUNT);

    let mut group = c.benchmark_group("orient2d_collinear_points");

    group.bench_function("orient2d_fast", |b| {
        b.iter(|| orient2d_fast_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_adaptive", |b| {
        b.iter(|| orient2d_adaptive_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_exact", |b| {
        b.iter(|| orient2d_exact_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_direct", |b| {
        b.iter(|| orient2d_direct_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_rational", |b| {
        b.iter(|| orient2d_rational_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_geometry_predicates", |b| {
        b.iter(|| orient2d_geometry_predicates_batch(black_box(&samples)))
    });

    group.finish();
}

criterion_group!(benches, bench_orient2d, bench_orient2d_collinear);
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

fn within_limits(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= MAG_LIMIT)
}

/// Generate samples where all three points are co-linear (lie on the same line).
/// This creates the most challenging test cases for geometric predicates.
fn generate_collinear_samples(count: usize) -> Vec<(Coord, Coord, Coord)> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut samples = Vec::with_capacity(count);

    while samples.len() < count {
        // Generate a base line defined by two points
        let x1 = lcg(&mut state);
        let y1 = lcg(&mut state);
        let x2 = lcg(&mut state);
        let y2 = lcg(&mut state);

        // Skip if base points are too close (degenerate line)
        let dx = x2 - x1;
        let dy = y2 - y1;
        if (dx * dx + dy * dy).sqrt() < 1e-6 {
            continue;
        }

        // Generate three points along this line with small perturbations
        // This creates near-co-linear configurations that challenge floating-point precision
        let t1 = lcg(&mut state) * 0.1; // Small parameter along line
        let t2 = 0.5 + lcg(&mut state) * 0.1; // Middle region
        let t3 = 1.0 + lcg(&mut state) * 0.1; // End region

        // Calculate points along the line
        let px1 = x1 + t1 * dx;
        let py1 = y1 + t1 * dy;
        let px2 = x1 + t2 * dx;
        let py2 = y1 + t2 * dy;
        let px3 = x1 + t3 * dx;
        let py3 = y1 + t3 * dy;

        // Add tiny perturbations to make them slightly non-co-linear
        // This creates the challenging near-co-linear cases
        let eps = 1e-15; // Extremely small perturbation for near-collinear cases
        let px3_pert = px3 + eps * lcg(&mut state);
        let py3_pert = py3 + eps * lcg(&mut state);

        let coords = [px1, py1, px2, py2, px3_pert, py3_pert];
        if within_limits(&coords) {
            samples.push((
                Coord::new(px1, py1),
                Coord::new(px2, py2),
                Coord::new(px3_pert, py3_pert),
            ));
        }
    }

    samples
}
