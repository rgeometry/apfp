use apfp::{Coord, orient2d};
use criterion::{Criterion, criterion_group, criterion_main};
use geometry_predicates::orient2d as gp_orient2d;
use std::hint::black_box;

/// Number of random test cases to generate for benchmarking
const SAMPLE_COUNT: usize = 5_000;

/// Maximum absolute value for coordinate components to avoid overflow
const MAG_LIMIT: f64 = 1.0e6;

fn orient2d_apfp_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d(a, b, c));
    }
}

fn orient2d_geometry_predicates_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(gp_orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y]));
    }
}

fn bench_orient2d(c: &mut Criterion) {
    let samples = generate_samples(SAMPLE_COUNT);
    let mut group = c.benchmark_group("orient2d_implementations");

    group.bench_function("orient2d_apfp", |b| {
        b.iter(|| orient2d_apfp_batch(black_box(&samples)))
    });

    group.bench_function("orient2d_geometry_predicates", |b| {
        b.iter(|| orient2d_geometry_predicates_batch(black_box(&samples)))
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

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}

fn within_limits(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= MAG_LIMIT)
}
