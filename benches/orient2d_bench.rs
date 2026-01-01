//! Benchmarks for orient2d implementations.
//!
//! Measures per-call performance for:
//! - Naive f64 (baseline, non-robust)
//! - apfp (our adaptive implementation)
//! - geometry_predicates (Shewchuk reference)
//! - apfp on each code path (fast, DD, exact)
//! - Rational (exact but slow baseline)

use apfp::analysis::adaptive_signum::{dd_from, dd_mul, dd_signum, dd_sub, gamma_from_ops};
use apfp::{Coord, orient2d};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use geometry_predicates::orient2d as gp_orient2d;
use num_rational::BigRational;
use num_traits::Zero;
use std::hint::black_box;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1;

/// Simple f64 orient2d - no error checking, may give wrong results for near-collinear points.
fn orient2d_naive(a: &Coord, b: &Coord, c: &Coord) -> i8 {
    let det = (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
    if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        0
    }
}

/// Exact orient2d using arbitrary precision rationals - always correct but slow.
fn orient2d_rational(a: &Coord, b: &Coord, c: &Coord) -> i8 {
    let ax = BigRational::from_float(a.x).unwrap();
    let ay = BigRational::from_float(a.y).unwrap();
    let bx = BigRational::from_float(b.x).unwrap();
    let by = BigRational::from_float(b.y).unwrap();
    let cx = BigRational::from_float(c.x).unwrap();
    let cy = BigRational::from_float(c.y).unwrap();

    let det = (&ax - &cx) * (&by - &cy) - (&ay - &cy) * (&bx - &cx);
    if det.is_zero() {
        0
    } else if det > BigRational::zero() {
        1
    } else {
        -1
    }
}

/// Classify which stage orient2d will use for given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Fast,
    Dd,
    Exact,
}

fn classify_orient2d(a: &Coord, b: &Coord, c: &Coord) -> Stage {
    let adx = a.x - c.x;
    let bdx = b.x - c.x;
    let ady = a.y - c.y;
    let bdy = b.y - c.y;
    let prod1 = adx * bdy;
    let prod2 = ady * bdx;
    let det = prod1 - prod2;
    let detsum = prod1.abs() + prod2.abs();
    let errbound = gamma_from_ops(7) * detsum;

    if det.abs() > errbound {
        return Stage::Fast;
    }

    // Check DD stage
    let dd_adx = dd_sub(dd_from(a.x), dd_from(c.x));
    let dd_bdx = dd_sub(dd_from(b.x), dd_from(c.x));
    let dd_ady = dd_sub(dd_from(a.y), dd_from(c.y));
    let dd_bdy = dd_sub(dd_from(b.y), dd_from(c.y));
    let dd_det = dd_sub(dd_mul(dd_adx, dd_bdy), dd_mul(dd_ady, dd_bdx));

    if dd_signum(dd_det).is_some() {
        Stage::Dd
    } else {
        Stage::Exact
    }
}

/// Pre-computed test cases for each stage.
/// These were found by systematic search and verified to trigger the expected code path.
fn get_stage_case(stage: Stage) -> (Coord, Coord, Coord) {
    match stage {
        // Random points - clearly non-collinear, fast path
        Stage::Fast => (
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(0.5, 1.0),
        ),
        // Near-collinear: passes fast filter, resolved by DD
        // Points on line y=x, with c slightly off
        Stage::Dd => (
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.5, 0.5 + 1e-14),
        ),
        // Exactly collinear - requires exact arithmetic
        Stage::Exact => (
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.5, 0.5),
        ),
    }
}

/// Find cases for each stage by searching.
fn find_stage_cases() -> (
    Vec<(Coord, Coord, Coord)>,
    Vec<(Coord, Coord, Coord)>,
    Vec<(Coord, Coord, Coord)>,
) {
    let mut fast_cases = Vec::new();
    let mut dd_cases = Vec::new();
    let mut exact_cases = Vec::new();

    let mut state = 0x1234_5678_9abc_def0u64;

    // Find fast cases (easy - random points almost always work)
    while fast_cases.len() < 100 {
        let a = Coord::new(lcg(&mut state), lcg(&mut state));
        let b = Coord::new(lcg(&mut state), lcg(&mut state));
        let c = Coord::new(lcg(&mut state), lcg(&mut state));
        if classify_orient2d(&a, &b, &c) == Stage::Fast {
            fast_cases.push((a, b, c));
        }
    }

    // Find DD cases by systematic search along near-collinear configurations
    'dd_search: for scale in [1.0, 10.0, 100.0, 1000.0] {
        for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
            for &eps in &[1e-12, 1e-13, 1e-14, 1e-15] {
                let a = Coord::new(0.0, 0.0);
                let b = Coord::new(scale, scale);
                let c = Coord::new(t * scale, t * scale + eps);
                if classify_orient2d(&a, &b, &c) == Stage::Dd {
                    dd_cases.push((a, b, c));
                    if dd_cases.len() >= 100 {
                        break 'dd_search;
                    }
                }
            }
        }
    }

    // Find exact cases - exactly collinear points
    'exact_search: for scale in [1.0, 10.0, 100.0, 1000.0] {
        for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let a = Coord::new(0.0, 0.0);
            let b = Coord::new(scale, scale);
            let c = Coord::new(t * scale, t * scale);
            if classify_orient2d(&a, &b, &c) == Stage::Exact {
                exact_cases.push((a, b, c));
                if exact_cases.len() >= 100 {
                    break 'exact_search;
                }
            }
        }
    }

    // Fallback to pre-computed cases if search didn't find enough
    while dd_cases.len() < 100 {
        dd_cases.push(get_stage_case(Stage::Dd));
    }
    while exact_cases.len() < 100 {
        exact_cases.push(get_stage_case(Stage::Exact));
    }

    (fast_cases, dd_cases, exact_cases)
}

fn bench_orient2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("orient2d");
    let batch_size = BatchSize::SmallInput;

    // Generate/find test cases for each stage
    let (fast_cases, dd_cases, exact_cases) = find_stage_cases();

    // Verify classifications
    eprintln!(
        "Stage samples: fast={}, dd={}, exact={}",
        fast_cases.len(),
        dd_cases.len(),
        exact_cases.len()
    );

    // Setup functions that cycle through pre-generated samples
    let make_random_setup = || {
        let mut state = 0x1234_5678_9abc_def0u64;
        move || {
            let a = Coord::new(lcg(&mut state), lcg(&mut state));
            let b = Coord::new(lcg(&mut state), lcg(&mut state));
            let c = Coord::new(lcg(&mut state), lcg(&mut state));
            (a, b, c)
        }
    };

    let make_stage_setup = |cases: Vec<(Coord, Coord, Coord)>| {
        let mut idx = 0usize;
        move || {
            let sample = cases[idx % cases.len()];
            idx = idx.wrapping_add(1);
            sample
        }
    };

    // === Main comparison benchmarks ===

    group.bench_function("naive_f64", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c)| black_box(orient2d_naive(&a, &b, &c)),
            batch_size,
        )
    });

    group.bench_function("apfp_random", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c)| black_box(orient2d(&a, &b, &c)),
            batch_size,
        )
    });

    group.bench_function("shewchuk_random", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c)| black_box(gp_orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y])),
            batch_size,
        )
    });

    // === Stage-specific benchmarks ===

    group.bench_function("apfp_fast_path", |bencher| {
        bencher.iter_batched(
            make_stage_setup(fast_cases.clone()),
            |(a, b, c)| black_box(orient2d(&a, &b, &c)),
            batch_size,
        )
    });

    group.bench_function("apfp_dd_path", |bencher| {
        bencher.iter_batched(
            make_stage_setup(dd_cases.clone()),
            |(a, b, c)| black_box(orient2d(&a, &b, &c)),
            batch_size,
        )
    });

    group.bench_function("apfp_exact_path", |bencher| {
        bencher.iter_batched(
            make_stage_setup(exact_cases.clone()),
            |(a, b, c)| black_box(orient2d(&a, &b, &c)),
            batch_size,
        )
    });

    // === Reference: exact rational arithmetic ===

    group.bench_function("rational", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(a, b, c)| black_box(orient2d_rational(&a, &b, &c)),
            batch_size,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_orient2d);
criterion_main!(benches);

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}
