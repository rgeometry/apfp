//! Benchmarks for cmp_dist implementations.
//!
//! Measures per-call performance for:
//! - Naive f64 (baseline, non-robust)
//! - apfp (our adaptive implementation)
//! - apfp on each code path (fast, DD, exact)
//! - Rational (exact but slow baseline)

use apfp::analysis::adaptive_signum::{
    dd_add, dd_from, dd_signum, dd_square, dd_sub, gamma_from_ops,
};
use apfp::geometry::f64::{Coord, cmp_dist};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use num_rational::BigRational;
use num_traits::Signed;
use std::cmp::Ordering;
use std::hint::black_box;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1;

/// Type alias for benchmark test cases: (origin, p, q) coordinate triples.
type StageCases = Vec<(Coord, Coord, Coord)>;

/// Simple f64 distance comparison - no error checking, may give wrong results.
fn cmp_dist_naive(origin: &Coord, p: &Coord, q: &Coord) -> Ordering {
    let pdx = p.x - origin.x;
    let pdy = p.y - origin.y;
    let qdx = q.x - origin.x;
    let qdy = q.y - origin.y;
    let pdist = pdx * pdx + pdy * pdy;
    let qdist = qdx * qdx + qdy * qdy;
    pdist.partial_cmp(&qdist).unwrap_or(Ordering::Equal)
}

/// Exact distance comparison using arbitrary precision rationals.
fn cmp_dist_rational(origin: &Coord, p: &Coord, q: &Coord) -> Ordering {
    let ox = BigRational::from_float(origin.x).unwrap();
    let oy = BigRational::from_float(origin.y).unwrap();
    let px = BigRational::from_float(p.x).unwrap();
    let py = BigRational::from_float(p.y).unwrap();
    let qx = BigRational::from_float(q.x).unwrap();
    let qy = BigRational::from_float(q.y).unwrap();

    let pdx = &px - &ox;
    let pdy = &py - &oy;
    let qdx = &qx - &ox;
    let qdy = &qy - &oy;

    let pdist = &pdx * &pdx + &pdy * &pdy;
    let qdist = &qdx * &qdx + &qdy * &qdy;
    let diff = pdist - qdist;

    if diff.is_positive() {
        Ordering::Greater
    } else if diff.is_negative() {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

/// Classify which stage cmp_dist will use for given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Fast,
    Dd,
    Exact,
}

fn classify_cmp_dist(origin: &Coord, p: &Coord, q: &Coord) -> Stage {
    let pdx = p.x - origin.x;
    let pdy = p.y - origin.y;
    let qdx = q.x - origin.x;
    let qdy = q.y - origin.y;
    let pdist = pdx * pdx + pdy * pdy;
    let qdist = qdx * qdx + qdy * qdy;
    let diff = pdist - qdist;
    let sum = pdist.abs() + qdist.abs();
    let errbound = gamma_from_ops(11) * sum;

    if diff.abs() > errbound {
        return Stage::Fast;
    }

    // Check DD stage
    let dd_pdx = dd_sub(dd_from(p.x), dd_from(origin.x));
    let dd_pdy = dd_sub(dd_from(p.y), dd_from(origin.y));
    let dd_qdx = dd_sub(dd_from(q.x), dd_from(origin.x));
    let dd_qdy = dd_sub(dd_from(q.y), dd_from(origin.y));
    let dd_pdist = dd_add(dd_square(dd_pdx), dd_square(dd_pdy));
    let dd_qdist = dd_add(dd_square(dd_qdx), dd_square(dd_qdy));
    let dd_diff = dd_sub(dd_pdist, dd_qdist);

    if dd_signum(dd_diff).is_some() {
        Stage::Dd
    } else {
        Stage::Exact
    }
}

/// Find cases for each stage by searching.
fn find_stage_cases() -> (StageCases, StageCases, StageCases) {
    let mut fast_cases = Vec::new();
    let mut dd_cases = Vec::new();
    let mut exact_cases = Vec::new();

    let mut state = 0xfedcba9876543210u64;
    let origin = Coord::new(0.0, 0.0);

    // Find fast cases (easy - random points almost always work)
    while fast_cases.len() < 100 {
        let p = Coord::new(lcg(&mut state), lcg(&mut state));
        let q = Coord::new(lcg(&mut state), lcg(&mut state));
        if classify_cmp_dist(&origin, &p, &q) == Stage::Fast {
            fast_cases.push((origin, p, q));
        }
    }

    // Find DD cases: points at nearly equal distances
    'dd_search: for r in [10.0, 100.0, 1000.0] {
        for &eps in &[1e-11, 1e-12, 1e-13, 1e-14] {
            let p = Coord::new(r, 0.0);
            let q = Coord::new(r + eps, 0.0);
            if classify_cmp_dist(&origin, &p, &q) == Stage::Dd {
                dd_cases.push((origin, p, q));
                if dd_cases.len() >= 100 {
                    break 'dd_search;
                }
            }
        }
    }

    // Find exact cases: points at exactly equal distances
    'exact_search: for r in [1.0, 10.0, 100.0, 1000.0] {
        // Points at same distance but different angles
        let p = Coord::new(r, 0.0);
        let q = Coord::new(0.0, r);
        if classify_cmp_dist(&origin, &p, &q) == Stage::Exact {
            exact_cases.push((origin, p, q));
            if exact_cases.len() >= 100 {
                break 'exact_search;
            }
        }
        // Try other angle combinations
        let q2 = Coord::new(r * 0.6, r * 0.8); // 3-4-5 triangle scaling
        if classify_cmp_dist(&origin, &p, &q2) == Stage::Exact {
            exact_cases.push((origin, p, q2));
            if exact_cases.len() >= 100 {
                break 'exact_search;
            }
        }
    }

    // Fallback: duplicate what we have
    while dd_cases.len() < 100 {
        if dd_cases.is_empty() {
            // Hardcoded fallback
            dd_cases.push((
                origin,
                Coord::new(100.0, 0.0),
                Coord::new(100.0 + 1e-13, 0.0),
            ));
        } else {
            dd_cases.push(dd_cases[0]);
        }
    }
    while exact_cases.len() < 100 {
        if exact_cases.is_empty() {
            exact_cases.push((origin, Coord::new(1.0, 0.0), Coord::new(0.0, 1.0)));
        } else {
            exact_cases.push(exact_cases[0]);
        }
    }

    (fast_cases, dd_cases, exact_cases)
}

fn bench_cmp_dist(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp_dist");
    let batch_size = BatchSize::SmallInput;

    let (fast_cases, dd_cases, exact_cases) = find_stage_cases();

    eprintln!(
        "Stage samples: fast={}, dd={}, exact={}",
        fast_cases.len(),
        dd_cases.len(),
        exact_cases.len()
    );

    let make_random_setup = || {
        let mut state = 0xfedcba9876543210u64;
        move || {
            let origin = Coord::new(lcg(&mut state), lcg(&mut state));
            let p = Coord::new(lcg(&mut state), lcg(&mut state));
            let q = Coord::new(lcg(&mut state), lcg(&mut state));
            (origin, p, q)
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
            |(origin, p, q)| black_box(cmp_dist_naive(&origin, &p, &q)),
            batch_size,
        )
    });

    group.bench_function("apfp_random", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(origin, p, q)| black_box(cmp_dist(&origin, &p, &q)),
            batch_size,
        )
    });

    // === Stage-specific benchmarks ===

    group.bench_function("apfp_fast_path", |bencher| {
        bencher.iter_batched(
            make_stage_setup(fast_cases.clone()),
            |(origin, p, q)| black_box(cmp_dist(&origin, &p, &q)),
            batch_size,
        )
    });

    group.bench_function("apfp_dd_path", |bencher| {
        bencher.iter_batched(
            make_stage_setup(dd_cases.clone()),
            |(origin, p, q)| black_box(cmp_dist(&origin, &p, &q)),
            batch_size,
        )
    });

    group.bench_function("apfp_exact_path", |bencher| {
        bencher.iter_batched(
            make_stage_setup(exact_cases.clone()),
            |(origin, p, q)| black_box(cmp_dist(&origin, &p, &q)),
            batch_size,
        )
    });

    // === Reference: exact rational arithmetic ===

    group.bench_function("rational", |bencher| {
        bencher.iter_batched(
            make_random_setup(),
            |(origin, p, q)| black_box(cmp_dist_rational(&origin, &p, &q)),
            batch_size,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_cmp_dist);
criterion_main!(benches);

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 32) as f64) / (u32::MAX as f64);
    (val * 2000.0) - 1000.0
}
