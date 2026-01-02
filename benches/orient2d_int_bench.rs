//! Benchmarks for integer orient2d implementations.
//!
//! Measures per-call performance for:
//! - Naive i32 (baseline, may overflow)
//! - Hand-written i64 (exact, uses u128 intermediate)
//! - apfp i32/i64 orient2d (exact, no overflow)
//! - BigInt (exact but slow baseline)

use apfp::geometry::i32 as i32_types;
use apfp::geometry::i64 as i64_types;
use apfp::int_signum;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use num_bigint::{BigInt, Sign};
use std::cmp::Ordering;
use std::hint::black_box;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1;

/// Simple i32 orient2d - may overflow for large inputs.
fn orient2d_naive_i32(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
    let adx = ax as i64 - cx as i64;
    let ady = ay as i64 - cy as i64;
    let bdx = bx as i64 - cx as i64;
    let bdy = by as i64 - cy as i64;

    let det = adx * bdy - ady * bdx;
    if det > 0 {
        1
    } else if det < 0 {
        -1
    } else {
        0
    }
}

/// Hand-written exact orient2d for i64 using u128 intermediate.
/// Works by computing absolute differences with sign tracking, then comparing
/// products in u128 space. Never overflows for any i64 inputs.
fn orient2d_handwritten_i64(ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64) -> Ordering {
    // Return the absolute difference along with its sign.
    // diff(0, 10) => (10, true)  // positive: b > a
    // diff(10, 0) => (10, false) // negative: a > b
    #[inline(always)]
    fn diff(a: i64, b: i64) -> (u128, bool) {
        if b > a {
            ((b.wrapping_sub(a)) as u64 as u128, true)
        } else {
            ((a.wrapping_sub(b)) as u64 as u128, false)
        }
    }

    // Compute (b - a) relative to c: ux = bx - cx, uy = by - cy
    // and (c - a) relative to c: vx = cx - cx = 0... wait, that's not right.
    // Actually for orient2d we want:
    //   det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    // Let's rename: ux = ax - cx, vy = by - cy, uy = ay - cy, vx = bx - cx
    let (ux, ux_neg) = diff(cx, ax); // ax - cx: positive if ax > cx
    let (vy, vy_neg) = diff(cy, by); // by - cy: positive if by > cy
    let ux_vy_neg = ux_neg != vy_neg && ux != 0 && vy != 0;

    let (uy, uy_neg) = diff(cy, ay); // ay - cy: positive if ay > cy
    let (vx, vx_neg) = diff(cx, bx); // bx - cx: positive if bx > cx
    let uy_vx_neg = uy_neg != vx_neg && uy != 0 && vx != 0;

    // det = ux * vy - uy * vx
    // We need to compare ux * vy against uy * vx, accounting for signs
    match (ux_vy_neg, uy_vx_neg) {
        // ux*vy negative, uy*vx non-negative => det < 0
        (true, false) => Ordering::Less,
        // ux*vy non-negative, uy*vx negative => det > 0
        (false, true) => Ordering::Greater,
        // Both negative: det = -|ux*vy| - (-|uy*vx|) = |uy*vx| - |ux*vy|
        (true, true) => (uy * vx).cmp(&(ux * vy)),
        // Both non-negative: det = |ux*vy| - |uy*vx|
        (false, false) => (ux * vy).cmp(&(uy * vx)),
    }
}

/// Exact orient2d using BigInt (slow).
fn orient2d_bigint_i32(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
    let ax = BigInt::from(ax);
    let ay = BigInt::from(ay);
    let bx = BigInt::from(bx);
    let by = BigInt::from(by);
    let cx = BigInt::from(cx);
    let cy = BigInt::from(cy);

    let adx = &ax - &cx;
    let ady = &ay - &cy;
    let bdx = &bx - &cx;
    let bdy = &by - &cy;

    let det = &adx * &bdy - &ady * &bdx;
    match det.sign() {
        Sign::Plus => 1,
        Sign::Minus => -1,
        Sign::NoSign => 0,
    }
}

/// Exact orient2d using BigInt for i64 inputs.
fn orient2d_bigint_i64(ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64) -> i32 {
    let ax = BigInt::from(ax);
    let ay = BigInt::from(ay);
    let bx = BigInt::from(bx);
    let by = BigInt::from(by);
    let cx = BigInt::from(cx);
    let cy = BigInt::from(cy);

    let adx = &ax - &cx;
    let ady = &ay - &cy;
    let bdx = &bx - &cx;
    let bdy = &by - &cy;

    let det = &adx * &bdy - &ady * &bdx;
    match det.sign() {
        Sign::Plus => 1,
        Sign::Minus => -1,
        Sign::NoSign => 0,
    }
}

fn bench_orient2d_int(c: &mut Criterion) {
    let mut group = c.benchmark_group("orient2d_int");
    let batch_size = BatchSize::SmallInput;

    // Setup function for random i32 coordinates
    let make_i32_setup = || {
        let mut state = 0x1234_5678_9abc_def0u64;
        move || {
            let ax = lcg_i32(&mut state);
            let ay = lcg_i32(&mut state);
            let bx = lcg_i32(&mut state);
            let by = lcg_i32(&mut state);
            let cx = lcg_i32(&mut state);
            let cy = lcg_i32(&mut state);
            (ax, ay, bx, by, cx, cy)
        }
    };

    // Setup function for random i64 coordinates (limited to avoid BigInt overhead)
    let make_i64_setup = || {
        let mut state = 0x1234_5678_9abc_def0u64;
        move || {
            let ax = lcg_i64(&mut state, 1_000_000_000);
            let ay = lcg_i64(&mut state, 1_000_000_000);
            let bx = lcg_i64(&mut state, 1_000_000_000);
            let by = lcg_i64(&mut state, 1_000_000_000);
            let cx = lcg_i64(&mut state, 1_000_000_000);
            let cy = lcg_i64(&mut state, 1_000_000_000);
            (ax, ay, bx, by, cx, cy)
        }
    };

    // Naive i32 (uses i64 internally, may overflow for extreme inputs)
    group.bench_function("naive_i32", |bencher| {
        bencher.iter_batched(
            make_i32_setup(),
            |(ax, ay, bx, by, cx, cy)| black_box(orient2d_naive_i32(ax, ay, bx, by, cx, cy)),
            batch_size,
        )
    });

    // apfp i32 (uses int_signum! macro)
    group.bench_function("apfp_i32", |bencher| {
        bencher.iter_batched(
            || {
                let mut state = 0x1234_5678_9abc_def0u64;
                let ax = lcg_i32(&mut state);
                let ay = lcg_i32(&mut state);
                let bx = lcg_i32(&mut state);
                let by = lcg_i32(&mut state);
                let cx = lcg_i32(&mut state);
                let cy = lcg_i32(&mut state);
                (ax, ay, bx, by, cx, cy)
            },
            |(ax, ay, bx, by, cx, cy)| {
                black_box(int_signum!((ax - cx) * (by - cy) - (ay - cy) * (bx - cx)))
            },
            batch_size,
        )
    });

    // apfp i32 with Coord (struct-based API)
    group.bench_function("apfp_i32_coord", |bencher| {
        bencher.iter_batched(
            || {
                let mut state = 0x1234_5678_9abc_def0u64;
                let a = i32_types::Coord::new(lcg_i32(&mut state), lcg_i32(&mut state));
                let b = i32_types::Coord::new(lcg_i32(&mut state), lcg_i32(&mut state));
                let c = i32_types::Coord::new(lcg_i32(&mut state), lcg_i32(&mut state));
                (a, b, c)
            },
            |(a, b, c)| black_box(i32_types::orient2d(&a, &b, &c)),
            batch_size,
        )
    });

    // BigInt i32 (exact but slow)
    group.bench_function("bigint_i32", |bencher| {
        bencher.iter_batched(
            make_i32_setup(),
            |(ax, ay, bx, by, cx, cy)| black_box(orient2d_bigint_i32(ax, ay, bx, by, cx, cy)),
            batch_size,
        )
    });

    // Hand-written i64 (exact, uses u128 intermediate)
    group.bench_function("handwritten_i64", |bencher| {
        bencher.iter_batched(
            make_i64_setup(),
            |(ax, ay, bx, by, cx, cy)| black_box(orient2d_handwritten_i64(ax, ay, bx, by, cx, cy)),
            batch_size,
        )
    });

    // apfp i64 (uses int_signum! macro with BigInt<4> internally)
    group.bench_function("apfp_i64", |bencher| {
        bencher.iter_batched(
            || {
                let mut state = 0x1234_5678_9abc_def0u64;
                let ax = lcg_i64(&mut state, 1_000_000_000);
                let ay = lcg_i64(&mut state, 1_000_000_000);
                let bx = lcg_i64(&mut state, 1_000_000_000);
                let by = lcg_i64(&mut state, 1_000_000_000);
                let cx = lcg_i64(&mut state, 1_000_000_000);
                let cy = lcg_i64(&mut state, 1_000_000_000);
                (ax, ay, bx, by, cx, cy)
            },
            |(ax, ay, bx, by, cx, cy)| {
                black_box(int_signum!((ax - cx) * (by - cy) - (ay - cy) * (bx - cx)))
            },
            batch_size,
        )
    });

    // apfp i64 with Coord
    group.bench_function("apfp_i64_coord", |bencher| {
        bencher.iter_batched(
            || {
                let mut state = 0x1234_5678_9abc_def0u64;
                let a = i64_types::Coord::new(
                    lcg_i64(&mut state, 1_000_000_000),
                    lcg_i64(&mut state, 1_000_000_000),
                );
                let b = i64_types::Coord::new(
                    lcg_i64(&mut state, 1_000_000_000),
                    lcg_i64(&mut state, 1_000_000_000),
                );
                let c = i64_types::Coord::new(
                    lcg_i64(&mut state, 1_000_000_000),
                    lcg_i64(&mut state, 1_000_000_000),
                );
                (a, b, c)
            },
            |(a, b, c)| black_box(i64_types::orient2d(&a, &b, &c)),
            batch_size,
        )
    });

    // BigInt i64
    group.bench_function("bigint_i64", |bencher| {
        bencher.iter_batched(
            make_i64_setup(),
            |(ax, ay, bx, by, cx, cy)| black_box(orient2d_bigint_i64(ax, ay, bx, by, cx, cy)),
            batch_size,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_orient2d_int);
criterion_main!(benches);

fn lcg_i32(state: &mut u64) -> i32 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    ((*state >> 33) as i32).wrapping_sub(i32::MAX / 2)
}

fn lcg_i64(state: &mut u64, limit: i64) -> i64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let val = ((*state >> 1) as i64).wrapping_rem(limit * 2);
    val - limit
}
