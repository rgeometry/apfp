use std::cmp::Ordering;

#[allow(dead_code)]
pub(crate) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    debug_assert!(
        a.abs() >= b.abs(),
        "FAST-TWO-SUM requires |a| >= |b| ({} vs {})",
        a,
        b
    );
    let sum = a + b;
    let b_virtual = sum - a;
    let err = b - b_virtual;
    (sum, err)
}

pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    let err = a_roundoff + b_roundoff;
    (sum, err)
}

pub(crate) fn two_product(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let err1 = product - (ahi * bhi);
    let err2 = err1 - (alo * bhi);
    let err3 = err2 - (ahi * blo);
    let err = (alo * blo) - err3;
    (product, err)
}

#[allow(dead_code)]
pub(crate) fn compare_magnitude(a: f64, b: f64) -> Ordering {
    match a.abs().partial_cmp(&b.abs()) {
        Some(order) => order,
        None => Ordering::Equal,
    }
}

fn split(value: f64) -> (f64, f64) {
    const SPLITTER: f64 = 134_217_729.0;
    let c = SPLITTER * value;
    let abig = c - value;
    let high = c - abig;
    let low = value - high;
    (high, low)
}
