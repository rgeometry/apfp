use std::cmp::Ordering;

/// Adds two nonoverlapping expansions and returns the summed expansion.
pub(crate) fn expansion_sum(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    if lhs.is_empty() {
        return rhs.to_vec();
    } else if rhs.is_empty() {
        return lhs.to_vec();
    }

    let mut result: Vec<f64> = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < lhs.len() && j < rhs.len() {
        let take_lhs = match compare_magnitude(lhs[i], rhs[j]) {
            Ordering::Less => true,
            Ordering::Equal => true,
            Ordering::Greater => false,
        };

        if take_lhs {
            result = grow_expansion_zeroelim(&result, lhs[i]);
            i += 1;
        } else {
            result = grow_expansion_zeroelim(&result, rhs[j]);
            j += 1;
        }
    }

    while i < lhs.len() {
        result = grow_expansion_zeroelim(&result, lhs[i]);
        i += 1;
    }

    while j < rhs.len() {
        result = grow_expansion_zeroelim(&result, rhs[j]);
        j += 1;
    }

    result
}

/// Returns true if the slice is sorted by nondecreasing magnitude.
pub(crate) fn is_sorted_by_magnitude(components: &[f64]) -> bool {
    components
        .windows(2)
        .all(|pair| compare_magnitude(pair[0], pair[1]) != Ordering::Greater)
}

/// Returns true if the slice is sorted and nonoverlapping according to Shewchuk.
pub(crate) fn is_nonoverlapping_sorted(components: &[f64]) -> bool {
    if components.len() <= 1 {
        return true;
    }

    for pair in components.windows(2) {
        let low = pair[0];
        let high = pair[1];

        if compare_magnitude(low, high) == Ordering::Greater {
            return false;
        }

        if high == 0.0 {
            if low != 0.0 {
                return false;
            }
            continue;
        }

        if low.abs() > ulp(high) {
            return false;
        }
    }

    true
}

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

pub(crate) fn compare_magnitude(a: f64, b: f64) -> Ordering {
    match a.abs().partial_cmp(&b.abs()) {
        Some(order) => order,
        None => Ordering::Equal,
    }
}

fn grow_expansion_zeroelim(expansion: &[f64], component: f64) -> Vec<f64> {
    debug_assert!(!component.is_nan(), "NaN components are not supported");

    let mut h = Vec::with_capacity(expansion.len() + 1);
    let mut q = component;

    for &enow in expansion {
        let (sum, err) = two_sum(q, enow);
        if err != 0.0 {
            h.push(err);
        }
        q = sum;
    }

    if q != 0.0 || h.is_empty() {
        h.push(q);
    }

    h
}

fn ulp(value: f64) -> f64 {
    if value == 0.0 {
        f64::MIN_POSITIVE
    } else {
        let abs = value.abs();
        let bits = abs.to_bits();
        if bits == f64::INFINITY.to_bits() {
            f64::INFINITY
        } else {
            let next = f64::from_bits(bits + 1);
            next - abs
        }
    }
}
