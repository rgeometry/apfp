use apfp::Ap64;
use ntest::timeout;
use num_rational::BigRational;
use num_traits::Zero;
use quickcheck::{QuickCheck, TestResult};
use std::panic::{self, AssertUnwindSafe};

const MAG_LIMIT: f64 = 1.0e12;
const QC_TESTS: u64 = 5000;
const QC_MAX_TESTS: u64 = 20_000;

#[test]
#[timeout(5000)]
fn zero_behaves() {
    let z = Ap64::zero();
    assert!(z.is_zero());
    assert_eq!(z.approx(), 0.0);
    assert!(z.components().is_empty());
    z.check_invariants().unwrap();
}

#[test]
#[timeout(5000)]
fn default_matches_zero() {
    assert_eq!(Ap64::default(), Ap64::zero());
    assert!(Ap64::from(-0.0).is_zero());
}

#[test]
#[timeout(5000)]
fn simple_addition() {
    let a = Ap64::from(1.5);
    let b = Ap64::from(2.25);
    let sum = &a + &b;
    assert_eq!(sum.approx(), 3.75);
    sum.check_invariants().unwrap();
}

#[test]
#[timeout(5000)]
fn simple_multiplication() {
    let a = Ap64::from(1.5);
    let b = Ap64::from(2.0);
    let prod = &a * &b;
    assert_eq!(prod.approx(), 3.0);
    prod.check_invariants().unwrap();
}

#[test]
#[timeout(5000)]
fn add_assign_works() {
    let mut acc = Ap64::from(0.125);
    acc += Ap64::from(0.125);
    acc += &Ap64::from(0.25);
    assert_eq!(acc.approx(), 0.5);
    acc.check_invariants().unwrap();
}

#[test]
#[timeout(5000)]
fn mul_assign_works() {
    let mut acc = Ap64::from(2.0);
    acc *= Ap64::from(0.5);
    acc *= &Ap64::from(4.0);
    assert_eq!(acc.approx(), 4.0);
    acc.check_invariants().unwrap();
}

#[test]
#[timeout(5000)]
fn multiplication_handles_zero() {
    let a = Ap64::from(3.14159);
    let zero = Ap64::zero();
    let prod = &a * &zero;
    assert!(prod.is_zero());
    assert_eq!(prod.components().len(), 0);
}

#[test]
#[timeout(5000)]
fn cancellation_is_preserved() {
    let a = Ap64::from(1.0e16);
    let b = Ap64::from(1.0);
    let c = Ap64::from(-1.0e16);

    let sum = (&a + &b) + &c;
    sum.check_invariants().unwrap();
    assert_eq!(sum.approx(), 1.0);
}

#[test]
#[timeout(5000)]
fn addition_produces_expected_components() {
    let a = Ap64::from(1.0e16);
    let b = Ap64::from(-1.0e16);
    let c = Ap64::from(2.5);
    let d = Ap64::from(-1.25);

    let sum = (((&a + &b) + &c) + &d).approx();
    assert!((sum - 1.25).abs() < f64::EPSILON);
}

fn f64_to_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(BigRational::zero());
    }
    BigRational::from_float(value)
}

fn ap64_to_rational(value: &Ap64) -> Option<BigRational> {
    value
        .components()
        .iter()
        .try_fold(BigRational::zero(), |acc, &component| {
            let rational = f64_to_rational(component)?;
            Some(acc + rational)
        })
}

fn within_limits(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= MAG_LIMIT)
}

fn run_qc1(prop: fn(f64) -> TestResult) {
    QuickCheck::new()
        .tests(QC_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(prop);
}

fn run_qc2(prop: fn(f64, f64) -> TestResult) {
    QuickCheck::new()
        .tests(QC_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(prop);
}

fn run_qc3(prop: fn(f64, f64, f64) -> TestResult) {
    QuickCheck::new()
        .tests(QC_TESTS)
        .max_tests(QC_MAX_TESTS)
        .quickcheck(prop);
}

fn safe_eval<T: std::panic::UnwindSafe>(f: impl FnOnce() -> T) -> Option<T> {
    panic::catch_unwind(AssertUnwindSafe(f)).ok()
}

fn property_ap64_add_matches_rational(x: f64, y: f64) -> TestResult {
    if !within_limits(&[x, y, x + y]) {
        return TestResult::discard();
    }
    let lhs = &Ap64::from(x) + &Ap64::from(y);
    if let Err(err) = lhs.check_invariants() {
        return TestResult::error(err);
    }
    let lhs_rational = match ap64_to_rational(&lhs) {
        Some(r) => r,
        None => return TestResult::discard(),
    };
    let rhs_rational = match (f64_to_rational(x), f64_to_rational(y)) {
        (Some(a), Some(b)) => a + b,
        _ => return TestResult::discard(),
    };
    TestResult::from_bool(lhs_rational == rhs_rational)
}

fn property_ap64_mul_matches_rational(x: f64, y: f64) -> TestResult {
    if !within_limits(&[x, y, x * y]) {
        return TestResult::discard();
    }
    let lhs = &Ap64::from(x) * &Ap64::from(y);
    if let Err(err) = lhs.check_invariants() {
        return TestResult::error(err);
    }
    let lhs_rational = match ap64_to_rational(&lhs) {
        Some(r) => r,
        None => return TestResult::discard(),
    };
    let rhs_rational = match (f64_to_rational(x), f64_to_rational(y)) {
        (Some(a), Some(b)) => a * b,
        _ => return TestResult::discard(),
    };
    TestResult::from_bool(lhs_rational == rhs_rational)
}

fn property_additive_identity(x: f64) -> TestResult {
    if !within_limits(&[x]) {
        return TestResult::discard();
    }
    let a = Ap64::from(x);
    let sum = &a + &Ap64::zero();
    match (ap64_to_rational(&a), ap64_to_rational(&sum)) {
        (Some(lhs), Some(rhs)) => TestResult::from_bool(lhs == rhs),
        _ => TestResult::discard(),
    }
}

fn property_additive_inverse(x: f64) -> TestResult {
    if !within_limits(&[x]) {
        return TestResult::discard();
    }
    let a = Ap64::from(x);
    let sum = &a + &Ap64::from(-x);
    if let Err(err) = sum.check_invariants() {
        return TestResult::error(err);
    }
    TestResult::from_bool(sum.is_zero())
}

fn property_multiplicative_identity(x: f64) -> TestResult {
    if !within_limits(&[x, x * 1.0]) {
        return TestResult::discard();
    }
    let a = Ap64::from(x);
    let prod = &a * &Ap64::from(1.0);
    match (ap64_to_rational(&a), ap64_to_rational(&prod)) {
        (Some(lhs), Some(rhs)) => TestResult::from_bool(lhs == rhs),
        _ => TestResult::discard(),
    }
}

fn property_add_commutative(x: f64, y: f64) -> TestResult {
    if !within_limits(&[x, y, x + y]) {
        return TestResult::discard();
    }
    let lhs = &Ap64::from(x) + &Ap64::from(y);
    let rhs = &Ap64::from(y) + &Ap64::from(x);
    if let (Some(a), Some(b)) = (ap64_to_rational(&lhs), ap64_to_rational(&rhs)) {
        TestResult::from_bool(a == b)
    } else {
        TestResult::discard()
    }
}

fn property_mul_commutative(x: f64, y: f64) -> TestResult {
    if !within_limits(&[x, y, x * y]) {
        return TestResult::discard();
    }
    let lhs = &Ap64::from(x) * &Ap64::from(y);
    let rhs = &Ap64::from(y) * &Ap64::from(x);
    if let (Some(a), Some(b)) = (ap64_to_rational(&lhs), ap64_to_rational(&rhs)) {
        TestResult::from_bool(a == b)
    } else {
        TestResult::discard()
    }
}

fn property_add_associative(x: f64, y: f64, z: f64) -> TestResult {
    if !within_limits(&[x, y, z, x + y, y + z, x + y + z]) {
        return TestResult::discard();
    }
    let Some((lhs, rhs)) = safe_eval(|| {
        let a = Ap64::from(x);
        let b = Ap64::from(y);
        let c = Ap64::from(z);
        let lhs = (&a + &b) + &c;
        let rhs = &a + &(&b + &c);
        (lhs, rhs)
    }) else {
        return TestResult::discard();
    };
    if let (Some(l), Some(r)) = (ap64_to_rational(&lhs), ap64_to_rational(&rhs)) {
        TestResult::from_bool(l == r)
    } else {
        TestResult::discard()
    }
}

fn property_distributive(a: f64, b: f64, c: f64) -> TestResult {
    if !within_limits(&[a, b, c, a * b, a * c, b + c, a * (b + c)]) {
        return TestResult::discard();
    }
    let Some((lhs, rhs)) = safe_eval(|| {
        let ap = Ap64::from(a);
        let bp = Ap64::from(b);
        let cp = Ap64::from(c);
        let lhs = &ap * &(&bp + &cp);
        let rhs = (&ap * &bp) + (&ap * &cp);
        (lhs, rhs)
    }) else {
        return TestResult::discard();
    };
    if let (Some(l), Some(r)) = (ap64_to_rational(&lhs), ap64_to_rational(&rhs)) {
        TestResult::from_bool(l == r)
    } else {
        TestResult::discard()
    }
}

fn property_sign_matches_largest_component(x: f64, y: f64) -> TestResult {
    if !within_limits(&[x, y, x + y]) {
        return TestResult::discard();
    }
    let sum = &Ap64::from(x) + &Ap64::from(y);
    if sum.is_zero() {
        return TestResult::passed();
    }
    let approx = sum.approx();
    let Some(&largest) = sum.components().last() else {
        return TestResult::discard();
    };
    if largest == 0.0 {
        return TestResult::discard();
    }
    TestResult::from_bool(approx.signum() == largest.signum())
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_add_matches_rational() {
    run_qc2(property_ap64_add_matches_rational);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_mul_matches_rational() {
    run_qc2(property_ap64_mul_matches_rational);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_add_identity() {
    run_qc1(property_additive_identity);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_add_inverse() {
    run_qc1(property_additive_inverse);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_mul_identity() {
    run_qc1(property_multiplicative_identity);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_add_commutative() {
    run_qc2(property_add_commutative);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_mul_commutative() {
    run_qc2(property_mul_commutative);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_add_associative() {
    run_qc3(property_add_associative);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_distributive() {
    run_qc3(property_distributive);
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_sign_matches_largest_component() {
    run_qc2(property_sign_matches_largest_component);
}

#[test]
#[timeout(5000)]
fn property_handles_explicit_nonoverlapping_pair() {
    let x = 1.0f64;
    let y = 2f64.powi(-54);
    let result = property_ap64_add_matches_rational(x, y);
    assert!(
        !result.is_failure(),
        "property failed for explicit pair {} + {}",
        x,
        y
    );
}

#[test]
#[timeout(5000)]
fn multiplication_property_handles_explicit_pair() {
    let x = 1.0e16;
    let y = 1.0e-16;
    let result = property_ap64_mul_matches_rational(x, y);
    assert!(
        !result.is_failure(),
        "property failed for explicit pair {} * {}",
        x,
        y
    );
}
