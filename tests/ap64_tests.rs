use apfp::Ap64;
use ntest::timeout;
use num_rational::BigRational;
use num_traits::Zero;
use quickcheck::{QuickCheck, TestResult};

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
fn simple_addition() {
    let a = Ap64::from(1.5);
    let b = Ap64::from(2.25);
    let sum = &a + &b;
    assert!(!sum.is_zero());
    assert_eq!(sum.approx(), 3.75);
    sum.check_invariants().unwrap();
    assert_eq!(sum.components().len(), 1);
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
fn simple_multiplication() {
    let a = Ap64::from(1.5);
    let b = Ap64::from(2.0);
    let prod = &a * &b;
    assert_eq!(prod.approx(), 3.0);
    prod.check_invariants().unwrap();
    assert_eq!(prod.components().len(), 1);
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

fn property_ap64_add_matches_rational(x: f64, y: f64) -> TestResult {
    const MAG_LIMIT: f64 = 1.0e150;
    if !x.is_finite() || !y.is_finite() || !(x + y).is_finite() {
        return TestResult::discard();
    }
    if x.abs() > MAG_LIMIT || y.abs() > MAG_LIMIT {
        return TestResult::discard();
    }
    let x_rational = match f64_to_rational(x) {
        Some(r) => r,
        None => return TestResult::discard(),
    };
    let y_rational = match f64_to_rational(y) {
        Some(r) => r,
        None => return TestResult::discard(),
    };

    let lhs = &Ap64::from(x) + &Ap64::from(y);
    if let Err(err) = lhs.check_invariants() {
        return TestResult::error(err);
    }
    let lhs_rational = match ap64_to_rational(&lhs) {
        Some(r) => r,
        None => return TestResult::discard(),
    };

    let rhs_rational = x_rational + y_rational;
    TestResult::from_bool(lhs_rational == rhs_rational)
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_add_matches_rational() {
    fn property(x: f64, y: f64) -> TestResult {
        property_ap64_add_matches_rational(x, y)
    }
    QuickCheck::new()
        .tests(1_000)
        .max_tests(50_000)
        .quickcheck(property as fn(f64, f64) -> TestResult);
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

fn property_ap64_mul_matches_rational(x: f64, y: f64) -> TestResult {
    const MAG_LIMIT: f64 = 1.0e150;
    if !x.is_finite() || !y.is_finite() || !(x * y).is_finite() {
        return TestResult::discard();
    }
    if x.abs() > MAG_LIMIT || y.abs() > MAG_LIMIT {
        return TestResult::discard();
    }
    let x_rational = match f64_to_rational(x) {
        Some(r) => r,
        None => return TestResult::discard(),
    };
    let y_rational = match f64_to_rational(y) {
        Some(r) => r,
        None => return TestResult::discard(),
    };

    let lhs = &Ap64::from(x) * &Ap64::from(y);
    if let Err(err) = lhs.check_invariants() {
        return TestResult::error(err);
    }
    let lhs_rational = match ap64_to_rational(&lhs) {
        Some(r) => r,
        None => return TestResult::discard(),
    };

    let rhs_rational = x_rational * y_rational;
    TestResult::from_bool(lhs_rational == rhs_rational)
}

#[test]
#[timeout(5000)]
fn quickcheck_ap64_mul_matches_rational() {
    fn property(x: f64, y: f64) -> TestResult {
        property_ap64_mul_matches_rational(x, y)
    }

    QuickCheck::new()
        .tests(1_000)
        .max_tests(50_000)
        .quickcheck(property as fn(f64, f64) -> TestResult);
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
