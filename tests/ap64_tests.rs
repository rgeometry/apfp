use apfp::Ap64;
use num_rational::BigRational;
use num_traits::Zero;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;

#[test]
fn zero_behaves() {
    let z = Ap64::zero();
    assert!(z.is_zero());
    assert_eq!(z.approx(), 0.0);
    assert!(z.components().is_empty());
    z.check_invariants().unwrap();
}

#[test]
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
fn cancellation_is_preserved() {
    let a = Ap64::from(1.0e16);
    let b = Ap64::from(1.0);
    let c = Ap64::from(-1.0e16);

    let sum = (&a + &b) + &c;
    sum.check_invariants().unwrap();
    assert_eq!(sum.approx(), 1.0);
}

#[test]
fn addition_produces_expected_components() {
    let a = Ap64::from(1.0e16);
    let b = Ap64::from(-1.0e16);
    let c = Ap64::from(2.5);
    let d = Ap64::from(-1.25);

    let sum = (((&a + &b) + &c) + &d).approx();
    assert!((sum - 1.25).abs() < f64::EPSILON);
}

#[test]
fn add_assign_works() {
    let mut acc = Ap64::from(0.125);
    acc += Ap64::from(0.125);
    acc += &Ap64::from(0.25);
    assert_eq!(acc.approx(), 0.5);
    acc.check_invariants().unwrap();
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

#[quickcheck]
fn quickcheck_ap64_add_matches_rational(x: f64, y: f64) -> TestResult {
    property_ap64_add_matches_rational(x, y)
}

#[test]
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
