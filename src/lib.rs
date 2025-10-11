use std::cmp::Ordering;
use std::ops::{Add, AddAssign};

/// Adaptive precision floating-point value represented as a nonoverlapping
/// expansion of IEEE `f64` components stored in increasing magnitude order.
#[derive(Clone, Debug, PartialEq)]
pub struct Ap64 {
    components: Vec<f64>,
}

impl Ap64 {
    /// Returns the zero value.
    pub fn zero() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Creates an adaptive value from a single `f64`.
    pub fn from_f64(value: f64) -> Self {
        debug_assert!(!value.is_nan(), "NaN components are not supported");
        if value == 0.0 {
            Self::zero()
        } else {
            Self {
                components: vec![value],
            }
        }
    }

    /// Returns a simple floating-point approximation (sum of all components).
    pub fn approx(&self) -> f64 {
        self.components.iter().copied().sum()
    }

    /// Exposes the underlying expansion.
    pub fn components(&self) -> &[f64] {
        &self.components
    }

    /// Reports whether the value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.components.is_empty()
    }

    /// Adds another adaptive precision value, returning the sum.
    pub fn add_expansion(&self, rhs: &Self) -> Self {
        let components = expansion_sum(self.components(), rhs.components());
        let result = Self::from_components(components);
        debug_assert!(result.check_invariants().is_ok());
        result
    }

    /// Ensures the internal expansion satisfies required invariants.
    pub fn check_invariants(&self) -> Result<(), &'static str> {
        for &component in &self.components {
            if !component.is_finite() {
                return Err("Ap64 component must be finite");
            }
        }
        if !is_sorted_by_magnitude(&self.components) {
            return Err("Ap64 components must be sorted by increasing magnitude");
        }
        if !is_nonoverlapping_sorted(&self.components) {
            return Err("Ap64 components must be nonoverlapping");
        }
        Ok(())
    }

    fn from_components(mut components: Vec<f64>) -> Self {
        if components.len() == 1 && components[0] == 0.0 {
            components.clear();
        } else if components.len() > 1 {
            components.retain(|c| *c != 0.0);
        }
        let result = Self { components };
        debug_assert!(result.check_invariants().is_ok());
        result
    }
}

impl Default for Ap64 {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<f64> for Ap64 {
    fn from(value: f64) -> Self {
        Ap64::from_f64(value)
    }
}

impl<'a, 'b> Add<&'b Ap64> for &'a Ap64 {
    type Output = Ap64;

    fn add(self, rhs: &'b Ap64) -> Ap64 {
        self.add_expansion(rhs)
    }
}

impl Add for Ap64 {
    type Output = Ap64;

    fn add(self, rhs: Ap64) -> Ap64 {
        (&self).add(&rhs)
    }
}

impl Add<&Ap64> for Ap64 {
    type Output = Ap64;

    fn add(self, rhs: &Ap64) -> Ap64 {
        (&self).add(rhs)
    }
}

impl AddAssign for Ap64 {
    fn add_assign(&mut self, rhs: Self) {
        *self = (&*self).add(&rhs);
    }
}

impl AddAssign<&Ap64> for Ap64 {
    fn add_assign(&mut self, rhs: &Ap64) {
        *self = (&*self).add(rhs);
    }
}

fn expansion_sum(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
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

fn compare_magnitude(a: f64, b: f64) -> Ordering {
    match a.abs().partial_cmp(&b.abs()) {
        Some(order) => order,
        None => Ordering::Equal,
    }
}

fn is_sorted_by_magnitude(components: &[f64]) -> bool {
    components
        .windows(2)
        .all(|pair| compare_magnitude(pair[0], pair[1]) != Ordering::Greater)
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    let err = a_roundoff + b_roundoff;
    (sum, err)
}

#[allow(dead_code)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
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

fn is_nonoverlapping_sorted(components: &[f64]) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    #[should_panic(expected = "Ap64 components must be nonoverlapping")]
    fn overlapping_components_fail_invariants() {
        let invalid = Ap64 {
            components: vec![0.75, 1.0],
        };
        invalid.check_invariants().unwrap();
    }
}
