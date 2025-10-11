use crate::expansion::{
    expansion_product, expansion_sum, is_nonoverlapping_sorted, is_sorted_by_magnitude,
};
use std::ops::{Add, AddAssign, Mul, MulAssign};

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

    /// Multiplies another adaptive precision value, returning the product.
    pub fn mul_expansion(&self, rhs: &Self) -> Self {
        let components = expansion_product(self.components(), rhs.components());
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

    #[cfg(test)]
    fn from_raw_components(components: Vec<f64>) -> Self {
        Self { components }
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

impl<'a, 'b> Mul<&'b Ap64> for &'a Ap64 {
    type Output = Ap64;

    fn mul(self, rhs: &'b Ap64) -> Ap64 {
        self.mul_expansion(rhs)
    }
}

impl Mul for Ap64 {
    type Output = Ap64;

    fn mul(self, rhs: Ap64) -> Ap64 {
        (&self).mul(&rhs)
    }
}

impl Mul<&Ap64> for Ap64 {
    type Output = Ap64;

    fn mul(self, rhs: &Ap64) -> Ap64 {
        (&self).mul(rhs)
    }
}

impl MulAssign for Ap64 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = (&*self).mul(&rhs);
    }
}

impl MulAssign<&Ap64> for Ap64 {
    fn mul_assign(&mut self, rhs: &Ap64) {
        *self = (&*self).mul(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "Ap64 components must be nonoverlapping")]
    fn overlapping_components_fail_invariants() {
        let invalid = Ap64::from_raw_components(vec![0.75, 1.0]);
        invalid.check_invariants().unwrap();
    }
}
