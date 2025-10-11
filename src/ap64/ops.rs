use crate::expansion::{expansion_product, expansion_sum};
use std::ops::{Add, AddAssign, Mul, MulAssign};

use super::Ap64;

impl Ap64 {
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
    use super::Ap64;

    #[test]
    #[should_panic(expected = "Ap64 components must be nonoverlapping")]
    fn overlapping_components_fail_invariants() {
        let invalid = Ap64::from_raw_components(vec![0.75, 1.0]);
        invalid.check_invariants().unwrap();
    }
}
