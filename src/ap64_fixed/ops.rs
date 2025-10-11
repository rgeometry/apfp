use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Ap64Fixed;

impl<const N: usize> Add for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(&rhs)
            .expect("Ap64Fixed addition overflow: increase capacity")
    }
}

impl<'b, const N: usize> Add<&'b Ap64Fixed<N>> for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn add(self, rhs: &'b Ap64Fixed<N>) -> Self::Output {
        self.try_add(rhs)
            .expect("Ap64Fixed addition overflow: increase capacity")
    }
}

impl<'b, const N: usize> Add<&'b Ap64Fixed<N>> for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn add(self, rhs: &'b Ap64Fixed<N>) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<const N: usize> Add<Ap64Fixed<N>> for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn add(self, rhs: Ap64Fixed<N>) -> Self::Output {
        self.add(&rhs)
    }
}

impl<const N: usize> AddAssign<&Ap64Fixed<N>> for Ap64Fixed<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Ap64Fixed<N>) {
        let result = self
            .try_add(rhs)
            .expect("Ap64Fixed addition overflow: increase capacity");
        *self = result;
    }
}

impl<const N: usize> AddAssign for Ap64Fixed<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<const N: usize> Mul for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.try_mul(&rhs)
            .expect("Ap64Fixed multiplication overflow: increase capacity")
    }
}

impl<'b, const N: usize> Mul<&'b Ap64Fixed<N>> for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn mul(self, rhs: &'b Ap64Fixed<N>) -> Self::Output {
        self.try_mul(rhs)
            .expect("Ap64Fixed multiplication overflow: increase capacity")
    }
}

impl<'b, const N: usize> Mul<&'b Ap64Fixed<N>> for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn mul(self, rhs: &'b Ap64Fixed<N>) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl<const N: usize> Mul<Ap64Fixed<N>> for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn mul(self, rhs: Ap64Fixed<N>) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<const N: usize> MulAssign<&Ap64Fixed<N>> for Ap64Fixed<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Ap64Fixed<N>) {
        let result = self
            .try_mul(rhs)
            .expect("Ap64Fixed multiplication overflow: increase capacity");
        *self = result;
    }
}

impl<const N: usize> MulAssign for Ap64Fixed<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.mul_assign(&rhs);
    }
}

impl<const N: usize> Neg for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<const N: usize> Neg for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<const N: usize> Sub for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(&rhs)
            .expect("Ap64Fixed subtraction overflow: increase capacity")
    }
}

impl<'b, const N: usize> Sub<&'b Ap64Fixed<N>> for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn sub(self, rhs: &'b Ap64Fixed<N>) -> Self::Output {
        self.try_sub(rhs)
            .expect("Ap64Fixed subtraction overflow: increase capacity")
    }
}

impl<'b, const N: usize> Sub<&'b Ap64Fixed<N>> for Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn sub(self, rhs: &'b Ap64Fixed<N>) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl<const N: usize> Sub<Ap64Fixed<N>> for &Ap64Fixed<N> {
    type Output = Ap64Fixed<N>;

    #[inline(always)]
    fn sub(self, rhs: Ap64Fixed<N>) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<const N: usize> SubAssign<&Ap64Fixed<N>> for Ap64Fixed<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Ap64Fixed<N>) {
        let result = self
            .try_sub(rhs)
            .expect("Ap64Fixed subtraction overflow: increase capacity");
        *self = result;
    }
}

impl<const N: usize> SubAssign for Ap64Fixed<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}
