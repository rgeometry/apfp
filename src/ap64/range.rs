#![allow(clippy::needless_lifetimes, dead_code)]

use crate::expansion::{two_product, two_sum};
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Range {
    pub(crate) center: f64,
    pub(crate) radius: f64,
}

impl Range {
    pub fn compare(self, rhs: Self) -> Ordering {
        let (self_lower, self_upper) = self.bounds();
        let (rhs_lower, rhs_upper) = rhs.bounds();
        if self_lower > rhs_upper {
            return Ordering::Greater;
        }
        if self_upper < rhs_lower {
            return Ordering::Less;
        }
        Ordering::Equal
    }

    pub fn bounds(self) -> (f64, f64) {
        let Range { center, radius } = self;
        (center - radius, center + radius)
    }

    pub(crate) fn new(center: f64, radius: f64) -> Self {
        Self {
            center,
            radius: radius.abs(),
        }
    }

    pub(crate) fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    pub(crate) fn from_components(components: &[f64]) -> Self {
        if components.is_empty() {
            return Self::zero();
        }
        let (center, radius) = approximate_with_error(components);
        Self::new(center, radius)
    }

    pub(crate) fn contains(&self, value: f64) -> bool {
        let lower = self.center - self.radius;
        let upper = self.center + self.radius;
        lower <= value && value <= upper
    }

    pub(crate) fn include(&mut self, other: &Self) {
        let required = (self.center - other.center).abs() + other.radius;
        let new_radius = self.radius.max(other.radius).max(required);
        self.center = other.center;
        self.radius = new_radius;
    }

    #[inline]
    fn add_range(&self, rhs: &Self) -> Self {
        let (center, err) = two_sum(self.center, rhs.center);
        let radius = self.radius + rhs.radius + err.abs();
        Self::new(center, radius)
    }

    #[inline]
    fn sub_range(&self, rhs: &Self) -> Self {
        let (center, err) = two_sum(self.center, -rhs.center);
        let radius = self.radius + rhs.radius + err.abs();
        Self::new(center, radius)
    }

    #[inline]
    fn mul_range(&self, rhs: &Self) -> Self {
        let (center, err) = two_product(self.center, rhs.center);
        let radius = self.center.abs() * rhs.radius
            + rhs.center.abs() * self.radius
            + self.radius * rhs.radius
            + err.abs();
        Self::new(center, radius)
    }
}

impl Add for Range {
    type Output = Range;

    fn add(self, rhs: Range) -> Range {
        self.add_range(&rhs)
    }
}

impl<'a> Add<&'a Range> for Range {
    type Output = Range;

    fn add(self, rhs: &'a Range) -> Range {
        self.add_range(rhs)
    }
}

impl<'a> Add<Range> for &'a Range {
    type Output = Range;

    fn add(self, rhs: Range) -> Range {
        self.add_range(&rhs)
    }
}

impl<'a, 'b> Add<&'b Range> for &'a Range {
    type Output = Range;

    fn add(self, rhs: &'b Range) -> Range {
        self.add_range(rhs)
    }
}

impl Sub for Range {
    type Output = Range;

    fn sub(self, rhs: Range) -> Range {
        self.sub_range(&rhs)
    }
}

impl<'a> Sub<&'a Range> for Range {
    type Output = Range;

    fn sub(self, rhs: &'a Range) -> Range {
        self.sub_range(rhs)
    }
}

impl<'a> Sub<Range> for &'a Range {
    type Output = Range;

    fn sub(self, rhs: Range) -> Range {
        self.sub_range(&rhs)
    }
}

impl<'a, 'b> Sub<&'b Range> for &'a Range {
    type Output = Range;

    fn sub(self, rhs: &'b Range) -> Range {
        self.sub_range(rhs)
    }
}

impl Mul for Range {
    type Output = Range;

    fn mul(self, rhs: Range) -> Range {
        self.mul_range(&rhs)
    }
}

impl<'a> Mul<&'a Range> for Range {
    type Output = Range;

    fn mul(self, rhs: &'a Range) -> Range {
        self.mul_range(rhs)
    }
}

impl<'a> Mul<Range> for &'a Range {
    type Output = Range;

    fn mul(self, rhs: Range) -> Range {
        self.mul_range(&rhs)
    }
}

impl<'a, 'b> Mul<&'b Range> for &'a Range {
    type Output = Range;

    fn mul(self, rhs: &'b Range) -> Range {
        self.mul_range(rhs)
    }
}

impl Neg for Range {
    type Output = Range;

    fn neg(self) -> Range {
        Range::new(-self.center, self.radius)
    }
}

impl<'a> Neg for &'a Range {
    type Output = Range;

    fn neg(self) -> Range {
        Range::new(-self.center, self.radius)
    }
}

fn approximate_with_error(components: &[f64]) -> (f64, f64) {
    let mut approx = 0.0;
    let mut error = 0.0;
    for &component in components.iter().rev() {
        let (sum, err) = two_sum(approx, component);
        approx = sum;
        error += err.abs();
    }
    (approx, error)
}
