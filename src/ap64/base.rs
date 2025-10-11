#[cfg(feature = "short-circuit")]
use super::range::Range;
use crate::expansion::{is_nonoverlapping_sorted, is_sorted_by_magnitude};
use std::cmp::Ordering;

/// Adaptive precision floating-point value represented as a nonoverlapping
/// expansion of IEEE `f64` components stored in increasing magnitude order.
#[derive(Clone, Debug, PartialEq)]
pub struct Ap64 {
    pub(crate) components: Vec<f64>,
    #[cfg(feature = "short-circuit")]
    pub(crate) range: Range,
}

impl Ap64 {
    /// Returns the zero value.
    pub fn zero() -> Self {
        Self {
            components: Vec::new(),
            #[cfg(feature = "short-circuit")]
            range: Range::zero(),
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
                #[cfg(feature = "short-circuit")]
                range: Range::new(value, 0.0),
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

    /// Returns the stored uncertainty interval as `(center, radius)`.
    #[cfg(feature = "short-circuit")]
    pub fn interval(&self) -> (f64, f64) {
        (self.range.center, self.range.radius)
    }

    /// Returns the lower and upper bounds of the stored uncertainty interval.
    #[cfg(feature = "short-circuit")]
    pub fn bounds(&self) -> (f64, f64) {
        let (center, radius) = self.interval();
        (center - radius, center + radius)
    }

    /// Reports whether the value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.components.is_empty()
    }

    /// Compares this value with another adaptive value.
    pub fn compare(&self, rhs: &Self) -> Ordering {
        #[cfg(feature = "short-circuit")]
        {
            let (self_lower, self_upper) = self.bounds();
            let (rhs_lower, rhs_upper) = rhs.bounds();
            if self_lower > rhs_upper {
                return Ordering::Greater;
            }
            if self_upper < rhs_lower {
                return Ordering::Less;
            }
        }

        if self.is_zero() && rhs.is_zero() {
            return Ordering::Equal;
        }

        let mut i = self.components.len();
        let mut j = rhs.components.len();

        loop {
            while i > 0 && self.components[i - 1] == 0.0 {
                i -= 1;
            }
            while j > 0 && rhs.components[j - 1] == 0.0 {
                j -= 1;
            }

            if i == 0 && j == 0 {
                return Ordering::Equal;
            }

            if i == 0 {
                let b = rhs.components[j - 1];
                if b > 0.0 {
                    return Ordering::Less;
                }
                if b < 0.0 {
                    return Ordering::Greater;
                }
                j -= 1;
                continue;
            }

            if j == 0 {
                let a = self.components[i - 1];
                if a > 0.0 {
                    return Ordering::Greater;
                }
                if a < 0.0 {
                    return Ordering::Less;
                }
                i -= 1;
                continue;
            }

            let a = self.components[i - 1];
            let b = rhs.components[j - 1];
            if a > b {
                return Ordering::Greater;
            }
            if a < b {
                return Ordering::Less;
            }
            i -= 1;
            j -= 1;
        }
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
        #[cfg(feature = "short-circuit")]
        {
            let range = Range::from_components(&self.components);
            debug_assert!(
                (self.range.center - range.center).abs()
                    <= range.radius + f64::EPSILON * range.center.abs(),
                "stored interval center must match recomputed center"
            );
            debug_assert!(
                self.range.radius + f64::EPSILON * range.radius.abs() >= range.radius,
                "stored interval radius must be at least the recomputed radius"
            );
            let exact = self.components.iter().copied().sum::<f64>();
            debug_assert!(
                self.range.contains(exact),
                "stored interval must contain the exact sum of components"
            );
        }
        Ok(())
    }

    pub(crate) fn from_components(
        mut components: Vec<f64>,
        #[cfg(feature = "short-circuit")] mut range: Range,
    ) -> Self {
        if components.len() == 1 && components[0] == 0.0 {
            components.clear();
        } else if components.len() > 1 {
            components.retain(|c| *c != 0.0);
        }
        #[cfg(feature = "short-circuit")]
        {
            let derived = Range::from_components(&components);
            range.include(&derived);
        }
        let result = Self {
            #[cfg(feature = "short-circuit")]
            range,
            components,
        };
        debug_assert!(result.check_invariants().is_ok());
        result
    }

    #[cfg(test)]
    pub(crate) fn from_raw_components(components: Vec<f64>) -> Self {
        #[cfg(feature = "short-circuit")]
        let range = Range::from_components(&components);
        Self {
            #[cfg(feature = "short-circuit")]
            range,
            components,
        }
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

#[cfg(all(test, feature = "short-circuit"))]
mod short_circuit_tests {
    use super::Ap64;

    #[test]
    fn interval_contains_exact_value() {
        let a = Ap64::from(1.0);
        let b = Ap64::from(1.0e-16);
        let value = &a + &b;
        let (center, radius) = value.interval();
        let exact: f64 = value.components().iter().copied().sum();
        assert!(
            exact >= center - radius && exact <= center + radius,
            "exact value {exact} not within interval [{}, {}]",
            center - radius,
            center + radius
        );
    }
}
