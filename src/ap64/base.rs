use crate::expansion::{is_nonoverlapping_sorted, is_sorted_by_magnitude};

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

    pub(crate) fn from_components(mut components: Vec<f64>) -> Self {
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
    pub(crate) fn from_raw_components(components: Vec<f64>) -> Self {
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
