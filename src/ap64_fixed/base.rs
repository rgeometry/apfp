use crate::expansion::{compare_magnitude, fast_two_sum, two_product, two_sum};
use std::cmp::Ordering;

/// Fixed-capacity variant for experimenting with allocation-free expansions.
#[derive(Clone, Debug, PartialEq)]
pub struct Ap64Fixed<const N: usize> {
    len: usize,
    components: [f64; N],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedExpansionOverflow;

impl<const N: usize> Ap64Fixed<N> {
    /// Creates a zero value with space for `N` components.
    pub const fn zero() -> Self {
        Self {
            len: 0,
            components: [0.0; N],
        }
    }

    /// Returns the number of populated components.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Reports whether every slot is currently used.
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Reports whether the expansion is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a view of the populated tail of the buffer.
    pub fn components(&self) -> &[f64] {
        &self.components[..self.len]
    }

    /// Clears the expansion back to zero without touching the backing array.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Pushes a new component, returning an overflow error if the buffer is full.
    pub fn push(&mut self, component: f64) -> Result<(), FixedExpansionOverflow> {
        debug_assert!(
            component.is_finite(),
            "Ap64Fixed components must remain finite"
        );
        if component == 0.0 {
            return Ok(());
        }
        if self.len == N {
            return Err(FixedExpansionOverflow);
        }
        unsafe {
            *self.components.as_mut_ptr().add(self.len) = component;
        }
        self.len += 1;
        Ok(())
    }

    /// Initializes the expansion from an existing slice, truncating on overflow.
    pub fn from_slice(slice: &[f64]) -> Result<Self, FixedExpansionOverflow> {
        let mut result = Self::zero();
        for &value in slice {
            result.push(value)?;
        }
        Ok(result)
    }

    /// Computes the sum of two fixed-capacity expansions without allocating.
    #[inline(always)]
    pub fn try_add(&self, rhs: &Self) -> Result<Self, FixedExpansionOverflow> {
        let mut buffer = [0.0; N];
        let len = expansion_sum_fixed_into::<N>(self.components(), rhs.components(), &mut buffer);
        let mut result = Self::zero();
        result.len = len;
        unsafe {
            copy_prefix_unsafe(result.components.as_mut_ptr(), buffer.as_ptr(), len);
        }
        Ok(result)
    }

    /// Computes the product of two fixed-capacity expansions without allocating.
    #[inline(always)]
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, FixedExpansionOverflow> {
        let mut buffer = [0.0; N];
        let len =
            expansion_product_fixed_into::<N>(self.components(), rhs.components(), &mut buffer);
        let mut result = Self::zero();
        result.len = len;
        unsafe {
            copy_prefix_unsafe(result.components.as_mut_ptr(), buffer.as_ptr(), len);
        }
        Ok(result)
    }

    /// Returns the sum of all stored components.
    #[inline(always)]
    pub fn approx(&self) -> f64 {
        let mut sum = 0.0;
        let ptr = self.components.as_ptr();
        let mut i = 0;
        while i < self.len {
            unsafe {
                sum += *ptr.add(i);
            }
            i += 1;
        }
        sum
    }

    /// Reports whether the expansion is exactly zero.
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.len == 0
    }

    /// Returns the negated expansion.
    #[inline(always)]
    pub fn negate(&self) -> Self {
        let mut result = Self::zero();
        result.len = self.len;
        let src = self.components.as_ptr();
        let dst = result.components.as_mut_ptr();
        let mut i = 0;
        while i < self.len {
            unsafe {
                *dst.add(i) = -*src.add(i);
            }
            i += 1;
        }
        result
    }

    /// Computes the difference between two fixed-capacity expansions.
    #[inline(always)]
    pub fn try_sub(&self, rhs: &Self) -> Result<Self, FixedExpansionOverflow> {
        self.try_add(&rhs.negate())
    }
}

impl<const N: usize> Default for Ap64Fixed<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl Ap64Fixed<0> {
    /// Convenience constructor for the zero-capacity zero value.
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Ap64Fixed<1> {
    /// Creates an expansion from a single `f64`. Zero is stored as the empty expansion.
    pub fn from_f64(value: f64) -> Self {
        debug_assert!(!value.is_nan(), "NaN components are not supported");
        if value == 0.0 {
            Self::zero()
        } else {
            Self {
                len: 1,
                components: [value],
            }
        }
    }
}

impl From<f64> for Ap64Fixed<1> {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

#[inline(always)]
fn grow_expansion_zeroelim_into<const N: usize>(
    expansion: &[f64],
    component: f64,
    out: &mut [f64; N],
) -> usize {
    if expansion.len() > N {
        panic!(
            "Ap64Fixed overflow: expansion length {} exceeds capacity {}",
            expansion.len(),
            N
        );
    }

    if component == 0.0 {
        unsafe {
            copy_prefix_unsafe(out.as_mut_ptr(), expansion.as_ptr(), expansion.len());
        }
        return expansion.len();
    }

    let mut q = component;
    let mut out_len = 0usize;
    let exp_ptr = expansion.as_ptr();
    let mut idx = 0usize;
    let exp_len = expansion.len();

    let out_ptr = out.as_mut_ptr();

    while idx < exp_len {
        let enow = unsafe { *exp_ptr.add(idx) };
        let (sum, err) = two_sum(q, enow);
        if err != 0.0 {
            if out_len >= N {
                panic!("Ap64Fixed overflow while growing expansion");
            }
            unsafe {
                *out_ptr.add(out_len) = err;
            }
            out_len += 1;
        }
        q = sum;
        idx += 1;
    }

    if q != 0.0 || out_len == 0 {
        if out_len >= N {
            panic!("Ap64Fixed overflow while growing expansion");
        }
        unsafe {
            *out_ptr.add(out_len) = q;
        }
        out_len += 1;
    }

    out_len
}

#[inline(always)]
fn expansion_sum_fixed_into<const N: usize>(lhs: &[f64], rhs: &[f64], out: &mut [f64; N]) -> usize {
    let lhs_len = lhs.len();
    let rhs_len = rhs.len();
    if lhs_len == 0 {
        if rhs_len > N {
            panic!(
                "Ap64Fixed overflow: sum length {} exceeds capacity {}",
                rhs_len, N
            );
        }
        unsafe {
            copy_prefix_unsafe(out.as_mut_ptr(), rhs.as_ptr(), rhs_len);
        }
        return rhs_len;
    }
    if rhs_len == 0 {
        if lhs_len > N {
            panic!(
                "Ap64Fixed overflow: sum length {} exceeds capacity {}",
                lhs_len, N
            );
        }
        unsafe {
            copy_prefix_unsafe(out.as_mut_ptr(), lhs.as_ptr(), lhs_len);
        }
        return lhs_len;
    }

    let lhs_ptr = lhs.as_ptr();
    let rhs_ptr = rhs.as_ptr();
    let mut current = [0.0; N];
    let mut current_len = 0usize;
    let mut scratch = [0.0; N];

    let mut i = 0usize;
    let mut j = 0usize;

    while i < lhs_len && j < rhs_len {
        let lhs_val = unsafe { *lhs_ptr.add(i) };
        let rhs_val = unsafe { *rhs_ptr.add(j) };
        let take_lhs = matches!(
            compare_magnitude(lhs_val, rhs_val),
            Ordering::Less | Ordering::Equal
        );
        let component = if take_lhs {
            i += 1;
            lhs_val
        } else {
            j += 1;
            rhs_val
        };
        let current_slice = unsafe { std::slice::from_raw_parts(current.as_ptr(), current_len) };
        let next = grow_expansion_zeroelim_into::<N>(current_slice, component, &mut scratch);
        if next > 0 {
            unsafe {
                copy_prefix_unsafe(current.as_mut_ptr(), scratch.as_ptr(), next);
            }
        }
        current_len = next;
    }

    while i < lhs_len {
        let component = unsafe { *lhs_ptr.add(i) };
        i += 1;
        let current_slice = unsafe { std::slice::from_raw_parts(current.as_ptr(), current_len) };
        let next = grow_expansion_zeroelim_into::<N>(current_slice, component, &mut scratch);
        if next > 0 {
            unsafe {
                copy_prefix_unsafe(current.as_mut_ptr(), scratch.as_ptr(), next);
            }
        }
        current_len = next;
    }

    while j < rhs_len {
        let component = unsafe { *rhs_ptr.add(j) };
        j += 1;
        let current_slice = unsafe { std::slice::from_raw_parts(current.as_ptr(), current_len) };
        let next = grow_expansion_zeroelim_into::<N>(current_slice, component, &mut scratch);
        if next > 0 {
            unsafe {
                copy_prefix_unsafe(current.as_mut_ptr(), scratch.as_ptr(), next);
            }
        }
        current_len = next;
    }

    if current_len > 0 {
        unsafe {
            copy_prefix_unsafe(out.as_mut_ptr(), current.as_ptr(), current_len);
        }
    }
    current_len
}

#[inline(always)]
fn scale_expansion_fixed_into<const N: usize>(
    expansion: &[f64],
    scalar: f64,
    out: &mut [f64; N],
) -> usize {
    if expansion.is_empty() || scalar == 0.0 {
        return 0;
    }

    let mut out_len = 0usize;
    let out_ptr = out.as_mut_ptr();
    let exp_ptr = expansion.as_ptr();
    let (product1, product0) = two_product(unsafe { *exp_ptr }, scalar);
    if product0 != 0.0 {
        if out_len >= N {
            panic!("Ap64Fixed overflow while scaling expansion");
        }
        unsafe {
            *out_ptr.add(out_len) = product0;
        }
        out_len += 1;
    }

    let mut q = product1;

    let mut idx = 1usize;
    while idx < expansion.len() {
        let component = unsafe { *exp_ptr.add(idx) };
        let (p1, p0) = two_product(component, scalar);
        let (sum, err3) = two_sum(q, p0);
        if err3 != 0.0 {
            if out_len >= N {
                panic!("Ap64Fixed overflow while scaling expansion");
            }
            unsafe {
                *out_ptr.add(out_len) = err3;
            }
            out_len += 1;
        }
        let (new_q, err4) = fast_two_sum(p1, sum);
        if err4 != 0.0 {
            if out_len >= N {
                panic!("Ap64Fixed overflow while scaling expansion");
            }
            unsafe {
                *out_ptr.add(out_len) = err4;
            }
            out_len += 1;
        }
        q = new_q;
        idx += 1;
    }

    if q != 0.0 || out_len == 0 {
        if out_len >= N {
            panic!("Ap64Fixed overflow while scaling expansion");
        }
        unsafe {
            *out_ptr.add(out_len) = q;
        }
        out_len += 1;
    }

    out_len
}

#[inline(always)]
fn expansion_product_fixed_into<const N: usize>(
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [f64; N],
) -> usize {
    if lhs.is_empty() || rhs.is_empty() {
        return 0;
    }

    let mut acc = [0.0; N];
    let mut acc_len = 0usize;
    let mut temp = [0.0; N];
    let mut scaled = [0.0; N];

    let rhs_ptr = rhs.as_ptr();
    let mut idx = 0usize;
    let rhs_len = rhs.len();
    while idx < rhs_len {
        let component = unsafe { *rhs_ptr.add(idx) };
        idx += 1;
        let scaled_len = scale_expansion_fixed_into::<N>(lhs, component, &mut scaled);
        if scaled_len == 0 {
            continue;
        }
        let acc_slice = unsafe { std::slice::from_raw_parts(acc.as_ptr(), acc_len) };
        let scaled_slice = unsafe { std::slice::from_raw_parts(scaled.as_ptr(), scaled_len) };
        let new_len = expansion_sum_fixed_into::<N>(acc_slice, scaled_slice, &mut temp);
        unsafe {
            copy_prefix_unsafe(acc.as_mut_ptr(), temp.as_ptr(), new_len);
        }
        acc_len = new_len;
    }

    unsafe {
        copy_prefix_unsafe(out.as_mut_ptr(), acc.as_ptr(), acc_len);
    }
    acc_len
}

#[inline(always)]
unsafe fn copy_prefix_unsafe(dest: *mut f64, src: *const f64, len: usize) {
    let dst_ptr = dest;
    let src_ptr = src;
    match len {
        0 => {}
        1 => unsafe {
            *dst_ptr = *src_ptr;
        },
        2 => unsafe {
            *dst_ptr = *src_ptr;
            *dst_ptr.add(1) = *src_ptr.add(1);
        },
        3 => unsafe {
            *dst_ptr = *src_ptr;
            *dst_ptr.add(1) = *src_ptr.add(1);
            *dst_ptr.add(2) = *src_ptr.add(2);
        },
        4 => unsafe {
            *dst_ptr = *src_ptr;
            *dst_ptr.add(1) = *src_ptr.add(1);
            *dst_ptr.add(2) = *src_ptr.add(2);
            *dst_ptr.add(3) = *src_ptr.add(3);
        },
        _ => unsafe {
            let mut i = 0;
            while i < len {
                *dst_ptr.add(i) = *src_ptr.add(i);
                i += 1;
            }
        },
    }
}
