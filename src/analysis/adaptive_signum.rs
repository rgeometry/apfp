//! Adaptive, allocation-free sign computation for arbitrary expressions.
//!
//! This module provides a general solution for evaluating the sign of a static
//! arithmetic expression without dynamic allocation. The core idea is:
//! 1) Evaluate the expression in `f64` and compute a conservative error bound
//!    based on the exact operation count and the magnitude of subexpressions.
//! 2) If the sign is provably correct, return it immediately.
//! 3) Otherwise, fall back to exact expansion arithmetic using fixed stack
//!    buffers computed from the expression type.
//!
//! A proc macro (`apfp_signum!`) expands an input expression into a static AST
//! of expression types defined here and calls `signum_adaptive`, which handles
//! both the fast filter and the exact fallback.
//!
//! # Example
//! ```rust
//! # use apfp::{apfp_signum, Coord};
//! # use apfp::analysis::adaptive_signum::square;
//! let a = Coord::new(0.0, 0.0);
//! let b = Coord::new(1.0, 0.0);
//! let c = Coord::new(0.0, 1.0);
//! let sign = apfp_signum!((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x));
//! assert_eq!(sign, 1);
//! let dist = apfp_signum!(
//!     (square(a.x - c.x) + square(a.y - c.y))
//!         - (square(b.x - c.x) + square(b.y - c.y))
//! );
//! assert!(dist != 0);
//! ```

use crate::expansion::{fast_two_sum, two_product, two_sum};

const U: f64 = 0.5 * f64::EPSILON;

#[derive(Debug, Clone, Copy)]
pub struct Scalar(pub f64);

#[derive(Debug, Clone, Copy)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

#[inline(always)]
pub fn dd_from(value: f64) -> Dd {
    Dd { hi: value, lo: 0.0 }
}

#[inline(always)]
pub fn dd_neg(value: Dd) -> Dd {
    Dd {
        hi: -value.hi,
        lo: -value.lo,
    }
}

#[inline(always)]
pub fn dd_add(lhs: Dd, rhs: Dd) -> Dd {
    let (sum, err) = two_sum(lhs.hi, rhs.hi);
    let t = lhs.lo + rhs.lo;
    let (sum2, err2) = two_sum(sum, t);
    let lo = err + err2;
    let (hi, lo2) = two_sum(sum2, lo);
    Dd { hi, lo: lo2 }
}

#[inline(always)]
pub fn dd_sub(lhs: Dd, rhs: Dd) -> Dd {
    dd_add(lhs, dd_neg(rhs))
}

#[inline(always)]
pub fn dd_mul(lhs: Dd, rhs: Dd) -> Dd {
    let (prod, err) = two_product(lhs.hi, rhs.hi);
    let t = lhs.hi * rhs.lo + lhs.lo * rhs.hi;
    let (sum, err2) = two_sum(prod, t);
    let lo = err + err2 + lhs.lo * rhs.lo;
    let (hi, lo2) = two_sum(sum, lo);
    Dd { hi, lo: lo2 }
}

#[inline(always)]
pub fn dd_square(value: Dd) -> Dd {
    dd_mul(value, value)
}

#[inline(always)]
pub fn dd_signum(value: Dd) -> Option<i32> {
    let bound = value.lo.abs();
    if value.hi > bound {
        Some(1)
    } else if value.hi < -bound {
        Some(-1)
    } else {
        None
    }
}
pub struct Negate<T>(pub T);
pub struct Square<T>(pub T);
pub struct Sum<A, B>(pub A, pub B);
pub struct Product<A, B>(pub A, pub B);
pub struct Diff<A, B>(pub A, pub B);

#[inline(always)]
pub fn square<T>(value: T) -> Square<T> {
    Square(value)
}

macro_rules! impl_ops {
    ($ty:ident) => {
        impl<A> std::ops::Add<A> for $ty {
            type Output = Sum<Self, A>;

            fn add(self, rhs: A) -> Self::Output {
                Sum(self, rhs)
            }
        }

        impl<A> std::ops::Sub<A> for $ty {
            type Output = Diff<Self, A>;

            fn sub(self, rhs: A) -> Self::Output {
                Diff(self, rhs)
            }
        }

        impl<A> std::ops::Mul<A> for $ty {
            type Output = Product<Self, A>;

            fn mul(self, rhs: A) -> Self::Output {
                Product(self, rhs)
            }
        }

        impl std::ops::Neg for $ty {
            type Output = Negate<Self>;

            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
    ($ty:ident [ $($gen:ident),* ]) => {
        impl<$($gen,)* C> std::ops::Add<C> for $ty<$($gen),*> {
            type Output = Sum<Self, C>;

            fn add(self, rhs: C) -> Self::Output {
                Sum(self, rhs)
            }
        }

        impl<$($gen,)* C> std::ops::Sub<C> for $ty<$($gen),*> {
            type Output = Diff<Self, C>;

            fn sub(self, rhs: C) -> Self::Output {
                Diff(self, rhs)
            }
        }

        impl<$($gen,)* C> std::ops::Mul<C> for $ty<$($gen),*> {
            type Output = Product<Self, C>;

            fn mul(self, rhs: C) -> Self::Output {
                Product(self, rhs)
            }
        }

        impl<$($gen,)*> std::ops::Neg for $ty<$($gen),*> {
            type Output = Negate<Self>;

            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
}

impl_ops!(Scalar);
impl_ops!(Negate[T]);
impl_ops!(Square[T]);
impl_ops!(Sum[A, B]);
impl_ops!(Product[A, B]);
impl_ops!(Diff[A, B]);

pub trait Eval {
    fn eval(&self) -> f64;
}

pub trait EvalBounded {
    fn eval_bounded(&self) -> (f64, f64);
}

pub trait OperationCount {
    const OPERATION_COUNT: usize;
}

pub trait Signum {
    const MAX_LEN: usize;
    const STACK_LEN: usize;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64];

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        let expansion = self.eval_exact(buffer);
        expansion_signum(expansion)
    }
}

#[inline(always)]
pub fn signum_adaptive<T: EvalBounded + OperationCount + Signum>(
    expr: T,
    buffer: &mut [f64],
) -> i32 {
    let (value, errbound) = expr.eval_bounded();
    if value > errbound {
        return 1;
    }
    if value < -errbound {
        return -1;
    }
    expr.signum(buffer)
}

#[inline(always)]
pub fn signum_exact<T: Signum>(expr: T, buffer: &mut [f64]) -> i32 {
    expr.signum(buffer)
}

impl Eval for Scalar {
    fn eval(&self) -> f64 {
        self.0
    }
}

impl EvalBounded for Scalar {
    fn eval_bounded(&self) -> (f64, f64) {
        (self.0, 0.0)
    }
}

impl OperationCount for Scalar {
    const OPERATION_COUNT: usize = 0;
}

impl Signum for Scalar {
    const MAX_LEN: usize = 1;
    const STACK_LEN: usize = 1;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        buffer[0] = self.0;
        &buffer[..1]
    }
}

impl<T: Eval> Eval for Negate<T> {
    fn eval(&self) -> f64 {
        -self.0.eval()
    }
}

impl<T: EvalBounded> EvalBounded for Negate<T> {
    fn eval_bounded(&self) -> (f64, f64) {
        let (value, err) = self.0.eval_bounded();
        (-value, err)
    }
}

impl<T: OperationCount> OperationCount for Negate<T> {
    const OPERATION_COUNT: usize = T::OPERATION_COUNT;
}

impl<T: Signum> Signum for Negate<T> {
    const MAX_LEN: usize = T::MAX_LEN;
    const STACK_LEN: usize = T::STACK_LEN + T::MAX_LEN;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (input, output) = buffer.split_at_mut(T::STACK_LEN);
        let inner = self.0.eval_exact(input);
        for (dst, &val) in output.iter_mut().zip(inner.iter()) {
            *dst = -val;
        }
        &output[..inner.len()]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        -self.0.signum(buffer)
    }
}

impl<T: Eval> Eval for Square<T> {
    fn eval(&self) -> f64 {
        let value = self.0.eval();
        value * value
    }
}

impl<T: EvalBounded> EvalBounded for Square<T> {
    fn eval_bounded(&self) -> (f64, f64) {
        let (value, err) = self.0.eval_bounded();
        let value_sq = value * value;
        let errbound = 2.0 * value.abs() * err + err * err + U * value_sq.abs();
        (value_sq, errbound)
    }
}

impl<T: OperationCount> OperationCount for Square<T> {
    const OPERATION_COUNT: usize = 1 + T::OPERATION_COUNT;
}

impl<T: Signum> Signum for Square<T> {
    const MAX_LEN: usize = 2 * T::MAX_LEN * T::MAX_LEN;
    const STACK_LEN: usize =
        T::STACK_LEN + T::STACK_LEN + Self::MAX_LEN + (2 * T::MAX_LEN + Self::MAX_LEN);

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (left_buf, rest) = buffer.split_at_mut(T::STACK_LEN);
        let (right_buf, rest) = rest.split_at_mut(T::STACK_LEN);
        let (result_buf, scratch) = rest.split_at_mut(Self::MAX_LEN);
        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.0.eval_exact(right_buf);
        let result_len = expansion_product_stack_with_scratch(
            left_exp,
            right_exp,
            result_buf,
            scratch,
            T::MAX_LEN,
        );
        &result_buf[..result_len]
    }
}

impl<A: Eval, B: Eval> Eval for Sum<A, B> {
    fn eval(&self) -> f64 {
        self.0.eval() + self.1.eval()
    }
}

impl<A: EvalBounded, B: EvalBounded> EvalBounded for Sum<A, B> {
    fn eval_bounded(&self) -> (f64, f64) {
        let (left, left_err) = self.0.eval_bounded();
        let (right, right_err) = self.1.eval_bounded();
        let value = left + right;
        let errbound = left_err + right_err + U * (left.abs() + right.abs());
        (value, errbound)
    }
}

impl<A: OperationCount, B: OperationCount> OperationCount for Sum<A, B> {
    const OPERATION_COUNT: usize = 1 + A::OPERATION_COUNT + B::OPERATION_COUNT;
}

impl<A: Signum, B: Signum> Signum for Sum<A, B> {
    const MAX_LEN: usize = A::MAX_LEN + B::MAX_LEN;
    const STACK_LEN: usize = A::STACK_LEN + B::STACK_LEN + Self::MAX_LEN;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (left_buf, rest) = buffer.split_at_mut(A::STACK_LEN);
        let (right_buf, rest) = rest.split_at_mut(B::STACK_LEN);
        let (result_buf, _) = rest.split_at_mut(Self::MAX_LEN);
        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);
        let result_len = expansion_sum_stack(left_exp, right_exp, result_buf);
        &result_buf[..result_len]
    }
}

impl<A: Eval, B: Eval> Eval for Diff<A, B> {
    fn eval(&self) -> f64 {
        self.0.eval() - self.1.eval()
    }
}

impl<A: EvalBounded, B: EvalBounded> EvalBounded for Diff<A, B> {
    fn eval_bounded(&self) -> (f64, f64) {
        let (left, left_err) = self.0.eval_bounded();
        let (right, right_err) = self.1.eval_bounded();
        let value = left - right;
        let errbound = left_err + right_err + U * (left.abs() + right.abs());
        (value, errbound)
    }
}

impl<A: OperationCount, B: OperationCount> OperationCount for Diff<A, B> {
    const OPERATION_COUNT: usize = 1 + A::OPERATION_COUNT + B::OPERATION_COUNT;
}

impl<A: Signum, B: Signum> Signum for Diff<A, B> {
    const MAX_LEN: usize = A::MAX_LEN + B::MAX_LEN;
    const STACK_LEN: usize = A::STACK_LEN + B::STACK_LEN + B::MAX_LEN + Self::MAX_LEN;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (left_buf, rest) = buffer.split_at_mut(A::STACK_LEN);
        let (right_buf, rest) = rest.split_at_mut(B::STACK_LEN);
        let (temp_buf, rest) = rest.split_at_mut(B::MAX_LEN);
        let (result_buf, _) = rest.split_at_mut(Self::MAX_LEN);
        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);
        for (i, &val) in right_exp.iter().enumerate() {
            temp_buf[i] = -val;
        }
        let neg_right = &temp_buf[..right_exp.len()];
        let result_len = expansion_sum_stack(left_exp, neg_right, result_buf);
        &result_buf[..result_len]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        let (left_buf, rest) = buffer.split_at_mut(A::STACK_LEN);
        let (right_buf, _) = rest.split_at_mut(B::STACK_LEN);
        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);
        expansion_diff_signum_direct(left_exp, right_exp)
    }
}

impl<A: Eval, B: Eval> Eval for Product<A, B> {
    fn eval(&self) -> f64 {
        self.0.eval() * self.1.eval()
    }
}

impl<A: EvalBounded, B: EvalBounded> EvalBounded for Product<A, B> {
    fn eval_bounded(&self) -> (f64, f64) {
        let (left, left_err) = self.0.eval_bounded();
        let (right, right_err) = self.1.eval_bounded();
        let value = left * right;
        let errbound = left_err * right.abs()
            + right_err * left.abs()
            + left_err * right_err
            + U * value.abs();
        (value, errbound)
    }
}

impl<A: OperationCount, B: OperationCount> OperationCount for Product<A, B> {
    const OPERATION_COUNT: usize = 1 + A::OPERATION_COUNT + B::OPERATION_COUNT;
}

impl<A: Signum, B: Signum> Signum for Product<A, B> {
    const MAX_LEN: usize = 2 * A::MAX_LEN * B::MAX_LEN;
    const STACK_LEN: usize =
        A::STACK_LEN + B::STACK_LEN + Self::MAX_LEN + (2 * A::MAX_LEN + Self::MAX_LEN);

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (left_buf, rest) = buffer.split_at_mut(A::STACK_LEN);
        let (right_buf, rest) = rest.split_at_mut(B::STACK_LEN);
        let (result_buf, scratch) = rest.split_at_mut(Self::MAX_LEN);
        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);
        let result_len = expansion_product_stack_with_scratch(
            left_exp,
            right_exp,
            result_buf,
            scratch,
            A::MAX_LEN,
        );
        &result_buf[..result_len]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        let (left_buf, rest) = buffer.split_at_mut(A::STACK_LEN);
        let (right_buf, _) = rest.split_at_mut(B::STACK_LEN);
        let left_sign = self.0.signum(left_buf);
        let right_sign = self.1.signum(right_buf);
        left_sign * right_sign
    }
}

#[inline(always)]
fn expansion_sum_stack(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> usize {
    if lhs.is_empty() {
        output[..rhs.len()].copy_from_slice(rhs);
        return rhs.len();
    } else if rhs.is_empty() {
        output[..lhs.len()].copy_from_slice(lhs);
        return lhs.len();
    }
    if lhs.len() == 1 && rhs.len() == 1 {
        let (sum, err) = two_sum(lhs[0], rhs[0]);
        if err != 0.0 {
            output[0] = err;
            output[1] = sum;
            return 2;
        }
        output[0] = sum;
        return 1;
    }

    let mut enow = lhs[0];
    let mut fnow = rhs[0];
    let mut eindex = 0usize;
    let mut findex = 0usize;
    let mut hindex = 0usize;

    let mut q = if (fnow > enow) == (fnow > -enow) {
        eindex += 1;
        enow
    } else {
        findex += 1;
        fnow
    };

    if eindex < lhs.len() && findex < rhs.len() {
        enow = lhs[eindex];
        fnow = rhs[findex];
        let (qnew, hh) = if (fnow > enow) == (fnow > -enow) {
            eindex += 1;
            fast_two_sum(enow, q)
        } else {
            findex += 1;
            fast_two_sum(fnow, q)
        };
        q = qnew;
        if hh != 0.0 {
            output[hindex] = hh;
            hindex += 1;
        }

        while eindex < lhs.len() && findex < rhs.len() {
            enow = lhs[eindex];
            fnow = rhs[findex];
            let (qnew, hh) = if (fnow > enow) == (fnow > -enow) {
                eindex += 1;
                two_sum(q, enow)
            } else {
                findex += 1;
                two_sum(q, fnow)
            };
            q = qnew;
            if hh != 0.0 {
                output[hindex] = hh;
                hindex += 1;
            }
        }
    }

    while eindex < lhs.len() {
        enow = lhs[eindex];
        let (qnew, hh) = two_sum(q, enow);
        q = qnew;
        eindex += 1;
        if hh != 0.0 {
            output[hindex] = hh;
            hindex += 1;
        }
    }

    while findex < rhs.len() {
        fnow = rhs[findex];
        let (qnew, hh) = two_sum(q, fnow);
        q = qnew;
        findex += 1;
        if hh != 0.0 {
            output[hindex] = hh;
            hindex += 1;
        }
    }

    if q != 0.0 || hindex == 0 {
        output[hindex] = q;
        hindex += 1;
    }

    hindex
}

#[inline(always)]
fn scale_expansion_stack(expansion: &[f64], scalar: f64, output: &mut [f64]) -> usize {
    if expansion.is_empty() || output.is_empty() {
        return 0;
    }

    let mut h_len = 0;
    let (product1, product0) = two_product(expansion[0], scalar);
    if product0 != 0.0 && h_len < output.len() {
        output[h_len] = product0;
        h_len += 1;
    }
    let mut q = product1;

    for &component in &expansion[1..] {
        if h_len >= output.len() {
            break;
        }

        let (product1, product0) = two_product(component, scalar);
        let (sum, err3) = two_sum(q, product0);
        if err3 != 0.0 && h_len < output.len() {
            output[h_len] = err3;
            h_len += 1;
        }
        let (new_q, err4) = fast_two_sum(product1, sum);
        if err4 != 0.0 && h_len < output.len() {
            output[h_len] = err4;
            h_len += 1;
        }
        q = new_q;
    }

    if (q != 0.0 || h_len == 0) && h_len < output.len() {
        output[h_len] = q;
        h_len += 1;
    }

    h_len
}

#[inline(always)]
fn expansion_product_stack_with_scratch(
    lhs: &[f64],
    rhs: &[f64],
    output: &mut [f64],
    scratch: &mut [f64],
    lhs_max_len: usize,
) -> usize {
    let lhs_len = lhs.len();
    let rhs_len = rhs.len();
    if lhs_len == 0 || rhs_len == 0 || output.is_empty() {
        return 0;
    }

    if lhs_len == 1 && rhs_len == 1 {
        let (p1, p0) = two_product(lhs[0], rhs[0]);
        if p0 != 0.0 {
            output[0] = p0;
            output[1] = p1;
            return 2;
        }
        output[0] = p1;
        return 1;
    }

    if lhs_len == 1 {
        return scale_expansion_stack(rhs, lhs[0], output);
    }

    if rhs_len == 1 {
        return scale_expansion_stack(lhs, rhs[0], output);
    }

    let max_needed = lhs_len.saturating_mul(rhs_len);
    debug_assert!(
        output.len() >= max_needed,
        "output buffer too small for expansion product"
    );

    let scaled_cap = lhs_max_len * 2;
    let sum_cap = output.len();
    let (scaled_buf, rest) = scratch.split_at_mut(scaled_cap);
    let (sum_buf, _) = rest.split_at_mut(sum_cap);
    let mut result_len = 0usize;

    for &component in rhs {
        let scaled_len = scale_expansion_stack(lhs, component, scaled_buf);
        if scaled_len == 0 {
            continue;
        }

        let temp_len =
            expansion_sum_stack(&output[..result_len], &scaled_buf[..scaled_len], sum_buf);
        output[..temp_len].copy_from_slice(&sum_buf[..temp_len]);
        result_len = temp_len.min(output.len());
    }

    result_len
}

#[inline(always)]
fn expansion_signum(expansion: &[f64]) -> i32 {
    let mut i = expansion.len();
    while i > 0 {
        let val = expansion[i - 1];
        if val != 0.0 {
            return val.signum() as i32;
        }
        i -= 1;
    }
    0
}

#[inline(always)]
fn expansion_diff_signum_direct(lhs: &[f64], rhs: &[f64]) -> i32 {
    let mut i = lhs.len();
    let mut j = rhs.len();

    loop {
        while i > 0 && lhs[i - 1] == 0.0 {
            i -= 1;
        }
        while j > 0 && rhs[j - 1] == 0.0 {
            j -= 1;
        }

        if i == 0 && j == 0 {
            return 0;
        }
        if i == 0 {
            let val = rhs[j - 1];
            return if val > 0.0 { -1 } else { 1 };
        }
        if j == 0 {
            let val = lhs[i - 1];
            return if val > 0.0 { 1 } else { -1 };
        }

        let left = lhs[i - 1];
        let right = rhs[j - 1];
        if left > right {
            return 1;
        }
        if left < right {
            return -1;
        }
        i -= 1;
        j -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_eval_exact_matches_expansion_product() {
        let a = 1.0e16;
        let b = 1.0;
        let diff = Diff(Scalar(a), Scalar(b));
        let square = Square(Diff(Scalar(a), Scalar(b)));

        let mut square_buf = [0.0_f64; <Square<Diff<Scalar, Scalar>> as Signum>::STACK_LEN];
        let square_exp = square.eval_exact(&mut square_buf);

        let mut diff_buf = [0.0_f64; <Diff<Scalar, Scalar> as Signum>::STACK_LEN];
        let diff_exp = diff.eval_exact(&mut diff_buf);

        let mut expected = [0.0_f64; <Square<Diff<Scalar, Scalar>> as Signum>::MAX_LEN];
        let mut scratch = [0.0_f64;
            2 * <Diff<Scalar, Scalar> as Signum>::MAX_LEN
                + <Square<Diff<Scalar, Scalar>> as Signum>::MAX_LEN];
        let expected_len = expansion_product_stack_with_scratch(
            diff_exp,
            diff_exp,
            &mut expected,
            &mut scratch,
            <Diff<Scalar, Scalar> as Signum>::MAX_LEN,
        );

        assert_eq!(square_exp, &expected[..expected_len]);
    }
}
