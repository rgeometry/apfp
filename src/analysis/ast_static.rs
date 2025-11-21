use crate::expansion::expansion_product;
use crate::geometry::Coord;
use num_rational::BigRational;
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Debug, Clone, Copy)]
struct Scalar(f64);
struct Negate<T>(T);
struct Square<T>(T);

struct Sum<A, B>(A, B);
struct Product<A, B>(A, B);

struct Diff<A, B>(A, B);

impl<A: Eval + OperationCount, B: Eval + OperationCount> Diff<A, B> {
    fn eval_bounded(&self, variables: &[f64]) -> (f64, f64) {
        const U: f64 = 0.5 * f64::EPSILON;
        let op_count = <Diff<A, B> as OperationCount>::OPERATION_COUNT;
        let gamma = (op_count as f64 * U) / (1.0 - op_count as f64 * U);
        let lhs = self.0.eval(variables);
        let rhs = self.1.eval(variables);
        (lhs - rhs, gamma * (lhs.abs() + rhs.abs()))
    }
}

macro_rules! impl_ops {
    // Zero-generic case: Type - generates Add, Sub, Mul, Neg
    ($ty:ident) => {
        impl<A> Add<A> for $ty {
            type Output = Sum<Self, A>;

            fn add(self, rhs: A) -> Self::Output {
                Sum(self, rhs)
            }
        }

        impl<A> Sub<A> for $ty {
            type Output = Diff<Self, A>;

            fn sub(self, rhs: A) -> Self::Output {
                Diff(self, rhs)
            }
        }

        impl<A> Mul<A> for $ty {
            type Output = Product<Self, A>;

            fn mul(self, rhs: A) -> Self::Output {
                Product(self, rhs)
            }
        }

        impl Neg for $ty {
            type Output = Negate<Self>;

            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
    // Generic case: Type[Gen1, Gen2, ...] - generates Add, Sub, Mul, Neg
    ($ty:ident [ $($gen:ident),* ]) => {
        impl<$($gen,)* C> Add<C> for $ty<$($gen),*> {
            type Output = Sum<Self, C>;

            fn add(self, rhs: C) -> Self::Output {
                Sum(self, rhs)
            }
        }

        impl<$($gen,)* C> Sub<C> for $ty<$($gen),*> {
            type Output = Diff<Self, C>;

            fn sub(self, rhs: C) -> Self::Output {
                Diff(self, rhs)
            }
        }

        impl<$($gen,)* C> Mul<C> for $ty<$($gen),*> {
            type Output = Product<Self, C>;

            fn mul(self, rhs: C) -> Self::Output {
                Product(self, rhs)
            }
        }

        impl<$($gen,)*> Neg for $ty<$($gen),*> {
            type Output = Negate<Self>;

            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
}

// Implement Add, Sub, Mul, Neg for all arithmetic types
impl_ops!(Scalar);
// impl_ops!(Float);
impl_ops!(Negate[A]);
impl_ops!(Square[A]);
impl_ops!(Sum[A, B]);
impl_ops!(Product[A, B]);
impl_ops!(Diff[A, B]);

macro_rules! impl_eval {
    // Zero-arg case: Type => expr (expr can use 'self' and 'variables')
    ($ty:ty => $expr:expr) => {
        impl Eval for $ty {
            #[inline]
            fn eval(&self, variables: &[f64]) -> f64 {
                // Use a block to ensure proper scoping
                { $expr }
            }
        }
    };
    // Unary case: Type[field] => expr
    ($ty:ident [ $field:ident ] => $expr:expr) => {
        impl<T: Eval> Eval for $ty<T> {
            #[inline]
            fn eval(&self, variables: &[f64]) -> f64 {
                let $field = self.0.eval(variables);
                $expr
            }
        }
    };
    // Binary case: Type[field1, field2] => expr
    ($ty:ident [ $field1:ident , $field2:ident ] => $expr:expr) => {
        impl<A: Eval, B: Eval> Eval for $ty<A, B> {
            #[inline]
            fn eval(&self, variables: &[f64]) -> f64 {
                let $field1 = self.0.eval(variables);
                let $field2 = self.1.eval(variables);
                $expr
            }
        }
    };
}

trait Eval {
    fn eval(&self, variables: &[f64]) -> f64;
}

impl Eval for Scalar {
    fn eval(&self, _variables: &[f64]) -> f64 {
        self.0
    }
}

impl Signum for Scalar {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        buffer[0] = self.0;
        &buffer[0..1]
    }

    fn signum(&self, _buffer: &mut [f64]) -> i32 {
        self.0.signum() as i32
    }
}

impl ToRational for Scalar {
    fn to_rational(&self) -> BigRational {
        // Convert f64 to rational. For exact computation, we assume the f64 represents
        // an exact value (like coordinates). In practice, this may introduce small errors
        // for non-exact f64 values, but works for geometric coordinates.
        BigRational::from_float(self.0).unwrap_or(BigRational::from_integer(0.into()))
    }
}

impl Eval for f64 {
    fn eval(&self, _variables: &[f64]) -> f64 {
        *self
    }
}

impl Signum for f64 {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        buffer[0] = *self;
        &buffer[0..1]
    }

    fn signum(&self, _buffer: &mut [f64]) -> i32 {
        (*self).signum() as i32
    }
}

impl ToRational for f64 {
    fn to_rational(&self) -> BigRational {
        BigRational::from_float(*self).unwrap_or(BigRational::from_integer(0.into()))
    }
}

impl_eval!(Negate[a] => -a);
impl_eval!(Square[a] => a.powi(2));

impl<T: Signum> Signum for Negate<T> {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let layout = BufferLayout::new(buffer);
        let (input_buf, output_buf) = layout.unary_split();

        let inner = self.0.eval_exact(input_buf);
        for (i, &val) in inner.iter().enumerate() {
            output_buf[i] = -val;
        }
        &output_buf[0..inner.len()]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        -self.0.signum(buffer)
    }
}

impl<T: ToRational> ToRational for Negate<T> {
    fn to_rational(&self) -> BigRational {
        -self.0.to_rational()
    }
}

impl<T: Signum> Signum for Square<T> {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let layout = BufferLayout::new(buffer);
        let (input_buf, output_buf) = layout.unary_split();

        let inner = self.0.eval_exact(input_buf);
        // For squaring, we need to compute the square of the expansion
        // For now, approximate by squaring the sum
        let sum: f64 = inner.iter().sum();
        output_buf[0] = sum * sum;
        &output_buf[0..1]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        let inner_sign = self.0.signum(buffer);
        // Square is always non-negative
        if inner_sign == 0 { 0 } else { 1 }
    }
}

impl<T: ToRational> ToRational for Square<T> {
    fn to_rational(&self) -> BigRational {
        let inner = self.0.to_rational();
        &inner * &inner
    }
}
impl_eval!(Sum[a, b] => a + b);
impl_eval!(Diff[a, b] => a - b);
impl_eval!(Product[a, b] => a * b);

impl<A: Signum, B: Signum> Signum for Sum<A, B> {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let layout = BufferLayout::new(buffer);
        let (left_buf, right_buf, result_buf) = layout.binary_split();

        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);

        let result_len = expansion_sum_stack(left_exp, right_exp, result_buf);
        &result_buf[0..result_len]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        // For sum, we need the actual expansion to determine the sign
        let expansion = self.eval_exact(buffer);
        expansion_signum(expansion)
    }
}

impl<A: ToRational, B: ToRational> ToRational for Sum<A, B> {
    fn to_rational(&self) -> BigRational {
        self.0.to_rational() + self.1.to_rational()
    }
}

impl<A: Signum, B: Signum> Signum for Diff<A, B> {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let layout = BufferLayout::new(buffer);
        let (left_buf, right_buf, temp_buf, result_buf) = layout.binary_with_temp();

        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);

        // For subtraction, negate the right expansion first
        for (i, &val) in right_exp.iter().enumerate() {
            temp_buf[i] = -val;
        }
        let neg_right_exp = &temp_buf[0..right_exp.len()];

        let result_len = expansion_sum_stack(left_exp, neg_right_exp, result_buf);
        &result_buf[0..result_len]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        let layout = BufferLayout::new(buffer);
        let (left_buf, right_buf, _, _) = layout.binary_with_temp();

        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);

        let left_sign = expansion_signum(left_exp);
        let right_sign = expansion_signum(right_exp);

        if left_sign == 0 && right_sign == 0 {
            return 0;
        }

        if left_sign == right_sign {
            // Same signs: compare magnitudes
            match compare_expansion_magnitudes(left_exp, right_exp) {
                std::cmp::Ordering::Greater => left_sign,
                std::cmp::Ordering::Less => -left_sign,
                std::cmp::Ordering::Equal => 0,
            }
        } else {
            // Different signs: result has sign of the larger magnitude
            match compare_expansion_magnitudes(left_exp, right_exp) {
                std::cmp::Ordering::Greater => left_sign,
                std::cmp::Ordering::Less => right_sign,
                std::cmp::Ordering::Equal => 0,
            }
        }
    }
}

impl<A: ToRational, B: ToRational> ToRational for Diff<A, B> {
    fn to_rational(&self) -> BigRational {
        self.0.to_rational() - self.1.to_rational()
    }
}

impl<A: Signum, B: Signum> Signum for Product<A, B> {
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let layout = BufferLayout::new(buffer);
        let (left_buf, right_buf, result_buf) = layout.binary_split();

        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);

        let result_len = expansion_product_stack(left_exp, right_exp, result_buf);
        &result_buf[0..result_len]
    }

    fn signum(&self, buffer: &mut [f64]) -> i32 {
        let mid = buffer.len() / 2;
        let (left_buf, right_buf) = buffer.split_at_mut(mid);
        let left_sign = self.0.signum(left_buf);
        let right_sign = self.1.signum(right_buf);
        left_sign * right_sign
    }
}

impl<A: ToRational, B: ToRational> ToRational for Product<A, B> {
    fn to_rational(&self) -> BigRational {
        self.0.to_rational() * self.1.to_rational()
    }
}

macro_rules! impl_operation_count {
    // Zero-arg case: Type => expr
    ($ty:ty => $expr:expr) => {
        impl OperationCount for $ty {
            const OPERATION_COUNT: usize = $expr;
        }
    };
    // Unary case: Type[field] => expr
    ($ty:ident [ $field:ident ] => $expr:expr) => {
        impl<$field: OperationCount> OperationCount for $ty<$field> {
            const OPERATION_COUNT: usize = $expr;
        }
    };
    // Binary case: Type[field1, field2] => expr
    ($ty:ident [ $field1:ident , $field2:ident ] => $expr:expr) => {
        impl<$field1: OperationCount, $field2: OperationCount> OperationCount
            for $ty<$field1, $field2>
        {
            const OPERATION_COUNT: usize = $expr;
        }
    };
}

trait OperationCount {
    const OPERATION_COUNT: usize;
}

/// Trait for exact arithmetic evaluation of expression signs.
/// Provides two methods: eval_exact for computing full expansions, and signum for optimized sign computation.
trait Signum {
    /// Compute the exact expansion of this expression.
    /// Returns a slice into the buffer containing the expansion components.
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64];

    /// Compute the sign of this expression using optimized algorithms.
    /// For simple cases like Diff, this avoids computing the full expansion.
    fn signum(&self, buffer: &mut [f64]) -> i32;
}

/// Trait for evaluating expressions to exact rational numbers.
trait ToRational {
    /// Evaluate this expression to an exact BigRational.
    fn to_rational(&self) -> BigRational;
}

impl_operation_count!(Scalar => 0);
impl_operation_count!(Negate[T] => T::OPERATION_COUNT);
impl_operation_count!(Square[T] => 1 + T::OPERATION_COUNT);
impl_operation_count!(Sum[A, B] => 1 + A::OPERATION_COUNT + B::OPERATION_COUNT);
impl_operation_count!(Diff[A, B] => 1 + A::OPERATION_COUNT + B::OPERATION_COUNT);
impl_operation_count!(Product[A, B] => 1 + A::OPERATION_COUNT + B::OPERATION_COUNT);

impl OperationCount for f64 {
    const OPERATION_COUNT: usize = 0;
}

const fn operation_count<T: OperationCount>(_value: &T) -> usize {
    T::OPERATION_COUNT
}

/// Helper for managing buffer allocation in expression evaluation.
/// Provides structured access to buffer regions for subexpressions and results.
/// This abstracts away manual buffer splitting and makes allocation patterns explicit.
struct BufferLayout<'a> {
    buffer: &'a mut [f64],
}

impl<'a> BufferLayout<'a> {
    fn new(buffer: &'a mut [f64]) -> Self {
        Self { buffer }
    }

    /// Split buffer for unary operation: input region and output region.
    /// Used for operations like Negate and Square.
    fn unary_split(self) -> (&'a mut [f64], &'a mut [f64]) {
        let half = self.buffer.len() / 2;
        self.buffer.split_at_mut(half)
    }

    /// Split buffer for binary operation: left input, right input, and result.
    /// Used for Sum and Product operations.
    fn binary_split(self) -> (&'a mut [f64], &'a mut [f64], &'a mut [f64]) {
        let third = self.buffer.len() / 3;
        let (left, rest) = self.buffer.split_at_mut(third);
        let (right, result) = rest.split_at_mut(third);
        (left, right, result)
    }

    /// Split buffer for binary operation with intermediate storage.
    /// Used for Diff operations that need temporary space for negation.
    fn binary_with_temp(self) -> (&'a mut [f64], &'a mut [f64], &'a mut [f64], &'a mut [f64]) {
        let quarter = self.buffer.len() / 4;
        let (a, rest) = self.buffer.split_at_mut(quarter);
        let (b, rest) = rest.split_at_mut(quarter);
        let (temp, result) = rest.split_at_mut(quarter);
        (a, b, temp, result)
    }
}

/// Stack-allocated version of expansion_sum that writes result to output buffer
fn expansion_sum_stack(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> usize {
    if lhs.is_empty() {
        output[..rhs.len()].copy_from_slice(rhs);
        return rhs.len();
    } else if rhs.is_empty() {
        output[..lhs.len()].copy_from_slice(lhs);
        return lhs.len();
    }

    let mut result_len = 0;
    let mut i = 0;
    let mut j = 0;

    while i < lhs.len() && j < rhs.len() {
        let take_lhs = match lhs[i].abs().partial_cmp(&rhs[j].abs()) {
            Some(Ordering::Less) => true,
            Some(Ordering::Equal) => true,
            Some(Ordering::Greater) => false,
            None => true,
        };

        if take_lhs {
            let (current, rest) = output.split_at_mut(result_len);
            result_len = grow_expansion_zeroelim_stack(current, lhs[i], rest);
            i += 1;
        } else {
            let (current, rest) = output.split_at_mut(result_len);
            result_len = grow_expansion_zeroelim_stack(current, rhs[j], rest);
            j += 1;
        }
    }

    while i < lhs.len() {
        let (current, rest) = output.split_at_mut(result_len);
        result_len = grow_expansion_zeroelim_stack(current, lhs[i], rest);
        i += 1;
    }

    while j < rhs.len() {
        let (current, rest) = output.split_at_mut(result_len);
        result_len = grow_expansion_zeroelim_stack(current, rhs[j], rest);
        j += 1;
    }

    result_len
}

/// Stack-allocated version of grow_expansion_zeroelim
fn grow_expansion_zeroelim_stack(expansion: &[f64], component: f64, output: &mut [f64]) -> usize {
    debug_assert!(!component.is_nan(), "NaN components are not supported");

    let mut h_len = 0;
    let mut q = component;

    for &enow in expansion {
        let sum = q + enow;
        let b_virtual = sum - q;
        let err = enow - b_virtual;
        if err != 0.0 {
            output[h_len] = err;
            h_len += 1;
        }
        q = sum;
    }

    if q != 0.0 || h_len == 0 {
        output[h_len] = q;
        h_len += 1;
    }

    h_len
}

/// Stack-allocated version of expansion_product
fn expansion_product_stack(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> usize {
    if lhs.is_empty() || rhs.is_empty() {
        return 0;
    }

    // For now, use the heap-allocated version and copy result to stack buffer
    // TODO: Implement fully stack-allocated version
    let result = expansion_product(lhs, rhs);
    let len = result.len().min(output.len());
    output[..len].copy_from_slice(&result[..len]);
    len
}

/// Compute the sign of an expansion (sum of components)
fn expansion_signum(expansion: &[f64]) -> i32 {
    let mut sum = 0.0;
    for &component in expansion {
        sum += component;
    }
    sum.signum() as i32
}

/// Compare the magnitudes of two expansions.
/// Returns Ordering::Greater if |a| > |b|, Ordering::Less if |a| < |b|, Ordering::Equal if |a| == |b|.
/// Used for optimizing sign computation in Diff operations.
fn compare_expansion_magnitudes(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    use crate::expansion::compare_magnitude;
    use std::cmp::Ordering;

    // Compare from highest magnitude components down
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match compare_magnitude(a[i].abs(), b[j].abs()) {
            Ordering::Greater => return Ordering::Greater,
            Ordering::Less => return Ordering::Less,
            Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }

    // If one expansion is exhausted, compare remaining components of the other
    if i < a.len() {
        // a has more components, check if any are non-zero
        for &val in &a[i..] {
            if val != 0.0 {
                return Ordering::Greater;
            }
        }
    } else if j < b.len() {
        // b has more components, check if any are non-zero
        for &val in &b[j..] {
            if val != 0.0 {
                return Ordering::Less;
            }
        }
    }

    Ordering::Equal
}

pub fn orient2d_fast(a: Coord, b: Coord, c: Coord) -> Option<Ordering> {
    let ax = Scalar(a.x);
    let ay = Scalar(a.y);
    let bx = Scalar(b.x);
    let by = Scalar(b.y);
    let cx = Scalar(c.x);
    let cy = Scalar(c.y);

    let det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    assert_eq!(operation_count(&det), 7);
    let (diff, errbound) = det.eval_bounded(&[]);
    if diff > errbound {
        Some(Ordering::Greater)
    } else if diff < -errbound {
        Some(Ordering::Less)
    } else {
        None
    }
}

pub fn orient2d_exact(a: Coord, b: Coord, c: Coord) -> Option<Ordering> {
    // Allocate a fixed-size buffer on the stack for expansion arithmetic
    // The buffer needs to be large enough for the expression tree evaluation
    // For orient2d, we have operations like: (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    // This involves multiple levels of operations, so we need sufficient space
    const BUFFER_SIZE: usize = 512; // Need larger buffer for complex expressions
    let mut buffer: [f64; BUFFER_SIZE] = [0.0; BUFFER_SIZE];

    let ax = Scalar(a.x);
    let ay = Scalar(a.y);
    let bx = Scalar(b.x);
    let by = Scalar(b.y);
    let cx = Scalar(c.x);
    let cy = Scalar(c.y);

    let det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    // signum computes the exact sign of the determinant.
    let sign = det.signum(&mut buffer);
    match sign {
        1 => Some(Ordering::Greater),
        -1 => Some(Ordering::Less),
        0 => Some(Ordering::Equal),
        _ => unreachable!(),
    }
}

/// Compute the orientation of three points using exact rational arithmetic.
/// This provides a reference implementation that computes the exact determinant value.
pub fn orient2d_rational(a: Coord, b: Coord, c: Coord) -> Ordering {
    let ax = Scalar(a.x);
    let ay = Scalar(a.y);
    let bx = Scalar(b.x);
    let by = Scalar(b.y);
    let cx = Scalar(c.x);
    let cy = Scalar(c.y);

    let det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    let rational_result = det.to_rational();

    rational_result.cmp(&BigRational::from_integer(0.into()))
}

/// Compare which of `p` or `q` lies closer to `origin` (squared distance) using
/// the same filtered/robust strategy.
pub fn cmp_dist_fast(origin: Coord, p: Coord, q: Coord) -> Option<Ordering> {
    let ox = Scalar(origin.x);
    let oy = Scalar(origin.y);

    let px = Scalar(p.x);
    let py = Scalar(p.y);
    let qx = Scalar(q.x);
    let qy = Scalar(q.y);

    let pdist = Square(px - ox) + Square(py - oy);
    let qdist = Square(qx - ox) + Square(qy - oy);

    let diff = pdist - qdist;
    assert_eq!(operation_count(&diff), 11);
    let (diff, errbound) = diff.eval_bounded(&[]);

    if diff > errbound {
        Some(Ordering::Greater)
    } else if diff < -errbound {
        Some(Ordering::Less)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_count() {
        orient2d_fast(
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(0.0, 1.0),
        );
    }
}
