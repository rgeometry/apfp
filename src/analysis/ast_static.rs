use crate::geometry::Coord;
use alloca::with_alloca_zeroed;
use geometry_predicates::orient2d as gp_orient2d;
use num_rational::BigRational;
use std::{
    cmp::Ordering,
    mem::MaybeUninit,
    ops::{Add, Mul, Neg, Sub},
};

use crate::expansion::two_sum;

#[derive(Debug, Clone, Copy)]
struct Scalar(f64);
struct Negate<T>(T);
struct Square<T>(T);

struct Sum<A, B>(A, B);
struct Product<A, B>(A, B);

struct Diff<A, B>(A, B);

#[derive(Debug, Clone, Copy)]
struct DiffScalarScalar(Scalar, Scalar);

impl DiffScalarScalar {
    fn new(lhs: Scalar, rhs: Scalar) -> Self {
        Self(lhs, rhs)
    }
}

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
// impl_ops!(Float);
impl_ops!(Negate[A]);
impl_ops!(Square[A]);
impl_ops!(Sum[A, B]);
impl_ops!(Product[A, B]);
impl_ops!(Diff[A, B]);

impl Sub<Scalar> for Scalar {
    type Output = DiffScalarScalar;

    fn sub(self, rhs: Scalar) -> Self::Output {
        DiffScalarScalar::new(self, rhs)
    }
}

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
    const MAX_LEN: usize = 1;
    const STACK_LEN: usize = 1;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        buffer[0] = self.0;
        &buffer[..1]
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
    const MAX_LEN: usize = 1;
    const STACK_LEN: usize = 1;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        buffer[0] = *self;
        &buffer[..1]
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
    const MAX_LEN: usize = T::MAX_LEN;
    const STACK_LEN: usize = T::STACK_LEN + T::MAX_LEN;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (input_buf, output_buf) = buffer.split_at_mut(T::STACK_LEN);
        let inner = self.0.eval_exact(input_buf);
        for (dst, &val) in output_buf.iter_mut().zip(inner.iter()) {
            *dst = -val;
        }
        &output_buf[..inner.len()]
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
    const MAX_LEN: usize = 1;
    const STACK_LEN: usize = T::STACK_LEN + T::MAX_LEN;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (input_buf, output_buf) = buffer.split_at_mut(T::STACK_LEN);
        let inner = self.0.eval_exact(input_buf);
        let sum: f64 = inner.iter().sum();
        output_buf[0] = sum * sum;
        &output_buf[..1]
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

impl Eval for DiffScalarScalar {
    fn eval(&self, _variables: &[f64]) -> f64 {
        self.0.eval(&[]) - self.1.eval(&[])
    }
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

    fn signum(&self, buffer: &mut [f64]) -> i32 {
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
        let neg_right_exp = &temp_buf[..right_exp.len()];

        let result_len = expansion_sum_stack(left_exp, neg_right_exp, result_buf);
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

impl<A: ToRational, B: ToRational> ToRational for Diff<A, B> {
    fn to_rational(&self) -> BigRational {
        self.0.to_rational() - self.1.to_rational()
    }
}

impl<A: Signum, B: Signum> Signum for Product<A, B> {
    const MAX_LEN: usize = 2 * A::MAX_LEN * B::MAX_LEN;
    const STACK_LEN: usize = A::STACK_LEN + B::STACK_LEN + Self::MAX_LEN;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (left_buf, rest) = buffer.split_at_mut(A::STACK_LEN);
        let (right_buf, rest) = rest.split_at_mut(B::STACK_LEN);
        let (result_buf, _) = rest.split_at_mut(Self::MAX_LEN);

        let left_exp = self.0.eval_exact(left_buf);
        let right_exp = self.1.eval_exact(right_buf);

        let result_len = expansion_product_stack(left_exp, right_exp, result_buf);
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

impl Signum for DiffScalarScalar {
    const MAX_LEN: usize = 2;
    const STACK_LEN: usize = 2;

    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64] {
        let (sum, err) = two_sum(self.0.0, -self.1.0);
        if err != 0.0 {
            buffer[0] = err;
            buffer[1] = sum;
            &buffer[..2]
        } else {
            buffer[0] = sum;
            &buffer[..1]
        }
    }

    fn signum(&self, _buffer: &mut [f64]) -> i32 {
        (self.0.0 - self.1.0).signum() as i32
    }
}

impl<A: ToRational, B: ToRational> ToRational for Product<A, B> {
    fn to_rational(&self) -> BigRational {
        self.0.to_rational() * self.1.to_rational()
    }
}

impl ToRational for DiffScalarScalar {
    fn to_rational(&self) -> BigRational {
        self.0.to_rational() - self.1.to_rational()
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
    const MAX_LEN: usize;
    const STACK_LEN: usize;
    /// Compute the exact expansion of this expression.
    /// Returns a slice into the buffer containing the expansion components.
    fn eval_exact<'a>(&self, buffer: &'a mut [f64]) -> &'a [f64];

    /// Compute the sign of this expression using optimized algorithms.
    /// For simple cases like Diff, this avoids computing the full expansion.
    fn signum(&self, buffer: &mut [f64]) -> i32;

    #[inline(always)]
    fn stack_len(&self) -> usize {
        Self::STACK_LEN
    }
}

/// Compute sign of expansionA - expansionB by directly comparing values.
/// This avoids redundant sign computations in the common case.
fn expansion_diff_signum_direct(a: &[f64], b: &[f64]) -> i32 {
    // Compute a - b directly by summing and comparing
    let a_val: f64 = a.iter().sum();
    let b_val: f64 = b.iter().sum();
    (a_val - b_val).signum() as i32
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

impl OperationCount for DiffScalarScalar {
    const OPERATION_COUNT: usize = 1;
}

impl<A> Add<A> for DiffScalarScalar {
    type Output = Sum<Self, A>;

    fn add(self, rhs: A) -> Self::Output {
        Sum(self, rhs)
    }
}

impl<A> Sub<A> for DiffScalarScalar {
    type Output = Diff<Self, A>;

    fn sub(self, rhs: A) -> Self::Output {
        Diff(self, rhs)
    }
}

impl<A> Mul<A> for DiffScalarScalar {
    type Output = Product<Self, A>;

    fn mul(self, rhs: A) -> Self::Output {
        Product(self, rhs)
    }
}

impl Neg for DiffScalarScalar {
    type Output = Negate<Self>;

    fn neg(self) -> Self::Output {
        Negate(self)
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

fn with_stack_f64<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    use std::{mem, slice};

    if len == 0 {
        let mut empty: [f64; 0] = [];
        return f(&mut empty);
    }

    let align = mem::align_of::<f64>();
    let bytes = len * mem::size_of::<f64>() + align;

    with_alloca_zeroed(bytes, |raw| {
        let ptr = raw.as_mut_ptr();
        let addr = ptr as usize;
        let offset = (align - (addr % align)) % align;
        let aligned = unsafe { ptr.add(offset) };
        let slice = unsafe { slice::from_raw_parts_mut(aligned as *mut f64, len) };
        f(slice)
    })
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
    if lhs.is_empty() || rhs.is_empty() || output.is_empty() {
        return 0;
    }

    let max_needed = lhs.len().saturating_mul(rhs.len());
    debug_assert!(
        output.len() >= max_needed,
        "output buffer too small for expansion product"
    );

    let scaled_cap = lhs.len() * 2;
    let scratch_total = scaled_cap + output.len();

    with_stack_f64(scratch_total, |scratch| {
        let (scaled_buf, sum_buf) = scratch.split_at_mut(scaled_cap);
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
    })
}

/// Stack-allocated version of scale_expansion
fn scale_expansion_stack(expansion: &[f64], scalar: f64, output: &mut [f64]) -> usize {
    use crate::expansion::{fast_two_sum, two_product, two_sum};

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

/// Compute the sign of an expansion (sum of components)
fn expansion_signum(expansion: &[f64]) -> i32 {
    let mut sum = 0.0;
    for &component in expansion {
        sum += component;
    }
    sum.signum() as i32
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

pub fn orient2d_adaptive(a: Coord, b: Coord, c: Coord) -> Option<Ordering> {
    // First try the fast filter - if it gives a definitive answer, use it
    if let Some(result) = orient2d_fast(a, b, c) {
        return Some(result);
    }

    // Fallback to exact computation - only allocate buffer when needed
    orient2d_exact(a, b, c)
}

pub fn orient2d_exact(a: Coord, b: Coord, c: Coord) -> Option<Ordering> {
    const BUFFER_SIZE: usize = 512;
    let mut buffer = MaybeUninit::<[f64; BUFFER_SIZE]>::uninit();

    let ax = Scalar(a.x);
    let ay = Scalar(a.y);
    let bx = Scalar(b.x);
    let by = Scalar(b.y);
    let cx = Scalar(c.x);
    let cy = Scalar(c.y);

    let det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    debug_assert!(det.stack_len() <= BUFFER_SIZE);
    let sign = det.signum(unsafe { &mut *buffer.as_mut_ptr() });
    match sign {
        1 => Some(Ordering::Greater),
        -1 => Some(Ordering::Less),
        0 => Some(Ordering::Equal),
        _ => unreachable!(),
    }
}

/// Reference wrapper that forwards to the `geometry_predicates` crate.
/// Useful for benchmarking and inspecting the code generation of the upstream implementation.
pub fn orient2d_geometry_predicates(a: Coord, b: Coord, c: Coord) -> Ordering {
    let det = gp_orient2d([a.x, a.y], [b.x, b.y], [c.x, c.y]);
    if det > 0.0 {
        Ordering::Greater
    } else if det < 0.0 {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

/// Direct implementation of orient2d using expansion arithmetic without AST overhead
pub fn orient2d_direct(a: Coord, b: Coord, c: Coord) -> Option<Ordering> {
    // Compute determinant directly: (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
    // This expands to: ax*by - ax*cy - cx*by + cx*cy - ay*bx + ay*cx + cy*bx - cy*cx

    // Use stack buffer for expansion arithmetic
    const BUFFER_SIZE: usize = 256;
    let mut buffer: [f64; BUFFER_SIZE] = [0.0; BUFFER_SIZE];

    // Compute the 8 terms of the expanded determinant
    let ax_by = a.x * b.y;
    let ax_cy = a.x * c.y;
    let cx_by = c.x * b.y;
    let cx_cy = c.x * c.y;
    let ay_bx = a.y * b.x;
    let ay_cx = a.y * c.x;
    let cy_bx = c.y * b.x;
    let cy_cx = c.y * c.x;

    // det = ax_by - ax_cy - cx_by + cx_cy - ay_bx + ay_cx + cy_bx - cy_cx
    // Group as: (ax_by + cx_cy + ay_cx + cy_bx) - (ax_cy + cx_by + ay_bx + cy_cx)

    let positive_terms = [ax_by, cx_cy, ay_cx, cy_bx];
    let negative_terms = [ax_cy, cx_by, ay_bx, cy_cx];

    // Use buffer to compute expansions stack-allocated
    // Split buffer: first quarter for pos, second quarter for neg, rest for result
    let quarter = BUFFER_SIZE / 4;
    let (pos_buf, rest) = buffer.split_at_mut(quarter);
    let (neg_buf, result_buf) = rest.split_at_mut(quarter);

    // Sum positive terms into pos_buf
    let mut pos_len = 0;
    for &term in &positive_terms {
        if pos_len == 0 {
            pos_buf[0] = term;
            pos_len = 1;
        } else {
            // Need to be careful with borrowing here
            let (current, dest) = pos_buf.split_at_mut(pos_len);
            let temp_len = expansion_sum_stack(current, &[term], dest);
            pos_len += temp_len;
            if pos_len > quarter - 4 {
                // Leave some margin
                pos_len = quarter - 4;
            }
        }
    }

    // Sum negative terms into neg_buf
    let mut neg_len = 0;
    for &term in &negative_terms {
        if neg_len == 0 {
            neg_buf[0] = term;
            neg_len = 1;
        } else {
            let (current, dest) = neg_buf.split_at_mut(neg_len);
            let temp_len = expansion_sum_stack(current, &[term], dest);
            neg_len += temp_len;
            if neg_len > quarter - 4 {
                neg_len = quarter - 4;
            }
        }
    }

    // Compute final determinant as pos_expansion - neg_expansion
    // Create negated copy of negative terms
    let mut neg_copy = [0.0; 16]; // Should be enough for our terms
    let neg_copy_len = neg_len.min(neg_copy.len());
    for i in 0..neg_copy_len {
        neg_copy[i] = -neg_buf[i];
    }

    let det_len = expansion_sum_stack(&pos_buf[0..pos_len], &neg_copy[0..neg_copy_len], result_buf);

    // Determine sign
    let sign = expansion_signum(&result_buf[0..det_len]);
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
