//! Allocation-free exact sign computation for integer arithmetic expressions.
//!
//! This module provides exact sign computation for expressions involving
//! signed integers (i8, i16, i32, i64) without any heap allocations.
//! Unlike floating-point arithmetic, integer arithmetic is exact, but
//! intermediate results may overflow. This module tracks bit widths at
//! compile time and uses appropriately-sized fixed arrays to store results.
//!
//! # Supported Operations
//! - Addition (`+`), subtraction (`-`), multiplication (`*`)
//! - Unary negation (`-x`)
//! - `square(x)` for computing `x * x`
//!
//! # Example
//! ```rust
//! use apfp::{int_signum, square};
//!
//! let a: i32 = 100;
//! let b: i32 = 200;
//! let c: i32 = 150;
//!
//! // Compute sign of arbitrary expression
//! let sign = int_signum!((a - c) * (b - c) - (b - a) * c);
//! assert!(sign == 1 || sign == -1 || sign == 0);
//! ```

use core::cmp::Ordering;

// ============================================================================
// Multi-precision integer representation
// ============================================================================

/// Maximum number of 64-bit limbs we support (256 bits total).
pub const MAX_LIMBS: usize = 4;

/// A fixed-size signed integer represented as an array of u64 limbs.
/// Stored in little-endian order (limbs[0] is least significant).
/// The sign is tracked separately.
#[derive(Debug, Clone, Copy)]
pub struct BigInt<const N: usize> {
    pub limbs: [u64; N],
    pub negative: bool,
}

impl<const N: usize> BigInt<N> {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            limbs: [0; N],
            negative: false,
        }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&x| x == 0)
    }

    #[inline(always)]
    pub fn signum(&self) -> i32 {
        if self.is_zero() {
            0
        } else if self.negative {
            -1
        } else {
            1
        }
    }

    /// Compare absolute values. Returns ordering of |self| vs |other|.
    #[inline(always)]
    pub fn cmp_abs<const M: usize>(&self, other: &BigInt<M>) -> Ordering {
        // Compare from most significant limb
        let max_len = N.max(M);
        for i in (0..max_len).rev() {
            let a = if i < N { self.limbs[i] } else { 0 };
            let b = if i < M { other.limbs[i] } else { 0 };
            match a.cmp(&b) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// Widen to a larger BigInt.
    #[inline(always)]
    pub fn widen<const M: usize>(&self) -> BigInt<M> {
        let mut result = BigInt::zero();
        let copy_len = N.min(M);
        result.limbs[..copy_len].copy_from_slice(&self.limbs[..copy_len]);
        result.negative = self.negative;
        result
    }
}

// ============================================================================
// Input type conversions
// ============================================================================

/// Trait for types that can be converted to BigInt.
pub trait ToBigInt: Copy {
    const BITS: usize;

    fn to_bigint_1(&self) -> BigInt<1>;
    fn to_bigint_2(&self) -> BigInt<2>;
    fn to_bigint_4(&self) -> BigInt<4>;
}

macro_rules! impl_to_bigint_signed {
    ($ty:ty, $bits:expr) => {
        impl ToBigInt for $ty {
            const BITS: usize = $bits;

            #[inline(always)]
            fn to_bigint_1(&self) -> BigInt<1> {
                let negative = *self < 0;
                let abs = if negative {
                    (self.wrapping_neg()) as u64
                } else {
                    *self as u64
                };
                BigInt {
                    limbs: [abs],
                    negative,
                }
            }

            #[inline(always)]
            fn to_bigint_2(&self) -> BigInt<2> {
                let negative = *self < 0;
                let abs = if negative {
                    (self.wrapping_neg()) as u64
                } else {
                    *self as u64
                };
                BigInt {
                    limbs: [abs, 0],
                    negative,
                }
            }

            #[inline(always)]
            fn to_bigint_4(&self) -> BigInt<4> {
                let negative = *self < 0;
                let abs = if negative {
                    (self.wrapping_neg()) as u64
                } else {
                    *self as u64
                };
                BigInt {
                    limbs: [abs, 0, 0, 0],
                    negative,
                }
            }
        }
    };
}

impl_to_bigint_signed!(i8, 8);
impl_to_bigint_signed!(i16, 16);
impl_to_bigint_signed!(i32, 32);
impl_to_bigint_signed!(i64, 64);

// ============================================================================
// Arithmetic operations on BigInt
// ============================================================================

/// Add two unsigned limb arrays, returning result and final carry.
#[inline(always)]
fn add_limbs<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut result = [0u64; N];
    let mut carry = 0u64;

    for i in 0..N {
        let (sum1, c1) = a[i].overflowing_add(b[i]);
        let (sum2, c2) = sum1.overflowing_add(carry);
        result[i] = sum2;
        carry = (c1 as u64) + (c2 as u64);
    }

    (result, carry != 0)
}

/// Subtract b from a (a - b), assuming a >= b. Returns result.
#[inline(always)]
fn sub_limbs<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut result = [0u64; N];
    let mut borrow = 0u64;

    for i in 0..N {
        let (diff1, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = (b1 as u64) + (b2 as u64);
    }

    result
}

/// Multiply two limb arrays, storing result in larger array.
/// If R < N + M, the result may overflow (only lower R limbs stored).
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn mul_limbs<const N: usize, const M: usize, const R: usize>(
    a: &[u64; N],
    b: &[u64; M],
) -> [u64; R] {
    let mut result = [0u64; R];

    for i in 0..N {
        if a[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..M {
            let idx = i + j;
            if idx >= R {
                break;
            }
            let prod = (a[i] as u128) * (b[j] as u128) + (result[idx] as u128) + carry;
            result[idx] = prod as u64;
            carry = prod >> 64;
        }
        // Store remaining carry
        let mut idx = i + M;
        while carry != 0 && idx < R {
            let sum = (result[idx] as u128) + carry;
            result[idx] = sum as u64;
            carry = sum >> 64;
            idx += 1;
        }
    }

    result
}

/// Add two BigInt<N> values.
#[inline(always)]
pub fn bigint_add_same<const N: usize>(a: &BigInt<N>, b: &BigInt<N>) -> BigInt<N> {
    if a.negative == b.negative {
        // Same sign: add magnitudes
        let (limbs, _) = add_limbs(&a.limbs, &b.limbs);
        BigInt {
            limbs,
            negative: a.negative,
        }
    } else {
        // Different signs: subtract smaller from larger
        match a.cmp_abs(b) {
            Ordering::Greater => {
                let limbs = sub_limbs(&a.limbs, &b.limbs);
                BigInt {
                    limbs,
                    negative: a.negative,
                }
            }
            Ordering::Less => {
                let limbs = sub_limbs(&b.limbs, &a.limbs);
                BigInt {
                    limbs,
                    negative: b.negative,
                }
            }
            Ordering::Equal => BigInt::zero(),
        }
    }
}

/// Subtract b from a.
#[inline(always)]
pub fn bigint_sub_same<const N: usize>(a: &BigInt<N>, b: &BigInt<N>) -> BigInt<N> {
    if b.is_zero() {
        return *a;
    }
    let neg_b = BigInt {
        limbs: b.limbs,
        negative: !b.negative,
    };
    bigint_add_same(a, &neg_b)
}

/// Multiply two BigInt<2> values to get BigInt<4>.
#[inline(always)]
pub fn bigint_mul_2_2_4(a: &BigInt<2>, b: &BigInt<2>) -> BigInt<4> {
    let limbs = mul_limbs::<2, 2, 4>(&a.limbs, &b.limbs);
    let negative = a.negative != b.negative && !a.is_zero() && !b.is_zero();
    BigInt { limbs, negative }
}

/// Multiply two BigInt<4> values to get BigInt<4> (may truncate).
#[inline(always)]
pub fn bigint_mul_4_4_4(a: &BigInt<4>, b: &BigInt<4>) -> BigInt<4> {
    let limbs = mul_limbs::<4, 4, 4>(&a.limbs, &b.limbs);
    let negative = a.negative != b.negative && !a.is_zero() && !b.is_zero();
    BigInt { limbs, negative }
}

/// Negate a BigInt.
#[inline(always)]
pub fn bigint_neg<const N: usize>(a: &BigInt<N>) -> BigInt<N> {
    if a.is_zero() {
        *a
    } else {
        BigInt {
            limbs: a.limbs,
            negative: !a.negative,
        }
    }
}

// ============================================================================
// AST types for expression representation
// ============================================================================

/// A scalar integer value.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct IntScalar<T>(pub T);

/// Negation of an expression.
#[doc(hidden)]
pub struct IntNegate<T>(pub T);

/// Square of an expression.
#[doc(hidden)]
pub struct IntSquare<T>(pub T);

/// Sum of two expressions.
#[doc(hidden)]
pub struct IntSum<A, B>(pub A, pub B);

/// Difference of two expressions.
#[doc(hidden)]
pub struct IntDiff<A, B>(pub A, pub B);

/// Product of two expressions.
#[doc(hidden)]
pub struct IntProduct<A, B>(pub A, pub B);

// ============================================================================
// Generic expression evaluation using BigInt<4> arithmetic
// ============================================================================

/// Evaluate any expression to BigInt<4> and return signum.
/// This is the fallback for complex expressions.
#[doc(hidden)]
#[inline(always)]
pub fn eval_expr_signum<E: EvalToBigInt4>(expr: &E) -> i32 {
    expr.eval_to_bigint4().signum()
}

/// Trait for evaluating to BigInt<4>.
#[doc(hidden)]
pub trait EvalToBigInt4 {
    fn eval_to_bigint4(&self) -> BigInt<4>;
}

impl<T: ToBigInt> EvalToBigInt4 for IntScalar<T> {
    #[inline(always)]
    fn eval_to_bigint4(&self) -> BigInt<4> {
        self.0.to_bigint_4()
    }
}

impl<T: EvalToBigInt4> EvalToBigInt4 for IntNegate<T> {
    #[inline(always)]
    fn eval_to_bigint4(&self) -> BigInt<4> {
        bigint_neg(&self.0.eval_to_bigint4())
    }
}

impl<T: EvalToBigInt4> EvalToBigInt4 for IntSquare<T> {
    #[inline(always)]
    fn eval_to_bigint4(&self) -> BigInt<4> {
        let inner = self.0.eval_to_bigint4();
        bigint_mul_4_4_4(&inner, &inner)
    }
}

impl<A: EvalToBigInt4, B: EvalToBigInt4> EvalToBigInt4 for IntSum<A, B> {
    #[inline(always)]
    fn eval_to_bigint4(&self) -> BigInt<4> {
        bigint_add_same(&self.0.eval_to_bigint4(), &self.1.eval_to_bigint4())
    }
}

impl<A: EvalToBigInt4, B: EvalToBigInt4> EvalToBigInt4 for IntDiff<A, B> {
    #[inline(always)]
    fn eval_to_bigint4(&self) -> BigInt<4> {
        bigint_sub_same(&self.0.eval_to_bigint4(), &self.1.eval_to_bigint4())
    }
}

impl<A: EvalToBigInt4, B: EvalToBigInt4> EvalToBigInt4 for IntProduct<A, B> {
    #[inline(always)]
    fn eval_to_bigint4(&self) -> BigInt<4> {
        bigint_mul_4_4_4(&self.0.eval_to_bigint4(), &self.1.eval_to_bigint4())
    }
}

// ============================================================================
// Macro for building expression AST and evaluating signum
// ============================================================================

/// Wraps a value for use in the `int_signum!` macro to compute its square.
#[inline(always)]
pub fn int_square<T>(value: T) -> IntSquare<T> {
    IntSquare(value)
}

// ============================================================================
// Macro implementation
// ============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_finish {
    ($expr:expr,) => {
        $expr
    };
    ($expr:expr, $($rest:tt)+) => {
        compile_error!(
            "unsupported expression; use literals, identifiers, field access, +, -, *, unary -, or square(x)"
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_call {
    (($callback:ident), $expr:expr, $($rest:tt)*) => {
        $callback!($expr, $($rest)*)
    };
    (($callback:ident, $($args:tt)+), $expr:expr, $($rest:tt)*) => {
        $callback!($($args)+, $expr, $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_expr {
    ($($tokens:tt)+) => {
        $crate::_int_signum_add!((_int_signum_finish); $($tokens)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_add {
    ($cb:tt; $($tokens:tt)+) => {
        $crate::_int_signum_mul!((_int_signum_add_tail, $cb); $($tokens)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_add_tail {
    ($cb:tt, $left:expr, + $($rest:tt)+) => {
        $crate::_int_signum_mul!((_int_signum_add_fold, $cb, $left, +); $($rest)+)
    };
    ($cb:tt, $left:expr, - $($rest:tt)+) => {
        $crate::_int_signum_mul!((_int_signum_add_fold, $cb, $left, -); $($rest)+)
    };
    ($cb:tt, $left:expr, $($rest:tt)*) => {
        $crate::_int_signum_call!($cb, $left, $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_add_fold {
    ($cb:tt, $left:expr, +, $right:expr, $($rest:tt)*) => {
        $crate::_int_signum_add_tail!(
            $cb,
            $crate::analysis::integer_signum::IntSum($left, $right),
            $($rest)*
        )
    };
    ($cb:tt, $left:expr, -, $right:expr, $($rest:tt)*) => {
        $crate::_int_signum_add_tail!(
            $cb,
            $crate::analysis::integer_signum::IntDiff($left, $right),
            $($rest)*
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_mul {
    ($cb:tt; $($tokens:tt)+) => {
        $crate::_int_signum_unary!((_int_signum_mul_tail, $cb); $($tokens)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_mul_tail {
    ($cb:tt, $left:expr, * $($rest:tt)+) => {
        $crate::_int_signum_unary!((_int_signum_mul_fold, $cb, $left); $($rest)+)
    };
    ($cb:tt, $left:expr, $($rest:tt)*) => {
        $crate::_int_signum_call!($cb, $left, $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_mul_fold {
    ($cb:tt, $left:expr, $right:expr, $($rest:tt)*) => {
        $crate::_int_signum_mul_tail!(
            $cb,
            $crate::analysis::integer_signum::IntProduct($left, $right),
            $($rest)*
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_unary {
    ($cb:tt; - $($rest:tt)+) => {
        $crate::_int_signum_unary!((_int_signum_unary_neg, $cb); $($rest)+)
    };
    ($cb:tt; $($tokens:tt)+) => {
        $crate::_int_signum_atom!($cb; $($tokens)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_unary_neg {
    ($cb:tt, $expr:expr, $($rest:tt)*) => {
        $crate::_int_signum_call!(
            $cb,
            $crate::analysis::integer_signum::IntNegate($expr),
            $($rest)*
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _int_signum_atom {
    ($cb:tt; square ( $($inner:tt)+ ) $($rest:tt)*) => {
        $crate::_int_signum_call!(
            $cb,
            $crate::analysis::integer_signum::IntSquare(
                $crate::_int_signum_expr!($($inner)+)
            ),
            $($rest)*
        )
    };
    ($cb:tt; ( $($inner:tt)+ ) $($rest:tt)*) => {
        $crate::_int_signum_call!(
            $cb,
            $crate::_int_signum_expr!($($inner)+),
            $($rest)*
        )
    };
    ($cb:tt; $base:ident . $field:ident $( . $tail:ident )* $($rest:tt)*) => {
        $crate::_int_signum_call!(
            $cb,
            $crate::analysis::integer_signum::IntScalar($base.$field $(.$tail)*),
            $($rest)*
        )
    };
    ($cb:tt; $lit:literal $($rest:tt)*) => {
        $crate::_int_signum_call!(
            $cb,
            $crate::analysis::integer_signum::IntScalar($lit),
            $($rest)*
        )
    };
    ($cb:tt; $ident:ident ( $($inner:tt)* ) $($rest:tt)*) => {
        compile_error!("unsupported function call; only square(x) is supported")
    };
    ($cb:tt; $ident:ident $($rest:tt)*) => {
        $crate::_int_signum_call!(
            $cb,
            $crate::analysis::integer_signum::IntScalar($ident),
            $($rest)*
        )
    };
}

/// Computes the exact sign of an integer arithmetic expression.
///
/// Returns `1` if the expression is positive, `-1` if negative, and `0` if exactly zero.
///
/// Unlike floating-point arithmetic, integer arithmetic is exact. This macro
/// handles overflow by using appropriately-sized fixed arrays to store
/// intermediate results, determined at compile time based on the input types
/// and operations.
///
/// # Supported Operations
/// - Addition (`+`) and subtraction (`-`)
/// - Multiplication (`*`)
/// - [`square()`](crate::square) for squared terms
/// - Unary negation (`-x`)
/// - Parentheses for grouping
///
/// # Supported Input Types
/// - `i8`, `i16`, `i32`, `i64`
///
/// # Example
/// ```rust
/// use apfp::{int_signum, square};
///
/// let x: i32 = 1000000;
/// let y: i32 = 999999;
/// // Compute sign of x^2 - y^2 (would overflow in i32)
/// let sign = int_signum!(square(x) - square(y));
/// assert_eq!(sign, 1);
/// ```
#[macro_export]
macro_rules! int_signum {
    ($($tokens:tt)+) => {{
        #[allow(unused_imports)]
        use $crate::{
            _int_signum_add, _int_signum_add_fold, _int_signum_add_tail,
            _int_signum_atom, _int_signum_call, _int_signum_expr, _int_signum_finish,
            _int_signum_mul, _int_signum_mul_fold, _int_signum_mul_tail,
            _int_signum_unary, _int_signum_unary_neg,
        };
        use $crate::analysis::integer_signum as _int;
        let expr = $crate::_int_signum_expr!($($tokens)+);
        _int::eval_expr_signum(&expr)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigint_basic_operations() {
        let a: BigInt<4> = 100i32.to_bigint_4();
        let b: BigInt<4> = 50i32.to_bigint_4();

        assert_eq!(a.signum(), 1);
        assert_eq!(b.signum(), 1);

        let sum = bigint_add_same(&a, &b);
        assert_eq!(sum.limbs[0], 150);
        assert!(!sum.negative);

        let diff = bigint_sub_same(&a, &b);
        assert_eq!(diff.limbs[0], 50);
        assert!(!diff.negative);

        let neg_diff = bigint_sub_same(&b, &a);
        assert_eq!(neg_diff.limbs[0], 50);
        assert!(neg_diff.negative);
    }

    #[test]
    fn bigint_multiplication() {
        let a: BigInt<2> = 1000000i32.to_bigint_2();
        let b: BigInt<2> = 1000000i32.to_bigint_2();

        let prod = bigint_mul_2_2_4(&a, &b);
        assert_eq!(prod.limbs[0], 1_000_000_000_000u64);
        assert!(!prod.negative);
    }

    #[test]
    fn bigint_negative_numbers() {
        let a: BigInt<4> = (-100i32).to_bigint_4();
        let b: BigInt<4> = 50i32.to_bigint_4();

        assert_eq!(a.signum(), -1);
        assert_eq!(b.signum(), 1);

        let prod = bigint_mul_4_4_4(&a, &b);
        assert_eq!(prod.limbs[0], 5000);
        assert!(prod.negative);
    }

    #[test]
    fn int_signum_macro_basic() {
        let a: i32 = 10;
        let b: i32 = 5;
        assert_eq!(int_signum!(a - b), 1);
        assert_eq!(int_signum!(b - a), -1);
        assert_eq!(int_signum!(a - a), 0);
    }

    #[test]
    fn int_signum_macro_products() {
        let a: i32 = 3;
        let b: i32 = 4;
        let c: i32 = 12;
        assert_eq!(int_signum!(a * b - c), 0);
        assert_eq!(int_signum!(a * b - c + 1), 1);
        assert_eq!(int_signum!(a * b - c - 1), -1);
    }

    #[test]
    fn int_signum_macro_cross_product() {
        // Test cross product pattern: (a - c) * (b - d) - (e - f) * (g - h)
        let ax: i32 = 0;
        let ay: i32 = 0;
        let bx: i32 = 10;
        let by: i32 = 0;
        let cx: i32 = 5;
        let cy: i32 = 10;

        let sign = int_signum!((ax - cx) * (by - cy) - (ay - cy) * (bx - cx));
        assert_eq!(sign, 1);
    }

    #[test]
    fn int_signum_macro_square() {
        let x: i32 = 1000000;
        let y: i32 = 999999;
        // x^2 - y^2 = (x+y)(x-y) = 1999999 * 1 = 1999999 > 0
        let sign = int_signum!(square(x) - square(y));
        assert_eq!(sign, 1);
    }

    #[test]
    fn int_signum_large_values() {
        // Values that would overflow i32 * i32
        let big: i32 = i32::MAX / 2;
        // This should not overflow thanks to BigInt arithmetic
        let sign = int_signum!(big * big - (big - 1) * (big + 1));
        // big^2 - (big-1)(big+1) = big^2 - big^2 + 1 = 1
        assert_eq!(sign, 1);
    }
}
