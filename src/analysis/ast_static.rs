use crate::geometry::Coord;
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

impl Eval for f64 {
    fn eval(&self, _variables: &[f64]) -> f64 {
        *self
    }
}

impl_eval!(Negate[a] => -a);
impl_eval!(Square[a] => a.powi(2));
impl_eval!(Sum[a, b] => a + b);
impl_eval!(Diff[a, b] => a - b);
impl_eval!(Product[a, b] => a * b);

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
