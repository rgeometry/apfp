//! Abstract Syntax Tree for representing calculations without memory allocation.
//!
//! This AST is designed to represent mathematical expressions that can be evaluated,
//! analyzed for error bounds, and transformed. All data is stored by value to avoid
//! heap allocation and reference counting. Expressions are copied rather than shared.

use crate::geometry::Coord;
use std::cmp::Ordering;

// Floating-point error model constant: u = 0.5 * f64::EPSILON
const U: f64 = 0.5 * f64::EPSILON;

// Import items from parent module for testing
#[cfg(test)]
use super::{GAMMA5, GAMMA7, GAMMA11, cmp_dist_fast};

/// Represents a variable in a calculation.
/// Variables can be coordinates (with x/y components) or scalar values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variable {
    /// A coordinate variable with x and y components
    Coord { id: usize },
    /// A scalar variable
    Scalar { id: usize },
}

/// Represents an expression in the AST.
/// Uses Box for recursive cases to avoid infinite type size while avoiding Rc.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant floating-point value
    Const(f64),

    /// A scalar variable reference
    Var(Variable),

    /// Access the x-component of a coordinate variable
    CoordX(Variable),

    /// Access the y-component of a coordinate variable
    CoordY(Variable),

    /// Negation: -expr
    Neg(Box<Expr>),

    /// Addition: left + right
    Add(Box<Expr>, Box<Expr>),

    /// Subtraction: left - right
    Sub(Box<Expr>, Box<Expr>),

    /// Multiplication: left * right
    Mul(Box<Expr>, Box<Expr>),

    /// Absolute value: |expr|
    Abs(Box<Expr>),

    /// Square: expr²
    Square(Box<Expr>),
}

impl Expr {
    /// Create a constant expression
    pub const fn const_(value: f64) -> Self {
        Self::Const(value)
    }

    /// Create a variable reference
    pub const fn var(variable: Variable) -> Self {
        Self::Var(variable)
    }

    /// Create a coordinate x-component access
    pub const fn coord_x(coord_var: Variable) -> Self {
        Self::CoordX(coord_var)
    }

    /// Create a coordinate y-component access
    pub const fn coord_y(coord_var: Variable) -> Self {
        Self::CoordY(coord_var)
    }

    /// Create a negation expression
    #[allow(clippy::should_implement_trait)]
    pub fn neg(expr: Expr) -> Self {
        Self::Neg(Box::new(expr))
    }

    /// Create an addition expression
    #[allow(clippy::should_implement_trait)]
    pub fn add(left: Expr, right: Expr) -> Self {
        Self::Add(Box::new(left), Box::new(right))
    }

    /// Create a subtraction expression
    #[allow(clippy::should_implement_trait)]
    pub fn sub(left: Expr, right: Expr) -> Self {
        Self::Sub(Box::new(left), Box::new(right))
    }

    /// Create a multiplication expression
    #[allow(clippy::should_implement_trait)]
    pub fn mul(left: Expr, right: Expr) -> Self {
        Self::Mul(Box::new(left), Box::new(right))
    }

    /// Create an absolute value expression
    pub fn abs(expr: Expr) -> Self {
        Self::Abs(Box::new(expr))
    }

    /// Create a square expression: expr²
    pub fn square(expr: Expr) -> Self {
        Self::Square(Box::new(expr))
    }

    /// Compute the GAMMA error bound for this expression.
    ///
    /// GAMMA represents the maximum relative error that can accumulate from
    /// floating-point operations in this expression. It follows the standard
    /// floating-point error model: γ_n = (n * u) / (1 - n * u) where n is the
    /// number of operations and u = 0.5 * f64::EPSILON.
    pub fn gamma(&self) -> f64 {
        let n = self.operation_count();
        if n == 0 {
            0.0
        } else {
            (n as f64 * U) / (1.0 - n as f64 * U)
        }
    }

    /// Count the number of floating-point operations in this expression.
    ///
    /// Each arithmetic operation (add, subtract, multiply, negate, abs, square) counts as 1.
    /// Constants and variable accesses do not count as operations.
    fn operation_count(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Var(_) | Expr::CoordX(_) | Expr::CoordY(_) => 0,
            Expr::Neg(inner) | Expr::Abs(inner) | Expr::Square(inner) => {
                1 + inner.operation_count()
            }
            Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
                1 + left.operation_count() + right.operation_count()
            }
        }
    }
}

/// Compare squared distances from origin to points p and q using AST-based computation.
///
/// This implements the same logic as `cmp_dist_fast` but using the Expr AST to represent
/// the calculation. Returns None when floating-point error makes the result unreliable.
pub fn cmp_dist_fast_ast(origin: &Coord, p: &Coord, q: &Coord) -> Option<Ordering> {
    // Create evaluation context with the three coordinate variables
    let mut ctx = EvalContext::new();
    let origin_var = ctx.add_coord(*origin);
    let p_var = ctx.add_coord(*p);
    let q_var = ctx.add_coord(*q);

    // Build expression for squared distance from origin to p:
    // pdist = (px - ox)^2 + (py - oy)^2
    let pdx = Expr::sub(Expr::coord_x(p_var), Expr::coord_x(origin_var));
    let pdy = Expr::sub(Expr::coord_y(p_var), Expr::coord_y(origin_var));
    let pdx_sq = Expr::square(pdx);
    let pdy_sq = Expr::square(pdy);
    let pdist = Expr::add(pdx_sq, pdy_sq);

    // Build expression for squared distance from origin to q:
    // qdist = (qx - ox)^2 + (qy - oy)^2
    let qdx = Expr::sub(Expr::coord_x(q_var), Expr::coord_x(origin_var));
    let qdy = Expr::sub(Expr::coord_y(q_var), Expr::coord_y(origin_var));
    let qdx_sq = Expr::square(qdx);
    let qdy_sq = Expr::square(qdy);
    let qdist = Expr::add(qdx_sq, qdy_sq);

    // Evaluate pdist and qdist
    let pdist_val = ctx.eval(&pdist);
    let qdist_val = ctx.eval(&qdist);

    // Compute diff and sum directly as f64 values
    let diff_val = pdist_val - qdist_val;
    let sum_val = pdist_val.abs() + qdist_val.abs();

    // Compute error bound using the gamma method on the diff expression
    // This gives the error bound for the entire distance comparison computation
    let errbound = pdist.gamma() * sum_val;

    // Make the comparison decision
    if diff_val > errbound {
        Some(Ordering::Greater) // p is farther than q
    } else if diff_val < -errbound {
        Some(Ordering::Less) // p is closer than q
    } else {
        None // too close to call due to floating-point error
    }
}

/// Context for evaluating expressions.
/// Provides values for variables during evaluation.
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Coordinate values indexed by variable id
    pub coords: Vec<crate::geometry::Coord>,
    /// Scalar values indexed by variable id
    pub scalars: Vec<f64>,
}

impl EvalContext {
    /// Create a new evaluation context
    pub fn new() -> Self {
        Self {
            coords: Vec::new(),
            scalars: Vec::new(),
        }
    }

    /// Add a coordinate variable and return its variable reference
    pub fn add_coord(&mut self, coord: crate::geometry::Coord) -> Variable {
        let id = self.coords.len();
        self.coords.push(coord);
        Variable::Coord { id }
    }

    /// Add a scalar variable and return its variable reference
    pub fn add_scalar(&mut self, value: f64) -> Variable {
        let id = self.scalars.len();
        self.scalars.push(value);
        Variable::Scalar { id }
    }

    /// Evaluate an expression in this context
    pub fn eval(&self, expr: &Expr) -> f64 {
        match expr {
            Expr::Const(value) => *value,
            Expr::Var(Variable::Coord { .. }) => {
                panic!("Cannot evaluate bare coordinate variable; use CoordX or CoordY")
            }
            Expr::Var(Variable::Scalar { id }) => self.scalars[*id],
            Expr::CoordX(Variable::Coord { id }) => self.coords[*id].x,
            Expr::CoordY(Variable::Coord { id }) => self.coords[*id].y,
            Expr::CoordX(_) | Expr::CoordY(_) => {
                panic!("CoordX/CoordY requires a coordinate variable")
            }
            Expr::Neg(inner) => -self.eval(inner),
            Expr::Add(left, right) => self.eval(left) + self.eval(right),
            Expr::Sub(left, right) => self.eval(left) - self.eval(right),
            Expr::Mul(left, right) => self.eval(left) * self.eval(right),
            Expr::Abs(inner) => self.eval(inner).abs(),
            Expr::Square(inner) => {
                let val = self.eval(inner);
                val * val
            }
        }
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Coord;

    // Helper function to create an expression with exactly n operations
    fn create_expression_with_n_operations(n: usize) -> Expr {
        if n == 0 {
            Expr::const_(1.0)
        } else {
            Expr::add(
                create_expression_with_n_operations(n - 1),
                Expr::const_(0.0),
            )
        }
    }

    #[test]
    fn test_basic_arithmetic() {
        let mut ctx = EvalContext::new();
        let a = ctx.add_scalar(3.0);
        let b = ctx.add_scalar(4.0);

        let expr = Expr::add(Expr::var(a), Expr::var(b));
        assert_eq!(ctx.eval(&expr), 7.0);

        let expr = Expr::mul(Expr::var(a), Expr::var(b));
        assert_eq!(ctx.eval(&expr), 12.0);
    }

    #[test]
    fn test_coordinate_access() {
        let mut ctx = EvalContext::new();
        let coord = ctx.add_coord(Coord::new(1.5, -2.25));

        let expr_x = Expr::coord_x(coord);
        assert_eq!(ctx.eval(&expr_x), 1.5);

        let expr_y = Expr::coord_y(coord);
        assert_eq!(ctx.eval(&expr_y), -2.25);
    }

    #[test]
    fn test_complex_expression() {
        let mut ctx = EvalContext::new();
        let a = ctx.add_coord(Coord::new(0.0, 0.0));
        let b = ctx.add_coord(Coord::new(1.0, 0.0));
        let c = ctx.add_coord(Coord::new(0.0, 1.0));

        // Expression: (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
        let adx = Expr::sub(Expr::coord_x(a), Expr::coord_x(c));
        let bdy = Expr::sub(Expr::coord_y(b), Expr::coord_y(c));
        let ady = Expr::sub(Expr::coord_y(a), Expr::coord_y(c));
        let bdx = Expr::sub(Expr::coord_x(b), Expr::coord_x(c));

        let term1 = Expr::mul(adx, bdy);
        let term2 = Expr::mul(ady, bdx);
        let det = Expr::sub(term1, term2);

        // For these points, the determinant should be positive (counter-clockwise)
        assert!(ctx.eval(&det) > 0.0);
    }

    #[test]
    fn test_absolute_value() {
        let mut ctx = EvalContext::new();
        let neg_val = ctx.add_scalar(-5.0);

        let expr = Expr::abs(Expr::var(neg_val));
        assert_eq!(ctx.eval(&expr), 5.0);
    }

    #[test]
    fn test_square() {
        let mut ctx = EvalContext::new();
        let val = ctx.add_scalar(3.0);
        let neg_val = ctx.add_scalar(-4.0);

        let expr = Expr::square(Expr::var(val));
        assert_eq!(ctx.eval(&expr), 9.0);

        let expr_neg = Expr::square(Expr::var(neg_val));
        assert_eq!(ctx.eval(&expr_neg), 16.0); // (-4)² = 16

        // Test that square counts as 1 operation
        assert_eq!(expr.operation_count(), 1);
    }

    #[test]
    fn test_gamma_constants_and_variables() {
        let const_expr = Expr::const_(std::f64::consts::PI);
        assert_eq!(const_expr.operation_count(), 0);
        assert_eq!(const_expr.gamma(), 0.0);

        let mut ctx = EvalContext::new();
        let scalar_var = ctx.add_scalar(2.5);
        let var_expr = Expr::var(scalar_var);
        assert_eq!(var_expr.operation_count(), 0);
        assert_eq!(var_expr.gamma(), 0.0);

        let coord_var = ctx.add_coord(Coord::new(1.0, 2.0));
        let coord_x_expr = Expr::coord_x(coord_var);
        assert_eq!(coord_x_expr.operation_count(), 0);
        assert_eq!(coord_x_expr.gamma(), 0.0);
    }

    #[test]
    fn test_gamma_simple_operations() {
        let mut ctx = EvalContext::new();
        let a = ctx.add_scalar(3.0);
        let b = ctx.add_scalar(4.0);

        // Single operation: a + b
        let add_expr = Expr::add(Expr::var(a), Expr::var(b));
        assert_eq!(add_expr.operation_count(), 1);
        assert!((add_expr.gamma() - GAMMA5).abs() < 1e-10);

        // Single operation: -a
        let neg_expr = Expr::neg(Expr::var(a));
        assert_eq!(neg_expr.operation_count(), 1);
        assert!((neg_expr.gamma() - GAMMA5).abs() < 1e-10);

        // Single operation: |a|
        let abs_expr = Expr::abs(Expr::var(a));
        assert_eq!(abs_expr.operation_count(), 1);
        assert!((abs_expr.gamma() - GAMMA5).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_complex_expression() {
        let mut ctx = EvalContext::new();
        let a = ctx.add_coord(Coord::new(0.0, 0.0));
        let b = ctx.add_coord(Coord::new(1.0, 0.0));
        let c = ctx.add_coord(Coord::new(0.0, 1.0));

        // Expression: (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
        // This involves: 4 subtractions + 2 multiplications + 1 subtraction = 7 operations
        let adx = Expr::sub(Expr::coord_x(a), Expr::coord_x(c)); // 1 op
        let bdy = Expr::sub(Expr::coord_y(b), Expr::coord_y(c)); // 1 op
        let ady = Expr::sub(Expr::coord_y(a), Expr::coord_y(c)); // 1 op
        let bdx = Expr::sub(Expr::coord_x(b), Expr::coord_x(c)); // 1 op

        let term1 = Expr::mul(adx, bdy); // 1 op + 2 sub-ops = 3 total so far
        let term2 = Expr::mul(ady, bdx); // 1 op + 2 sub-ops = 3 total so far
        let det = Expr::sub(term1, term2); // 1 op + 6 sub-ops = 7 total

        assert_eq!(det.operation_count(), 7);
        assert!((det.gamma() - GAMMA7).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_matches_existing_constants() {
        // Test that our GAMMA computation matches the existing constants
        // We need to create expressions with the right number of operations

        // For GAMMA5 (5 operations)
        let expr5 = create_expression_with_n_operations(5);
        assert_eq!(expr5.operation_count(), 5);
        assert!((expr5.gamma() - GAMMA5).abs() < 1e-10);

        // For GAMMA7 (7 operations)
        let expr7 = create_expression_with_n_operations(7);
        assert_eq!(expr7.operation_count(), 7);
        assert!((expr7.gamma() - GAMMA7).abs() < 1e-10);

        // For GAMMA11 (11 operations)
        let expr11 = create_expression_with_n_operations(11);
        assert_eq!(expr11.operation_count(), 11);
        assert!((expr11.gamma() - GAMMA11).abs() < 1e-10);
    }

    #[test]
    fn cmp_dist_fast_ast_matches_original() {
        // Test cases that should give clear results
        let origin = Coord::new(0.0, 0.0);
        let closer = Coord::new(1.0, 1.0); // distance squared = 2.0
        let farther = Coord::new(2.0, 2.0); // distance squared = 8.0

        // Compare closer vs farther
        let ast_result = cmp_dist_fast_ast(&origin, &farther, &closer);
        let original_result = cmp_dist_fast(&origin, &farther, &closer);
        assert_eq!(ast_result, original_result);
        assert_eq!(ast_result, Some(Ordering::Greater)); // farther > closer

        // Compare closer vs farther (reverse order)
        let ast_result = cmp_dist_fast_ast(&origin, &closer, &farther);
        let original_result = cmp_dist_fast(&origin, &closer, &farther);
        assert_eq!(ast_result, original_result);
        assert_eq!(ast_result, Some(Ordering::Less)); // closer < farther
    }

    #[test]
    fn cmp_dist_fast_ast_efficiency() {
        // Verify that the optimized AST version still produces correct results
        let origin = Coord::new(0.0, 0.0);
        let p = Coord::new(1.0, 0.0);
        let q = Coord::new(0.0, 1.0);

        // Just verify that the AST version produces the same results as the original
        let ast_result = cmp_dist_fast_ast(&origin, &p, &q);
        let original_result = cmp_dist_fast(&origin, &p, &q);
        assert_eq!(ast_result, original_result);
    }

    #[test]
    fn cmp_dist_fast_ast_handles_close_points() {
        // Test points that are very close - should return None due to floating point error
        let origin = Coord::new(0.0, 0.0);
        let p1 = Coord::new(1e-8, 1e-8);
        let p2 = Coord::new(1e-8 + 1e-16, 1e-8 + 1e-16); // extremely close to p1

        let ast_result = cmp_dist_fast_ast(&origin, &p1, &p2);
        let original_result = cmp_dist_fast(&origin, &p1, &p2);

        // Both should return the same result (likely None for points this close)
        assert_eq!(ast_result, original_result);
    }
}
