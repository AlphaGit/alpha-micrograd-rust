//! A simple library for creating and backpropagating through expression trees.
//! 
//! This package includes the following elements to construct expression trees:
//! 
//! - [`value::Expr`]: a node in the expression tree.
#![deny(missing_docs)]
use std::collections::VecDeque;
use std::ops::{Add, Div, Mul, Sub};
use std::iter::Sum;
use crate::operations::{Operation, OperationType};

/// Expression representing a node in a calculation graph.
/// 
/// This struct represents a node in a calculation graph. It can be a leaf node, a unary operation or a binary operation.
/// 
/// A leaf node holds a value, which is the one that is used in the calculation.
/// 
/// A unary expression is the result of applying a unary operation to another expression. For example, the result of applying the `tanh` operation to a leaf node.
/// 
/// A binary expression is the result of applying a binary operation to two other expressions. For example, the result of adding two leaf nodes.
#[derive(Debug, Clone)]
pub struct Expr {
    pub(crate) operand1: Option<Box<Expr>>,
    pub(crate) operand2: Option<Box<Expr>>,
    /// The operation applied to the operands, if any.
    pub(crate) operation: Operation,
    /// The numeric result of the expression, as result of applying the operation to the operands.
    pub result: f64,
    /// Whether the expression is learnable or not. Only learnable [`Expr`] will have their values updated during backpropagation (learning).
    pub is_learnable: bool,
    pub(crate) grad: f64,
    /// The name of the expression, used to identify it in the calculation graph.
    pub name: Option<String>,
}

impl Expr {
    /// Creates a new leaf expression with the given value.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// ```
    pub fn new_leaf(value: f64) -> Expr {
        Expr {
            operand1: None,
            operand2: None,
            operation: Operation::None,
            result: value,
            is_learnable: true,
            grad: 0.0,
            name: None,
        }
    }

    /// Creates a new leaf expression with the given value and name.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf_with_name(1.0, "x");
    /// 
    /// assert_eq!(expr.name, Some("x".to_string()));
    /// ```
    pub fn new_leaf_with_name(value: f64, name: &str) -> Expr {
        let mut expr = Expr::new_leaf(value);
        expr.name = Some(name.to_string());
        expr
    }

    fn new_unary(operand: Expr, operation: Operation, result: f64) -> Expr {
        operation.assert_is_type(OperationType::Unary);
        Expr {
            operand1: Some(Box::new(operand)),
            operand2: None,
            operation,
            result,
            is_learnable: false,
            grad: 0.0,
            name: None,
        }
    }

    fn new_binary(operand1: Expr, operand2: Expr, operation: Operation, result: f64) -> Expr {
        operation.assert_is_type(OperationType::Binary);
        Expr {
            operand1: Some(Box::new(operand1)),
            operand2: Some(Box::new(operand2)),
            operation,
            result,
            is_learnable: false,
            grad: 0.0,
            name: None,
        }
    }

    /// Applies the hyperbolic tangent function to the expression and returns it as a new expression.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// let expr2 = expr.tanh();
    /// 
    /// assert_eq!(expr2.result, 0.7615941559557649);
    /// ```
    pub fn tanh(self) -> Expr {
        let result = self.result.tanh();
        Expr::new_unary(self, Operation::Tanh, result)
    }

    /// Applies the rectified linear unit function to the expression and returns it as a new expression.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(-1.0);
    /// let expr2 = expr.relu();
    /// 
    /// assert_eq!(expr2.result, 0.0);
    /// ```
    pub fn relu(self) -> Expr {
        let result = self.result.max(0.0);
        Expr::new_unary(self, Operation::ReLU, result)
    }

    /// Applies the exponential function (e^x) to the expression and returns it as a new expression.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// let expr2 = expr.exp();
    /// 
    /// assert_eq!(expr2.result, 2.718281828459045);
    /// ```
    pub fn exp(self) -> Expr {
        let result = self.result.exp();
        Expr::new_unary(self, Operation::Exp, result)
    }

    /// Raises the expression to the power of the given exponent (expression) and returns it as a new expression.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(2.0);
    /// let exponent = Expr::new_leaf(3.0);
    /// let result = expr.pow(exponent);
    /// 
    /// assert_eq!(result.result, 8.0);
    /// ```
    pub fn pow(self, exponent: Expr) -> Expr {
        let result = self.result.powf(exponent.result);
        Expr::new_binary(self, exponent, Operation::Pow, result)
    }

    /// Applies the natural logarithm function to the expression and returns it as a new expression.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(2.0);
    /// let expr2 = expr.log();
    /// 
    /// assert_eq!(expr2.result, 0.6931471805599453);
    /// ```
    pub fn log(self) -> Expr {
        let result = self.result.ln();
        Expr::new_unary(self, Operation::Log, result)
    }

    /// Negates the expression and returns it as a new expression.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// let expr2 = expr.neg();
    /// 
    /// assert_eq!(expr2.result, -1.0);
    /// ```
    pub fn neg(self) -> Expr {
        let result = -self.result;
        Expr::new_unary(self, Operation::Neg, result)
    }

    /// Recalculates the value of the expression recursively, from new values of the operands.
    /// 
    /// Usually will be used after a call to [`Expr::learn`], where the gradients have been calculated and
    /// the internal values of the expression tree have been updated.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// let mut expr2 = expr.tanh();
    /// expr2.learn(1e-09);
    /// expr2.recalculate();
    /// 
    /// assert_eq!(expr2.result, 0.7615941557793864);
    /// ```
    /// 
    /// You can also vary the values of the operands and recalculate the expression:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf_with_name(1.0, "x");
    /// let mut expr2 = expr.tanh();
    /// 
    /// let mut original = expr2.find_mut("x").expect("Could not find x");
    /// original.result = 2.0;
    /// expr2.recalculate();
    /// 
    /// assert_eq!(expr2.result, 0.9640275800758169);
    /// ```
    pub fn recalculate(&mut self) {
        // TODO: Since we can't borrow the operands mutably without inferring multible borrows from
        // the current node, this approach will need to stay recursive for now.
        // We can replace it with an iterative approach after we implement an allocation arena at the
        // tree level and then we can just visit them in a regular loop.
        match self.operation.expr_type() {
            OperationType::Leaf => {}
            OperationType::Unary => {
                let operand1 = self.operand1.as_mut().expect("Unary expression did not have an operand");
                operand1.recalculate();

                self.result = match self.operation {
                    Operation::Tanh => operand1.result.tanh(),
                    Operation::Exp => operand1.result.exp(),
                    Operation::ReLU => operand1.result.max(0.0),
                    Operation::Log => operand1.result.ln(),
                    Operation::Neg => -operand1.result,
                    _ => panic!("Invalid unary operation {:?}", self.operation),
                };
            }
            OperationType::Binary => {
                let operand1 = self.operand1.as_mut().expect("Binary expression did not have an operand");
                let operand2 = self.operand2.as_mut().expect("Binary expression did not have a second operand");

                operand1.recalculate();
                operand2.recalculate();

                self.result = match self.operation {
                    Operation::Add => operand1.result + operand2.result,
                    Operation::Sub => operand1.result - operand2.result,
                    Operation::Mul => operand1.result * operand2.result,
                    Operation::Div => operand1.result / operand2.result,
                    Operation::Pow => operand1.result.powf(operand2.result),
                    _ => panic!("Invalid binary operation: {:?}", self.operation),
                };
            }
        }
    }

    /// Applies backpropagation to the expression, updating the values of the
    /// gradients and the expression itself.
    /// 
    /// This method will change the gradients based on the gradient of the last
    /// expression in the calculation graph.
    /// 
    /// Example:
    /// 
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// let mut expr2 = expr.tanh();
    /// expr2.learn(1e-09);
    /// ```
    /// 
    /// After adjusting the gradients, the method will update the values of the
    /// individual expression tree nodes to minimize the loss function.
    /// 
    /// In order to get a new calculation of the expression tree, you'll need to call
    /// [`Expr::recalculate`] after calling [`Expr::learn`].
    pub fn learn(&mut self, learning_rate: f64) {
        self.grad = 1.0;

        let mut queue = VecDeque::from([self]);

        while let Some(node) = queue.pop_front() {
            match node.operation.expr_type() {
                OperationType::Leaf => {
                    node.learn_internal_leaf(learning_rate);
                }
                OperationType::Unary => {
                    let operand1 = node.operand1.as_mut().expect("Unary expression did not have an operand");
                    operand1.adjust_grad_unary(&node.operation, node.grad, node.result);
                    queue.push_back(operand1);
                }
                OperationType::Binary => {
                    let operand1 = node.operand1.as_mut().expect("Binary expression did not have an operand");
                    let operand2 = node.operand2.as_mut().expect("Binary expression did not have a second operand");

                    operand1.adjust_grad_binary_op1(&node.operation, node.grad, operand2);
                    operand2.adjust_grad_binary_op2(&node.operation, node.grad, operand1);

                    queue.push_back(operand1);
                    queue.push_back(operand2);
                }
            }
        }
    }

    fn learn_internal_leaf(&mut self, learning_rate: f64) {
        // leaves have their gradient set externally by other nodes in the tree
        // leaves can be learnable, in which case we update the value
        if self.is_learnable {
            self.result -= learning_rate * self.grad;
        }
    }

    fn adjust_grad_unary(&mut self, child_operation: &Operation, child_grad: f64, child_result: f64) {
        match child_operation {
            Operation::Tanh => {
                let tanh_grad = 1.0 - (child_result * child_result);
                self.grad = child_grad * tanh_grad;
            }
            Operation::Exp => {
                self.grad = child_grad * child_result;
            }
            Operation::ReLU => {
                self.grad = child_grad * if child_result > 0.0 { 1.0 } else { 0.0 };
            }
            Operation::Log => {
                self.grad = child_grad / child_result;
            }
            Operation::Neg => {
                self.grad = -child_grad;
            }
            _ => panic!("Invalid unary operation {child_operation:?}"),
        }
    }

    fn adjust_grad_binary_op1(&mut self, child_operation: &Operation, child_grad: f64, operand2: &Expr) {
        match child_operation {
            Operation::Add => {
                self.grad = child_grad;
            }
            Operation::Sub => {
                self.grad = child_grad;
            }
            Operation::Mul => {
                let operand2_result = operand2.result;

                self.grad = child_grad * operand2_result;
            }
            Operation::Div => {
                let operand2_result = operand2.result;

                self.grad = child_grad / operand2_result;
            }
            Operation::Pow => {
                let exponent = operand2.result;
                let base = self.result;

                self.grad = child_grad * exponent * base.powf(exponent - 1.0);
            }
            _ => panic!("Invalid binary operation: {child_operation:?}"),
        }
    }

    fn adjust_grad_binary_op2(&mut self,child_operation: &Operation, child_grad: f64, operand1: &Expr) {
        match child_operation {
            Operation::Add => {
                self.grad = child_grad;
            }
            Operation::Sub => {
                self.grad = -child_grad;
            }
            Operation::Mul => {
                let operand1_result = operand1.result;
                self.grad = child_grad * operand1_result;
            }
            Operation::Div => {
                let operand2_result = self.result;
                let operand1_result = operand1.result;

                self.grad = -child_grad * operand1_result / (operand2_result * operand2_result);
            }
            Operation::Pow => {
                let exponent = self.result;
                let base = operand1.result;

                self.grad = child_grad * base.powf(exponent) * base.ln();
            }
            _ => panic!("Invalid binary operation: {child_operation:?}"),
        }
    }

    /// Finds a node in the expression tree by its name.
    /// 
    /// This method will search the expression tree for a node with the given name.
    /// If the node is not found, it will return [None].
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf_with_name(1.0, "x");
    /// let expr2 = expr.tanh();
    /// let original = expr2.find("x");
    /// 
    /// assert_eq!(original.expect("Could not find x").result, 1.0);
    /// ```
    pub fn find(&self, name: &str) -> Option<&Expr> {
        let mut stack = vec![self];

        while let Some(node) = stack.pop() {
            if node.name == Some(name.to_string()) {
                return Some(node);
            }

            if let Some(operand1) = node.operand1.as_ref() {
                stack.push(operand1);
            }
            if let Some(operand2) = node.operand2.as_ref() {
                stack.push(operand2);
            }
        }

        None
    }

    /// Finds a node in the expression tree by its name and returns a mutable reference to it.
    /// 
    /// This method will search the expression tree for a node with the given name.
    /// If the node is not found, it will return [None].
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf_with_name(1.0, "x");
    /// let mut expr2 = expr.tanh();
    /// let mut original = expr2.find_mut("x").expect("Could not find x");
    /// original.result = 2.0;
    /// expr2.recalculate();
    /// 
    /// assert_eq!(expr2.result, 0.9640275800758169);
    /// ```
    pub fn find_mut(&mut self, name: &str) -> Option<&mut Expr> {
        let mut stack = vec![self];

        while let Some(node) = stack.pop() {
            if node.name == Some(name.to_string()) {
                return Some(node);
            }

            if let Some(operand1) = node.operand1.as_mut() {
                stack.push(operand1);
            }
            if let Some(operand2) = node.operand2.as_mut() {
                stack.push(operand2);
            }
        }

        None
    }

    /// Returns the count of nodes (parameters)in the expression tree.
    /// 
    /// This method will return the total number of nodes in the expression tree,
    /// including the root node.
    /// 
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::value::Expr;
    /// 
    /// let expr = Expr::new_leaf(1.0);
    /// let expr2 = expr.tanh();
    /// 
    /// assert_eq!(expr2.parameter_count(false), 2);
    /// assert_eq!(expr2.parameter_count(true), 1);
    /// ```
    pub fn parameter_count(&self, learnable_only: bool) -> usize {
        let mut stack = vec![self];
        let mut count = 0;

        while let Some(node) = stack.pop() {
            if node.is_learnable || !learnable_only {
                count += 1;
            }

            if let Some(operand1) = node.operand1.as_ref() {
                stack.push(operand1);
            }
            if let Some(operand2) = node.operand2.as_ref() {
                stack.push(operand2);
            }
        }

        count
    }
}

/// Implements the [`Add`] trait for the [`Expr`] struct.
/// 
/// This implementation allows the addition of two [`Expr`] objects.
/// 
/// Example:
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr + expr2;
/// 
/// assert_eq!(result.result, 3.0);
/// ```
impl Add for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        let result = self.result + other.result;
        Expr::new_binary(self, other, Operation::Add, result)
    }
}

/// Implements the [`Add`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the addition of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr + 2.0;
/// 
/// assert_eq!(result.result, 3.0);
/// ```
impl Add<f64> for Expr {
    type Output = Expr;

    fn add(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self + operand2
    }
}

/// Implements the [`Add`] trait for the [`f64`] type, when the right operand is an [`Expr`].
/// 
/// This implementation allows the addition of a [`f64`] value and an [`Expr`] object.
/// 
/// Example:
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = 2.0 + expr;
/// 
/// assert_eq!(result.result, 3.0);
/// ```
impl Add<Expr> for f64 {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        let operand1 = Expr::new_leaf(self);
        operand1 + other
    }
}

/// Implements the [`Mul`] trait for the [`Expr`] struct.
/// 
/// This implementation allows the multiplication of two [`Expr`] objects.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr * expr2;
/// 
/// assert_eq!(result.result, 2.0);
/// ```
impl Mul for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        let result = self.result * other.result;
        Expr::new_binary(self, other, Operation::Mul, result)
    }
}

/// Implements the [`Mul`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the multiplication of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr * 2.0;
/// 
/// assert_eq!(result.result, 2.0);
/// ```
impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self * operand2
    }
}

/// Implements the [`Mul`] trait for the [`f64`] type, when the right operand is an [`Expr`].
/// 
/// This implementation allows the multiplication of a [`f64`] value and an [`Expr`] object.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = 2.0 * expr;
/// 
/// assert_eq!(result.result, 2.0);
/// ```
impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        let operand1 = Expr::new_leaf(self);
        operand1 * other
    }
}

/// Implements the [`Sub`] trait for the [`Expr`] struct.
/// 
/// This implementation allows the subtraction of two [`Expr`] objects.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr - expr2;
/// 
/// assert_eq!(result.result, -1.0);
/// ```
impl Sub for Expr {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        let result = self.result - other.result;
        Expr::new_binary(self, other, Operation::Sub, result)
    }
}

/// Implements the [`Sub`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the subtraction of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr - 2.0;
/// 
/// assert_eq!(result.result, -1.0);
/// ```
impl Sub<f64> for Expr {
    type Output = Expr;

    fn sub(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self - operand2
    }
}

/// Implements the [`Sub`] trait for the [`f64`] type, when the right operand is an [`Expr`].
/// 
/// This implementation allows the subtraction of a [`f64`] value and an [`Expr`] object.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = 2.0 - expr;
/// 
/// assert_eq!(result.result, 1.0);
/// ```
impl Sub<Expr> for f64 {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        let operand1 = Expr::new_leaf(self);
        operand1 - other
    }
}

/// Implements the [`Div`] trait for the [`Expr`] struct.
/// 
/// This implementation allows the division of two [`Expr`] objects.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr / expr2;
/// 
/// assert_eq!(result.result, 0.5);
/// ```
impl Div for Expr {
    type Output = Expr;

    fn div(self, other: Expr) -> Expr {
        let result = self.result / other.result;
        Expr::new_binary(self, other, Operation::Div, result)
    }
}

/// Implements the [`Div`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the division of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr / 2.0;
/// 
/// assert_eq!(result.result, 0.5);
/// ```
impl Div<f64> for Expr {
    type Output = Expr;

    fn div(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self / operand2
    }
}

/// Implements the [`Sum`] trait for the [`Expr`] struct.
/// 
/// Note that this implementation will generate temporary [`Expr`] objects,
/// which may not be the most efficient way to sum a collection of [`Expr`] objects.
/// However, it is provided as a convenience method for users that want to use sum
/// over an [`Iterator<Expr>`].
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::value::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// let expr3 = Expr::new_leaf(3.0);
/// 
/// let sum = vec![expr, expr2, expr3].into_iter().sum::<Expr>();
/// 
/// assert_eq!(sum.result, 6.0);
/// ```
impl Sum for Expr {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|acc, x| acc + x)
            .unwrap_or(Expr::new_leaf(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_eq(f1: f64, f2: f64) {
        let places = 7;
        let tolerance = 10.0_f64.powi(-places);
        assert!((f1 - f2).abs() < tolerance, "{} != {} (tol: {})", f1, f2, tolerance);
    }

    #[test]
    fn test() {
        let expr = Expr::new_leaf(1.0);
        assert_eq!(expr.result, 1.0);
    }

    #[test]
    fn test_unary() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_unary(expr, Operation::Tanh, 1.1);

        assert_eq!(expr2.result, 1.1);
        assert_eq!(expr2.operand1.unwrap().result, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_unary_expression_type_check() {
        let expr = Expr::new_leaf(1.0);
        let _expr2 = Expr::new_unary(expr, Operation::Add, 1.1);
    }

    #[test]
    fn test_binary() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let expr3 = Expr::new_binary(expr, expr2, Operation::Add, 1.1);

        assert_eq!(expr3.result, 1.1);
        assert_eq!(expr3.operand1.unwrap().result, 1.0);
        assert_eq!(expr3.operand2.unwrap().result, 2.0);
    }

    #[test]
    #[should_panic]
    fn test_binary_expression_type_check() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let _expr3 = Expr::new_binary(expr, expr2, Operation::Tanh, 3.0);
    }

    #[test]
    fn test_mixed_tree() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let expr3 = Expr::new_binary(expr, expr2, Operation::Sub, 1.1);
        let expr4 = Expr::new_unary(expr3, Operation::Tanh, 1.2);

        assert_eq!(expr4.result, 1.2);
        let expr3 = expr4.operand1.unwrap();
        assert_eq!(expr3.result, 1.1);
        assert_eq!(expr3.operand1.unwrap().result, 1.0);
        assert_eq!(expr3.operand2.unwrap().result, 2.0);
    }

    #[test]
    fn test_tanh() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.tanh();

        assert_eq!(expr2.result, 0.7615941559557649);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 1.0);
        assert_eq!(expr2.operation, Operation::Tanh);
        assert!(expr2.operand2.is_none());

        // Some other known values
        fn get_tanh(x: f64) -> f64 {
            Expr::new_leaf(x).tanh().result
        }

        assert_float_eq(get_tanh(10.74), 0.9999999);
        assert_float_eq(get_tanh(-10.74), -0.9999999);
        assert_float_eq(get_tanh(0.0), 0.0);
    }

    #[test]
    fn test_exp() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.exp();

        assert_eq!(expr2.result, 2.718281828459045);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 1.0);
        assert_eq!(expr2.operation, Operation::Exp);
        assert!(expr2.operand2.is_none());
    }

    #[test]
    fn test_relu() {
        // negative case
        let expr = Expr::new_leaf(-1.0);
        let expr2 = expr.relu();

        assert_eq!(expr2.result, 0.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, -1.0);
        assert_eq!(expr2.operation, Operation::ReLU);
        assert!(expr2.operand2.is_none());

        // positive case
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.relu();

        assert_eq!(expr2.result, 1.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 1.0);
        assert_eq!(expr2.operation, Operation::ReLU);
        assert!(expr2.operand2.is_none());
    }

    #[test]
    fn test_pow() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = Expr::new_leaf(3.0);
        let result = expr.pow(expr2);

        assert_eq!(result.result, 8.0);
        assert!(result.operand1.is_some());
        assert_eq!(result.operand1.unwrap().result, 2.0);
        assert_eq!(result.operation, Operation::Pow);
        
        assert!(result.operand2.is_some());
        assert_eq!(result.operand2.unwrap().result, 3.0);
    }

    #[test]
    fn test_add() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let expr3 = expr + expr2;

        assert_eq!(expr3.result, 3.0);
        assert!(expr3.operand1.is_some());
        assert_eq!(expr3.operand1.unwrap().result, 1.0);
        assert!(expr3.operand2.is_some());
        assert_eq!(expr3.operand2.unwrap().result, 2.0);
        assert_eq!(expr3.operation, Operation::Add);
    }

    #[test]
    fn test_add_f64() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr + 2.0;

        assert_eq!(expr2.result, 3.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 1.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 2.0);
        assert_eq!(expr2.operation, Operation::Add);
    }

    #[test]
    fn test_add_f64_expr() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = 2.0 + expr;

        assert_eq!(expr2.result, 3.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 2.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 1.0);
        assert_eq!(expr2.operation, Operation::Add);
    }

    #[test]
    fn test_mul() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = Expr::new_leaf(3.0);
        let expr3 = expr * expr2;

        assert_eq!(expr3.result, 6.0);
        assert!(expr3.operand1.is_some());
        assert_eq!(expr3.operand1.unwrap().result, 2.0);
        assert!(expr3.operand2.is_some());
        assert_eq!(expr3.operand2.unwrap().result, 3.0);
        assert_eq!(expr3.operation, Operation::Mul);
    }

    #[test]
    fn test_mul_f64() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr * 3.0;

        assert_eq!(expr2.result, 6.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 2.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 3.0);
        assert_eq!(expr2.operation, Operation::Mul);
    }

    #[test]
    fn test_mul_f64_expr() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = 3.0 * expr;

        assert_eq!(expr2.result, 6.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 3.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 2.0);
        assert_eq!(expr2.operation, Operation::Mul);
    }

    #[test]
    fn test_sub() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = Expr::new_leaf(3.0);
        let expr3 = expr - expr2;

        assert_eq!(expr3.result, -1.0);
        assert!(expr3.operand1.is_some());
        assert_eq!(expr3.operand1.unwrap().result, 2.0);
        assert!(expr3.operand2.is_some());
        assert_eq!(expr3.operand2.unwrap().result, 3.0);
        assert_eq!(expr3.operation, Operation::Sub);
    }

    #[test]
    fn test_sub_f64() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr - 3.0;

        assert_eq!(expr2.result, -1.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 2.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 3.0);
        assert_eq!(expr2.operation, Operation::Sub);
    }

    #[test]
    fn test_sub_f64_expr() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = 3.0 - expr;

        assert_eq!(expr2.result, 1.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 3.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 2.0);
        assert_eq!(expr2.operation, Operation::Sub);
    }

    #[test]
    fn test_div() {
        let expr = Expr::new_leaf(6.0);
        let expr2 = Expr::new_leaf(3.0);
        let expr3 = expr / expr2;

        assert_eq!(expr3.result, 2.0);
        assert!(expr3.operand1.is_some());
        assert_eq!(expr3.operand1.unwrap().result, 6.0);
        assert!(expr3.operand2.is_some());
        assert_eq!(expr3.operand2.unwrap().result, 3.0);
        assert_eq!(expr3.operation, Operation::Div);
    }

    #[test]
    fn test_div_f64() {
        let expr = Expr::new_leaf(6.0);
        let expr2 = expr / 3.0;

        assert_eq!(expr2.result, 2.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 6.0);
        assert!(expr2.operand2.is_some());
        assert_eq!(expr2.operand2.unwrap().result, 3.0);
        assert_eq!(expr2.operation, Operation::Div);
    }

    #[test]
    fn test_log() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr.log();

        assert_eq!(expr2.result, 0.6931471805599453);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 2.0);
        assert_eq!(expr2.operation, Operation::Log);
        assert!(expr2.operand2.is_none());
    }

    #[test]
    fn test_neg() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr.neg();

        assert_eq!(expr2.result, -2.0);
        assert!(expr2.operand1.is_some());
        assert_eq!(expr2.operand1.unwrap().result, 2.0);
        assert_eq!(expr2.operation, Operation::Neg);
        assert!(expr2.operand2.is_none());
    }

    #[test]
    fn test_backpropagation_add() {
        let operand1 = Expr::new_leaf(1.0);
        let operand2 = Expr::new_leaf(2.0);
        let mut expr3 = operand1 + operand2;

        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 1.0);
        assert_eq!(operand2.grad, 1.0);
    }

    #[test]
    fn test_backpropagation_sub() {
        let operand1 = Expr::new_leaf(1.0);
        let operand2 = Expr::new_leaf(2.0);
        let mut expr3 = operand1 - operand2;

        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 1.0);
        assert_eq!(operand2.grad, -1.0);
    }

    #[test]
    fn test_backpropagation_mul() {
        let operand1 = Expr::new_leaf(3.0);
        let operand2 = Expr::new_leaf(4.0);
        let mut expr3 = operand1 * operand2;

        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 4.0);
        assert_eq!(operand2.grad, 3.0);
    }

    #[test]
    fn test_backpropagation_div() {
        let operand1 = Expr::new_leaf(3.0);
        let operand2 = Expr::new_leaf(4.0);
        let mut expr3 = operand1 / operand2;

        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 0.25);
        assert_eq!(operand2.grad, -0.1875);
    }

    #[test]
    fn test_backpropagation_tanh() {
        let operand1 = Expr::new_leaf(0.0);
        let mut expr2 = operand1.tanh();

        expr2.learn(1e-09);

        let operand1 = expr2.operand1.unwrap();
        assert_eq!(operand1.grad, 1.0);
    }

    #[test]
    fn test_backpropagation_relu() {
        let operand1 = Expr::new_leaf(-1.0);
        let mut expr2 = operand1.relu();

        expr2.learn(1e-09);

        let operand1 = expr2.operand1.unwrap();
        assert_eq!(operand1.grad, 0.0);
    }

    #[test]
    fn test_backpropagation_exp() {
        let operand1 = Expr::new_leaf(0.0);
        let mut expr2 = operand1.exp();

        expr2.learn(1e-09);

        let operand1 = expr2.operand1.unwrap();
        assert_eq!(operand1.grad, 1.0);
    }

    #[test]
    fn test_backpropagation_pow() {
        let operand1 = Expr::new_leaf(2.0);
        let operand2 = Expr::new_leaf(3.0);
        let mut expr3 = operand1.pow(operand2);

        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 12.0);
        assert_eq!(operand2.grad, 5.545177444479562);
    }

    #[test]
    fn test_backpropagation_mixed_tree() {
        let operand1 = Expr::new_leaf(1.0);
        let operand2 = Expr::new_leaf(2.0);
        let expr3 = operand1 + operand2;
        let mut expr4 = expr3.tanh();

        expr4.learn(1e-09);

        let expr3 = expr4.operand1.unwrap();
        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();

        assert_eq!(expr3.grad, 0.009866037165440211);
        assert_eq!(operand1.grad, 0.009866037165440211);
        assert_eq!(operand2.grad, 0.009866037165440211);
    }

    #[test]
    fn test_backpropagation_karpathys_example() {
        let x1 = Expr::new_leaf(2.0);
        let x2 = Expr::new_leaf(0.0);
        let w1 = Expr::new_leaf(-3.0);
        let w2 = Expr::new_leaf(1.0);
        let b = Expr::new_leaf(6.8813735870195432);

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1_x2w2 = x1w1 + x2w2;
        let n = x1w1_x2w2 + b;
        let mut o = n.tanh();

        o.learn(1e-09);

        assert_eq!(o.operation, Operation::Tanh);
        assert_eq!(o.grad, 1.0);

        let n = o.operand1.unwrap();
        assert_eq!(n.operation, Operation::Add);
        assert_float_eq(n.grad, 0.5);

        let x1w1_x2w2 = n.operand1.unwrap();
        assert_eq!(x1w1_x2w2.operation, Operation::Add);
        assert_float_eq(x1w1_x2w2.grad, 0.5);

        let b = n.operand2.unwrap();
        assert_eq!(b.operation, Operation::None);
        assert_float_eq(b.grad, 0.5);

        let x1w1 = x1w1_x2w2.operand1.unwrap();
        assert_eq!(x1w1.operation, Operation::Mul);
        assert_float_eq(x1w1.grad, 0.5);

        let x2w2 = x1w1_x2w2.operand2.unwrap();
        assert_eq!(x2w2.operation, Operation::Mul);
        assert_float_eq(x2w2.grad, 0.5);

        let x1 = x1w1.operand1.unwrap();
        assert_eq!(x1.operation, Operation::None);
        assert_float_eq(x1.grad, -1.5);

        let w1 = x1w1.operand2.unwrap();
        assert_eq!(w1.operation, Operation::None);
        assert_float_eq(w1.grad, 1.0);

        let x2 = x2w2.operand1.unwrap();
        assert_eq!(x2.operation, Operation::None);
        assert_float_eq(x2.grad, 0.5);

        let w2 = x2w2.operand2.unwrap();
        assert_eq!(w2.operation, Operation::None);
        assert_float_eq(w2.grad, 0.0);
    }

    #[test]
    fn test_learn_simple() {
        let mut expr = Expr::new_leaf(1.0);
        expr.learn(1e-01);

        assert_float_eq(expr.result, 0.9);
    }

    #[test]
    fn test_learn_skips_non_learnable() {
        let mut expr = Expr::new_leaf(1.0);
        expr.is_learnable = false;
        expr.learn(1e-01);

        assert_float_eq(expr.result, 1.0);
    }

    #[test]
    fn test_find_simple() {
        let expr = Expr::new_leaf_with_name(1.0, "x");
        let expr2 = expr.tanh();

        let found = expr2.find("x");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, Some("x".to_string()));
    }

    #[test]
    fn test_find_not_found() {
        let expr = Expr::new_leaf_with_name(1.0, "x");
        let expr2 = expr.tanh();

        let found = expr2.find("y");
        assert!(found.is_none());
    }

    #[test]
    fn test_sum_iterator() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let expr3 = Expr::new_leaf(3.0);

        let sum: Expr = vec![expr, expr2, expr3].into_iter().sum::<Expr>();
        assert_eq!(sum.result, 6.0);
    }

    #[test]
    fn test_find_after_clone() {
        let expr = Expr::new_leaf_with_name(1.0, "x");
        let expr2 = expr.tanh();
        let expr2_clone = expr2.clone();

        let found = expr2_clone.find("x");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, Some("x".to_string()));
    }
}