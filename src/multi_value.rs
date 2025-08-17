//! A simple library for creating and backpropagating through expression trees.
//!
//! This package includes the following elements to construct expression trees:
//!
//! - [`value::Expr`]: a node in the expression tree.
use crate::operations::{Operation, OperationType};
use std::collections::VecDeque;
use std::ops::{Add, Div, Mul, Sub};

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
pub struct MultiValueExpr<'a> {
    pub(crate) operand1: Option<&'a MultiValueExpr<'a>>,
    pub(crate) operand2: Option<&'a MultiValueExpr<'a>>,
    /// The operation applied to the operands, if any.
    pub(crate) operation: Operation,
    /// The numeric result of the expression, as result of applying the operation to the operands.
    pub value: Vec<f64>,
    /// The shape of the expression's result and expression gradients.
    pub shape: (usize, usize),
    /// Whether the expression is learnable or not. Only learnable [`Expr`] will have their values updated during backpropagation (learning).
    pub is_learnable: bool,
    pub(crate) grad: Vec<f64>,
}

#[allow(missing_docs)]
impl<'a> MultiValueExpr<'a> {
    pub fn new_leaf(value: Vec<f64>, shape: (usize, usize), is_learnable: bool) -> Self {
        Self {
            operand1: None,
            operand2: None,
            operation: Operation::None,
            value,
            shape,
            is_learnable,
            grad: vec![0.0; shape.0 * shape.1],
        }
    }

    fn get_unary_mapped_value(&self, operation: Operation) -> Vec<f64> {
        let lambda = operation.get_unary_operation_lambda();
        self.value.iter().map(lambda).collect()
    }

    fn get_binary_mapped_value(&self, other: &Self, operation: Operation) -> Vec<f64> {
        let lambda = operation.get_binary_operation_lambda();
        self.value
            .iter()
            .zip(other.value.iter())
            .map(lambda)
            .collect()
    }

    fn map_unary_operation(&'a self, operation: Operation) -> Self {
        let value = self.get_unary_mapped_value(operation);
        Self {
            operand1: Some(&self),
            operand2: None,
            operation,
            value,
            shape: self.shape,
            is_learnable: false,
            grad: vec![0.0; self.shape.0 * self.shape.1],
        }
    }

    fn map_binary_operation(&'a self, other: &'a Self, operation: Operation) -> Self {
        let value = self.get_binary_mapped_value(other, operation);
        Self {
            operand1: Some(self),
            operand2: Some(other),
            operation,
            value,
            shape: self.shape,
            is_learnable: false,
            grad: vec![0.0; self.shape.0 * self.shape.1],
        }
    }

    pub fn tanh(&'a self) -> Self {
        self.map_unary_operation(Operation::Tanh)
    }

    pub fn relu(&'a self) -> Self {
        self.map_unary_operation(Operation::ReLU)
    }

    pub fn exp(&'a self) -> Self {
        self.map_unary_operation(Operation::Exp)
    }

    pub fn log(&'a self) -> Self {
        self.map_unary_operation(Operation::Log)
    }

    pub fn neg(&'a self) -> Self {
        self.map_unary_operation(Operation::Neg)
    }

    pub fn pow(&'a self, exponent: &'a MultiValueExpr<'a>) -> Self {
        self.map_binary_operation(exponent, Operation::Pow)
    }
}

impl<'a> Add for &'a MultiValueExpr<'a> {
    type Output = MultiValueExpr<'a>;

    fn add(self: &'a MultiValueExpr<'a>, other: &'a MultiValueExpr<'a>) -> Self::Output {
        self.map_binary_operation(other, Operation::Add)
    }
}

impl<'a> Sub for &'a MultiValueExpr<'a> {
    type Output = MultiValueExpr<'a>;

    fn sub(self: &'a MultiValueExpr<'a>, other: &'a MultiValueExpr<'a>) -> Self::Output {
        self.map_binary_operation(other, Operation::Sub)
    }
}

impl<'a> Mul for &'a MultiValueExpr<'a> {
    type Output = MultiValueExpr<'a>;

    fn mul(self: &'a MultiValueExpr<'a>, other: &'a MultiValueExpr<'a>) -> Self::Output {
        self.map_binary_operation(other, Operation::Mul)
    }
}

impl<'a> Div for &'a MultiValueExpr<'a> {
    type Output = MultiValueExpr<'a>;

    fn div(self: &'a MultiValueExpr<'a>, other: &'a MultiValueExpr<'a>) -> Self::Output {
        self.map_binary_operation(other, Operation::Div)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_leaf() {
        let expr = MultiValueExpr::new_leaf(vec![1.0, 2.0], (2, 1), true);
        assert_eq!(expr.value, vec![1.0, 2.0]);
        assert_eq!(expr.shape, (2, 1));
        assert!(expr.is_learnable);
        assert_eq!(expr.grad, vec![0.0, 0.0]);
    }

    #[test]
    fn test_tanh() {
        let expr = MultiValueExpr::new_leaf(vec![0.0, 1.0], (2, 1), false);
        let tanh_expr = expr.tanh();
        assert_eq!(tanh_expr.value.len(), 2);
        // tanh(0.0) == 0.0, tanh(1.0) ~ 0.76159
        assert!((tanh_expr.value[0] - 0.0).abs() < 1e-6);
        assert!((tanh_expr.value[1] - 0.76159).abs() < 1e-4);
    }

    #[test]
    fn test_relu() {
        let expr = MultiValueExpr::new_leaf(vec![-1.0, 2.0], (2, 1), true);
        let relu_expr = expr.relu();
        assert_eq!(relu_expr.value, vec![0.0, 2.0]);
        // is_learnable is always false for relu output (see implementation)
        assert_eq!(relu_expr.is_learnable, false);
    }

    #[test]
    fn test_add() {
        let expr1 = MultiValueExpr::new_leaf(vec![1.0, 2.0], (2, 1), false);
        let expr2 = MultiValueExpr::new_leaf(vec![3.0, 4.0], (2, 1), false);
        let sum_expr = &expr1 + &expr2;
        assert_eq!(sum_expr.value, vec![4.0, 6.0]);
        assert_eq!(sum_expr.shape, (2, 1));
    }

    #[test]
    fn test_sub() {
        let expr1 = MultiValueExpr::new_leaf(vec![5.0, 7.0], (2, 1), false);
        let expr2 = MultiValueExpr::new_leaf(vec![2.0, 3.0], (2, 1), false);
        let sub_expr = &expr1 - &expr2;
        assert_eq!(sub_expr.value, vec![3.0, 4.0]);
    }

    #[test]
    fn test_mul() {
        let expr1 = MultiValueExpr::new_leaf(vec![2.0, 3.0], (2, 1), false);
        let expr2 = MultiValueExpr::new_leaf(vec![4.0, 5.0], (2, 1), false);
        let mul_expr = &expr1 * &expr2;
        assert_eq!(mul_expr.value, vec![8.0, 15.0]);
    }

    #[test]
    fn test_div() {
        let expr1 = MultiValueExpr::new_leaf(vec![8.0, 9.0], (2, 1), false);
        let expr2 = MultiValueExpr::new_leaf(vec![2.0, 3.0], (2, 1), false);
        let div_expr = &expr1 / &expr2;
        assert_eq!(div_expr.value, vec![4.0, 3.0]);
    }

    #[test]
    fn test_exp() {
        let expr = MultiValueExpr::new_leaf(vec![0.0, 1.0], (2, 1), false);
        let exp_expr = expr.exp();
        assert!((exp_expr.value[0] - 1.0).abs() < 1e-6); // exp(0) = 1
        assert!((exp_expr.value[1] - std::f64::consts::E).abs() < 1e-6); // exp(1) = e
    }

    #[test]
    fn test_log() {
        let expr = MultiValueExpr::new_leaf(vec![1.0, std::f64::consts::E], (2, 1), false);
        let log_expr = expr.log();
        assert!((log_expr.value[0] - 0.0).abs() < 1e-6); // log(1) = 0
        assert!((log_expr.value[1] - 1.0).abs() < 1e-6); // log(e) = 1
    }

    #[test]
    fn test_neg() {
        let expr = MultiValueExpr::new_leaf(vec![2.0, -3.0], (2, 1), false);
        let neg_expr = expr.neg();
        assert_eq!(neg_expr.value, vec![-2.0, 3.0]);
    }

    #[test]
    fn test_pow() {
        let base = MultiValueExpr::new_leaf(vec![2.0, 3.0], (2, 1), false);
        let exp = MultiValueExpr::new_leaf(vec![3.0, 2.0], (2, 1), false);
        let _pow_expr = base.pow(&exp);
    }
}
