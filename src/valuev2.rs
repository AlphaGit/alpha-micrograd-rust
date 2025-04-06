//! A simple library for creating and backpropagating through expression trees.
//! 
//! This package includes the following elements to construct expression trees:
//! - [`Expr`]: a node in the expression tree
#![deny(missing_docs)]
use std::collections::{HashMap, VecDeque};
use std::fmt::{Formatter, Display};
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Sub};
use std::iter::Sum;

#[derive(Debug, Clone, PartialEq)]
enum Operation {
    None,
    Add,
    Sub,
    Mul,
    Div,
    Tanh,
    Exp,
    Pow,
    ReLU,
    Log,
    Neg,
}

impl Operation {
    fn assert_is_type(&self, expr_type: ExprType) {
        match self {
            Operation::None => assert_eq!(expr_type, ExprType::Leaf),
            Operation::Tanh | Operation::Exp | Operation::ReLU | Operation::Log | Operation::Neg => assert_eq!(expr_type, ExprType::Unary),
            _ => assert_eq!(expr_type, ExprType::Binary),
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let op = match self {
            Operation::None => "(Constant)",
            Operation::Add => "Addition",
            Operation::Sub => "Subtraction",
            Operation::Mul => "Multiplication",
            Operation::Div => "Division",
            Operation::Tanh => "tanh",
            Operation::Exp => "exp",
            Operation::Pow => "Power",
            Operation::ReLU => "ReLU",
            Operation::Log => "log",
            Operation::Neg => "Negation",
        };
        write!(f, "{}", op)
    }
}

#[derive(Debug, PartialEq)]
enum ExprType {
    Leaf,
    Unary,
    Binary,
}

/// Expression representing a calculation graph.
/// 
/// This struct represents a whole calculation graph, even if it contains a single element.
#[derive(Debug)]
pub struct Expr {
    tree: Vec<Option<ExprNode>>,
    names: HashMap<String, usize>,
}

#[derive(Debug)]
struct ExprNode {
    operation: Operation,
    result: f64,
    is_learnable: bool,
    grad: f64
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
        let mut tree = Vec::with_capacity(1);
        tree[0] = Some(ExprNode {
            operation: Operation::None,
            result: value,
            is_learnable: true,
            grad: 0.0
        });
        Expr {
            tree,
            names: HashMap::new(),
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
        expr.names.insert(name.to_string(), 0);
        expr
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
        let root_value = self.tree[0].expect("root cannot be None").result;
        let result = root_value.tanh();

        let new_root = ExprNode {
            operation: Operation::Tanh,
            result,
            is_learnable: false,
            grad: 0.0
        };
    }
}