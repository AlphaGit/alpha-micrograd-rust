//! A simple library for creating and backpropagating through expression trees.
//! 
//! This package includes the following elements to construct expression trees:
//! - [`Expr`]: a node in the expression tree
#![deny(missing_docs)]
use std::collections::HashMap;
use std::fmt::{Formatter, Display};
use std::ops::{Add, Div, Mul, Sub};
use std::iter::Sum;

/// Represents the operation type of an expression node.
///
/// This enum defines the types of operations that can be performed on expression nodes.
/// It includes binary operations (addition, subtraction, multiplication, division),
/// unary operations (tanh, exp, ReLU, log, negation), and a None operation for leaf nodes.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Operation {
    /// No operation, used for leaf nodes
    /// in the expression tree.
    None,
    /// Addition operation.
    Add,
    /// Subtraction operation.
    Sub,
    /// Multiplication operation.
    Mul,
    /// Division operation.
    Div,
    /// Hyperbolic tangent operation.
    Tanh,
    /// Exponential operation.
    Exp,
    /// Power operation.
    Pow,
    /// Rectified Linear Unit operation.
    ReLU,
    /// Logarithm operation.
    Log,
    /// Negation operation (-x).
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
        tree.push(Some(ExprNode {
            operation: Operation::None,
            result: value,
            is_learnable: true,
            grad: 0.0
        }));
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

    /// Returns the (current) value of the expression.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::Expr;
    ///
    /// let expr = Expr::new_leaf(1.0);
    /// assert_eq!(expr.result(), 1.0);
    /// ```
    pub fn result(&self) -> f64 {
        self.tree
            .last()
            .expect("tree cannot be empty")
            .as_ref()
            .expect("root cannot be None")
            .result
    }

    /// Returns the (current) operation of the expression.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::{Expr, Operation};
    ///
    /// let expr = Expr::new_leaf(1.0);
    /// assert_eq!(expr.operation(), Operation::None);
    /// ```
    pub fn operation(&self) -> Operation {
        self.tree
            .last()
            .expect("tree cannot be empty")
            .as_ref()
            .expect("root cannot be None")
            .operation
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
        let current_value = self.tree[0].as_ref().expect("root cannot be None").result;
        let result = current_value.tanh();

        let new_root = ExprNode {
            operation: Operation::Tanh,
            result,
            is_learnable: false,
            grad: 0.0
        };

        add_new_root(self, new_root)
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
        let current_value = self.tree[0].as_ref().expect("root cannot be None").result;
        let result = current_value.max(0.0);
        let new_root = ExprNode {
            operation: Operation::ReLU,
            result,
            is_learnable: false,
            grad: 0.0
        };
        add_new_root(self, new_root)
    }
}

fn add_new_root(subtree_a: Expr, new_root: ExprNode) -> Expr {
    let subtree_b: Expr = Expr {
        tree: vec![None],
        names: HashMap::new(),
    };

    merge_trees(subtree_a, subtree_b, new_root)
}

fn merge_trees(tree_a: Expr, tree_b: Expr, new_root: ExprNode) -> Expr {
    let mut subtree_a = tree_a.tree;
    let mut subtree_b = tree_b.tree;

    // levels are 1-based, level 1 is the root
    let len_a = subtree_a.len();
    let len_b = subtree_b.len();
    assert!(len_a > 0, "subtree_a must not be empty");
    assert!(len_b > 0, "subtree_b must not be empty");
    let levels_a = (len_a + 1).ilog2();
    let levels_b = (len_b + 1).ilog2();

    let final_level_number = levels_a.max(levels_b) + 1; // +1 for the new root
    let final_node_count = 2_usize.pow(final_level_number) - 1;
    dbg!(levels_a, levels_b, final_level_number, final_node_count);

    let mut new_tree = Vec::with_capacity(final_node_count);
    let mut new_names = HashMap::new();

    let mut original_tree_section_end_index = 0;
    let mut new_tree_section_start_index = 0;
    for level_number in (2..=final_level_number).rev() {
        // simplifying 2^(l-1)/2 to 2^(l-2) creates a subtraction overflow for l=1
        let nodes_per_tree = 2_usize.pow(level_number - 1) / 2;
        original_tree_section_end_index += nodes_per_tree;
        dbg!(level_number, nodes_per_tree, original_tree_section_end_index);

        // extract nodes for this level for tree_b (right side)
        let mut nodes_b = if levels_b + 1 >= level_number {
            subtree_b.drain(..nodes_per_tree).collect()
        } else {
            (0..nodes_per_tree).map(|_| None).collect::<Vec<Option<ExprNode>>>()
        };
        new_tree.append(&mut nodes_b);

        tree_b.names.iter()
            .filter(|(_, &index)| index < original_tree_section_end_index)
            .for_each(|(name, old_index)| {
                let new_index = old_index + new_tree_section_start_index;
                new_names.insert(name.clone(), new_index);
            });

        let mut nodes_a = if levels_a + 1 >= level_number {
            subtree_a.drain(..nodes_per_tree).collect()
        } else {
            (0..nodes_per_tree).map(|_| None).collect::<Vec<Option<ExprNode>>>()
        };
        new_tree.append(&mut nodes_a);

        tree_a.names.iter()
            .filter(|(_, &index)| index < original_tree_section_end_index)
            .map(|(name, old_index)| (name.clone(), old_index + new_tree_section_start_index + nodes_per_tree))
            .for_each(|(name, new_index)| {
                new_names.insert(name, new_index);
            });

        new_tree_section_start_index += nodes_per_tree * 2;
    }

    new_tree.push(Some(new_root));

    assert_eq!(new_tree.len(), final_node_count, "new_tree should have {} nodes, but has {}", final_node_count, new_tree.len());
    assert_eq!(new_tree.capacity(), final_node_count, "new_tree should have {} capacity, but has {}", final_node_count, new_tree.capacity());

    Expr {
        tree: new_tree,
        names: new_names,
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
    fn test_add_new_root() {
        let tree_a = Expr::new_leaf(1.0);

        assert_eq!(tree_a.tree.len(), 1);

        let new_root = ExprNode {
            operation: Operation::Mul,
            result: 3.0,
            is_learnable: true,
            grad: 0.0
        };

        let merged_tree = add_new_root(tree_a, new_root);
        assert_eq!(merged_tree.tree.len(), 3);
        assert!(merged_tree.tree[0].is_none());
        assert_eq!(merged_tree.tree[1].as_ref().unwrap().result, 1.0);
        assert_eq!(merged_tree.tree[2].as_ref().unwrap().result, 3.0);
    }

    #[test]
    fn test_add_new_root_with_name() {
        let tree_a = Expr::new_leaf_with_name(1.0, "x");

        let new_root = ExprNode {
            operation: Operation::Mul,
            result: 3.0,
            is_learnable: true,
            grad: 0.0
        };

        let merged_tree = add_new_root(tree_a, new_root);
        dbg!(&merged_tree);
        assert_eq!(merged_tree.names.len(), 1);
        assert_eq!(merged_tree.names.get("x"), Some(&1));
    }

    #[test]
    fn test_merge_trees() {
        let tree_a = Expr::new_leaf(1.0);
        let tree_b = Expr::new_leaf(2.0);

        let new_root = ExprNode {
            operation: Operation::Mul,
            result: 3.0,
            is_learnable: true,
            grad: 0.0
        };

        let merged_tree = merge_trees(tree_a, tree_b, new_root);
        assert_eq!(merged_tree.tree.len(), 3);
    }

    #[test]
    fn test_merge_trees_with_names() {
        let tree_a = Expr::new_leaf_with_name(1.0, "x");
        let tree_b = Expr::new_leaf_with_name(2.0, "y");

        let new_root = ExprNode {
            operation: Operation::Add,
            result: 3.0,
            is_learnable: false,
            grad: 0.0
        };

        let merged_tree = merge_trees(tree_a, tree_b, new_root);
        assert_eq!(merged_tree.tree.len(), 3);
        assert_eq!(merged_tree.names.len(), 2);
        assert_eq!(merged_tree.names.get("x"), Some(&1));
        assert_eq!(merged_tree.names.get("y"), Some(&0));
    }

    #[test]
    fn test_tanh() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.tanh();

        assert_eq!(expr2.result(), 0.7615941559557649);
        assert_eq!(expr2.operation(), Operation::Tanh);

        // Some other known values
        fn get_tanh(x: f64) -> f64 {
            Expr::new_leaf(x).tanh().result()
        }

        assert_float_eq(get_tanh(10.74), 0.9999999);
        assert_float_eq(get_tanh(-10.74), -0.9999999);
        assert_float_eq(get_tanh(0.0), 0.0);
    }

    #[test]
    fn test_relu() {
        // negative case
        let expr = Expr::new_leaf(-1.0);
        let expr2 = expr.relu();

        assert_eq!(expr2.result(), 0.0);
        assert_eq!(expr2.operation(), Operation::ReLU);

        // positive case
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.relu();

        assert_eq!(expr2.result(), 1.0);
        assert_eq!(expr2.operation(), Operation::ReLU);
    }
}
