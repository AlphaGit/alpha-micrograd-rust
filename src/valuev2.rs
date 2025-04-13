//! A simple library for creating and backpropagating through expression trees.
//! 
//! This package includes the following elements to construct expression trees:
//! - [`Expr`]: a node in the expression tree
#![deny(missing_docs)]
#![feature(hash_extract_if)]
use std::collections::HashMap;
use std::fmt::{Formatter, Display};
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
        let root_value = self.tree[0].as_ref().expect("root cannot be None").result;
        let result = root_value.tanh();

        let new_root = ExprNode {
            operation: Operation::Tanh,
            result,
            is_learnable: false,
            grad: 0.0
        };

        let new_tree = add_new_root(self.tree, new_root);
        Expr {
            tree: new_tree,
            names: self.names,
        }
    }
}

fn add_new_root(subtree_a: Vec<Option<ExprNode>>, new_root: ExprNode) -> Vec<Option<ExprNode>> {
    let mut subtree_b = Vec::with_capacity(1);
    subtree_b.push(None);
    merge_trees(subtree_a, subtree_b, new_root)
}

fn merge_trees(mut tree_a: Expr, mut tree_b: Expr, new_root: ExprNode) -> Expr {
    let mut subtree_a = tree_a.tree;
    let mut subtree_b = tree_b.tree;

    // levels are 1-based, level 1 is the root
    let len_a = subtree_a.len();
    let len_b = subtree_b.len();
    assert!(len_a > 0, "subtree_a must not be empty");
    assert!(len_b > 0, "subtree_b must not be empty");
    let levels_a = len_a.ilog2() + 1;
    let levels_b = len_b.ilog2() + 1;

    let mut level_number = levels_a.max(levels_b) + 1; // +1 for the new root

    let final_node_count = 2_usize.pow(level_number) - 1;
    let mut new_tree = Vec::with_capacity(final_node_count);
    let mut new_names = HashMap::new();

    dbg!(levels_a, levels_b, level_number, final_node_count);

    let mut min_moved_index: usize = 0;
    let mut max_moved_index: usize = 0;
    while level_number > 0 {
        let nodes_per_tree = 2_usize.pow(level_number - 1);
        dbg!(level_number, nodes_per_tree);

        // extract nodes for this level for tree_b (right side)
        let mut level_b = if levels_b >= level_number {
            subtree_b.split_off(subtree_b.len() - nodes_per_tree)
        } else {
            (1..nodes_per_tree).map(|_| None).collect::<Vec<Option<ExprNode>>>()
        };
        new_tree.append(&mut level_b);

        // renaming name mappings for tree_b, for this level
        mat_moved_index = 2_usize.pow(level_number - 1) - 1;

        let nodes_in_this_level = 2_usize.pow(level_number - 1);
        dbg!(level_number, nodes_in_this_level);

        let mut level_a = if levels_a >= level_number {
            subtree_a.split_off(subtree_a.len() - nodes_per_tree)
        } else {
            (1..nodes_per_tree).map(|_| None).collect::<Vec<Option<ExprNode>>>()
        };

        new_tree.append(&mut level_a);

        let level_min_old_index = 2_usize.pow(level_number - 1) - 1;
        let level_max_old_index = 2_usize.pow(level_number) - 1 - 1;

        tree_a.names.extract_if(|_, old_index| {
            return *old_index >= level_min_old_index && *old_index <= level_max_old_index;
        }).for_each(|(name, old_index)| {
            let new_index = old_index - level_min_old_index + new_tree.len() - nodes_per_tree * 2;
            new_names.insert(name, new_index);
        });

        level_number -= 1;
    }

    new_tree.append(&mut subtree_a);
    new_tree.append(&mut subtree_b);
    new_tree.push(Some(new_root));

    Expr {
        tree: new_tree,
        names: new_names,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_new_root() {
        let mut tree_a = Vec::new();
        let node_a = ExprNode {
            operation: Operation::Add,
            result: 1.0,
            is_learnable: true,
            grad: 0.0
        };
        tree_a.push(Some(node_a));

        assert_eq!(tree_a.len(), 1);

        let new_root = ExprNode {
            operation: Operation::Mul,
            result: 3.0,
            is_learnable: true,
            grad: 0.0
        };

        let merged_tree = add_new_root(tree_a, new_root);
        assert_eq!(merged_tree.len(), 3);
    }

    #[test]
    fn test_merge_trees() {
        let mut tree_a = Vec::new();
        let mut tree_b = Vec::new();

        let node_a = ExprNode {
            operation: Operation::Add,
            result: 1.0,
            is_learnable: true,
            grad: 0.0
        };

        let node_b = ExprNode {
            operation: Operation::Sub,
            result: 2.0,
            is_learnable: true,
            grad: 0.0
        };

        tree_a.push(Some(node_a));
        tree_b.push(Some(node_b));

        let new_root = ExprNode {
            operation: Operation::Mul,
            result: 3.0,
            is_learnable: true,
            grad: 0.0
        };

        let merged_tree = merge_trees(tree_a, tree_b, new_root);
        assert_eq!(merged_tree.len(), 3);
    }
}
