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

    /// Sets the name of the expression, making sure
    /// it is preserved in the tree operations.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::Expr;
    ///
    /// let mut expr = Expr::new_leaf(1.0);
    /// expr.set_name("x");
    /// ```
    pub fn set_name(&mut self, name: &str) {
        self.names.insert(name.to_string(), self.tree.len() - 1);
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
        let current_value = self.result();
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
        let current_value = self.result();
        let result = current_value.max(0.0);
        let new_root = ExprNode {
            operation: Operation::ReLU,
            result,
            is_learnable: false,
            grad: 0.0
        };
        add_new_root(self, new_root)
    }

    /// Applies the exponential function (e^x) to the expression and returns it as a new expression.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::Expr;
    ///
    /// let expr = Expr::new_leaf(1.0);
    /// let expr2 = expr.exp();
    ///
    /// assert_eq!(expr2.result(), 2.718281828459045);
    /// ```
    pub fn exp(self) -> Expr {
        let current_value = self.result();
        let result = current_value.exp();
        let new_root = ExprNode {
            operation: Operation::Exp,
            result,
            is_learnable: false,
            grad: 0.0
        };
        add_new_root(self, new_root)
    }

    /// Raises the expression to the power of the given exponent (expression) and returns it as a new expression.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::Expr;
    ///
    /// let expr = Expr::new_leaf(2.0);
    /// let exponent = Expr::new_leaf(3.0);
    /// let result = expr.pow(exponent);
    ///
    /// assert_eq!(result.result(), 8.0);
    /// ```
    pub fn pow(self, exponent: Expr) -> Expr {
        let current_value = self.result();
        let exponent_value = exponent.result();
        let result = current_value.powf(exponent_value);
        let new_root = ExprNode {
            operation: Operation::Pow,
            result,
            is_learnable: false,
            grad: 0.0
        };
        add_new_root(self, new_root)
    }

    /// Applies the natural logarithm function to the expression and returns it as a new expression.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::Expr;
    ///
    /// let expr = Expr::new_leaf(2.0);
    /// let expr2 = expr.log();
    ///
    /// assert_eq!(expr2.result(), 0.6931471805599453);
    /// ```
    pub fn log(self) -> Expr {
        let current_value = self.result();
        let result = current_value.ln();
        let new_root = ExprNode {
            operation: Operation::Log,
            result,
            is_learnable: false,
            grad: 0.0
        };
        add_new_root(self, new_root)
    }

    /// Negates the expression and returns it as a new expression.
    ///
    /// Example:
    /// ```rust
    /// use alpha_micrograd_rust::valuev2::Expr;
    ///
    /// let expr = Expr::new_leaf(1.0);
    /// let expr2 = expr.neg();
    ///
    /// assert_eq!(expr2.result(), -1.0);
    /// ```
    pub fn neg(self) -> Expr {
        let current_value = self.result();
        let result = -current_value;
        let new_root = ExprNode {
            operation: Operation::Neg,
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

fn reindex_names(names: &HashMap<String, usize>, new_names: &mut HashMap<String, usize>, tree_levels: u32, level_to_extract: u32, index_offset: usize) {
    if level_to_extract > tree_levels {
        return;
    }

    let nodes_in_tree = 2_usize.pow(tree_levels) - 1;
    let nodes_in_tree_of_level_to_extract = 2_usize.pow(level_to_extract) - 1;
    let nodes_in_this_level = 2_usize.pow(level_to_extract - 1);
    let lower_index = nodes_in_tree - nodes_in_tree_of_level_to_extract;
    let upper_index = lower_index + nodes_in_this_level - 1;

    names.iter()
        .filter(|&(_, index)| *index <= upper_index && *index >= lower_index)
        .for_each(|(name, old_index)| {
            let new_index = old_index - lower_index + index_offset;
            new_names.insert(name.clone(), new_index);
        });
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

    let mut new_tree = Vec::with_capacity(final_node_count);
    let mut new_names = HashMap::new();

    let mut new_tree_section_start_index = 0;
    // level 1 is skipped: it's the root and it's done after the loop
    for level_number in (2..=final_level_number).rev() {
        // simplifying 2^(l-1)/2 to 2^(l-2) creates a subtraction overflow for l=1
        let nodes_per_tree = 2_usize.pow(level_number - 1) / 2;

        // extract nodes for this level for tree_b (right side)
        let mut nodes_b = if levels_b + 1 >= level_number {
            subtree_b.drain(..nodes_per_tree).collect()
        } else {
            (0..nodes_per_tree).map(|_| None).collect::<Vec<Option<ExprNode>>>()
        };
        new_tree.append(&mut nodes_b);

        reindex_names(
            &tree_b.names,
            &mut new_names, 
            levels_b,
            level_number - 1,
            new_tree_section_start_index,
        );
        new_tree_section_start_index += nodes_per_tree;

        let mut nodes_a = if levels_a + 1 >= level_number {
            subtree_a.drain(..nodes_per_tree).collect()
        } else {
            (0..nodes_per_tree).map(|_| None).collect::<Vec<Option<ExprNode>>>()
        };
        new_tree.append(&mut nodes_a);

        reindex_names(
            &tree_a.names,
            &mut new_names, 
            levels_a,
            level_number - 1,
            new_tree_section_start_index
        );

        new_tree_section_start_index += nodes_per_tree;
    }

    new_tree.push(Some(new_root));

    assert_eq!(new_tree.len(), final_node_count, "new_tree should have {} nodes, but has {}", final_node_count, new_tree.len());
    assert_eq!(new_tree.capacity(), final_node_count, "new_tree should have {} capacity, but has {}", final_node_count, new_tree.capacity());

    Expr {
        tree: new_tree,
        names: new_names,
    }
}

/// Implements the [`Add`] trait for the [`Expr`] struct.
/// 
/// This implementation allows the addition of two [`Expr`] objects.
/// 
/// Example:
/// ```rust
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr + expr2;
/// 
/// assert_eq!(result.result(), 3.0);
/// ```
impl Add for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        let result = self.result() + other.result();
        let new_root = ExprNode {
            operation: Operation::Add,
            result,
            is_learnable: true,
            grad: 0.0
        };
        merge_trees(self, other, new_root)
    }
}

/// Implements the [`Add`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the addition of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// ```rust
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr + 2.0;
/// 
/// assert_eq!(result.result(), 3.0);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = 2.0 + expr;
/// 
/// assert_eq!(result.result(), 3.0);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr * expr2;
/// 
/// assert_eq!(result.result(), 2.0);
/// ```
impl Mul for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        let result = self.result() * other.result();
        let new_root = ExprNode {
            operation: Operation::Mul,
            result,
            is_learnable: true,
            grad: 0.0
        };
        merge_trees(self, other, new_root)
    }
}

/// Implements the [`Mul`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the multiplication of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr * 2.0;
/// 
/// assert_eq!(result.result(), 2.0);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = 2.0 * expr;
/// 
/// assert_eq!(result.result(), 2.0);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr - expr2;
/// 
/// assert_eq!(result.result(), -1.0);
/// ```
impl Sub for Expr {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        let result = self.result() - other.result();
        let new_root = ExprNode {
            operation: Operation::Sub,
            result,
            is_learnable: true,
            grad: 0.0
        };
        merge_trees(self, other, new_root)
    }
}

/// Implements the [`Sub`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the subtraction of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr - 2.0;
/// 
/// assert_eq!(result.result(), -1.0);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = 2.0 - expr;
/// 
/// assert_eq!(result.result(), 1.0);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// 
/// let result = expr / expr2;
/// 
/// assert_eq!(result.result(), 0.5);
/// ```
impl Div for Expr {
    type Output = Expr;

    fn div(self, other: Expr) -> Expr {
        let result = self.result() / other.result();
        let new_root = ExprNode {
            operation: Operation::Div,
            result,
            is_learnable: true,
            grad: 0.0
        };
        merge_trees(self, other, new_root)
    }
}

/// Implements the [`Div`] trait for the [`Expr`] struct, when the right operand is a [`f64`].
/// 
/// This implementation allows the division of an [`Expr`] object and a [`f64`] value.
/// 
/// Example:
/// 
/// ```rust
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let result = expr / 2.0;
/// 
/// assert_eq!(result.result(), 0.5);
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
/// use alpha_micrograd_rust::valuev2::Expr;
/// 
/// let expr = Expr::new_leaf(1.0);
/// let expr2 = Expr::new_leaf(2.0);
/// let expr3 = Expr::new_leaf(3.0);
/// 
/// let sum = vec![expr, expr2, expr3].into_iter().sum::<Expr>();
/// 
/// assert_eq!(sum.result(), 6.0);
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

    #[test]
    fn test_exp() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.exp();

        assert_eq!(expr2.result(), 2.718281828459045);
        assert_eq!(expr2.operation(), Operation::Exp);
    }

    #[test]
    fn test_pow() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = Expr::new_leaf(3.0);
        let result = expr.pow(expr2);

        assert_eq!(result.result(), 8.0);
        assert_eq!(result.operation(), Operation::Pow);
    }

    #[test]
    fn test_log() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr.log();

        assert_eq!(expr2.result(), 0.6931471805599453);
        assert_eq!(expr2.operation(), Operation::Log);
    }

    #[test]
    fn test_neg() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr.neg();

        assert_eq!(expr2.result(), -2.0);
        assert_eq!(expr2.operation(), Operation::Neg);
    }

    #[test]
    fn test_add() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let expr3 = expr + expr2;

        assert_eq!(expr3.result(), 3.0);
        assert_eq!(expr3.operation(), Operation::Add);
    }

    #[test]
    fn test_add_f64() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr + 2.0;

        assert_eq!(expr2.result(), 3.0);
        assert_eq!(expr2.operation(), Operation::Add);
    }

    #[test]
    fn test_add_f64_expr() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = 2.0 + expr;

        assert_eq!(expr2.result(), 3.0);
        assert_eq!(expr2.operation(), Operation::Add);
    }

    #[test]
    fn test_mul() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = Expr::new_leaf(3.0);
        let expr3 = expr * expr2;

        assert_eq!(expr3.result(), 6.0);
        assert_eq!(expr3.operation(), Operation::Mul);
    }

    #[test]
    fn test_mul_f64() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr * 3.0;

        assert_eq!(expr2.result(), 6.0);
        assert_eq!(expr2.operation(), Operation::Mul);
    }

    #[test]
    fn test_mul_f64_expr() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = 3.0 * expr;

        assert_eq!(expr2.result(), 6.0);
        assert_eq!(expr2.operation(), Operation::Mul);
    }

    #[test]
    fn test_sub() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = Expr::new_leaf(3.0);
        let expr3 = expr - expr2;

        assert_eq!(expr3.result(), -1.0);
        assert_eq!(expr3.operation(), Operation::Sub);
    }

    #[test]
    fn test_sub_f64() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = expr - 3.0;

        assert_eq!(expr2.result(), -1.0);
        assert_eq!(expr2.operation(), Operation::Sub);
    }

    #[test]
    fn test_sub_f64_expr() {
        let expr = Expr::new_leaf(2.0);
        let expr2 = 3.0 - expr;

        assert_eq!(expr2.result(), 1.0);
        assert_eq!(expr2.operation(), Operation::Sub);
    }

    #[test]
    fn test_div() {
        let expr = Expr::new_leaf(6.0);
        let expr2 = Expr::new_leaf(3.0);
        let expr3 = expr / expr2;

        assert_eq!(expr3.result(), 2.0);
        assert_eq!(expr3.operation(), Operation::Div);
    }

    #[test]
    fn test_div_f64() {
        let expr = Expr::new_leaf(6.0);
        let expr2 = expr / 3.0;

        assert_eq!(expr2.result(), 2.0);
        assert_eq!(expr2.operation(), Operation::Div);
    }

    #[test]
    fn test_sum_iterator() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = Expr::new_leaf(2.0);
        let expr3 = Expr::new_leaf(3.0);

        let sum: Expr = vec![expr, expr2, expr3].into_iter().sum::<Expr>();
        assert_eq!(sum.result(), 6.0);
    }

    #[test]
    fn sum_empty_iterator() {
        let sum: Expr = Vec::<Expr>::new().into_iter().sum();
        assert_eq!(sum.result(), 0.0);
    }

    #[test]
    fn test_tree_storage() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.relu();
        let expr3 = expr2 + 2.0;
        let expr4 = expr3 * 3.0;

        // tree is now:
        //        9 (mul)
        //      /         \
        //     3 (add)    3 (leaf)
        //    /       \
        //   1 (relu) 2 (leaf)
        //  /
        // 1 (leaf)
        // With missing terms:
        //          9
        //       /     \
        //      3       3
        //    /   \    /  \
        //   1    2   n    n
        //  / \  / \ / \  / \
        // 1  n n  n n  n n n
        // represented in array (right to left):
        // [n, n, n, n, n, n, n, 1, n, n, 2, 1, 3, 3, 9]

        assert_eq!(expr4.tree.len(), 15);
        assert_eq!(expr4.tree.capacity(), 15);
        assert_eq!(expr4.tree[0].is_none(), true);
        assert_eq!(expr4.tree[1].is_none(), true);
        assert_eq!(expr4.tree[2].is_none(), true);
        assert_eq!(expr4.tree[3].is_none(), true);
        assert_eq!(expr4.tree[4].is_none(), true);
        assert_eq!(expr4.tree[5].is_none(), true);
        assert_eq!(expr4.tree[6].is_none(), true);
        assert_eq!(
            expr4.tree[7].as_ref().unwrap().result,
            1.0
        );
        assert_eq!(expr4.tree[8].is_none(), true);
        assert_eq!(expr4.tree[9].is_none(), true);
        assert_eq!(
            expr4.tree[10].as_ref().unwrap().result,
            2.0
        );
        assert_eq!(
            expr4.tree[11].as_ref().unwrap().result,
            1.0
        );
        assert_eq!(
            expr4.tree[12].as_ref().unwrap().result,
            3.0
        );
        assert_eq!(
            expr4.tree[13].as_ref().unwrap().result,
            3.0
        );
        assert_eq!(
            expr4.tree[14].as_ref().unwrap().result,
            9.0
        );
    }

    #[test]
    fn test_tree_storage_with_names() {
        let expr = Expr::new_leaf_with_name(1.0, "a");
        // Tree is now:
        //        0 (leaf)
        //        a (leaf)
        assert_eq!(expr.names.len(), 1);
        assert_eq!(expr.names.get("a"), Some(&0));

        let mut expr2 = expr.relu();
        expr2.set_name("b");
        // Tree is now:
        //        2
        //        b (relu)
        //      /       \
        //     1         0
        //     a         n

        assert_eq!(expr2.names.len(), 2);
        assert_eq!(expr2.names.get("a"), Some(&1));
        assert_eq!(expr2.names.get("b"), Some(&2));

        let mut expr3 = expr2 + 2.0;
        expr3.set_name("c");

        // Tree is now:
        //        6
        //        c (add)
        //      /       \
        //     5         4
        //     b (relu)  n
        //    /   \     / \
        //   3     2   1   0
        //   a     n   n   n
        assert_eq!(expr3.names.len(), 3);
        assert_eq!(expr3.names.get("a"), Some(&3));
        assert_eq!(expr3.names.get("b"), Some(&5));
        assert_eq!(expr3.names.get("c"), Some(&6));

        let mut expr4 = expr3 * 3.0;
        expr4.set_name("d");
        // Tree is now:
        //        14
        //        d (mul)
        //      /       \
        //     13        12
        //     c (add)   n
        //    /   \     / \
        //   11    10  9   8
        //   b     n   n   n
        //  / \   / \ / \ / \
        // 7   6  5 4 3 2 1 0
        // a   n  n n n n n n

        assert_eq!(expr4.names.len(), 4);
        assert_eq!(expr4.names.get("a"), Some(&7));
        assert_eq!(expr4.names.get("b"), Some(&11));
        assert_eq!(expr4.names.get("c"), Some(&13));
        assert_eq!(expr4.names.get("d"), Some(&14));
    }

    #[test]
    fn test_reindex_names() {
        let mut names = HashMap::new();
        names.insert("a".to_string(), 0);
        // Tree is now:
        //        0 (leaf)
        //        a (leaf)

        let mut names2 = HashMap::new();
        // Tree will now be:
        //        2
        //        b (relu)
        //      /       \
        //     1         0
        //     a         n
        // moving left tree: (a)
        reindex_names(&names, &mut names2, 1, 1, 1);

        assert_eq!(names2.len(), 1);
        assert_eq!(names2.get("a"), Some(&1));

        // adding root (b)
        names2.insert("b".to_string(), 2);

        let mut names3 = HashMap::new();
        // Tree will now be:
        //        6
        //        c (add)
        //      /       \
        //     5         4
        //     b (relu)  n
        //    /   \     / \
        //   3     2   1   0
        //   a     n   n   n

        // moving left tree, level 2: (a)
        reindex_names(&names2, &mut names3, 2, 2, 2);
        assert_eq!(names3.len(), 1);
        assert_eq!(names3.get("a"), Some(&3));

        // moving left tree, level 1: (b)
        reindex_names(&names2, &mut names3, 2, 1, 5);
        assert_eq!(names3.len(), 2);
        assert_eq!(names3.get("b"), Some(&5));

        // adding root (c)
        names3.insert("c".to_string(), 6);

        let mut names4 = HashMap::new();
        // Tree will now be:
        //        14
        //        d (mul)
        //      /       \
        //     13        12
        //     c (add)   n
        //    /   \     / \
        //   11    10  9   8
        //   b     n   n   n
        //  / \   / \ / \ / \
        // 7   6  5 4 3 2 1 0
        // a   n  n n n n n n

        // moving left tree, level 3 (a)
        reindex_names(&names3, &mut names4, 3, 3, 4);
        assert_eq!(names4.len(), 1);
        assert_eq!(names4.get("a"), Some(&7));

        // moving left tree, level 2 (b)
        reindex_names(&names3, &mut names4, 3, 2, 10);
        assert_eq!(names4.len(), 2);
        assert_eq!(names4.get("b"), Some(&11));

        // moving left tree, level 1 (c)
        reindex_names(&names3, &mut names4, 3, 1, 13);
        assert_eq!(names4.len(), 3);
        assert_eq!(names4.get("c"), Some(&13));
    }
}
