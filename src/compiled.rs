use std::collections::HashMap;

use crate::value::{Expr, Operation};

pub struct CompiledExpr {
    operations: Vec<Operation>,
    lhs: Vec<Option<usize>>,
    rhs: Vec<Option<usize>>,
    results: Vec<f64>,
    gradients: Vec<f64>,
    is_learnable: Vec<bool>,
    names_to_index: HashMap<String, usize>,
}

impl CompiledExpr {
    fn consume_expr(&mut self, expr: Expr) {
        let lhs = if let Some(operand1) = expr.operand1 {
            self.consume_expr(*operand1);
            Some(self.results.len() - 1)
        } else {
            None
        };

        let rhs = if let Some(operand2) = expr.operand2 {
            self.consume_expr(*operand2);
            Some(self.results.len() - 1)
        } else {
            None
        };

        self.lhs.push(lhs);
        self.rhs.push(rhs);
        self.results.push(expr.result);
        self.operations.push(expr.operation);
        self.gradients.push(expr.grad);
        self.is_learnable.push(expr.is_learnable);
        if let Some(name) = expr.name {
            self.names_to_index.insert(name, self.results.len() - 1);
        }
    }

    pub fn from_expr(expr: Expr) -> Self {
        let parameter_count = expr.parameter_count(false);
        let mut tape = CompiledExpr {
            operations: Vec::with_capacity(parameter_count),
            lhs: Vec::with_capacity(parameter_count),
            rhs: Vec::with_capacity(parameter_count),
            results: Vec::with_capacity(parameter_count),
            gradients: Vec::with_capacity(parameter_count),
            is_learnable: Vec::with_capacity(parameter_count),
            names_to_index: HashMap::new(),
        };

        tape.consume_expr(expr);

        tape
    }

    pub fn recalculate(&mut self) {
        for i in 0..self.results.len() {
            let operation = self.operations[i];
            let lhs_index = self.lhs[i];
            let rhs_index = self.rhs[i];

            let lhs_value = if let Some(index) = lhs_index {
                self.results[index]
            } else {
                0.0 // Default value for leaf nodes
            };

            let rhs_value = if let Some(index) = rhs_index {
                self.results[index]
            } else {
                0.0 // Default value for leaf nodes
            };

            self.results[i] = match operation {
                Operation::Add => lhs_value + rhs_value,
                Operation::Sub => lhs_value - rhs_value,
                Operation::Mul => lhs_value * rhs_value,
                Operation::Div => lhs_value / rhs_value,
                Operation::None => self.results[i], // No operation, keep the value
                Operation::Tanh => lhs_value.tanh(),
                Operation::Exp => lhs_value.exp(),
                Operation::Pow => lhs_value.powf(rhs_value),
                Operation::Log => lhs_value.ln(),
                Operation::ReLU => lhs_value.max(0.0),
                Operation::Neg => -lhs_value,
            };
        }
    }

    pub fn learn(&mut self, learning_rate: f64) {
        // set last gradient to 1.0
        self.gradients[self.results.len() - 1] = 1.0;

        for i in (0..self.results.len()).rev() {
            let operation = self.operations[i];
            let lhs_index = self.lhs[i].unwrap_or(0);
            let rhs_index = self.rhs[i].unwrap_or(0);

            let lhs_result = if let Some(index) = self.lhs[i] {
                self.results[index]
            } else {
                0.0 // Default value for leaf nodes
            };

            let rhs_result = if let Some(index) = self.rhs[i] {
                self.results[index]
            } else {
                0.0 // Default value for leaf nodes
            };
            let result = self.results[i];
            let gradient = self.gradients[i];

            match operation {
                // Learnable leaves
                Operation::None => {
                    // For learnable leaves only, update the result directly
                    // (the gradient is already set by a previous operation)
                    if self.is_learnable[i] {
                        self.results[i] -= learning_rate * self.gradients[i];
                    }
                }
                // Unary operations
                Operation::Tanh => {
                    let tanh_grad = 1.0 - (result * result);
                    self.gradients[lhs_index] = gradient * tanh_grad;
                }
                Operation::Exp => {
                    self.gradients[lhs_index] = gradient * result;
                }
                Operation::ReLU => {
                    self.gradients[lhs_index] = if result > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                }
                Operation::Log => {
                    self.gradients[lhs_index] = gradient / result;
                }
                Operation::Neg => {
                    self.gradients[lhs_index] = -gradient;
                }
                // Binary operations
                Operation::Add => {
                    self.gradients[lhs_index] = gradient;
                    self.gradients[rhs_index] = gradient;
                }
                Operation::Sub => {
                    self.gradients[lhs_index] = gradient;
                    self.gradients[rhs_index] = -gradient;
                }
                Operation::Mul => {
                    self.gradients[lhs_index] = gradient * rhs_result;
                    self.gradients[rhs_index] = gradient * lhs_result;
                }
                Operation::Div => {
                    self.gradients[lhs_index] = gradient / rhs_result;
                    self.gradients[rhs_index] = -gradient * lhs_result / (rhs_result * rhs_result);
                }
                Operation::Pow => {
                    let exponent = rhs_result;
                    let base = lhs_result;

                    self.gradients[lhs_index] = gradient * exponent * base.powf(exponent - 1.0);
                    self.gradients[rhs_index] = gradient * lhs_result.ln() * result;
                }
            }
        }
    }

    pub fn result(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            *self.results.last().unwrap()
        }
    }

    pub fn get_grad_by_name(&self, name: &str) -> Option<f64> {
        if let Some(&index) = self.names_to_index.get(name) {
            return Some(self.gradients[index]);
        }
        None
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
    fn test_from_expr_multilevel() {
        // Create a multilevel expression: (a + b) * (c - d)
        // where a=2.0, b=3.0, c=5.0, d=1.0
        // This should result in (2.0 + 3.0) * (5.0 - 1.0) = 5.0 * 4.0 = 20.0

        // Create leaf nodes
        let a = Expr::new_leaf(2.0);
        let b = Expr::new_leaf_with_name(3.0, "b");
        let c = Expr::new_leaf(5.0);
        let d = Expr::new_leaf_with_name(1.0, "d");

        // Create (a + b)
        let add = a + b;

        // Create (c - d)
        let sub = c - d;

        // Create (a + b) * (c - d)
        let mul = add * sub;

        // Convert to tape
        let tape = CompiledExpr::from_expr(mul);

        // Verify that all elements of the tape have the same length
        assert_eq!(tape.results.len(), 7);
        assert_eq!(tape.operations.len(), 7);
        assert_eq!(tape.lhs.len(), 7);
        assert_eq!(tape.rhs.len(), 7);
        assert_eq!(tape.gradients.len(), 7);

        // Verify each operation in the tape
        // a: leaf, 2.0
        assert_eq!(tape.results[0], 2.0);
        assert_eq!(tape.lhs[0], None);
        assert_eq!(tape.rhs[0], None); // Leaf node
        assert_eq!(tape.operations[0], Operation::None);
        assert_eq!(tape.gradients[0], 0.0); // Default gradient for leaf

        // b: leaf, 3.0
        assert_eq!(tape.results[1], 3.0);
        assert_eq!(tape.lhs[1], None);
        assert_eq!(tape.rhs[1], None); // Leaf node
        assert_eq!(tape.operations[1], Operation::None);
        assert_eq!(tape.gradients[1], 0.0); // Default gradient for leaf

        // add: (a + b)
        assert_eq!(tape.results[2], 5.0);
        assert_eq!(tape.lhs[2], Some(0)); // Index of a
        assert_eq!(tape.rhs[2], Some(1)); // Index of b
        assert_eq!(tape.operations[2], Operation::Add);
        assert_eq!(tape.gradients[2], 0.0); // Default gradient for result

        // c: leaf, 5.0
        assert_eq!(tape.results[3], 5.0);
        assert_eq!(tape.lhs[3], None);
        assert_eq!(tape.rhs[3], None); // Leaf node
        assert_eq!(tape.operations[3], Operation::None);
        assert_eq!(tape.gradients[3], 0.0); // Default gradient for leaf

        // d: leaf, 1.0
        assert_eq!(tape.results[4], 1.0);
        assert_eq!(tape.lhs[4], None);
        assert_eq!(tape.rhs[4], None); // Leaf node
        assert_eq!(tape.operations[4], Operation::None);
        assert_eq!(tape.gradients[4], 0.0); // Default gradient for leaf

        // sub: (c - d)
        assert_eq!(tape.results[5], 4.0);
        assert_eq!(tape.lhs[5], Some(3)); // Index of c
        assert_eq!(tape.rhs[5], Some(4)); // Index of d
        assert_eq!(tape.operations[5], Operation::Sub);
        assert_eq!(tape.gradients[5], 0.0); // Default gradient for result

        // mul: (a + b) * (c - d)
        assert_eq!(tape.results[6], 20.0);
        assert_eq!(tape.lhs[6], Some(2)); // Index of add
        assert_eq!(tape.rhs[6], Some(5)); // Index of sub
        assert_eq!(tape.operations[6], Operation::Mul);
        assert_eq!(tape.gradients[6], 0.0); // Default gradient for result

        // Verify names to index mapping
        assert_eq!(tape.names_to_index.get("b"), Some(&1));
        assert_eq!(tape.names_to_index.get("d"), Some(&4));
        assert!(tape.names_to_index.get("a").is_none());
        assert!(tape.names_to_index.get("c").is_none());
    }

    #[test]
    fn test_recalculate() {
        // Create a simple expression: a + b
        let a = Expr::new_leaf(2.0);
        let b = Expr::new_leaf(3.0);
        let expr = a + b;

        // Convert to tape
        let mut tape = CompiledExpr::from_expr(expr);

        // Recalculate the results
        tape.recalculate();

        // Verify the result
        assert_eq!(tape.results[2], 5.0); // Result of a + b

        tape.results[0] = 4.0; // Change a to 4.0
        tape.results[1] = 6.0; // Change b to 6.0
        tape.recalculate();

        // Verify the recalculated result
        assert_eq!(tape.results[2], 10.0); // Result of 4.0 + 6.0
    }

    #[test]
    fn test_learn_simple() {
        let expr = Expr::new_leaf(1.0);
        let mut tape = CompiledExpr::from_expr(expr);
        assert_eq!(tape.result(), 1.0);

        tape.learn(1e-01);
        assert_eq!(tape.result(), 0.9); // 1.0 - 0.1 = 0.9
    }

    #[test]
    fn test_learn_skips_non_learnable() {
        let mut expr = Expr::new_leaf(1.0);
        expr.is_learnable = false;
        let mut tape = CompiledExpr::from_expr(expr);
        assert_eq!(tape.result(), 1.0);

        tape.learn(1e-01);
        assert_eq!(tape.result(), 1.0);
    }

    #[test]
    fn test_learn_multilevel() {
        let expr = Expr::new_leaf(1.0);
        let expr2 = expr.tanh();
        let mut tape = CompiledExpr::from_expr(expr2);
        assert_eq!(tape.result(), 0.7615941559557649); // tanh(1.0)
        tape.learn(1e-09);
        tape.recalculate();

        assert_eq!(tape.result(), 0.7615941557793864);
    }

    #[test]
    fn test_backpropagation_add() {
        let mut operand1 = Expr::new_leaf(1.0);
        operand1.name = Some("a".to_string());

        let mut operand2 = Expr::new_leaf(2.0);
        operand2.name = Some("b".to_string());

        let expr3 = operand1 + operand2;
        let mut tape = CompiledExpr::from_expr(expr3);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        let grad_b = tape.get_grad_by_name("b").unwrap();
        assert_eq!(grad_a, 1.0);
        assert_eq!(grad_b, 1.0);
    }

    #[test]
    fn test_backpropagation_sub() {
        let mut operand1 = Expr::new_leaf(1.0);
        operand1.name = Some("a".to_string());

        let mut operand2 = Expr::new_leaf(2.0);
        operand2.name = Some("b".to_string());

        let expr3 = operand1 - operand2;
        let mut tape = CompiledExpr::from_expr(expr3);
        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        let grad_b = tape.get_grad_by_name("b").unwrap();
        assert_eq!(grad_a, 1.0);
        assert_eq!(grad_b, -1.0);
    }

    #[test]
    fn test_backpropagation_mul() {
        let mut operand1 = Expr::new_leaf(3.0);
        operand1.name = Some("a".to_string());

        let mut operand2 = Expr::new_leaf(4.0);
        operand2.name = Some("b".to_string());

        let expr3 = operand1 * operand2;
        let mut tape = CompiledExpr::from_expr(expr3);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        let grad_b = tape.get_grad_by_name("b").unwrap();
        assert_eq!(grad_a, 4.0);
        assert_eq!(grad_b, 3.0);
    }

    #[test]
    fn test_backpropagation_div() {
        let mut operand1 = Expr::new_leaf(3.0);
        operand1.name = Some("a".to_string());

        let mut operand2 = Expr::new_leaf(4.0);
        operand2.name = Some("b".to_string());
        let expr3 = operand1 / operand2;
        let mut tape = CompiledExpr::from_expr(expr3);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        let grad_b = tape.get_grad_by_name("b").unwrap();
        assert_eq!(grad_a, 0.25);
        assert_eq!(grad_b, -0.1875);
    }

    #[test]
    fn test_backpropagation_tanh() {
        let mut operand1 = Expr::new_leaf(0.0);
        operand1.name = Some("a".to_string());
        let expr2 = operand1.tanh();
        let mut tape = CompiledExpr::from_expr(expr2);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        assert_float_eq(grad_a, 1.0);
    }

    #[test]
    fn test_backpropagation_relu() {
        let mut operand1 = Expr::new_leaf(-1.0);
        operand1.name = Some("a".to_string());
        let expr2 = operand1.relu();
        let mut tape = CompiledExpr::from_expr(expr2);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        assert_eq!(grad_a, 0.0);
    }

    #[test]
    fn test_backpropagation_exp() {
        let mut operand1 = Expr::new_leaf(0.0);
        operand1.name = Some("a".to_string());
        let expr2 = operand1.exp();
        let mut tape = CompiledExpr::from_expr(expr2);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        assert_eq!(grad_a, 1.0);
    }

    #[test]
    fn test_backpropagation_pow() {
        let mut operand1 = Expr::new_leaf(2.0);
        operand1.name = Some("a".to_string());
        let mut operand2 = Expr::new_leaf(3.0);
        operand2.name = Some("b".to_string());
        let expr3 = operand1.pow(operand2);
        let mut tape = CompiledExpr::from_expr(expr3);

        tape.learn(1e-09);

        let grad_a = tape.get_grad_by_name("a").unwrap();
        let grad_b = tape.get_grad_by_name("b").unwrap();
        assert_eq!(grad_a, 12.0);
        assert_eq!(grad_b, 5.545177444479562);
    }

    #[test]
    fn test_backpropagation_mixed_tree() {
        let mut operand1 = Expr::new_leaf(1.0);
        operand1.name = Some("operand1".to_string());
        let mut operand2 = Expr::new_leaf(2.0);
        operand2.name = Some("operand2".to_string());
        let mut expr3 = operand1 + operand2;
        expr3.name = Some("expr3".to_string());
        let expr4 = expr3.tanh();
        let mut tape = CompiledExpr::from_expr(expr4);

        tape.learn(1e-09);

        let expr3_grad = tape.get_grad_by_name("expr3").unwrap();
        let operand1_grad = tape.get_grad_by_name("operand1").unwrap();
        let operand2_grad = tape.get_grad_by_name("operand2").unwrap();

        assert_eq!(expr3_grad, 0.009866037165440211);
        assert_eq!(operand1_grad, 0.009866037165440211);
        assert_eq!(operand2_grad, 0.009866037165440211);
    }

    #[test]
    fn test_backpropagation_karpathys_example() {
        let mut x1 = Expr::new_leaf(2.0);
        x1.name = Some("x1".to_string());
        let mut x2 = Expr::new_leaf(0.0);
        x2.name = Some("x2".to_string());
        let mut w1 = Expr::new_leaf(-3.0);
        w1.name = Some("w1".to_string());
        let mut w2 = Expr::new_leaf(1.0);
        w2.name = Some("w2".to_string());
        let mut b = Expr::new_leaf(6.8813735870195432);
        b.name = Some("b".to_string());

        let mut x1w1 = x1 * w1;
        x1w1.name = Some("x1w1".to_string());
        let mut x2w2 = x2 * w2;
        x2w2.name = Some("x2w2".to_string());
        let mut x1w1_x2w2 = x1w1 + x2w2;
        x1w1_x2w2.name = Some("x1w1_x2w2".to_string());
        let mut n = x1w1_x2w2 + b;
        n.name = Some("n".to_string());
        let o = n.tanh();
        let mut tape = CompiledExpr::from_expr(o);

        tape.learn(1e-09);

        let n_grad = tape.get_grad_by_name("n").unwrap();
        assert_float_eq(n_grad, 0.5);

        let x1w1_x2w2_grad = tape.get_grad_by_name("x1w1_x2w2").unwrap();
        assert_float_eq(x1w1_x2w2_grad, 0.5);

        let b_grad = tape.get_grad_by_name("b").unwrap();
        assert_float_eq(b_grad, 0.5);

        let x1w1_grad = tape.get_grad_by_name("x1w1").unwrap();
        assert_float_eq(x1w1_grad, 0.5);

        let x2w2_grad = tape.get_grad_by_name("x2w2").unwrap();
        assert_float_eq(x2w2_grad, 0.5);

        let x1_grad = tape.get_grad_by_name("x1").unwrap();
        assert_float_eq(x1_grad, -1.5);

        let w1_grad = tape.get_grad_by_name("w1").unwrap();
        assert_float_eq(w1_grad, 1.0);

        let x2_grad = tape.get_grad_by_name("x2").unwrap();
        assert_float_eq(x2_grad, 0.5);

        let w2_grad = tape.get_grad_by_name("w2").unwrap();
        assert_float_eq(w2_grad, 0.0);
    }
}
