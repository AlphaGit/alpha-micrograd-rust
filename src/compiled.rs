use std::collections::HashMap;

use crate::value::{Expr, Operation};

struct CompiledExpr {
    operations: Vec<Operation>,
    lhs: Vec<Option<usize>>,
    rhs: Vec<Option<usize>>,
    results: Vec<f64>,
    gradients: Vec<f64>,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
