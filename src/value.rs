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
}

impl Operation {
    fn assert_is_type(&self, expr_type: ExprType) {
        match self {
            Operation::None => assert_eq!(expr_type, ExprType::Leaf),
            Operation::Tanh | Operation::Exp | Operation::ReLU => assert_eq!(expr_type, ExprType::Unary),
            _ => assert_eq!(expr_type, ExprType::Binary),
        }
    }
}

#[derive(Debug, PartialEq)]
enum ExprType {
    Leaf,
    Unary,
    Binary,
}

#[derive(Debug, Clone)]
pub struct Expr {
    operand1: Option<Box<Expr>>,
    operand2: Option<Box<Expr>>,
    operation: Operation,
    pub result: f64,
    is_learnable: bool,
    pub grad: f64,
}

impl Expr {
    pub fn new_leaf(value: f64) -> Expr {
        Expr {
            operand1: None,
            operand2: None,
            operation: Operation::None,
            result: value,
            is_learnable: true,
            grad: 0.0,
        }
    }

    fn expr_type(&self) -> ExprType {
        match self.operation {
            Operation::None => ExprType::Leaf,
            Operation::Tanh | Operation::Exp | Operation::ReLU => ExprType::Unary,
            _ => ExprType::Binary,
        }
    }

    fn new_unary(operand: Expr, operation: Operation, result: f64) -> Expr {
        operation.assert_is_type(ExprType::Unary);
        Expr {
            operand1: Some(Box::new(operand)),
            operand2: None,
            operation,
            result,
            is_learnable: true,
            grad: 0.0,
        }
    }

    fn new_binary(operand1: Expr, operand2: Expr, operation: Operation, result: f64) -> Expr {
        operation.assert_is_type(ExprType::Binary);
        Expr {
            operand1: Some(Box::new(operand1)),
            operand2: Some(Box::new(operand2)),
            operation,
            result,
            is_learnable: true,
            grad: 0.0,
        }
    }

    pub fn tanh(self) -> Expr {
        let e_2x = (self.result * 2.0).exp();
        let numerator = e_2x - 1.0;
        let denominator = e_2x + 1.0;
        let result = numerator / denominator;

        Expr::new_unary(self, Operation::Tanh, result)
    }

    pub fn relu(self) -> Expr {
        let result = self.result.max(0.0);
        Expr::new_unary(self, Operation::ReLU, result)
    }

    pub fn exp(self) -> Expr {
        let result = self.result.exp();
        Expr::new_unary(self, Operation::Exp, result)
    }

    pub fn pow(self, exponent: Expr) -> Expr {
        let result = self.result.powf(exponent.result);
        Expr::new_binary(self, exponent, Operation::Pow, result)
    }

    pub fn set_learnable(&mut self, learnable: bool) {
        self.is_learnable = learnable;
    }

    pub fn recalculate(&mut self) {
        match self.expr_type() {
            ExprType::Leaf => {}
            ExprType::Unary => {
                let operand1 = self.operand1.as_mut().expect("Unary expression did not have an operand");
                operand1.recalculate();

                self.result = match self.operation {
                    Operation::Tanh => operand1.result.tanh(),
                    Operation::Exp => operand1.result.exp(),
                    Operation::ReLU => operand1.result.max(0.0),
                    _ => panic!("Invalid unary operation {:?}", self.operation),
                };
            }
            ExprType::Binary => {
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

    pub fn learn(&mut self, learning_rate: f64) {
        match self.expr_type() {
            ExprType::Leaf => {}
            ExprType::Unary => {
                let operand1 = self.operand1.as_mut().expect("Unary expression did not have an operand");

                match self.operation {
                    Operation::Tanh => {
                        let tanh_grad = 1.0 - (self.result * self.result);
                        operand1.grad = self.grad * tanh_grad;
                    }
                    Operation::Exp => {
                        operand1.grad = self.grad * self.result;
                    }
                    Operation::ReLU => {
                        operand1.grad = self.grad * if self.result > 0.0 { 1.0 } else { 0.0 };
                    }
                    _ => panic!("Invalid unary operation {:?}", self.operation),
                }

                operand1.learn(learning_rate);
            }
            ExprType::Binary => {
                let operand1 = self.operand1.as_mut().expect("Binary expression did not have an operand");
                let operand2 = self.operand2.as_mut().expect("Binary expression did not have a second operand");

                match self.operation {
                    Operation::Add => {
                        operand1.grad = self.grad;
                        operand2.grad = self.grad;
                    }
                    Operation::Sub => {
                        operand1.grad = self.grad;
                        operand2.grad = -self.grad;
                    }
                    Operation::Mul => {
                        let operand2_result = operand2.result;
                        let operand1_result = operand1.result;

                        operand1.grad = self.grad * operand2_result;
                        operand2.grad = self.grad * operand1_result;
                    }
                    Operation::Div => {
                        let operand2_result = operand2.result;
                        let operand1_result = operand1.result;

                        operand1.grad = self.grad / operand2_result;
                        operand2.grad = -self.grad * operand1_result / (operand2_result * operand2_result);
                    }
                    Operation::Pow => {
                        let exponent = operand2.result;
                        let base = operand1.result;

                        operand1.grad = self.grad * exponent * base.powf(exponent - 1.0);
                        operand2.grad = self.grad * base.powf(exponent) * base.ln();
                    }
                    _ => panic!("Invalid binary operation: {:?}", self.operation),
                }

                operand1.learn(learning_rate);
                operand2.learn(learning_rate);
            }
        }

        if self.is_learnable {
           self.result -= learning_rate * self.grad;
        }
    }
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        let result = self.result + other.result;
        Expr::new_binary(self, other, Operation::Add, result)
    }
}

impl Add<f64> for Expr {
    type Output = Expr;

    fn add(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self + operand2
    }
}

impl Add<Expr> for f64 {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        let operand1 = Expr::new_leaf(self);
        operand1 + other
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        let result = self.result * other.result;
        Expr::new_binary(self, other, Operation::Mul, result)
    }
}


impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self * operand2
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        let operand1 = Expr::new_leaf(self);
        operand1 * other
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        let result = self.result - other.result;
        Expr::new_binary(self, other, Operation::Sub, result)
    }
}

impl Sub<f64> for Expr {
    type Output = Expr;

    fn sub(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self - operand2
    }
}

impl Sub<Expr> for f64 {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        let operand1 = Expr::new_leaf(self);
        operand1 - other
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, other: Expr) -> Expr {
        let result = self.result / other.result;
        Expr::new_binary(self, other, Operation::Div, result)
    }
}

impl Div<f64> for Expr {
    type Output = Expr;

    fn div(self, other: f64) -> Expr {
        let operand2 = Expr::new_leaf(other);
        self / operand2
    }
}

impl Sum for Expr {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Expr::new_leaf(0.0), |acc, x| acc + x)
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
    fn test_backpropagation_add() {
        let operand1 = Expr::new_leaf(1.0);
        let operand2 = Expr::new_leaf(2.0);
        let mut expr3 = operand1 + operand2;

        expr3.grad = 2.0;
        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 2.0);
        assert_eq!(operand2.grad, 2.0);
    }

    #[test]
    fn test_backpropagation_sub() {
        let operand1 = Expr::new_leaf(1.0);
        let operand2 = Expr::new_leaf(2.0);
        let mut expr3 = operand1 - operand2;

        expr3.grad = 2.0;
        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 2.0);
        assert_eq!(operand2.grad, -2.0);
    }

    #[test]
    fn test_backpropagation_mul() {
        let operand1 = Expr::new_leaf(3.0);
        let operand2 = Expr::new_leaf(4.0);
        let mut expr3 = operand1 * operand2;

        expr3.grad = 2.0;
        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 8.0);
        assert_eq!(operand2.grad, 6.0);
    }

    #[test]
    fn test_backpropagation_div() {
        let operand1 = Expr::new_leaf(3.0);
        let operand2 = Expr::new_leaf(4.0);
        let mut expr3 = operand1 / operand2;

        expr3.grad = 2.0;
        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 0.5);
        assert_eq!(operand2.grad, -0.375);
    }

    #[test]
    fn test_backpropagation_tanh() {
        let operand1 = Expr::new_leaf(0.0);
        let mut expr2 = operand1.tanh();

        expr2.grad = 2.0;
        expr2.learn(1e-09);

        let operand1 = expr2.operand1.unwrap();
        assert_eq!(operand1.grad, 2.0);
    }

    #[test]
    fn test_backpropagation_relu() {
        let operand1 = Expr::new_leaf(-1.0);
        let mut expr2 = operand1.relu();

        expr2.grad = 2.0;
        expr2.learn(1e-09);

        let operand1 = expr2.operand1.unwrap();
        assert_eq!(operand1.grad, 0.0);
    }

    #[test]
    fn test_backpropagation_exp() {
        let operand1 = Expr::new_leaf(0.0);
        let mut expr2 = operand1.exp();

        expr2.grad = 2.0;
        expr2.learn(1e-09);

        let operand1 = expr2.operand1.unwrap();
        assert_eq!(operand1.grad, 2.0);
    }

    #[test]
    fn test_backpropagation_pow() {
        let operand1 = Expr::new_leaf(2.0);
        let operand2 = Expr::new_leaf(3.0);
        let mut expr3 = operand1.pow(operand2);

        expr3.grad = 2.0;
        expr3.learn(1e-09);

        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();
        assert_eq!(operand1.grad, 24.0);
        assert_eq!(operand2.grad, 11.090354888959125);
    }

    #[test]
    fn test_backpropagation_mixed_tree() {
        let operand1 = Expr::new_leaf(1.0);
        let operand2 = Expr::new_leaf(2.0);
        let expr3 = operand1 + operand2;
        let mut expr4 = expr3.tanh();

        expr4.grad = 2.0;
        expr4.learn(1e-09);

        let expr3 = expr4.operand1.unwrap();
        let operand1 = expr3.operand1.unwrap();
        let operand2 = expr3.operand2.unwrap();

        assert_eq!(expr3.grad, 0.019732074330880423);
        assert_eq!(operand1.grad, 0.019732074330880423);
        assert_eq!(operand2.grad, 0.019732074330880423);
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

        o.grad = 1.0;
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
        expr.grad = 1.0;
        expr.learn(1e-01);

        assert_float_eq(expr.result, 0.9);
    }

    #[test]
    fn test_learn_skips_non_learnable() {
        let mut expr = Expr::new_leaf(1.0);
        expr.set_learnable(false);
        expr.grad = 1.0;
        expr.learn(1e-01);

        assert_float_eq(expr.result, 1.0);
    }
}