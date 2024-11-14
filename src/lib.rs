use std::ops::Add;
use std::ops::Mul;

#[derive(Debug)]
struct Expr {
    data: f64,
    operand1: Option<Box<Expr>>,
    operand2: Option<Box<Expr>>,
    operation: Option<Operation>,
}

#[derive(Debug, PartialEq)]
enum Operation {
    Add,
    Mul,
    Tanh,
}

impl Expr {
    fn new(data: f64) -> Expr {
        Expr {
            data,
            operand1: None,
            operand2: None,
            operation: None,
        }
    }

    fn new_unary(data: f64, previous: Expr, operation: Operation) -> Expr {
        Expr {
            data,
            operand1: Some(Box::new(previous)),
            operand2: None,
            operation: Some(operation),
        }
    }

    fn new_binary(data: f64, operand1: Expr, operand2: Expr, operation: Operation) -> Expr {
        Expr {
            data,
            operand1: Some(Box::new(operand1)),
            operand2: Some(Box::new(operand2)),
            operation: Some(operation),
        }
    }

    fn tanh(self) -> Expr {
        let e_2x = self.data.powi(2).exp();
        let numerator = e_2x - 1.0;
        let denominator = e_2x + 1.0;
        return Expr::new_unary(numerator / denominator, self, Operation::Tanh);
    }
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        Expr::new_binary(self.data + other.data, self, other, Operation::Add)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        Expr::new_binary(self.data * other.data, self, other, Operation::Mul)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_be_instantiated() {
        let value = Expr::new(3.0);
        assert_eq!(value.data, 3.0);
    }

    #[test]
    fn can_be_instantiated_with_dependencies() {
        let value1 = Expr::new(3.0);
        let value2 = Expr::new(4.0);
        let value3 = Expr::new_binary(5.0, value1, value2, Operation::Add);

        assert_eq!(value3.data, 5.0);
        assert_eq!(value3.operand1.unwrap().data, 3.0);
        assert_eq!(value3.operand2.unwrap().data, 4.0);
    }

    #[test]
    fn can_add() {
        let value1 = Expr::new(3.0);
        let value2 = Expr::new(4.0);

        let result = value1 + value2;
        assert_eq!(result.data, 7.0);
        assert_eq!(result.operand1.unwrap().data, 3.0);
        assert_eq!(result.operand2.unwrap().data, 4.0);
        assert_eq!(result.operation, Some(Operation::Add));
    }

    #[test]
    fn can_multiply() {
        let value1 = Expr::new(3.0);
        let value2 = Expr::new(4.0);
        let result = value1 * value2;
        assert_eq!(result.data, 12.0);
        assert_eq!(result.operand1.unwrap().data, 3.0);
        assert_eq!(result.operand2.unwrap().data, 4.0);
        assert_eq!(result.operation, Some(Operation::Mul));
    }

    #[test]
    fn can_compute_tanh() {
        let value = Expr::new(0.0);
        let result = value.tanh();
        assert_eq!(result.data, 0.0);
        assert_eq!(result.operand1.unwrap().data, 0.0);
        assert_eq!(result.operation, Some(Operation::Tanh));
    }
}
