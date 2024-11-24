use std::borrow::BorrowMut;
use std::ops::Add;
use std::ops::Mul;

struct LeafExpr {
    data: f64,
    grad: f64,
}

impl LeafExpr {
    fn new(data: f64) -> LeafExpr {
        LeafExpr { data, grad: 0.0 }
    }
}

impl Into<Expr> for LeafExpr {
    fn into(self) -> Expr {
        Expr::Leaf(self)
    }
}

struct UnaryExpr {
    data: f64,
    operand: Box<Expr>,
    operation: Operation,
    grad: f64,
}

impl UnaryExpr {
    fn new(data: f64, previous: Box<Expr>, operation: Operation) -> UnaryExpr {
        let operand = previous;
        UnaryExpr {
            data,
            operand,
            operation,
            grad: 0.0,
        }
    }
}

impl Into<Expr> for UnaryExpr {
    fn into(self) -> Expr {
        Expr::Unary(self)
    }
}

struct BinaryExpr {
    data: f64,
    operand1: Box<Expr>,
    operand2: Box<Expr>,
    operation: Operation,
    grad: f64,
}

impl BinaryExpr {
    fn new(
        data: f64,
        operand1: Box<Expr>,
        operand2: Box<Expr>,
        operation: Operation,
    ) -> BinaryExpr {
        BinaryExpr {
            data,
            operand1,
            operand2,
            operation,
            grad: 0.0,
        }
    }
}

impl Into<Expr> for BinaryExpr {
    fn into(self) -> Expr {
        Expr::Binary(self)
    }
}

enum Expr {
    Leaf(LeafExpr),
    Unary(UnaryExpr),
    Binary(BinaryExpr),
}

impl Expr {
    fn tanh(self) -> Expr {
        let e_2x = self.data().powi(2).exp();
        let numerator = e_2x - 1.0;
        let denominator = e_2x + 1.0;

        let operand = Box::new(self);
        return UnaryExpr::new(numerator / denominator, operand, Operation::Tanh).into();
    }

    fn data(&self) -> f64 {
        match self {
            Expr::Leaf(leaf) => leaf.data,
            Expr::Unary(unary) => unary.data,
            Expr::Binary(binary) => binary.data,
        }
    }

    fn grad(&self) -> f64 {
        match self {
            Expr::Leaf(leaf) => leaf.grad,
            Expr::Unary(unary) => unary.grad,
            Expr::Binary(binary) => binary.grad,
        }
    }

    fn set_grad(&mut self, grad: f64) {
        match self {
            Expr::Leaf(leaf) => leaf.grad = grad,
            Expr::Unary(unary) => unary.grad = grad,
            Expr::Binary(binary) => binary.grad = grad,
        }
    }

    fn backpropagate(&mut self) {
        let out_grad = self.grad();
        match self {
            Expr::Leaf(_) => {}
            Expr::Unary(_) => {}
            Expr::Binary(binary) => {
                let operand1: &mut Expr = binary.operand1.borrow_mut();
                let operand2: &mut Expr = binary.operand2.borrow_mut();

                match binary.operation {
                    Operation::Add => {
                        operand1.set_grad(out_grad);
                        operand2.set_grad(out_grad);
                    }
                    Operation::Mul => {
                        operand1.set_grad(out_grad * operand2.data());
                        operand2.set_grad(out_grad * operand1.data());
                    }
                    Operation::Tanh => {}
                }
            }
        }
    }
}

#[derive(PartialEq, Debug)]
enum Operation {
    Add,
    Mul,
    Tanh,
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.data() + rhs.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(rhs);

        BinaryExpr::new(data, operand1, operand2, Operation::Add).into()
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.data() * rhs.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(rhs);

        BinaryExpr::new(data, operand1, operand2, Operation::Mul).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_can_be_instantiated() {
        let value = LeafExpr::new(3.0);
        assert_eq!(value.data, 3.0);
    }

    #[test]
    fn unary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = UnaryExpr::new(4.0, Box::new(value1.into()), Operation::Tanh);
        assert_eq!(value2.data, 4.0);
        assert_eq!(value2.operand.data(), 3.0);
    }

    #[test]
    fn binary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);
        let value3 = BinaryExpr::new(
            5.0,
            Box::new(value1.into()),
            Box::new(value2.into()),
            Operation::Add,
        );
        assert_eq!(value3.data, 5.0);
        assert_eq!(value3.operand1.data(), 3.0);
        assert_eq!(value3.operand2.data(), 4.0);
    }

    #[test]
    fn can_add() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = value1 + value2;
        assert_eq!(result.data(), 7.0);

        if let Expr::Binary(binary) = result {
            assert_eq!(binary.operand1.data(), 3.0);
            assert_eq!(binary.operand2.data(), 4.0);
            assert_eq!(binary.operation, Operation::Add);
        } else {
            assert_eq!(false, true)
        }
    }

    #[test]
    fn can_multiply() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = value1 * value2;
        assert_eq!(result.data(), 12.0);

        if let Expr::Binary(binary) = result {
            assert_eq!(binary.operand1.data(), 3.0);
            assert_eq!(binary.operand2.data(), 4.0);
            assert_eq!(binary.operation, Operation::Mul);
        } else {
            assert_eq!(false, true)
        }
    }

    #[test]
    fn can_compute_tanh() {
        let value: Expr = LeafExpr::new(0.0).into();

        let result = value.tanh();
        assert_eq!(result.data(), 0.0);

        if let Expr::Unary(unary) = result {
            assert_eq!(unary.operand.data(), 0.0);
            assert_eq!(unary.operation, Operation::Tanh);
        } else {
            assert_eq!(false, true)
        }
    }

    #[test]
    fn addition_backpropagation() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();
        let mut addition = value1 + value2;

        addition.set_grad(2.0);
        addition.backpropagate();

        if let Expr::Binary(binary) = addition {
            assert_eq!(binary.operand1.grad(), 2.0);
            assert_eq!(binary.operand2.grad(), 2.0);
        } else {
            assert_eq!(false, true)
        }
    }

    #[test]
    fn multiplication_backpropagation() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();
        let mut multiplication = value1 * value2;

        multiplication.set_grad(2.0);
        multiplication.backpropagate();

        if let Expr::Binary(binary) = multiplication {
            assert_eq!(binary.operand1.grad(), 8.0);
            assert_eq!(binary.operand2.grad(), 6.0);
        } else {
            assert_eq!(false, true)
        }
    }
}
