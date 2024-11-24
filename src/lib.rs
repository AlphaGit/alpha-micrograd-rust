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

struct UnaryExpr {
    data: f64,
    operand: Box<dyn Expr>,
    operation: Operation,
    grad: f64,
}

impl UnaryExpr {
    fn new(data: f64, previous: Box<dyn Expr>, operation: Operation) -> UnaryExpr {
        let operand = previous;
        UnaryExpr {
            data,
            operand,
            operation,
            grad: 0.0,
        }
    }
}

struct BinaryExpr {
    data: f64,
    operand1: Box<dyn Expr>,
    operand2: Box<dyn Expr>,
    operation: Operation,
    grad: f64,
}

impl BinaryExpr {
    fn new(
        data: f64,
        operand1: Box<dyn Expr>,
        operand2: Box<dyn Expr>,
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

trait Expr {
    fn data(&self) -> f64;

    fn grad(&self) -> f64;
    fn backpropagate(&self);
}

impl Expr for LeafExpr {
    fn data(&self) -> f64 {
        self.data
    }

    fn grad(&self) -> f64 {
        self.grad
    }

    fn backpropagate(&self) {}
}

impl Expr for UnaryExpr {
    fn data(&self) -> f64 {
        self.data
    }

    fn grad(&self) -> f64 {
        self.grad
    }

    fn backpropagate(&self) {}
}

impl Expr for BinaryExpr {
    fn data(&self) -> f64 {
        self.data
    }

    fn grad(&self) -> f64 {
        self.grad
    }

    fn backpropagate(&self) {}
}

#[derive(PartialEq, Debug)]
enum Operation {
    Add,
    Mul,
    Tanh,
}

impl<T> Add<T> for LeafExpr
where
    T: Expr + 'static,
{
    type Output = BinaryExpr;

    fn add(self, other: T) -> BinaryExpr {
        let data = self.data() + other.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(other);
        BinaryExpr::new(data, operand1, operand2, Operation::Add)
    }
}

impl<T> Add<T> for UnaryExpr
where
    T: Expr + 'static,
{
    type Output = BinaryExpr;

    fn add(self, other: T) -> BinaryExpr {
        let data = self.data() + other.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(other);
        BinaryExpr::new(data, operand1, operand2, Operation::Add)
    }
}

impl<T> Add<T> for BinaryExpr
where
    T: Expr + 'static,
{
    type Output = BinaryExpr;

    fn add(self, other: T) -> BinaryExpr {
        let data = self.data() + other.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(other);
        BinaryExpr::new(data, operand1, operand2, Operation::Add)
    }
}

impl<T> Mul<T> for LeafExpr
where
    T: Expr + 'static,
{
    type Output = BinaryExpr;

    fn mul(self, other: T) -> BinaryExpr {
        let data = self.data() * other.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(other);
        BinaryExpr::new(data, operand1, operand2, Operation::Mul)
    }
}

impl<T> Mul<T> for UnaryExpr
where
    T: Expr + 'static,
{
    type Output = BinaryExpr;

    fn mul(self, other: T) -> BinaryExpr {
        let data = self.data() * other.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(other);
        BinaryExpr::new(data, operand1, operand2, Operation::Mul)
    }
}

impl<T> Mul<T> for BinaryExpr
where
    T: Expr + 'static,
{
    type Output = BinaryExpr;

    fn mul(self, other: T) -> BinaryExpr {
        let data = self.data() * other.data();
        let operand1 = Box::new(self);
        let operand2 = Box::new(other);
        BinaryExpr::new(data, operand1, operand2, Operation::Mul)
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
        let value2 = UnaryExpr::new(4.0, Box::new(value1), Operation::Tanh);
        assert_eq!(value2.data, 4.0);
        assert_eq!(value2.operand.data(), 3.0);
    }

    #[test]
    fn binary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);
        let value3 = BinaryExpr::new(5.0, Box::new(value1), Box::new(value2), Operation::Add);
        assert_eq!(value3.data, 5.0);
        assert_eq!(value3.operand1.data(), 3.0);
        assert_eq!(value3.operand2.data(), 4.0);
    }

    #[test]
    fn can_add() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);

        let result = value1 + value2;
        assert_eq!(result.data(), 7.0);
        assert_eq!(result.operand1.data(), 3.0);
        assert_eq!(result.operand2.data(), 4.0);
        assert_eq!(result.operation, Operation::Add);
    }

    #[test]
    fn can_multiply() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);
        let result = value1 * value2;
        assert_eq!(result.data, 12.0);
        assert_eq!(result.operand1.data(), 3.0);
        assert_eq!(result.operand2.data(), 4.0);
        assert_eq!(result.operation, Operation::Mul);
    }
}
