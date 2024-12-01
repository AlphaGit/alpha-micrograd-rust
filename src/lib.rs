use std::cell::RefCell;
use std::ops::Add;
use std::ops::Mul;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct LeafExpr {
    data: f64,
    grad: f64,
}

#[derive(Debug, Clone)]
struct UnaryExpr {
    data: f64,
    operand: Rc<RefCell<Expr>>,
    operation: Operation,
    grad: f64,
}

#[derive(Debug, Clone)]
struct BinaryExpr {
    data: f64,
    operand1: Rc<RefCell<Expr>>,
    operand2: Rc<RefCell<Expr>>,
    operation: Operation,
    grad: f64,
}

#[derive(Debug, Clone)]
enum Expr {
    Leaf(LeafExpr),
    Unary(UnaryExpr),
    Binary(BinaryExpr),
}

impl LeafExpr {
    fn new(data: f64) -> Expr {
        let leaf = LeafExpr { data, grad: 0.0 };
        Expr::Leaf(leaf)
    }
}

impl UnaryExpr {
    fn new(data: f64, previous: Expr, operation: Operation) -> Expr {
        let operand = Rc::new(RefCell::new(previous));
        let unary = UnaryExpr {
            data,
            operand,
            operation,
            grad: 0.0,
        };
        Expr::Unary(unary)
    }
}

impl BinaryExpr {
    fn new(data: f64, operand1: Expr, operand2: Expr, operation: Operation) -> Expr {
        let operand1 = Rc::new(RefCell::new(operand1));
        let operand2 = Rc::new(RefCell::new(operand2));

        let binary = BinaryExpr {
            data,
            operand1,
            operand2,
            operation,
            grad: 0.0,
        };
        Expr::Binary(binary)
    }
}

impl Expr {
    fn tanh(self: Expr) -> Expr {
        let e_2x = (self.data() * 2.0).exp();
        let numerator = e_2x - 1.0;
        let denominator = e_2x + 1.0;

        return UnaryExpr::new(numerator / denominator, self, Operation::Tanh);
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
            Expr::Leaf(leaf) => leaf.grad += grad,
            Expr::Unary(unary) => unary.grad += grad,
            Expr::Binary(binary) => binary.grad += grad,
        }
    }

    fn backpropagate(&mut self) {
        let out_grad = self.grad();
        let out_data = self.data();
        match self {
            Expr::Leaf(_) => {}
            Expr::Unary(unary) => {
                let mut operand = unary.operand.borrow_mut();

                match unary.operation {
                    Operation::Tanh => {
                        let tanh_grad = 1.0 - out_data.powi(2);
                        operand.set_grad(out_grad * tanh_grad);
                    }
                    Operation::Add => {
                        panic!("Add is not a Unary operation.")
                    }
                    Operation::Mul => {
                        panic!("Mul is not a Unary operation.")
                    }
                }
            }
            Expr::Binary(binary) => {
                let mut operand1 = binary.operand1.borrow_mut();
                let mut operand2 = binary.operand2.borrow_mut();

                match binary.operation {
                    Operation::Add => {
                        operand1.set_grad(out_grad);
                        operand2.set_grad(out_grad);
                    }
                    Operation::Mul => {
                        operand1.set_grad(out_grad * operand2.data());
                        operand2.set_grad(out_grad * operand1.data());
                    }
                    Operation::Tanh => {
                        panic!("Tanh is not a Binary operation.")
                    }
                }
            }
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Operation {
    Add,
    Mul,
    Tanh,
}

impl Add for Expr {
    type Output = Expr;

    fn add(self: Self, rhs: Expr) -> Self::Output {
        let data = self.data() + rhs.data();
        BinaryExpr::new(data, self, rhs, Operation::Add)
    }
}

impl Add<i32> for Expr {
    type Output = Expr;

    fn add(self: Self, rhs: i32) -> Self::Output {
        let rhs = LeafExpr::new(rhs as f64);
        self + rhs
    }
}

impl Add<Expr> for i32 {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        let lhs = LeafExpr::new(self as f64);
        lhs + rhs
    }
}

impl Add<f64> for Expr {
    type Output = Expr;

    fn add(self: Self, rhs: f64) -> Self::Output {
        let rhs = LeafExpr::new(rhs);
        self + rhs
    }
}

impl Add<Expr> for f64 {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        let lhs = LeafExpr::new(self);
        lhs + rhs
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        let data = self.data() * rhs.data();
        BinaryExpr::new(data, self, rhs, Operation::Mul)
    }
}

impl Mul<i32> for Expr {
    type Output = Expr;

    fn mul(self, rhs: i32) -> Self::Output {
        let rhs = LeafExpr::new(rhs as f64);
        self * rhs
    }
}

impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = LeafExpr::new(rhs);
        self * rhs
    }
}

impl Mul<Expr> for i32 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        let lhs = LeafExpr::new(self as f64);
        lhs * rhs
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        let lhs = LeafExpr::new(self);
        lhs * rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_can_be_instantiated() {
        let value = LeafExpr::new(3.0);
        assert_eq!(value.data(), 3.0);
    }

    #[test]
    fn unary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = UnaryExpr::new(4.0, value1, Operation::Tanh);
        assert_eq!(value2.data(), 4.0);

        let value2 = assert_get_unary_expr(value2);
        let operand = value2.operand.borrow();

        assert_eq!(operand.data(), 3.0);
    }

    #[test]
    fn binary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);

        let value3 = BinaryExpr::new(5.0, value1, value2, Operation::Add);
        assert_eq!(value3.data(), 5.0);

        let value3 = assert_get_binary_expr(value3);
        assert_eq!(value3.operation, Operation::Add);

        let operand1 = value3.operand1.borrow();
        let operand2 = value3.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_add_expr() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = value1 + value2;
        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_add_int_lhs() {
        let value1 = LeafExpr::new(3.0);
        let result = value1 + 4;

        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_add_int_rhs() {
        let value1 = LeafExpr::new(3.0);
        let result = 4 + value1;

        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 4.0);
        assert_eq!(operand2.data(), 3.0);
    }

    #[test]
    fn can_add_float_rhs() {
        let value1 = LeafExpr::new(3.0);
        let result = value1 + 4.0;

        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_add_float_lhs() {
        let value1 = LeafExpr::new(3.0);
        let result = 4.0 + value1;

        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 4.0);
        assert_eq!(operand2.data(), 3.0);
    }

    #[test]
    fn can_multiply_expr() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = value1 * value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_multiply_int_rhs() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2 = 4;

        let result = value1 * value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_multiply_int_lhs() {
        let value1 = 4;
        let value2: Expr = LeafExpr::new(3.0).into();

        let result = value1 * value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 4.0);
        assert_eq!(operand2.data(), 3.0);
    }

    #[test]
    fn can_multiply_float_rhs() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2 = 4.0;

        let result = value1 * value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }
    #[test]
    fn can_multiply_float_lhs() {
        let value1 = 3.0;
        let value2 = LeafExpr::new(4.0);

        let result = value1 * value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1.borrow();
        let operand2 = result.operand2.borrow();

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }
    #[test]
    fn can_compute_tanh() {
        let value: Expr = LeafExpr::new(0.0).into();

        let result = value.tanh();
        assert_eq!(result.data(), 0.0);

        let result = assert_get_unary_expr(result);
        assert_eq!(result.operation, Operation::Tanh);

        let operand = result.operand.borrow();
        assert_eq!(operand.data(), 0.0);
    }

    #[test]
    fn addition_backpropagation() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();
        let mut addition = value1 + value2;

        addition.set_grad(2.0);
        addition.backpropagate();

        let addition = assert_get_binary_expr(addition);
        let operand1 = addition.operand1.borrow();
        let operand2 = addition.operand2.borrow();

        assert_eq!(operand1.grad(), 2.0);
        assert_eq!(operand2.grad(), 2.0);
    }

    #[test]
    fn multiplication_backpropagation() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();
        let mut multiplication = value1 * value2;

        multiplication.set_grad(2.0);
        multiplication.backpropagate();

        let multiplication = assert_get_binary_expr(multiplication);
        let operand1 = multiplication.operand1.borrow();
        let operand2 = multiplication.operand2.borrow();

        assert_eq!(operand1.grad(), 8.0);
        assert_eq!(operand2.grad(), 6.0);
    }

    #[test]
    fn tanh_backpropagation() {
        let value: Expr = LeafExpr::new(0.0).into();
        let mut tanh = value.tanh();

        tanh.set_grad(2.0);
        tanh.backpropagate();

        let tanh = assert_get_unary_expr(tanh);
        assert_eq!(tanh.grad, 2.0);
    }

    fn assert_get_binary_ref(e: std::cell::Ref<'_, Expr>) -> BinaryExpr {
        assert_get_binary_expr(e.clone())
    }

    fn assert_get_binary_expr(e: Expr) -> BinaryExpr {
        match e {
            Expr::Binary(binary) => binary,
            _ => panic!("Expected binary expression"),
        }
    }

    fn assert_get_unary_ref(e: std::cell::Ref<'_, Expr>) -> UnaryExpr {
        assert_get_unary_expr(e.clone())
    }

    fn assert_get_unary_expr(e: Expr) -> UnaryExpr {
        match e {
            Expr::Unary(unary) => unary,
            _ => panic!("Expected unary expression"),
        }
    }

    fn assert_get_leaf_ref(e: std::cell::Ref<'_, Expr>) -> LeafExpr {
        assert_get_leaf_expr(e.clone())
    }

    fn assert_get_leaf_expr(e: Expr) -> LeafExpr {
        match e {
            Expr::Leaf(leaf) => leaf,
            _ => panic!("Expected leaf expression"),
        }
    }

    fn assert_float_eq(f1: f64, f2: f64) {
        assert!(f1 - f1 < f64::EPSILON)
    }

    #[test]
    fn karpathys_example() {
        let x1: Expr = LeafExpr::new(2.0);
        let x2: Expr = LeafExpr::new(0.0);
        let w1: Expr = LeafExpr::new(-3.0);
        let w2: Expr = LeafExpr::new(1.0);
        let b: Expr = LeafExpr::new(6.8813735870195432);
        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1_x2w2 = x1w1 + x2w2;
        let n = x1w1_x2w2 + b;
        let mut o = n.tanh();

        o.set_grad(1.0);
        o.backpropagate();

        let o = assert_get_unary_expr(o);
        assert_float_eq(o.grad, 1.0);

        let n = o.operand.borrow();
        let n = assert_get_binary_ref(n);
        assert_float_eq(n.grad, 0.5);

        let x1w1_x2w2 = n.operand1.borrow();
        let x1w1_x2w2 = assert_get_binary_ref(x1w1_x2w2);
        assert_float_eq(x1w1_x2w2.grad, 0.5);

        let b = n.operand2.borrow();
        let b = assert_get_leaf_ref(b);
        assert_float_eq(b.grad, 0.5);

        let x1w1 = x1w1_x2w2.operand1.borrow();
        let x1w1 = assert_get_binary_ref(x1w1);
        assert_float_eq(x1w1.grad, 0.5);

        let x2w2 = x1w1_x2w2.operand2.borrow();
        let x2w2 = assert_get_binary_ref(x2w2);
        assert_float_eq(x2w2.grad, 0.5);

        let x1 = x1w1.operand1.borrow();
        let x1 = assert_get_leaf_ref(x1);
        assert_float_eq(x1.grad, -1.5);

        let w1 = x1w1.operand2.borrow();
        let w1 = assert_get_leaf_ref(w1);
        assert_float_eq(w1.grad, 1.0);

        let x2 = x2w2.operand1.borrow();
        let x2 = assert_get_leaf_ref(x2);
        assert_float_eq(x2.grad, 0.5);

        let w2 = x2w2.operand2.borrow();
        let w2 = assert_get_leaf_ref(w2);
        assert_float_eq(w2.grad, 0.0);
    }

    #[test]
    fn fix_same_object_error_addition() {
        let a = LeafExpr::new(3.0);
        let mut b = a.clone() + a.clone();

        b.backpropagate();

        let a = assert_get_leaf_expr(a);
        assert_float_eq(a.grad, 2.0);
    }

    #[test]
    fn crossing_paths_test() {
        let a = LeafExpr::new(-2.0);
        let b = LeafExpr::new(3.0);

        let d = a.clone() * b.clone();
        let e = a * b;
        let mut f = d + e;

        f.backpropagate();

        let f = assert_get_binary_expr(f);
        let d = f.operand1.borrow();
        let e = f.operand2.borrow();
        assert_float_eq(e.grad(), -6.0);
        assert_float_eq(d.grad(), 1.0);

        let d = assert_get_binary_expr(d.clone());
        let a = d.operand1.borrow();
        let b = d.operand2.borrow();
        assert_float_eq(a.grad(), -3.0);
        assert_float_eq(b.grad(), -8.0);

        let e = assert_get_binary_expr(e.clone());
        let a = e.operand1.borrow();
        let b = e.operand2.borrow();
        assert_float_eq(a.grad(), -3.0);
        assert_float_eq(b.grad(), -8.0);
    }
}
