use std::iter::Sum;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::convert::From;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct LeafExpr {
    data: f64,
    grad: f64,
}

#[derive(Debug, Clone)]
pub struct UnaryExpr {
    data: f64,
    grad: f64,
    operand: Rc<&Expr>,
    operation: Operation,
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    data: f64,
    grad: f64,
    operand1: Rc<&Expr>,
    operand2: Rc<&Expr>,
    operation: Operation,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Leaf(LeafExpr),
    Unary(UnaryExpr),
    Binary(BinaryExpr),
}

impl LeafExpr {
    pub fn new(data: f64) -> Expr {
        let leaf = LeafExpr {
            data: data,
            grad: 0.0,
        };
        Expr::Leaf(leaf)
    }
}

impl From<f64> for Expr {
    fn from(data: f64) -> Expr {
        LeafExpr::new(data)
    }
}

impl From<f32> for Expr {
    fn from(data: f32) -> Self {
        Expr::from(data as f64)
    }
}

impl From<i32> for Expr {
    fn from(data: i32) -> Self {
        Expr::from(data as f64)
    }
}

impl UnaryExpr {
    fn new(data: f64, previous: &Rc<Expr>, operation: Operation) -> Expr {
        let unary = UnaryExpr {
            data: data,
            grad: 0.0,
            operand: Rc::clone(previous),
            operation,
        };
        Expr::Unary(unary)
    }
}

impl BinaryExpr {
    fn new(
        data: f64,
        operand1: &Rc<Expr>,
        operand2: &Rc<Expr>,
        operation: Operation,
    ) -> Expr {
        let binary = BinaryExpr {
            data: data,
            grad: 0.0,
            operand1: Rc::clone(operand1),
            operand2: Rc::clone(operand2),
            operation,
        };
        Expr::Binary(binary)
    }
}

impl Expr {
    pub fn tanh(self) -> Expr {
        let e_2x = (self.data() * 2.0).exp();
        let numerator = e_2x - 1.0;
        let denominator = e_2x + 1.0;
        let previous = Rc::new(self);

        UnaryExpr::new(numerator / denominator, &previous, Operation::Tanh)
    }

    pub fn exp(self) -> Expr {
        let data = self.data().exp();
        UnaryExpr::new(data, self, Operation::Exp)
    }

    pub fn pow(&self, exponent: &'a Expr) -> Expr {
        let data = self.data().powf(exponent.data());
        BinaryExpr::new(data, self, exponent, Operation::Pow)
    }

    pub fn data(&self) -> f64 {
        match self {
            Expr::Leaf(leaf) => leaf.data.get(),
            Expr::Unary(unary) => unary.data.get(),
            Expr::Binary(binary) => binary.data.get(),
        }
    }

    pub fn set_data(&mut self, data: f64) {
        match self {
            Expr::Leaf(leaf) => leaf.data.set(data),
            Expr::Unary(unary) => unary.data.set(data),
            Expr::Binary(binary) => binary.data.set(data),
        }
    }

    fn grad(&self) -> f64 {
        match self {
            Expr::Leaf(leaf) => leaf.grad.get(),
            Expr::Unary(unary) => unary.grad.get(),
            Expr::Binary(binary) => binary.grad.get(),
        }
    }

    pub fn backpropagate(&mut self) {
        let out_grad = self.grad();
        let out_data = self.data();
        match self {
            Expr::Leaf(_) => {
                // TODO
            }
            Expr::Unary(unary) => {
                let operand = unary.operand;

                match unary.operation {
                    Operation::Add => {
                        panic!("Add is not a Unary operation.")
                    }
                    Operation::Sub => {
                        panic!("Sub is not a Unary operation.")
                    }
                    Operation::Mul => {
                        panic!("Mul is not a Unary operation.")
                    }
                    Operation::Tanh => {
                        let tanh_grad = 1.0 - out_data.powi(2);
                        operand.incr_grad(out_grad * tanh_grad);
                    }
                    Operation::Exp => {
                        operand.incr_grad(out_grad * out_data);
                    }
                    Operation::Pow => {
                        panic!("Pow is not a Unary operation.")
                    }
                }
            }
            Expr::Binary(binary) => {
                let operand1 = binary.operand1;
                let operand2 = binary.operand2;

                match binary.operation {
                    Operation::Add => {
                        operand1.incr_grad(out_grad);
                        operand2.incr_grad(out_grad);
                    }
                    Operation::Sub => {
                        operand1.incr_grad(out_grad);
                        operand2.incr_grad(-out_grad);
                    }
                    Operation::Mul => {
                        operand1.incr_grad(out_grad * operand2.data());
                        operand2.incr_grad(out_grad * operand1.data());
                    }
                    Operation::Tanh => {
                        panic!("Tanh is not a Binary operation.")
                    }
                    Operation::Exp => {
                        panic!("Exp is not a Binary operation.")
                    }
                    Operation::Pow => {
                        let exponent = operand2.data();
                        operand1.incr_grad(out_grad * exponent * out_data.powf(exponent - 1.0));
                        operand2.incr_grad(out_grad * out_data.powf(exponent) * out_data.ln());
                    }
                }
            }
        }
    }

    fn incr_grad(&self, grad: f64) {
        match self {
            Expr::Leaf(leaf) => leaf.grad.set(leaf.grad.get() + grad),
            Expr::Unary(unary) => unary.grad.set(unary.grad.get() + grad),
            Expr::Binary(binary) => binary.grad.set(binary.grad.get() + grad),
        }
    }

    pub fn reset_grads(&self) {
        match self {
            Expr::Leaf(leaf) => leaf.grad.set(0.0),
            Expr::Unary(unary) => {
                unary.grad.set(0.0);
                unary.operand.reset_grads();
            }
            Expr::Binary(binary) => {
                binary.grad.set(0.0);
                binary.operand1.reset_grads();
                binary.operand2.reset_grads();
            }
        }
    }

    pub fn learn_grads(&self, learning_rate: f64) {
        match self {
            Expr::Leaf(leaf) => leaf
                .data
                .set(leaf.data.get() - leaf.grad.get() * learning_rate),
            Expr::Unary(unary) => {
                unary
                    .data
                    .set(unary.data.get() - unary.grad.get() * learning_rate);
                unary.operand.learn_grads(learning_rate);
            }
            Expr::Binary(binary) => {
                binary
                    .data
                    .set(binary.data.get() - binary.grad.get() * learning_rate);
                binary.operand1.learn_grads(learning_rate);
                binary.operand2.learn_grads(learning_rate);
            }
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Operation {
    Add,
    Sub,
    Mul,
    Pow,
    Tanh,
    Exp,
}

impl<'a> Add for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn add(self, rhs: &'a Expr<'a>) -> Self::Output {
        let data = self.data() + rhs.data();
        BinaryExpr::new(data, self, rhs, Operation::Add)
    }
}

impl<'a> Mul for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn mul(self, rhs: &'a Expr) -> Self::Output {
        let data = self.data() * rhs.data();
        BinaryExpr::new(data, self, rhs, Operation::Mul)
    }
}

impl<'a> Sub for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn sub(self, rhs: &'a Expr<'a>) -> Self::Output {
        BinaryExpr::new(self.data() - rhs.data(), self, rhs, Operation::Sub)
    }
}

impl<'a> Sum for Expr<'a> {
    fn sum<I>(iter: I) -> Expr<'a>
    where
        I: Iterator<Item = Self>,
    {
        let mut sum = 0.0;
        for x in iter {
            sum += x.data();
        }
        LeafExpr::new(sum)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use super::*;

    #[test]
    fn leaf_can_be_instantiated() {
        let value = LeafExpr::new(3.0);
        assert_eq!(value.data(), 3.0);
    }

    #[test]
    fn unary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = UnaryExpr::new(4.0, &value1, Operation::Tanh);
        assert_eq!(value2.data(), 4.0);

        let value2 = assert_get_unary_expr(&value2);
        let operand = value2.operand;

        assert_eq!(operand.data(), 3.0);
    }

    #[test]
    fn binary_can_be_instantiated() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);

        let value3 = BinaryExpr::new(5.0, &value1, &value2, Operation::Add);
        assert_eq!(value3.data(), 5.0);

        let value3 = assert_get_binary_expr(&value3);
        assert_eq!(value3.operation, Operation::Add);

        let operand1 = value3.operand1;
        let operand2 = value3.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_add_expr() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = &value1 + &value2;
        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(&result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1;
        let operand2 = result.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_add_int() {
        let value1 = LeafExpr::new(3.0);
        let value2 = LeafExpr::new(4.0);
        let result = &value1 + &value2;

        assert_eq!(result.data(), 7.0);

        let result = assert_get_binary_expr(&result);
        assert_eq!(result.operation, Operation::Add);

        let operand1 = result.operand1;
        let operand2 = result.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_multiply_expr() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = &value1 * &value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(&result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1;
        let operand2 = result.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_multiply_int() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2 = Expr::from(4);

        let result = &value1 * &value2;
        assert_eq!(result.data(), 12.0);

        let result = assert_get_binary_expr(&result);
        assert_eq!(result.operation, Operation::Mul);

        let operand1 = result.operand1;
        let operand2 = result.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_sub_expr() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();

        let result = &value1 - &value2;
        assert_eq!(result.data(), -1.0);

        let result = assert_get_binary_expr(&result);
        assert_eq!(result.operation, Operation::Sub);

        let operand1 = result.operand1;
        let operand2 = result.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_sub_float() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2 = Expr::from(4.0);

        let result = &value1 - &value2;
        assert_eq!(result.data(), -1.0);

        let result = assert_get_binary_expr(&result);
        assert_eq!(result.operation, Operation::Sub);

        let operand1 = result.operand1;
        let operand2 = result.operand2;

        assert_eq!(operand1.data(), 3.0);
        assert_eq!(operand2.data(), 4.0);
    }

    #[test]
    fn can_compute_tanh() {
        let value: Expr = LeafExpr::new(0.0).into();

        let result = value.tanh();
        assert_eq!(result.data(), 0.0);

        let result = assert_get_unary_expr(&result);
        assert_eq!(result.operation, Operation::Tanh);

        let operand = result.operand;
        assert_eq!(operand.data(), 0.0);
    }

    #[test]
    fn can_compute_exp() {
        let value = LeafExpr::new(0.0);

        let result = value.exp();
        assert_eq!(result.data(), 1.0);

        let result = assert_get_unary_expr(&result);
        assert_eq!(result.operation, Operation::Exp);

        let operand = result.operand;
        assert_eq!(operand.data(), 0.0);
    }

    #[test]
    fn addition_backpropagation() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();
        let mut addition = &value1 + &value2;

        addition.incr_grad(2.0);
        addition.backpropagate();

        let addition = assert_get_binary_expr(&addition);
        let operand1 = addition.operand1;
        let operand2 = addition.operand2;

        assert_eq!(operand1.grad(), 2.0);
        assert_eq!(operand2.grad(), 2.0);
    }

    #[test]
    fn multiplication_backpropagation() {
        let value1: Expr = LeafExpr::new(3.0).into();
        let value2: Expr = LeafExpr::new(4.0).into();
        let mut multiplication = &value1 * &value2;

        multiplication.incr_grad(2.0);
        multiplication.backpropagate();

        let multiplication = assert_get_binary_expr(&multiplication);
        let operand1 = multiplication.operand1;
        let operand2 = multiplication.operand2;

        assert_eq!(operand1.grad(), 8.0);
        assert_eq!(operand2.grad(), 6.0);
    }

    #[test]
    fn tanh_backpropagation() {
        let value: Expr = LeafExpr::new(0.0).into();
        let mut tanh = value.tanh();

        tanh.incr_grad(2.0);
        tanh.backpropagate();

        let tanh = assert_get_unary_expr(&tanh);
        assert_eq!(tanh.grad.get(), 2.0);
    }

    fn assert_get_binary_expr<'a>(e: &'a Expr<'a>) -> &'a BinaryExpr<'a> {
        match e {
            Expr::Binary(binary) => binary,
            _ => panic!("Expected binary expression"),
        }
    }

    fn assert_get_unary_expr<'a>(e: &'a Expr<'a>) -> &'a UnaryExpr<'a> {
        match e {
            Expr::Unary(unary) => unary,
            _ => panic!("Expected unary expression"),
        }
    }

    fn assert_get_leaf_expr<'a>(e: &'a Expr<'a>) -> &'a LeafExpr {
        match e {
            Expr::Leaf(leaf) => leaf,
            _ => panic!("Expected leaf expression"),
        }
    }

    fn assert_float_eq(f1: f64, f2: f64) {
        assert!(f1 - f2 < f64::EPSILON)
    }

    #[test]
    fn karpathys_example() {
        let x1: Expr = LeafExpr::new(2.0);
        let x2: Expr = LeafExpr::new(0.0);
        let w1: Expr = LeafExpr::new(-3.0);
        let w2: Expr = LeafExpr::new(1.0);
        let b: Expr = LeafExpr::new(6.8813735870195432);
        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1_x2w2 = &x1w1 + &x2w2;
        let n = &x1w1_x2w2 + &b;
        let mut o = n.tanh();

        o.incr_grad(1.0);
        o.backpropagate();

        let o = assert_get_unary_expr(&o);
        assert_float_eq(o.grad.get(), 1.0);

        let n = o.operand;
        let n = assert_get_binary_expr(n);
        assert_float_eq(n.grad.get(), 0.5);

        let x1w1_x2w2 = n.operand1;
        let x1w1_x2w2 = assert_get_binary_expr(x1w1_x2w2);
        assert_float_eq(x1w1_x2w2.grad.get(), 0.5);

        let b = n.operand2;
        let b = assert_get_leaf_expr(&b);
        assert_float_eq(b.grad.get(), 0.5);

        let x1w1 = x1w1_x2w2.operand1;
        let x1w1 = assert_get_binary_expr(x1w1);
        assert_float_eq(x1w1.grad.get(), 0.5);

        let x2w2 = x1w1_x2w2.operand2;
        let x2w2 = assert_get_binary_expr(x2w2);
        assert_float_eq(x2w2.grad.get(), 0.5);

        let x1 = x1w1.operand1;
        let x1 = assert_get_leaf_expr(&x1);
        assert_float_eq(x1.grad.get(), -1.5);

        let w1 = x1w1.operand2;
        let w1 = assert_get_leaf_expr(&w1);
        assert_float_eq(w1.grad.get(), 1.0);

        let x2 = x2w2.operand1;
        let x2 = assert_get_leaf_expr(&x2);
        assert_float_eq(x2.grad.get(), 0.5);

        let w2 = x2w2.operand2;
        let w2 = assert_get_leaf_expr(&w2);
        assert_float_eq(w2.grad.get(), 0.0);
    }

    #[test]
    fn fix_same_object_error_addition() {
        let a = LeafExpr::new(3.0);
        let mut b = &a + &a;

        b.backpropagate();

        let a = assert_get_leaf_expr(&a);
        assert_float_eq(a.grad.get(), 2.0);
    }

    #[test]
    fn crossing_paths_test() {
        let a = LeafExpr::new(-2.0);
        let b = LeafExpr::new(3.0);

        let d = &a * &b;
        let e = &a * &b;
        let mut f = &d + &e;

        f.backpropagate();

        let f = assert_get_binary_expr(&f);
        let d = f.operand1;
        let e = f.operand2;
        assert_float_eq(e.grad(), -6.0);
        assert_float_eq(d.grad(), 1.0);

        let d = assert_get_binary_expr(&d);
        let a = d.operand1;
        let b = d.operand2;
        assert_float_eq(a.grad(), -3.0);
        assert_float_eq(b.grad(), -8.0);

        let e = assert_get_binary_expr(&e);
        let a = e.operand1;
        let b = e.operand2;
        assert_float_eq(a.grad(), -3.0);
        assert_float_eq(b.grad(), -8.0);
    }
}
