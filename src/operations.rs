#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Operation {
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
    pub(crate) fn assert_is_type(&self, operation_type: OperationType) {
        assert_eq!(self.expr_type(), operation_type, "Operation type mismatch: expected {:?}, found {:?}", operation_type, self.expr_type());
    }

    pub(crate) fn expr_type(&self) -> OperationType {
        match self {
            Operation::None => OperationType::Leaf,
            Operation::Tanh | Operation::Exp | Operation::ReLU | Operation::Log | Operation::Neg => OperationType::Unary,
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Pow => OperationType::Binary,
        }
    }

    pub(crate) fn get_binary_operation_lambda(
        &self,
    ) -> Box<dyn for<'a> FnMut((&'a f64, &'a f64)) -> f64> {
        assert_eq!(self.expr_type(), OperationType::Binary, "Operation is not binary");

        match self {
            Operation::Add => Box::new(|(a, b)| a + b),
            Operation::Sub => Box::new(|(a, b)| a - b),
            Operation::Mul => Box::new(|(a, b)| a * b),
            Operation::Div => Box::new(|(a, b)| a / b),
            Operation::Pow => Box::new(|(a, b)| a.powf(*b)),
            _ => panic!("Unsupported binary operation: {self:?}"),
        }
    }

    pub(crate) fn get_unary_operation_lambda(&self) -> Box<dyn FnMut(&f64) -> f64> {
        assert_eq!(self.expr_type(), OperationType::Unary, "Operation is not unary");
        match self {
            Operation::Neg => Box::new(|x| -x),
            Operation::Exp => Box::new(|x| x.exp()),
            Operation::Log => Box::new(|x| x.ln()),
            Operation::Tanh => Box::new(|x| x.tanh()),
            Operation::ReLU => Box::new(|x| x.max(0.0)),
            _ => panic!("Unsupported unary operation: {self:?}"),
        }
    }
}


#[derive(Debug, PartialEq)]
pub(crate) enum OperationType {
    Leaf,
    Unary,
    Binary,
}
