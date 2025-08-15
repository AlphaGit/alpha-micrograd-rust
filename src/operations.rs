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
}


#[derive(Debug, PartialEq)]
pub(crate) enum OperationType {
    Leaf,
    Unary,
    Binary,
}
