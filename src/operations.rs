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
        match self {
            Operation::None => assert_eq!(operation_type, OperationType::Leaf),
            Operation::Tanh
            | Operation::Exp
            | Operation::ReLU
            | Operation::Log
            | Operation::Neg => assert_eq!(operation_type, OperationType::Unary),
            _ => assert_eq!(operation_type, OperationType::Binary),
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum OperationType {
    Leaf,
    Unary,
    Binary,
}
