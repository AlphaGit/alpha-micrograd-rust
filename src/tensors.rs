use std::vec;

use crate::operations::{Operation, OperationType};

struct TensorShape {
    dimensions: Vec<usize>,
}

impl TensorShape {
    fn new(dimensions: Vec<usize>) -> Self {
        TensorShape { dimensions }
    }

    fn size(&self) -> usize {
        self.dimensions.iter().product()
    }
}

struct Tensor {
    data: Vec<f32>,
    shape: TensorShape,
}

impl Tensor {
    fn zeros(shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        let size = shape.size();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        assert_eq!(
            data.len(),
            shape.size(),
            "Data size does not match shape size"
        );
        Tensor { data, shape }
    }
}

struct TensorExpression {
    operand1: Option<Box<TensorExpression>>,
    operand2: Option<Box<TensorExpression>>,
    operation: Operation,
    result: Tensor,
    is_learnable: bool,
    grad: Option<Tensor>,
    name: Option<String>,
}

impl TensorExpression {
    fn new_leaf(tensor: Tensor, is_learnable: bool, name: Option<String>) -> Self {
        TensorExpression {
            operand1: None,
            operand2: None,
            operation: Operation::None,
            result: tensor,
            is_learnable,
            grad: None,
            name,
        }
    }

    fn new_leaf_with_name(tensor: Tensor, name: String) -> Self {
        TensorExpression::new_leaf(tensor, false, Some(name))
    }

    fn new_unary(operand: TensorExpression, operation: Operation, result: Tensor) -> Self {
        operation.assert_is_type(OperationType::Unary);
        TensorExpression {
            operand1: Some(Box::new(operand)),
            operand2: None,
            operation,
            result,
            is_learnable: false,
            grad: None,
            name: None,
        }
    }

    fn new_binary(
        operand1: TensorExpression,
        operand2: TensorExpression,
        operation: Operation,
        result: Tensor,
    ) -> Self {
        operation.assert_is_type(OperationType::Binary);
        TensorExpression {
            operand1: Some(Box::new(operand1)),
            operand2: Some(Box::new(operand2)),
            operation,
            result,
            is_learnable: false,
            grad: None,
            name: None,
        }
    }

    fn tanh(self) -> Self {
        let result = self
            .result
            .data
            .iter()
            .map(|&x| x.tanh())
            .collect::<Vec<f32>>();
        let result_tensor = Tensor::from_data(result, self.result.shape.dimensions.clone());

        TensorExpression::new_unary(self, Operation::Tanh, result_tensor)
    }

    fn relu(self) -> Self {
        let result = self
            .result
            .data
            .iter()
            .map(|&x| x.max(0.0))
            .collect::<Vec<f32>>();
        let result_tensor = Tensor::from_data(result, self.result.shape.dimensions.clone());

        TensorExpression::new_unary(self, Operation::ReLU, result_tensor)
    }

    fn exp(self) -> Self {
        let result = self
            .result
            .data
            .iter()
            .map(|&x| x.exp())
            .collect::<Vec<f32>>();
        let result_tensor = Tensor::from_data(result, self.result.shape.dimensions.clone());

        TensorExpression::new_unary(self, Operation::Exp, result_tensor)
    }

    fn pow(self, exponent: f32) -> Self {
        let result = self
            .result
            .data
            .iter()
            .map(|&x| x.powf(exponent))
            .collect::<Vec<f32>>();

        let result_tensor = Tensor::from_data(result, self.result.shape.dimensions.clone());

        let exponent_tensor = Tensor::from_data(vec![exponent], vec![1]);
        let exponent_tensor = TensorExpression::new_leaf(exponent_tensor, false, None);

        TensorExpression::new_binary(self, exponent_tensor, Operation::Pow, result_tensor)
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = vec![2, 3];
        let tensor = Tensor::zeros(shape.clone());
        assert_eq!(tensor.shape.dimensions, shape);
        assert_eq!(tensor.data.len(), 6);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_multi_dimensional() {
        let shape = vec![2, 3, 4];
        let tensor = Tensor::zeros(shape.clone());
        assert_eq!(tensor.shape.dimensions, shape);
        assert_eq!(tensor.data.len(), 24);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::from_data(data.clone(), shape.clone());
        assert_eq!(tensor.shape.dimensions, shape);
        assert_eq!(tensor.data, data);
    }

    #[test]
    #[should_panic(expected = "Data size does not match shape size")]
    fn test_tensor_from_data_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2];
        Tensor::from_data(data, shape);
    }

    #[test]
    fn test_tensor_shape_size() {
        let shape = vec![2, 3, 4];
        let tensor_shape = TensorShape::new(shape);
        assert_eq!(tensor_shape.size(), 24);
    }
}
