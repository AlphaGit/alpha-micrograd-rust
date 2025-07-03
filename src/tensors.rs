use std::{ops::Add, vec};

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

    fn broadcast(shape1: &Self, shape2: &Self) -> Self {
        let max_len = shape1.dimensions.len().max(shape2.dimensions.len());
        let mut dimensions1 = shape1.dimensions.clone();
        dimensions1.reverse();
        dimensions1.resize_with(max_len, || { 1usize });
        dimensions1.reverse();

        let mut dimensions2 = shape2.dimensions.clone();
        dimensions2.reverse();
        dimensions2.resize_with(max_len, || { 1usize });
        dimensions2.reverse();

        let mut new_shape = vec![1usize; max_len];

        for i in 0..max_len {
            let dim1 = dimensions1.get(i).cloned().unwrap_or(1);
            let dim2 = dimensions2.get(i).cloned().unwrap_or(1);

            if dim1 == 1 {
                new_shape[i] = dim2;
            } else if dim2 == 1 {
                new_shape[i] = dim1;
            } else if dim1 == dim2 {
                new_shape[i] = dim1;
            } else {
                panic!("Shapes are not compatible for broadcasting: {:?} and {:?}", shape1.dimensions, shape2.dimensions);
            }
        } 

        TensorShape::new(new_shape)
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

    fn from_scalar(value: f32) -> Self {
        Tensor {
            data: vec![value],
            shape: TensorShape::new(vec![1]),
        }
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

        let exponent_tensor = Tensor::from_scalar(exponent);
        let exponent_tensor = TensorExpression::new_leaf(exponent_tensor, false, None);

        TensorExpression::new_binary(self, exponent_tensor, Operation::Pow, result_tensor)
    }
}

impl Add for TensorExpression {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(
            self.result.shape.dimensions,
            other.result.shape.dimensions,
            "Tensors must have the same shape for addition"
        );

        let result_data: Vec<f32> = self
            .result
            .data
            .iter()
            .zip(other.result.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let result_tensor = Tensor::from_data(result_data, self.result.shape.dimensions.clone());

        TensorExpression::new_binary(self, other, Operation::Add, result_tensor)
    }
}

impl Add<f32> for TensorExpression {
    type Output = Self;

    fn add(self, scalar: f32) -> Self::Output {
        let scalar_tensor = Tensor::from_scalar(scalar);
        let scalar_expression = TensorExpression::new_leaf(scalar_tensor, false, None);

        self + scalar_expression
    }
}

#[cfg(test)]
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

    #[test]
    fn test_tensor_expression_add() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 2];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 + expr2;

        // Check that result is correctly computed
        assert_eq!(result.result.data, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result.result.shape.dimensions, shape);

        // Check that operation type is correctly set
        assert_eq!(result.operation, Operation::Add);

        // Check that operands are correctly set
        assert!(result.operand1.is_some());
        assert!(result.operand2.is_some());
    }

    #[test]
    fn test_tensor_expression_add_scalar() {
        let data = vec![1.0];
        let shape = vec![1];
        let scalar = 5.0;

        let tensor = Tensor::from_data(data, shape.clone());
        let expr = TensorExpression::new_leaf(tensor, false, None);
 
        let result = expr + scalar;

        // Check that result is correctly computed
        assert_eq!(result.result.data, vec![6.0]);
        assert_eq!(result.result.shape.dimensions, shape);

        // Check that operation type is correctly set
        assert_eq!(result.operation, Operation::Add);

        // Check that operands are correctly set
        assert!(result.operand1.is_some());
        assert!(result.operand2.is_some());

        // Check that the scalar was converted to a tensor expression with a scalar tensor
        if let Some(op2) = &result.operand2 {
            assert_eq!(op2.result.data, vec![scalar]);
            assert_eq!(op2.result.shape.dimensions, vec![1]);
        }
    }

    #[test]
    #[should_panic(expected = "Tensors must have the same shape for addition")]
    fn test_tensor_expression_add_shape_mismatch() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0];

        let tensor1 = Tensor::from_data(data1, vec![2, 2]);
        let tensor2 = Tensor::from_data(data2, vec![1, 2]);

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let _result = expr1 + expr2; // This should panic
    }

    #[test]
    fn test_tensor_expression_chained_add() {
        let data1 = vec![1.0, 2.0];
        let data2 = vec![3.0, 4.0];
        let data3 = vec![5.0, 6.0];
        let shape = vec![1, 2];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());
        let tensor3 = Tensor::from_data(data3, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);
        let expr3 = TensorExpression::new_leaf(tensor3, false, None);

        // Chain multiple additions: expr1 + expr2 + expr3
        let result = expr1 + expr2 + expr3;

        // Check that result is correctly computed
        assert_eq!(result.result.data, vec![9.0, 12.0]);
        assert_eq!(result.result.shape.dimensions, shape);

        // Check that operation type is correctly set
        assert_eq!(result.operation, Operation::Add);
    }

    #[test]
    fn test_tensor_expression_mixed_add() {
        let data = vec![1.0];
        let shape = vec![1];
        
        let tensor = Tensor::from_data(data, shape.clone());
        let expr = TensorExpression::new_leaf(tensor, false, None);
        
        // Mix tensor + scalar + tensor
        let scalar = 5.0;
        let data2 = vec![10.0];
        let tensor2 = Tensor::from_data(data2, shape.clone());
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);
        
        let result = (expr + scalar) + expr2;
        
        // Check that result is correctly computed (1+5+10)
        assert_eq!(result.result.data, vec![16.0]);
        assert_eq!(result.result.shape.dimensions, shape);
    }

    #[test]
    #[ignore]
    fn test_tensor_expression_add_broadcasting() {
        let data1 = vec![1.0, 2.0, 3.0];
        let shape1 = vec![3];
        let tensor1 = Tensor::from_data(data1, shape1.clone());
        let expr1 = TensorExpression::new_leaf(tensor1, false, None);

        let tensor2 = Tensor::from_scalar(10.0);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 + expr2;

        // Check that result is correctly computed
        assert_eq!(result.result.data, vec![11.0, 12.0, 13.0]);
        assert_eq!(result.result.shape.dimensions, shape1);
    }

    #[test]
    fn test_tensor_shape_broadcast_same_dimensions() {
        // Test broadcasting with same dimensions
        let shape1 = TensorShape::new(vec![2, 3]);
        let shape2 = TensorShape::new(vec![2, 3]);
        
        let result = TensorShape::broadcast(&shape1, &shape2);
        
        assert_eq!(result.dimensions, vec![2, 3]);
    }

    #[test]
    fn test_tensor_shape_broadcast_scalar() {
        // Test broadcasting a scalar with a tensor
        let shape1 = TensorShape::new(vec![2, 3, 4]);
        let shape2 = TensorShape::new(vec![1]);
        
        let result = TensorShape::broadcast(&shape1, &shape2);
        
        assert_eq!(result.dimensions, vec![2, 3, 4]);
    }

    #[test]
    fn test_tensor_shape_broadcast_different_dimensions() {
        // Test broadcasting tensors with different dimensions
        let shape1 = TensorShape::new(vec![5, 1, 6]);
        let shape2 = TensorShape::new(vec![1, 4, 6]);
        
        let result = TensorShape::broadcast(&shape1, &shape2);
        
        assert_eq!(result.dimensions, vec![5, 4, 6]);
    }

    #[test]
    fn test_tensor_shape_broadcast_prepend_ones() {
        // Test broadcasting with dimensions of different lengths
        let shape1 = TensorShape::new(vec![3]);
        let shape2 = TensorShape::new(vec![2, 1]);
        
        let result = TensorShape::broadcast(&shape1, &shape2);
        
        assert_eq!(result.dimensions, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "Shapes are not compatible for broadcasting")]
    fn test_tensor_shape_broadcast_incompatible() {
        // Test broadcasting incompatible shapes
        let shape1 = TensorShape::new(vec![2, 3]);
        let shape2 = TensorShape::new(vec![2, 4]);
        
        // This should panic with a message about incompatible shapes
        let _result = TensorShape::broadcast(&shape1, &shape2);
    }
}
