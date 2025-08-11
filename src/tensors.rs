use std::{fmt::Display, ops::{Add, Mul, Sub}, vec};

use crate::operations::{Operation, OperationType};

/// Represents the shape (dimensions) of a tensor.
#[derive(Debug, Clone)]
pub struct TensorShape {
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

    fn position_iter(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        let mut indices = vec![0; self.dimensions.len()];
        let total_size = self.size();
        let mut count = 0;

        std::iter::from_fn(move || {
            if count >= total_size {
                return None;
            }

            let current_indices = indices.clone();
            count += 1;

            for i in (0..indices.len()).rev() {
                indices[i] += 1;
                if indices[i] < self.dimensions[i] {
                    break;
                }
                indices[i] = 0;
            }

            Some(current_indices)
        })
    }

    fn is_scalar(&self) -> bool {
        self.dimensions.len() == 1 && self.dimensions[0] == 1
    }
}

impl PartialEq for TensorShape {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions
    }
}

/// Represents a multi-dimensional array of f64 values with a specific shape.
#[derive(Clone)]
pub struct Tensor {
    /// The flattened data of the tensor, stored in row-major order.
    pub data: Vec<f64>,
    /// The shape (dimensions) of the tensor.
    pub shape: TensorShape,
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

    /// Creates a tensor from the given data and shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f64 values representing the tensor's elements.
    /// * `shape` - A vector of usize values representing the dimensions of the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in `data` does not match the product of the dimensions in `shape`.
    pub fn from_data(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        assert_eq!(
            data.len(),
            shape.size(),
            "Data size does not match shape size"
        );
        Tensor { data, shape }
    }

    /// Creates a tensor from a single scalar value.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value to be stored in the tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing the scalar value with shape `[1]`.
    pub fn from_scalar(value: f64) -> Self {
        Tensor {
            data: vec![value],
            shape: TensorShape::new(vec![1]),
        }
    }

    fn get(&self, indices: &[usize]) -> f64 {
        assert_eq!(indices.len(), self.shape.dimensions.len());
        let mut flat_index = 0;
        let mut stride = 1;

        for (i, &dim) in self.shape.dimensions.iter().rev().enumerate() {
            let idx = indices[self.shape.dimensions.len() - 1 - i];
            assert!(idx < dim, "Index out of bounds");
            flat_index += idx * stride;
            stride *= dim;
        }

        self.data[flat_index]
    }

    fn get_broadcasted(&self, indices: &[usize], target_shape: &TensorShape) -> f64 {
        assert!(target_shape.dimensions.len() >= self.shape.dimensions.len());
        let mut adjusted_indices = vec![0; self.shape.dimensions.len()];
        let offset = target_shape.dimensions.len() - self.shape.dimensions.len();

        for i in 0..self.shape.dimensions.len() {
            if self.shape.dimensions[i] == 1 {
                adjusted_indices[i] = 0;
            } else {
                adjusted_indices[i] = indices[i + offset];
            }
        }

        self.get(&adjusted_indices)
    }

    /// Returns the sum of all elements in the tensor.
    /// 
    /// # Returns
    /// 
    /// The sum of all elements as a single f64 value.
    pub fn sum(self) -> f64 {
        self.data.iter().sum::<f64>()
    }

    fn tensor_contraction(&self, other: &Self) -> Self {
        // self: A, i, j, ..., k
        // other: B, k, ...l, m
        // result: C, i, j, ..., l, m
        let a_k = self.shape.dimensions.last().unwrap().clone();
        let pos_k = self.shape.dimensions.len() - 1;
        let b_k = other.shape.dimensions.first().unwrap().clone();

        assert!(a_k == b_k || a_k == 1 || b_k == 1,
            "Matrix multiplication requires the inner dimensions to match or be 1, got {:?} (from {:?}) and {:?} (from {:?})",
            a_k, self.shape, b_k, other.shape);
        let max_k = a_k.max(b_k);

        let final_dimensions = self.shape.dimensions[..pos_k]
            .iter()
            .chain(other.shape.dimensions.iter().skip(1))
            .cloned()
            .collect::<Vec<usize>>();
        let final_shape = TensorShape::new(final_dimensions);

        let result_data = final_shape
            .position_iter()
            .map(|pos| {
                let mut sum = 0.0;
                let mut a_indexing = pos.iter().take(pos_k).cloned().collect::<Vec<usize>>();
                a_indexing.push(0); // Placeholder for the k dimension

                let mut b_indexing = pos.iter().skip(pos_k).cloned().collect::<Vec<usize>>();
                b_indexing.insert(0, 0); // Placeholder for the k dimension

                for k in 0..max_k {
                    a_indexing[pos_k] = k.min(a_k - 1); // k or 0 if a_k is 1
                    let a = self.get(&a_indexing);

                    b_indexing[0] = k.min(b_k - 1); // k or 0 if b_k is 1
                    let b = other.get(&b_indexing);
                    sum += a * b;
                }
                sum
            })
            .collect::<Vec<f64>>();

        Tensor::from_data(result_data, final_shape.dimensions)
    }

    fn tanh(&self) -> Self {
        let result_data = self.data.iter().map(|&x| x.tanh()).collect::<Vec<f64>>();
        Tensor::from_data(result_data, self.shape.dimensions.clone())
    }

    fn relu(&self) -> Self {
        let result_data = self.data.iter().map(|&x| x.max(0.0)).collect::<Vec<f64>>();
        Tensor::from_data(result_data, self.shape.dimensions.clone())
    }

    fn exp(&self) -> Self {
        let result_data = self.data.iter().map(|&x| x.exp()).collect::<Vec<f64>>();
        Tensor::from_data(result_data, self.shape.dimensions.clone())
    }

    fn powf(&self, exponent: f64) -> Self {
        let result_data = self.data.iter().map(|&x| x.powf(exponent)).collect::<Vec<f64>>();
        Tensor::from_data(result_data, self.shape.dimensions.clone())
    }

    fn element_wise_operation(&self, other: &Self, operation: Operation) -> Self {
        let combined_shape = TensorShape::broadcast(&self.shape, &other.shape);
        let result_data = combined_shape
            .position_iter()
            .map(|pos| {
                let value1 = self.get_broadcasted(&pos, &combined_shape);
                let value2 = other.get_broadcasted(&pos, &combined_shape);
                match operation {
                    Operation::Add => value1 + value2,
                    Operation::Sub => value1 - value2,
                    Operation::Mul => value1 * value2,
                    Operation::Div => value1 / value2,
                    _ => panic!("Unsupported operation for element-wise operation"),
                }
            })
            .collect::<Vec<f64>>();

        Tensor::from_data(result_data, combined_shape.dimensions)
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        self.element_wise_operation(other, Operation::Add)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        self.element_wise_operation(other, Operation::Sub)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        if self.shape.is_scalar() || other.shape.is_scalar() {
            return self.element_wise_operation(other, Operation::Mul);
        }

        if self.shape.dimensions.len() > 1 || 
           other.shape.dimensions.len() > 1 {
            return self.tensor_contraction(other);
        }

        self.element_wise_operation(other, Operation::Mul)
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Self::Output {
        let self_tensor = Tensor::from_scalar(self);
        &self_tensor * tensor
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Self::Output {
        let scalar_tensor = Tensor::from_scalar(scalar);
        self * &scalar_tensor
    }
}

#[derive(Clone)]
/// Represents a node in a tensor computation graph, which can be a leaf tensor or an operation applied to one or more tensors.
pub struct TensorExpression {
    operand1: Option<Box<TensorExpression>>,
    operand2: Option<Box<TensorExpression>>,
    operation: Operation,
    /// The result tensor of this expression node.
    pub result: Tensor,
    /// Indicates whether this tensor is a learnable parameter.
    pub is_learnable: bool,
    grad: Option<Tensor>,
    name: Option<String>,
}

impl TensorExpression {
    /// Creates a new leaf node in the tensor expression tree.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor value for this leaf node.
    /// * `is_learnable` - Whether this tensor is a learnable parameter.
    /// * `name` - An optional name for this tensor expression.
    pub fn new_leaf(tensor: Tensor, is_learnable: bool, name: Option<String>) -> Self {
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

    /// Applies the hyperbolic tangent (tanh) activation function element-wise to the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor expression where each element is replaced by its tanh value.
    pub fn tanh(self) -> Self {
        let result_tensor = self.result.tanh();
        TensorExpression::new_unary(self, Operation::Tanh, result_tensor)
    }

    /// Applies the ReLU (Rectified Linear Unit) activation function element-wise to the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor expression where each element is replaced by the maximum of 0.0 and the original value.
    pub fn relu(self) -> Self {
        let result_tensor = self.result.relu();
        TensorExpression::new_unary(self, Operation::ReLU, result_tensor)
    }

    fn exp(self) -> Self {
        let result_tensor = self.result.exp(); 
        TensorExpression::new_unary(self, Operation::Exp, result_tensor)
    }

    /// Raises each element of the tensor to the specified power.
    /// 
    /// # Arguments
    /// 
    /// * `exponent` - The power to raise each element to.
    /// 
    /// # Returns
    /// 
    /// A new tensor expression where each element is raised to the given power.
    pub fn pow(self, exponent: f64) -> Self {
        let result_tensor = self.result.powf(exponent);
        let exponent_tensor = Tensor::from_scalar(exponent);
        let exponent_tensor = TensorExpression::new_leaf(exponent_tensor, false, None);

        TensorExpression::new_binary(self, exponent_tensor, Operation::Pow, result_tensor)
    }

    /// Counts the number of parameters in the tensor expression tree.
    /// 
    /// # Arguments
    /// 
    /// * `learnable_only` - If true, only counts learnable parameters.
    /// 
    /// # Returns
    /// 
    /// The total number of parameters in this expression and its operands.
    pub fn parameter_count(&self, learnable_only: bool) -> usize {
        let mut count = 0;
        if self.is_learnable {
            count += self.result.data.len();
        }
        if let Some(operand1) = &self.operand1 {
            count += operand1.parameter_count(learnable_only);
        }
        if let Some(operand2) = &self.operand2 {
            count += operand2.parameter_count(learnable_only);
        }
        count
    }
}

impl Add for TensorExpression {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let result_tensor = &self.result + &other.result;
        TensorExpression::new_binary(self, other, Operation::Add, result_tensor)
    }
}

impl Mul for TensorExpression {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let result_tensor = &self.result * &other.result;
        TensorExpression::new_binary(self, other, Operation::Mul, result_tensor)
    }
}

impl Add<f64> for TensorExpression {
    type Output = Self;

    fn add(self, scalar: f64) -> Self::Output {
        let scalar_tensor = Tensor::from_scalar(scalar);
        let scalar_expression = TensorExpression::new_leaf(scalar_tensor, false, None);

        self + scalar_expression
    }
}

impl Mul<f64> for TensorExpression {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        let scalar_tensor = Tensor::from_scalar(scalar);
        let scalar_expression = TensorExpression::new_leaf(scalar_tensor, false, None);

        self * scalar_expression
    }
}

impl Sub for TensorExpression {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let result_tensor = &self.result - &other.result;
        TensorExpression::new_binary(self, other, Operation::Sub, result_tensor)
    }
}

impl Sub<f64> for TensorExpression {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self::Output {
        let scalar_tensor = Tensor::from_scalar(scalar);
        let scalar_expression = TensorExpression::new_leaf(scalar_tensor, false, None);

        self - scalar_expression
    }
}

impl Display for TensorExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}", name)
        } else {
            let op1_str = self.operand1.as_ref().map_or("None".to_string(), |op| op.to_string());
            let op2_str = self.operand2.as_ref().map_or("None".to_string(), |op| op.to_string());
            write!(f, "TensorExpression: {} {:?} {}", 
                   op1_str,
                   self.operation,
                   op2_str)
        }
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
    #[should_panic(expected = "Shapes are not compatible for broadcasting: [2, 2] and [3, 1]")]
    fn test_tensor_expression_add_shape_mismatch() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0];

        let tensor1 = Tensor::from_data(data1, vec![2, 2]);
        let tensor2 = Tensor::from_data(data2, vec![3, 1]);

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let _result = expr1 + expr2;
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

    #[test]
    fn test_tensor_get_1d() {
        // Test getting values from a 1D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];
        let tensor = Tensor::from_data(data, shape);
        
        assert_eq!(tensor.get(&[0]), 1.0);
        assert_eq!(tensor.get(&[2]), 3.0);
        assert_eq!(tensor.get(&[4]), 5.0);
    }

    #[test]
    fn test_tensor_get_2d() {
        // Test getting values from a 2D tensor
        // Create a 2x3 matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::from_data(data, shape);
        
        assert_eq!(tensor.get(&[0, 0]), 1.0);
        assert_eq!(tensor.get(&[0, 2]), 3.0);
        assert_eq!(tensor.get(&[1, 0]), 4.0);
        assert_eq!(tensor.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_tensor_get_3d() {
        // Test getting values from a 3D tensor
        // Create a 2x2x2 tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 2, 2];
        let tensor = Tensor::from_data(data, shape);
        
        assert_eq!(tensor.get(&[0, 0, 0]), 1.0);
        assert_eq!(tensor.get(&[0, 0, 1]), 2.0);
        assert_eq!(tensor.get(&[0, 1, 0]), 3.0);
        assert_eq!(tensor.get(&[0, 1, 1]), 4.0);
        assert_eq!(tensor.get(&[1, 0, 0]), 5.0);
        assert_eq!(tensor.get(&[1, 1, 1]), 8.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_tensor_get_out_of_bounds() {
        // Test that accessing an out-of-bounds index panics
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        let tensor = Tensor::from_data(data, shape);
        
        // This should panic
        tensor.get(&[3]);
    }

    #[test]
    #[should_panic]
    fn test_tensor_get_wrong_dimensions() {
        // Test that providing the wrong number of indices panics
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::from_data(data, shape);
        
        // This should panic - trying to access a 2D tensor with only one index
        tensor.get(&[1]);
    }

    #[test]
    fn test_tensor_get_scalar() {
        // Test getting the value from a scalar tensor
        let tensor = Tensor::from_scalar(42.0);
        
        assert_eq!(tensor.get(&[0]), 42.0);
    }

    #[test]
    fn test_get_broadcasted_scalar_to_2d() {
        // Broadcasting a scalar to a 2D tensor
        let tensor = Tensor::from_scalar(7.0);
        let target_shape = TensorShape::new(vec![3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(tensor.get_broadcasted(&[i, j], &target_shape), 7.0);
            }
        }
    }

    #[test]
    fn test_get_broadcasted_1d_to_2d() {
        // Broadcasting a 1D tensor to a 2D tensor
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let target_shape = TensorShape::new(vec![2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.get_broadcasted(&[i, j], &target_shape), tensor.get(&[j]));
            }
        }
    }

    #[test]
    fn test_get_broadcasted_2d_singleton_to_2d() {
        // Broadcasting a 2D tensor with a singleton dimension to a larger 2D tensor
        let tensor = Tensor::from_data(vec![5.0, 6.0], vec![1, 2]);
        let target_shape = TensorShape::new(vec![4, 2]);
        for i in 0..4 {
            for j in 0..2 {
                assert_eq!(tensor.get_broadcasted(&[i, j], &target_shape), tensor.get(&[0, j]));
            }
        }
    }

    #[test]
    fn test_get_broadcasted_1d_to_3d() {
        // Broadcasting a 1D tensor to a 3D tensor
        let tensor = Tensor::from_data(vec![10.0, 20.0], vec![2]);
        let target_shape = TensorShape::new(vec![3, 2, 2]);
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_eq!(tensor.get_broadcasted(&[i, j, k], &target_shape), tensor.get(&[k]));
                }
            }
        }
    }

    #[test]
    fn test_get_broadcasted_2_1_2_to_2_4_2() {
        // Original shape: (2, 1, 2)
        // Broadcasted shape: (2, 4, 2)
        // Data layout: [[[a, b]], [[c, d]]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 1, 2]);
        let target_shape = TensorShape::new(vec![2, 4, 2]);
        // For each i in 0..2, j in 0..4, k in 0..2, the value should be tensor.get(&[i, 0, k])
        for i in 0..2 {
            for j in 0..4 {
                for k in 0..2 {
                    let expected = tensor.get(&[i, 0, k]);
                    assert_eq!(tensor.get_broadcasted(&[i, j, k], &target_shape), expected, "Mismatch at [{}, {}, {}]", i, j, k);
                }
            }
        }
    }

    #[test]
    fn test_position_iter_1d() {
        let shape = TensorShape::new(vec![3]);
        let positions: Vec<_> = shape.position_iter().collect();
        assert_eq!(positions, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn test_position_iter_2d() {
        let shape = TensorShape::new(vec![2, 2]);
        let positions: Vec<_> = shape.position_iter().collect();
        assert_eq!(positions, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    }

    #[test]
    fn test_position_iter_3d() {
        let shape = TensorShape::new(vec![2, 1, 2]);
        let positions: Vec<_> = shape.position_iter().collect();
        assert_eq!(positions, vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![1, 0, 0],
            vec![1, 0, 1],
        ]);
    }

    #[test]
    fn test_position_iter_empty() {
        let shape = TensorShape::new(vec![]);
        let positions: Vec<_> = shape.position_iter().collect();
        assert_eq!(positions, vec![vec![]]);
    }

    #[test]
    fn test_tensor_expression_mul_2d() {
        // 2x2 * 2x2
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 2];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        // Manual matrix multiplication result
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result.result.data, vec![19.0, 22.0, 43.0, 50.0]);
        assert_eq!(result.result.shape.dimensions, shape);
    }

    #[test]
    fn test_tensor_expression_mul_2d_1d() {
        // dimensionality: (2, 2) * (2) => (2)
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0];
        let shape1 = vec![2, 2];
        let shape2 = vec![2];

        let tensor1 = Tensor::from_data(data1, shape1.clone());
        let tensor2 = Tensor::from_data(data2, shape2.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        // Manual result: for each i in 0..2, j in 0..2: result[i] = tensor1[i,j] * tensor2[j]
        // [1*5+2*6, 3*5+4*6] = [17.0, 39.0]
        assert_eq!(result.result.data, vec![17.0, 39.0]);
        assert_eq!(result.result.shape.dimensions, vec![2]);
    }

    #[test]
    fn test_tensor_expression_mul_element_wise() {
        // [1,2,3] * [4,5,6] = [1*4, 2*5, 3*6] = [4, 10, 18]
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0];
        let shape = vec![3];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        // Should be a scalar
        assert_eq!(result.result.data, vec![4.0, 10.0, 18.0]);
        assert_eq!(result.result.shape.dimensions, shape);
    }

    #[test]
    fn test_tensor_expression_mul_scalar() {
        // 2x2 * scalar
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let scalar = 10.0;

        let tensor = Tensor::from_data(data.clone(), shape.clone());
        let expr = TensorExpression::new_leaf(tensor, false, None);

        let result = expr * scalar;

        // Each element multiplied by scalar
        assert_eq!(result.result.data, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(result.result.shape.dimensions, shape);
    }

    #[test]
    fn test_tensor_expression_mul_2x2x2_times_2x2x2() {
        // 2x2x2 * 2x2x2
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let shape = vec![2, 2, 2];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        assert_eq!(result.result.data, vec![35.0, 38.0, 41.0, 44.0, 79.0, 86.0, 93.0, 100.0, 123.0, 134.0, 145.0, 156.0, 167.0, 182.0, 197.0, 212.0]);
        assert_eq!(result.result.shape.dimensions, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_tensor_expression_mul_3x3_times_3x3() {
        // 3x3 * 3x3
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let data2 = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let shape = vec![3, 3];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        assert_eq!(result.result.data, vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0]);
        assert_eq!(result.result.shape.dimensions, shape);
    }

    #[test]
    fn test_numpy_matmul_example_2x2_times_2x2() {
        // https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        // Numpy example: 2x2 * 2x2
        let data1 = vec![1.0, 0.0, 0.0, 1.0];
        let data2 = vec![4.0, 1.0, 2.0, 2.0];
        let shape = vec![2, 2];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        assert_eq!(result.result.data, vec![4.0, 1.0, 2.0, 2.0]);
        assert_eq!(result.result.shape.dimensions, shape);
    }

    #[test]
    fn test_numpy_matmul_example_2x2_times_2() {
        // Numpy example: 2x2 * 2
        let data1 = vec![1.0, 0.0, 0.0, 1.0];
        let data2 = vec![1.0, 2.0];
        let shape1 = vec![2, 2];
        let shape2 = vec![2];

        let tensor1 = Tensor::from_data(data1.clone(), shape1.clone());
        let tensor2 = Tensor::from_data(data2.clone(), shape2.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 * expr2;

        // Should be a vector of size 2
        assert_eq!(result.result.data, vec![1.0, 2.0]);
        assert_eq!(result.result.shape.dimensions, vec![2]);

        // The other way around: 2 * 2x2
        let tensor1 = Tensor::from_data(data1, shape1.clone());
        let expr1 = TensorExpression::new_leaf(tensor1, false, None);

        let tensor2 = Tensor::from_data(data2, shape2.clone());
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);
        let result = expr2 * expr1;

        // Should be a vector of size 2
        assert_eq!(result.result.data, vec![1.0, 2.0]);
        assert_eq!(result.result.shape.dimensions, vec![2]);
    }

    #[test]
    fn test_tensor_expression_sub() {
        let data1 = vec![5.0, 7.0, 9.0, 11.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let tensor1 = Tensor::from_data(data1.clone(), shape.clone());
        let tensor2 = Tensor::from_data(data2.clone(), shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);

        let result = expr1 - expr2;
        assert_eq!(result.result.data, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(result.result.shape.dimensions, shape);
        assert_eq!(result.operation, Operation::Sub);
        assert!(result.operand1.is_some());
        assert!(result.operand2.is_some());
    }

    #[test]
    fn test_tensor_expression_sub_scalar() {
        let data = vec![10.0, 20.0];
        let shape = vec![2];
        let scalar = 3.0;

        let tensor = Tensor::from_data(data.clone(), shape.clone());
        let expr = TensorExpression::new_leaf(tensor, false, None);

        let result = expr - scalar;
        assert_eq!(result.result.data, vec![7.0, 17.0]);
        assert_eq!(result.result.shape.dimensions, shape);
        assert_eq!(result.operation, Operation::Sub);
        assert!(result.operand1.is_some());
        assert!(result.operand2.is_some());
        if let Some(op2) = &result.operand2 {
            assert_eq!(op2.result.data, vec![scalar]);
            assert_eq!(op2.result.shape.dimensions, vec![1]);
        }
    }

    #[test]
    fn test_tensor_expression_chained_sub() {
        let data1 = vec![10.0, 20.0];
        let data2 = vec![1.0, 2.0];
        let data3 = vec![3.0, 4.0];
        let shape = vec![2];

        let tensor1 = Tensor::from_data(data1, shape.clone());
        let tensor2 = Tensor::from_data(data2, shape.clone());
        let tensor3 = Tensor::from_data(data3, shape.clone());

        let expr1 = TensorExpression::new_leaf(tensor1, false, None);
        let expr2 = TensorExpression::new_leaf(tensor2, false, None);
        let expr3 = TensorExpression::new_leaf(tensor3, false, None);

        let result = expr1 - expr2 - expr3;
        assert_eq!(result.result.data, vec![6.0, 14.0]); // (10-1)-3, (20-2)-4
        assert_eq!(result.result.shape.dimensions, shape);
        assert_eq!(result.operation, Operation::Sub);
    }
}
