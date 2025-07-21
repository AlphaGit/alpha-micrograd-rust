//! A simple implementation of simple neural network's elements.
//!
//! This package includes the following elements to construct neural networks:
//! - [`Neuron`]: weight and bias
//! - [`Layer`]: a collection of neurons
//! - [`MLP`]: multilayer-perceptron, a collection of layers and activation function
//!
//! All of them have a `forward` method to calculate the output of the element.
#![deny(missing_docs)]
use crate::tensors::{Tensor, TensorExpression};
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use std::fmt::Display;
use std::iter;

/// A layer in a neural network.
///
/// A layer is a collection of [`Neuron`s](Neuron). It calculates the output of each neuron in the layer.
/// The output of the layer is the collection of the outputs of the neurons.
pub struct Layer {
    w: TensorExpression,
    b: TensorExpression,
    activation: Activation,
}

/// A multilayer perceptron.
///
/// A multilayer perceptron is a collection of [`Layer`s](Layer). It calculates the output of each layer
/// and passes it to the next layer.
/// The output of the MLP is the output of the last layer.
pub struct MLP {
    layers: Vec<Layer>,
}

/// Activation functions for neurons.
///
/// The activation function is applied to the weighted sum of the inputs and the bias.
#[derive(Debug, Copy, Clone)]
pub enum Activation {
    /// No activation function.
    None,
    /// Rectified Linear Unit ([`Expr::relu`]).
    ReLU,
    /// Hyperbolic Tangent ([`Expr::tanh`]).
    Tanh,
}

impl Layer {
    /// Create a new [`Layer`] with `n_inputs` to the neurons and `n_outputs` neurons.
    ///
    /// The layer is a collection of [`neuron`s](Neuron). The number of neurons is `n_outputs`.
    /// Each neuron has `n_inputs` inputs.
    pub fn new(n_inputs: u32, n_outputs: u32, activation: Activation) -> Layer {
        let between = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = thread_rng();

        let weights_values = (0..n_inputs * n_outputs)
            .map(|_| between.sample(&mut rng))
            .collect::<Vec<_>>();

        let weights_tensor =
            Tensor::from_data(weights_values, vec![n_inputs as usize, n_outputs as usize]);
        let weights_tensor_expr = TensorExpression::new_leaf(weights_tensor, true, None);

        let bias_values = (0..n_outputs)
            .map(|_| between.sample(&mut rng))
            .collect::<Vec<_>>();
        let bias_tensor = Tensor::from_data(bias_values, vec![n_outputs as usize]);
        let bias_tensor_expr = TensorExpression::new_leaf(bias_tensor, true, None);

        Layer {
            w: weights_tensor_expr,
            b: bias_tensor_expr,
            activation,
        }
    }

    /// Calculate the output of the neuron for the given inputs.
    ///
    /// The output of the neuron is the weighted sum of the inputs and the bias.
    /// The activation function is applied to the result.
    pub fn forward(&self, x: TensorExpression) -> TensorExpression {
        // The sum is calculated as the dot product of the inputs and the weights, plus the bias.
        let multiplication = x.clone() * self.w.clone();
        let sum = self.b.clone() + multiplication;

        match self.activation {
            Activation::None => sum,
            Activation::ReLU => sum.relu(),
            Activation::Tanh => sum.tanh(),
        }
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer:\n - {}", self.w)
            .and_then(|_| write!(f, "\n - {}", self.b))
            .and_then(|_| write!(f, "\n - Activation: {:?}", self.activation))
    }
}

impl MLP {
    /// Create a new multilayer perceptron ([`MLP`]) with the given number of inputs, hidden layers, and outputs.
    ///
    /// The [`MLP`] is a collection of [`Layer`s](Layer). The total number of layers is `n_hidden.len() + 2`.
    /// The first layer has `n_inputs` inputs and `n_hidden[0]` neurons.
    /// The hidden layer `i` has `n_hidden[i]` neurons. There are `n_hidden.len()` hidden layers.
    /// The last layer has `n_hidden[n_hidden.len() - 1]` inputs and `n_outputs` neurons.
    /// The activation functions for the input, hidden, and output layers are `input_activation`, `hidden_activation`, and `output_activation`, respectively.
    pub fn new(
        n_inputs: u32,
        input_activation: Activation,
        n_hidden: Vec<u32>,
        hidden_activation: Activation,
        n_outputs: u32,
        output_activation: Activation,
    ) -> MLP {
        let mut layers = Vec::new();

        let layer_sizes = iter::once(n_inputs)
            .into_iter()
            .chain(n_hidden.into_iter())
            .chain(iter::once(n_outputs))
            .collect::<Vec<_>>();

        for i in 1..layer_sizes.len() {
            let activation = match i {
                0 => input_activation,
                _ if i == layer_sizes.len() - 1 => output_activation,
                _ => hidden_activation,
            };

            layers.push(Layer::new(
                layer_sizes[i - 1],
                layer_sizes[i],
                activation,
            ));
        }

        MLP { layers }
    }

    /// Calculate the output of the MLP for the given inputs.
    ///
    /// The output of the MLP is the output of the last layer.
    pub fn forward(&self, x: TensorExpression) -> TensorExpression {
        let mut y = x;
        for layer in &self.layers {
            y = layer.forward(y);
        }
        y
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let layers = self
            .layers
            .iter()
            .map(|l| format!("{l:}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        write!(f, "MLP:\n{layers:}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_instantiate_layer() {
        let l = Layer::new(3, 2, Activation::None);

        assert_eq!(l.w.parameter_count(false), 3 * 2);
        assert_eq!(l.b.parameter_count(false), 2);
    }

    #[test]
    fn can_do_forward_pass_layer() {
        let l = Layer::new(3, 2, Activation::Tanh);

        let x = TensorExpression::new_leaf(
            Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3]),
            true,
            None,
        );

        let y = l.forward(x);

        assert_eq!(y.result.data.len(), 2);
    }

    #[test]
    fn can_instantiate_mlp() {
        let m = MLP::new(
            3,
            Activation::None,
            vec![2, 2],
            Activation::Tanh,
            1,
            Activation::None,
        );

        assert_eq!(m.layers.len(), 3);
        // 3 inputs (from inputs) x 2 outputs (from neurons)
        assert_eq!(m.layers[0].w.parameter_count(false), 6);
        assert_eq!(m.layers[0].b.parameter_count(false), 2);

        // 2 inputs (from previous layer) x 2 outputs (from neurons)
        assert_eq!(m.layers[1].w.parameter_count(false), 4);
        assert_eq!(m.layers[1].b.parameter_count(false), 2);

        // 2 inputs (from previous layer) x 1 output (from neuron)
        assert_eq!(m.layers[2].w.parameter_count(false), 2);
        assert_eq!(m.layers[2].b.parameter_count(false), 1);
    }

    #[test]
    fn can_do_forward_pass_mlp() {
        let m = MLP::new(
            3,
            Activation::None,
            vec![2, 2],
            Activation::Tanh,
            1,
            Activation::None,
        );

        let x = TensorExpression::new_leaf(
            Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3]),
            true,
            None,
        );

        let y = m.forward(x);

        assert_eq!(y.result.data.len(), 1);
    }

    #[test]
    #[ignore = "learn() is not implemented yet"]
    fn can_learn() {
        let mlp = MLP::new(
            3,
            Activation::None,
            vec![2, 2],
            Activation::Tanh,
            1,
            Activation::None,
        );

        let mut inputs = vec![
            TensorExpression::new_leaf(
                Tensor::from_data(vec![2.0, 3.0, -1.0], vec![3]),
                true,
                None,
            ),
            TensorExpression::new_leaf(
                Tensor::from_data(vec![3.0, -1.0, 0.5], vec![3]),
                true,
                None,
            ),
            TensorExpression::new_leaf(
                Tensor::from_data(vec![0.5, 1.0, 1.0], vec![3]),
                true,
                None,
            ),
            TensorExpression::new_leaf(
                Tensor::from_data(vec![1.0, 1.0, -1.0], vec![3]),
                true,
                None,
            ),
        ];

        // make these non-learnable
        inputs.iter_mut().for_each(|input| {
            input.is_learnable = false
        });

        let mut targets = vec![
            TensorExpression::new_leaf(Tensor::from_scalar(1.0), true, None),
            TensorExpression::new_leaf(Tensor::from_scalar(-1.0), true, None),
            TensorExpression::new_leaf(Tensor::from_scalar(-1.0), true, None),
            TensorExpression::new_leaf(Tensor::from_scalar(1.0), true, None),
        ];
        // make these non-learnable
        targets
            .iter_mut()
            .for_each(|target| target.is_learnable = false);

        let predicted = inputs
            .iter()
            .map(|x| mlp.forward(x.clone()))
            // n_outputs == 1 so we want the only output neuron
            .map(|x| x.clone())
            .collect::<Vec<_>>();

        // calculating loss: MSE
        let loss = predicted
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let mut diff = p.clone() - t.clone();
                diff.is_learnable = false;

                let mut mse = diff.clone().pow(2.0);
                mse.is_learnable = false;

                mse
            })
            .reduce(|acc, x| {
                let mut sum = acc + x;
                sum.is_learnable = false;
                sum
            })
            .expect("No loss calculated");

        let first_loss = loss.result.clone();
        // loss.learn(1e-04);
        // loss.recalculate();
        let second_loss = loss.result.clone();

        let first_loss_sum = first_loss.sum();
        let second_loss_sum = second_loss.sum();
        assert!(
            second_loss_sum < first_loss_sum,
            "Loss should decrease after learning ({} >= {})",
            second_loss_sum,
            first_loss_sum
        );
    }
}
