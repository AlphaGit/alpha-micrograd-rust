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

/// A neuron in a neural network.
///
/// A neuron has a collection of weights and a bias. It calculates the weighted sum of the inputs
/// and applies an activation function to the result.
pub struct Neuron {
    w: TensorExpression,
    b: TensorExpression,
    activation: Activation,
}

/// A layer in a neural network.
///
/// A layer is a collection of [`Neuron`s](Neuron). It calculates the output of each neuron in the layer.
/// The output of the layer is the collection of the outputs of the neurons.
pub struct Layer {
    neurons: Vec<Neuron>,
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

impl Neuron {
    /// Create a new [`Neuron`] with `n_inputs` inputs.
    ///
    /// The weights and bias are initialized randomly from a uniform distribution between -1 and 1.
    ///
    /// The weights are named `w_i` where `i` is the index of the weight (starting from 1).
    /// The bias is named `b`.
    pub fn new(n_inputs: u32, n_outputs: u32, activation: Activation) -> Neuron {
        let between = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = thread_rng();

        let weights_values = (0..n_inputs * n_outputs)
            .map(|_| between.sample(&mut rng))
            .collect::<Vec<_>>();

        let weights_tensor =
            Tensor::from_data(weights_values, vec![n_inputs as usize, n_outputs as usize]);
        let weights_tensor_expr = TensorExpression::new_leaf(weights_tensor, true, None);

        let bias_value = between.sample(&mut rng);
        let bias_tensor = Tensor::from_scalar(bias_value);
        let bias_tensor_expr = TensorExpression::new_leaf(bias_tensor, true, None);

        Neuron {
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

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neuron: w: [{:}], b: {:.2}", self.w, self.b)
    }
}

impl Layer {
    /// Create a new [`Layer`] with `n_inputs` to the neurons and `n_outputs` neurons.
    ///
    /// The layer is a collection of [`neuron`s](Neuron). The number of neurons is `n_outputs`.
    /// Each neuron has `n_inputs` inputs.
    pub fn new(n_inputs: u32, n_outputs: u32, neuron_count: u32, activation: Activation) -> Layer {
        Layer {
            neurons: (0..neuron_count)
                .map(|_| Neuron::new(n_inputs, n_outputs, activation))
                .collect(),
        }
    }

    /// Calculate the output of the layer for the given inputs.
    ///
    /// The output of the layer is the collection of the outputs of the neurons.
    pub fn forward(&self, x: Vec<TensorExpression>) -> Vec<TensorExpression> {
        self.neurons
            .iter()
            .zip(x)
            .map(|(n_i, x_i)| n_i.forward(x_i.clone()))
            .collect()
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let neurons = self
            .neurons
            .iter()
            .map(|n| format!("{n:}"))
            .collect::<Vec<_>>()
            .join("\n - ");
        write!(f, "Layer:\n - {neurons:}")
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

        layers.push(Layer::new(n_inputs, n_hidden[0], 1, input_activation));
        for i in 1..n_hidden.len() {
            layers.push(Layer::new(
                n_hidden[i - 1],
                n_hidden[i],
                1,
                hidden_activation,
            ));
        }
        layers.push(Layer::new(
            n_hidden[n_hidden.len() - 1],
            n_outputs,
            1,
            output_activation,
        ));

        MLP { layers }
    }

    /// Calculate the output of the MLP for the given inputs.
    ///
    /// The output of the MLP is the output of the last layer.
    pub fn forward(&self, x: Vec<TensorExpression>) -> Vec<TensorExpression> {
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
    fn can_instantiate_neuron() {
        let n = Neuron::new(3, 1, Activation::None);

        assert_eq!(n.w.parameter_count(false), 3);
        for i in 0..3 {
            assert!(n.w.result.data[i] >= -1.0 && n.w.result.data[i] <= 1.0);
        }
    }

    #[test]
    fn can_do_forward_pass_neuron() {
        let n = Neuron::new(3, 1, Activation::None);

        let x =
            TensorExpression::new_leaf(Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3]), true, None);

        let _ = n.forward(x);
    }

    #[test]
    fn can_instantiate_layer() {
        let l = Layer::new(3, 2, 1, Activation::None);

        assert_eq!(l.neurons.len(), 1);
        assert_eq!(l.neurons[0].w.parameter_count(false), 6);
    }

    #[test]
    fn can_do_forward_pass_layer() {
        let l = Layer::new(3, 2, 1, Activation::Tanh);

        let x = vec![TensorExpression::new_leaf(
            Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3]),
            true,
            None,
        )];

        let y = l.forward(x);

        assert_eq!(y[0].result.data.len(), 2);
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
        assert_eq!(m.layers[0].neurons.len(), 1);
        // 3 inputs (from inputs) x 2 outputs (from neurons)
        assert_eq!(m.layers[0].neurons[0].w.parameter_count(false), 6);

        assert_eq!(m.layers[1].neurons.len(), 1);
        // 2 inputs (from previous layer) x 2 outputs (from neurons)
        assert_eq!(m.layers[1].neurons[0].w.parameter_count(false), 4);

        assert_eq!(m.layers[2].neurons.len(), 1);
        // 2 inputs (from previous layer) x 1 output (from neuron)
        assert_eq!(m.layers[2].neurons[0].w.parameter_count(false), 2);
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

        let x = vec![TensorExpression::new_leaf(
            Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3]),
            true,
            None,
        )];

        let y = m.forward(x);

        assert_eq!(y.len(), 1);
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
            vec![TensorExpression::new_leaf(
                Tensor::from_data(vec![2.0, 3.0, -1.0], vec![3]),
                true,
                None,
            )],
            vec![TensorExpression::new_leaf(
                Tensor::from_data(vec![3.0, -1.0, 0.5], vec![3]),
                true,
                None,
            )],
            vec![TensorExpression::new_leaf(
                Tensor::from_data(vec![0.5, 1.0, 1.0], vec![3]),
                true,
                None,
            )],
            vec![TensorExpression::new_leaf(
                Tensor::from_data(vec![1.0, 1.0, -1.0], vec![3]),
                true,
                None,
            )],
        ];

        // make these non-learnable
        inputs.iter_mut().for_each(|instance| {
            instance
                .iter_mut()
                .for_each(|input| input.is_learnable = false)
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
            .map(|x| x[0].clone())
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
