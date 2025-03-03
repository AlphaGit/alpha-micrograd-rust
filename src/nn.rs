//! A simple implementation of simple neural network's elements.
//!
//! This package includes the following elements to construct neural networks:
//! - [`Neuron`]: weight and bias
//! - [`Layer`]: a collection of neurons
//! - [`MLP`]: multilayer-perceptron, a collection of layers and activation function
//! 
//! All of them have a `forward` method to calculate the output of the element.
#![deny(missing_docs)]
use std::fmt::Display;
use crate::value::Expr;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

/// A neuron in a neural network.
/// 
/// A neuron has a collection of weights and a bias. It calculates the weighted sum of the inputs
/// and applies an activation function to the result.
pub struct Neuron {
    w: Vec<Expr>,
    b: Expr,
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
    pub fn new(n_inputs: u32, activation: Activation) -> Neuron {
        let between = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = thread_rng();

        let weights = (1..=n_inputs)
            .map(|_| between.sample(&mut rng))
            .enumerate()
            .map(|(i, n)| Expr::new_leaf(n, &format!("w_{:}", i)))
            .collect();

        Neuron {
            w: weights,
            b: Expr::new_leaf(between.sample(&mut rng), "b"),
            activation,
        }
    }

    /// Calculate the output of the neuron for the given inputs.
    ///
    /// The output of the neuron is the weighted sum of the inputs and the bias.
    /// The activation function is applied to the result.
    pub fn forward(&self, x: Vec<Expr>) -> Expr {
        assert_eq!(
            x.len(),
            self.w.len(),
            "Number of inputs must match number of weights"
        );

        let mut sum = Expr::new_leaf(0.0, "0.0");

        // cloning to avoid consuming these values that we need to keep
        for (i, x_i) in x.iter().enumerate() {
            sum = sum + (x_i.clone() * self.w[i].clone());
        }

        let sum = sum + self.b.clone();
        match self.activation {
            Activation::None => sum,
            Activation::ReLU => sum.relu("activation"),
            Activation::Tanh => sum.tanh("activation"),
        }
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let weights = self.w
            .iter()
            .map(|w| format!("{:.2}", w.result))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Neuron: w: [{:}], b: {:.2}", weights, self.b.result)
    }
}

impl Layer {
    /// Create a new [`Layer`] with `n_inputs` to the neurons and `n_outputs` neurons.
    ///
    /// The layer is a collection of [`neuron`s](Neuron). The number of neurons is `n_outputs`.
    /// Each neuron has `n_inputs` inputs.
    pub fn new(n_inputs: u32, n_outputs: u32, activation: Activation) -> Layer {
        Layer {
            neurons: (0..n_outputs).map(|_| Neuron::new(n_inputs, activation)).collect(),
        }
    }

    /// Calculate the output of the layer for the given inputs.
    ///
    /// The output of the layer is the collection of the outputs of the neurons.
    pub fn forward(&self, x: Vec<Expr>) -> Vec<Expr> {
        self.neurons.iter().map(|n| n.forward(x.clone())).collect()
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let neurons = self.neurons
            .iter()
            .map(|n| format!("{:}", n))
            .collect::<Vec<_>>()
            .join("\n - ");
        write!(f, "Layer:\n - {:}", neurons)
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
        n_inputs: u32, input_activation: Activation,
        n_hidden: Vec<u32>, hidden_activation: Activation,
        n_outputs: u32, output_activation: Activation) -> MLP {

        let mut layers = Vec::new();

        layers.push(Layer::new(n_inputs, n_hidden[0], input_activation));
        for i in 1..n_hidden.len() {
            layers.push(Layer::new(n_hidden[i - 1], n_hidden[i], hidden_activation));
        }
        layers.push(Layer::new(n_hidden[n_hidden.len() - 1], n_outputs, output_activation));

        MLP { layers }
    }

    /// Calculate the output of the MLP for the given inputs.
    /// 
    /// The output of the MLP is the output of the last layer.
    pub fn forward(&self, x: Vec<Expr>) -> Vec<Expr> {
        let mut y = x;
        for layer in &self.layers {
            y = layer.forward(y);
        }
        y
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let layers = self.layers
            .iter()
            .map(|l| format!("{:}", l))
            .collect::<Vec<_>>()
            .join("\n\n");
        write!(f, "MLP:\n{:}", layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_instantiate_neuron() {
        let n = Neuron::new(3, Activation::None);

        assert_eq!(n.w.len(), 3);
        for i in 0..3 {
            assert!(n.w[i].result >= -1.0 && n.w[i].result <= 1.0);
        }
    }

    #[test]
    fn can_do_forward_pass_neuron() {
        let n = Neuron::new(3, Activation::None);

        let x = vec![
            Expr::new_leaf(0.0, "x_1"),
            Expr::new_leaf(1.0, "x_2"), 
            Expr::new_leaf(2.0, "x_3")
        ];

        let _ = n.forward(x);
    }

    #[test]
    fn can_instantiate_layer() {
        let l = Layer::new(3, 2, Activation::None);

        assert_eq!(l.neurons.len(), 2);
        assert_eq!(l.neurons[0].w.len(), 3);
    }

    #[test]
    fn can_do_forward_pass_layer() {
        let l = Layer::new(3, 2, Activation::Tanh);

        let x = vec![
            Expr::new_leaf(0.0, "x_1"),
            Expr::new_leaf(1.0, "x_2"),
            Expr::new_leaf(2.0, "x_3")
        ];

        let y = l.forward(x);

        assert_eq!(y.len(), 2);
    }

    #[test]
    fn can_instantiate_mlp() {
        let m = MLP::new(3, Activation::None,
            vec![2, 2], Activation::Tanh,
            1, Activation::None);

        assert_eq!(m.layers.len(), 3);
        assert_eq!(m.layers[0].neurons.len(), 2); // 2 neurons
        assert_eq!(m.layers[0].neurons[0].w.len(), 3); // 3 inputs (from inputs)

        assert_eq!(m.layers[1].neurons.len(), 2); // 2 neurons
        assert_eq!(m.layers[1].neurons[0].w.len(), 2); // 2 inputs (from neurons)

        assert_eq!(m.layers[2].neurons.len(), 1); // 1 neuron
        assert_eq!(m.layers[2].neurons[0].w.len(), 2); // 2 inputs (from neurons)
    }

    #[test]
    fn can_do_forward_pass_mlp() {
        let m = MLP::new(3, Activation::None,
            vec![2, 2], Activation::Tanh,
            1, Activation::None);

        let x = vec![
            Expr::new_leaf(0.0, "x_1"),
            Expr::new_leaf(1.0, "x_2"),
            Expr::new_leaf(2.0, "x_3")
        ];

        let y = m.forward(x);

        assert_eq!(y.len(), 1);
    }

    #[test]
    fn can_learn() {
        let mlp = MLP::new(3, Activation::None,
            vec![2, 2], Activation::Tanh,
            1, Activation::None);

        let mut inputs = vec![
            vec![Expr::new_leaf(2.0, "x_1,1"), Expr::new_leaf(3.0, "x_1,2"), Expr::new_leaf(-1.0, "x_1,3")],
            vec![Expr::new_leaf(3.0, "x_2,1"), Expr::new_leaf(-1.0, "x_2,2"), Expr::new_leaf(0.5, "x_2,3")],
            vec![Expr::new_leaf(0.5, "x_3,1"), Expr::new_leaf(1.0, "x_3,2"), Expr::new_leaf(1.0, "x_3,3")],
            vec![Expr::new_leaf(1.0, "x_4,1"), Expr::new_leaf(1.0, "x_4,2"), Expr::new_leaf(-1.0, "x_4,3")],
        ];

        // make these non-learnable
        inputs.iter_mut().for_each(|instance| 
            instance.iter_mut().for_each(|input| 
                input.is_learnable = false
            )
        );

        let mut targets = vec![
            Expr::new_leaf(1.0, "y_1"),
            Expr::new_leaf(-1.0, "y_2"),
            Expr::new_leaf(-1.0, "y_3"),
            Expr::new_leaf(1.0, "y_4"),
        ];
        // make these non-learnable
        targets.iter_mut().for_each(|target| target.is_learnable = false);

        let predicted = inputs
            .iter()
            .map(|x| mlp.forward(x.clone()))
            // n_outputs == 1 so we want the only output neuron
            .map(|x| x[0].clone())
            .collect::<Vec<_>>();

        // calculating loss: MSE
        let mut loss = predicted
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let mut diff = p.clone() - t.clone();
                diff.is_learnable = false;

                let mut squared_exponent = Expr::new_leaf(2.0, "2");
                squared_exponent.is_learnable = false;

                let mut mse = diff.clone().pow(squared_exponent, "mse");
                mse.is_learnable = false;

                mse
            })
            .sum::<Expr>();

        let first_loss = loss.result.clone();
        loss.learn(1e-04);
        loss.recalculate();
        let second_loss = loss.result.clone();

        assert!(second_loss < first_loss, "Loss should decrease after learning ({} >= {})", second_loss, first_loss);
    }
}
