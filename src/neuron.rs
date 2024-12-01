use crate::value::{Expr, LeafExpr};
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

struct Neuron {
    w: Vec<f64>,
    b: f64,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct MLP {
    layers: Vec<Layer>,
}

impl Neuron {
    fn new(n_inputs: u32) -> Neuron {
        let between = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = thread_rng();
        Neuron {
            w: (0..n_inputs).map(|_| between.sample(&mut rng)).collect(),
            b: between.sample(&mut rng),
        }
    }

    fn forward(&self, x: Vec<Expr>) -> Expr {
        assert_eq!(
            x.len(),
            self.w.len(),
            "Number of inputs must match number of weights"
        );

        let mut sum = LeafExpr::new(0.0);

        for (i, x_i) in x.iter().enumerate() {
            sum = sum + (x_i.clone() * self.w[i]);
        }

        let activation = sum + self.b;
        activation.tanh()
    }
}

impl Layer {
    fn new(n_inputs: u32, n_outputs: u32) -> Layer {
        Layer {
            neurons: (0..n_outputs).map(|_| Neuron::new(n_inputs)).collect(),
        }
    }

    fn forward(&self, x: Vec<Expr>) -> Vec<Expr> {
        self.neurons.iter().map(|n| n.forward(x.clone())).collect()
    }
}

impl MLP {
    fn new(n_inputs: u32, n_hidden: Vec<u32>, n_outputs: u32) -> MLP {
        let mut layers = Vec::new();
        let mut n_in = n_inputs;
        for n_out in n_hidden {
            layers.push(Layer::new(n_in, n_out));
            n_in = n_out;
        }
        layers.push(Layer::new(n_in, n_outputs));

        MLP { layers }
    }

    fn forward(&self, x: Vec<Expr>) -> Vec<Expr> {
        let mut y = x;
        for layer in &self.layers {
            y = layer.forward(y);
        }
        y.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_instantiate_neuron() {
        let n = Neuron::new(3);

        assert_eq!(n.w.len(), 3);
        for i in 0..3 {
            assert!(n.w[i] >= -1.0 && n.w[i] <= 1.0);
        }
    }

    #[test]
    fn can_do_forward_pass_neuron() {
        let n = Neuron::new(3);

        let x = vec![LeafExpr::new(0.0), LeafExpr::new(1.0), LeafExpr::new(2.0)];

        let _ = n.forward(x);
    }

    #[test]
    fn can_instantiate_layer() {
        let l = Layer::new(3, 2);

        assert_eq!(l.neurons.len(), 2);
        assert_eq!(l.neurons[0].w.len(), 3);
    }

    #[test]
    fn can_do_forward_pass_layer() {
        let l = Layer::new(3, 2);

        let x = vec![LeafExpr::new(0.0), LeafExpr::new(1.0), LeafExpr::new(2.0)];

        let y = l.forward(x);

        assert_eq!(y.len(), 2);
    }

    #[test]
    fn can_instantiate_mlp() {
        let m = MLP::new(3, vec![2, 2], 1);

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
        let m = MLP::new(3, vec![2, 2], 1);

        let x = vec![LeafExpr::new(0.0), LeafExpr::new(1.0), LeafExpr::new(2.0)];

        let y = m.forward(x);

        assert_eq!(y.len(), 1);
    }
}
