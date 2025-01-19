use crate::value::Expr;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

struct Neuron<'a> {
    w: Vec<Expr<'a>>,
    b: Expr<'a>,
}

struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

struct MLP<'a> {
    layers: Vec<Layer<'a>>,
}

impl<'a> Neuron<'a> {
    fn new(n_inputs: u32) -> Neuron<'a> {
        let between = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = thread_rng();
        Neuron {
            w: (0..n_inputs)
                .map(|_| between.sample(&mut rng))
                .map(|v| Expr::from(v))
                .collect(),
            b: Expr::from(between.sample(&mut rng)),
        }
    }

    fn forward(&mut self, x: Vec<Expr<'a>>) -> Expr<'a> {
        assert_eq!(
            x.len(),
            self.w.len(),
            "Number of inputs must match number of weights"
        );

        let sum = x
            .iter()
            .enumerate()
            .map(|(i, x_i)| x_i * &self.w[i])
            .reduce(|a, b| &a + &b)
            .unwrap();

        let activation = &sum + &self.b;
        activation.tanh()
    }
}

impl Layer {
    fn new(n_inputs: u32, n_outputs: u32) -> Layer {
        Layer {
            neurons: (0..n_outputs).map(|_| Neuron::new(n_inputs)).collect(),
        }
    }

    fn forward(&mut self, x: Vec<Expr>) -> Vec<Expr> {
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
        for layer in &self.layers.iter() {
            x = layer.forward(x);
        }
        let mut y = x;
        for layer in &self.layers {
            y = layer.forward(y);
        }
        y
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

    #[test]
    fn can_learn() {
        let mlp = MLP::new(3, vec![2, 2], 1);

        let inputs = vec![
            vec![LeafExpr::new(2.0), LeafExpr::new(3.0), LeafExpr::new(-1.0)],
            vec![LeafExpr::new(3.0), LeafExpr::new(-1.0), LeafExpr::new(0.5)],
            vec![LeafExpr::new(0.5), LeafExpr::new(1.0), LeafExpr::new(1.0)],
            vec![LeafExpr::new(1.0), LeafExpr::new(1.0), LeafExpr::new(-1.0)],
        ];

        let targets = vec![
            LeafExpr::new(1.0),
            LeafExpr::new(-1.0),
            LeafExpr::new(-1.0),
            LeafExpr::new(1.0),
        ];

        let predicted_1 = inputs
            .iter()
            .map(|x| mlp.forward(x.clone()))
            .map(|x| x[0].clone())
            .collect::<Vec<_>>();

        let mut loss_1 = predicted_1
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).pow(2))
            .sum::<Expr>();

        loss_1.reset_grads();
        loss_1.backpropagate();
        loss_1.learn_grads(0.1);

        let predicted_2 = inputs
            .iter()
            .map(|x| mlp.forward(x.clone()))
            .map(|x| x[0].clone())
            .collect::<Vec<_>>();

        let loss_2 = predicted_2
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).pow(2))
            .sum::<Expr>();

        assert!(loss_2.data() < loss_1.data(), "Loss should decrease");
    }
}
