use crate::value::Expr;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

struct Neuron {
    w: Vec<Expr>,
    b: Expr,
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
            w: (0..n_inputs).map(|_| between.sample(&mut rng)).map(Expr::new_leaf).collect(),
            b: Expr::new_leaf(between.sample(&mut rng)),
        }
    }

    fn forward(&self, x: Vec<Expr>) -> Expr {
        assert_eq!(
            x.len(),
            self.w.len(),
            "Number of inputs must match number of weights"
        );

        let mut sum = Expr::new_leaf(0.0);

        for (i, x_i) in x.iter().enumerate() {
            sum = sum + (x_i.clone() * self.w[i].clone());
        }

        let activation = sum + self.b.clone();
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
            assert!(n.w[i].result >= -1.0 && n.w[i].result <= 1.0);
        }
    }

    #[test]
    fn can_do_forward_pass_neuron() {
        let n = Neuron::new(3);

        let x = vec![Expr::new_leaf(0.0), Expr::new_leaf(1.0), Expr::new_leaf(2.0)];

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

        let x = vec![Expr::new_leaf(0.0), Expr::new_leaf(1.0), Expr::new_leaf(2.0)];

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

        let x = vec![Expr::new_leaf(0.0), Expr::new_leaf(1.0), Expr::new_leaf(2.0)];

        let y = m.forward(x);

        assert_eq!(y.len(), 1);
    }

    #[test]
    fn can_learn() {
        let mlp = MLP::new(3, vec![2, 2], 1);

        let mut inputs = vec![
            vec![Expr::new_leaf(2.0), Expr::new_leaf(3.0), Expr::new_leaf(-1.0)],
            vec![Expr::new_leaf(3.0), Expr::new_leaf(-1.0), Expr::new_leaf(0.5)],
            vec![Expr::new_leaf(0.5), Expr::new_leaf(1.0), Expr::new_leaf(1.0)],
            vec![Expr::new_leaf(1.0), Expr::new_leaf(1.0), Expr::new_leaf(-1.0)],
        ];

        // make these non-learnable
        inputs.iter_mut().for_each(|instance| 
            instance.iter_mut().for_each(|input| 
                input.set_learnable(false)
            )
        );

        let mut targets = vec![
            Expr::new_leaf(1.0),
            Expr::new_leaf(-1.0),
            Expr::new_leaf(-1.0),
            Expr::new_leaf(1.0),
        ];
        // make these non-learnable
        targets.iter_mut().for_each(|target| target.set_learnable(false));

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
                diff.set_learnable(false);

                let mut squared_exponent = Expr::new_leaf(2.0);
                squared_exponent.set_learnable(false);

                let mut mse = diff.clone().pow(squared_exponent);
                mse.set_learnable(false);

                mse
            })
            .sum::<Expr>();

        loss.grad = 1.0;
        let first_loss = loss.result.clone();
        loss.learn(1e-04);
        loss.recalculate();
        let second_loss = loss.result.clone();

        assert!(second_loss < first_loss, "Loss should decrease after learning ({} >= {})", second_loss, first_loss);
    }
}
