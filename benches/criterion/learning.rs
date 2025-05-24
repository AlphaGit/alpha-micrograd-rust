use alpha_micrograd_rust::nn::MLP;
use alpha_micrograd_rust::value::Expr;

use criterion::{BatchSize, Criterion, Throughput};
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

fn get_inputs() -> Vec<Vec<Expr>> {
    let dist = Uniform::new(0.0, 10.0);
    let mut rng = thread_rng();
    let mut random_input = || Expr::new_leaf(dist.sample(&mut rng));

    let mut inputs = vec![
        vec![random_input(), random_input(), random_input()],
        vec![random_input(), random_input(), random_input()],
        vec![random_input(), random_input(), random_input()],
    ];

    inputs.iter_mut().for_each(|instance| {
        instance.iter_mut().for_each(|value| {
            value.is_learnable = false;
        });
    });

    inputs
}

fn get_targets() -> Vec<Expr> {
    let dist = Uniform::new(100.0, 1000.0);
    let mut rng = thread_rng();
    let mut random_target = || Expr::new_leaf(dist.sample(&mut rng));

    let mut targets = vec![random_target(), random_target(), random_target()];

    targets.iter_mut().for_each(|target| {
        target.is_learnable = false;
    });

    targets
}

fn get_mlp() -> MLP {
    MLP::new(
        3,
        alpha_micrograd_rust::nn::Activation::Tanh,
        vec![2, 2],
        alpha_micrograd_rust::nn::Activation::Tanh,
        1,
        alpha_micrograd_rust::nn::Activation::None,
    )
}

fn build_prediction_network(mlp: MLP, inputs: Vec<Vec<Expr>>) -> Vec<Expr> {
    inputs
        // for each example, make a prediction
        .iter()
        .map(|example| mlp.forward(example.clone()))
        // name these predictions y1, y2, y3
        .enumerate()
        .map(|(i, mut y)| {
            // the result is a vector but it's a single value because we specified 1 output neuron
            let mut result = y.remove(0);
            result.name = Some(format!("y{:}", i + 1));
            result
        })
        // collect them into a single vector
        .collect::<Vec<_>>()
}

fn build_loss_function(predictions: Vec<Expr>, targets: Vec<Expr>) -> Expr {
    let differences = predictions
        .iter()
        .zip(targets.iter())
        .map(|(y, t)| y.clone() - t.clone())
        .collect::<Vec<_>>();

    let loss = differences
        .iter()
        .map(|d| d.clone() * d.clone())
        .sum::<Expr>();

    loss
}

pub(crate) fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("learning");
    group.throughput(Throughput::Elements(25 * 25 * 2));
    group.bench_function("learning", |b| {
        b.iter_batched(
            || {
                let inputs = get_inputs();
                let targets = get_targets();
                let mlp = get_mlp();

                let predictions = build_prediction_network(mlp, inputs);
                let loss = build_loss_function(predictions, targets);

                loss
            },
            |mut loss| {
                let learning_rate = 0.01;
                loss.learn(learning_rate);
                loss.recalculate();
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}
