/// A simple example of a multi-layer perceptron (MLP) with 3 input neurons, 2 hidden neurons, and 1 output neuron.
/// The output neuron has no activation function.
/// The loss function is the sum of the squared differences between the predicted and target values.
/// The network is trained to minimize the loss using gradient descent.
/// Contrast this example with the `neuron` example.
/// The network is trained on 3 examples, each with 3 input values and 1 target value.
/// The target values are fixed and not learned.
/// 
/// For more information, see [`MLP`].
extern crate alpha_micrograd_rust;

use alpha_micrograd_rust::nn::{Activation, MLP};
use alpha_micrograd_rust::value::Expr;

fn main() {
    let mut targets = vec![
        Expr::new_leaf_with_name(150.0, "t1"),
        Expr::new_leaf_with_name(250.0, "t2"),
        Expr::new_leaf_with_name(350.0, "t3"),
    ];
    targets.iter_mut().for_each(|target| {
        target.is_learnable = false;
    });

    let mlp = MLP::new(
        3,
        Activation::Tanh,
        vec![2, 2],
        Activation::Tanh,
        1,
        Activation::None,
    );
    println!("Initial values: {:}", mlp);

    let mut inputs = vec![
        vec![
            Expr::new_leaf(1.0),
            Expr::new_leaf(2.0),
            Expr::new_leaf(3.0),
        ],
        vec![
            Expr::new_leaf(4.0),
            Expr::new_leaf(5.0),
            Expr::new_leaf(6.0),
        ],
        vec![
            Expr::new_leaf(7.0),
            Expr::new_leaf(8.0),
            Expr::new_leaf(9.0),
        ],
    ];

    inputs.iter_mut().for_each(|instance| {
        instance.iter_mut().for_each(|value| {
            value.is_learnable = false;
        });
    });

    let predictions = inputs
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
        .collect::<Vec<_>>();

    let differences = predictions
        .iter()
        .zip(targets.iter())
        .map(|(y, t)| y.clone() - t.clone())
        .collect::<Vec<_>>();
    let mut loss = differences
        .iter()
        .map(|d| d.clone() * d.clone())
        .sum::<Expr>();

    let y1 = loss.find("y1").unwrap();
    let y2 = loss.find("y2").unwrap();
    let y3 = loss.find("y3").unwrap();
    println!("Initial loss: {:.2}", loss.result);
    println!(
        "Initial predictions: {:5.2} {:5.2} {:5.2}",
        y1.result, y2.result, y3.result
    );

    println!("\nTraining:");
    let learning_rate = 0.025;
    for i in 1..=100 {
        loss.learn(learning_rate);
        loss.recalculate();

        let t1 = loss.find("t1").unwrap();
        let t2 = loss.find("t2").unwrap();
        let t3 = loss.find("t3").unwrap();

        let y1 = loss.find("y1").unwrap();
        let y2 = loss.find("y2").unwrap();
        let y3 = loss.find("y3").unwrap();

        println!(
            "Iteration {:3}, loss: {:11.4} / predicted: {:5.2}, {:5.2}, {:5.2} (targets: {:5.2}, {:5.2}, {:5.2})",
            i, loss.result, y1.result, y2.result, y3.result, t1.result, t2.result, t3.result
        );
    }

    println!("Final values: {:}", mlp);
}
