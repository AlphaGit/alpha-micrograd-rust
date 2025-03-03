/// Example of a simple neuron with 3 inputs and no activation function.
///
/// This example demonstrates how to create a simple neuron with 3 inputs and no activation function.
/// The neuron has 3 inputs and 1 output. The output is the weighted sum of the inputs.
///
/// Contrast this example with the `layer` example.
/// 
/// For more information, see [`Neuron`].
extern crate alpha_micrograd_rust;

use alpha_micrograd_rust::nn::{Activation, Neuron};
use alpha_micrograd_rust::value::Expr;

fn main() {
    let mut target = Expr::new_leaf(50.0, "target");
    target.is_learnable = false;

    let neuron = Neuron::new(3, Activation::None);
    println!("Initial values: {:}", neuron);

    let mut inputs = vec![
        Expr::new_leaf(1.0, "x_1"),
        Expr::new_leaf(2.0, "x_2"),
        Expr::new_leaf(3.0, "x_3"),
    ];

    inputs.iter_mut().for_each(|input| {
        input.is_learnable = false;
    });

    let mut y = neuron.forward(inputs);
    y.name = "y".to_string();

    let difference = y - target;
    let mut square_exponent = Expr::new_leaf(2.0, "square_exponent");
    square_exponent.is_learnable = false;

    let mut loss = difference.pow(square_exponent, "loss");

    let target = loss.find("target").unwrap();
    let y = loss.find("y").unwrap();
    println!("Initial target: {:.2}", target.result);
    println!("Predicted: {:.2}", y.result);
    println!("Initial loss: {:.2}", loss.result);

    println!("\nTraining:");
    let learning_rate = 0.01;
    for i in 1..=100 {
        loss.learn(learning_rate);
        loss.recalculate();

        let y = loss.find("y").expect("Node not found");
        let target = loss.find("target").expect("Node not found");

        println!(
            "Iteration {:3}, loss: {:9.4} / predicted: {:.2} (target: {:.2})",
            i, loss.result, y.result, target.result
        );
    }

    println!("Final values: {:}", neuron);
}
