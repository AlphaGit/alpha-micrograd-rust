/// Example of a simple neural network layer with a layer of neurons.
///
/// This example demonstrates how to create a simple neural network layer with a layer of neurons.
/// The layer has three inputs and one output. The output is the weighted sum of the inputs.
///
/// Contrast this example with the `neuron` example.
/// 
/// For more information, see [`Layer`].
extern crate alpha_micrograd_rust;

use alpha_micrograd_rust::nn::{Activation, Layer};
use alpha_micrograd_rust::value::Expr;

fn main() {
    let mut target = vec![Expr::new_leaf_with_name(15.0, "t1"), Expr::new_leaf_with_name(85.0, "t2")];
    target[0].is_learnable = false;
    target[1].is_learnable = false;

    let layer = Layer::new(3, 2, Activation::None);
    println!("Initial values: {:}", layer);

    let mut inputs = vec![
        Expr::new_leaf(1.0),
        Expr::new_leaf(2.0),
        Expr::new_leaf(3.0),
    ];

    inputs.iter_mut().for_each(|input| {
        input.is_learnable = false;
    });

    let mut y = layer.forward(inputs);
    let mut y1 = y.remove(0);
    y1.name = Some("y1".to_string());
    let mut y2 = y.remove(0);
    y2.name = Some("y2".to_string());

    let d1 = y1 - target[0].clone();
    let mut sqr1 = Expr::new_leaf(2.0);
    sqr1.is_learnable = false;

    let d2 = y2 - target[1].clone();
    let mut sqr2 = Expr::new_leaf(2.0);
    sqr2.is_learnable = false;

    let mut loss = d1.pow(sqr1) + d2.pow(sqr2);

    let t1 = loss.find("t1").unwrap();
    let t2 = loss.find("t2").unwrap();
    let y1 = loss.find("y1").unwrap();
    let y2 = loss.find("y2").unwrap();

    println!("Initial targets: {:.2}, {:.2}", t1.result, t2.result);
    println!("Predicted: {:.2}, {:.2}", y1.result, y2.result);
    println!("Initial loss: {:.2}", loss.result);

    println!("\nTraining:");
    let learning_rate = 0.004;
    for i in 1..=100 {
        loss.learn(learning_rate);
        loss.recalculate();

        let t1 = loss.find("t1").unwrap();
        let t2 = loss.find("t2").unwrap();
        let y1 = loss.find("y1").unwrap();
        let y2 = loss.find("y2").unwrap();

        println!(
            "Iteration {:3}, loss: {:9.4} / predicted: {:5.2}, {:5.2} (targets: {:5.2}, {:5.2})",
            i, loss.result, y1.result, y2.result, t1.result, t2.result
        );
    }

    println!("Final values: {:}", layer);
}
