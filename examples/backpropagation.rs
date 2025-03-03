/// This example shows how to implement a simple adaptive linear neuron using the
/// backpropagation algorithm.
///
/// We will compute the expression `x1*w1 + x2*w2 + b` and apply a non-linear activation
/// function to it: `tanh(x1*w1 + x2*w2 + b)`.
///
/// We will only allow the values of `w1`, `w2` and `b` to be learned to reach a fixed
/// target value.
/// 
/// For more information, see [`Expr`].
extern crate alpha_micrograd_rust;

use alpha_micrograd_rust::value::Expr;

fn main() {
    // these are the initial values for the nodes of the graph
    let mut x1 = Expr::new_leaf(2.0, "x1");
    x1.is_learnable = false;

    let mut x2 = Expr::new_leaf(1.0, "x2");
    x2.is_learnable = false;

    let w1 = Expr::new_leaf(-3.0, "w1");
    let w2 = Expr::new_leaf(1.0, "w2");
    let b = Expr::new_leaf(6.5, "b");

    // here we compute the expression x1*w1 + x2*w2 + b
    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1_x2w2 = x1w1 + x2w2;
    let n = x1w1_x2w2 + b;

    // we add a non-linear activation function: tanh(x1*w1 + x2*w2 + b)
    let o = n.tanh("o");

    println!("Initial output: {:.2}", o.result);

    // we set the target value
    let target_value = 0.2;
    let mut target = Expr::new_leaf(target_value, "target");
    target.is_learnable = false;

    // we compute the loss function
    let mut squared_exponent = Expr::new_leaf(2.0, "squared_exponent");
    squared_exponent.is_learnable = false;

    let mut loss = (o - target).pow(squared_exponent, "loss");
    loss.is_learnable = false;

    // we print the initial loss
    println!("Initial loss: {:.4}", loss.result);

    println!("\nTraining:");
    let learning_rate = 0.01;
    for i in 1..=50 {
        loss.learn(learning_rate);
        loss.recalculate();

        let target = loss.find("o").expect("Node not found");

        println!(
            "Iteration {:2}, loss: {:.4} / result: {:.2}",
            i, loss.result, target.result
        );
    }

    let w1 = loss.find("w1").expect("Node not found");
    let w2 = loss.find("w2").expect("Node not found");
    let b = loss.find("b").expect("Node not found");

    println!(
        "\nFinal values: w1: {:.2}, w2: {:.2}, b: {:.2}",
        w1.result, w2.result, b.result
    );

    let x1 = loss.find("x1").expect("Node not found");
    let x2 = loss.find("x2").expect("Node not found");

    let n = loss
        .find("(((x1 * w1) + (x2 * w2)) + b)") // auto-generated node name
        .expect("Node not found");
    let o = loss.find("o").expect("Node not found");

    println!(
        "Final formula: tanh({:.2}*{:.2} + {:.2}*{:.2} + {:.2}) = tanh({:.2}) = {:.2}",
        x1.result, w1.result, x2.result, w2.result, b.result, n.result, o.result
    )
}
