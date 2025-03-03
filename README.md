# Alpha's Micrograd in Rust

This repository holds yet another version of [Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd).

This version was created mostly with learning purposes, but I also took the chance for some liberties which, hopefully, make this library more Rust idiomatic.

## What is Micrograd?

From Karpathy's README:

> A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. 

## Values

You can construct values that get added into a tree structure automatically.

In this example we model the function `tanh(x1 * w1 + x2 * w2 + b)`, which might as well be a simple neuron.

```rust
use alpha_micrograd_rust::value::Expr;

let x1 = Expr::new_leaf(2.0, "x1");
let x2 = Expr::new_leaf(0.0, "x2");
let w1 = Expr::new_leaf(-3.0, "w1");
let w2 = Expr::new_leaf(1.0, "w2");
let b = Expr::new_leaf(6.8813735870195432, "b");
```

As you can see, each value is represented by an `Expr` struct, which holds the value and the internal gradient required for backpropagation (to be seen later on).

Values can be combined in an ergonomic approach:

```rust
use alpha_micrograd_rust::value::Expr;

let x1 = Expr::new_leaf(2.0, "x1");
let x2 = Expr::new_leaf(0.0, "x2");
let w1 = Expr::new_leaf(-3.0, "w1");
let w2 = Expr::new_leaf(1.0, "w2");
let b = Expr::new_leaf(6.8813735870195432, "b");

let x1w1 = x1 * w1;
let x2w2 = x2 * w2;
let x1w1_x2w2 = x1w1 + x2w2;
let n = x1w1_x2w2 + b;
```

Or you can use helper functions already embedded in the `Expr` struct:

```rust
use alpha_micrograd_rust::value::Expr;

let x1 = Expr::new_leaf(2.0, "x1");
let mut o = x1.tanh("output");
```

For simplicity of the construction of the in-memory tree, the `Expr` values are owned by the new `Expr` values created by the operations. This means that you can't use the same `Expr` value in different branches of the tree as of right now.

You can also evaluate the value of the expression:

```rust
use alpha_micrograd_rust::value::Expr;

let o = Expr::new_leaf(2.0, "x1") + Expr::new_leaf(3.0, "x2");
println!("{}", o.result)
```

But more interestingly, you can backpropagate the gradients and have the learnable parameters be adjusted accordingly:

```rust
use alpha_micrograd_rust::value::Expr;

let x1 = Expr::new_leaf(2.0, "x1");
let mut o = x1.tanh("output");

let lr = 0.001; // learning rate
o.learn(lr);
```

Maybe some parameters (like inputs, loss functions) should not be learnable, so you can set the `is_learnable` flag to `false`:

```rust
use alpha_micrograd_rust::value::Expr;

let mut x1 = Expr::new_leaf(2.0, "x1");
x1.is_learnable = false;
```

### Available operations

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)
- Hyperbolic tangent (`tanh`)
- Exponential (`exp`)
- Power (`pow`)
- ReLU (`relu`)

## Neurons, Layers and Networks

For your convience, you can create a `Neuron` struct that holds the weights and bias of a neuron, and a `Layer` struct that holds a vector of neurons.

```rust
use alpha_micrograd_rust::nn::{Neuron, Activation};
use alpha_micrograd_rust::value::Expr;

let n = Neuron::new(2, Activation::None); // 2 inputs
let x = vec![Expr::new_leaf(1.0, "x1"), Expr::new_leaf(2.0, "x2")];
let y = n.forward(x);
```

Similarly, you can declare a single Layer of neurons:

```rust
use alpha_micrograd_rust::nn::{Layer, Activation};
use alpha_micrograd_rust::value::Expr;

let mut l = Layer::new(2, 2, Activation::None);
let x = vec![Expr::new_leaf(1.0, "x1"), Expr::new_leaf(2.0, "x2")];
let y = l.forward(x);
```

And a multi-layer perceptron:

```rust
use alpha_micrograd_rust::nn::{MLP, Activation};
use alpha_micrograd_rust::value::Expr;

let mut mlp = MLP::new(
    2, Activation::Tanh, // input layer
    vec![2, 2], Activation::Tanh, // hidden layers
    1, Activation::None, // output layer
);
let x = vec![Expr::new_leaf(1.0, "x1"), Expr::new_leaf(2.0, "x2")];
let y = mlp.forward(x);
```

You can tie it all together by setting up a loss function and backpropagating the gradients:

```rust
use alpha_micrograd_rust::nn::{MLP, Activation};
use alpha_micrograd_rust::value::Expr;

let mut mlp = MLP::new(
    2, Activation::Tanh, // input layer
    vec![2, 2], Activation::Tanh, // hidden layers
    1, Activation::None, // output layer
);
let x = vec![Expr::new_leaf(1.0, "x1"), Expr::new_leaf(2.0, "x2")];
let mut y = mlp.forward(x);

let target = Expr::new_leaf(0.0, "target");
let exponent = Expr::new_leaf(2.0, "exponent");
let mut loss = (y.remove(0) - target).pow(exponent, "loss");
loss.learn(0.001);
```

Put this in a loop and you have a simple neural network training!