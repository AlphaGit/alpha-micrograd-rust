/// This example demonstrates how to perform basic operations on [`Expr`] objects.
/// The example calculates the squared difference between two [`Expr`] objects.
/// The [`Expr`] objects are created using the [`Expr::new_leaf`] function.
/// The squared difference is calculated using the [`Expr::pow`] method.
/// The result is printed to the console.
extern crate alpha_micrograd_rust;

use alpha_micrograd_rust::value::Expr;

fn main() {
    let a = Expr::new_leaf(4.0);
    let b = Expr::new_leaf(2.0);

    let difference = a - b;

    let square_exponent = Expr::new_leaf(2.0);
    let squared_diff = difference.pow(square_exponent);

    println!("squared difference: {:.2}", squared_diff.result);
}
