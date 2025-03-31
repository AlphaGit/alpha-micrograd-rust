use alpha_micrograd_rust::value::Expr;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

pub fn mat_mul(xs: &Vec<Vec<Expr>>, w: &Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    let result: Vec<Vec<Expr>> = xs
        .iter()
        .map(|x| {
            w.iter()
                .map(|w_row| {
                    x.iter()
                        .zip(w_row.iter())
                        .map(|(x, w)| x.clone() * w.clone())
                        .sum()
                })
                .collect()
        })
        .collect();
    result
}

fn get_random_row(n_inputs: u32) -> Vec<Expr> {
    let between = Uniform::new_inclusive(-1.0, 1.0);
    let mut rng = thread_rng();
    (1..=n_inputs)
        .map(|_| between.sample(&mut rng))
        .enumerate()
        .map(|(i, n)| Expr::new_leaf(n, &format!("w_{:}", i)))
        .collect()
}

fn get_random_matrix(n_inputs: u32, n_rows: u32) -> Vec<Vec<Expr>> {
    (0..n_rows)
        .map(|_| get_random_row(n_inputs))
        .collect()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for dimension in 1..=50 {
        let matrix1 = get_random_matrix(dimension, dimension);
        let matrix2 = get_random_matrix(dimension, dimension);

        group.bench_with_input(
            BenchmarkId::from_parameter(dimension), 
            &(matrix1, matrix2), 
            |b, (matrix1, matrix2)| {
                b.iter(|| mat_mul(matrix1, matrix2));
            }
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);



