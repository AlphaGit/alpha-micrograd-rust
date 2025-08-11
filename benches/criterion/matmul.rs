use alpha_micrograd_rust::{tensors::Tensor, value::Expr};

use criterion::{BatchSize, Criterion, Throughput};
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

fn get_random_numbers(n: u32) -> Vec<f64> {
    let between = Uniform::new_inclusive(-1.0, 1.0);
    let mut rng = thread_rng();
    (1..=n).map(|_| between.sample(&mut rng)).collect()
}

fn get_random_row(n_inputs: u32) -> Vec<Expr> {
    let random_f32s = get_random_numbers(n_inputs);
    random_f32s.into_iter().map(|n| Expr::new_leaf(n))
        .collect()
}

fn get_random_matrix(n_inputs: u32, n_rows: u32) -> Vec<Vec<Expr>> {
    (0..n_rows)
        .map(|_| get_random_row(n_inputs))
        .collect()
}

pub(crate) fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations");
    group.throughput(Throughput::Elements(25 * 25 * 2));
    group.bench_function("matmul", |b| {
        b.iter_batched(|| {
            let matrix1 = get_random_matrix(25, 25);
            let matrix2 = get_random_matrix(25, 25);
            return (matrix1, matrix2);
        }, |(matrix1, matrix2)| {
            mat_mul(&matrix1, &matrix2)
        }, BatchSize::LargeInput);
    });
    group.finish();
}

fn get_random_tensor(n_rows: u32, n_columns: u32) -> Tensor {
    let total_elements = n_rows * n_columns;
    let tensor_data = get_random_numbers(total_elements);
    Tensor::from_data(tensor_data, vec![n_rows as usize, n_columns as usize])
}

pub(crate) fn criterion_benchmark_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations");
    group.throughput(Throughput::Elements(25 * 25 * 2));
    group.bench_function("tensor_matmul", |b| {
        b.iter_batched(|| {
            let tensor1 = get_random_tensor(25, 25);
            let tensor2 = get_random_tensor(25, 25);
            return (tensor1, tensor2);
        }, |(tensor1, tensor2)| {
            &tensor1 * &tensor2
        }, BatchSize::LargeInput);
    });
    group.finish();
}