use alpha_micrograd_rust::value::Expr;

use criterion::{BatchSize, Criterion, Throughput};
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

fn get_random_vec(n_inputs: u32) -> Vec<Expr> {
    let between = Uniform::new_inclusive(-1.0, 1.0);
    let mut rng = thread_rng();
    (1..=n_inputs)
        .map(|_| between.sample(&mut rng))
        .map(|n| Expr::new_leaf(n))
        .collect()
}

pub(crate) fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations");
    group.throughput(Throughput::Elements(1000));
    group.bench_function("itersum", |b| {
        b.iter_batched(|| {
            get_random_vec(1000)
        }, |vec| {
            vec.into_iter().sum::<Expr>()
        }, BatchSize::LargeInput);
    });
    group.finish();
}



