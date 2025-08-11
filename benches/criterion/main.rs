mod matmul;
mod itersum;
mod learning;

use criterion::{criterion_group, criterion_main};

use matmul::criterion_benchmark as matmul;
use matmul::criterion_benchmark_tensor as matmul_tensor;
use itersum::criterion_benchmark as itersum;
use learning::criterion_benchmark as learning;

criterion_group!(benches, matmul, matmul_tensor, itersum, learning);
criterion_main!(benches);