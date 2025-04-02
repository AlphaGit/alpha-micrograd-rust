mod matmul;
mod itersum;

use matmul::criterion_benchmark as matmul;
use itersum::criterion_benchmark as itersum;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, matmul, itersum);
criterion_main!(benches);