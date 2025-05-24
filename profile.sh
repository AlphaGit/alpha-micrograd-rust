CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench criterion -o matmul.svg -- --bench matmul
CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench criterion -o itersum.svg -- --bench itersum
CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench criterion -o learning.svg -- --bench learning