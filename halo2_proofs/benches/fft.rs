#[macro_use]
extern crate criterion;

use crate::arithmetic::best_fft;
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;

use criterion::{BenchmarkId, Criterion};
use poly::EvaluationDomain;
use rand_core::OsRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for k in 3..19 {
        group.bench_function(BenchmarkId::new("k", k), |b| {
            let domain = EvaluationDomain::<Fp>::new(1, k);
            let n = domain.get_n() as usize;

            let input = vec![Fp::random(OsRng); n];

            let mut a = input.clone();
            let l_a = a.len();

            let omega = Fp::random(OsRng); // would be weird if this mattered
            b.iter(|| {
                best_fft(&mut a, omega, k, domain.get_fft_data(l_a), false);
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
