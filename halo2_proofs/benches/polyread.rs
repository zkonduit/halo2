use std::io::Cursor;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use halo2_proofs::{poly::Polynomial, SerdeFormat, SerdePrimeField};
use halo2curves::bn256::Fr;
use maybe_rayon::{iter::ParallelIterator, slice::ParallelSlice};
use rand_core::OsRng;

pub fn parallel_poly_read_benchmark_unchecked(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_poly_read_unchecked");

    for batch_size in [64, 256, 1024, 4096, 100000, 1000000].iter() {
        let data = setup_random_poly(100_000_000);
        group.bench_function(format!("batch_{}", batch_size), |b| {
            b.iter(|| {
                let mut reader = Cursor::new(data.clone());
                black_box(
                    read::<_, Fr>(&mut reader, SerdeFormat::RawBytesUnchecked, *batch_size)
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

pub fn parallel_poly_read_benchmark_checked(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_poly_read_checked");

    for batch_size in [64, 256, 1024, 4096, 100000, 1000000].iter() {
        let data = setup_random_poly(100_000_000);
        group.bench_function(format!("batch_{}", batch_size), |b| {
            b.iter(|| {
                let mut reader = Cursor::new(data.clone());
                black_box(read::<_, Fr>(&mut reader, SerdeFormat::RawBytes, *batch_size).unwrap())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    parallel_poly_read_benchmark_checked,
    parallel_poly_read_benchmark_unchecked
);
criterion_main!(benches);

fn setup_random_poly(n: usize) -> Vec<u8> {
    let mut rng = OsRng;
    let random_poly = Polynomial::<Fr, usize>::random(n, &mut rng);
    let mut vector_bytes = vec![];
    random_poly
        .write(&mut vector_bytes, SerdeFormat::RawBytes)
        .unwrap();
    vector_bytes
}

pub fn read<R: std::io::Read, F: SerdePrimeField>(
    reader: &mut R,
    format: SerdeFormat,
    batch_size: usize,
) -> std::io::Result<Vec<F>> {
    let poly_len = u32::from_be_bytes({
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        buf
    }) as usize;

    let repr_len = F::default().to_repr().as_ref().len();
    let buffer = {
        let mut buf = vec![0u8; poly_len * repr_len];
        reader.read_exact(&mut buf)?;
        buf
    };

    Ok(buffer
        .par_chunks(repr_len * batch_size)
        .map(|batch| {
            batch
                .chunks(repr_len)
                .map(|chunk| F::read(&mut std::io::Cursor::new(chunk), format))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect())
}
