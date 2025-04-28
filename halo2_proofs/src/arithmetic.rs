//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

#[cfg(feature = "icicle_gpu")]
use super::icicle;
#[cfg(feature = "icicle_gpu")]
use std::env;
use super::multicore;
pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    prime::PrimeCurveAffine,
    Curve, GroupOpsOwned, ScalarMulOwned,
};

use halo2curves::msm::msm_best;
pub use halo2curves::{CurveAffine, CurveExt};

use maybe_rayon::iter::ParallelIterator;
use maybe_rayon::prelude::ParallelSliceMut;
use maybe_rayon::iter::IndexedParallelIterator;

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

/// Best MSM
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    #[cfg(feature = "icicle_gpu")]
    if env::var("ENABLE_ICICLE_GPU").is_ok()
        && !icicle::should_use_cpu_msm(coeffs.len())
        && icicle::is_gpu_supported_field(&coeffs[0])
    {
        return best_multiexp_gpu(coeffs, bases);
    }

    #[cfg(feature = "metal")]
    {
        use mopro_msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
        use std::sync::Once;

        // Static mutex to block concurrent Metal acceleration calls
        static PRINT_ONCE: Once = Once::new();

        // Print the warning message only once
        PRINT_ONCE.call_once(|| {
            log::warn!(
                "WARNING: Using Experimental Metal Acceleration for MSM. \
                 Best performance improvements are observed with log row size >= 20. \
                 Current log size: {}",
                coeffs.len().ilog2()
            );
        });

        // Perform MSM using Metal acceleration
        return mopro_msm::metal::msm_best::<C, H2GAffine, H2G, H2Fr>(coeffs, bases);
    }

    #[allow(unreachable_code)]
    best_multiexp_cpu(coeffs, bases)
}

// [JPW] Keep this adapter to halo2curves to minimize code changes.
/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp_cpu<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    msm_best(coeffs, bases)
}

#[cfg(feature = "icicle_gpu")]
/// Performs a multi-exponentiation operation on GPU using Icicle library
pub fn best_multiexp_gpu<C: CurveAffine>(coeffs: &[C::Scalar], g: &[C]) -> C::Curve {
    icicle::multiexp_on_device::<C>(coeffs, g)
}

/// Dispatcher
pub fn best_fft_cpu<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    fft::fft(a, omega, log_n, data, inverse);
}

/// Best FFT
pub fn best_fft<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    #[cfg(feature = "icicle_gpu")]
    if env::var("ENABLE_ICICLE_GPU").is_ok()
        && !icicle::should_use_cpu_fft(scalars.len())
        && icicle::is_gpu_supported_field(&omega)
    {
        best_fft_gpu(scalars, omega, log_n, inverse);
    } else {
        best_fft_cpu(scalars, omega, log_n, data, inverse);
    }

    #[cfg(not(feature = "icicle_gpu"))]
    best_fft_cpu(scalars, omega, log_n, data, inverse);
}

/// Performs a NTT operation on GPU using Icicle library
#[cfg(feature = "icicle_gpu")]
pub fn best_fft_gpu<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    inverse: bool,
) {
    println!("icicle_fft");
    icicle::fft_on_device::<Scalar, G>(a, omega, log_n, inverse);
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: PrimeCurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let n_inv = C::Scalar::TWO_INV.pow_vartime([k as u64, 0, 0, 0]);
    let omega = C::Scalar::ROOT_OF_UNITY;
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }

    let mut g_lagrange_projective = g_projective;
    let n = g_lagrange_projective.len();
    let fft_data = FFTData::new(n, omega, omega_inv);

    best_fft_cpu(&mut g_lagrange_projective, omega_inv, k, &fft_data, true);
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });

    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });

    g_lagrange
}
/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::ZERO, |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ZERO; num_threads];
        multicore::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime([start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::ZERO, |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    // TODO: parallelize?
    assert_eq!(a.len(), b.len());

    let mut acc = F::ZERO;
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Performs element-wise addition of vector `b` multiplied by scalar `k` to vector `a`.
/// This function executes in parallel to improve performance.
fn vector_add_mul_scalar_inplace<F: Field>(a: &mut Vec<F>, b: &Vec<F>, k: F)
{
    parallelize(a, |lhs, start| {
        for (lhs, rhs) in lhs
            .iter_mut()
            .zip(b[start..].iter())
        {
            *lhs += (*rhs) *k;
        }
    });
}

/// Multiplies each element of vector `a` by scalar `k` in place.
/// This function executes in parallel to enhance performance.
pub fn vector_mul_scalar_inplace<F: Field>(a: &mut Vec<F>, k: F)
{
    parallelize(a, |lhs, _| {
        for lh in lhs
                    .iter_mut()
        {
            *lh *= k;
        }
    });
}

/// Computes the first `n` powers of the element `b`.
/// /// This function utilizes parallel computation to optimize the power calculations.
pub fn compute_powers<F: Field>(b: F, n: usize) -> Vec<F>{

    let num_threads = multicore::current_num_threads();
    let chunk_size = (n + num_threads-1)/ num_threads;
    let mut pows = vec!(F::ZERO; n);

    pows.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(cid, pows)| {

            let start = cid * chunk_size;
            let mut prvb  = b.pow_vartime([start as u64, 0, 0, 0]);

            for pow in pows.iter_mut() {
                *pow  = prvb;
                prvb *= b;
            }
    });

    pows
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - 1];

    let mut tmp = F::ZERO;
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division_par<F: Field>(poly: &[F], b: F) -> Vec<F>
{
    fn poly_div<F: Field>(a: &[F], b: F, q: &mut [F], remd: &mut F) {
        let b_neg = -b;

        let mut tmp = F::ZERO;
        for i in (0..a.len()-1).rev() {

            let r = a[i+1];
            let mut lead_coeff = r;
            lead_coeff.sub_assign(&tmp);

            q[i] = lead_coeff;
            tmp = lead_coeff;
            tmp.mul_assign(&b_neg);
        }

        *remd = a[0] - tmp;
    }

    let n = poly.len();
    let num_threads  = multicore::current_num_threads(); // n*2; // multicore::current_num_threads();
    let mut quotient = vec![F::ZERO; n];
    let mut remaindr = vec![F::ZERO; num_threads];


    if n < num_threads {
        poly_div(&poly[0..n], b, &mut quotient, &mut remaindr[0]);
    }


    else
    {
        let start = instant::Instant::now();
        let chunk_size = (n + num_threads - 1) / num_threads;


        multicore::scope(|scope| {
            for (chunk_idx, (r, mut q)) in remaindr
                                        .chunks_mut(1)
                                        .zip(quotient.chunks_mut(chunk_size))
                                        .enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    let max   = start + chunk_size;
                    let end   = if max<n {max} else {n};

                    poly_div(&poly[start..end], b, &mut q, &mut r[0]);
                });
            }
        });
        /*---------------------------------------------------
         let n = num_threads = 2, c = chunk_size
         quotient[0] = 0, 1, ..., c-2   ; remainders[0] = 0;
         quotient[1] = c + (0, ..., c-2); remainders[1] = c;

         poly     = 0, 1, ..., 2c - 1
         quotient = 0, 1, ..., 2c - 2
         ----------------------------------------------------*/
        let comp_div_poly_t = start.elapsed();


        let start = instant::Instant::now();

        let mut bpows = compute_powers(b, chunk_size)[0..chunk_size].to_vec();
        bpows.reverse();

        for tid in (0..num_threads-1).rev() {
            let r = remaindr[tid+1];
            remaindr[tid] += r*bpows[0]*b;
        }

        let pows_of_b_t = start.elapsed();


        let start = instant::Instant::now();

        quotient.par_chunks_mut(chunk_size)
                .enumerate()
                .take(num_threads - 1)
                .for_each(|(tid, qs)| {

                    let r = remaindr[tid+1];
                    for (i, q) in qs.iter_mut().enumerate() {

                        let pow = bpows[i];
                        *q = *q + (pow*r);
                    }

                });

        for tid in 1..num_threads {
            quotient[tid*chunk_size-1] = remaindr[tid];
        }
        let vec_sub_t = start.elapsed();


        // println!("kate_division_par comp_div_poly_t: {:?}, pows_of_b_t: {:?}, vec_sub_t: {:?}, chunk_size: {chunk_size}.",
        //           comp_div_poly_t, pows_of_b_t, vec_sub_t, );
    };


    quotient[0..n-1].to_vec()
}

/// Divides polynomial `a` in `X` by `(X - b0) (X - b1)` with
/// no remainder.
pub fn kate_division_deg2<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, b: &[F]) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    let (b1,b2) = (b[0],b[1]);
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - b.len()];
    let mut prv1 = F::ZERO;
    let mut prv2 = F::ZERO;

    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;

        lead_coeff.add_assign(&(prv2 * b2));
        prv2 = lead_coeff;

        lead_coeff.add_assign(&prv1);
        *q = lead_coeff;

        prv1 = lead_coeff * b1;
    }

    q
}

/// Divides polynomial `a` in `X` by `(X - b0) (X - b1) (X - b2)` with
/// no remainder.
pub fn kate_division_deg3<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, b: &[F]) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    let (b1,b2,b3) = (b[0],b[1],b[2]);
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - b.len()];
    let mut prv1 = F::ZERO;
    let mut prv2 = F::ZERO;
    let mut prv3 = F::ZERO;
    let mut prvq = F::ZERO;

    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;

        lead_coeff.add_assign(&(prv3 * b3));
        prv3 = lead_coeff;

        lead_coeff.add_assign(&prv1);
        lead_coeff.add_assign(&((prvq - prv2) * b2));
        *q = lead_coeff;

        prv2 = prv1;
        prv1 = lead_coeff * b1;
        prvq = lead_coeff;
    }

    q
}

/// This utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    // Algorithm rationale:
    //
    // Using the stdlib `chunks_mut` will lead to severe load imbalance.
    // From https://github.com/rust-lang/rust/blob/e94bda3/library/core/src/slice/iter.rs#L1607-L1637
    // if the division is not exact, the last chunk will be the remainder.
    //
    // Dividing 40 items on 12 threads will lead to a chunk size of 40/12 = 3,
    // There will be a 13 chunks of size 3 and 1 of size 1 distributed on 12 threads.
    // This leads to 1 thread working on 6 iterations, 1 on 4 iterations and 10 on 3 iterations,
    // a load imbalance of 2x.
    //
    // Instead we can divide work into chunks of size
    // 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3 = 4*4 + 3*8 = 40
    //
    // This would lead to a 6/4 = 1.5x speedup compared to naive chunks_mut
    //
    // See also OpenMP spec (page 60)
    // http://www.openmp.org/mp-documents/openmp-4.5.pdf
    // "When no chunk_size is specified, the iteration space is divided into chunks
    // that are approximately equal in size, and at most one chunk is distributed to
    // each thread. The size of the chunks is unspecified in this case."
    // This implies chunks are the same size ±1

    let f = &f;
    let total_iters = v.len();
    let num_threads = multicore::current_num_threads();
    let base_chunk_size = total_iters / num_threads;
    let cutoff_chunk_id = total_iters % num_threads;
    let split_pos = cutoff_chunk_id * (base_chunk_size + 1);
    let (v_hi, v_lo) = v.split_at_mut(split_pos);

    multicore::scope(|scope| {
        // Skip special-case: number of iterations is cleanly divided by number of threads.
        if cutoff_chunk_id != 0 {
            for (chunk_id, chunk) in v_hi.chunks_exact_mut(base_chunk_size + 1).enumerate() {
                let offset = chunk_id * (base_chunk_size + 1);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
        // Skip special-case: less iterations than number of threads.
        if base_chunk_size != 0 {
            for (chunk_id, chunk) in v_lo.chunks_exact_mut(base_chunk_size).enumerate() {
                let offset = split_pos + (chunk_id * base_chunk_size);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
    });
}

///
pub fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms)
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: Field>(roots: &[F], z: F) -> F {
    fn evaluate<F: Field>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::ONE, |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ONE; num_threads];
        multicore::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::ONE, |acc, part| acc * part)
    }
}

pub(crate) fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::ONE), move |power| Some(base * power))
}

pub(crate) fn powers_of_x<F: Field>(base: F, n: usize) -> Vec<F> {

    let mut pows = Vec::<F>::with_capacity(n);

    pows.push(F::ONE);
    for i in 1..n {
        pows.push(pows[i-1]*base);
    }
    pows
}

/// Reverse `l` LSBs of bitvector `n`
pub fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[cfg(test)]
use rand_core::OsRng;

use crate::fft::{self, recursive::FFTData};
#[cfg(test)]
use crate::halo2curves::pasta::Fp;
// use crate::plonk::{get_duration, get_time, start_measure, stop_measure};

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}
