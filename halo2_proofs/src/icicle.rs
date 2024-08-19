use group::ff::PrimeField;
use std::sync::Arc;

use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField, BaseField};
use halo2curves::bn256::Fr as Bn256Fr;
use icicle_cuda_runtime::{stream::CudaStream, memory::{DeviceVec, HostSlice}};
use crate::arithmetic::FftGroup;
use std::any::TypeId;
use std::any::Any;
pub use halo2curves::CurveAffine;
use icicle_core::{
    curve::Affine,
    msm,
    ntt::{get_root_of_unity, initialize_domain, ntt, FieldImpl, NTTConfig, NTTDir},
};
use std::time::{Duration, Instant};
use lazy_static::lazy_static;
use std::sync::Mutex;
use maybe_rayon::iter::IntoParallelRefIterator;
use maybe_rayon::iter::ParallelIterator;

lazy_static! {
    /// a
    pub static ref TOTAL_DURATION_NTT: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// b
    pub static ref TOTAL_DURATION_INITIALIZATION: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// c
    pub static ref TOTAL_DURATION_EXECUTION: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// d
    pub static ref TOTAL_DURATION_COPY_TO_HOST: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// e
    pub static ref TOTAL_DURATION_CONVERT_TO_ORIGINAL: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// f
    pub static ref TOTAL_DURATION_CONVERT_TO_ICICLE: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// f
    pub static ref TOTAL_DURATION_CONVERT_TO_ORIGINAL_1: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// g
    pub static ref TOTAL_DURATION_CONVERT_TO_ORIGINAL_2: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    /// h
    pub static ref TOTAL_DURATION_CONVERT_TO_ORIGINAL_3: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
}

lazy_static! {
    pub static ref TOTAL_DURATION_MULTIEXP_CONVERT_TO_ICICLE: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    pub static ref TOTAL_DURATION_MULTIEXP_INITIALIZATION: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    pub static ref TOTAL_DURATION_MULTIEXP_EXECUTION: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    pub static ref TOTAL_DURATION_MULTIEXP_COPY_TO_HOST: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
    pub static ref TOTAL_DURATION_MULTIEXP_CONVERT_TO_ORIGINAL: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
}

use std::{env, mem};

pub fn should_use_cpu_msm(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K").unwrap_or("8".to_string()), 10).unwrap())
}

pub fn should_use_cpu_fft(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K_FFT").unwrap_or("8".to_string()), 10).unwrap())
}

pub fn is_gpu_supported_field<G: Any>(_sample_element: &G) -> bool {
    match TypeId::of::<G>() {
        id if id == TypeId::of::<Bn256Fr>() => true,
        _ => false,
    }
}

fn u32_from_u8(u8_arr: &[u8; 32]) -> [u32; 8] {
    unsafe { mem::transmute(*u8_arr) }
}

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
    let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
        unsafe { mem::transmute(&u32_arr[..]) };
    return PrimeField::from_repr(t[0]).unwrap();
}

fn is_infinity_point(point: &G1Projective) -> bool {
    BaseField::zero().eq(&point.z)
}

fn icicle_scalars_from_c<C: CurveAffine>(coeffs: &[C::Scalar]) -> Vec<ScalarField> {
    let results: Vec<ScalarField> = coeffs.par_iter().map(|coef| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
        ScalarField::from(repr)
    }).collect();

    results
}

fn icicle_scalars_from_c_scalars<G: PrimeField>(coeffs: &[G]) -> Vec<ScalarField> {
    let start_time = Instant::now();

    let results: Vec<ScalarField> =  coeffs.par_iter().map(|coef| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
        ScalarField::from(repr)
    }).collect();

    let duration = start_time.elapsed();
    *TOTAL_DURATION_CONVERT_TO_ICICLE.lock().unwrap() += duration;

    results
}

fn c_scalars_from_icicle_scalars<G: PrimeField>(scalars: &[ScalarField]) -> Vec<G> {
    let start_time = Instant::now();

    let results: Vec<G> = scalars.par_iter().map(|scalar| {
        let repr: G::Repr = unsafe { mem::transmute_copy(scalar) };
        G::from_repr(repr).unwrap()
    }).collect();

    let duration = start_time.elapsed();
    *TOTAL_DURATION_CONVERT_TO_ORIGINAL.lock().unwrap() += duration;

    results
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    // Convert base points to their byte representation in parallel
    let results: Vec<Affine<CurveCfg>> = bases.par_iter().map(|p| {
        let coordinates = p.coordinates().unwrap();
        let x_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.x().to_repr()) };
        let y_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.y().to_repr()) };

        Affine::<CurveCfg>::from_limbs(x_repr, y_repr)
    }).collect();

    results
}

fn c_from_icicle_point<C: CurveAffine>(point: &G1Projective) -> C::Curve {
    let (x, y) = {
        let affine: Affine<CurveCfg> = Affine::<CurveCfg>::from(*point);

        (
            repr_from_u32::<C>(&affine.x.into()),
            repr_from_u32::<C>(&affine.y.into()),
        )
    };

    let affine = C::from_xy(x, y);

    return affine.unwrap().to_curve();
}

pub fn multiexp_on_device<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    // Measure the time taken to convert coefficients and bases to ICICLE format
    let start = Instant::now();
    let binding = icicle_scalars_from_c::<C>(coeffs);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);
    let duration = start.elapsed();
    *TOTAL_DURATION_MULTIEXP_CONVERT_TO_ICICLE.lock().unwrap() += duration;

    // Measure the time taken to initialize MSM on device
    let start = Instant::now();
    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
    let mut cfg = msm::MSMConfig::default();
    let stream = CudaStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    cfg.large_bucket_factor = 10;
    cfg.c = 16;
    let duration = start.elapsed();
    *TOTAL_DURATION_MULTIEXP_INITIALIZATION.lock().unwrap() += duration;

    // Measure the time taken to execute MSM
    let start = Instant::now();
    msm::msm(coeffs, bases, &cfg, &mut msm_results[..]).unwrap();
    stream.synchronize().unwrap();
    let duration = start.elapsed();
    *TOTAL_DURATION_MULTIEXP_EXECUTION.lock().unwrap() += duration;

    // Measure the time taken to copy the results to host
    let start = Instant::now();
    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();
    let duration = start.elapsed();
    *TOTAL_DURATION_MULTIEXP_COPY_TO_HOST.lock().unwrap() += duration;

    // Measure the time taken to convert the result back to the original format
    let start = Instant::now();
    let msm_point = c_from_icicle_point::<C>(&msm_host_result[0]);
    let duration = start.elapsed();
    *TOTAL_DURATION_MULTIEXP_CONVERT_TO_ORIGINAL.lock().unwrap() += duration;

    msm_point
}
pub fn ntt_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    _omega: Scalar, 
    log_n: u32, 
    inverse: bool
) {
    let size: usize = 1 << log_n;

    let start_initialization = Instant::now();
    let mut cfg = NTTConfig::<'_, ScalarField>::default();
    cfg.is_async = false;

    let icicle_omega = get_root_of_unity::<ScalarField>((size as u64) * 10);
    initialize_domain(icicle_omega, &cfg.ctx, true).unwrap();

    let mut ntt_results = DeviceVec::<ScalarField>::cuda_malloc(size).unwrap();
    let icicle_scalars: Vec<ScalarField> = icicle_scalars_from_c_scalars(scalars);
    let host_scalars = HostSlice::from_slice(&icicle_scalars);

    let duration_initialization = start_initialization.elapsed();
    *TOTAL_DURATION_INITIALIZATION.lock().unwrap() += duration_initialization;

    let start_execution = Instant::now();
    ntt::<ScalarField, ScalarField>(
        host_scalars,
        if inverse { NTTDir::kInverse } else { NTTDir::kForward },
        &cfg,
        &mut ntt_results[..],
    ).unwrap();

    let duration_execution = start_execution.elapsed();
    *TOTAL_DURATION_EXECUTION.lock().unwrap() += duration_execution;

    // Measure time to copy results to host
    let start_copy_to_host = Instant::now();
    let mut ntt_host_result = vec![ScalarField::zero(); size];
    ntt_results
        .copy_to_host(HostSlice::from_mut_slice(&mut ntt_host_result[..]))
        .unwrap();
    let duration_copy_to_host = start_copy_to_host.elapsed();
    *TOTAL_DURATION_COPY_TO_HOST.lock().unwrap() += duration_copy_to_host;

    let start_convert_to_original = Instant::now();
    // Convert back to original scalar type
    let c_scalars = &c_scalars_from_icicle_scalars::<G>(&mut ntt_host_result)[..];
    scalars.copy_from_slice(&c_scalars);
    let duration_convert_to_original = start_convert_to_original.elapsed();
    *TOTAL_DURATION_CONVERT_TO_ORIGINAL.lock().unwrap() += duration_convert_to_original;

    // Measure total time for the NTT operation
    let duration_ntt = start_initialization.elapsed();
    *TOTAL_DURATION_NTT.lock().unwrap() += duration_ntt;
}
