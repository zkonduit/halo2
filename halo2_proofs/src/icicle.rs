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
use std::{env, mem};

pub fn should_use_cpu_msm(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K").unwrap_or("8".to_string()), 10).unwrap())
}

pub fn should_use_cpu_fft(size: u32) -> bool {
    size as u8 <= (u8::from_str_radix(&env::var("ICICLE_SMALL_K_FFT").unwrap_or("5".to_string()), 10).unwrap())
}

pub fn is_gpu_supported_field<G: Any>(sample_element: &G) -> bool {
    match TypeId::of::<G>() {
        id if id == TypeId::of::<Bn256Fr>() => true,
        _ => false,
    }
}

fn u32_from_u8(u8_arr: &[u8; 32]) -> [u32; 8] {
    let mut t = [0u32; 8];
    for i in 0..8 {
        t[i] = u32::from_le_bytes([
            u8_arr[4 * i],
            u8_arr[4 * i + 1],
            u8_arr[4 * i + 2],
            u8_arr[4 * i + 3],
        ]);
    }
    return t;
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
    let _coeffs = [Arc::new(
        coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>(),
    )];

    let _coeffs: &Arc<Vec<[u32; 8]>> = unsafe { mem::transmute(&_coeffs) };
    _coeffs
        .iter()
        .map(|x| ScalarField::from(*x))
        .collect::<Vec<_>>()
}

fn icicle_scalars_from_c_scalars<G: PrimeField>(coeffs: &[G]) -> Vec<ScalarField> {
    let _coeffs = [Arc::new(
        coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>(),
    )];

    let _coeffs: &Arc<Vec<[u32; 8]>> = unsafe { mem::transmute(&_coeffs) };
    _coeffs
        .iter()
        .map(|x| ScalarField::from(*x))
        .collect::<Vec<_>>()
}

fn c_scalars_from_icicle_scalars<G: PrimeField>(scalars: &[ScalarField]) -> Vec<G> {
    scalars
        .iter()
        .map(|x| {
            let mut repr = G::Repr::default();
            repr.as_mut().copy_from_slice(&x.to_bytes_le()[..]);
            G::from_repr(repr).unwrap()
        })
        .collect()
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    let _bases = [Arc::new(
        bases
            .iter()
            .map(|p| {
                let coordinates = p.coordinates().unwrap();
                [coordinates.x().to_repr(), coordinates.y().to_repr()]
            })
            .collect::<Vec<_>>(),
    )];

    let _bases: &Arc<Vec<[[u8; 32]; 2]>> = unsafe { mem::transmute(&_bases) };
    _bases
        .iter()
        .map(|x| {
            let tx = u32_from_u8(&x[0]);
            let ty = u32_from_u8(&x[1]);

            Affine::<CurveCfg>::from_limbs(tx, ty)
        })
        .collect::<Vec<_>>()
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
    let binding = icicle_scalars_from_c::<C>(coeffs);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);

    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
    let mut cfg = msm::MSMConfig::default();
    let stream = CudaStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    cfg.large_bucket_factor = 10;
    cfg.c = 16;
    msm::msm(coeffs, bases, &cfg, &mut msm_results[..]).unwrap();
    stream.synchronize().unwrap();

    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    let msm_point = c_from_icicle_point::<C>(&msm_host_result[0]);

    msm_point
}

pub fn ntt_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(scalars: &mut [G], omega: Scalar, log_n: u32, inverse: bool) {
    let size: usize = 1 << log_n;

    let mut cfg = NTTConfig::<'_, ScalarField>::default();
    let stream = CudaStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    
    let icicle_omega = get_root_of_unity::<ScalarField>((size as u64) * 10);
    // let icicle_omega = icicle_scalars_from_c_scalars(&[omega])[0];
    initialize_domain(icicle_omega, &cfg.ctx, true).unwrap();

    let mut ntt_results = DeviceVec::<ScalarField>::cuda_malloc(size).unwrap();
    let mut icicle_scalars: Vec<ScalarField> = icicle_scalars_from_c_scalars(scalars);

    let host_scalars = HostSlice::from_slice(&icicle_scalars);
    
    ntt::<ScalarField, ScalarField>(
        host_scalars,
        if inverse { NTTDir::kInverse } else { NTTDir::kForward },
        &cfg,
        &mut ntt_results[..],
    ).unwrap();

    stream.synchronize().unwrap();

    let mut ntt_host_result = vec![ScalarField::zero(); size];
    ntt_results
        .copy_to_host(HostSlice::from_mut_slice(&mut ntt_host_result[..]))
        .unwrap();

    let mut c_scalars = &c_scalars_from_icicle_scalars::<G>(&mut ntt_host_result)[..];
    scalars.copy_from_slice(&c_scalars);
}
