use group::ff::PrimeField;
use std::sync::Arc;

use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarCfg};

use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};

pub use halo2curves::CurveAffine;
use icicle_core::field::Field;
use icicle_core::{
    curve::{Affine, Curve},
    msm,
};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_cuda_runtime::stream::CudaStream;
use std::{env, mem};

type Abc = Field<8, ScalarCfg>;

pub fn should_use_cpu_msm(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K").unwrap_or("8".to_string()), 10).unwrap())
}

fn is_infinity_point(point: &G1Projective) -> bool {
    let inf_point = G1Projective::zero();
    inf_point.z.eq(&point.z)
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

fn icicle_scalars_from_c<C: CurveAffine>(coeffs: &[C::Scalar]) -> Vec<Abc> {
    let _coeffs = [Arc::new(
        coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>(),
    )];

    let _coeffs: &Arc<Vec<[u32; 8]>> = unsafe { mem::transmute(&_coeffs) };
    _coeffs.iter().map(|x| {
        Abc::from(*x)
    }
    ).collect::<Vec<_>>()
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

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
    let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
        unsafe { mem::transmute(&u32_arr[..]) };
    return PrimeField::from_repr(t[0]).unwrap();
}

fn c_from_icicle_point<C: CurveAffine>(point: &G1Projective) -> C::Curve {
    if is_infinity_point(point) {
        let affine_result = C::from_xy(
            repr_from_u32::<C>(&[0u32; 8]),
            repr_from_u32::<C>(&[0u32; 8]),
        );

        affine_result.unwrap().to_curve()
    } else {
        let mut point_aff = Affine::<CurveCfg>::from(*point);

        let x_limbs: [u32; 8] = point_aff.x.into();
        let y_limbs: [u32; 8] = point_aff.y.into();

        let x_limbs = repr_from_u32::<C>(&x_limbs);
        let y_limbs = repr_from_u32::<C>(&y_limbs);

        let affine_result = C::from_xy(
            x_limbs, 
            y_limbs 
        );

        affine_result.unwrap().to_curve()
    }
}

pub fn multiexp_on_device<C: CurveAffine>(mut coeffs: &[C::Scalar], g: &[C]) -> C::Curve {
    let mut cfg = msm::MSMConfig::default();
    let stream = CudaStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    cfg.c = 16;
    cfg.are_scalars_montgomery_form = false;
    let binding = icicle_scalars_from_c::<C>(coeffs);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let binding = icicle_points_from_c(g);
    let g = HostSlice::from_slice(&binding[..]);

    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
    // println!("coeffs gpu: {:?}", coeffs);
    // println!("bases gpu: {:?}", g);

    msm::msm(coeffs, g, &cfg, &mut msm_results[..]).unwrap();

    let mut msm_host_result = vec![G1Projective::zero(); 1];
    stream
        .synchronize()
        .unwrap();

    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    println!("msm point: {:?}", msm_host_result);

    let msm_point = c_from_icicle_point::<C>(&msm_host_result[0]);

    println!("msm point: {:?}", msm_point);
    msm_point
}
