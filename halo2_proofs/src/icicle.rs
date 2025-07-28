//! icicle

use ff::Field;
use group::ff::PrimeField;
use halo2curves::bn256::Fr as Bn256Fr;
use crate::{arithmetic::FftGroup, plonk::{Calculation, CalculationInfo, ValueSource}, poly::{ExtendedLagrangeCoeff, Polynomial}};
use std::{any::{Any, TypeId}, ptr};
pub use halo2curves::CurveAffine;
use icicle_runtime::{get_available_memory, memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice}, set_default_device, set_device, stream::IcicleStream};
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField};
use icicle_core::{
    curve::Affine, msm, ntt::{ntt_inplace, NTTConfig, NTTDir}, traits::MontgomeryConvertible, vec_ops::{add_scalars, inv_scalars, mul_scalars, scalar_add, scalar_mul, scalar_sub, sub_scalars, VecOpsConfig}
};
use maybe_rayon::iter::IntoParallelRefIterator;
use maybe_rayon::iter::ParallelIterator;
use std::{env, mem};

/// load cpu or cuda backend
pub fn try_load_and_set_backend_device(device_type: &str) {
    if device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device {}", device_type);
    let device = icicle_runtime::Device::new(device_type, 0 /* =device_id*/);
    set_device(&device).unwrap();
    set_default_device(&device).unwrap();
}

/// should use cpu for fft
pub fn should_use_cpu_fft(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K_FFT").unwrap_or("2".to_string()), 10).unwrap())
}

/// is icicle support the field
pub fn is_gpu_supported_field<G: Any>(_sample_element: &G) -> bool {
    match TypeId::of::<G>() {
        id if id == TypeId::of::<Bn256Fr>() => true,
        _ => false,
    }
}

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
    let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
        unsafe { mem::transmute(&u32_arr[..]) };
    PrimeField::from_repr(t[0]).unwrap()
}

/// process value source
pub fn process_value_source(
    value: &ValueSource,
    value_types: &mut Vec<u32>,
    value_indices: &mut Vec<u32>,
) {
    let (code, idx0, idx1) = match value {
        ValueSource::Constant(v) => (0, *v, 0),
        ValueSource::Intermediate(v) => (1, *v, 0),
        ValueSource::Fixed(a, b) => (2, *a, *b),
        ValueSource::Advice(a, b) => (3, *a, *b),
        ValueSource::Instance(a, b) => (4, *a, *b),
        ValueSource::Challenge(v) => (5, *v, 0),
        ValueSource::Beta() => (6, 0, 0),
        ValueSource::Gamma() => (7, 0, 0),
        ValueSource::Theta() => (8, 0, 0),
        ValueSource::Y() => (9, 0, 0),
        ValueSource::PreviousValue() => (10, 0, 0),
    };
    value_types.push(code);
    value_indices.push(idx0 as u32);
    value_indices.push(idx1 as u32);
}

/// create calculation data
pub fn create_calculation_data<C: CurveAffine>(
    calculations: &[CalculationInfo],
    constants: &[C::ScalarExt],
    rotations: &[i32],
    num_intermediates: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<ScalarField>, Vec<i32>, u32, u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) 
{
    let size = calculations.len();
    let mut icicle_calculations = Vec::with_capacity(size);
    let mut targets = Vec::with_capacity(size);
    let mut value_types = Vec::with_capacity(size * 2);
    let mut value_indices = Vec::with_capacity(size * 4);
    
    let mut horner_value_types: Vec<u32> = Vec::new();
    let mut horner_value_indices: Vec<u32> = Vec::new();
    let mut horner_offsets: Vec<u32> = vec![0; size];
    let mut horner_sizes: Vec<u32> = vec![0; size];

    for (i, info) in calculations.iter().enumerate() {
        let calc_code = match info.calculation {
            Calculation::Add(_, _) => 0,
            Calculation::Sub(_, _) => 1,
            Calculation::Mul(_, _) => 2,
            Calculation::Square(_) => 3,
            Calculation::Double(_) => 4,
            Calculation::Negate(_) => 5,
            Calculation::Horner(_, _, _) => 6,
            Calculation::Store(_) => 7,
        };
        icicle_calculations.push(calc_code);
        targets.push(info.target as u32);

        let (first, second) = match &info.calculation {
            Calculation::Add(a, b)
            | Calculation::Sub(a, b)
            | Calculation::Mul(a, b) => (a, b),
            Calculation::Square(a)
            | Calculation::Double(a)
            | Calculation::Negate(a)
            | Calculation::Store(a) => (a, a),
            Calculation::Horner(a, b_values, c) => {
                horner_offsets[i] = horner_value_types.len() as u32;
                horner_sizes[i] = b_values.len() as u32;
                for b in b_values {
                    process_value_source(b, &mut horner_value_types, &mut horner_value_indices);
                }
                (a, c)
            },
        };

        process_value_source(first, &mut value_types, &mut value_indices);
        process_value_source(second, &mut value_types, &mut value_indices);
    }

    let icicle_rotations: Vec<i32> = rotations.iter().map(|&x| x).collect();
    let icicle_constants = icicle_scalars_from_c_scalars::<C::Scalar>(constants);

    (icicle_calculations, targets, value_types, value_indices, icicle_constants, icicle_rotations, size as u32, num_intermediates as u32, horner_value_types, horner_value_indices, horner_offsets, horner_sizes)
}

/// unsafe flatten
pub fn unsafe_flatten<C: CurveAffine>(poly_vec: &[Polynomial<C::Scalar, ExtendedLagrangeCoeff>]) -> Vec<C::ScalarExt> {
    let total_size: usize = poly_vec.iter().map(|poly| poly.num_coeffs()).sum();
    
    let mut result: Vec<C::ScalarExt> = Vec::with_capacity(total_size);

    unsafe {
        let result_ptr = result.as_mut_ptr();
        let mut offset = 0;

        for poly in poly_vec {
            let poly_len = poly.num_coeffs();
            let poly_ptr = poly.as_ptr();
            
            // Perform direct memory copy
            ptr::copy_nonoverlapping(poly_ptr, result_ptr.add(offset), poly_len);
            offset += poly_len;
        }
        
        // Manually set the length of result
        result.set_len(total_size);
    }

    result
}

/// create gate data
pub fn create_gate_data<C: CurveAffine>(
    advice: &[Polynomial<C::Scalar, ExtendedLagrangeCoeff>],
    instance: &[Polynomial<C::Scalar, ExtendedLagrangeCoeff>],
    challenges: &[C::ScalarExt],
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    y: C::ScalarExt,
) -> (DeviceVec<ScalarField>, DeviceVec<ScalarField>, Vec<ScalarField>, Vec<ScalarField>, Vec<ScalarField>, Vec<ScalarField>, Vec<ScalarField>)
{
    let icicle_instance = device_vec_from_poly_vec::<C, ExtendedLagrangeCoeff>(instance, &IcicleStream::default());
    let icicle_advice = device_vec_from_poly_vec::<C, ExtendedLagrangeCoeff>(advice, &IcicleStream::default());

    let icicle_challenges = icicle_scalars_from_c_scalars::<C::Scalar>(challenges);
    let icicle_beta = icicle_scalars_from_c_scalars::<C::Scalar>(&[beta]);
    let icicle_gamma = icicle_scalars_from_c_scalars::<C::Scalar>(&[gamma]);
    let icicle_theta = icicle_scalars_from_c_scalars::<C::Scalar>(&[theta]); 
    let icicle_y = icicle_scalars_from_c_scalars::<C::Scalar>(&[y]);

    (icicle_advice, icicle_instance, icicle_challenges, icicle_beta, icicle_gamma, icicle_theta, icicle_y)
}

/// device vec from poly vec
pub fn device_vec_from_poly_vec<C: CurveAffine, B>(
    poly_vec: &[Polynomial<C::Scalar, B>],
    stream: &IcicleStream,
) -> DeviceVec<ScalarField> {
    if poly_vec.len() == 0 {
        return DeviceVec::<ScalarField>::device_malloc_async(0, stream).unwrap();
    }
    let total_len: usize = poly_vec.iter().map(|poly| poly.num_coeffs()).sum();

    let mut device_vec = DeviceVec::<ScalarField>::device_malloc_async(total_len, stream).unwrap();

    let mut offset = 0;
    for poly in poly_vec {
        let values: &[C::Scalar] = poly.as_ref();
        let len = values.len();

        let icicle_values: &[ScalarField] = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const ScalarField, len)
        };

        let h_icicle_values = HostSlice::from_slice(icicle_values);
        device_vec[offset..offset + len].copy_from_host_async(h_icicle_values, stream).unwrap();
        offset += len;
    }

    ScalarField::from_mont(&mut device_vec, stream);

    device_vec
}


/// device vec from c scalars
pub fn device_vec_from_c_scalars<G: Field>(coeffs: &[G], stream: &IcicleStream) -> DeviceVec<ScalarField> {
    let icicle_scalars = unsafe { &*(coeffs as *const _ as *const [ScalarField]) };

    // Create a HostSlice from the mutable slice
    let icicle_host_slice = HostSlice::from_slice(icicle_scalars);

    let mut icicle_scalars = DeviceVec::<ScalarField>::device_malloc_async(coeffs.len(), stream).unwrap();
    icicle_scalars
        .copy_from_host_async(icicle_host_slice, stream)
        .unwrap();

    // Convert from Montgomery representation using the Icicle type's conversion method
    ScalarField::from_mont(&mut icicle_scalars, stream);

    icicle_scalars
}

/// c scalars from device vec 
pub fn c_scalars_from_device_vec<G: PrimeField>(device_vec: &mut DeviceVec<ScalarField>, stream: &IcicleStream) -> Vec<G> {
    let len = device_vec.len();
    let mut host_vec = Vec::with_capacity(len);
    unsafe { host_vec.set_len(len); }
    let host_slice = HostSlice::from_mut_slice(&mut host_vec[..]);
    // Convert from Icicle's representation back to Montgomery representation
    ScalarField::to_mont(&mut device_vec[..], stream);

    device_vec.copy_to_host_async(host_slice, stream).unwrap();

    stream.synchronize().unwrap();
    unsafe { std::mem::transmute::<_, Vec<G>>(host_vec) }
}

/// icicle invert
pub fn icicle_invert<G: PrimeField>(coeffs: &[G], stream: &IcicleStream) -> Vec<G> {    
    let mut d_result = device_vec_from_c_scalars(coeffs, stream);
    inplace_invert(&mut d_result, stream);
    c_scalars_from_device_vec(&mut d_result, stream)
}

/// icicle scalars from c scalars
pub fn icicle_scalars_from_c_scalars<G: PrimeField>(coeffs: &[G]) -> Vec<ScalarField> {
    coeffs.par_iter().map(|coef| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
        ScalarField::from(repr)
    }).collect()
}

/// icicle scalar from c scalar
pub fn icicle_scalar_from_c_scalar<G: PrimeField>(coef: &G) -> ScalarField {
    let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
    ScalarField::from(repr)
}

/// icicle points from c
pub fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    bases.par_iter().map(|p| {
        let coordinates = p.coordinates().unwrap();
        let x_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.x().to_repr()) };
        let y_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.y().to_repr()) };

        Affine::<CurveCfg>::from_limbs(x_repr, y_repr)
    }).collect()
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

    affine.unwrap().to_curve()
}

/// multiexp on device
pub fn multiexp_on_device<C: CurveAffine>(coeffs: &[C::Scalar], bases: &DeviceSlice<Affine<CurveCfg>>, stream: &IcicleStream) -> C::Curve {
    let coeffs = device_vec_from_c_scalars::<C::ScalarExt>(coeffs, stream);

    let mut msm_results = DeviceVec::<G1Projective>::device_malloc_async(1, stream).unwrap();
    let mut cfg = msm::MSMConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;

    msm::msm(&coeffs, bases, &cfg, &mut msm_results[..]).unwrap();

    let mut msm_host_result = [G1Projective::zero(); 1];
    msm_results
        .copy_to_host_async(HostSlice::from_mut_slice(&mut msm_host_result[..]), stream)
        .unwrap();

    c_from_icicle_point::<C>(&msm_host_result[0])
}

/// fft on device
pub fn fft_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    inverse: bool,
    stream: &IcicleStream,
) {
    let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };
    let mut cfg = NTTConfig::<ScalarField>::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;

    let mut icicle_scalars = device_vec_from_c_scalars(scalars, stream);

    ntt_inplace::<ScalarField, ScalarField>(
        &mut icicle_scalars,
        dir,
        &cfg,
    ).unwrap();

    let c_scalars = &c_scalars_from_device_vec::<G>(&mut icicle_scalars, stream)[..];
    scalars.copy_from_slice(c_scalars);

}

/// fft on device
pub fn fft_on_device_vec<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    inverse: bool,
    stream: &IcicleStream,
) -> DeviceVec<ScalarField> {
    let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };
    let mut cfg = NTTConfig::<ScalarField>::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;

    let mut icicle_scalars = device_vec_from_c_scalars(scalars, stream);

    ntt_inplace::<ScalarField, ScalarField>(
        &mut icicle_scalars,
        dir,
        &cfg,
    ).unwrap();

    icicle_scalars
}

/// print available memory
pub fn print_available_memory(label: &str) {
    let (total, free) = get_available_memory().unwrap();
    let used = total - free;
    println!("{:?}", label);
    println!("Total memory: {} MB", total / 1024 / 1024);
    println!("Free memory: {} MB", free / 1024 / 1024);
    println!("Used memory: {} MB", used / 1024 / 1024);
    println!("--------------------------------");
}

/// inplace invert
pub fn inplace_invert(device_vec: &mut DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    inv_scalars(d_copy, device_vec, &cfg).unwrap();
}

/// inplace mul
pub fn inplace_mul(device_vec: &mut DeviceSlice<ScalarField>, other: &DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    mul_scalars(d_copy, other, device_vec, &cfg).unwrap();
}

/// inplace add
pub fn inplace_add(device_vec: &mut DeviceSlice<ScalarField>, other: &DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    add_scalars(d_copy, other, device_vec, &cfg).unwrap();
}

/// inplace sub
pub fn inplace_sub(device_vec: &mut DeviceSlice<ScalarField>, other: &DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    sub_scalars(d_copy, other, device_vec, &cfg).unwrap();
}

/// inplace scalar add
pub fn inplace_scalar_add(device_vec: &mut DeviceSlice<ScalarField>, scalar: &DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    scalar_add(scalar, d_copy, device_vec, &cfg).unwrap();
}

/// inplace scalar mul
pub fn inplace_scalar_mul(device_vec: &mut DeviceSlice<ScalarField>, scalar: &DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    scalar_mul(scalar, d_copy, device_vec, &cfg).unwrap();
}

/// inplace scalar sub
pub fn inplace_scalar_sub(device_vec: &mut DeviceSlice<ScalarField>, scalar: &DeviceSlice<ScalarField>, stream: &IcicleStream) {
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = stream.into();
    cfg.is_async = true;
    let d_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            device_vec.as_mut_ptr(),
            device_vec.len(),
        ))
    };
    scalar_sub(scalar, d_copy, device_vec, &cfg).unwrap();
}
