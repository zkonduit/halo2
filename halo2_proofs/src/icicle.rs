use group::ff::PrimeField;
use icicle::{curves::bn254::{Point_BN254, ScalarField_BN254}, test_bn254::{commit_bn254, commit_batch_bn254}};
use std::sync::{Arc, Once};

pub use icicle::curves::bn254::PointAffineNoInfinity_BN254;
use rustacuda::memory::CopyDestination;
use rustacuda::prelude::*;

pub use halo2curves::CurveAffine;
use std::{mem, env};
use log::info;

static mut GPU_CONTEXT: Option<Context> = None;
static mut GPU_G: Option<DeviceBuffer<PointAffineNoInfinity_BN254>> = None;
static mut GPU_G_LAGRANGE: Option<DeviceBuffer<PointAffineNoInfinity_BN254>> = None;
static GPU_INIT: Once = Once::new();

pub fn is_small_circuit(size: usize) -> bool {
    size <= (1 << u8::from_str_radix(&env::var("ICICLE_SMALL_CIRCUIT").unwrap_or("8".to_string()), 10).unwrap())
}

pub fn init_gpu<C: CurveAffine>(g: &[C], g_lagrange: &[C]) {
    unsafe {
        GPU_INIT.call_once(|| {
            GPU_CONTEXT = Some(rustacuda::quick_init().unwrap());
            GPU_G = Some(copy_points_to_device(g));
            GPU_G_LAGRANGE = Some(copy_points_to_device(g_lagrange));
            info!("GPU initialized");
        });
    }
}

fn u32_from_u8(u8_arr: &[u8;32]) -> [u32;8]{
    let mut t = [0u32;8];
    for i in 0..8{
        t[i] = u32::from_le_bytes([u8_arr[4*i],u8_arr[4*i+1],u8_arr[4*i+2],u8_arr[4*i+3]]);
    }
    return t; 
}

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32;8]) -> <C as CurveAffine>::Base {
    let t : &[<<C as CurveAffine>::Base as PrimeField>::Repr] = unsafe { mem::transmute(&u32_arr[..]) };
    return PrimeField::from_repr(t[0]).unwrap();
}

fn is_infinity_point(point: Point_BN254) -> bool {
    let inf_point = Point_BN254::infinity();
    point.z.s.eq(&inf_point.z.s)
}

fn icicle_scalars_from_c<C: CurveAffine>(coeffs: &[C::Scalar]) -> Vec<ScalarField_BN254> {
    let _coeffs = [Arc::new(
        coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>(),
    )];
    
    let _coeffs: &Arc<Vec<[u32;8]>> = unsafe { mem::transmute(&_coeffs) };
    _coeffs.iter().map(|x| {
        ScalarField_BN254::from_limbs(x)
    }).collect::<Vec<_>>()
}

pub fn copy_scalars_to_device<C: CurveAffine>(coeffs: &[C::Scalar]) -> DeviceBuffer<ScalarField_BN254> {
    let scalars = icicle_scalars_from_c::<C>(coeffs);

    DeviceBuffer::from_slice(scalars.as_slice()).unwrap()
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<PointAffineNoInfinity_BN254> {
    let _bases = [Arc::new(
        bases.iter().map(|p| {
            let coordinates = p.coordinates().unwrap();
            [coordinates.x().to_repr(),coordinates.y().to_repr()]
        }).collect::<Vec<_>>(),
    )];
    
    let _bases: &Arc<Vec<[[u8;32];2]>> = unsafe { mem::transmute(&_bases) };
    _bases.iter().map(|x| {
        let tx = u32_from_u8(&x[0]);
        let ty = u32_from_u8(&x[1]);
        PointAffineNoInfinity_BN254::from_limbs(&tx,&ty)
    }).collect::<Vec<_>>()
}

pub fn copy_points_to_device<C: CurveAffine>(bases: &[C]) -> DeviceBuffer<PointAffineNoInfinity_BN254> {
    let points = icicle_points_from_c(bases);
    
    DeviceBuffer::from_slice(points.as_slice()).unwrap()
}

fn c_from_icicle_point<C: CurveAffine>(commit_res: Point_BN254) -> C::Curve {    
    let (x , y) = if is_infinity_point(commit_res){
        (repr_from_u32::<C>(&[0u32;8]), repr_from_u32::<C>(&[0u32;8]))
    } else{
        let affine_res_from_cuda = commit_res.to_affine();
        (repr_from_u32::<C>(&affine_res_from_cuda.x.s), repr_from_u32::<C>(&affine_res_from_cuda.y.s))
    };

    let affine = C::from_xy(x,y).unwrap();
    return affine.to_curve();
}

pub fn multiexp_on_device<C: CurveAffine>(mut coeffs: DeviceBuffer<ScalarField_BN254>, is_lagrange: bool) -> C::Curve {    
    let base_ptr: &mut DeviceBuffer<PointAffineNoInfinity_BN254>;
    unsafe {
        if is_lagrange {
            base_ptr = GPU_G_LAGRANGE.as_mut().unwrap();
        } else {
            base_ptr = GPU_G.as_mut().unwrap();
        };
    }

    let d_commit_result = commit_bn254(base_ptr, &mut coeffs, 10);

    let mut h_commit_result = Point_BN254::zero();
    d_commit_result
        .copy_to(&mut h_commit_result)
        .unwrap();

    c_from_icicle_point::<C>(h_commit_result)
}

pub fn batch_multiexp_on_device<C: CurveAffine>(mut coeffs: DeviceBuffer<ScalarField_BN254>, mut bases: DeviceBuffer<PointAffineNoInfinity_BN254>, batch_size: usize) -> Vec<C::Curve> {    
    let d_commit_result = commit_batch_bn254(&mut bases, &mut coeffs, batch_size);
    let mut h_commit_result: Vec<Point_BN254> = (0..batch_size)
        .map(|_| Point_BN254::zero())
        .collect();
    d_commit_result
        .copy_to(&mut h_commit_result[..])
        .unwrap();

    h_commit_result.iter().map(|commit_result| c_from_icicle_point::<C>(*commit_result)).collect()
}
