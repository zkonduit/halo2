//! # halo2_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(
    clippy::op_ref,
    clippy::many_single_char_names,
    clippy::empty_docs,
    clippy::doc_lazy_continuation,
    clippy::single_range_in_vec_init
)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![feature(int_roundings)]

#[cfg(feature = "counter")]
#[macro_use]
extern crate lazy_static;

#[cfg(feature = "counter")]
use lazy_static::lazy_static;

#[cfg(feature = "counter")]
use std::sync::Mutex;

#[cfg(feature = "counter")]
use std::collections::BTreeMap;

#[cfg(feature = "counter")]
lazy_static! {
    static ref FFT_COUNTER: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
    static ref MSM_COUNTER: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
}

pub mod arithmetic;
pub mod circuit;
pub mod fft;
pub use halo2curves;
mod multicore;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
mod helpers;
pub use helpers::SerdeFormat;
pub use helpers::SerdePrimeField;

#[cfg(feature = "icicle_gpu")]
#[allow(unsafe_code)]
mod icicle;
