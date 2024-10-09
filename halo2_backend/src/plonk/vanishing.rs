use std::marker::PhantomData;

use crate::arithmetic::CurveAffine;

pub(crate) mod prover;
pub(crate) mod verifier;

/// A vanishing argument.
pub(crate) struct Argument<C: CurveAffine> {
    _marker: PhantomData<C>,
}
