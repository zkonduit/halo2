use super::{
    construct_intermediate_sets, ChallengeU, ChallengeV, ChallengeY, Commitment, RotationSet,
};
use crate::arithmetic::{
    eval_polynomial, evaluate_vanishing_polynomial, kate_division, lagrange_interpolate,
    kate_division_deg2, kate_division_deg3, kate_division_par, vector_mul_scalar_inplace, powers_of_x,
    parallelize, powers, CurveAffine,
};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::{Blind, ParamsProver, Prover};
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::{PolynomialPointer, ProverQuery};
use crate::poly::{Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use ff::Field;
use group::Curve;
use halo2curves::pairing::Engine;
use halo2curves::CurveExt;
use maybe_rayon::iter::ParallelIterator;
use maybe_rayon::prelude::IntoParallelIterator;
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::MulAssign;
use std::hash::Hash;
use maybe_rayon::iter::IndexedParallelIterator;

fn div_by_vanishing<F: Field>(poly: Polynomial<F, Coeff>, roots: &[F]) -> Vec<F> {
    let poly = roots
        .iter()
        .fold(poly.values, |poly, point| kate_division(&poly, *point));

    poly
}

fn div_by_vanishing_par<F: Field>(poly: Polynomial<F, Coeff>, roots: &[F]) -> Vec<F> {
    let poly = roots
        .iter()
        .fold(poly.values, |poly, point| kate_division_par(&poly, *point));
    poly
}

struct CommitmentExtension<'a, C: CurveAffine> {
    commitment: Commitment<C::Scalar, PolynomialPointer<'a, C>>,
    low_degree_equivalent: Polynomial<C::Scalar, Coeff>,
}

impl<'a, C: CurveAffine> Commitment<C::Scalar, PolynomialPointer<'a, C>> {
    fn extend(&self, points: &[C::Scalar]) -> CommitmentExtension<'a, C> {
        let poly = lagrange_interpolate(points, &self.evals()[..]);

        let low_degree_equivalent = Polynomial {
            values: poly,
            _marker: PhantomData,
        };

        CommitmentExtension {
            commitment: self.clone(),
            low_degree_equivalent,
        }
    }
}

impl<'a, C: CurveAffine> CommitmentExtension<'a, C> {
    fn linearisation_contribution(&self, u: C::Scalar) -> Polynomial<C::Scalar, Coeff> {
        let p_x = self.commitment.get().poly;
        let r_eval = eval_polynomial(&self.low_degree_equivalent.values[..], u);
        p_x - r_eval
    }

    fn quotient_contribution(&self) -> Polynomial<C::Scalar, Coeff> {
        let len = self.low_degree_equivalent.len();
        let mut p_x = self.commitment.get().poly.clone();
        parallelize(&mut p_x.values[0..len], |lhs, start| {
            for (lhs, rhs) in lhs
                .iter_mut()
                .zip(self.low_degree_equivalent.values[start..].iter())
            {
                *lhs -= *rhs;
            }
        });
        p_x
    }
}

struct RotationSetExtension<'a, C: CurveAffine> {
    commitments: Vec<CommitmentExtension<'a, C>>,
    points: Vec<C::Scalar>,
}

impl<'a, C: CurveAffine> RotationSet<C::Scalar, PolynomialPointer<'a, C>> {
    fn extend(self, commitments: Vec<CommitmentExtension<'a, C>>) -> RotationSetExtension<'a, C> {
        RotationSetExtension {
            commitments,
            points: self.points,
        }
    }
}

/// Concrete KZG prover with SHPLONK variant
#[derive(Debug)]
pub struct ProverSHPLONK<'a, E: Engine> {
    params: &'a ParamsKZG<E>,
}

impl<'a, E: Engine> ProverSHPLONK<'a, E> {
    /// Given parameters creates new prover instance
    pub fn new(params: &'a ParamsKZG<E>) -> Self {
        Self { params }
    }
}

/// Create a multi-opening proof
impl<'params, E: Engine + Debug> Prover<'params, KZGCommitmentScheme<E>>
    for ProverSHPLONK<'params, E>
where
    E::Fr: Ord + Hash,
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
    const QUERY_INSTANCE: bool = false;

    fn new(params: &'params ParamsKZG<E>) -> Self {
        Self { params }
    }

    /// Create a multi-opening proof
    fn create_proof<
        'com,
        Ch: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, Ch>,
        R,
        I,
    >(
        &self,
        _: R,
        transcript: &mut T,
        queries: I,
    ) -> io::Result<()>
    where
        I: IntoIterator<Item = ProverQuery<'com, E::G1Affine>> + Clone,
        R: RngCore,
    {
        // TODO: explore if it is safe to use same challenge
        // for different sets that are already combined with another challenge
        let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

        let quotient_contribution = |i: usize, rotation_set: &RotationSetExtension<E::G1Affine>| {
            // [P_i_0(X) - R_i_0(X), P_i_1(X) - R_i_1(X), ... ]

            // #[allow(clippy::needless_collect)]
            // let numerators = rotation_set
            //     .commitments
            //     .as_slice()
            //     .into_par_iter()
            //     .map(|commitment| commitment.quotient_contribution())
            //     .collect::<Vec<_>>();

            // define numerator polynomial as
            // N_i_j(X) = (P_i_j(X) - R_i_j(X))
            // and combine polynomials with same evaluation point set
            // N_i(X) = linear_combination(y, N_i_j(X))
            // where y is random scalar to combine numerator polynomials

            let poly_num = rotation_set.commitments.len();
            let ys  = powers_of_x(*y, poly_num);

            let start = instant::Instant::now();
            let mut comb_poly = rotation_set
                                      .commitments[0].commitment.get().poly.clone();
            let mut_comb_poly = &mut comb_poly;

            let polys: Vec<_> = rotation_set
                                      .commitments[1..poly_num]
                                      .iter()
                                      .map(|cmmt| cmmt.commitment.get().poly)
                                      .collect();
            mut_comb_poly.add_mul_scalars(&polys, &ys[1..poly_num]);
            let cmmt_poly_comb_t = start.elapsed();

            let start = instant::Instant::now();
            let mut n_x = comb_poly.clone();
            let mut_nx  = &mut n_x;

            for i in 0..poly_num {
                let cmmt = &rotation_set.commitments[i];
                let poly = &cmmt.low_degree_equivalent;
                mut_nx.sub_low_poly_mul_scalar(poly, ys[i]);
            };
            let nx_sub_low_poly_t = start.elapsed();

            let points = &rotation_set.points[..];

            // quotient contribution of this evaluation set is
            // Q_i(X) = N_i(X) / Z_i(X) where
            // Z_i(X) = (x - r_i_0) * (x - r_i_1) * ...

            let start = instant::Instant::now();
            let mut poly = match points.len() {
                2 => kate_division_deg2(&n_x.values, points),
                3 => kate_division_deg3(&n_x.values, points),
                _ => div_by_vanishing(n_x, points),
            };

            poly.resize(self.params.n as usize, E::Fr::ZERO);
            let nx_div_vanish_t = start.elapsed();

            let quot_poly = Polynomial {
                values: poly,
                _marker: PhantomData, };

            ((i,comb_poly), quot_poly)

        };

        let intermediate_sets = construct_intermediate_sets(queries);
        let (rotation_sets, super_point_set) = (
            intermediate_sets.rotation_sets,
            intermediate_sets.super_point_set,
        );

        let rotation_sets: Vec<RotationSetExtension<E::G1Affine>> = rotation_sets
            .into_par_iter()
            .map(|rotation_set| {
                let commitments: Vec<CommitmentExtension<E::G1Affine>> = rotation_set
                    .commitments
                    .as_slice()
                    .into_par_iter()
                    .map(|commitment_data| commitment_data.extend(&rotation_set.points))
                    .collect();
                rotation_set.extend(commitments)
            })
            .collect();

        let v: ChallengeV<_> = transcript.squeeze_challenge_scalar();

        let start = instant::Instant::now();

        #[allow(clippy::needless_collect)]
        let (comb_polys, quotient_polynomials): (
            Vec<(usize, Polynomial<E::Fr, Coeff>)>,
            Vec<Polynomial<E::Fr, Coeff>>,
        ) = rotation_sets
                    .as_slice()
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, rotation_set)| quotient_contribution(i, rotation_set))
                    .unzip();
        let quot_polys_t = start.elapsed();

        let start = instant::Instant::now();
        let vs  = powers_of_x(*v, quotient_polynomials.len());

        let mut h_x = quotient_polynomials[0].clone();
        let polynomials_refs: Vec<&Polynomial<E::Fr, Coeff>> = quotient_polynomials[1..].iter().collect();
        &h_x.add_mul_scalars(&polynomials_refs, &vs[1..quotient_polynomials.len()]);

        let quot_polys_comb_t = start.elapsed();


        let h = self.params.commit(&h_x, Blind::default()).to_affine();
        transcript.write_point(h)?;
        let u: ChallengeU<_> = transcript.squeeze_challenge_scalar();

        let linearisation_contribution = |i: usize, rotation_set: RotationSetExtension<E::G1Affine>| {
            let mut diffs = super_point_set.clone();
            for point in rotation_set.points.iter() {
                diffs.remove(point);
            }
            let diffs = diffs.into_iter().collect::<Vec<_>>();

            // calculate difference vanishing polynomial evaluation
            let z_i = evaluate_vanishing_polynomial(&diffs[..], *u);

            // inner linearisation contributions are
            // [P_i_0(X) - r_i_0, P_i_1(X) - r_i_1, ... ] where
            // r_i_j = R_i_j(u) is the evaluation of low degree equivalent polynomial
            // where u is random evaluation point

            let low_poly_vals = rotation_set.commitments
                                            .as_slice()
                                            .into_iter()
                                            .map(|cmmt| eval_polynomial(&cmmt.low_degree_equivalent.values[..], *u))
                                            .collect::<Vec<_>>();

            let num = rotation_set.commitments.len();
            let ys  = powers_of_x(*y, num);

            let mut l_x  = comb_polys.iter()
                                     .find(|(j, _)| *j == i)
                                     .map(|(_, poly)| poly)
                                     .expect("none sense")
                                     .clone();
            let mut_lx = &mut l_x;

            for i in 0..num {
                let poly  = Polynomial {
                                values: [low_poly_vals[i]].to_vec(),
                                _marker: PhantomData,
                            };
                mut_lx.sub_low_poly_mul_scalar(&poly, ys[i]);
            };

            // define inner contributor polynomial as
            // L_i_j(X) = (P_i_j(X) - r_i_j)
            // and combine polynomials with same evaluation point set
            // L_i(X) = linear_combination(y, L_i_j(X))
            // where y is random scalar to combine inner contributors

            // finally scale l_x by difference vanishing polynomial evaluation z_i
            (l_x * z_i, z_i)
        };


        let start = instant::Instant::now();

        #[allow(clippy::type_complexity)]
        let (linearisation_contributions, z_diffs): (
            Vec<Polynomial<E::Fr, Coeff>>,
            Vec<E::Fr>,
        ) = rotation_sets
            .into_par_iter()
            .enumerate()
            .map(|(i, rotation_set)| linearisation_contribution(i, rotation_set))
            .unzip();
        let lin_contris_t = start.elapsed();

        let start = instant::Instant::now();
        let mut l_x = linearisation_contributions[0].clone();// * vs[linearisation_contributions.len()];
        let polynomials_refs: Vec<&Polynomial<E::Fr, Coeff>> = linearisation_contributions[1..].iter().collect();

        &l_x.add_mul_scalars(&polynomials_refs, &vs[1..]);
        let lin_contri_comb_t = start.elapsed();


        let super_point_set = super_point_set.into_iter().collect::<Vec<_>>();
        let zt_eval = evaluate_vanishing_polynomial(&super_point_set[..], *u);
        let l_x = l_x - &(h_x * zt_eval);

        // sanity check
        #[cfg(debug_assertions)]
        {
            let must_be_zero = eval_polynomial(&l_x.values[..], *u);
            assert_eq!(must_be_zero, E::Fr::ZERO);
        }

        let mut h_x = div_by_vanishing_par(l_x, &[*u]);

        // normalize coefficients by the coefficient of the first polynomial
        let z_0_diff_inv = z_diffs[0].invert().unwrap();

        vector_mul_scalar_inplace(&mut h_x, z_0_diff_inv);

        let h_x = Polynomial {
            values: h_x,
            _marker: PhantomData,
        };

        let h = self.params.commit(&h_x, Blind::default()).to_affine();
        transcript.write_point(h)?;

        Ok(())
    }
}
