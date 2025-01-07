use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeTheta, ChallengeX, Error, ProvingKey, VerifyingKey,
};
use super::Argument;
use crate::helpers::SerdeCurveAffine;
use crate::plonk::evaluation::evaluate;
use crate::SerdeFormat;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    poly::{
        commitment::{Blind, Params},
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::WithSmallOrderMulGroup;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rustc_hash::FxHashMap as HashMap;

use maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    iter,
    ops::{Mul, MulAssign},
};

#[derive(Debug)]
pub(in crate::plonk) struct Prepared<C: CurveAffine> {
    compressed_inputs_expressions: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    m_values: Polynomial<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) commitment: C,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) m_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) phi_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) commitment: C,
}

impl<C: SerdeCurveAffine> Committed<C> {
    #[allow(dead_code)]
    pub fn write<W: std::io::Write>(
        &self,
        writer: &mut W,
        format: SerdeFormat,
    ) -> std::io::Result<()>
    where
        <C as CurveAffine>::ScalarExt: crate::helpers::SerdePrimeField,
    {
        self.m_poly.write(writer, format)?;
        self.phi_poly.write(writer, format)?;
        self.commitment.write(writer, format)
    }

    #[allow(dead_code)]
    pub fn read<R: std::io::Read>(reader: &mut R, format: SerdeFormat) -> std::io::Result<Self>
    where
        <C as CurveAffine>::ScalarExt: crate::helpers::SerdePrimeField,
    {
        let m_poly = Polynomial::read(reader, format)?;
        let phi_poly = Polynomial::read(reader, format)?;
        let commitment = C::read(reader, format)?;

        Ok(Committed {
            m_poly,
            phi_poly,
            commitment,
        })
    }
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

impl<F: WithSmallOrderMulGroup<3>> Argument<F> {
    pub(in crate::plonk) fn prepare<'a, 'params: 'a, C, P: Params<'params, C>>(
        &self,
        vk: &VerifyingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
    ) -> Result<Prepared<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        let n = params.n() as usize;
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        n,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        let start = instant::Instant::now();
        // Get values of input expressions involved in the lookup and compress them
        let compressed_inputs_expressions: Vec<_> = self
            .inputs_expressions
            .par_iter()
            .map(|input_expressions| compress_expressions(input_expressions))
            .collect();
        log::trace!("compressed_inputs_expressions {:?}", start.elapsed());

        // Get values of table expressions involved in the lookup and compress them
        let start = instant::Instant::now();
        let compressed_table_expression = compress_expressions(&self.table_expressions);
        log::trace!("compressed_table_expression {:?}", start.elapsed());

        let blinding_factors = vk.cs.blinding_factors();

        let chunk_size = n - blinding_factors - 1;

        // compute m(X)
        let start = instant::Instant::now();
        let table_index_value_mapping: HashMap<Vec<u8>, usize> = compressed_table_expression
            .par_iter()
            .take(chunk_size)
            .enumerate()
            .map(|(i, &x)| (x.to_repr().as_ref().to_owned(), i))
            .collect();
        log::trace!("table_index_value_mapping {:?}", start.elapsed());

        let start = instant::Instant::now();
        let m_values: Vec<F> = {
            use std::sync::atomic::{AtomicU64, Ordering};
            let m_values: Vec<AtomicU64> = (0..params.n()).map(|_| AtomicU64::new(0)).collect();

            compressed_inputs_expressions
                .par_iter()
                .for_each(|compressed_input_expression| {
                    compressed_input_expression
                        .iter()
                        .take(chunk_size)
                        .for_each(|fi| {
                            let index = match table_index_value_mapping
                                .get(&fi.to_repr().as_ref().to_owned())
                            {
                                Some(value) => value,
                                None => {
                                    log::error!("value is OOR of lookup");
                                    return;
                                }
                            };
                            m_values[*index].fetch_add(1, Ordering::Relaxed);
                        });
                });

            m_values
                .par_iter()
                .map(|mi| F::from(mi.load(Ordering::Relaxed)))
                .collect()
        };
        log::trace!("m_values {:?}", start.elapsed());
        let m_values = vk.domain.lagrange_from_vec(m_values);

        #[cfg(feature = "sanity-checks")]
        {
            // check that m is zero after blinders
            let invalid_ms = m_values
                .iter()
                .skip(params.n() as usize - blinding_factors)
                .collect::<Vec<_>>();
            assert_eq!(invalid_ms.len(), blinding_factors);
            for mi in invalid_ms {
                assert_eq!(*mi, C::Scalar::ZERO);
            }

            // check sums
            let alpha = C::Scalar::random(&mut rng);
            let cs_input_sum =
                |compressed_input_expression: &Polynomial<C::Scalar, LagrangeCoeff>| {
                    let mut lhs_sum = C::Scalar::ZERO;
                    for &fi in compressed_input_expression
                        .iter()
                        .take(params.n() as usize - blinding_factors - 1)
                    {
                        lhs_sum += (fi + alpha).invert().unwrap();
                    }

                    lhs_sum
                };

            let mut lhs_sum = C::Scalar::ZERO;

            for compressed_input_expression in compressed_inputs_expressions.iter() {
                lhs_sum += cs_input_sum(compressed_input_expression);
            }

            let mut rhs_sum = C::Scalar::ZERO;
            for (&ti, &mi) in compressed_table_expression.iter().zip(m_values.iter()) {
                rhs_sum += mi * (ti + alpha).invert().unwrap();
            }

            assert_eq!(lhs_sum, rhs_sum);
        }

        // commit to m(X)
        let blind = Blind(C::Scalar::ZERO);
        let start = instant::Instant::now();
        let m_commitment = params.commit_lagrange(&m_values, blind.clone()).to_affine();
        log::trace!("m_commitment {:?}", start.elapsed());

        // write commitment of m(X) to transcript
        // transcript.write_point(m_commitment)?;

        Ok(Prepared {
            compressed_inputs_expressions,
            compressed_table_expression,
            m_values,
            commitment: m_commitment,
        })
    }
}

impl<C: CurveAffine> Prepared<C> {
    pub(in crate::plonk) fn commit_grand_sum<'params, P: Params<'params, C>>(
        self,
        vk: &VerifyingKey<C>,
        params: &P,
        beta: ChallengeBeta<C>,
        phi_blinds: &[C::Scalar],
    ) -> Result<Committed<C>, Error> {
        /*
            φ_i(X) = f_i(X) + α
            τ(X) = t(X) + α
            LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
            RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
        */

        let start = instant::Instant::now();
        // ∑ 1/(φ_i(X))
        let mut inputs_log_derivatives = vec![C::Scalar::ZERO; params.n() as usize];
        for compressed_input_expression in self.compressed_inputs_expressions.iter() {
            let mut input_log_derivatives = vec![C::Scalar::ZERO; params.n() as usize];

            parallelize(
                &mut input_log_derivatives,
                |input_log_derivatives, start| {
                    for (input_log_derivative, fi) in input_log_derivatives
                        .iter_mut()
                        .zip(compressed_input_expression[start..].iter())
                    {
                        *input_log_derivative = *beta + fi;
                    }
                },
            );
            input_log_derivatives.iter_mut().batch_invert();

            // TODO: remove last blinders from this
            for i in 0..params.n() as usize {
                inputs_log_derivatives[i] += input_log_derivatives[i];
            }
        }

        log::trace!(" - inputs_log_derivatives {:?}", start.elapsed());

        let start = instant::Instant::now();
        // 1 / τ(X)
        let mut table_log_derivatives = vec![C::Scalar::ZERO; params.n() as usize];
        parallelize(
            &mut table_log_derivatives,
            |table_log_derivatives, start| {
                for (table_log_derivative, ti) in table_log_derivatives
                    .iter_mut()
                    .zip(self.compressed_table_expression[start..].iter())
                {
                    *table_log_derivative = *beta + ti;
                }
            },
        );

        log::trace!(" - table_log_derivatives {:?}", start.elapsed());

        let start = instant::Instant::now();
        table_log_derivatives.iter_mut().batch_invert();
        log::trace!(
            " - table_log_derivatives batch_invert {:?}",
            start.elapsed()
        );

        let start = instant::Instant::now();
        // (Σ 1/(φ_i(X)) - m(X) / τ(X))
        let mut log_derivatives_diff = vec![C::Scalar::ZERO; params.n() as usize];
        parallelize(&mut log_derivatives_diff, |log_derivatives_diff, start| {
            for (((log_derivative_diff, fi), ti), mi) in log_derivatives_diff
                .iter_mut()
                .zip(inputs_log_derivatives[start..].iter())
                .zip(table_log_derivatives[start..].iter())
                .zip(self.m_values[start..].iter())
            {
                // (Σ 1/(φ_i(X)) - m(X) / τ(X))
                *log_derivative_diff = *fi - *mi * *ti;
            }
        });

        log::trace!(" - log_derivatives_diff {:?}", start.elapsed());

        let start = instant::Instant::now();
        // Compute the evaluations of the lookup grand sum polynomial
        // over our domain, starting with phi[0] = 0
        let blinding_factors = vk.cs.blinding_factors();

        assert!(
            phi_blinds.len() == blinding_factors,
            "invalid number of blinding factors"
        );

        let phi = iter::once(C::Scalar::ZERO)
            .chain(log_derivatives_diff)
            .scan(C::Scalar::ZERO, |state, cur| {
                *state += &cur;
                Some(*state)
            })
            // Take all rows including the "last" row which should
            // be a 0
            .take(params.n() as usize - blinding_factors)
            // Chain random blinding factors.
            .chain(phi_blinds.into_iter().map(|&x| x))
            .collect::<Vec<_>>();
        assert_eq!(phi.len(), params.n() as usize);
        let phi = vk.domain.lagrange_from_vec(phi);

        log::trace!(" - phi {:?}", start.elapsed());

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n() as usize) - (blinding_factors + 1);

            /*
                φ_i(X) = f_i(X) + α
                τ(X) = t(X) + α
                LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
            */

            // q(X) = LHS - RHS mod zH(X)
            for i in 0..u {
                // Π(φ_i(X))
                let fi_prod = || {
                    let mut prod = C::Scalar::ONE;
                    for compressed_input_expression in self.compressed_inputs_expressions.iter() {
                        prod *= *beta + compressed_input_expression[i];
                    }

                    prod
                };

                let fi_log_derivative = || {
                    let mut sum = C::Scalar::ZERO;
                    for compressed_input_expression in self.compressed_inputs_expressions.iter() {
                        sum += (*beta + compressed_input_expression[i]).invert().unwrap();
                    }

                    sum
                };

                // LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                let lhs = {
                    (*beta + self.compressed_table_expression[i])
                        * fi_prod()
                        * (phi[i + 1] - phi[i])
                };

                // RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                let rhs = {
                    (*beta + self.compressed_table_expression[i])
                        * fi_prod()
                        * (fi_log_derivative()
                            - self.m_values[i]
                                * (*beta + self.compressed_table_expression[i])
                                    .invert()
                                    .unwrap())
                };

                assert_eq!(lhs - rhs, C::Scalar::ZERO);
            }

            assert_eq!(phi[u], C::Scalar::ZERO);
        }

        let grand_sum_blind = Blind(C::Scalar::ZERO);
        let start = instant::Instant::now();
        let phi_commitment = params
            .commit_lagrange(&phi, grand_sum_blind.clone())
            .to_affine();
        log::trace!(" - phi_commitment {:?}", start.elapsed());

        // Hash grand sum commitment
        // transcript.write_point(phi_commitment)?;

        Ok(Committed {
            m_poly: vk.domain.lagrange_to_coeff(self.m_values),
            phi_poly: vk.domain.lagrange_to_coeff(phi),
            commitment: phi_commitment,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        vk: &VerifyingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &vk.domain;
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let phi_eval = eval_polynomial(&self.phi_poly, *x);
        let phi_next_eval = eval_polynomial(&self.phi_poly, x_next);
        let m_eval = eval_polynomial(&self.m_poly, *x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(phi_eval))
            .chain(Some(phi_next_eval))
            .chain(Some(m_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.phi_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.phi_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.m_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
    }
}
