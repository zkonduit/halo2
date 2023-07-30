use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::Curve;
use halo2curves::CurveExt;
use rand_core::RngCore;
use std::collections::BTreeSet;
use std::env::var;
use std::ops::RangeTo;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;
use std::{collections::HashMap, iter, mem, sync::atomic::Ordering};

use procinfo::pid::{stat_self, statm_self};

use super::{
    circuit::{
        sealed::{self, SealedPhase},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, FirstPhase, Fixed,
        FloorPlanner, Instance, Selector,
    },
    lookup, permutation, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, Error, Expression, ProvingKey,
};
use crate::circuit::layouter::SyncDeps;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    circuit::Value,
    plonk::Assigned,
    poly::{
        self,
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurveAffine;

/// get the mem usage of process
pub fn getmem(mark: &str) {
    println!(
        "{}: total {:.3} resident {:.3}",
        mark,
        ((statm_self().unwrap().size * 4096) as f64) / 1073741824f64,
        ((statm_self().unwrap().resident * 4096) as f64) / 1073741824f64
    );
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    'params,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[Scheme::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
{
    getmem("start");
    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    }

    let instance: Vec<InstanceSingle<Scheme::Curve>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<Scheme::Curve>, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = domain.empty_lagrange();
                    assert_eq!(poly.len(), params.n() as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    for (poly, value) in poly.iter_mut().zip(values.iter()) {
                        if !P::QUERY_INSTANCE {
                            transcript.common_scalar(*value)?;
                        }
                        *poly = *value;
                    }
                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            if P::QUERY_INSTANCE {
                let instance_commitments_projective: Vec<_> = instance_values
                    .iter()
                    .map(|poly| params.commit_lagrange(poly, Blind::default()))
                    .collect();
                let mut instance_commitments =
                    vec![Scheme::Curve::identity(); instance_commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &instance_commitments_projective,
                    &mut instance_commitments,
                );
                let instance_commitments = instance_commitments;
                drop(instance_commitments_projective);

                for commitment in &instance_commitments {
                    transcript.common_point(*commitment)?;
                }
            }

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    getmem("2");

    #[derive(Clone)]
    struct AdviceSingle<C: CurveAffine, B: Basis> {
        pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
        pub advice_blinds: Vec<Blind<C::Scalar>>,
    }

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: Field> SyncDeps for WitnessCollection<'a, F> {}

    impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase != column.column_type().phase {
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            *self
                .advice
                .get_mut(column.index())
                .and_then(|v| v.get_mut(row))
                .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    getmem("4");

    let (advice, challenges) = {
        let mut advice = vec![
            AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                advice_polys: Vec::new(), //vec![domain.empty_lagrange(); meta.num_advice_columns],
                advice_blinds: Vec::new(), //vec![Blind::default(); meta.num_advice_columns],
            };
            instances.len()
        ];
        let mut challenges = HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);

        getmem("5");
        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
        for current_phase in pk.vk.cs.phases() {
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            getmem("6");

            for ((circuit, advice), instances) in
                circuits.iter().zip(advice.iter_mut()).zip(instances)
            {
                let mut witness = WitnessCollection {
                    k: params.k(),
                    current_phase,
                    advice: vec![domain.empty_lagrange_assigned(); meta.num_advice_columns],
                    instances,
                    challenges: &challenges,
                    // The prover will not be allowed to assign values to advice
                    // cells that exist within inactive rows, which include some
                    // number of blinding factors and an extra row for use in the
                    // permutation argument.
                    usable_rows: ..unusable_rows_start,
                    _marker: std::marker::PhantomData,
                };

                // Synthesize the circuit to obtain the witness and other information.
                ConcreteCircuit::FloorPlanner::synthesize(
                    &mut witness,
                    circuit,
                    config.clone(),
                    meta.constants.clone(),
                )?;

                getmem("7");

                for (column_index, advice_value) in witness
                    .advice
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            Some(advice)
                        } else {
                            None
                        }
                    })
                    .enumerate()
                {
                    getmem("inside col loop");
                    // batch invert the advice values
                    let advice_value = &mut batch_invert_assigned::<Scheme::Scalar>(
                        vec![advice_value].into_iter().collect(),
                    )[0];

                    // Pad advice values with blinded values
                    for cell in &mut advice_value[unusable_rows_start..] {
                        *cell = Scheme::Scalar::random(&mut rng);
                    }

                    advice
                        .advice_blinds
                        .push(Blind(Scheme::Scalar::random(&mut rng)));
                    // let advice_blind = &mut advice.advice_blinds[column_index];
                    // *advice_blind = Blind(Scheme::Scalar::random(&mut rng));

                    // Compute advice column commitment
                    let advice_commitment_projective =
                        params.commit_lagrange(advice_value, advice.advice_blinds[column_index]);
                    let advice_commitment = advice_commitment_projective.to_affine();
                    transcript.write_point(advice_commitment)?;

                    let advice_value: &_ = advice_value;

                    advice.advice_polys.push(advice_value.to_owned());
                    // let advice = &mut advice.advice_polys[column_index];
                    // *advice = advice_value.clone();
                }
            }
            getmem("8");

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        getmem("before challenges");
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    getmem("before lookups");

    let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    getmem("after lookups");

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    getmem("before permutation commit");

    // Commit to permutations.
    let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    getmem("after perm, before lookups");
    let lookups: Vec<Vec<lookup::prover::Committed<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    getmem("before vanishing");
    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;

    getmem("after vanishing");
    // Obtain challenge for keeping all separate gates linearly independent
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    getmem("calc advice polys");
    // Calculate the advice polys
    let advice: Vec<AdviceSingle<Scheme::Curve, Coeff>> = advice
        .into_iter()
        .map(
            |AdviceSingle {
                 advice_polys,
                 advice_blinds,
             }| {
                AdviceSingle {
                    advice_polys: advice_polys
                        .into_iter()
                        .map(|poly| domain.lagrange_to_coeff(poly))
                        .collect::<Vec<_>>(),
                    advice_blinds,
                }
            },
        )
        .collect();

    getmem("h poly");
    // Evaluate the h(X) polynomial
    let h_poly = pk.ev.evaluate_h(
        pk,
        &advice
            .iter()
            .map(|a| a.advice_polys.as_slice())
            .collect::<Vec<_>>(),
        &instance
            .iter()
            .map(|i| i.instance_polys.as_slice())
            .collect::<Vec<_>>(),
        &challenges,
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &permutations,
    );

    getmem("vanishing");
    // Construct the vanishing argument's h(X) commitments
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow(&[params.n() as u64, 0, 0, 0]);

    if P::QUERY_INSTANCE {
        // Compute and hash instance evals for each circuit instance
        for instance in instance.iter() {
            // Evaluate polynomials at omega^i x
            let instance_evals: Vec<_> = meta
                .instance_queries
                .iter()
                .map(|&(column, at)| {
                    eval_polynomial(
                        &instance.instance_polys[column.index()],
                        domain.rotate_omega(*x, at),
                    )
                })
                .collect();

            // Hash each instance column evaluation
            for eval in instance_evals.iter() {
                transcript.write_scalar(*eval)?;
            }
        }
    }

    getmem("advice evals");
    // Compute and hash advice evals for each circuit instance
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &advice.advice_polys[column.index()],
                    domain.rotate_omega(*x, at),
                )
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    }

    getmem("fixed  evals");
    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
        })
        .collect();

    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }

    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the lookups, if any, at omega^i x.
    let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .flat_map(|(((instance, advice), permutation), lookups)| {
            iter::empty()
                .chain(
                    P::QUERY_INSTANCE
                        .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| {
                            ProverQuery {
                                point: domain.rotate_omega(*x, at),
                                poly: &instance.instance_polys[column.index()],
                                blind: Blind::default(),
                            }
                        }))
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            poly: &advice.advice_polys[column.index()],
                            blind: advice.advice_blinds[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)).into_iter())
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    poly: &pk.fixed_polys[column.index()],
                    blind: Blind::default(),
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));

    let prover = P::new(params);
    prover
        .create_proof(rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure)
}
