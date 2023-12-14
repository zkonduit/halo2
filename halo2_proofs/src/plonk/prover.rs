use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use rand_core::RngCore;
use std::collections::BTreeSet;
use std::ops::RangeTo;
use std::{collections::HashMap, iter};

use super::{
    circuit::{
        sealed::{self},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    mv_lookup, permutation, shuffle, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta,
    ChallengeX, ChallengeY, Error, Expression, ProvingKey,
};

use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    circuit::Value,
    plonk::Assigned,
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurveAffine;

#[cfg(feature = "icicle_gpu")]
use crate::icicle;
use log::{debug, info};
use std::time::Instant;

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
    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;

        // reset counters at the beginning of the prove
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

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

    #[derive(Clone)]
    struct AdviceSingle<C: CurveAffine, B: Basis> {
        pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
        pub advice_blinds: Vec<Blind<C::Scalar>>,
    }

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
        unblinded_advice: std::collections::HashSet<usize>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

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

    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let (advice, challenges) = {
        let mut advice = vec![
            AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
                advice_blinds: vec![Blind::default(); meta.num_advice_columns],
            };
            instances.len()
        ];
        let mut challenges = HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);

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

            for ((circuit, advice), instances) in
                circuits.iter().zip(advice.iter_mut()).zip(instances)
            {
                let mut witness = WitnessCollection {
                    k: params.k(),
                    current_phase,
                    advice: vec![domain.empty_lagrange_assigned(); meta.num_advice_columns],
                    unblinded_advice: std::collections::HashSet::from_iter(
                        meta.unblinded_advice_columns.clone().into_iter(),
                    ),
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

                let mut advice_values = batch_invert_assigned::<Scheme::Scalar>(
                    witness
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
                        .collect(),
                );

                // Add blinding factors to advice columns
                for (column_index, advice_values) in &mut advice_values.iter_mut().enumerate() {
                    for cell in &mut advice_values[unusable_rows_start..] {
                        if witness.unblinded_advice.contains(&column_index) {
                            *cell = Blind::default().0;
                        } else {
                            *cell = Scheme::Scalar::random(&mut rng);
                        }
                    }
                }

                // Compute commitments to advice column polynomials
                let blinds: Vec<_> = (0..meta.num_advice_columns)
                    .map(|i| {
                        if witness.unblinded_advice.contains(&i) {
                            Blind::default()
                        } else {
                            Blind(Scheme::Scalar::random(&mut rng))
                        }
                    })
                    .collect();

                let now = std::time::Instant::now();
                #[cfg(feature = "icicle_gpu")]
                let mut advice_commitments_projective: Vec<_>;
                #[cfg(feature = "icicle_gpu")]
                if std::env::var("ENABLE_ICICLE_GPU").is_ok()
                    && icicle::is_small_circuit(advice_values[0].len())
                {
                    advice_commitments_projective =
                        params.commit_lagrange_batch(&advice_values, &blinds);
                    debug!(
                        "GPU: advice_commitments_projective of length {} took: {}",
                        advice_commitments_projective.len(),
                        now.elapsed().as_millis()
                    );
                } else {
                    advice_commitments_projective = advice_values
                        .iter()
                        .zip(blinds.iter())
                        .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                        .collect();
                }

                #[cfg(not(feature = "icicle_gpu"))]
                let advice_commitments_projective: Vec<_> = advice_values
                    .iter()
                    .zip(blinds.iter())
                    .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                    .collect();
                #[cfg(not(feature = "icicle_gpu"))]
                debug!(
                    "CPU: advice_commitments_projective of length {} took: {}",
                    advice_commitments_projective.len(),
                    now.elapsed().as_millis()
                );

                let mut advice_commitments =
                    vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &advice_commitments_projective,
                    &mut advice_commitments,
                );
                let advice_commitments = advice_commitments;
                drop(advice_commitments_projective);

                for commitment in &advice_commitments {
                    transcript.write_point(*commitment)?;
                }
                for ((column_index, advice_values), blind) in
                    column_indices.iter().zip(advice_values).zip(blinds)
                {
                    advice.advice_polys[*column_index] = advice_values;
                    advice.advice_blinds[*column_index] = blind;
                }
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };
    #[cfg(feature = "profile")]
    info!(
        "Advice and Challenge generation: {} ms",
        start.elapsed().as_millis()
    );

    // Sample theta challenge for keeping lookup columns linearly independent
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
    #[cfg(feature = "profile")]
    info!("theta generation: {} ms", start.elapsed().as_millis());

    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let lookups: Vec<Vec<mv_lookup::prover::Prepared<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.prepare(
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
    #[cfg(feature = "profile")]
    info!("Lookups prepare: {} ms", start.elapsed().as_millis());
    // Sample beta challenge
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    #[cfg(feature = "profile")]
    info!("beta generation: {} ms", start.elapsed().as_millis());

    // Sample gamma challenge
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();
    #[cfg(feature = "profile")]
    info!("gamma generation: {} ms", start.elapsed().as_millis());

    // Commit to permutations.
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
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
    #[cfg(feature = "profile")]
    info!("permutation commit: {} ms", start.elapsed().as_millis());

    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let lookups: Vec<Vec<mv_lookup::prover::Committed<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| lookup.commit_grand_sum(pk, params, beta, &mut rng, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "profile")]
    info!(
        "lookups commit_grand_sum: {} ms",
        start.elapsed().as_millis()
    );

    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let shuffles: Vec<Vec<shuffle::prover::Committed<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, _> {
            // Compress expressions for each shuffle
            pk.vk
                .cs
                .shuffles
                .iter()
                .map(|shuffle| {
                    shuffle.commit_product(
                        pk,
                        params,
                        domain,
                        theta,
                        gamma,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "profile")]
    info!("shuffle commit_product: {} ms", start.elapsed().as_millis());

    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;
    #[cfg(feature = "profile")]
    info!("vanishing commit: {} ms", start.elapsed().as_millis());

    // Obtain challenge for keeping all separate gates linearly independent
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();
    #[cfg(feature = "profile")]
    info!("y generation: {} ms", start.elapsed().as_millis());

    // Calculate the advice polys
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
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
    #[cfg(feature = "profile")]
    info!(
        "advice langrange_to_coeff: {} ms",
        start.elapsed().as_millis()
    );
    // Evaluate the h(X) polynomial
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
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
        &shuffles,
        &permutations,
    );
    #[cfg(feature = "profile")]
    info!("h_poly: {} ms", start.elapsed().as_millis());

    // Construct the vanishing argument's h(X) commitments
    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;
    #[cfg(feature = "profile")]
    info!("vanishing construction: {} ms", start.elapsed().as_millis());

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow([params.n()]);

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

    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
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
    let lookups: Vec<Vec<mv_lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the shuffles, if any, at omega^i x.
    let shuffles: Vec<Vec<shuffle::prover::Evaluated<Scheme::Curve>>> = shuffles
        .into_iter()
        .map(|shuffles| -> Result<Vec<_>, _> {
            shuffles
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
        .zip(shuffles.iter())
        .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
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
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                .chain(shuffles.iter().flat_map(move |p| p.open(pk, x)))
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

    #[cfg(feature = "profile")]
    info!("evaluations: {} ms", start.elapsed().as_millis());

    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;
        log::debug!("MSM_COUNTER: {:?}", MSM_COUNTER.lock().unwrap());
        log::debug!("FFT_COUNTER: {:?}", *FFT_COUNTER.lock().unwrap());

        // reset counters at the end of the proving
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    #[cfg(feature = "profile")]
    let start = std::time::Instant::now();
    let prover = P::new(params);
    let proof = prover
        .create_proof(rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure);
    #[cfg(feature = "profile")]
    info!("prover.create_proof : {} ms", start.elapsed().as_millis());
    proof
}

#[test]
fn test_create_proof() {
    use crate::{
        circuit::SimpleFloorPlanner,
        plonk::{keygen_pk, keygen_vk},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::Bn256;
    use rand_core::OsRng;

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, OsRng);
    let vk = keygen_vk(&params, &MyCircuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &MyCircuit).expect("keygen_pk should not fail");
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // Create proof with wrong number of instances
    let proof = create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[],
        OsRng,
        &mut transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[&[], &[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
}
