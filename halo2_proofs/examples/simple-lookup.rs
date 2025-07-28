extern crate criterion;

use ff::PrimeField;
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::*;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::kzg::multiopen::VerifierGWC;
use halo2_proofs::poly::{commitment::ParamsProver, Rotation};
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use icicle_bn254::curve::ScalarField;
use icicle_core::ntt::release_domain;
use icicle_runtime::{stream::IcicleStream, warmup};
use rand_core::OsRng;

use halo2_proofs::{
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::ProverGWC,
        strategy::SingleStrategy,
    },
    transcript::{TranscriptReadBuffer, TranscriptWriterBuffer},
    icicle::try_load_and_set_backend_device
};

use std::marker::PhantomData;
use std::time::{Instant};

#[derive(Clone, Default)]
struct MyCircuit<F: PrimeField> {
    _marker: PhantomData<F>,
    k: u32,
}

#[derive(Clone)]
struct MyConfig {
    qlookup: Selector,
    identity: Selector,
    table_input: TableColumn,
    table_output: TableColumn,
    advice: Column<Advice>,
    other_advice: Column<Advice>,
    instance: Column<Instance>,
}

impl<F: PrimeField> Circuit<F> for MyCircuit<F> {
    type Config = MyConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> MyConfig {
        let config = MyConfig {
            qlookup: meta.complex_selector(),
            identity: meta.complex_selector(),
            table_input: meta.lookup_table_column(),
            table_output: meta.lookup_table_column(),
            advice: meta.advice_column(),
            other_advice: meta.advice_column(),
            instance: meta.instance_column(),
        };

        meta.enable_equality(config.advice);
        meta.enable_equality(config.instance);
        meta.enable_equality(config.other_advice);

        meta.lookup("", |cs| {
            let qlookup = cs.query_selector(config.qlookup);
            let not_qlookup = Expression::Constant(F::ONE) - qlookup.clone();
            let (default_x, default_y): (F, F) = (F::from(0), F::from(0));
            vec![
                (
                    qlookup.clone() * cs.query_advice(config.advice, Rotation(0))
                        + not_qlookup.clone() * default_x,
                    config.table_input,
                ),
                (
                    qlookup * cs.query_advice(config.other_advice, Rotation(0))
                        + not_qlookup * default_y,
                    config.table_output,
                ),
            ]
        });

        // This is kind of dumb but it helps illustrate the issue
        meta.create_gate("identity", |meta| {
            // To implement identity, we need 2 advice cells and a selector
            // cell. We arrange them like so:
            //
            // | a1  | a2  | s_mul |
            // |-----|-----|-------|
            // | lhs | rhs | s_iden |
            //
            let lhs = meta.query_advice(config.advice, Rotation::cur());
            let rhs = meta.query_advice(config.other_advice, Rotation::cur());
            let s_mul = meta.query_selector(config.identity);

            let diff = lhs - rhs;

            let constraint = diff.clone() * (Expression::Constant(F::ZERO) - diff);

            Constraints::with_selector(s_mul, vec![constraint])
        });

        config
    }

    fn synthesize(
        &self,
        config: MyConfig,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "8-bit 2x table",
            |mut table| {
                for row in 0u64..2_u64.pow(self.k - 1) {
                    table.assign_cell(
                        || format!("input row {row}"),
                        config.table_input,
                        row as usize,
                        || Value::known(F::from(row)),
                    )?;
                    // table output (2x the input) -- yeehaw
                    table.assign_cell(
                        || format!("output row {row}"),
                        config.table_output,
                        row as usize,
                        || Value::known(F::from(2 * row)),
                    )?;
                }

                Ok(())
            },
        )?;

        layouter.assign_region(
            || "assign values",
            |mut region| {
                for offset in 0u64..2_u64.pow(self.k - 2) {
                    // enable the 2x lookup table selector
                    config.qlookup.enable(&mut region, offset as usize)?;
                    // input
                    region.assign_advice(
                        || format!("offset {offset}"),
                        config.advice,
                        offset as usize,
                        || Value::known(F::from(offset)),
                    )?;
                    // 2x
                    let cell = region.assign_advice(
                        || format!("offset {offset}"),
                        config.other_advice,
                        offset as usize,
                        || Value::known(F::from(2 * offset)),
                    )?;

                    let output_offset = offset + 2_u64.pow(self.k - 2);
                    // copy it further down
                    cell.copy_advice(
                        || "",
                        &mut region,
                        config.advice,
                        output_offset as usize,
                    )?;
                    // now make it public
                    region.assign_advice_from_instance(
                        || "pub input anchor",
                        config.instance,
                        offset as usize,
                        config.other_advice,
                        output_offset as usize,
                    )?;
                    config
                        .identity
                        .enable(&mut region, output_offset as usize)?;
                }

                Ok(())
            },
        )
    }
}

fn keygen(k: u32) -> (ParamsKZG<Bn256>, ProvingKey<G1Affine>) {
    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
    let empty_circuit: MyCircuit<Fr> = MyCircuit {
        _marker: PhantomData,
        k
    };
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");
    (params, pk)
}

fn prover(k: u32, params: &ParamsKZG<Bn256>, pk: &ProvingKey<G1Affine>) -> Vec<u8> {
    let rng = OsRng;

    let circuit: MyCircuit<Fr> = MyCircuit {
        _marker: PhantomData,
        k,
    };

    let mut transcript = Blake2bWrite::<_, _, Challenge255<G1Affine>>::init(vec![]);

    // 2x the first 2^k-1 rows
    let instances = (0..2_u64.pow(k - 1))
        .map(|row| Fr::from(2 * row))
        .collect::<Vec<_>>();

    create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<'_, Bn256>, _, _, _, _>(
        params,
        pk,
        &[circuit],
        &[&[instances.as_slice()]],
        rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    transcript.finalize()
}

fn verifier(k: u32, params: &ParamsKZG<Bn256>, vk: &VerifyingKey<G1Affine>, proof: &[u8]) {
    let strategy = SingleStrategy::new(params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<G1Affine>>::init(proof);

    let instances = (0..2_u64.pow(k - 1))
        .map(|row| Fr::from(2 * row))
        .collect::<Vec<_>>();

    assert!(verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierGWC<'_, Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<'_, Bn256>,
    >(
        params,
        vk,
        strategy,
        &[&[instances.as_slice()]],
        &mut transcript,
        params.n()
    )
    .is_ok());
}

fn main() {
    env_logger::init();

    try_load_and_set_backend_device("CUDA");
    warmup(&IcicleStream::default()).unwrap();
    
    for k in 10..20 {
        release_domain::<ScalarField>().unwrap();

        let (params, pk) = keygen(k);

        let start = Instant::now();
        let proof = prover(k, &params, &pk);
        let end = Instant::now();
        let prove_time = end - start;
        let vk = pk.get_vk();
        verifier(k, &params, &vk, &proof);
        println!("proving took: {:?} for k = {}", prove_time, k);
    }
}