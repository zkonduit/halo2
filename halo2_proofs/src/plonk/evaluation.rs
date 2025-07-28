use crate::icicle::{c_scalars_from_device_vec, create_calculation_data, create_gate_data, device_vec_from_c_scalars, inplace_invert, inplace_mul};
use crate::plonk::{permutation, Any, ProvingKey};

#[cfg(feature = "mv-lookup")]
use crate::plonk::mv_lookup as lookup;

#[cfg(not(feature = "mv-lookup"))]
use crate::plonk::lookup;

use crate::poly::Basis;
use crate::{
    arithmetic::{parallelize, CurveAffine},
    poly::{Coeff, ExtendedLagrangeCoeff, Polynomial, Rotation}
};

use group::ff::{Field, PrimeField, WithSmallOrderMulGroup};
use icicle_bn254::curve::ScalarField;
use icicle_core::{
    traits::FieldImpl,
    gate_ops::{GateData, LookupConfig, CalculationData, HornerData, GateOpsConfig, LookupData, gate_evaluation, lookups_constraint},
    vec_ops::{accumulate_scalars, VecOpsConfig}
};
use icicle_runtime::{
    memory::{DeviceVec, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};
use maybe_rayon::iter::IntoParallelRefIterator;
use maybe_rayon::iter::ParallelIterator;
use maybe_rayon::join;

use super::{shuffle, ConstraintSystem, Expression};

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[allow(clippy::too_many_arguments)]
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Store(v) => get_value(v),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<C>,
    ///  Lookups evalution
    #[cfg(feature = "mv-lookup")]
    pub lookups: Vec<(Vec<GraphEvaluator<C>>, GraphEvaluator<C>)>,
    #[cfg(not(feature = "mv-lookup"))]
    pub lookups: Vec<GraphEvaluator<C>>,
    ///  Shuffle evalution
    pub shuffles: Vec<GraphEvaluator<C>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CalculationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| ev.custom_gates.add_expression(poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        #[cfg(feature = "mv-lookup")]
        for lookup in cs.lookups.iter() {
            let mut graph_table = GraphEvaluator::default();
            let mut graph_inputs: Vec<_> = (0..lookup.inputs_expressions.len())
                .map(|_| GraphEvaluator::default())
                .collect();

            let evaluate_lc = |graph: &mut GraphEvaluator<C>, expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Inputs cosets
            for (input_expressions, graph_input) in lookup
                .inputs_expressions
                .iter()
                .zip(graph_inputs.iter_mut())
            {
                let compressed_input_coset = evaluate_lc(graph_input, input_expressions);

                graph_input.add_calculation(Calculation::Add(
                    compressed_input_coset,
                    ValueSource::Beta(),
                ));
            }

            // table coset
            let compressed_table_coset = evaluate_lc(&mut graph_table, &lookup.table_expressions);

            graph_table.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Beta(),
            ));

            /*
                a) f_i + beta
                b) t + beta
            */
            ev.lookups.push((graph_inputs.to_vec(), graph_table));
        }

        #[cfg(not(feature = "mv-lookup"))]
        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        // Shuffles
        for shuffle in cs.shuffles.iter() {
            let evaluate_lc = |expressions: &Vec<Expression<_>>, graph: &mut GraphEvaluator<C>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            let mut graph_input = GraphEvaluator::default();
            let compressed_input_coset = evaluate_lc(&shuffle.input_expressions, &mut graph_input);
            let _ = graph_input.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Gamma(),
            ));

            let mut graph_shuffle = GraphEvaluator::default();
            let compressed_shuffle_coset =
                evaluate_lc(&shuffle.shuffle_expressions, &mut graph_shuffle);
            let _ = graph_shuffle.add_calculation(Calculation::Add(
                compressed_shuffle_coset,
                ValueSource::Gamma(),
            ));

            ev.shuffles.push(graph_input);
            ev.shuffles.push(graph_shuffle);
        }

        ev
    }

    /// Evaluate h poly
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        instance_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        challenges: &[C::ScalarExt],
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        shuffles: &[Vec<shuffle::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let fixed = &pk.fixed_cosets[..];
        let icicle_fixed = &pk.icicle_fixed[..];
        let extended_omega = domain.get_extended_omega();
        let isize = size as i32;
        let one = C::ScalarExt::ONE;
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let icicle_l0 = &pk.icicle_l0;
        let icicle_l_last = &pk.icicle_l_last;
        let icicle_l_active_row = &pk.icicle_l_active_row;
        let p = &pk.vk.cs.permutation;

        let mut values = domain.empty_extended();

        // Core expression evaluations
        for ((((advice, instance), lookups), shuffles), permutation) in advice_polys
            .iter()
            .zip(instance_polys.iter())
            .zip(lookups.iter())
            .zip(shuffles.iter())
            .zip(permutations.iter())
        {
            let (instance, advice) = join(
                || {
                    let instance: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>> = instance
                        .par_iter()
                        .map(|poly| {
                            let mut stream = IcicleStream::create().unwrap();
                            let result = domain.coeff_to_extended(poly, &stream);
                            stream.synchronize().unwrap();
                            stream.destroy().unwrap();

                            result
                        })
                        .collect();

                    instance
                },
                || {
                    let advice: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>> = advice
                        .par_iter()
                        .map(|poly| {
                            let mut stream = IcicleStream::create().unwrap();
                            let result = domain.coeff_to_extended(poly, &stream);
                            stream.synchronize().unwrap();
                            stream.destroy().unwrap();

                            result
                        })
                        .collect();

                    advice
                },
            );

            let (icicle_advice, icicle_instance, icicle_challenges, icicle_beta, icicle_gamma, icicle_theta, icicle_y) = create_gate_data::<C>(
                &advice[..],
                &instance[..],
                challenges,
                beta,
                gamma,
                theta,
                y,
            );

            let num_instance_rows = instance.len();
            let num_advice_rows = advice.len();
            let num_instance_cols = if num_instance_rows > 0 { instance[0].len() } else { 0 };
            let num_advice_cols = if num_advice_rows > 0 { advice[0].len() } else { 0 };

            let gate_data = GateData::new(
                unsafe { icicle_fixed.as_ptr() },
                fixed.len() as u32,
                fixed[0].len() as u32,
                unsafe { icicle_advice.as_ptr() },
                num_advice_rows as u32,
                num_advice_cols as u32,
                unsafe { icicle_instance.as_ptr() },
                num_instance_rows as u32,
                num_instance_cols as u32,
                icicle_challenges.as_ptr(),
                challenges.len() as u32,
                icicle_beta.as_ptr(),
                icicle_gamma.as_ptr(),
                icicle_theta.as_ptr(),
                icicle_y.as_ptr(),
            );

            // Custom gates
            {
                let (icicle_calculations, targets, value_types, value_indices, icicle_constants, icicle_rotations, size, num_intermediates, horner_value_types, horner_value_indices, horner_offsets, horner_sizes) = create_calculation_data::<C>(
                    &self.custom_gates.calculations,
                    &self.custom_gates.constants,
                    &self.custom_gates.rotations,
                    self.custom_gates.num_intermediates,
                );

                let calculation_data = CalculationData::new(
                    icicle_calculations.as_ptr(),
                    targets.as_ptr(),
                    value_types.as_ptr(),
                    value_indices.as_ptr(),
                    icicle_constants.as_ptr(),
                    icicle_constants.len() as u32,
                    icicle_rotations.as_ptr(),
                    icicle_rotations.len() as u32,
                    std::ptr::null(),
                    true,
                    size,
                    num_intermediates,
                    values.len() as u32,
                    rot_scale as u32,
                    isize as u32,
                );
    
                let horner_data = HornerData::new(
                    horner_value_types.as_ptr(),
                    horner_value_indices.as_ptr(),
                    horner_offsets.as_ptr(),
                    horner_sizes.as_ptr(),
                    horner_value_types.len() as u32
                );
    
                let mut d_result = DeviceVec::device_malloc_async(values.len(), &IcicleStream::default()).unwrap();
    
                let mut cfg = GateOpsConfig::default();
                cfg.is_fixed_on_device = true;
                cfg.is_advice_on_device = true;
                cfg.is_instance_on_device = true;
                cfg.is_previous_value_on_device = true;
                cfg.is_result_on_device = true;
                
                gate_evaluation(
                    &gate_data,
                    &calculation_data,
                    &horner_data,
                    &mut d_result,
                    &cfg,
                )
                .unwrap();
    
    
                let halo2_result: Vec<C::ScalarExt> = c_scalars_from_device_vec(&mut d_result, &IcicleStream::default());
    
                values =  Polynomial::from_vec(halo2_result);
            }

            // Permutations
            {
                let sets = &permutation.sets;
                if !sets.is_empty() {
                    let blinding_factors = pk.vk.cs.blinding_factors();
                    let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                    let chunk_len = pk.vk.cs.degree() - 2;
                    let delta_start = beta * &C::Scalar::ZETA;

                    let first_set = sets.first().unwrap();
                    let last_set = sets.last().unwrap();

                    // Permutation constraints
                    parallelize(&mut values, |values, start| {
                        let mut beta_term = extended_omega.pow_vartime([start as u64, 0, 0, 0]);
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;
                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                            let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                            // Enforce only for the first set.
                            // l_0(X) * (1 - z_0(X)) = 0
                            *value = *value * y
                                + ((one - first_set.permutation_product_coset[idx]) * l0[idx]);
                            // Enforce only for the last set.
                            // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                            *value = *value * y
                                + ((last_set.permutation_product_coset[idx]
                                    * last_set.permutation_product_coset[idx]
                                    - last_set.permutation_product_coset[idx])
                                    * l_last[idx]);
                            // Except for the first set, enforce.
                            // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                            for (set_idx, set) in sets.iter().enumerate() {
                                if set_idx != 0 {
                                    *value = *value * y
                                        + ((set.permutation_product_coset[idx]
                                            - permutation.sets[set_idx - 1].permutation_product_coset
                                                [r_last])
                                            * l0[idx]);
                                }
                            }
                            // And for all the sets we enforce:
                            // (1 - (l_last(X) + l_blind(X))) * (
                            //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                            // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                            // )
                            let mut current_delta = delta_start * beta_term;
                            for ((set, columns), cosets) in sets
                                .iter()
                                .zip(p.columns.chunks(chunk_len))
                                .zip(pk.permutation.cosets.chunks(chunk_len))
                            {
                                let mut left = set.permutation_product_coset[r_next];
                                for (values, permutation) in columns
                                    .iter()
                                    .map(|&column| match column.column_type() {
                                        Any::Advice(_) => &advice[column.index()],
                                        Any::Fixed => &fixed[column.index()],
                                        Any::Instance => &instance[column.index()],
                                    })
                                    .zip(cosets.iter())
                                {
                                    left *= values[idx] + beta * permutation[idx] + gamma;
                                }

                                let mut right = set.permutation_product_coset[idx];
                                for values in columns.iter().map(|&column| match column.column_type() {
                                    Any::Advice(_) => &advice[column.index()],
                                    Any::Fixed => &fixed[column.index()],
                                    Any::Instance => &instance[column.index()],
                                }) {
                                    right *= values[idx] + current_delta + gamma;
                                    current_delta *= &C::Scalar::DELTA;
                                }

                                *value = *value * y + ((left - right) * l_active_row[idx]);
                            }
                            beta_term *= &extended_omega;
                        }
                    });
                }
            }

            // Merged Lookups section
            #[cfg(feature = "mv-lookup")]
            {
                let y_vec = [y];
                let icicle_y = device_vec_from_c_scalars(&y_vec, &IcicleStream::default());
                let mut icicle_previous_value = device_vec_from_c_scalars(&values, &IcicleStream::default());

                let inputs_inv_sum_vec = vec![ScalarField::zero(); values.len()];
                let h_inputs_inv_sum = HostSlice::from_slice(&inputs_inv_sum_vec);

                for (n, lookup) in lookups.iter().enumerate() {

                    let ((m_coset, phi_coset), (inputs_inv_sums, (inputs_prods, table_values))) = join(
                        || {
                            let mut stream_m_poly = IcicleStream::create().unwrap();
                            let mut stream_phi_poly = IcicleStream::create().unwrap();

                            let m_coset = domain.coeff_to_extended_device_vec(&lookup.m_poly, &stream_m_poly);
                            let phi_coset = domain.coeff_to_extended_device_vec(&lookup.phi_poly, &stream_phi_poly);

                            stream_m_poly.synchronize().unwrap();
                            stream_phi_poly.synchronize().unwrap();
                            stream_m_poly.destroy().unwrap();
                            stream_phi_poly.destroy().unwrap();

                            (m_coset, phi_coset)
                        },
                        || join(
                            || {
                                let mut stream: IcicleStream = IcicleStream::create().unwrap();

                                let mut d_result = DeviceVec::device_malloc_async(domain.extended_len(), &stream).unwrap();

                                let gate_data = GateData::new(
                                    unsafe { icicle_fixed.as_ptr() },
                                    fixed.len() as u32,
                                    fixed[0].len() as u32,
                                    unsafe { icicle_advice.as_ptr() },
                                    num_advice_rows as u32,
                                    num_advice_cols as u32,
                                    unsafe { icicle_instance.as_ptr() },
                                    num_instance_rows as u32,
                                    num_instance_cols as u32,
                                    icicle_challenges.as_ptr(),
                                    challenges.len() as u32,
                                    icicle_beta.as_ptr(),
                                    icicle_gamma.as_ptr(),
                                    icicle_theta.as_ptr(),
                                    unsafe { icicle_y.as_ptr() },
                                );

                                let (inputs_lookup_evaluator, _) = &self.lookups[n];


                                let mut d_inputs_inv_sum = DeviceVec::<ScalarField>::device_malloc_async(values.len(), &stream).unwrap();
                                d_inputs_inv_sum.copy_from_host_async(h_inputs_inv_sum, &stream).unwrap();

                                // For each compressed input column, evaluate at ω^i and add beta
                                // This is a vector of length self.lookups[n].0.len()
                                inputs_lookup_evaluator.iter().for_each(
                                    |input_lookup_evaluator| {
                                        let (
                                            icicle_calculations,
                                            targets,
                                            value_types,
                                            value_indices,
                                            icicle_constants,
                                            icicle_rotations,
                                            size,
                                            num_intermediates,
                                            horner_value_types,
                                            horner_value_indices,
                                            horner_offsets,
                                            horner_sizes,
                                        ) = create_calculation_data::<C>(
                                            &input_lookup_evaluator.calculations,
                                            &input_lookup_evaluator.constants,
                                            &input_lookup_evaluator.rotations,
                                            input_lookup_evaluator.num_intermediates,
                                        );

                                        let calculation_data = CalculationData::new(
                                            icicle_calculations.as_ptr(),
                                            targets.as_ptr(),
                                            value_types.as_ptr(),
                                            value_indices.as_ptr(),
                                            icicle_constants.as_ptr(),
                                            icicle_constants.len() as u32,
                                            icicle_rotations.as_ptr(),
                                            icicle_rotations.len() as u32,
                                            std::ptr::null(),
                                            true,
                                            size,
                                            num_intermediates,
                                            domain.extended_len() as u32,
                                            rot_scale as u32,
                                            isize as u32,
                                        );
                                        
                                        let horner_data = HornerData::new(
                                            horner_value_types.as_ptr(),
                                            horner_value_indices.as_ptr(),
                                            horner_offsets.as_ptr(),
                                            horner_sizes.as_ptr(),
                                            horner_value_types.len() as u32,
                                        );
                                        
                                        let mut cfg = GateOpsConfig::default();
                                        cfg.is_async = true;
                                        cfg.stream_handle = (&stream).into();
                                        cfg.is_fixed_on_device = true;
                                        cfg.is_advice_on_device = true;
                                        cfg.is_instance_on_device = true;
                                        cfg.is_previous_value_on_device = true;
                                        cfg.is_result_on_device = true;

                                        gate_evaluation(
                                            &gate_data,
                                            &calculation_data,
                                            &horner_data,
                                            &mut d_result[..],
                                            &cfg,
                                        )
                                        .unwrap();

                                        let cfg = VecOpsConfig::default();
                                        inplace_invert(&mut d_result, &stream);

                                        accumulate_scalars(&mut d_inputs_inv_sum, &d_result, &cfg).unwrap();
                                    },
                                );

                                stream.synchronize().unwrap();
                                stream.destroy().unwrap();

                                d_inputs_inv_sum
                            },
                            || {
                                let mut stream = IcicleStream::create().unwrap();
                                let mut d_result = DeviceVec::device_malloc_async(domain.extended_len(), &stream).unwrap();

                                let gate_data = GateData::new(
                                    unsafe { icicle_fixed.as_ptr() },
                                    fixed.len() as u32,
                                    fixed[0].len() as u32,
                                    unsafe { icicle_advice.as_ptr() },
                                    num_advice_rows as u32,
                                    num_advice_cols as u32,
                                    unsafe { icicle_instance.as_ptr() },
                                    num_instance_rows as u32,
                                    num_instance_cols as u32,
                                    icicle_challenges.as_ptr(),
                                    challenges.len() as u32,
                                    icicle_beta.as_ptr(),
                                    icicle_gamma.as_ptr(),
                                    icicle_theta.as_ptr(),
                                    unsafe { icicle_y.as_ptr() },
                                );

                                let (inputs_lookup_evaluator, table_lookup_evaluator) = &self.lookups[n];

                                let inputs_prods_vec = vec![ScalarField::one(); values.len()];
                                let h_inputs_prods = HostSlice::from_slice(&inputs_prods_vec);
                                let mut d_inputs_prods =
                                    DeviceVec::<ScalarField>::device_malloc_async(values.len(), &stream).unwrap();
                                d_inputs_prods.copy_from_host_async(h_inputs_prods, &stream).unwrap();

                                inputs_lookup_evaluator
                                    .iter()
                                    .for_each(|input_lookup_evaluator| {
                                        let (
                                            icicle_calculations,
                                            targets,
                                            value_types,
                                            value_indices,
                                            icicle_constants,
                                            icicle_rotations,
                                            size,
                                            num_intermediates,
                                            horner_value_types,
                                            horner_value_indices,
                                            horner_offsets,
                                            horner_sizes,
                                        ) = create_calculation_data::<C>(
                                            &input_lookup_evaluator.calculations,
                                            &input_lookup_evaluator.constants,
                                            &input_lookup_evaluator.rotations,
                                            input_lookup_evaluator.num_intermediates,
                                        );

                                        let calculation_data = CalculationData::new(
                                            icicle_calculations.as_ptr(),
                                            targets.as_ptr(),
                                            value_types.as_ptr(),
                                            value_indices.as_ptr(),
                                            icicle_constants.as_ptr(),
                                            icicle_constants.len() as u32,
                                            icicle_rotations.as_ptr(),
                                            icicle_rotations.len() as u32,
                                            unsafe { icicle_previous_value.as_ptr() },
                                            false,
                                            size,
                                            num_intermediates,
                                            values.len() as u32,
                                            rot_scale as u32,
                                            isize as u32,
                                        );

                                        let horner_data = HornerData::new(
                                            horner_value_types.as_ptr(),
                                            horner_value_indices.as_ptr(),
                                            horner_offsets.as_ptr(),
                                            horner_sizes.as_ptr(),
                                            horner_value_types.len() as u32,
                                        );

                                        let mut cfg = GateOpsConfig::default();

                                        cfg.is_fixed_on_device = true;
                                        cfg.is_advice_on_device = true;
                                        cfg.is_instance_on_device = true;
                                        cfg.is_previous_value_on_device = true;
                                        cfg.is_result_on_device = true;

                                        gate_evaluation(
                                            &gate_data,
                                            &calculation_data,
                                            &horner_data,
                                            &mut d_result[..],
                                            &cfg,
                                        )
                                        .unwrap();

                                        inplace_mul(&mut d_inputs_prods, &d_result, &stream);
                                    });

                                let table_values: DeviceVec<ScalarField> = {
                                    let (
                                        icicle_calculations,
                                        targets,
                                        value_types,
                                        value_indices,
                                        icicle_constants,
                                        icicle_rotations,
                                        size,
                                        num_intermediates,
                                        horner_value_types,
                                        horner_value_indices,
                                        horner_offsets,
                                        horner_sizes,
                                    ) = create_calculation_data::<C>(
                                        &table_lookup_evaluator.calculations,
                                        &table_lookup_evaluator.constants,
                                        &table_lookup_evaluator.rotations,
                                        table_lookup_evaluator.num_intermediates,
                                    );

                                    let calculation_data = CalculationData::new(
                                        icicle_calculations.as_ptr(),
                                        targets.as_ptr(),
                                        value_types.as_ptr(),
                                        value_indices.as_ptr(),
                                        icicle_constants.as_ptr(),
                                        icicle_constants.len() as u32,
                                        icicle_rotations.as_ptr(),
                                        icicle_rotations.len() as u32,
                                        unsafe { icicle_previous_value.as_ptr() },
                                        false,
                                        size,
                                        num_intermediates,
                                        domain.extended_len() as u32,
                                        rot_scale as u32,
                                        isize as u32,
                                    );

                                    let horner_data = HornerData::new(
                                        horner_value_types.as_ptr(),
                                        horner_value_indices.as_ptr(),
                                        horner_offsets.as_ptr(),
                                        horner_sizes.as_ptr(),
                                        horner_value_types.len() as u32,
                                    );

                                    let mut cfg = GateOpsConfig::default();
                                    cfg.is_fixed_on_device = true;
                                    cfg.is_advice_on_device = true;
                                    cfg.is_instance_on_device = true;
                                    cfg.is_previous_value_on_device = true;

                                    let mut d_table_values =
                                        DeviceVec::device_malloc_async(values.len(), &stream).unwrap();

                                    gate_evaluation(
                                        &gate_data,
                                        &calculation_data,
                                        &horner_data,
                                        &mut d_table_values[..],
                                        &cfg,
                                    )
                                    .unwrap();

                                    d_table_values
                                };

                                stream.synchronize().unwrap();
                                stream.destroy().unwrap();

                                (d_inputs_prods, table_values)
                            }
                        )
                    );

                    let lookup_data = LookupData::new(
                        unsafe { table_values.as_ptr() },
                        table_values.len() as u32,
                        unsafe { inputs_prods.as_ptr() },
                        inputs_prods.len() as u32,
                        unsafe { inputs_inv_sums.as_ptr() },
                        inputs_inv_sums.len() as u32,
                        unsafe { phi_coset.as_ptr() },
                        phi_coset.len() as u32,
                        unsafe { m_coset.as_ptr() },
                        m_coset.len() as u32,
                        unsafe { icicle_l0.as_ptr() },
                        l0.len() as u32,
                        unsafe { icicle_l_last.as_ptr() },
                        l_last.len() as u32,
                        unsafe { icicle_l_active_row.as_ptr() },
                        l_active_row.len() as u32,
                        unsafe { icicle_y.as_ptr() },
                        unsafe { icicle_previous_value.as_ptr() },
                        values.len() as u32,
                        rot_scale as u32,
                        isize as u32,
                    );

                    let mut cfg = LookupConfig::default();
                    cfg.is_coset_on_device = true;
                    cfg.is_table_values_on_device = true;
                    cfg.is_inputs_inv_sums_on_device = true;
                    cfg.is_inputs_prods_on_device = true;
                    cfg.is_l_on_device = true;
                    cfg.is_y_on_device = true;
                    cfg.is_result_on_device = true;
                    lookups_constraint(&lookup_data, &mut icicle_previous_value, &cfg).unwrap();
                }

                let result: Vec<C::ScalarExt> = c_scalars_from_device_vec(&mut icicle_previous_value, &IcicleStream::default());
                values = Polynomial::from_vec(result);
            }

            #[cfg(all(not(feature = "mv-lookup"), feature = "precompute-coset"))]
            let mut cosets: Vec<_> = {
                let domain = &pk.vk.domain;
                lookups
                    .par_iter()
                    .map(|lookup| {
                        (
                            domain.coeff_to_extended(lookup.product_poly.clone()),
                            domain.coeff_to_extended(lookup.permuted_input_poly.clone()),
                            domain.coeff_to_extended(lookup.permuted_table_poly.clone()),
                        )
                    })
                    .collect()
            };

            // Lookup Constraints
            #[cfg(not(feature = "mv-lookup"))]
            {
                for (n, lookup) in lookups.iter().enumerate() {
                    // Polynomials required for this lookup.
                    // Calculated here so these only have to be kept in memory for the short time
                    // they are actually needed.
    
                    #[cfg(feature = "precompute-coset")]
                    let (product_coset, permuted_input_coset, permuted_table_coset) = &cosets.remove(0);
    
                    #[cfg(not(feature = "precompute-coset"))]
                    let (product_coset, permuted_input_coset, permuted_table_coset) = {
                        let mut stream_coset = IcicleStream::create().unwrap();
                        let mut stream_input_coset = IcicleStream::create().unwrap();
                        let mut stream_table_coset = IcicleStream::create().unwrap();
    
                        let product_coset = pk.vk.domain.coeff_to_extended(&lookup.product_poly, &stream_coset);
                        let permuted_input_coset =
                            pk.vk.domain.coeff_to_extended(&lookup.permuted_input_poly, &stream_input_coset);
                        let permuted_table_coset =
                            pk.vk.domain.coeff_to_extended(&lookup.permuted_table_poly, &stream_table_coset);
    
                        stream_coset.synchronize().unwrap();
                        stream_input_coset.synchronize().unwrap();
                        stream_table_coset.synchronize().unwrap();
    
                        stream_coset.destroy().unwrap();
                        stream_input_coset.destroy().unwrap();
                        stream_table_coset.destroy().unwrap();
    
                        (product_coset, permuted_input_coset, permuted_table_coset)
    
                    };
    
                    // Lookup constraints
                    parallelize(&mut values, |values, start| {
                        let lookup_evaluator = &self.lookups[n];
                        let mut eval_data = lookup_evaluator.instance();
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;
    
                            let table_value = lookup_evaluator.evaluate(
                                &mut eval_data,
                                fixed,
                                &advice[..],
                                &instance[..],
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                            );
    
                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                            let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);
    
                            let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
                            // l_0(X) * (1 - z(X)) = 0
                            *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                            // l_last(X) * (z(X)^2 - z(X)) = 0
                            *value = *value * y
                                + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                    * l_last[idx]);
                            // (1 - (l_last(X) + l_blind(X))) * (
                            //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
                            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
                            //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
                            // ) = 0
                            *value = *value * y
                                + ((product_coset[r_next]
                                    * (permuted_input_coset[idx] + beta)
                                    * (permuted_table_coset[idx] + gamma)
                                    - product_coset[idx] * table_value)
                                    * l_active_row[idx]);
                            // Check that the first values in the permuted input expression and permuted
                            // fixed expression are the same.
                            // l_0(X) * (a'(X) - s'(X)) = 0
                            *value = *value * y + (a_minus_s * l0[idx]);
                            // Check that each value in the permuted lookup input expression is either
                            // equal to the value above it, or the value at the same index in the
                            // permuted table expression.
                            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
                            *value = *value * y
                                + (a_minus_s
                                    * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
                                    * l_active_row[idx]);
                        }
                    });
                }
            }

            // Shuffle constraints
            {
                for (n, shuffle) in shuffles.iter().enumerate() {
                    let product_coset = pk.vk.domain.coeff_to_extended(&shuffle.product_poly, &IcicleStream::default());

                    // Shuffle constraints
                    parallelize(&mut values, |values, start| {
                        let input_evaluator = &self.shuffles[2 * n];
                        let shuffle_evaluator = &self.shuffles[2 * n + 1];
                        let mut eval_data_input = shuffle_evaluator.instance();
                        let mut eval_data_shuffle = shuffle_evaluator.instance();
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;

                            let input_value = input_evaluator.evaluate(
                                &mut eval_data_input,
                                fixed,
                                &advice[..],
                                &instance[..],
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                            );

                            let shuffle_value = shuffle_evaluator.evaluate(
                                &mut eval_data_shuffle,
                                fixed,
                                &advice[..],
                                &instance[..],
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                            );

                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);

                            // l_0(X) * (1 - z(X)) = 0
                            *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                            // l_last(X) * (z(X)^2 - z(X)) = 0
                            *value = *value * y
                                + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                    * l_last[idx]);
                            // (1 - (l_last(X) + l_blind(X))) * (z(\omega X) (s(X) + \gamma) - z(X) (a(X) + \gamma)) = 0
                            *value = *value * y
                                + l_active_row[idx]
                                    * (product_coset[r_next] * shuffle_value
                                        - product_coset[idx] * input_value)
                        }
                    });
                }
            }
        }
        values
    }
}

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::ZERO,
                C::ScalarExt::ONE,
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<C::ScalarExt>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == C::ScalarExt::ZERO {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<C> {
        EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> C::ScalarExt {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            let idx = get_rotation_idx(idx, *rot, rot_scale, isize);
            data.rotations[rot_idx] = idx;
        }

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            C::ScalarExt::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}