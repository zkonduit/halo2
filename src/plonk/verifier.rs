use super::{hash_point, Error, Proof, VerifyingKey};
use crate::arithmetic::{get_challenge_scalar, Challenge, CurveAffine, Field};
use crate::poly::{
    commitment::{Guard, Params, MSM},
    Rotation,
};
use crate::transcript::Hasher;

impl<'a, C: CurveAffine> Proof<C> {
    /// Returns a boolean indicating whether or not the proof is valid
    pub fn verify<HBase: Hasher<C::Base>, HScalar: Hasher<C::Scalar>>(
        &self,
        params: &'a Params<C>,
        vk: &VerifyingKey<C>,
        mut msm: MSM<'a, C>,
        aux_commitments: &[C],
    ) -> Result<Guard<'a, C>, Error> {
        self.check_lengths(vk, aux_commitments)?;

        // Scale the MSM by a random factor to ensure that if the existing MSM
        // has is_zero() == false then this argument won't be able to interfere
        // with it to make it true, with high probability.
        msm.scale(C::Scalar::random());

        // Create a transcript for obtaining Fiat-Shamir challenges.
        let mut transcript = HBase::init(C::Base::one());

        // Hash the aux (external) commitments into the transcript
        for commitment in aux_commitments {
            hash_point(&mut transcript, commitment)?;
        }

        // Hash the prover's advice commitments into the transcript
        for commitment in &self.advice_commitments {
            hash_point(&mut transcript, commitment)?;
        }

        // Sample x_0 challenge
        let x_0: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Sample x_1 challenge
        let x_1: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Hash each permutation product commitment
        for c in &self.permutation_product_commitments {
            hash_point(&mut transcript, c)?;
        }

        // Sample x_2 challenge, which keeps the gates linearly independent.
        let x_2: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Obtain a commitment to h(X) in the form of multiple pieces of degree n - 1
        for c in &self.h_commitments {
            hash_point(&mut transcript, c)?;
        }

        // Sample x_3 challenge, which is used to ensure the circuit is
        // satisfied with high probability.
        let x_3: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // This check ensures the circuit is satisfied so long as the polynomial
        // commitments open to the correct values.
        self.check_hx(params, vk, x_0, x_1, x_2, x_3)?;

        // Hash together all the openings provided by the prover into a new
        // transcript on the scalar field.
        let mut transcript_scalar = HScalar::init(C::Scalar::one());

        for eval in self
            .advice_evals
            .iter()
            .chain(self.aux_evals.iter())
            .chain(self.fixed_evals.iter())
            .chain(self.h_evals.iter())
            .chain(self.permutation_product_evals.iter())
            .chain(self.permutation_product_inv_evals.iter())
            .chain(self.permutation_evals.iter().flat_map(|evals| evals.iter()))
        {
            transcript_scalar.absorb(*eval);
        }

        let transcript_scalar_point =
            C::Base::from_bytes(&(transcript_scalar.squeeze()).to_bytes()).unwrap();
        transcript.absorb(transcript_scalar_point);

        // Sample x_4 for compressing openings at the same points together
        let x_4: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Compress the commitments and expected evaluations at x_3 together
        // using the challenge x_4
        let mut q_commitments: Vec<_> = vec![params.empty_msm(); vk.cs.rotations.len()];
        let mut q_evals: Vec<_> = vec![C::Scalar::zero(); vk.cs.rotations.len()];
        {
            let mut accumulate = |point_index: usize, new_commitment, eval| {
                q_commitments[point_index].scale(x_4);
                q_commitments[point_index].add_term(C::Scalar::one(), new_commitment);
                q_evals[point_index] *= &x_4;
                q_evals[point_index] += &eval;
            };

            for (query_index, &(wire, ref at)) in vk.cs.advice_queries.iter().enumerate() {
                let point_index = (*vk.cs.rotations.get(at).unwrap()).0;
                accumulate(
                    point_index,
                    self.advice_commitments[wire.0],
                    self.advice_evals[query_index],
                );
            }

            for (query_index, &(wire, ref at)) in vk.cs.aux_queries.iter().enumerate() {
                let point_index = (*vk.cs.rotations.get(at).unwrap()).0;
                accumulate(
                    point_index,
                    aux_commitments[wire.0],
                    self.aux_evals[query_index],
                );
            }

            for (query_index, &(wire, ref at)) in vk.cs.fixed_queries.iter().enumerate() {
                let point_index = (*vk.cs.rotations.get(at).unwrap()).0;
                accumulate(
                    point_index,
                    vk.fixed_commitments[wire.0],
                    self.fixed_evals[query_index],
                );
            }

            let current_index = (*vk.cs.rotations.get(&Rotation::default()).unwrap()).0;
            for (commitment, eval) in self.h_commitments.iter().zip(self.h_evals.iter()) {
                accumulate(current_index, *commitment, *eval);
            }

            // Handle permutation arguments, if any exist
            if !vk.cs.permutations.is_empty() {
                // Open permutation product commitments at x_3
                for (commitment, eval) in self
                    .permutation_product_commitments
                    .iter()
                    .zip(self.permutation_product_evals.iter())
                {
                    accumulate(current_index, *commitment, *eval);
                }
                // Open permutation commitments for each permutation argument at x_3
                for (commitment, eval) in vk
                    .permutation_commitments
                    .iter()
                    .zip(self.permutation_evals.iter())
                    .flat_map(|(commitments, evals)| commitments.iter().zip(evals.iter()))
                {
                    accumulate(current_index, *commitment, *eval);
                }
                let current_index = (*vk.cs.rotations.get(&Rotation(-1)).unwrap()).0;
                // Open permutation product commitments at \omega^{-1} x_3
                for (commitment, eval) in self
                    .permutation_product_commitments
                    .iter()
                    .zip(self.permutation_product_inv_evals.iter())
                {
                    accumulate(current_index, *commitment, *eval);
                }
            }
        }

        // Sample a challenge x_5 for keeping the multi-point quotient
        // polynomial terms linearly independent.
        let x_5: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Obtain the commitment to the multi-point quotient polynomial f(X).
        hash_point(&mut transcript, &self.f_commitment)?;

        // Sample a challenge x_6 for checking that f(X) was committed to
        // correctly.
        let x_6: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        for eval in self.q_evals.iter() {
            transcript_scalar.absorb(*eval);
        }

        let transcript_scalar_point =
            C::Base::from_bytes(&(transcript_scalar.squeeze()).to_bytes()).unwrap();
        transcript.absorb(transcript_scalar_point);

        // We can compute the expected msm_eval at x_6 using the q_evals provided
        // by the prover and from x_5
        let mut msm_eval = C::Scalar::zero();
        for (&row, point_index) in vk.cs.rotations.iter() {
            let mut eval = self.q_evals[point_index.0];

            let point = vk.domain.rotate_omega(x_3, row);
            eval = eval - &q_evals[point_index.0];
            eval = eval * &(x_6 - &point).invert().unwrap();

            msm_eval *= &x_5;
            msm_eval += &eval;
        }

        // Sample a challenge x_7 that we will use to collapse the openings of
        // the various remaining polynomials at x_6 together.
        let x_7: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Compute the final commitment that has to be opened
        let mut commitment_msm = params.empty_msm();
        commitment_msm.add_term(C::Scalar::one(), self.f_commitment);
        for (_, &point_index) in vk.cs.rotations.iter() {
            commitment_msm.scale(x_7);
            commitment_msm.add_msm(&q_commitments[point_index.0]);
            msm_eval *= &x_7;
            msm_eval += &self.q_evals[point_index.0];
        }

        // Verify the opening proof
        self.opening
            .verify(params, msm, &mut transcript, x_6, commitment_msm, msm_eval)
            .map_err(|_| Error::OpeningError)
    }

    /// Checks that the lengths of vectors are consistent with the constraint
    /// system
    fn check_lengths(&self, vk: &VerifyingKey<C>, aux_commitments: &[C]) -> Result<(), Error> {
        // Check that aux_commitments matches the expected number of aux_wires
        // and self.aux_evals
        if aux_commitments.len() != vk.cs.num_aux_wires
            || self.aux_evals.len() != vk.cs.num_aux_wires
        {
            return Err(Error::IncompatibleParams);
        }

        if self.q_evals.len() != vk.cs.rotations.len() {
            return Err(Error::IncompatibleParams);
        }

        // TODO: check h_evals

        if self.fixed_evals.len() != vk.cs.fixed_queries.len() {
            return Err(Error::IncompatibleParams);
        }

        if self.advice_evals.len() != vk.cs.advice_queries.len() {
            return Err(Error::IncompatibleParams);
        }

        if self.permutation_evals.len() != vk.cs.permutations.len() {
            return Err(Error::IncompatibleParams);
        }

        for (permutation_evals, permutation) in
            self.permutation_evals.iter().zip(vk.cs.permutations.iter())
        {
            if permutation_evals.len() != permutation.len() {
                return Err(Error::IncompatibleParams);
            }
        }

        if self.permutation_product_inv_evals.len() != vk.cs.permutations.len() {
            return Err(Error::IncompatibleParams);
        }

        if self.permutation_product_evals.len() != vk.cs.permutations.len() {
            return Err(Error::IncompatibleParams);
        }

        if self.permutation_product_commitments.len() != vk.cs.permutations.len() {
            return Err(Error::IncompatibleParams);
        }

        // TODO: check h_commitments

        if self.advice_commitments.len() != vk.cs.num_advice_wires {
            return Err(Error::IncompatibleParams);
        }

        Ok(())
    }

    /// Checks that this proof's h_evals are correct, and thus that all of the
    /// rules are satisfied.
    fn check_hx(
        &self,
        params: &'a Params<C>,
        vk: &VerifyingKey<C>,
        x_0: C::Scalar,
        x_1: C::Scalar,
        x_2: C::Scalar,
        x_3: C::Scalar,
    ) -> Result<(), Error> {
        // x_3^n
        let x_3n = x_3.pow(&[params.n as u64, 0, 0, 0]);

        // TODO: bubble this error up
        // l_0(x_3)
        let l_0 = (x_3 - &C::Scalar::one()).invert().unwrap() // 1 / (x_3 - 1)
            * &(x_3n - &C::Scalar::one()) // (x_3^n - 1) / (x_3 - 1)
            * &vk.domain.get_barycentric_weight(); // l_0(x_3)

        // Compute the expected value of h(x_3)
        let expected_h_eval = std::iter::empty()
            // Evaluate the circuit using the custom gates provided
            .chain(vk.cs.gates.iter().map(|poly| {
                poly.evaluate(
                    &|index| self.fixed_evals[index],
                    &|index| self.advice_evals[index],
                    &|index| self.aux_evals[index],
                    &|a, b| a + &b,
                    &|a, b| a * &b,
                    &|a, scalar| a * &scalar,
                )
            }))
            // l_0(X) * (1 - z(X)) = 0
            .chain(
                self.permutation_product_evals
                    .iter()
                    .map(|product_eval| l_0 * &(C::Scalar::one() - &product_eval)),
            )
            // z(X) \prod (p(X) + \beta s_i(X) + \gamma)
            // - z(omega^{-1} X) \prod (p(X) + \delta^i \beta X + \gamma)
            .chain(
                vk.cs
                    .permutations
                    .iter()
                    .zip(self.permutation_evals.iter())
                    .zip(self.permutation_product_evals.iter())
                    .zip(self.permutation_product_inv_evals.iter())
                    .map(
                        |(((wires, permutation_evals), product_eval), product_inv_eval)| {
                            let mut left = *product_eval;
                            for (advice_eval, permutation_eval) in wires
                                .iter()
                                .map(|&wire| {
                                    self.advice_evals[vk.cs.get_advice_query_index(wire, 0)]
                                })
                                .zip(permutation_evals.iter())
                            {
                                left *= &(advice_eval + &(x_0 * permutation_eval) + &x_1);
                            }

                            let mut right = *product_inv_eval;
                            let mut current_delta = x_0 * &x_3;
                            for advice_eval in wires.iter().map(|&wire| {
                                self.advice_evals[vk.cs.get_advice_query_index(wire, 0)]
                            }) {
                                right *= &(advice_eval + &current_delta + &x_1);
                                current_delta *= &C::Scalar::DELTA;
                            }

                            left - &right
                        },
                    ),
            )
            .fold(C::Scalar::zero(), |h_eval, v| h_eval * &x_2 + &v);

        // Compute h(x_3) from the prover
        let (_, h_eval) = self
            .h_evals
            .iter()
            .fold((C::Scalar::one(), C::Scalar::zero()), |(cur, acc), eval| {
                (cur * &x_3n, acc + &(cur * eval))
            });

        // Did the prover commit to the correct polynomial?
        if expected_h_eval != (h_eval * &(x_3n - &C::Scalar::one())) {
            return Err(Error::ConstraintSystemFailure);
        }

        Ok(())
    }
}
