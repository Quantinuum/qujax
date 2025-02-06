from __future__ import annotations

from typing import Callable, Sequence, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.lax import fori_loop

from qujax.statetensor import apply_gate
from qujax.utils import check_hermitian, paulis


def statetensor_to_single_expectation(
    statetensor: jax.Array, hermitian: jax.Array, qubit_inds: Sequence[int]
) -> jax.Array:
    """
    Evaluates expectation value of an observable represented by a Hermitian matrix (in tensor form).

    Args:
        statetensor: Input statetensor.
        hermitian: Hermitian array
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of qubit indices for Hermitian matrix to be applied to.
            Must have 2 * len(qubit_inds) == hermitian.ndim

    Returns:
        Expected value (float).
    """
    statetensor_new = apply_gate(statetensor, hermitian, qubit_inds)
    axes = tuple(range(statetensor.ndim))
    return jnp.tensordot(
        statetensor.conjugate(), statetensor_new, axes=(axes, axes)
    ).real


def get_hermitian_tensor(hermitian_seq: Sequence[Union[str, jax.Array]]) -> jax.Array:
    """
    Convert a sequence of observables represented by Pauli strings or Hermitian matrices
    in tensor form into single array (in tensor form).

    Args:
        hermitian_seq: Sequence of Hermitian strings or arrays.

    Returns:
        Hermitian matrix in tensor form (array).
    """
    for h in hermitian_seq:
        check_hermitian(h)

    single_arrs = [paulis[h] if isinstance(h, str) else h for h in hermitian_seq]
    single_arrs = [
        h_arr.reshape((2,) * int(jnp.rint(jnp.log2(h_arr.size))))
        for h_arr in single_arrs
    ]

    full_mat = single_arrs[0]
    for single_matrix in single_arrs[1:]:
        full_mat = jnp.kron(full_mat, single_matrix)
    full_mat = full_mat.reshape((2,) * int(jnp.rint(jnp.log2(full_mat.size))))
    return full_mat


def _get_tensor_to_expectation_func(
    hermitian_seq_seq: Sequence[Sequence[Union[str, jax.Array]]],
    qubits_seq_seq: Sequence[Sequence[int]],
    coefficients: Union[Sequence[float], jax.Array],
    contraction_function: Callable,
) -> Callable[[jax.Array], float]:
    """
    Takes strings (or arrays) representing Hermitian matrices, along with qubit indices and
    a list of coefficients and returns a function that converts a tensor into an expected value.
    The contraction function performs the tensor contraction according to the type of tensor
    provided (i.e. whether it is a statetensor or a densitytensor).

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian matrix is either represented by a tensor (jax.Array) or by a
            list of 'X', 'Y' or 'Z' characters corresponding to the standard Pauli matrices.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.
        contraction_function: Function that performs the tensor contraction.

    Returns:
        Function that takes tensor and returns expected value (float).
    """

    hermitian_tensors = [get_hermitian_tensor(h_seq) for h_seq in hermitian_seq_seq]

    def tensor_to_expectation_func(tensor: jax.Array) -> float:
        """
        Maps tensor to expected value.

        Args:
            tensor: Input tensor.

        Returns:
            Expected value (float).
        """
        out = 0
        for hermitian, qubit_inds, coeff in zip(
            hermitian_tensors, qubits_seq_seq, coefficients
        ):
            out += coeff * contraction_function(tensor, hermitian, qubit_inds)
        return out

    return tensor_to_expectation_func


def get_statetensor_to_expectation_func(
    hermitian_seq_seq: Sequence[Sequence[Union[str, jax.Array]]],
    qubits_seq_seq: Sequence[Sequence[int]],
    coefficients: Union[Sequence[float], jax.Array],
) -> Callable[[jax.Array], float]:
    """
    Takes strings (or arrays) representing Hermitian matrices, along with qubit indices and
    a list of coefficients and returns a function that converts a statetensor into an expected
    value.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian matrix is either represented by a tensor (jax.Array)
            or by a list of 'X', 'Y' or 'Z' characters corresponding to the standard Pauli matrices.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """

    return _get_tensor_to_expectation_func(
        hermitian_seq_seq,
        qubits_seq_seq,
        coefficients,
        statetensor_to_single_expectation,
    )


def get_statetensor_to_sampled_expectation_func(
    hermitian_seq_seq: Sequence[Sequence[Union[str, jax.Array]]],
    qubits_seq_seq: Sequence[Sequence[int]],
    coefficients: Union[Sequence[float], jax.Array],
) -> Callable[[jax.Array, random.PRNGKeyArray, int], float]:
    """
    Converts strings (or arrays) representing Hermitian matrices, qubit indices and
    coefficients into a function that converts a statetensor into a sampled expected value.

    On a quantum device, measurements are always taken in the computational basis, as such
    sampled expectation values should be taken with respect to an observable that commutes
    with the Pauli Z - a warning will be raised if it does not.

    qujax applies an importance sampling heuristic for sampled expectation values that only
    reflects the physical notion of measurement in the case that the observable commutes with Z.
    In the case that it does not, the expectation value will still be asymptotically unbiased
    but not representative of an experiment on a real quantum device.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian is either a tensor (jax.Array) or a string in ('X', 'Y', 'Z').
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor, random key and integer number of shots
        and returns sampled expected value (float).
    """
    statetensor_to_expectation_func = get_statetensor_to_expectation_func(
        hermitian_seq_seq, qubits_seq_seq, coefficients
    )

    for hermitian_seq in hermitian_seq_seq:
        for h in hermitian_seq:
            check_hermitian(h, check_z_commutes=True)

    def statetensor_to_sampled_expectation_func(
        statetensor: jax.Array, random_key: random.PRNGKeyArray, n_samps: int
    ) -> float:
        """
        Maps statetensor to sampled expected value.

        Args:
            statetensor: Input statetensor.
            random_key: JAX random key
            n_samps: Number of samples contributing to sampled expectation.

        Returns:
            Sampled expected value (float).
        """
        measure_probs = jnp.abs(statetensor) ** 2
        sampled_probs = sample_probs(measure_probs, random_key, n_samps)
        iweights = jnp.sqrt(sampled_probs / measure_probs)
        return statetensor_to_expectation_func(statetensor * iweights)

    return statetensor_to_sampled_expectation_func


def sample_probs(
    measure_probs: jax.Array, random_key: random.PRNGKeyArray, n_samps: int
):
    """
    Generate an empirical distribution from a probability distribution.

    Args:
        measure_probs: Probability distribution.
        random_key: JAX random key
        n_samps: Number of samples contributing to empirical distribution.

    Returns:
        Empirical distribution (jax.Array).
    """
    measure_probs_flat = measure_probs.flatten()
    sampled_integers = random.choice(
        random_key,
        a=jnp.arange(measure_probs.size),
        shape=(n_samps,),
        p=measure_probs_flat,
    )
    sampled_probs = fori_loop(
        0,
        n_samps,
        lambda i, sv: sv.at[sampled_integers[i]].add(1 / n_samps),
        jnp.zeros_like(measure_probs_flat),
    )
    return sampled_probs.reshape(measure_probs.shape)
