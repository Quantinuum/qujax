from __future__ import annotations

from typing import Callable, Sequence, Union

import jax
from jax.typing import ArrayLike
from jax import numpy as jnp
from jax import random

from qujax.densitytensor import _kraus_single, partial_trace
from qujax.statetensor_observable import _get_tensor_to_expectation_func, sample_probs
from qujax.utils import bitstrings_to_integers, check_hermitian


def densitytensor_to_single_expectation(
    densitytensor: jax.Array, hermitian: jax.Array, qubit_inds: Sequence[int]
) -> jax.Array:
    """
    Evaluates expectation value of an observable represented by a Hermitian matrix (in tensor form).

    Args:
        densitytensor: Input densitytensor.
        hermitian: Hermitian matrix representing observable
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of qubit indices for Hermitian matrix to be applied to.
            Must have 2 * len(qubit_inds) == hermitian.ndim
    Returns:
        Expected value (float).
    """
    n_qubits = densitytensor.ndim // 2
    dt_indices = 2 * list(range(n_qubits))
    hermitian_indices = [i + densitytensor.ndim // 2 for i in range(hermitian.ndim)]
    for n, q in enumerate(qubit_inds):
        dt_indices[q] = hermitian_indices[n + len(qubit_inds)]
        dt_indices[q + n_qubits] = hermitian_indices[n]
    return jnp.einsum(densitytensor, dt_indices, hermitian, hermitian_indices).real


def get_densitytensor_to_expectation_func(
    hermitian_seq_seq: Sequence[Sequence[Union[str, jax.Array]]],
    qubits_seq_seq: Sequence[Sequence[int]],
    coefficients: Union[Sequence[float], jax.Array],
) -> Callable[[jax.Array], float]:
    """
    Takes strings (or arrays) representing Hermitian matrices, along with qubit indices and
    a list of coefficients and returns a function that converts a densitytensor into an
    expected value.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian matrix is either represented by a tensor (jax.Array)
            or by a list of 'X', 'Y' or 'Z' characters corresponding to the standard Pauli matrices.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes densitytensor and returns expected value (float).
    """

    return _get_tensor_to_expectation_func(
        hermitian_seq_seq,
        qubits_seq_seq,
        coefficients,
        densitytensor_to_single_expectation,
    )


def get_densitytensor_to_sampled_expectation_func(
    hermitian_seq_seq: Sequence[Sequence[Union[str, jax.Array]]],
    qubits_seq_seq: Sequence[Sequence[int]],
    coefficients: Union[Sequence[float], jax.Array],
) -> Callable[[jax.Array, random.PRNGKeyArray, int], float]:
    """
    Converts strings (or arrays) representing Hermitian matrices, qubit indices and
    coefficients into a function that converts a densitytensor into a sampled expected value.

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
        Function that takes densitytensor, random key and integer number of shots
        and returns sampled expected value (float).
    """
    densitytensor_to_expectation_func = get_densitytensor_to_expectation_func(
        hermitian_seq_seq, qubits_seq_seq, coefficients
    )

    for hermitian_seq in hermitian_seq_seq:
        for h in hermitian_seq:
            check_hermitian(h, check_z_commutes=True)

    def densitytensor_to_sampled_expectation_func(
        densitytensor: jax.Array, random_key: random.PRNGKeyArray, n_samps: int
    ) -> float:
        """
        Maps densitytensor to sampled expected value.

        Args:
            densitytensor: Input densitytensor.
            random_key: JAX random key
            n_samps: Number of samples contributing to sampled expectation.

        Returns:
            Sampled expected value (float).

        """
        n_qubits = densitytensor.ndim // 2
        dm = densitytensor.reshape((2**n_qubits, 2**n_qubits))
        measure_probs = jnp.diag(dm).real
        sampled_probs = sample_probs(measure_probs, random_key, n_samps)
        iweights = jnp.sqrt(sampled_probs / measure_probs)
        return densitytensor_to_expectation_func(
            densitytensor * jnp.outer(iweights, iweights).reshape(densitytensor.shape)
        )

    return densitytensor_to_sampled_expectation_func


def densitytensor_to_measurement_probabilities(
    densitytensor: jax.Array, qubit_inds: Sequence[int]
) -> jax.Array:
    """
    Extract array of measurement probabilities given a densitytensor and some qubit indices to
    measure (in the computational basis).
    I.e. the ith element of the array corresponds to the probability of observing the bitstring
    represented by the integer i on the measured qubits.

    Args:
        densitytensor: Input densitytensor.
        qubit_inds: Sequence of qubit indices to measure.

    Returns:
        Normalised array of measurement probabilities.
    """
    n_qubits = densitytensor.ndim // 2
    n_qubits_measured = len(qubit_inds)
    qubit_inds_trace_out = [i for i in range(n_qubits) if i not in qubit_inds]
    return jnp.diag(
        partial_trace(densitytensor, qubit_inds_trace_out).reshape(
            2 * n_qubits_measured, 2 * n_qubits_measured
        )
    ).real


def densitytensor_to_measured_densitytensor(
    densitytensor: jax.Array,
    qubit_inds: Sequence[int],
    measurement: ArrayLike,
) -> jax.Array:
    """
    Returns the post-measurement densitytensor assuming that qubit_inds are measured
    (in the computational basis) and the given measurement (integer or bitstring) is observed.

    Args:
        densitytensor: Input densitytensor.
        qubit_inds: Sequence of qubit indices to measure.
        measurement: Observed integer or bitstring.

    Returns:
        Post-measurement densitytensor (same shape as input densitytensor).
    """
    measurement = jnp.array(measurement)
    measured_int = (
        bitstrings_to_integers(measurement) if measurement.ndim == 1 else measurement
    )

    n_qubits = densitytensor.ndim // 2
    n_qubits_measured = len(qubit_inds)
    qubit_inds_projector = jnp.diag(
        jnp.zeros(2**n_qubits_measured).at[measured_int].set(1)
    ).reshape((2,) * 2 * n_qubits_measured)
    unnorm_densitytensor = _kraus_single(
        densitytensor, qubit_inds_projector, qubit_inds
    )
    norm_const = jnp.trace(unnorm_densitytensor.reshape(2**n_qubits, 2**n_qubits)).real
    return unnorm_densitytensor / norm_const
