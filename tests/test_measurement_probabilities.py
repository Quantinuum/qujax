"""Marginal probabilities in the caller's requested measurement order."""

import jax
from jax import numpy as jnp
import numpy as np
import pytest

import qujax


@pytest.mark.parametrize("qubits", [[], [0], [2], [0, 2], [2, 0], [2, 1, 0], [0, 1, 2]])
@pytest.mark.parametrize("compiled", [False, True])
def test_ordered_marginal_probabilities(qubits, compiled):
    """Sum basis populations independently, including zero and three measured qubits."""
    rng = np.random.default_rng(17)
    factor = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    matrix = factor @ factor.conj().T
    matrix /= np.trace(matrix)
    expected = np.zeros(2 ** len(qubits))
    for basis, probability in enumerate(np.diag(matrix).real):
        outcome = 0
        for qubit in qubits:
            outcome = 2 * outcome + ((basis >> (2 - qubit)) & 1)
        expected[outcome] += probability

    def probabilities(density):
        return qujax.densitytensor_to_measurement_probabilities(density, qubits)

    if compiled:
        probabilities = jax.jit(probabilities)
    actual = probabilities(jnp.asarray(matrix).reshape((2,) * 6))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.sum(actual), 1.0, atol=1e-6)


def test_reversed_measurement_matches_basis_bitstring():
    """In state |100>, measuring qubits (2,0) produces bitstring 01."""
    matrix = np.zeros((8, 8), dtype=complex)
    matrix[4, 4] = 1
    probabilities = qujax.densitytensor_to_measurement_probabilities(
        jnp.asarray(matrix).reshape((2,) * 6), [2, 0]
    )
    np.testing.assert_allclose(probabilities, [0, 1, 0, 0], atol=1e-6)
