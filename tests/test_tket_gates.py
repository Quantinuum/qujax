"""TKET canonical gate matrices and their JAX transformations."""

import jax
from jax import numpy as jnp
import numpy as np
import pytest

import qujax

PARAMETERS = [(0.0, 0.0, 0.0), (0.2, -0.3, 0.7), (-1.0, 0.5, 2.0)]


def reference_matrix(name, params):
    """Independent closed forms in half-turn units."""
    first, second, third = np.array(params) * np.pi / 2
    if name == "TK1":
        cosine, sine = np.cos(second), np.sin(second)
        return np.array(
            [
                [
                    cosine * np.exp(-1j * (first + third)),
                    -1j * sine * np.exp(-1j * (first - third)),
                ],
                [
                    -1j * sine * np.exp(1j * (first - third)),
                    cosine * np.exp(1j * (first + third)),
                ],
            ]
        )
    even_phase, odd_phase = np.exp(-1j * third), np.exp(1j * third)
    even_cos, even_sin = np.cos(first - second), np.sin(first - second)
    odd_cos, odd_sin = np.cos(first + second), np.sin(first + second)
    return np.array(
        [
            [even_phase * even_cos, 0, 0, -1j * even_phase * even_sin],
            [0, odd_phase * odd_cos, -1j * odd_phase * odd_sin, 0],
            [0, -1j * odd_phase * odd_sin, odd_phase * odd_cos, 0],
            [-1j * even_phase * even_sin, 0, 0, even_phase * even_cos],
        ]
    )


@pytest.mark.parametrize("name", ["TK1", "TK2"])
@pytest.mark.parametrize("params", PARAMETERS)
@pytest.mark.parametrize("compiled", [False, True])
def test_canonical_gate_and_circuit(name, params, compiled):
    """Verify the full matrix and the public state simulator by gate name."""
    gate = getattr(qujax.gates, name)
    if compiled:
        gate = jax.jit(gate)
    expected = reference_matrix(name, params)
    actual = np.asarray(gate(*params)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(len(actual)), atol=1e-6)

    n_qubits = 1 if name == "TK1" else 2
    simulate = qujax.get_params_to_statetensor_func(
        [name], [list(range(n_qubits))], [[0, 1, 2]], n_qubits
    )
    if compiled:
        simulate = jax.jit(simulate)
    state = np.asarray(simulate(jnp.array(params))).reshape(-1)
    np.testing.assert_allclose(state, expected[:, 0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["TK1", "TK2"])
@pytest.mark.parametrize("params", PARAMETERS)
def test_canonical_gate_gradients(name, params):
    """Check compiled gradients against analytic matrix-entry derivatives."""
    gate = getattr(qujax.gates, name)
    size = 2 if name == "TK1" else 4
    gradient = jax.jit(jax.grad(lambda p: jnp.real(gate(*p).reshape(size, size)[0, 0])))
    first, second, third = np.array(params) * np.pi / 2
    if name == "TK1":
        da = -np.cos(second) * np.sin(first + third)
        db = -np.sin(second) * np.cos(first + third)
        expected = np.pi / 2 * np.array([da, db, da])
    else:
        da = -np.cos(third) * np.sin(first - second)
        dc = -np.sin(third) * np.cos(first - second)
        expected = np.pi / 2 * np.array([da, -da, dc])
    np.testing.assert_allclose(
        gradient(jnp.array(params)), expected, rtol=1e-6, atol=1e-6
    )
