import jax
import jax.numpy as jnp
import pytest

import qujax

BUILDERS = [
    qujax.get_params_to_statetensor_func,
    qujax.get_params_to_densitytensor_func,
    qujax.get_params_to_unitarytensor_func,
]


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "indices,parameters", [([0, 1], [0.2]), ([5], [0.0] * 5), ([-2], [0.2])]
)
def test_reject_undersized_parameter_vectors(builder, compiled, indices, parameters):
    circuit = builder(
        ["Rx"] * len(indices), [[0]] * len(indices), [[i] for i in indices]
    )
    if compiled:
        circuit = jax.jit(circuit)
    with pytest.raises(ValueError, match="parameters"):
        circuit(jnp.array(parameters))


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("compiled", [False, True])
def test_valid_negative_index_and_unused_parameters(builder, compiled):
    circuit = builder(["Rx"], [[0]], [[-2]])
    if compiled:
        circuit = jax.jit(circuit)
    angle = 0.3
    result = circuit(jnp.array([0.7, angle, 0.9]))
    cosine, sine = jnp.cos(jnp.pi * angle / 2), jnp.sin(jnp.pi * angle / 2)
    state = jnp.array([cosine, -1j * sine])
    if builder is qujax.get_params_to_statetensor_func:
        expected = state
    elif builder is qujax.get_params_to_densitytensor_func:
        expected = jnp.outer(state, state.conj())
    else:
        expected = jnp.array([[cosine, -1j * sine], [-1j * sine, cosine]])
    assert jnp.allclose(result, expected)


def test_parameter_validation_preserves_jit_differentiation():
    circuit = qujax.get_params_to_statetensor_func(["Rx"], [[0]], [[1]])

    def probability(parameters):
        return jnp.abs(circuit(parameters)[0]) ** 2

    parameters = jnp.array([0.7, 0.3, 0.9])
    _, gradient = jax.jit(jax.value_and_grad(probability))(parameters)
    expected = jnp.array([0, -jnp.pi * jnp.sin(jnp.pi * 0.3) / 2, 0])
    assert jnp.allclose(gradient, expected)
