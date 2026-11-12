import jax
import jax.numpy as jnp
import pytest

import qujax


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("basis", range(4))
def test_sampled_basis_state_expectation(mixed, compiled, basis):
    state = jnp.zeros(4, dtype=complex).at[basis].set(1).reshape(2, 2)
    factory = (
        qujax.get_densitytensor_to_sampled_expectation_func
        if mixed
        else qujax.get_statetensor_to_sampled_expectation_func
    )
    expectation = factory([["Z"]], [[1]], [1.0])
    if compiled:
        expectation = jax.jit(expectation, static_argnums=2)
    tensor = qujax.statetensor_to_densitytensor(state) if mixed else state
    result = expectation(tensor, jax.random.PRNGKey(0), 128)
    assert jnp.isfinite(result)
    assert jnp.isclose(result, (-1) ** (basis % 2))


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_sampled_entangled_state_with_sparse_support(mixed, compiled):
    state = jnp.array([1, 0, 0, 1j]).reshape(2, 2) / jnp.sqrt(2)
    factory = (
        qujax.get_densitytensor_to_sampled_expectation_func
        if mixed
        else qujax.get_statetensor_to_sampled_expectation_func
    )
    expectation = factory([["Z", "Z"]], [[0, 1]], [1.0])
    if compiled:
        expectation = jax.jit(expectation, static_argnums=2)
    tensor = qujax.statetensor_to_densitytensor(state) if mixed else state
    result = expectation(tensor, jax.random.PRNGKey(1), 128)
    assert jnp.isfinite(result)
    assert jnp.isclose(result, 1.0)
