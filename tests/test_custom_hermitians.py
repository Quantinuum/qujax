import jax.numpy as jnp
import pytest

import qujax


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
@pytest.mark.parametrize("tensor_form", [False, True])
def test_custom_multiqubit_observable(n_qubits, tensor_form):
    dim = 2**n_qubits
    diagonal = jnp.arange(dim, dtype=float)
    observable = jnp.diag(diagonal)
    if tensor_form:
        observable = observable.reshape((2,) * (2 * n_qubits))

    qujax.check_hermitian(observable, check_z_commutes=True)
    state = qujax.all_zeros_statetensor(n_qubits).at[(1,) * n_qubits].set(1)
    state = state.at[(0,) * n_qubits].set(0)
    density = qujax.statetensor_to_densitytensor(state)
    qubits = [list(range(n_qubits))]
    pure_expectation = qujax.get_statetensor_to_expectation_func(
        [[observable]], qubits, [1.0]
    )
    mixed_expectation = qujax.get_densitytensor_to_expectation_func(
        [[observable]], qubits, [1.0]
    )
    assert jnp.isclose(pure_expectation(state), dim - 1)
    assert jnp.isclose(mixed_expectation(density), dim - 1)


@pytest.mark.parametrize("tensor_form", [False, True])
def test_reject_nonhermitian_multiqubit_observable(tensor_form):
    observable = jnp.eye(8, dtype=complex).at[0, 7].set(1j)
    if tensor_form:
        observable = observable.reshape((2,) * 6)
    with pytest.raises(TypeError, match="Array not Hermitian"):
        qujax.check_hermitian(observable)
