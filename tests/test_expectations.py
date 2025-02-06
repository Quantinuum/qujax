from jax import config, grad, jit
from jax import numpy as jnp
from jax import random

import qujax


def test_pauli_hermitian():
    for p_str in ("X", "Y", "Z"):
        qujax.check_hermitian(p_str)
        qujax.check_hermitian(qujax.gates.__dict__[p_str])


def test_single_expectation():
    Z = qujax.gates.Z

    st1 = jnp.zeros((2, 2, 2))
    st2 = jnp.zeros((2, 2, 2))
    st1 = st1.at[(0, 0, 0)].set(1.0)
    st2 = st2.at[(1, 0, 0)].set(1.0)
    dt1 = qujax.statetensor_to_densitytensor(st1)
    dt2 = qujax.statetensor_to_densitytensor(st2)
    ZZ = jnp.kron(Z, Z).reshape(2, 2, 2, 2)

    est1 = qujax.statetensor_to_single_expectation(st1, ZZ, [0, 1])
    est2 = qujax.statetensor_to_single_expectation(st2, ZZ, [0, 1])
    edt1 = qujax.densitytensor_to_single_expectation(dt1, ZZ, [0, 1])
    edt2 = qujax.densitytensor_to_single_expectation(dt2, ZZ, [0, 1])

    assert est1.item() == edt1.item() == 1
    assert est2.item() == edt2.item() == -1


def test_bitstring_expectation():
    n_qubits = 4

    gates = (
        ["H"] * n_qubits
        + ["Ry"] * n_qubits
        + ["Rz"] * n_qubits
        + ["CX"] * (n_qubits - 1)
        + ["Ry"] * n_qubits
        + ["Rz"] * n_qubits
    )
    qubits = (
        [[i] for i in range(n_qubits)] * 3
        + [[i, i + 1] for i in range(n_qubits - 1)]
        + [[i] for i in range(n_qubits)] * 2
    )
    param_inds = (
        [[]] * n_qubits
        + [[i] for i in range(n_qubits * 2)]
        + [[]] * (n_qubits - 1)
        + [[i] for i in range(n_qubits * 2, n_qubits * 4)]
    )

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds)

    n_params = n_qubits * 4
    params = random.uniform(random.PRNGKey(0), shape=(n_params,))

    costs = random.normal(random.PRNGKey(1), shape=(2**n_qubits,))

    def st_to_expectation(statetensor):
        probs = jnp.square(jnp.abs(statetensor.flatten()))
        return jnp.sum(costs * probs)

    param_to_expectation = lambda p: st_to_expectation(param_to_st(p))

    def brute_force_param_to_exp(p):
        sv = param_to_st(p).flatten()
        return jnp.dot(sv, jnp.diag(costs) @ sv.conj()).real

    true_expectation = brute_force_param_to_exp(params)

    expectation = param_to_expectation(params)
    expectation_jit = jit(param_to_expectation)(params)

    assert expectation.shape == ()
    assert expectation.dtype.name[:5] == "float"
    assert jnp.isclose(true_expectation, expectation)
    assert jnp.isclose(true_expectation, expectation_jit)

    true_expectation_grad = grad(brute_force_param_to_exp)(params)
    expectation_grad = grad(param_to_expectation)(params)
    expectation_grad_jit = jit(grad(param_to_expectation))(params)

    assert expectation_grad.shape == (n_params,)
    assert expectation_grad.dtype.name[:5] == "float"
    assert jnp.allclose(true_expectation_grad, expectation_grad, atol=1e-5)
    assert jnp.allclose(true_expectation_grad, expectation_grad_jit, atol=1e-5)


def _test_hermitian_observable(
    hermitian_str_seq_seq, qubit_inds_seq, coefs, st_in=None
):
    n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    if st_in is None:
        state = (
            random.uniform(random.PRNGKey(2), shape=(2**n_qubits,)) * 2
            + 1.0j * random.uniform(random.PRNGKey(1), shape=(2**n_qubits,)) * 2
        )
        state /= jnp.linalg.norm(state)
        st_in = state.reshape((2,) * n_qubits)

    dt_in = qujax.statetensor_to_densitytensor(st_in)

    st_to_exp = qujax.get_statetensor_to_expectation_func(
        hermitian_str_seq_seq, qubit_inds_seq, coefs
    )
    dt_to_exp = qujax.get_densitytensor_to_expectation_func(
        hermitian_str_seq_seq, qubit_inds_seq, coefs
    )

    def big_hermitian_matrix(hermitian_str_seq, qubit_inds):
        qubit_arrs = [getattr(qujax.gates, s) for s in hermitian_str_seq]
        hermitian_arrs = []
        j = 0
        for i in range(n_qubits):
            if i in qubit_inds:
                hermitian_arrs.append(qubit_arrs[j])
                j += 1
            else:
                hermitian_arrs.append(jnp.eye(2))

        big_h = hermitian_arrs[0]
        for k in range(1, n_qubits):
            big_h = jnp.kron(big_h, hermitian_arrs[k])
        return big_h

    sum_big_hs = jnp.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in range(len(hermitian_str_seq_seq)):
        sum_big_hs += coefs[i] * big_hermitian_matrix(
            hermitian_str_seq_seq[i], qubit_inds_seq[i]
        )

    assert jnp.allclose(sum_big_hs, sum_big_hs.conj().T)

    sv = st_in.flatten()
    true_exp = jnp.dot(sv.conj(), sum_big_hs @ sv).real

    qujax_exp = st_to_exp(st_in)
    qujax_dt_exp = dt_to_exp(dt_in)
    qujax_exp_jit = jit(st_to_exp)(st_in)
    qujax_dt_exp_jit = jit(dt_to_exp)(dt_in)

    assert jnp.array(qujax_exp).shape == ()
    assert jnp.array(qujax_exp).dtype.name[:5] == "float"
    assert jnp.isclose(true_exp, qujax_exp)
    assert jnp.isclose(true_exp, qujax_dt_exp)
    assert jnp.isclose(true_exp, qujax_exp_jit)
    assert jnp.isclose(true_exp, qujax_dt_exp_jit)

    st_to_samp_exp = qujax.get_statetensor_to_sampled_expectation_func(
        hermitian_str_seq_seq, qubit_inds_seq, coefs
    )
    dt_to_samp_exp = qujax.get_densitytensor_to_sampled_expectation_func(
        hermitian_str_seq_seq, qubit_inds_seq, coefs
    )
    qujax_samp_exp = st_to_samp_exp(st_in, random.PRNGKey(1), 1000000)
    qujax_samp_exp_jit = jit(st_to_samp_exp, static_argnums=2)(
        st_in, random.PRNGKey(1), 1000000
    )
    qujax_samp_exp_dt = dt_to_samp_exp(dt_in, random.PRNGKey(1), 1000000)
    qujax_samp_exp_dt_jit = jit(dt_to_samp_exp, static_argnums=2)(
        dt_in, random.PRNGKey(1), 1000000
    )
    assert jnp.array(qujax_samp_exp).shape == ()
    assert jnp.array(qujax_samp_exp).dtype.name[:5] == "float"
    assert jnp.isclose(true_exp, qujax_samp_exp, rtol=1e-2)
    assert jnp.isclose(true_exp, qujax_samp_exp_jit, rtol=1e-2)
    assert jnp.isclose(true_exp, qujax_samp_exp_dt, rtol=1e-2)
    assert jnp.isclose(true_exp, qujax_samp_exp_dt_jit, rtol=1e-2)


def test_X():
    hermitian_str_seq_seq = ["X"]
    qubit_inds_seq = [[0]]
    coefs = [1]

    gates = ["H", "Rz"]
    qubit = [[0], [0]]
    param_ind = [[], [0]]
    st_in = qujax.get_params_to_statetensor_func(gates, qubit, param_ind)(0.3)

    _test_hermitian_observable(hermitian_str_seq_seq, qubit_inds_seq, coefs, st_in)


def test_Y():
    n_qubits = 1

    hermitian_str_seq_seq = ["Y"] * n_qubits
    qubit_inds_seq = [[i] for i in range(n_qubits)]
    coefs = jnp.ones(len(hermitian_str_seq_seq))

    _test_hermitian_observable(hermitian_str_seq_seq, qubit_inds_seq, coefs)


def test_Z():
    n_qubits = 1

    hermitian_str_seq_seq = ["Z"] * n_qubits
    qubit_inds_seq = [[i] for i in range(n_qubits)]
    coefs = random.normal(random.PRNGKey(0), shape=(len(hermitian_str_seq_seq),))

    _test_hermitian_observable(hermitian_str_seq_seq, qubit_inds_seq, coefs)


def test_XYZ():
    n_qubits = 1

    hermitian_str_seq_seq = ["X", "Y", "Z"] * n_qubits
    qubit_inds_seq = [[i] for _ in range(3) for i in range(n_qubits)]
    coefs = random.normal(random.PRNGKey(0), shape=(len(hermitian_str_seq_seq),))

    _test_hermitian_observable(hermitian_str_seq_seq, qubit_inds_seq, coefs)


def test_ZZ_Y():
    config.update("jax_enable_x64", True)  # Run this test with 64 bit precision

    n_qubits = 4

    hermitian_str_seq_seq = [["Z", "Z"]] * (n_qubits - 1) + [["Y"]] * n_qubits
    qubit_inds_seq = [[i, i + 1] for i in range(n_qubits - 1)] + [
        [i] for i in range(n_qubits)
    ]
    coefs = random.normal(random.PRNGKey(1), shape=(len(hermitian_str_seq_seq),))

    _test_hermitian_observable(hermitian_str_seq_seq, qubit_inds_seq, coefs)


def test_sampling():
    target_pmf = jnp.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    target_pmf /= target_pmf.sum()

    target_st = jnp.sqrt(target_pmf).reshape(
        (2,) * int(jnp.rint(jnp.log2(target_pmf.size)))
    )

    n_samps = 7

    sample_ints = qujax.sample_integers(random.PRNGKey(0), target_st, n_samps)
    assert sample_ints.shape == (n_samps,)
    assert all(target_pmf[sample_ints] > 0)

    sample_bitstrings = qujax.sample_bitstrings(random.PRNGKey(0), target_st, n_samps)
    assert sample_bitstrings.shape == (
        n_samps,
        int(jnp.rint(jnp.log2(target_pmf.size))),
    )
    assert all(qujax.bitstrings_to_integers(sample_bitstrings) == sample_ints)
