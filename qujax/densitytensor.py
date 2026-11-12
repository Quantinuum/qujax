from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union, Optional

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
from jax.lax import scan
from jax._src.dtypes import canonicalize_dtype
from jax._src.typing import DTypeLike
from qujax.statetensor import (
    _arrayify_inds,
    _gate_func_to_unitary,
    _to_gate_func,
    apply_gate,
)
from qujax.utils import _required_params, check_circuit
from qujax.typing import (
    MixedCircuitFunction,
    KrausOp,
    GateFunction,
    GateParameterIndices,
)


def _kraus_single(
    densitytensor: jax.Array, array: jax.Array, qubit_inds: Sequence[int]
) -> jax.Array:
    r"""
    Performs single Kraus operation

    .. math::

        \rho_\text{out} = B \rho_\text{in} B^{\dagger}

    Args:
        densitytensor: Input density matrix of shape=(2, 2, ...) and ndim=2*n_qubits
        array: Array containing the Kraus operator (in tensor form).
        qubit_inds: Sequence of qubit indices on which to apply the Kraus operation.

    Returns:
        Updated density matrix.
    """
    n_qubits = densitytensor.ndim // 2
    densitytensor = apply_gate(densitytensor, array, qubit_inds)
    densitytensor = apply_gate(
        densitytensor, array.conj(), [n_qubits + i for i in qubit_inds]
    )
    return densitytensor


def kraus(
    densitytensor: jax.Array, arrays: Iterable[jax.Array], qubit_inds: Sequence[int]
) -> jax.Array:
    r"""
    Performs Kraus operation.

    .. math::
        \rho_\text{out} = \sum_i B_i \rho_\text{in} B_i^{\dagger}

    Args:
        densitytensor: Input density matrix of shape=(2, 2, ...) and ndim=2*n_qubits
        arrays: Sequence of arrays containing the Kraus operators (in tensor form).
        qubit_inds: Sequence of qubit indices on which to apply the Kraus operation.

    Returns:
        Updated density matrix.
    """
    arrays = jnp.array(arrays)
    if arrays.ndim % 2 == 0:
        arrays = arrays[jnp.newaxis]
        # ensure first dimensions indexes different kraus operators
    arrays = arrays.reshape((arrays.shape[0],) + (2,) * 2 * len(qubit_inds))

    new_densitytensor, _ = scan(
        lambda dt, arr: (dt + _kraus_single(densitytensor, arr, qubit_inds), None),
        init=jnp.zeros_like(densitytensor) * 0.0j,
        xs=arrays,
    )
    # new_densitytensor = vmap(_kraus_single, in_axes=(None, 0, None))(
    #     densitytensor, arrays, qubit_inds
    # ).sum(0)
    return new_densitytensor


def _to_kraus_operator_seq_funcs(
    kraus_op: KrausOp,
    param_inds: Optional[Union[GateParameterIndices, Sequence[GateParameterIndices]]],
) -> Tuple[Sequence[GateFunction], Sequence[jax.Array]]:
    """
    Ensures Kraus operators are a sequence of functions that map (possibly empty) parameters to
    tensors and that each element of param_inds_seq is a sequence of arrays that correspond to the
    parameter indices of each Kraus operator.

    Args:
        kraus_op: Either a normal Gate or a sequence of Gates representing Kraus operators.
        param_inds: If kraus_op is a normal Gate then a sequence of parameter indices,
            if kraus_op is a sequence of Kraus operators then a sequence of sequences of
            parameter indices

    Returns:
        Tuple containing sequence of functions mapping to Kraus operators
        and sequence of arrays with parameter indices

    """
    if isinstance(kraus_op, (list, tuple)):
        kraus_op_funcs = [_to_gate_func(ko) for ko in kraus_op]
        if param_inds is None:
            param_inds = [None for _ in kraus_op]
    elif isinstance(kraus_op, (str, jax.Array)) or callable(kraus_op):
        kraus_op_funcs = [_to_gate_func(kraus_op)]
        param_inds = [param_inds]
    else:
        raise ValueError(f"Invalid Kraus operator specification: {kraus_op}")
    return kraus_op_funcs, _arrayify_inds(param_inds)


def partial_trace(
    densitytensor: jax.Array, indices_to_trace: Sequence[int]
) -> jax.Array:
    """
    Traces out (discards) specified qubits, resulting in a densitytensor
    representing the mixed quantum state on the remaining qubits.

    Args:
        densitytensor: Input densitytensor.
        indices_to_trace: Indices of qubits to trace out/discard.

    Returns:
        Resulting densitytensor on remaining qubits.

    """
    n_qubits = densitytensor.ndim // 2
    einsum_indices = list(range(densitytensor.ndim))
    for i in indices_to_trace:
        einsum_indices[i + n_qubits] = einsum_indices[i]
    densitytensor = jnp.einsum(densitytensor, einsum_indices)
    return densitytensor


def all_zeros_densitytensor(n_qubits: int, dtype: DTypeLike = complex) -> jax.Array:
    """
    Returns a densitytensor representation of the all-zeros state |00...0> on `n_qubits` qubits

    Args:
        n_qubits: Number of qubits that the state is defined on.
        dtype: Data type of the densitytensor returned.

    Returns:
        Densitytensor representing the state having all qubits set to zero.
    """
    densitytensor = jnp.zeros((2,) * 2 * n_qubits, canonicalize_dtype(dtype))
    densitytensor = densitytensor.at[(0,) * 2 * n_qubits].set(1.0)
    return densitytensor


def get_params_to_densitytensor_func(
    kraus_ops_seq: Sequence[KrausOp],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[
        Union[GateParameterIndices, Sequence[GateParameterIndices]]
    ],
    n_qubits: Optional[int] = None,
) -> MixedCircuitFunction:
    """
    Creates a function that maps circuit parameters to a density tensor (a density matrix in
    tensor form).
    densitytensor = densitymatrix.reshape((2,) * 2 * n_qubits)
    densitymatrix = densitytensor.reshape(2 ** n_qubits, 2 ** n_qubits)

    Args:
        kraus_ops_seq: Sequence of gates.
            Each element is either a string matching a unitary array or function in qujax.gates,
            a custom unitary array or a custom function taking parameters and returning a unitary
            array. Unitary arrays will be reshaped into tensor form (2, 2,...)
        qubit_inds_seq: Sequences of sequences representing qubit indices (ints) that gates are
            acting on.
            i.e. [[0], [0,1], [1]] tells qujax the first gate is a single qubit gate acting on the
            zeroth qubit, the second gate is a two qubit gate acting on the zeroth and
            first qubit etc.
        param_inds_seq: Sequence of sequences representing parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the zeroth parameter
            (the float at position zero in the parameter vector/array), the second gate is not
            parameterised and the third gate uses the parameters at position five and two.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps parameters (and optional densitytensor_in) to a densitytensor.
        If no parameters are found then the function only takes optional densitytensor_in.

    """

    check_circuit(kraus_ops_seq, qubit_inds_seq, param_inds_seq, n_qubits, False)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    kraus_ops_seq_callable_and_param_inds = [
        _to_kraus_operator_seq_funcs(ko, param_inds)
        for ko, param_inds in zip(kraus_ops_seq, param_inds_seq)
    ]
    kraus_ops_seq_callable = [
        ko_pi[0] for ko_pi in kraus_ops_seq_callable_and_param_inds
    ]
    param_inds_array_seq = [ko_pi[1] for ko_pi in kraus_ops_seq_callable_and_param_inds]
    n_required_params = _required_params(
        [indices for op_indices in param_inds_array_seq for indices in op_indices]
    )

    def params_to_densitytensor_func(
        params: ArrayLike, densitytensor_in: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Applies parameterised circuit (series of gates) to a densitytensor_in
        (default is |0>^N <0|^N).

        Args:
            params: Parameters of the circuit.
            densitytensor_in: Optional. Input densitytensor.
                Defaults to |0>^N <0|^N (tensor of size 2^(2*N) with all zeroes except one in
                the [0]*(2*N) index).

        Returns:
            Updated densitytensor.

        """
        if densitytensor_in is None:
            densitytensor = all_zeros_densitytensor(n_qubits)
        else:
            densitytensor = densitytensor_in
        params = jnp.atleast_1d(params)
        if params.size < n_required_params:
            raise ValueError(
                f"Circuit requires at least {n_required_params} parameters; "
                f"received {params.size}"
            )
        # Guarantee `params` has the right type for type-checking purposes
        if not isinstance(params, jax.Array):
            raise ValueError("This should not happen. Please open an issue on GitHub.")
        for gate_func_single_seq, qubit_inds, param_inds_single_seq in zip(
            kraus_ops_seq_callable, qubit_inds_seq, param_inds_array_seq
        ):
            kraus_operators = [
                _gate_func_to_unitary(gf, qubit_inds, pi, params)
                for gf, pi in zip(gate_func_single_seq, param_inds_single_seq)
            ]
            densitytensor = kraus(densitytensor, kraus_operators, qubit_inds)
        return densitytensor

    non_parameterised = all(
        [all([pi.size == 0 for pi in pi_seq]) for pi_seq in param_inds_array_seq]
    )
    if non_parameterised:

        def no_params_to_densitytensor_func(
            densitytensor_in: Optional[jax.Array] = None,
        ) -> jax.Array:
            """
            Applies circuit (series of gates with no parameters) to a densitytensor_in
            (default is |0>^N <0|^N).

            Args:
                densitytensor_in: Optional. Input densitytensor.
                    Defaults to |0>^N <0|^N (tensor of size 2^(2*N) with all zeroes except one in
                    the [0]*(2*N) index).

            Returns:
                Updated densitytensor.

            """
            return params_to_densitytensor_func(jnp.array([]), densitytensor_in)

        return no_params_to_densitytensor_func

    return params_to_densitytensor_func
