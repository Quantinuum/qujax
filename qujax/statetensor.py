from __future__ import annotations

from functools import partial
from typing import Callable, Sequence, Optional

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
from jax._src.dtypes import canonicalize_dtype
from jax._src.typing import DTypeLike

from qujax import gates
from qujax.utils import _arrayify_inds, check_circuit

from qujax.typing import Gate, PureCircuitFunction, GateFunction, GateParameterIndices


def apply_gate(
    statetensor: jax.Array, gate_unitary: jax.Array, qubit_inds: Sequence[int]
) -> jax.Array:
    """
    Applies gate to statetensor and returns updated statetensor.
    Gate is represented by a unitary matrix in tensor form.

    Args:
        statetensor: Input statetensor.
        gate_unitary: Unitary array representing gate
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of indices for gate to be applied to.
            Must have 2 * len(qubit_inds) = gate_unitary.ndim

    Returns:
        Updated statetensor.

    Examples:
        >>> import qujax
        >>> from jax import numpy as jnp

        >>> state = qujax.all_zeros_statetensor(2)
        >>> state = qujax.apply_gate(state, qujax.gates.X, [1])
        >>> jnp.abs(state).tolist()
        [[0.0, 1.0], [0.0, 0.0]]
    """
    statetensor = jnp.tensordot(
        gate_unitary, statetensor, axes=(list(range(-len(qubit_inds), 0)), qubit_inds)
    )
    statetensor = jnp.moveaxis(statetensor, list(range(len(qubit_inds))), qubit_inds)
    return statetensor


def _to_gate_func(
    gate: Gate,
) -> GateFunction:
    """
    Ensures a gate_seq element is a function that map (possibly empty) parameters
    to a unitary tensor.

    Args:
        gate: Either a string matching an array or function in qujax.gates,
            a unitary array (which will be reshaped into a tensor of shape (2,2,2,...) )
            or a function taking parameters and returning gate unitary in tensor form.

    Returns:
        Gate parameter to unitary functions
    """

    def _array_to_callable(arr: jax.Array) -> Callable[[], jax.Array]:
        return lambda: arr

    if isinstance(gate, str):
        gate = gates.__dict__[gate]

    if callable(gate):
        gate_func = gate
    elif hasattr(gate, "__array__"):
        gate_func = _array_to_callable(jnp.array(gate))
    else:
        raise TypeError(
            f"Unsupported gate type - gate must be either a string in qujax.gates, an array or "
            f"callable: {gate}"
        )
    return gate_func


def _gate_func_to_unitary(
    gate_func: GateFunction,
    qubit_inds: Sequence[int],
    param_inds: jax.Array,
    params: jax.Array,
) -> jax.Array:
    """
    Extract gate unitary.

    Args:
        gate_func: Function that maps a (possibly empty) parameter array to a unitary tensor (array)
        qubit_inds: Indices of qubits to apply gate to
            (only needed to ensure gate is in tensor form)
        param_inds: Indices of full parameter to extract gate specific parameters
        params: Full parameter vector

    Returns:
        Array containing gate unitary in tensor form.
    """
    gate_params = jnp.take(params, param_inds)
    gate_unitary = gate_func(*gate_params)
    gate_unitary = gate_unitary.reshape(
        (2,) * (2 * len(qubit_inds))
    )  # Ensure gate is in tensor form
    return gate_unitary


def all_zeros_statetensor(n_qubits: int, dtype: DTypeLike = complex) -> jax.Array:
    """
    Returns a statetensor representation of the all-zeros state |00...0> on `n_qubits` qubits

    Args:
        n_qubits: Number of qubits that the state is defined on.
        dtype: Data type of the statetensor returned.

    Returns:
        Statetensor representing the state having all qubits set to zero.

    Examples:
        >>> import qujax
        >>> from jax import numpy as jnp

        >>> qujax.all_zeros_statetensor(2).shape
        (2, 2)
        >>> complex(qujax.all_zeros_statetensor(2)[0, 0])
        (1+0j)
    """
    statetensor = jnp.zeros((2,) * n_qubits, dtype=canonicalize_dtype(dtype))
    statetensor = statetensor.at[(0,) * n_qubits].set(1.0)
    return statetensor


def get_params_to_statetensor_func(
    gate_seq: Sequence[Gate],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[GateParameterIndices],
    n_qubits: Optional[int] = None,
) -> PureCircuitFunction:
    """
    Creates a function that maps circuit parameters to a statetensor.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching a unitary array or function in qujax.gates,
            a custom unitary array or a custom function taking parameters and returning a
            unitary array. Unitary arrays will be reshaped into tensor form (2, 2,...)
        qubit_inds_seq: Sequences of sequences representing qubit indices (ints) that gates are
            acting on.
            i.e. [[0], [0,1], [1]] tells qujax the first gate is a single qubit gate acting on the
            zeroth qubit, the second gate is a two qubit gate acting on the zeroth and first qubit
            etc.
        param_inds_seq: Sequence of sequences representing parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the zeroth parameter
            (the float at position zero in the parameter vector/array), the second gate is not
            parameterised and the third gate uses the parameters at position five and two.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps parameters (and optional statetensor_in) to a statetensor.
        If no parameters are found then the function only takes optional statetensor_in.


    Examples:
        >>> import qujax
        >>> from jax import numpy as jnp

        With no parameterised gates, call the returned function without parameters:

        >>> simulate = qujax.get_params_to_statetensor_func(["X"], [[0]], [[]])
        >>> jnp.abs(simulate()).tolist()
        [0.0, 1.0]

        A custom array and a custom callable can be mixed with named gates. Angles
        are in half turns; the callable receives the selected parameter values:

        >>> def custom_rotation(angle):
        ...     return qujax.gates.Ry(angle)
        >>> simulate = qujax.get_params_to_statetensor_func(
        ...     [jnp.eye(2), custom_rotation], [[0], [0]], [[], [0]])
        >>> bool(jnp.allclose(jnp.abs(simulate(jnp.array([1.0]))), jnp.array([0., 1.]), atol=1e-6))
        True

        Supply an initial state explicitly, or compile and differentiate the function:

        >>> import jax
        >>> initial = jnp.array([0., 1.], dtype=complex)
        >>> bool(jnp.allclose(jnp.abs(simulate(jnp.array([0.0]), initial)), jnp.array([0., 1.])))
        True
        >>> jax.jit(simulate)(jnp.array([0.25])).shape
        (2,)
        >>> jax.grad(lambda p: jnp.real(simulate(p)[0]))(jnp.array([0.25])).shape
        (1,)
    """

    check_circuit(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    gate_seq_callable = [_to_gate_func(g) for g in gate_seq]
    param_inds_array_seq = _arrayify_inds(param_inds_seq)

    def params_to_statetensor_func(
        params: ArrayLike, statetensor_in: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Applies parameterised circuit (series of gates) to a statetensor_in (default is |0>^N).

        Args:
            params: Parameters of the circuit.
            statetensor_in: Optional. Input statetensor.
                Defaults to |0>^N (tensor of size 2^n with all zeroes except one in [0]*N index).

        Returns:
            Updated statetensor.

        """
        if statetensor_in is None:
            statetensor = all_zeros_statetensor(n_qubits)
        else:
            statetensor = statetensor_in

        params = jnp.atleast_1d(params)
        # Guarantee `params` has the right type for type-checking purposes
        if not isinstance(params, jax.Array):
            raise ValueError("This should not happen. Please open an issue on GitHub.")

        for gate_func, qubit_inds, param_inds in zip(
            gate_seq_callable, qubit_inds_seq, param_inds_array_seq
        ):
            gate_unitary = _gate_func_to_unitary(
                gate_func, qubit_inds, param_inds, params
            )
            statetensor = apply_gate(statetensor, gate_unitary, qubit_inds)
        return statetensor

    non_parameterised = all([pi.size == 0 for pi in param_inds_array_seq])
    if non_parameterised:

        def no_params_to_statetensor_func(
            statetensor_in: Optional[jax.Array] = None,
        ) -> jax.Array:
            """
            Applies circuit (series of gates with no parameters) to a statetensor_in
            (default is |0>^N).

            Args:
                statetensor_in: Optional. Input statetensor.
                    Defaults to |0>^N (tensor of size 2^n with all zeroes except one in
                    the [0]*N index).

            Returns:
                Updated statetensor.

            """
            return params_to_statetensor_func(jnp.array([]), statetensor_in)

        return no_params_to_statetensor_func

    return params_to_statetensor_func


def get_params_to_unitarytensor_func(
    gate_seq: Sequence[Gate],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[GateParameterIndices],
    n_qubits: Optional[int] = None,
) -> PureCircuitFunction:
    """
    Creates a function that maps circuit parameters to a unitarytensor.
    The unitarytensor is an array with shape (2,) * 2 * n_qubits
    representing the full unitary matrix of the circuit.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching a unitary array or function in qujax.gates,
            a custom unitary array or a custom function taking parameters and returning a unitary
            array. Unitary arrays will be reshaped into tensor form (2, 2,...)
        qubit_inds_seq: Sequences of sequences representing qubit indices (ints) that gates are
            acting on.
            i.e. [[0], [0,1], [1]] tells qujax the first gate is a single qubit gate acting on the
            zeroth qubit, the second gate is a two qubit gate acting on the zeroth and first qubit
            etc.
        param_inds_seq: Sequence of sequences representing parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the zeroth parameter
            (the float at position zero in the parameter vector/array), the second gate is not
            parameterised and the third gate uses the parameters at position five and two.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps any parameters to a unitarytensor.


    Examples:
        >>> import qujax
        >>> from jax import numpy as jnp

        >>> unitary = qujax.get_params_to_unitarytensor_func(["CX"], [[0, 1]], [[]])
        >>> unitary().shape
        (2, 2, 2, 2)
        >>> jnp.abs(unitary().reshape(4, 4)).tolist()
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    """

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    param_to_st = get_params_to_statetensor_func(
        gate_seq, qubit_inds_seq, param_inds_seq, n_qubits
    )
    identity_unitarytensor = jnp.eye(2**n_qubits).reshape((2,) * 2 * n_qubits)
    return partial(param_to_st, statetensor_in=identity_unitarytensor)
