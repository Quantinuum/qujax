from __future__ import annotations

import collections.abc
from inspect import signature
from typing import Callable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import jax
from jax.typing import ArrayLike
from jax import numpy as jnp
from jax import random

from qujax import gates

from qujax.typing import (
    Gate,
    KrausOp,
    GateParameterIndices,
    PureParameterizedCircuit,
    MixedParameterizedCircuit,
)

paulis = {"X": gates.X, "Y": gates.Y, "Z": gates.Z}


def check_unitary(gate: Gate) -> None:
    """
    Checks whether a qujax Gate is unitary.
    Throws a TypeError if this is found not to be the case.

    Args:
        gate: array containing potentially unitary string, array
            or function (which will be evaluated with all arguments set to 0.1).

    """
    if isinstance(gate, str):
        if gate in gates.__dict__:
            gate = gates.__dict__[gate]
        else:
            raise KeyError(
                f"Gate string '{gate}' not found in qujax.gates "
                f"- consider changing input to an array or callable"
            )

    if callable(gate):
        num_args = len(signature(gate).parameters)
        gate_arr = gate(*jnp.ones(num_args) * 0.1)
    elif isinstance(gate, jax.Array):
        gate_arr = gate
    else:
        raise TypeError(
            f"Unsupported gate type - gate must be either a string in qujax.gates, an array or "
            f"callable: {gate}"
        )

    gate_square_dim = int(jnp.sqrt(gate_arr.size))
    gate_arr = gate_arr.reshape(gate_square_dim, gate_square_dim)

    if jnp.any(
        jnp.abs(gate_arr @ jnp.conjugate(gate_arr).T - jnp.eye(gate_square_dim)) > 1e-3
    ):
        raise TypeError(f"Gate not unitary: {gate}")


def check_hermitian(hermitian: Union[str, jax.Array], check_z_commutes: bool = False):
    """
    Checks whether a matrix or tensor is Hermitian.

    Args:
        hermitian: array containing potentially Hermitian matrix or tensor
        check_z_commutes: boolean on whether to check if the matrix commutes with Z

    """
    if isinstance(hermitian, str):
        if hermitian not in paulis:
            raise TypeError(
                f"qujax only accepts {tuple(paulis.keys())} as Hermitian strings,"
                "received: {hermitian}"
            )
        n_qubits = 1
        hermitian_mat = paulis[hermitian]

    else:
        n_qubits = hermitian.ndim // 2
        hermitian_mat = hermitian.reshape(2 * n_qubits, 2 * n_qubits)
        if not jnp.allclose(hermitian_mat, hermitian_mat.T.conj()):
            raise TypeError(f"Array not Hermitian: {hermitian}")

    if check_z_commutes:
        big_z = jnp.diag(jnp.where(jnp.arange(2**n_qubits) % 2 == 0, 1, -1))
        z_commutes = jnp.allclose(hermitian_mat @ big_z, big_z @ hermitian_mat)
        if not z_commutes:
            warn(
                "Hermitian matrix does not commute with Z. \n"
                "For sampled expectation values, this may lead to unexpected results, "
                "measurements on a quantum device are always taken in the computational basis. "
                "Additional gates can be applied in the circuit to change the basis such "
                "that an observable that commutes with Z can be measured."
            )


def _arrayify_inds(
    param_inds_seq: Optional[Sequence[GateParameterIndices]],
) -> Sequence[jax.Array]:
    """
    Ensure each element of param_inds_seq is an array (and therefore valid for jnp.take)

    Args:
        param_inds_seq: Sequence of sequences representing parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the zeroth parameter
            (the float at position zero in the parameter vector/array), the second gate is not
            parameterised and the third gates used the parameters at position five and two.

    Returns:
        Sequence of arrays representing parameter indices.
    """
    if param_inds_seq is None:
        param_inds_seq = [None]
    array_param_inds = [jnp.array([] if p is None else p) for p in param_inds_seq]
    array_param_inds = [
        jnp.array([]) if jnp.any(jnp.isnan(p)) else p.astype(int)
        for p in array_param_inds
    ]
    return array_param_inds


def check_circuit(
    gate_seq: Sequence[KrausOp],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[
        Union[GateParameterIndices, Sequence[GateParameterIndices]]
    ],
    n_qubits: Optional[int] = None,
    check_unitaries: bool = True,
):
    """
    Basic checks that circuit arguments conform.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching an array or function in qujax.gates,
            a unitary array (which will be reshaped into a tensor of shape (2,2,2,...) )
            or a function taking parameters and returning gate unitary in tensor form.
            Or alternatively a sequence of the above representing Kraus operators.
        qubit_inds_seq: Sequences of qubits (ints) that gates are acting on.
        param_inds_seq: Sequence of parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the first parameter,
            the second gate is not parameterised and the third gates used the fifth and second
            parameters.
        n_qubits: Number of qubits, if fixed.
        check_unitaries: boolean on whether to check if each gate represents a unitary matrix

    """
    if not isinstance(gate_seq, collections.abc.Sequence):
        raise TypeError("gate_seq must be Sequence e.g. ['H', 'Rx', 'CX']")

    if (not isinstance(qubit_inds_seq, collections.abc.Sequence)) or (
        any(
            [
                not (isinstance(q, collections.abc.Sequence) or hasattr(q, "__array__"))
                for q in qubit_inds_seq
            ]
        )
    ):
        raise TypeError(
            "qubit_inds_seq must be Sequence of Sequences e.g. [[0,1], [0], []]"
        )

    if (not isinstance(param_inds_seq, collections.abc.Sequence)) or (
        any(
            [
                not (
                    isinstance(p, collections.abc.Sequence)
                    or hasattr(p, "__array__")
                    or p is None
                )
                for p in param_inds_seq
            ]
        )
    ):
        raise TypeError(
            "param_inds_seq must be Sequence of Sequences e.g. [[0,1], [0], []]"
        )

    if len(gate_seq) != len(qubit_inds_seq) or len(param_inds_seq) != len(gate_seq):
        raise TypeError(
            f"gate_seq ({len(gate_seq)}), qubit_inds_seq ({len(qubit_inds_seq)})"
            f"and param_inds_seq ({len(param_inds_seq)}) must have matching lengths"
        )

    if n_qubits is not None and n_qubits < max([max(qi) for qi in qubit_inds_seq]) + 1:
        raise TypeError(
            "n_qubits must be larger than largest qubit index in qubit_inds_seq"
        )

    if check_unitaries:
        for g in gate_seq:
            check_unitary(g)


def _get_gate_str(
    gate_obj: KrausOp,
    param_inds: Union[GateParameterIndices, Sequence[GateParameterIndices]],
) -> str:
    """
    Maps single gate object to a four character string representation

    Args:
        gate_obj: Either a string matching a function in qujax.gates,
            a unitary array (which will be reshaped into a tensor of shape e.g. (2,2,2,...) )
            or a function taking parameters (can be empty) and returning gate unitary
            in tensor form.
            Or alternatively, a sequence of Krause operators represented by strings, arrays or
            functions.
        param_inds: Parameter indices that gates are using, i.e. gate uses 1st and 5th parameter.

    Returns:
        Four character string representation of the gate

    """
    if isinstance(gate_obj, (tuple, list)) or (
        hasattr(gate_obj, "__array__") and gate_obj.ndim % 2 == 1
    ):
        # Kraus operators
        gate_obj = "Kr"
        param_inds = jnp.unique(jnp.concatenate(_arrayify_inds(param_inds), axis=0))

    if isinstance(gate_obj, str):
        gate_str = gate_obj
    elif hasattr(gate_obj, "__array__"):
        gate_str = "Arr"
    elif callable(gate_obj):
        gate_str = "Func"
    else:
        if hasattr(gate_obj, "__name__"):
            gate_str = gate_obj.__name__
        elif hasattr(gate_obj, "__class__") and hasattr(gate_obj.__class__, "__name__"):
            gate_str = gate_obj.__class__.__name__
        else:
            gate_str = "Other"

    if hasattr(param_inds, "tolist"):
        param_inds = param_inds.tolist()

    if isinstance(param_inds, tuple):
        param_inds = list(param_inds)

    if param_inds == [] or param_inds == [None] or param_inds is None:
        if len(gate_str) > 7:
            gate_str = gate_str[:6] + "."
    else:
        param_str = str(param_inds).replace(" ", "")

        if len(param_str) > 5:
            param_str = "[.]"

        if (len(gate_str) + len(param_str)) > 7:
            gate_str = gate_str[:1] + "."

        gate_str += param_str

    gate_str = gate_str.center(7, "-")

    return gate_str


def _pad_rows(rows: List[str]) -> Tuple[List[str], List[bool]]:
    """
    Pad string representation of circuit to be rectangular.
    Fills qubit rows with '-' and between-qubit rows with ' '.

    Args:
        rows: String representation of circuit

    Returns:
        Rectangular string representation of circuit with right padding.

    """

    max_len = max([len(r) for r in rows])

    def extend_row(row: str, qubit_row: bool) -> str:
        lr = len(row)
        if lr < max_len:
            if qubit_row:
                row += "-" * (max_len - lr)
            else:
                row += " " * (max_len - lr)
        return row

    out_rows = [extend_row(r, i % 2 == 0) for i, r in enumerate(rows)]
    return out_rows, [True] * len(rows)


def print_circuit(
    gate_seq: Sequence[KrausOp],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[
        Union[GateParameterIndices, Sequence[GateParameterIndices]]
    ],
    n_qubits: Optional[int] = None,
    qubit_min: int = 0,
    qubit_max: Optional[int] = None,
    gate_ind_min: int = 0,
    gate_ind_max: Optional[int] = None,
    sep_length: int = 1,
) -> List[str]:
    """
    Returns and prints basic string representation of circuit.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching an array or function in qujax.gates,
            a unitary array (which will be reshaped into a tensor of shape (2,2,2,...) )
            or a function taking parameters and returning gate unitary in tensor form.
            Or alternatively a sequence of the above representing Kraus operators.
        qubit_inds_seq: Sequences of qubits (ints) that gates are acting on.
        param_inds_seq: Sequence of parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the first parameter,
            the second gate is not parameterised and the third gates used the fifth and
            second parameters.
        n_qubits: Number of qubits, if fixed.
        qubit_min: Index of first qubit to display.
        qubit_max: Index of final qubit to display.
        gate_ind_min: Index of gate to start circuit printing.
        gate_ind_max: Index of gate to stop circuit printing.
        sep_length: Number of dashes to separate gates.

    Returns:
        String representation of circuit

    """
    check_circuit(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits, False)

    if gate_ind_max is None:
        gate_ind_max = len(gate_seq) - 1
    else:
        gate_ind_max = min(len(gate_seq) - 1, gate_ind_max)

    if gate_ind_min > gate_ind_max:
        raise TypeError("gate_ind_max must be larger or equal to gate_ind_min")

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    if qubit_max is None:
        qubit_max = n_qubits - 1
    else:
        qubit_max = min(n_qubits - 1, qubit_max)

    if qubit_min > qubit_max:
        raise TypeError("qubit_max must be larger or equal to qubit_min")

    gate_str_seq = [_get_gate_str(g, p) for g, p in zip(gate_seq, param_inds_seq)]

    n_qubits_disp = qubit_max - qubit_min + 1

    rows = [f"q{qubit_min}: ".ljust(3) + "-" * sep_length]
    if n_qubits_disp > 1:
        for i in range(qubit_min + 1, qubit_max + 1):
            rows += [" ", f"q{i}: ".ljust(3) + "-" * sep_length]
    rows, rows_free = _pad_rows(rows)

    for gate_ind in range(gate_ind_min, gate_ind_max + 1):
        g = gate_str_seq[gate_ind]
        qi = qubit_inds_seq[gate_ind]

        qi_min = min(qi)
        qi_max = max(qi)
        ri_min = 2 * qi_min  # index of top row used by gate
        ri_max = 2 * qi_max  # index of bottom row used by gate

        if not all([rows_free[i] for i in range(ri_min, ri_max)]):
            rows, rows_free = _pad_rows(rows)

        for row_ind in range(ri_min, ri_max + 1):
            if row_ind == 2 * qi[-1]:
                rows[row_ind] += "-" * sep_length + g
            elif row_ind % 2 == 1:
                rows[row_ind] += " " * sep_length + "   " + "|" + "   "
            elif row_ind / 2 in qi:
                rows[row_ind] += "-" * sep_length + "---" + "◯" + "---"
            else:
                rows[row_ind] += "-" * sep_length + "---" + "|" + "---"

            rows_free[row_ind] = False

    rows, _ = _pad_rows(rows)

    for p in rows:
        print(p)

    return rows


def integers_to_bitstrings(
    integers: Union[int, jax.Array], nbits: Optional[int] = None
) -> jax.Array:
    """
    Convert integer or array of integers into their binary expansion(s).

    Args:
        integers: Integer or array of integers to be converted.
        nbits: Length of output binary expansion.
            Defaults to smallest possible.

    Returns:
        Array of binary expansion(s).
    """
    integers = jnp.atleast_1d(integers)
    # Guarantee `bitstrings` has the right type for type-checking purposes
    if not isinstance(integers, jax.Array):
        raise ValueError("This should not happen. Please open an issue on GitHub.")

    if nbits is None:
        nbits = int(jnp.ceil(jnp.log2(jnp.maximum(integers.max(), 1)) + 1e-5).item())

    return jnp.squeeze(
        ((integers[:, None] & (1 << jnp.arange(nbits - 1, -1, -1))) > 0).astype(int)
    )


def bitstrings_to_integers(bitstrings: ArrayLike) -> jax.Array:
    """
    Convert binary expansion(s) into integers.

    Args:
        bitstrings: Bitstring array or array of bitstring arrays.

    Returns:
        Array of integers.
    """
    bitstrings = jnp.atleast_2d(bitstrings)

    # Guarantee `bitstrings` has the right type for type-checking purposes
    if not isinstance(bitstrings, jax.Array):
        raise ValueError("This should not happen. Please open an issue on GitHub.")

    convarr = 2 ** jnp.arange(bitstrings.shape[-1] - 1, -1, -1)
    return jnp.squeeze(bitstrings.dot(convarr)).astype(int)


def sample_integers(
    random_key: random.PRNGKeyArray,
    statetensor: jax.Array,
    n_samps: int = 1,
) -> jax.Array:
    """
    Generate random integer samples according to statetensor.

    Args:
        random_key: JAX random key to seed samples.
        statetensor: Statetensor encoding sampling probabilities (in the form of amplitudes).
        n_samps: Number of samples to generate. Defaults to 1.

    Returns:
        Array with sampled integers, shape=(n_samps,).

    """
    sv_probs = jnp.square(jnp.abs(statetensor.flatten()))
    sampled_inds = random.choice(
        random_key, a=jnp.arange(statetensor.size), shape=(n_samps,), p=sv_probs
    )
    return sampled_inds


def sample_bitstrings(
    random_key: random.PRNGKeyArray,
    statetensor: jax.Array,
    n_samps: int = 1,
) -> jax.Array:
    """
    Generate random bitstring samples according to statetensor.

    Args:
        random_key: JAX random key to seed samples.
        statetensor: Statetensor encoding sampling probabilities (in the form of amplitudes).
        n_samps: Number of samples to generate. Defaults to 1.

    Returns:
        Array with sampled bitstrings, shape=(n_samps, statetensor.ndim).
    """
    return integers_to_bitstrings(
        sample_integers(random_key, statetensor, n_samps), statetensor.ndim
    )


def statetensor_to_densitytensor(statetensor: jax.Array) -> jax.Array:
    """
    Computes a densitytensor representation of a pure quantum state
    from its statetensor representaton

    Args:
        statetensor: Input statetensor.

    Returns:
        A densitytensor representing the quantum state.
    """
    n_qubits = statetensor.ndim
    st = statetensor
    dt = (st.reshape(-1, 1) @ st.reshape(1, -1).conj()).reshape(
        2 for _ in range(2 * n_qubits)
    )
    return dt


def repeat_circuit(
    circuit: Union[PureParameterizedCircuit, MixedParameterizedCircuit],
    nr_of_parameters: int,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Repeats circuit encoded by `circuit` an arbitrary number of times.
    Avoids compilation overhead with increasing circuit depth.

    Args:
        circuit: The function encoding the circuit.
        nr_of_parameters: The number of parameters that `circuit` takes.

    Returns:
        A function taking an arbitrary number of parameters and returning as
        many applications of `circuit` as the number of parameters allows.
        An exception is thrown if this function is supplied with a parameter array
        of size not divisible by `nr_of_parameters`.
    """

    def repeated_circuit(params: jax.Array, statetensor_in: jax.Array) -> jax.Array:
        def f(state, p):
            return circuit(p, state), None

        reshaped_parameters = params.reshape(-1, nr_of_parameters)
        result, _ = jax.lax.scan(f, statetensor_in, reshaped_parameters)
        return result

    return repeated_circuit
