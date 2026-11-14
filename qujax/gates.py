import jax
from jax import numpy as jnp

I = jnp.eye(2)

_0 = jnp.zeros((2, 2))

X = jnp.array([[0.0, 1.0], [1.0, 0.0]])

Y = jnp.array([[0.0, -1.0j], [1.0j, 0.0]])

Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])

H = jnp.array([[1.0, 1.0], [1.0, -1]]) / jnp.sqrt(2)

S = jnp.array([[1.0, 0.0], [0.0, 1.0j]])

Sdg = jnp.array([[1.0, 0.0], [0.0, -1.0j]])

T = jnp.array([[1.0, 0.0], [0.0, jnp.exp(jnp.pi * 1.0j / 4)]])

Tdg = jnp.array([[1.0, 0.0], [0.0, jnp.exp(-jnp.pi * 1.0j / 4)]])

V = jnp.array([[1.0, -1.0j], [-1.0j, 1.0]]) / jnp.sqrt(2)

Vdg = jnp.array([[1.0, 1.0j], [1.0j, 1.0]]) / jnp.sqrt(2)

SX = jnp.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]]) / 2

SXdg = jnp.array([[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]]) / 2

CX = jnp.block([[I, _0], [_0, X]]).reshape((2,) * 4)

CY = jnp.block([[I, _0], [_0, Y]]).reshape((2,) * 4)

CZ = jnp.block([[I, _0], [_0, Z]]).reshape((2,) * 4)

CH = jnp.block([[I, _0], [_0, H]]).reshape((2,) * 4)

CV = jnp.block([[I, _0], [_0, V]]).reshape((2,) * 4)

CVdg = jnp.block([[I, _0], [_0, Vdg]]).reshape((2,) * 4)

CSX = jnp.block([[I, _0], [_0, SX]]).reshape((2,) * 4)

CSXdg = jnp.block([[I, _0], [_0, SXdg]]).reshape((2,) * 4)

CCX = jnp.block(
    [[I, _0, _0, _0], [_0, I, _0, _0], [_0, _0, I, _0], [_0, _0, _0, X]]  # Toffoli gate
).reshape((2,) * 6)

ECR = jnp.block([[_0, Vdg], [V, _0]]).reshape((2,) * 4)

SWAP = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1],
    ]
)

CSWAP = jnp.block([[jnp.eye(4), jnp.zeros((4, 4))], [jnp.zeros((4, 4)), SWAP]]).reshape(
    (2,) * 6
)


def Rx(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * X * 1.0j


def Ry(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * Y * 1.0j


def Rz(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * Z * 1.0j


def CRx(param: float) -> jax.Array:
    return jnp.block([[I, _0], [_0, Rx(param)]]).reshape((2,) * 4)


def CRy(param: float) -> jax.Array:
    return jnp.block([[I, _0], [_0, Ry(param)]]).reshape((2,) * 4)


def CRz(param: float) -> jax.Array:
    return jnp.block([[I, _0], [_0, Rz(param)]]).reshape((2,) * 4)


def U1(param: float) -> jax.Array:
    return U3(0, 0, param)


def U2(param0: float, param1: float) -> jax.Array:
    return U3(0.5, param0, param1)


def U3(param0: float, param1: float, param2: float) -> jax.Array:
    return (
        jnp.exp((param1 + param2) * jnp.pi * 1.0j / 2)
        * Rz(param1)
        @ Ry(param0)
        @ Rz(param2)
    )


def CU1(param: float) -> jax.Array:
    return jnp.block([[I, _0], [_0, U1(param)]]).reshape((2,) * 4)


def CU2(param0: float, param1: float) -> jax.Array:
    return jnp.block([[I, _0], [_0, U2(param0, param1)]]).reshape((2,) * 4)


def CU3(param0: float, param1: float, param2: float) -> jax.Array:
    return jnp.block([[I, _0], [_0, U3(param0, param1, param2)]]).reshape((2,) * 4)


def ISWAP(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    c = jnp.cos(param_pi_2)
    i_s = 1.0j * jnp.sin(param_pi_2)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, i_s, 0.0],
            [0.0, i_s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).reshape((2,) * 4)


def PhasedISWAP(param0: float, param1: float) -> jax.Array:
    param1_pi_2 = param1 * jnp.pi / 2
    c = jnp.cos(param1_pi_2)
    i_s = 1.0j * jnp.sin(param1_pi_2)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, i_s * jnp.exp(2.0j * jnp.pi * param0), 0.0],
            [0.0, i_s * jnp.exp(-2.0j * jnp.pi * param0), c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).reshape((2,) * 4)


def XXPhase(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    c = jnp.cos(param_pi_2)
    i_s = 1.0j * jnp.sin(param_pi_2)
    return jnp.array(
        [
            [c, 0.0, 0.0, -i_s],
            [0.0, c, -i_s, 0.0],
            [0.0, -i_s, c, 0.0],
            [-i_s, 0.0, 0.0, c],
        ]
    ).reshape((2,) * 4)


def YYPhase(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    c = jnp.cos(param_pi_2)
    i_s = 1.0j * jnp.sin(param_pi_2)
    return jnp.array(
        [
            [c, 0.0, 0.0, i_s],
            [0.0, c, -i_s, 0.0],
            [0.0, -i_s, c, 0.0],
            [i_s, 0.0, 0.0, c],
        ]
    ).reshape((2,) * 4)


def ZZPhase(param: float) -> jax.Array:
    param_pi_2 = param * jnp.pi / 2
    e_m = jnp.exp(-1.0j * param_pi_2)
    e_p = jnp.exp(1.0j * param_pi_2)
    return jnp.diag(jnp.array([e_m, e_p, e_p, e_m])).reshape((2,) * 4)


ZZMax = ZZPhase(0.5)


def PhasedX(param0: float, param1: float) -> jax.Array:
    return Rz(param1) @ Rx(param0) @ Rz(-param1)


def TK1(param0: float, param1: float, param2: float) -> jax.Array:
    """TKET's canonical one-qubit gate, Rz(param0) Rx(param1) Rz(param2).

    All three parameters are measured in half-turns.
    """
    return Rz(param0) @ Rx(param1) @ Rz(param2)


def TK2(param0: float, param1: float, param2: float) -> jax.Array:
    """TKET's canonical two-qubit gate with XX, YY and ZZ half-turn angles."""
    matrix = (
        XXPhase(param0).reshape(4, 4)
        @ YYPhase(param1).reshape(4, 4)
        @ ZZPhase(param2).reshape(4, 4)
    )
    return matrix.reshape((2,) * 4)
