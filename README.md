# qujax

<div align="center">
<a href="https://quantinuum.github.io/qujax/">
<picture>
    <source srcset="docs/logo_dark_mode.svg"  media="(prefers-color-scheme: dark)">
    <img src="docs/logo.svg">
</picture>
</a>
</div>

[![PyPI - Version](https://img.shields.io/pypi/v/qujax)](https://pypi.org/project/qujax/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05504/status.svg)](https://doi.org/10.21105/joss.05504)

[**Documentation**](https://quantinuum.github.io/qujax/) | [**Installation**](#installation) | [**Quick start**](#quick-start) | [**Examples**](https://quantinuum.github.io/qujax/examples.html) | [**Contributing**](#contributing) | [**Citing qujax**](#citing-qujax)

qujax is a [JAX](https://github.com/google/jax)-based Python library for the classical simulation of quantum circuits. It is designed to be *simple*, *fast* and *flexible*.

It follows a functional programming design by translating circuits into pure functions. This allows qujax to [seamlessly interface with JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions), enabling direct access to its powerful automatic differentiation tools, just-in-time compiler, vectorization capabilities, GPU/TPU integration and growing ecosystem of packages.

qujax can be used both for pure and for mixed quantum state simulation. It not only supports the standard gate set, but also allows user-defined custom operations, including general quantum channels, enabling the user to e.g. model device noise and errors. 

A summary of the core functionalities of qujax can be found in the [Quick start](#quick-start) section. More advanced use-cases, including the training of parameterised quantum circuits, can be found in the [Examples](https://quantinuum.github.io/qujax/examples.html) section of the documentation.


## Installation

qujax is [hosted on PyPI](https://pypi.org/project/qujax/) and can be installed via the pip package manager
```
pip install qujax
```

## Quick start

**Important note: qujax circuit parameters are expressed in units of $\pi$ (e.g. in the range $[0,2]$ as opposed to $[0, 2\pi]$)**.

Start by defining the quantum gates making up the circuit, the qubits that they act on, and the indices of the parameters for each gate. 

A list of all gates can be found [here](https://github.com/Quantinuum/qujax/blob/main/qujax/gates.py) (custom operations can be included by [passing an array or function](https://quantinuum.github.io/qujax/statetensor/get_params_to_statetensor_func.html) instead of a string).

```python
from jax import numpy as jnp
import qujax

# List of quantum gates
circuit_gates = ['H', 'Ry', 'CZ']
# Indices of qubits the gates will be applied to
circuit_qubit_inds = [[0], [0], [0, 1]]
# Indices of parameters each parameterised gate will use
circuit_params_inds = [[], [0], []]

qujax.print_circuit(circuit_gates, circuit_qubit_inds, circuit_params_inds);
# q0: -----H-----Ry[0]-----◯---
#                          |   
# q1: ---------------------CZ--
```

Translate the circuit to a pure function `param_to_st` that takes a set of parameters and an (optional) initial quantum state as its input.

```python
param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                   circuit_qubit_inds,
                                                   circuit_params_inds)

param_to_st(jnp.array([0.1]))
# Array([[0.58778524+0.j, 0.        +0.j],
#        [0.80901706+0.j, 0.        +0.j]], dtype=complex64)
```

The optional initial state can be passed to `param_to_st` using the `statetensor_in` argument. When it is not provided, the initial state defaults to $\ket{0...0}$.

Map the state to an expectation value by defining an observable using lists of Pauli matrices, the qubits they act on, and the associated coefficients. 

```python
st_to_expectation = qujax.get_statetensor_to_expectation_func([['Z']], [[0]], [1.])
```

Combining `param_to_st` and `st_to_expectation` gives us a parameter to expectation function that can be automatically differentiated using JAX.

```python
from jax import value_and_grad

param_to_expectation = lambda param: st_to_expectation(param_to_st(param))
expectation_and_grad = value_and_grad(param_to_expectation)
expectation_and_grad(jnp.array([0.1]))
# (Array(-0.3090171, dtype=float32),
#  Array([-2.987832], dtype=float32))
```

Mixed state simulations are analogous to the above, but with calls to [`get_params_to_densitytensor_func`](https://quantinuum.github.io/qujax/densitytensor/get_params_to_densitytensor_func.html) and [`get_densitytensor_to_expectation_func`](https://quantinuum.github.io/qujax/densitytensor/get_densitytensor_to_expectation_func.html) instead.

A more in-depth version of the above can be found in the [Getting started](https://quantinuum.github.io/qujax/getting_started.html) section of the documentation. More advanced use-cases, including the training of parameterised quantum circuits, can be found in the [Examples](https://quantinuum.github.io/qujax/examples.html) section of the documentation.

## Converting from TKET

A [`pytket`](https://quantinuum.github.io/tket/pytket/api/) circuit can be directly converted using the [`tk_to_qujax`](https://quantinuum.github.io/pytket-qujax/api/api.html#pytket.extensions.qujax.qujax_convert.tk_to_qujax) and [`tk_to_qujax_symbolic`](https://quantinuum.github.io/pytket-qujax/api/api.html#pytket.extensions.qujax.qujax_convert.tk_to_qujax_symbolic) functions in the [**`pytket-qujax`**](https://github.com/Quantinuum/pytket-qujax) extension. See [`pytket-qujax_heisenberg_vqe.ipynb`](https://github.com/Quantinuum/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb) for an example.

## Contributing

You can open a bug report or a feature request by creating a new [issue on GitHub](https://github.com/Quantinuum/qujax/issues).

Pull requests are welcome! To open a new one, please go through the following steps:

1. First fork the repo and create your branch from [`develop`](https://github.com/Quantinuum/qujax/tree/develop).
2. Commit your code and tests.
4. Update the documentation, if required.
5. Check the code lints (run `black . --check` and `pylint */`).
6. Issue a pull request into the [`develop`](https://github.com/Quantinuum/qujax/tree/develop) branch.

New commits on [`develop`](https://github.com/Quantinuum/qujax/tree/develop) will be merged into
[`main`](https://github.com/Quantinuum/qujax/tree/main) in the next release.


## Citing qujax

If you have used qujax in your code or research, we kindly ask that you cite it. You can use the following BibTeX entry for this:

```bibtex
@article{qujax2023,
  author = {Duffield, Samuel and Matos, Gabriel and Johannsen, Melf},
  doi = {10.21105/joss.05504},
  journal = {Journal of Open Source Software},
  month = sep,
  number = {89},
  pages = {5504},
  title = {{qujax: Simulating quantum circuits with JAX}},
  url = {https://joss.theoj.org/papers/10.21105/joss.05504},
  volume = {8},
  year = {2023}
}
```