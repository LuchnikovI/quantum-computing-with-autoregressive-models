import jax.numpy as jnp
import jax
from jax import vmap, jit
from typing import Mapping, Callable

def push_two_qubit(pauli_string, u, sides):
    """Pushes pauli string through a two-qubit quantum gate.

    Args:
        pauli_string: int valued tensor of shape (n,), a sample
        u: complex valued tensor of shape (2, 2, 2, 2), a quantum gate
        sides: list with two integers, sides where to apply a gate

    Returns:
        int valued tensor of shape (4, n), and complex valued tensor
        of shape (4,). The first tensor is pushed pauil string through
        a gate, the second tensor is weights of each string"""

    ind1 = pauli_string[sides[1]]
    ind2 = pauli_string[sides[0]]
    not_ind1 = jnp.logical_not(ind1).astype(jnp.int32)
    not_ind2 = jnp.logical_not(ind2).astype(jnp.int32)
    weights = u[ind2, ind1]
    pauli_string1 = jax.ops.index_update(pauli_string, sides[1], not_ind1)
    pauli_string2 = jax.ops.index_update(pauli_string, sides[0], not_ind2)
    pauli_string3 = jax.ops.index_update(pauli_string2, sides[1], not_ind1)
    pusshed_pauli_strings = jnp.concatenate([pauli_string[jnp.newaxis],
                                             pauli_string1[jnp.newaxis],
                                             pauli_string2[jnp.newaxis],
                                             pauli_string3[jnp.newaxis]], axis=0)
    weights = weights[(ind2, ind2, not_ind2, not_ind2),
                      (ind1, not_ind1, ind1, not_ind1)]
    return pusshed_pauli_strings, weights

push_two_qubit_vec = vmap(push_two_qubit, (0, None, None), (0, 0))

def push_one_qubit(pauli_string, u, side):
    """Pushes pauli string through a one-qubit quantum gate.

    Args:
        pauli_string: int valued tensor of shape (n,), a sample
        u: complex valued tensor of shape (2, 2), a quantum gate
        sides: int number, side where to apply a gate

    Returns:
        int valued tensor of shape (2, n), and complex valued tensor
        of shape (2,). The first tensor is pushed pauil string through
        a gate, the second tensor is weights of each string"""

    ind = pauli_string[side]
    not_ind = jnp.logical_not(ind).astype(jnp.int32)
    weights = u[ind]
    pauli_string1 = jax.ops.index_update(pauli_string, side, not_ind)
    pusshed_pauli_strings = jnp.concatenate([pauli_string[jnp.newaxis],
                                             pauli_string1[jnp.newaxis]], axis=0)
    weights = jnp.array([weights[ind], weights[not_ind]])
    return pusshed_pauli_strings, weights

push_one_qubit_vec = vmap(push_one_qubit, (0, None, None), (0, 0))

@jit
def softsign(x):
    return x / (1 + jnp.abs(x))

Params = Mapping[str, Mapping[str, jnp.ndarray]]  # Neural network params type
PRNGKey = jnp.ndarray
NNet = Callable[[jnp.ndarray, Params], jnp.ndarray]  # Neural network type