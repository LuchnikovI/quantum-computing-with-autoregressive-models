import jax.numpy as jnp
import jax
from jax import vmap, random, value_and_grad
from jax.lax import pmean
from typing import Mapping, Callable, List, Tuple, Any
from functools import partial
import optax


Params = Mapping[str, Mapping[str, jnp.ndarray]]  # Neural network params type
PRNGKey = jnp.ndarray
NNet = Callable[[jnp.ndarray, Params], jnp.ndarray]  # Neural network type


@partial(vmap, in_axes=(0, None, None))
def _push_two_qubit(pauli_string, u, sides):
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


@partial(vmap, in_axes=(0, None, None))
def _push_one_qubit(pauli_string, u, side):
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


def _softsign(x):
    return x / (1 + jnp.abs(x))


def _sample(num_of_samples: int,
            key: PRNGKey,
            wave_function_number: int,
            params: List[Params],
            fwd: NNet,
            qubits_num: int) -> jnp.array:
    """Return samples from the wave function.

    Args:
        num_of_samples: number of samples
        key: PRNGKey
        wave_function_number: number of a wave function to sample from
        params: parameters
        fwd: network
        qubits_num: number of qubits

    Returns:
        (num_of_samples, length) array like"""

    # TODO check whether one has a problem with PRNGKey splitting
    samples = jnp.ones((num_of_samples, qubits_num+1), dtype=jnp.int32)
    ind = 0
    def f(carry, xs):
        samples, key, ind = carry
        key, subkey = random.split(key)
        #samples_slice = jax.lax.dynamic_slice(samples, (0, 0, 0), (num_of_samples, 1+ind, loc_dim))
        logp = fwd(x=samples, params=params[wave_function_number])[:, ind, :2]
        logp = jax.nn.log_softmax(logp)
        eps = random.gumbel(subkey, logp.shape)
        s = jnp.argmax(logp + eps, axis=-1)
        samples = jax.ops.index_update(samples, jax.ops.index[:, ind+1], s)
        return (samples, key, ind+1), None

    (samples, _, _), _ = jax.lax.scan(f, (samples, key, ind), None, length=qubits_num)
    return samples[:, 1:]


def _log_amplitude(string: jnp.ndarray,
                   wave_function_number: int,
                   params: List[Params],
                   fwd: NNet,
                   qubits_num: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return log(wave function) for a given set of bit strings.

    Args:
        string: (num_of_samples, length) array like
        wave_function_number: number of a wave function to evaluate
        params: parameters
        fwd: network
        qubits_num: number of qubits

    Returns:
        two array like (num_of_samples,) -- log of absolut value and phase"""

    shape = string.shape
    bs = shape[0]
    zero_spin = jnp.ones((bs, 1), dtype=jnp.int32)
    inp = jnp.concatenate([zero_spin, string], axis=1)
    out = fwd(x=inp[:, :-1], params=params[wave_function_number])
    logabs = out[..., :2]
    logabs = jax.nn.log_softmax(logabs)
    logabs = 0.5 * (logabs * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
    phi = out[..., 2:]
    phi = jnp.pi * _softsign(phi)
    phi = (phi * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
    return logabs, phi


def _two_qubit_gate_bracket(gate: jnp.ndarray,
                            sides: List[int],
                            wave_function_numbers: List[int],
                            key: PRNGKey,
                            num_of_samples: int,
                            params: List[Params],
                            fwd: NNet,
                            qubits_num: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates <psi_1|U^dagger|psi_2>

        Args:
            gate: (2, 2, 2, 2) array like
            sides: list with two int values specifying sides
                where to apply a gate
            wave_function_numbers: list with two int values specifying numbers
                of wave functions
            key: PRNGKey
            num_of_samples: number of samples
            params: parameters
            fwd: network
            qubits_num: number of qubits

        Returns:
            two array like of shape (1,)"""

        sample = _sample(num_of_samples,
                         key,
                         wave_function_numbers[0],
                         params,
                         fwd,
                         qubits_num)
        pushed_sample, ampls = _push_two_qubit(sample, gate.transpose((2, 3, 0, 1)).conj(), sides)
        denom = _log_amplitude(sample,
                               wave_function_numbers[0],
                               params,
                               fwd,
                               qubits_num)
        nom = _log_amplitude(pushed_sample.reshape((-1, qubits_num)),
                             wave_function_numbers[1],
                             params,
                             fwd,
                             qubits_num)
        log_abs = nom[0].reshape((-1, 4)) - denom[0][:, jnp.newaxis]
        phi = nom[1].reshape((-1, 4)) - denom[1][:, jnp.newaxis]
        re = jnp.exp(log_abs) * jnp.cos(phi)
        im = jnp.exp(log_abs) * jnp.sin(phi)
        ampls_re = jnp.real(ampls)
        ampls_im = jnp.imag(ampls)
        re, im = ampls_re * re - ampls_im * im, re * ampls_im + im * ampls_re
        re, im = re.sum(1).mean(), im.sum(1).mean()
        return re, im


def _train_step(gate: jnp.ndarray,
                loss: jnp.ndarray,
                sides: List[int],
                opt: Any,
                opt_state: Any,
                num_of_samples: int,
                key: PRNGKey,
                params: List[Params],
                fwd: NNet,
                qubits_num: int) -> Tuple[jnp.ndarray, List[Params], PRNGKey, Any]:
    """Makes one training step

    Args:
        gate: (2, 2, 2, 2) array like
        loss: (1,) array like
        sides: list with two elements showing where to apply a gate
        opt: optax optimizer
        opt_state: state of an optax optimizer
        num_of_samples: number of samples used to evaluate loss function
        key: PRGNKey
        params: parameters of wave function
        fwd: network
        qubit_num: number of qubits

    Returns:
        loss function value, new set of parameters, new PRNGKey,
        optimizer state"""

    key = random.split(key)[0]
    param1, param2 = params[0], params[1]
    loss_func = lambda param1, param2: 1 - _two_qubit_gate_bracket(gate, sides, [0, 1], key, num_of_samples, [param1, param2], fwd, qubits_num)[0]
    l, grad = value_and_grad(loss_func, 1)(param1, param2)
    l = pmean(l, axis_name='i')
    grad = pmean(grad, axis_name='i')
    update, opt_state = opt.update(grad, opt_state, param2)
    param2 = optax.apply_updates(param2, update)
    params[1] = param2
    return loss+l, params, key, opt_state


def _train_epoch(gate: jnp.ndarray,
                sides: List[int],
                opt: Any,
                opt_state: Any,
                num_of_samples: int,
                key: PRNGKey,
                epoch_size: int,
                params: List[Params],
                fwd: NNet,
                qubits_num: int) -> Tuple[jnp.ndarray, List[Params], PRNGKey, Any]:
    """Makes training epoch

    Args:
        gate: (2, 2, 2, 2) array like
        sides: list with two elements showing where to apply a gate
        opt: optax optimizer
        opt_state: state of an optax optimizer
        num_of_samples: number of samples used to evaluate loss function
        key: PRGNKey
        epoch_size: number of iterations
        params: parameters of wave function
        fwd: network
        qubit_num: number of qubits

    Returns:
        loss function value, new set of parameters, new PRNGKey,
        optimizer state"""

    body_fun = lambda i, val: _train_step(gate, val[0], sides, opt, val[3], num_of_samples, val[2], val[1], fwd, qubits_num)
    loss, params, key, opt_sate = jax.lax.fori_loop(0, epoch_size, body_fun, (jnp.array(0.), params, key, opt_state))
    return loss/epoch_size, params, key, opt_state
