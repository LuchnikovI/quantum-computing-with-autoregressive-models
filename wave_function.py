from typing import Tuple, List
import jax.numpy as jnp
from jax import random, pmap
import jax
import haiku as hk
from attention import AttentionEncoder
from utils import (
    Params,
    PRNGKey,
    NNet,
    _softsign,
    _push_two_qubit,
)
from functools import partial

class WaveFunction:
    """Autoregressive wave function based on self attention
    Args:
        number_of_heads: number of heads in MultiHeadAttention
        kqv_size: size of key, value and query for all layers
        number_of_layers: number of layers
        number_of_wave_functions: number of wave functions"""

    def __init__(self,
                 number_of_heads: int,
                 kqv_size: int,
                 number_of_layers: int,
                 number_of_wave_functions: int):

        self._number_of_heads = number_of_heads
        self._kqv_size = kqv_size
        self._number_of_layers = number_of_layers
        self._number_of_wave_functions = number_of_wave_functions

    def init(self, key: PRNGKey) -> Tuple[List[Params], NNet]:
        """Initializes wave function
        Args:

            key: PRNGKey

        Returns:
            network parameters, network"""

        def _forward(x):
            return AttentionEncoder(
                self._number_of_heads, self._kqv_size, self._number_of_layers
            )(x)

        # attention compilation
        forward = hk.without_apply_rng(hk.transform(_forward))
        params = forward.init(key, jnp.ones((1, 1), dtype=jnp.int32))

        return self._number_of_wave_functions * [params], forward.apply

    def sample(self,
               num_of_samples: int,
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
        samples = jnp.ones((num_of_samples, qubits_num + 1), dtype=jnp.int32)
        ind = 0
    
        def f(carry, xs):
            samples, key, ind = carry
            key, subkey = random.split(key)
            # samples_slice = jax.lax.dynamic_slice(samples, (0, 0, 0),
            #                                      (num_of_samples, 1+ind, loc_dim))
            logp = fwd(x=samples, params=params[wave_function_number])[:, ind, :2]
            logp = jax.nn.log_softmax(logp)
            eps = random.gumbel(subkey, logp.shape)
            s = jnp.argmax(logp + eps, axis=-1)
            samples = jax.ops.index_update(samples, jax.ops.index[:, ind + 1], s)
            return (samples, key, ind + 1), None
    
        (samples, _, _), _ = jax.lax.scan(f, (samples, key, ind), None, length=qubits_num)
        return samples[:, 1:]

    def log_amplitude(self,
                      sample: jnp.ndarray,
                      wave_function_number: int,
                      params: List[Params],
                      fwd: NNet,
                      qubits_num: int) -> jnp.ndarray:
        """Return log(wave function) for a given sample.
    
        Args:
            sample: (num_of_samples, length) array like
            wave_function_number: number of a wave function to evaluate
            params: parameters
            fwd: network
            qubits_num: number of qubits
    
        Returns:
            log(wave function)"""
    
        shape = sample.shape
        bs = shape[0]
        zero_spin = jnp.ones((bs, 1), dtype=jnp.int32)
        inp = jnp.concatenate([zero_spin, sample], axis=1)
        out = fwd(x=inp[:, :-1], params=params[wave_function_number])
        logabs = out[..., :2]
        logabs = jax.nn.log_softmax(logabs)
        logabs = 0.5 * (logabs * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
        phi = out[..., 2:]
        phi = jnp.pi * _softsign(phi)
        phi = (phi * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
        return logabs + 1j * phi

    def two_qubit_gate_log_amplitude(self,
                                     gate: jnp.ndarray,
                                     sides: List[int],
                                     sample: jnp.ndarray,
                                     wave_function_number: int,
                                     params: List[Params],
                                     fwd: NNet,
                                     qubits_num: int) -> jnp.ndarray:
        """Return log(gate.dot(wave function)) for a given sample

        Args:
            gate: (2, 2, 2, 2) array like
            sides: list with two integers, sides where to apply a gate
            sample: (num_of_samples, length) array like
            wave_function_number: number of a wave function to evaluate
            params: parameters
            fwd: network
            qubits_num: number of qubits

        Returns:
            two array like (num_of_samples,) -- log of absolut value and phase"""

        pushed_samples, weights = _push_two_qubit(sample, gate, sides)
        log_weights = jnp.log(weights)
        pushed_samples = pushed_samples.reshape((-1, qubits_num))
        log_psi = self.log_amplitude(pushed_samples, wave_function_number, params, fwd, qubits_num)
        log_psi = log_psi.reshape((-1, 4))
        log = log_psi + log_weights
        max_log = jnp.real(log).max(-1, keepdims=True)
        log = jnp.log(jnp.exp(log - max_log).sum(-1)) + max_log
        return log

    def bracket(self,
                log_bra: jnp.ndarray,
                log_ket: jnp.ndarray) -> jnp.ndarray:
        """Calculates <psi|psi>

        Args:
            log_bra: array like of shape (bs,)
            log_ket: array_like of shape (bs,)

        Returns:
            array like of shape (1,)"""

        return jnp.exp(log_ket - log_bra).mean()


class WaveFunctionParallel(WaveFunction):

    def init(self, key: PRNGKey) -> Tuple[List[Params], NNet]:
        params, fwd = super().init(key)
        num_devices = jax.local_device_count()
        params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)
        return params, fwd

    @partial(pmap,
             in_axes=(None, None, 0, None, 0, None, None),
             out_axes=0,
             static_broadcasted_argnums=(0, 1, 3, 5, 6))
    def sample(self,
               num_of_samples: int,
               key: PRNGKey,
               wave_function_number: int,
               params: List[Params],
               fwd: NNet,
               qubits_num: int):
        return super().sample(num_of_samples, key, wave_function_number, params, fwd, qubits_num)

    @partial(pmap,
             in_axes=(None, 0, None, 0, None, None),
             out_axes=0,
             static_broadcasted_argnums=(0, 2, 4, 5))
    def log_amplitude(self,
                      sample: jnp.ndarray,
                      wave_function_number: int,
                      params: List[Params],
                      fwd: NNet,
                      qubits_num: int):
        return super(WaveFunction, self).log_amplitude(sample, wave_function_number, params, fwd, qubits_num)

    @partial(pmap,
             in_axes=(None, None, None, 0, None, 0, None, None),
             out_axes=0,
             static_broadcasted_argnums=(0, 1, 2, 4, 6, 7))
    def two_qubit_gate_log_amplitude(self,
                                     gate: jnp.ndarray,
                                     sides: List[int],
                                     sample: jnp.ndarray,
                                     wave_function_number: int,
                                     params: List[Params],
                                     fwd: NNet,
                                     qubits_num: int):
        return super().two_qubit_gate_log_amplitude(gate, sides, sample, wave_function_number, params, fwd, qubits_num)

    @partial(pmap,
             in_axes=(0, 0),
             out_axes=0)
    def bracket(self,
                log_bra: jnp.ndarray,
                log_ket: jnp.ndarray):
        return super().bracket(log_bra, log_ket)
