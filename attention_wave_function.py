import jax
from jax import random, pmap
import jax.numpy as jnp
import haiku as hk
from attention import AttentionEncoder
from utils import push_two_qubit, softsign, Params, PRNGKey, NNet
from typing import Tuple, List
from functools import partial


class AttentionWaveFunction:
    """Attention network-based wave function

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

    def init(self,
             key: PRNGKey,
             qubits_num: int) -> Tuple[List[Params], NNet, int]:
        """Initializes wave function

        Args:
            key: PRNGKey
            qubits_num: number of qubits

        Returns:
            network parameters, network and number of qubits"""

        def _forward(x):
            return AttentionEncoder(self._number_of_heads,
                                    self._kqv_size,
                                    self._number_of_layers)(x)

        # attention compilation
        forward = hk.without_apply_rng(hk.transform(_forward))
        params = forward.init(key, jnp.ones((1, 1), dtype=jnp.int32))
        num_devices = jax.local_device_count()
        params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)

        return self._number_of_wave_functions*[params], forward.apply, qubits_num

    def _sample(self,
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

    def _log_amplitude(self,
                       string: jnp.ndarray,
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
        phi = jnp.pi * softsign(phi)
        phi = (phi * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
        return logabs, phi

    @partial(pmap, in_axes=(None, None, 0, None, 0, None, None), out_axes=0, static_broadcasted_argnums=(0, 1, 3, 5, 6))
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

        return self._sample(num_of_samples, key, wave_function_number, params, fwd, qubits_num)

    @partial(pmap, in_axes=(None, 0, None, 0, None, None), out_axes=0, static_broadcasted_argnums=(0, 2, 4, 5))
    def log_amplitude(self,
                      string: jnp.ndarray,
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

        return self._log_amplitude(string, wave_function_number, params, fwd, qubits_num)

    @partial(pmap, in_axes=(None, None, None, None, 0, None, 0, None, None), out_axes=0, static_broadcasted_argnums=(0, 2, 3, 5, 7, 8))
    def two_qubit_gate_bracket(self,
                               gate: jnp.ndarray,
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

        sample = self._sample(num_of_samples,
                              key,
                              wave_function_numbers[0],
                              params,
                              fwd,
                              qubits_num)
        pushed_sample, ampls = push_two_qubit(sample, gate.transpose((2, 3, 0, 1)).conj(), sides)
        denom = self._log_amplitude(sample,
                                    wave_function_numbers[0],
                                    params,
                                    fwd,
                                    qubits_num)
        nom = self._log_amplitude(pushed_sample.reshape((-1, qubits_num)),
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
