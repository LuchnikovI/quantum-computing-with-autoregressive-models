import jax
from jax import random, pmap
import jax.numpy as jnp
import haiku as hk
from attention import AttentionEncoder
from utils import softsign
from typing import Mapping, Tuple, Callable
from functools import partial

Params = Mapping[str, Mapping[str, jnp.ndarray]]
PRNGKey = jnp.ndarray
NNet = Callable[[jnp.ndarray, Params], jnp.ndarray]  # Neural network type


class AttentionWaveFunction:
    """Attention network-based wave function

    Args:
        number_of_heads: number of heads in MultiHeadAttention
        kqv_size: size of key, value and query for all layers
        number_of_layers: number of layers
        qubits_num: number of qubits
        key: PRNGKey

    Returns:
        network parameters, network and number of qubits"""

    def __init__(self,
                 number_of_heads: int,
                 kqv_size: int,
                 number_of_layers: int,
                 qubits_num: int,
                 key: PRNGKey) -> Tuple[Params, NNet, int]:

        def _forward(x):
            return AttentionEncoder(number_of_heads,
                                    kqv_size,
                                    number_of_layers)(x)
        
        # attention compilation
        forward = hk.without_apply_rng(hk.transform(_forward))
        params = forward.init(key, jnp.ones((1, 1), dtype=jnp.int32))
        num_devices = jax.local_device_count()
        params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)
        
        return params, forward.apply, qubits_num
        
    @partial(pmap, in_axes=(None, 0, 0, None, None), out_axes=0, static_broadcasted_argnums=(0, 3, 4))
    def sample(self,
               num_of_samples: int,
               key: PRNGKey,
               params: Params,
               fwd: NNet,
               qubits_num: int) -> jnp.array:
        """Return samples from wave function.

        Args:
            num_of_samples: number of samples
            key: PRNGKey
            state: state
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
            logp = fwd(x=samples, params=params)[:, ind, :2]
            logp = jax.nn.log_softmax(logp)
            eps = random.gumbel(subkey, logp.shape)
            s = jnp.argmax(logp + eps, axis=-1)
            samples = jax.ops.index_update(samples, jax.ops.index[:, ind+1], s)
            return (samples, key, ind+1), None

        (samples, _, _), _ = jax.lax.scan(f, (samples, key, ind), None, length=qubits_num)
        return samples[:, 1:]

    @partial(pmap, in_axes=(0, 0, None, None), out_axes=0, static_broadcasted_argnums=(2, 3))
    def log_psi(self,
                string: jnp.ndarray,
                params: Params,
                fwd: NNet,
                qubits_num: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return log(wave_function) for given set of bit strings.

        Args:
            string: (num_of_samples, length) array like
            state: state
            fwd: network
            qubits_num: number of qubits

        Returns:
            two array like (num_of_samples,) -- log of absolut value and phase"""

        shape = string.shape
        bs = shape[0]
        zero_spin = jnp.ones((bs, 1), dtype=jnp.int32)
        inp = jnp.concatenate([zero_spin, string], axis=1)
        out = fwd(x=inp[:, :-1], params=params)
        logabs = out[..., :2]
        logabs = jax.nn.log_softmax(logabs)
        logabs = 0.5 * (logabs * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
        phi = out[..., 2:]
        phi = jnp.pi * softsign(phi)
        phi = (phi * jax.nn.one_hot(inp[:, 1:], 2)).sum((-2, -1))
        return logabs, phi
