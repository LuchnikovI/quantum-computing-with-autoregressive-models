import jax
from jax import jit
from jax import random
import jax.numpy as jnp
import haiku as hk


def positional_encoding(T, d):
    """Retrurns a matrix of positional codes of fixed size.

    Args:
        T: int value, length of the encoding
        d: int value, depth of the encoding

    Returns:
        real valued tensor of shape (T, d)"""

    omega = (1 / 10000) ** (jnp.arange(0, d, step=2) / d)
    sins = jnp.sin(omega * jnp.arange(0, T)[:, jnp.newaxis])
    coss = jnp.cos(omega * jnp.arange(0, T)[:, jnp.newaxis])
    encod = jnp.concatenate([sins[..., jnp.newaxis], coss[..., jnp.newaxis]], axis=-1)
    encod = encod.reshape((encod.shape[0], -1))
    return encod


class AttentionEncoder(hk.Module):

    """Class instantiates autoregressive model based on the attention mechanism."""
    def __init__(self,
                 heads_layers,
                 KQ_layers,
                 V_layers,
                 out_size,
                 max_length=128,
                 depth=2,
                 name='AttentionEncoder'):
        super().__init__(name=name)
        self.heads_layers = heads_layers  # number of heads per attention layer
        self.KQ_layers = KQ_layers  # size of the key and query per attention layer
        self.V_layers = V_layers  # size of the value per attention layer
        self.positional_encoding = positional_encoding(max_length, 4*depth)  # positional encoding
        self.out_size = out_size  # size of the output
        
    def __call__(self, x):
        shape = x.shape
        length = shape[-2]
        mask = jnp.ones((length, length))
        mask = jnp.tril(mask, 0)
        enc = self.positional_encoding[:length]
        enc = jnp.tile(enc[jnp.newaxis], (shape[0], 1, 1))
        x = jnp.concatenate([x, enc], axis=-1)
        for heads, KQs, Vs in zip(self.heads_layers,
                                  self.KQ_layers,
                                  self.V_layers):
            x = hk.MultiHeadAttention(heads, KQs, 1, KQs, Vs)(x, x, x, mask=mask)
            skip = x
            x = hk.Linear(skip.shape[-1])(x)
            x = jax.nn.leaky_relu(x)
            x = hk.Linear(skip.shape[-1])(x) + skip
        x = jax.nn.leaky_relu(x)
        x = hk.Linear(self.out_size)(x)
        return x

@jit
def softsign(x):
    return x / (1 + jnp.abs(x))

def log_psi(string, loc_dim, params, fwd):
    """Returns real and imag parts of log(psi)

    Args:
        string: int valued tensor of shape (bs, length)
        loc_dim: int value, local Hilbert space dimension
        params: py tree, params of a model
        fwd: initialized Haiku model

    Returns two tensors of shape (bs,)"""

    shape = string.shape
    bs = shape[0]
    zero_spin = jnp.ones((bs, 1, loc_dim))
    inp = jnp.concatenate([zero_spin, jax.nn.one_hot(string, loc_dim)], axis=1)
    out = fwd(x=inp[:, :-1], params=params)
    logp = out[..., :loc_dim]
    logp = jax.nn.log_softmax(logp)
    logp = 0.5 * (logp * inp[:, 1:]).sum((-2, -1))
    logphi = out[..., loc_dim:]
    logphi = jnp.pi * softsign(logphi)
    logphi = (logphi * inp[:, 1:]).sum((-2, -1))
    return logp, logphi

def sample(num_of_samples, length, loc_dim, params, fwd, key):
    samples = jnp.ones((num_of_samples, length+1, loc_dim))
    for i in range(length):
        key, subkey = random.split(key)
        logp = fwd(x=samples[:, :i+1], params=params)[:, i+1, :loc_dim]
        logp = jax.nn.log_softmax(logp)
        eps = random.gumbel(subkey, logp.shape)
        s = jax.nn.one_hot(jnp.argmax(logp + eps, axis=-1), loc_dim)
        samples = jax.ops.index_update(samples, jax.ops.index[:, i+1], s)
    return jnp.argmax(samples[:, 1:], -1)
