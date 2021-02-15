import jax
from jax import jit
from jax import random
import jax.numpy as jnp
import haiku as hk
from utils import push_two_qubit_vec


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
        x = hk.Linear(self.out_size, w_init=hk.initializers.Constant(0))(x)
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

def two_qubit_gate_fidelity(params1, params2, key, gate, sides, num_of_samples, length, loc_dim, fwd):
    """Calculates <psi_old|U^dagger|psi_new>
    
    Args:
        params1: py tree, old parameters
        params2: py tree, new parameters
        key: PRNGKey
        gate: complex valued tensor of shape (2, 2, 2, 2)
        sides: list with two int values representing sides
            where to apply a gate
        num_of_samples: int value, number of samples from |psi|^2
        length: int value, chain length
        loc_dim: int value, local Hilbert space dimension
        fwd: neural net
    
    Returns:
        real valued tensor of shape (1,), <psi_old|U^dagger|psi_new>"""

    smpl = sample(num_of_samples, length, loc_dim, params1, fwd, key)
    pushed_smpl, ampls = push_two_qubit_vec(smpl, gate.transpose((2, 3, 0, 1)).conj(), sides)
    denom = log_psi(smpl, loc_dim, params1, fwd)
    nom = log_psi(pushed_smpl.reshape((-1, length)), loc_dim, params2, fwd)
    log_sq_abs = nom[0].reshape((-1, 4)) - denom[0][:, jnp.newaxis]
    phi = nom[1].reshape((-1, 4)) - denom[1][:, jnp.newaxis]
    re = jnp.exp(2*log_sq_abs) * jnp.cos(phi)
    im = jnp.exp(2*log_sq_abs) * jnp.sin(phi)
    ampls_re = jnp.real(ampls)
    ampls_im = jnp.imag(ampls)
    re, im = ampls_re * re - ampls_im * im, re * ampls_im + im * ampls_re
    re, im = pmean(re.sum(1).mean(), axis_name='i'), pmean(im.sum(1).mean(), axis_name='i')
    return jnp.sqrt(re ** 2 + im ** 2)
