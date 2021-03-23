import jax
from jax import jit
from jax import random
import jax.numpy as jnp
from jax.lax import pmean
import haiku as hk
from utils import push_two_qubit_vec
import optax


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
    """Haiku model of autoregressive attention.
        
    Args:
        number_of_heads: int number, number of heads in MultiHeadAttention
        kqv_size: int number, size of key, value and query for all layers
        number_of_layers: int number, number of layers
        max_length: int, max length of a chain
        depth: int, the dimension of the input vector at each side
        name: name of the network"""

    def __init__(self,
                 number_of_heads,
                 kqv_size,
                 number_of_layers,
                 max_length=128,
                 depth=2,
                 name='AttentionEncoder'):

        super().__init__(name=name)
        self.number_of_heads = number_of_heads
        self.kqv_size = kqv_size  # size of the key, value and query
        self.number_of_layers = number_of_layers
        self.positional_encoding = positional_encoding(max_length, kqv_size * number_of_heads)  # positional encoding
        self.out_size = 2 * depth
        self.depth = depth
        self.hidden_size = kqv_size * number_of_heads  # size of hidden representation
        
    def __call__(self, x):

        # build mask necessary for the autoregressive property
        shape = x.shape
        length = shape[-1]
        mask = jnp.ones((length, length))
        mask = jnp.tril(mask, 0)
        
        # build embedding of the input seq
        x = hk.Embed(self.depth, self.hidden_size)(x)
        
        # + pos. encoding
        x = x + self.positional_encoding[:length]
        
        # for loop over layers
        for _ in range(self.number_of_layers):
            # attention layer
            skip = x
            x = hk.MultiHeadAttention(self.number_of_heads,
                                      self.kqv_size,
                                      1,
                                      self.kqv_size,
                                      self.kqv_size)(x, x, x, mask=mask)

            # add & norm
            x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x + skip)
            
            # fnn
            skip = x
            x = jax.nn.leaky_relu(x, 0.2)
            x = hk.Linear(self.hidden_size)(x)
            
            # add & norm
            x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x + skip)
        
        # final linear layer
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
        params: py tree, parameters of a Haiku model
        fwd: initialized Haiku model

    Returns:
        two tensors of shape (bs,)"""

    shape = string.shape
    bs = shape[0]
    zero_spin = jnp.ones((bs, 1), dtype=jnp.int32)
    inp = jnp.concatenate([zero_spin, string], axis=1)
    out = fwd(x=inp[:, :-1], params=params)
    logabs = out[..., :loc_dim]
    logabs = jax.nn.log_softmax(logabs)
    logabs = 0.5 * (logabs * jax.nn.one_hot(inp[:, 1:], loc_dim)).sum((-2, -1))
    phi = out[..., loc_dim:]
    phi = jnp.pi * softsign(phi)
    phi = (phi * inp[:, 1:]).sum((-2, -1))
    return logabs, phi

def sample(num_of_samples, length, loc_dim, params, fwd, key):
    """Makes samples from |psi|^2

    Args:
        num_of_samples: int, number of samples
        length: int, length of a chain
        loc_dim: the dimension of a local space
        params: py tree, parameters of a Haiku model
        fwd: initialized Haiku model
        key: PRNGKey

    Returns:
        int valued tensor of shape (number_of_samples, length)"""

    # TODO check whether one has a problem with PNGKey
    samples = jnp.ones((num_of_samples, length+1), dtype=jnp.int32)
    ind = 0
    def f(carry, xs):
        samples, key, ind = carry
        key, subkey = random.split(key)
        #samples_slice = jax.lax.dynamic_slice(samples, (0, 0, 0), (num_of_samples, 1+ind, loc_dim))
        logp = fwd(x=samples, params=params)[:, ind, :loc_dim]
        logp = jax.nn.log_softmax(logp)
        eps = random.gumbel(subkey, logp.shape)
        s = jnp.argmax(logp + eps, axis=-1)
        samples = jax.ops.index_update(samples, jax.ops.index[:, ind+1], s)
        return (samples, key, ind+1), None

    (samples, _, _), _ = jax.lax.scan(f, (samples, key, ind), None, length=length)
    return samples[:, 1:]

def two_qubit_gate_braket(params1, params2, key, gate, sides, num_of_samples, length, loc_dim, fwd):
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
        two real valued tensors of shape (1,), Re(<psi_old|U^dagger|psi_new>)
        and Im(<psi_old|U^dagger|psi_new>)"""

    smpl = sample(num_of_samples, length, loc_dim, params1, fwd, key)
    pushed_smpl, ampls = push_two_qubit_vec(smpl, gate.transpose((2, 3, 0, 1)).conj(), sides)
    denom = log_psi(smpl, loc_dim, params1, fwd)
    nom = log_psi(pushed_smpl.reshape((-1, length)), loc_dim, params2, fwd)
    log_abs = nom[0].reshape((-1, 4)) - denom[0][:, jnp.newaxis]
    phi = nom[1].reshape((-1, 4)) - denom[1][:, jnp.newaxis]
    re = jnp.exp(log_abs) * jnp.cos(phi)
    im = jnp.exp(log_abs) * jnp.sin(phi)
    ampls_re = jnp.real(ampls)
    ampls_im = jnp.imag(ampls)
    re, im = ampls_re * re - ampls_im * im, re * ampls_im + im * ampls_re
    re, im = re.sum(1).mean(), im.sum(1).mean()
    return re, im

def train_step(loss, params1,
               params2, key,
               state, gate,
               sides, num_of_samples,
               opt, loss_and_grad):
    """Makes one training step

    Args:
        loss: real valued jnp scalar, previouse value of loss function
        params1: py tree with parameters of compilled Haiku model,
            old parameters
        params2: py tree with parameters of a compilled Haiku model,
            new parameters to be trained
        key: PRGNKey
        state: py tree, state of an initialized Optax optimizer
        gate: complex valued tensor of shape (2, 2, 2, 2)
        sides: list with two ints specifying positions where to apply a gate
        num_of_samples: int, number of samples to use while optimization step
        opt: an initialized Optax optimizer
        loss_and_grad: function providing loss and its gradient given parameters

    Returns:
        new value of loss, py tree with old parameters, py tree with updated
        new parameters, new PRNGKey, new py tree with state of an optimizer
        """
    key = random.split(key)[0]
    l, grad = loss_and_grad(params1, params2, key, gate, sides, num_of_samples)
    l = pmean(l, axis_name='i')
    grad = pmean(grad, axis_name='i')
    update, state = opt.update(grad, state, params2)
    params2 = optax.apply_updates(params2, update)
    return loss+l, params1, params2, key, state
