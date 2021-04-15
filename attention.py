"""
This module contains the Haiku model of autoregressive attention and relevant utilities.
"""

import jax
import jax.numpy as jnp
import haiku as hk


def positional_encoding(T, d):
    """Returns a matrix of positional codes of fixed size.

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
        name: name of the network"""

    def __init__(
        self,
        number_of_heads,
        kqv_size,
        number_of_layers,
        max_length=128,
        name="AttentionEncoder",
    ):

        super().__init__(name=name)
        self.number_of_heads = number_of_heads
        self.kqv_size = kqv_size  # size of the key, value and query
        self.number_of_layers = number_of_layers
        self.positional_encoding = positional_encoding(
            max_length, kqv_size * number_of_heads
        )  # positional encoding
        self.out_size = 4
        self.hidden_size = kqv_size * number_of_heads  # size of hidden representation

    def __call__(self, x):

        # build mask necessary for the autoregressive property
        shape = x.shape
        length = shape[-1]
        mask = jnp.ones((length, length))
        mask = jnp.tril(mask, 0)

        # build embedding of the input seq
        x = hk.Embed(2, self.hidden_size)(x)

        # + pos. encoding
        x = x + self.positional_encoding[:length]

        # for loop over layers
        for _ in range(self.number_of_layers):
            # attention layer
            skip = x
            x = hk.MultiHeadAttention(self.number_of_heads, self.kqv_size, 1)(
                x, x, x, mask=mask
            )

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
