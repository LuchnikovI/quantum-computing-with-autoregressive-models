import jax
from jax import jit, random, pmap, value_and_grad
from jax.lax import fori_loop
import jax.numpy as jnp
import haiku as hk
from attention import AttentionEncoder, log_psi, sample, two_qubit_gate_braket, train_step

class AttentionQC:
    """Attention network-based quantum computing emulator

    Args:
        number_of_heads: int number, number of heads in MultiHeadAttention
        kqv_size: int number, size of key, value and query for all layers
        number_of_layers: int number, number of layers
        length: int number, length of a chain
        key: PRNGKey
        loc_dim: int number, the dimension of a local Hilbert space"""

    def __init__(self, number_of_heads,
                 kqv_size, number_of_layers,
                 length, key, loc_dim=2):

        def _forward(x):
            return AttentionEncoder(number_of_heads,
                                    kqv_size,
                                    number_of_layers,
                                    depth=loc_dim)(x)

        self.num_devices = jax.local_device_count()
        forward = hk.without_apply_rng(hk.transform(_forward))
        params = forward.init(key, jnp.ones((1, 1), dtype=jnp.int32))
        params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * self.num_devices), params)
        self.params1 = pmap(lambda x: x)(params)
        self.params2 = pmap(lambda x: x)(params)
        fwd = jit(forward.apply)
        
        self.logpsi = pmap(lambda string, params: log_psi(string, loc_dim, params, fwd))
        
        self.smpl = pmap(lambda num_of_samples, params, key: sample(num_of_samples, length, loc_dim, params, fwd, key), in_axes=(None, 0, 0), static_broadcasted_argnums=0)
        
        self.braket = pmap(lambda params1, params2, key, gate, sides, num_of_samples: two_qubit_gate_braket(params1, params2, key, gate, sides, num_of_samples, length, loc_dim, fwd), in_axes=(0, 0, 0, None, None, None), static_broadcasted_argnums=(3, 4, 5))
        
        self.opt = None
        
        self.state = None
        
        def loss_func(params1, params2, key, gate, sides, num_of_samples):
            re, _ = two_qubit_gate_braket(params1, params2, key, gate, sides, num_of_samples, length, loc_dim, fwd)
            return 1 - re
        loss_and_grad = value_and_grad(loss_func, 1)
        self.train_step = lambda loss, params1, params2, key, state, gate, sides, num_of_samples, opt: train_step(loss, params1, params2, key, state, gate, sides, num_of_samples, opt, loss_and_grad)
        
        self.p_epoch_train = None

    def sample(self, num_of_samples, keys):
        """Makes samples from |psi|^2, (parallel on TPU kers or GPUs)
        
        Args:
            num_of_samples: int, number of samples
            keys: PRNGKeys distributed across devices

        Returns:
            real valued tensor of shape (num_of_devices, num_of_samples, length)"""

        return self.smpl(num_of_samples, self.params2, keys)
    
    def log_psi(self, string):
        """Returns real and imag parts of log(psi), (parallel on TPU kers or GPUs)

        Args:
            string: int valued tensor of shape (bs, length)
    
        Returns:
            two tensors of shape (num_of_devices, bs)"""

        return self.logpsi(string, self.params2)

    def gate_braket(self, num_of_samples, gate, sides, keys):
        """Returns <psi_old|gate^dagger|psi_new>

        Args:
            num_of_samples: int, number of samples
            gate: complex valued tensor of shape (2, 2, 2, 2)
            sides: list with to ints representing sides where to apply
                a gate
            keys: PRNGKeys distributed across devices

        Returns:
            two real valued tensors of shape (num_of_devices,),
            Re(<psi_old|U^dagger|psi_new>) and Im(<psi_old|U^dagger|psi_new>)"""

        return self.braket(self.params1, self.params2, keys, gate, sides, num_of_samples)

    def set_optimizer(self, opt):
        """Sets an optax optimizer"""

        self.opt = opt
        state = self.opt.init(jax.tree_util.tree_map(lambda x: x[0], self.params2))
        state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * self.num_devices), state)
        state = pmap(lambda x: x)(state)
        self.state = state
        def epoch_train(params1, params2, key, state, gate, sides, num_of_samples, iters):
            train = lambda i, vals: self.train_step(*vals, gate, sides, num_of_samples, opt)
            l, _, params2, key, state = fori_loop(0, iters, train, (jnp.array(0.), params1, params2, key, state))
            return l / iters, params2, state, key
        self.p_epoch_train = pmap(epoch_train, in_axes=(0, 0, 0, 0, None, None, None, None), static_broadcasted_argnums=(4, 5, 6, 7), axis_name='i')

    def reset_optimizer_state(self):
        """Resets the optimizer state"""
        
        # self.state = jax.tree_util.tree_map(pmap(lambda x: jnp.zeros(x.shape, dtype=x.dtype)), self.state)
        state = self.opt.init(jax.tree_util.tree_map(lambda x: x[0], self.params2))
        state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * self.num_devices), state)
        state = pmap(lambda x: x)(state)
        self.state = state
    
    def train_epoch(self, keys, gate, sides, num_of_samples, iters):
        """Trains one epoch

        Args:
            keys: PRNGKeys distributed across devices
            gate: complex valued tensor of shape (2, 2, 2, 2)
            sides: list with to ints representing sides where to apply
                a gate
            num_of_samples: int, number of samples
            iters: int, number of iterations within one epoch

        Returns:
            value of a loss function and new PRNGKeys distributed across devices"""

        l, self.params2, self.state, new_keys = self.p_epoch_train(self.params1, self.params2, keys, self.state, gate, sides, num_of_samples, iters)
        return l, new_keys
    
    def fix_training_result(self):
        """Sets value of the old parameters equal to new parameters"""

        self.params1 = self.params2
