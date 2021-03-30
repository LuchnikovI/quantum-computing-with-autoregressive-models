"""
This module contains 2 quantum computing backend wrappers -- neural-network and exact.
"""


import pickle
import time
import jax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
from ..wave_functions.attention_wave_function import AttentionWaveFunction


class NeuralQCWrapper:
    """Wrapper for neural networks based quantum computation.

    Args:
        number_of_heads: int number, number of heads in MultiHeadAttention
        kqv_size: int number, size of key, value and query for all layers
        number_of_layers: int number, number of layers
        length: int, length of a chain
        key: PRNGKey"""

    def __init__(self, number_of_heads, kqv_size, number_of_layers, length, key):

        self.wave_func = AttentionWaveFunction(
            number_of_heads, kqv_size, number_of_layers, 2
        )
        self.params, self.fwd, self.length = self.wave_func.init(key, length)
        self.circuit = []
        self.training_data = []
        self.key = key
        self.num_devices = jax.local_device_count()
        self.opt = None
        self.opt_state = None

    def add_gate(self, gate, sides):
        """Adds gate to the circuit.

        Args:
            gate: complex valued tensor of shape (2, 2, 2, 2)
            sides: list with two int values showing where to apply a gate"""

        self.circuit.append([gate, sides])

    def set_optimizer(self, opt):
        """Sets optax optimizer"""

        self.opt = opt
        one_device_params = jax.tree_util.tree_map(lambda x: x[0], self.params[0])
        self.opt_state = jax.tree_util.tree_map(
            lambda x: jnp.stack(8 * [x]), opt.init(one_device_params)
        )

    def train_qc(self, epoch_size, iters, num_of_samples):
        """Calculates output of a quantum circuit.

        Args:
            epoch_size: int, size of one epoch
            iters: int, number of epoch
            number_of_samples: int, number of samples per iteration"""

        for layer_num, layer in enumerate(self.circuit):
            gate_time = time.time()
            loss_dynamics = []
            self.key = random.split(self.key)[0]
            keys = random.split(self.key, self.num_devices)
            for i in tqdm(range(iters)):
                compilation_time = time.time()
                loss, self.params, keys, self.opt_state = self.wave_func.train_epoch(
                    layer[0],
                    layer[1],
                    self.opt,
                    self.opt_state,
                    num_of_samples,
                    keys,
                    epoch_size,
                    self.params,
                    self.fwd,
                    self.length,
                )
                loss_dynamics.append(loss[0])
                if i == 0:
                    print(", Compilation time = " + str(time.time() - compilation_time))
            opt_state = self.opt.init(
                jax.tree_util.tree_map(lambda x: x[0], self.params[1])
            )
            opt_state = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * self.num_devices), opt_state
            )
            self.opt_state = opt_state
            self.params[0] = self.params[1]
            self.training_data.append({"loss_dynamics": loss_dynamics})
            print("Gate time = " + str(time.time() - gate_time))
            print("Gate #" + str(layer_num) + ", infidelity = " + str(loss[0]))
            with open("qc_net_" + str(layer_num) + ".pickle", "wb") as f:
                pickle.dump(self.params[0], f)

    def get_network(self):
        """Returns output of a circuit in the form of NN"""

        return self.wave_func

    def get_training_data(self):
        """Returns log of training"""

        return self.training_data


class ExactQCWrapper:
    """Wrapper for exact simmulation of quantum computations

    Args:
        length: int, length of a chain"""

    def __init__(self, length):
        self.state = jnp.ones(2 ** length, dtype=jnp.complex64) / jnp.sqrt(2 ** length)
        self.state = self.state.reshape(length * (2,))
        self.circuit = []
        self.length = length

    def add_gate(self, gate, sides):
        """Adds gate to the circuit.

        Args:
            gate: complex valued tensor of shape (2, 2, 2, 2)
            sides: list with two int values showing where to apply a gate"""

        self.circuit.append([gate, sides])

    def train_qc(self):
        """Calculates output of a quantum circuit."""

        for layer in self.circuit:
            gate = layer[0]
            sides = layer[1]
            if sides[0] > sides[1]:
                min_index = sides[1]
                max_index = sides[0]
                first_edge = self.length - 1
                second_edge = self.length - 2
            else:
                min_index = sides[0]
                max_index = sides[1]
                first_edge = self.length - 2
                second_edge = self.length - 1
            new_ord = (
                tuple(range(min_index))
                + (first_edge,)
                + tuple(range(min_index, max_index - 1))
                + (second_edge,)
                + tuple(range(max_index - 1, self.length - 2))
            )
            self.state = jnp.tensordot(self.state, gate, axes=[sides, [2, 3]])
            self.state = self.state.transpose(new_ord)

    def get_output_state(self):
        """Returns output state"""

        return self.state
