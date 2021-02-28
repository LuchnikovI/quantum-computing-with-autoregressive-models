from attention_qc import AttentionQC
import jax
from jax import random

class NeuralQCWrapper:
    """Wrapper for neural networks based quantum computation.
    
    Args:
        Args:
        list_of_heads_sizes: list with number of heads per layer
        list_of_QK_sizes: list with dimensions of key and query per layer
        list_of_V_sizes: list with dimensions of value per layer
        length: int, length of a chain
        key: PRNGKey"""

    def __init__(self, list_of_heads_sizes,
                 list_of_QK_sizes,
                 list_of_V_sizes,
                 length,
                 key):
        self.qc = AttentionQC(list_of_heads_sizes,
                              list_of_QK_sizes,
                              list_of_V_sizes,
                              length,
                              key)
        self.circuit = []
        self.training_data = []
        self.key = key
        self.num_devices = jax.local_device_count()

    def add_gate(self, gate, sides):
        """Adds gate to the circuit.
        
        Args:
            gate: complex valued tensor of shape (2, 2, 2, 2)
            sides: list with two int values showing where to apply a gate"""

        self.circuit.append([gate, sides])
        
    def set_optimizer(self, opt):
        """Sets optax optimizer"""

        self.qc.set_optimizer(opt)
    
    def train_qc(self, epoch_size, iters, num_of_samples):
        """Calculates output of a quantum circuit.

        Args:
            epoch_size: int, size of one epoch
            iters: int, number of epoch
            number_of_samples: int, number of samples per iteration"""

        for layer in self.circuit:
            loss_dynamics = []
            self.key = random.split(self.key)[0]
            keys = random.split(self.key, self.num_devices)
            for _ in range(iters):
                loss, keys = self.qc.train_epoch(keys, layer[0], layer[1], num_of_samples, epoch_size)
                loss_dynamics.append(loss[0])
            self.qc.reset_optimizer_state()
            self.qc.fix_training_result()
            self.training_data.append({'loss_dynamics': loss_dynamics})
    
    def get_network(self):
        """Returns output of a circuit in the form of NN"""
        return self.qc
    
    def get_training_data(self):
        """Returns log of training"""

        return self.training_data