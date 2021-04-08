"""
This module contains the attention-based neural-network wave function.
"""


from functools import partial
from typing import Tuple, List, Any
import jax.numpy as jnp
from jax import pmap
import jax
import haiku as hk
from ..attention.attention import AttentionEncoder
from ..utils import (
    Params,
    PRNGKey,
    NNet,
    _log_amplitude,
    _sample,
    _two_qubit_gate_bracket,
    _train_epoch,
    _circ_bracket,
)


class AttentionWaveFunction:
    """Attention network-based wave function

    Args:
        number_of_heads: number of heads in MultiHeadAttention
        kqv_size: size of key, value and query for all layers
        number_of_layers: number of layers
        number_of_wave_functions: number of wave functions"""

    def __init__(
        self,
        number_of_heads: int,
        kqv_size: int,
        number_of_layers: int,
        number_of_wave_functions: int,
    ):

        self._number_of_heads = number_of_heads
        self._kqv_size = kqv_size
        self._number_of_layers = number_of_layers
        self._number_of_wave_functions = number_of_wave_functions

    def init(self, key: PRNGKey, qubits_num: int) -> Tuple[List[Params], NNet, int]:
        """Initializes wave function

        Args:
            key: PRNGKey
            qubits_num: number of qubits

        Returns:
            network parameters, network and number of qubits"""

        def _forward(x):
            return AttentionEncoder(
                self._number_of_heads, self._kqv_size, self._number_of_layers
            )(x)

        # attention compilation
        forward = hk.without_apply_rng(hk.transform(_forward))
        params = forward.init(key, jnp.ones((1, 1), dtype=jnp.int32))
        num_devices = jax.local_device_count()
        params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)

        return self._number_of_wave_functions * [params], forward.apply, qubits_num

    @partial(
        pmap,
        in_axes=(None, None, 0, None, 0, None, None),
        out_axes=0,
        static_broadcasted_argnums=(0, 1, 3, 5, 6),
    )
    def sample(
        self,
        num_of_samples: int,
        key: PRNGKey,
        wave_function_number: int,
        params: List[Params],
        fwd: NNet,
        qubits_num: int,
    ) -> jnp.array:
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

        return _sample(
            num_of_samples, key, wave_function_number, params, fwd, qubits_num
        )

    @partial(
        pmap,
        in_axes=(None, 0, None, 0, None, None),
        out_axes=0,
        static_broadcasted_argnums=(0, 2, 4, 5),
    )
    def log_amplitude(
        self,
        sample: jnp.ndarray,
        wave_function_number: int,
        params: List[Params],
        fwd: NNet,
        qubits_num: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return log(wave function) for a given set of bit strings.

        Args:
            sample: (num_of_samples, length) array like
            wave_function_number: number of a wave function to evaluate
            params: parameters
            fwd: network
            qubits_num: number of qubits

        Returns:
            two array like (num_of_samples,) -- log of absolut value and phase"""

        return _log_amplitude(sample, wave_function_number, params, fwd, qubits_num)

    @partial(
        pmap,
        in_axes=(None, None, None, None, 0, None, 0, None, None),
        out_axes=0,
        static_broadcasted_argnums=(0, 2, 3, 5, 7, 8),
    )
    def two_qubit_gate_bracket(
        self,
        gate: jnp.ndarray,
        sides: List[int],
        wave_function_numbers: List[int],
        key: PRNGKey,
        num_of_samples: int,
        params: List[Params],
        fwd: NNet,
        qubits_num: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

        return _two_qubit_gate_bracket(
            gate,
            sides,
            wave_function_numbers,
            key,
            num_of_samples,
            params,
            fwd,
            qubits_num,
        )

    @partial(
        pmap,
        in_axes=(None, None, None, None, 0, None, 0, None, None),
        out_axes=0,
        static_broadcasted_argnums=(0, 2, 3, 5, 7, 8),
    )
    def circ_bracket(
        self,
        mpo: List[jnp.ndarray],
        circ: Any,
        wave_function_numbers: List[int],
        key: PRNGKey,
        num_of_samples: int,
        params: List[Params],
        fwd: NNet,
        qubits_num: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates <psi_1|mpo|psi_2>

        Args:
            mpo: mpo representation of a circuit
            circ: object of class circuit
            wave_function_numbers: list with two int values specifying numbers
                of wave functions
            key: PRNGKey
            num_of_samples: number of samples
            params: parameters
            fwd: network
            qubits_num: number of qubits

        Returns:
            two array like of shape (1,)"""

        return _circ_bracket(mpo,
                             circ,
                             wave_function_numbers,
                             key,
                             num_of_samples,
                             params,
                             fwd,
                             qubits_num)

    @partial(
        pmap,
        in_axes=(None, None, None, None, 0, None, 0, None, 0, None, None),
        out_axes=0,
        static_broadcasted_argnums=(0, 2, 3, 5, 7, 9, 10),
        axis_name="i",
    )
    def train_epoch(
        self,
        gate: jnp.ndarray,
        sides: List[int],
        opt: Any,
        opt_state: Any,
        num_of_samples: int,
        key: PRNGKey,
        epoch_size: int,
        params: List[Params],
        fwd: NNet,
        qubits_num: int,
    ) -> Tuple[jnp.array, List[Params], PRNGKey, Any]:
        """Makes training epoch

        Args:
            gate: (2, 2, 2, 2) array like
            sides: list with two elements showing where to apply a gate
            opt: optax optimizer
            opt_state: state of an optax optimizer
            num_of_samples: number of samples used to evaluate loss function
            key: PRGNKey
            epoch_size: number of iterations
            params: parameters of wave function
            fwd: network
            qubit_num: number of qubits

        Returns:
            loss function value, new set of parameters, new PRNGKey,
            optimizer state"""

        return _train_epoch(
            gate,
            sides,
            opt,
            opt_state,
            num_of_samples,
            key,
            epoch_size,
            params,
            fwd,
            qubits_num,
        )

    @partial(
        pmap,
        in_axes=(None, None, None, None, 0, None, 0, None, 0, None, None),
        out_axes=0,
        static_broadcasted_argnums=(0, 2, 3, 5, 7, 9, 10),
        axis_name="i",
    )
    def train_epoch_circ(
        self,
        mpo: List[jnp.ndarray],
        circ: Any,
        opt: Any,
        opt_state: Any,
        num_of_samples: int,
        key: PRNGKey,
        epoch_size: int,
        params: List[Params],
        fwd: NNet,
        qubits_num: int,
    ) -> Tuple[jnp.array, List[Params], PRNGKey, Any]:
        """Makes training epoch

        Args:
            mpo: mpo representation of a circuit
            circ: circuit
            opt: optax optimizer
            opt_state: state of an optax optimizer
            num_of_samples: number of samples used to evaluate loss function
            key: PRGNKey
            epoch_size: number of iterations
            params: parameters of wave function
            fwd: network
            qubit_num: number of qubits

        Returns:
            loss function value, new set of parameters, new PRNGKey,
            optimizer state"""

        return _train_epoch(
            mpo,
            circ,
            opt,
            opt_state,
            num_of_samples,
            key,
            epoch_size,
            params,
            fwd,
            qubits_num,
        )
