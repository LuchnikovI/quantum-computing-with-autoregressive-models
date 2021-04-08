from jax import numpy as jnp
from jax import vmap, random
from typing import List
from functools import partial, reduce
from ..utils import _mpo_block_eye_prod


class CircuitMPO:

    def __init__(self, number_of_qubits: int):
        self.number_of_qubits = number_of_qubits

    def init(self) -> List[jnp.ndarray]:
        """Returns initial MPO (MPO representation of the identity matrix)"""
        return self.number_of_qubits * [jnp.eye(2, dtype=jnp.complex64).reshape((2, 1, 2, 1))]

    def apply_gate(self,
                   mpo: List[jnp.ndarray],
                   gate: jnp.ndarray,
                   sides: List[int],
                   eps=1e-6):
        """Update MPO inplace
        Args:
            mpo: list with MPO blocks
            gate: (2, 2, 2, 2) array like
            sides: list with number of spins where to apply a gate
            eps: threshold"""

        if sides[0] > sides[1]:
            sides[0], sides[1] = sides[1], sides[0]
            gate = gate.transpose((1, 0, 3, 2))
        # mpo blocks bond dimensions
        _, left_bond_0, _, right_bond_0 = mpo[sides[0]].shape
        _, left_bond_1, _, right_bond_1 = mpo[sides[1]].shape
        # gate splitting
        gate = gate.transpose((0, 2, 1, 3))
        gate = gate.reshape((4, 4))
        left, right = jnp.linalg.qr(gate)
        # left, s, right = jnp.linalg.svd(gate, full_matrices=False)
        # eta = (s > eps).sum()
        # s, left, right = s[:eta], left[:, :eta], right[:eta]
        # left, right = left * jnp.sqrt(s), right * jnp.sqrt(s)[:, jnp.newaxis]
        left, right = left.reshape((2, 2, -1)), right.reshape((-1, 2, 2))
        # gate mpo blocks dot product
        left = jnp.einsum('ijkl,min->mjkln', mpo[sides[0]], left)
        right = jnp.einsum('ijkl,nmi->mjnkl', mpo[sides[1]], right)
        # setting proper shape of new mpo blocks
        right, left = right.reshape((2, -1, 2, right_bond_1)), left.reshape((2, left_bond_0, 2, -1))
        # update left and right blocks
        mpo[sides[0]], mpo[sides[1]] = left, right
        # update inner blocks
        eye_matrix = jnp.eye(right.shape[0], dtype=gate.dtype)
        mpo[sides[0]+1:sides[1]] = list(map(partial(_mpo_block_eye_prod, eye_matrix=eye_matrix), mpo[sides[0]+1:sides[1]]))
    
    def truncate(self,
                 mpo: List[jnp.ndarray],
                 eps=1e-6):
        """Truncate MPO inplace

        Args:
            mpo: list with MPO blocks
            eps: threshold"""

        for i in range(len(mpo)-1):
            left, right = mpo[i], mpo[i+1].transpose((1, 0, 2, 3))
            left_shape, right_shape = left.shape, right.shape
            left, right = left.reshape((-1, left_shape[-1])), right.reshape((right_shape[0], -1))
            block = left @ right
            u, s, vh = jnp.linalg.svd(block, full_matrices=False)
            eta = (s > eps).sum()
            s, u, vh = s[:eta], u[:, :eta], vh[:eta]
            left, right = u, vh * s[:, jnp.newaxis]
            left, right = left.reshape(left_shape[:-1] + (-1,)), right.reshape((-1,) + right_shape[1:])
            mpo[i], mpo[i+1] = left, right.transpose((1, 0, 2, 3))

    @partial(vmap, in_axes=(None, 0, None, 0))
    def _push_sample(self,
                     inp_sample: jnp.ndarray,
                     mpo: List[jnp.ndarray],
                     key: jnp.ndarray) -> List[jnp.ndarray]:
        """Conditionally samples from MPO

        Args:
            in_samples: input sample
            mpo: mpo representation of a circuit
            key: PRNGKey

        Returns:
            log(u) and corresponding sample"""

        # don't touch this wtf
        inp_sample_list = [per_qubit_sample for per_qubit_sample in inp_sample]
        sample_block_pair = zip(mpo, inp_sample_list)
        mps = list(map(lambda x: x[0][:, :, x[1]], sample_block_pair))
        transfer_matrices =  list(map(lambda x: jnp.einsum('ijk,ilm->jlkm', x, x.conj()), mps))
        def plug_update(y, x):
            x_update = jnp.tensordot(y, x, axes=[[2, 3], [0, 1]])
            return x_update / jnp.linalg.norm(x_update)
        right_plugs =  reduce(lambda x, y: [plug_update(y, x[0])]+x, transfer_matrices[::-1], [jnp.ones((1, 1), dtype=jnp.complex64)])[1:]
        def local_sample(x, y):
            block, plug = y
            u, key, prev_tensor, inds = x
            trial_tensor = jnp.tensordot(prev_tensor, block, [[0], [1]])
            local_dens = trial_tensor @ plug @ trial_tensor.conj().T
            log_p = jnp.log(jnp.diag(local_dens))
            new_ind = jnp.argmax((random.gumbel(key, shape=(2,)) + log_p).astype(jnp.float32))
            new_tensor = trial_tensor[new_ind]
            norm = jnp.linalg.norm(new_tensor)
            new_tensor = new_tensor / norm
            u = jnp.log(norm) + u
            key, _ = random.split(key)
            return u, key, new_tensor, inds + [new_ind]
        u, _, final_tensor, out_sample = reduce(local_sample, zip(mps, right_plugs), (jnp.array(0.), key, jnp.ones((1,), dtype=jnp.complex64), []))
        return u + jnp.log(final_tensor[0]), jnp.stack(out_sample)

    def push_sample(self,
                    inp_sample: jnp.ndarray,
                    mpo: List[jnp.ndarray],
                    key: jnp.ndarray) -> List[jnp.ndarray]:
        """Conditionally samples from MPO

        Args:
            in_samples: input samples
            mpo: mpo representation of a circuit
            key: PRNGKey

        Returns:
            log(u) and corresponding samples"""

        return self._push_sample(inp_sample,
                                 mpo,
                                 random.split(key, inp_sample.shape[0]))