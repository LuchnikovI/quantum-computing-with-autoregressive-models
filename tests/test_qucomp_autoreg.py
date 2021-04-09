"""This module contains tests."""

import jax.numpy as jnp
import pytest
from jax import random
from qucomp_autoreg.utils import (
    _push_two_qubit,
    _push_one_qubit,
    _sample,
    _log_amplitude,
    _two_qubit_gate_bracket,
    _mpo_block_eye_prod,
    _contract_mpo_tensors,
    _contract_mpo_tensors,
)

# Question: how to test functions which have a fwd argument?
# where to take the NN from? train here from scratch?


def test_push_two_qubit():

    return None


def test_push_one_qubit():

    return None


def test_sample():

    return None


def test_log_amplitude():

    return None


def test_two_qubit_gate_bracket():

    return None


def test_mpo_block_eye_prod():

    key_1 = random.PRNGKey(42)

    for _ in range(100):

        key_1, _ = random.split(key_1)
        random_mpo = random.uniform(key_1, shape=(2, 3, 2, 7))
        phys_dim, left_bond, _, right_bond = random_mpo.shape

        eye_dim = 5
        eye = jnp.identity(eye_dim)
        mpo_eye_prod = _mpo_block_eye_prod(random_mpo, eye)

        mpo_eye_prod_to_compare = jnp.tensordot(eye, random_mpo, axes=0)
        mpo_eye_prod_to_compare = mpo_eye_prod_to_compare.transpose((2, 3, 0, 4, 5, 1))
        mpo_eye_prod_to_compare = mpo_eye_prod_to_compare.reshape(
            (phys_dim, eye_dim * left_bond, phys_dim, eye_dim * right_bond)
        )

        assert jnp.equal(mpo_eye_prod, mpo_eye_prod_to_compare).all()

    return None


def test_contract_mpo_tensors():

    key_1 = random.PRNGKey(42)

    for _ in range(100):

        key_1, key_2 = random.split(key_1)
        tensor_1 = random.uniform(key_1, shape=(2, 3, 2, 7))
        tensor_2 = random.uniform(key_2, shape=(2, 7, 2, 3))

        contracted_tensor = _contract_mpo_tensors(tensor_1, tensor_2)

        contracted_tensor_einsum = jnp.einsum(
            "ijkl, mlno -> ijkmno", tensor_1, tensor_2
        )

        assert contracted_tensor.shape == contracted_tensor_einsum.shape
        assert jnp.equal(contracted_tensor, contracted_tensor_einsum).all()

    return


def test_mpo_to_dense():

    return None
