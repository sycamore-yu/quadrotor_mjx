"""Tests for save/load functionality."""

import jax
import jax.numpy as jnp
import pytest
import tempfile
import os

from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from dva_quadrotor_mjx.algorithms.trainer import save_checkpoint, load_checkpoint


class DummyNetwork(nn.Module):
    """Simple network for testing."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        return x


class TestSaveLoad:
    """Tests for checkpoint save/load."""

    def test_save_and_load(self):
        """Test that saved checkpoint can be loaded correctly."""
        network = DummyNetwork()
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((23,))
        params = network.init(key, dummy_obs)

        tx = optax.adam(1e-3)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        # Save
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            path = f.name

        save_checkpoint(path, train_state)
        assert os.path.exists(path)

        # Load
        loaded_state = load_checkpoint(path, train_state)

        # Verify params match
        def check_match(p1, p2):
            return jnp.allclose(p1, p2)

        match = jax.tree_util.tree_map(
            check_match, train_state.params, loaded_state.params
        )
        all_match = all(jax.tree_util.tree_leaves(match))
        assert all_match

        # Cleanup
        os.unlink(path)

    def test_load_preserves_structure(self):
        """Test that loaded checkpoint preserves structure."""
        network = DummyNetwork()
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((23,))
        params = network.init(key, dummy_obs)

        tx = optax.adam(1e-3)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            path = f.name

        save_checkpoint(path, train_state)
        loaded_state = load_checkpoint(path, train_state)

        # Check structure matches
        assert loaded_state.step == train_state.step
        assert loaded_state.apply_fn == train_state.apply_fn

        os.unlink(path)
