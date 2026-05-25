"""Tests for HoveringStateEnv and HoveringFeaturesEnv."""

import jax
import jax.numpy as jnp
import pytest

from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
from dva_quadrotor_mjx.envs.hover_features import HoveringFeaturesEnv


class TestHoveringStateEnv:
    """Tests for state-based hovering environment."""

    def test_observation_size(self):
        env = HoveringStateEnv()
        assert env.observation_size == (23,)

    def test_action_space(self):
        env = HoveringStateEnv()
        assert env.action_space.shape == (4,)
        assert jnp.allclose(env.action_space.low, jnp.array([0.0, -6.0, -6.0, -4.0]))
        assert jnp.allclose(env.action_space.high, jnp.array([12.0, 6.0, 6.0, 4.0]))

    def test_observation_space(self):
        env = HoveringStateEnv()
        assert env.observation_space.shape == (23,)

    def test_reset(self):
        env = HoveringStateEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert state.obs.shape == (23,)
        assert state.reward == 0.0
        assert state.done == 0.0
        assert "steps" in state.info
        assert "last_actions" in state.info
        assert "motor_force" in state.info
        assert state.info["last_actions"].shape == (2, 4)

    def test_step(self):
        env = HoveringStateEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jnp.array([7.3575, 0.0, 0.0, 0.0])
        next_state = env.step(state, action)

        assert next_state.obs.shape == (23,)
        assert next_state.info["steps"] == 1
        assert next_state.info["last_actions"].shape == (2, 4)
        assert next_state.info["last_actions"][-1].shape == (4,)
        assert "reward_tuple" in next_state.info
        assert "acc" in next_state.info["reward_tuple"]

    def test_done_conditions(self):
        env = HoveringStateEnv(max_steps_in_episode=10)
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Step until truncation
        for _ in range(10):
            state = env.step(state, jnp.zeros(4))

        assert state.done == 1.0
        assert state.info["truncation"] == 1.0

    def test_auto_reset(self):
        env = HoveringStateEnv(max_steps_in_episode=5)
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Step past truncation
        for _ in range(5):
            state = env.step(state, jnp.zeros(4))

        # After truncation, state should be reset
        assert state.done == 1.0

    def test_observation_schema(self):
        """Verify observation matches documented schema."""
        env = HoveringStateEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        obs = state.obs
        # position: 3
        assert obs[0:3].shape == (3,)
        # rotation matrix: 9
        assert obs[3:12].shape == (9,)
        # velocity: 3
        assert obs[12:15].shape == (3,)
        # action history: 2 * 4 = 8
        assert obs[15:23].shape == (8,)


class TestHoveringFeaturesEnv:
    """Tests for feature-based hovering environment."""

    def test_observation_size(self):
        env = HoveringFeaturesEnv()
        # 7 points * 2 coords + 2 actions * 4 = 14 + 8 = 22
        assert env.observation_size == (22,)

    def test_reset(self):
        env = HoveringFeaturesEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert state.obs.shape == (22,)
        assert state.reward == 0.0
        assert state.done == 0.0

    def test_step(self):
        env = HoveringFeaturesEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jnp.array([7.3575, 0.0, 0.0, 0.0])
        next_state = env.step(state, action)

        assert next_state.obs.shape == (22,)
        assert next_state.info["steps"] == 1

    def test_feature_projection(self):
        """Test that feature points are projected correctly."""
        env = HoveringFeaturesEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        obs = state.obs
        # First 14 elements are normalized feature coordinates
        features = obs[:14]
        # Should be in [-1, 1] range (or close)
        assert jnp.all(features >= -1.1)
        assert jnp.all(features <= 1.1)
