"""Tests for scene environments."""

import jax
import jax.numpy as jnp
import pytest

from dva_quadrotor_mjx.envs.hover_obstacle import HoverObstacleEnv
from dva_quadrotor_mjx.envs.gate_crossing import GateCrossingEnv
from dva_quadrotor_mjx.envs.forest_navigation import ForestNavigationEnv


class TestHoverObstacleEnv:
    """Tests for hover obstacle environment."""

    def test_reset(self):
        env = HoverObstacleEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert state.obs.shape == (23,)
        assert state.reward == 0.0
        assert state.done == 0.0

    def test_step(self):
        env = HoverObstacleEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jnp.array([7.3575, 0.0, 0.0, 0.0])
        next_state = env.step(state, action)

        assert next_state.obs.shape == (23,)
        assert "collision" in next_state.info


class TestGateCrossingEnv:
    """Tests for gate crossing environment."""

    def test_reset(self):
        env = GateCrossingEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert state.obs.shape == (23,)
        assert "current_gate" in state.info
        assert "gates_crossed" in state.info

    def test_step(self):
        env = GateCrossingEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jnp.array([7.3575, 0.0, 0.0, 0.0])
        next_state = env.step(state, action)

        assert next_state.obs.shape == (23,)
        assert "current_gate" in next_state.info
        assert "gate_success" in next_state.info


class TestForestNavigationEnv:
    """Tests for forest navigation environment."""

    def test_reset(self):
        env = ForestNavigationEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert state.obs.shape == (23,)
        assert "tree_positions" in state.info
        assert "tree_radii" in state.info
        assert "goal" in state.info

    def test_step(self):
        env = ForestNavigationEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jnp.array([7.3575, 0.0, 0.0, 0.0])
        next_state = env.step(state, action)

        assert next_state.obs.shape == (23,)
        assert "collision" in next_state.info
        assert "dist_to_goal" in next_state.info

    def test_deterministic_reset(self):
        """Test that reset with same seed produces same state."""
        env = ForestNavigationEnv()
        rng = jax.random.PRNGKey(42)

        state1 = env.reset(rng)
        state2 = env.reset(rng)

        assert jnp.allclose(state1.obs, state2.obs)
        assert jnp.allclose(state1.pipeline_state.qpos, state2.pipeline_state.qpos)
