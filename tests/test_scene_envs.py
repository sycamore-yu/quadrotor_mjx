"""Tests for scene environments."""

import jax
import jax.numpy as jnp
import pytest
import yaml
from importlib import resources

from dva_quadrotor_mjx.envs.hover_obstacle import HoverObstacleEnv
from dva_quadrotor_mjx.envs.gate_crossing import GateCrossingEnv
from dva_quadrotor_mjx.envs.forest_navigation import ForestNavigationEnv


SCENE_ENVS = (HoverObstacleEnv, GateCrossingEnv, ForestNavigationEnv)
HOVER_ACTION = jnp.array([7.3575, 0.0, 0.0, 0.0])


def _with_pose(state, pos, vel=None):
    qpos = state.pipeline_state.qpos.at[0:3].set(jnp.array(pos))
    qvel = state.pipeline_state.qvel
    if vel is not None:
        qvel = qvel.at[0:3].set(jnp.array(vel))
    return state.replace(
        pipeline_state=state.pipeline_state.replace(qpos=qpos, qvel=qvel)
    )


def test_scene_sensor_mode_configs_cover_required_modes():
    """Scene configs expose state, feature, rangefinder, and RGB/depth modes."""

    required_modes = {"state", "feature", "rangefinder", "rgb_depth"}
    for name in ("hover_obstacle", "gate_crossing", "forest_navigation"):
        path = resources.files("dva_quadrotor_mjx.configs").joinpath(
            "tasks", f"{name}.yaml"
        )
        config = yaml.safe_load(path.read_text())
        sensor_modes = config["sensor_modes"]
        assert required_modes.issubset(set(sensor_modes["available"]))
        for mode in required_modes:
            assert mode in sensor_modes


@pytest.mark.parametrize("env_cls", SCENE_ENVS)
def test_scene_reset_is_deterministic_for_same_seed(env_cls):
    """Every scene reset must be seed-stable for reproducible gates."""

    env = env_cls()
    rng = jax.random.PRNGKey(42)

    state1 = env.reset(rng)
    state2 = env.reset(rng)

    assert jnp.allclose(state1.obs, state2.obs)
    assert jnp.allclose(state1.pipeline_state.qpos, state2.pipeline_state.qpos)


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

        next_state = env.step(state, HOVER_ACTION)

        assert next_state.obs.shape == (23,)
        assert "collision" in next_state.info

    def test_collision_failure(self):
        env = HoverObstacleEnv()
        state = env.reset(jax.random.PRNGKey(0))
        state = _with_pose(state, [0.0, 0.0, 0.5])

        next_state = env.step(state, HOVER_ACTION)

        assert bool(jnp.asarray(next_state.info["collision"]))
        assert float(jnp.asarray(next_state.reward)) < float(jnp.asarray(state.reward))


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

        next_state = env.step(state, HOVER_ACTION)

        assert next_state.obs.shape == (23,)
        assert "current_gate" in next_state.info
        assert "gate_success" in next_state.info

    def test_miss_gate_failure(self):
        env = GateCrossingEnv()
        state = env.reset(jax.random.PRNGKey(0))
        state = _with_pose(state, [1.99, 2.0, 1.0], vel=[3.0, 0.0, 0.0])

        next_state = env.step(state, HOVER_ACTION)

        assert not bool(jnp.asarray(next_state.info["gate_success"]))
        assert int(jnp.asarray(next_state.info["gates_crossed"])) == 0


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

        next_state = env.step(state, HOVER_ACTION)

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

    def test_tree_collision_failure(self):
        env = ForestNavigationEnv()
        state = env.reset(jax.random.PRNGKey(0))
        tree_pos = state.info["tree_positions"][0]
        state = _with_pose(state, tree_pos)

        next_state = env.step(state, HOVER_ACTION)

        assert bool(jnp.asarray(next_state.info["collision"]))

    def test_goal_reach_success(self):
        env = ForestNavigationEnv()
        state = env.reset(jax.random.PRNGKey(0))
        state = _with_pose(state, state.info["goal"])

        next_state = env.step(state, HOVER_ACTION)

        assert bool(jnp.asarray(next_state.info["success"]))
        assert float(jnp.asarray(next_state.done)) == 1.0
