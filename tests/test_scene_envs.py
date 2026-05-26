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


def _with_pose(state, pos, vel=None, quat=None):
    qpos = state.pipeline_state.qpos.at[0:3].set(jnp.asarray(pos))
    if quat is not None:
        qpos = qpos.at[3:7].set(jnp.asarray(quat))
    qvel = state.pipeline_state.qvel
    if vel is not None:
        qvel = qvel.at[0:3].set(jnp.asarray(vel))
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
    """Every scene reset must be seed-stable for reproducible layouts."""

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

    def test_clearance_collision_and_target_hold(self):
        env = HoverObstacleEnv()
        state = env.reset(jax.random.PRNGKey(0))

        clearance_state = env.step(
            _with_pose(state, [1.0, 1.0, 1.2], vel=[0.0, 0.0, 0.0]),
            HOVER_ACTION,
        )
        target_hold_state = env.step(
            _with_pose(state, [0.0, 0.0, 1.2], vel=[0.0, 0.0, 0.0]),
            HOVER_ACTION,
        )
        collision_state = env.step(
            _with_pose(state, [0.0, 0.0, 0.5], vel=[0.0, 0.0, 0.0]),
            HOVER_ACTION,
        )

        assert not bool(jnp.asarray(clearance_state.info["collision"]))
        assert not bool(jnp.asarray(target_hold_state.info["collision"]))
        assert bool(jnp.asarray(collision_state.info["collision"]))
        assert float(jnp.asarray(target_hold_state.reward)) > float(
            jnp.asarray(clearance_state.reward)
        )
        assert float(jnp.asarray(clearance_state.reward)) > float(
            jnp.asarray(collision_state.reward)
        )


class TestGateCrossingEnv:
    """Tests for gate crossing environment."""

    def test_reset(self):
        env = GateCrossingEnv()
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert state.obs.shape == (23,)
        assert "current_gate" in state.info
        assert "gates_crossed" in state.info

    def test_pass_gate_advances_waypoint(self):
        env = GateCrossingEnv(max_steps_in_episode=2)
        state = env.reset(jax.random.PRNGKey(0))

        next_state = env.step(
            _with_pose(state, [1.99, 0.0, 1.0], vel=[3.0, 0.0, 0.0]),
            HOVER_ACTION,
        )

        assert bool(jnp.asarray(next_state.info["gate_success"]))
        assert not bool(jnp.asarray(next_state.info["collision"]))
        assert int(jnp.asarray(next_state.info["gates_crossed"])) == 1
        assert int(jnp.asarray(next_state.info["current_gate"])) == 1
        assert float(jnp.asarray(next_state.done)) == 0.0

    def test_miss_gate_leaves_waypoint_unchanged(self):
        env = GateCrossingEnv(max_steps_in_episode=2)
        state = env.reset(jax.random.PRNGKey(0))

        next_state = env.step(
            _with_pose(state, [1.0, 2.0, 1.0], vel=[0.0, 0.0, 0.0]),
            HOVER_ACTION,
        )

        assert not bool(jnp.asarray(next_state.info["gate_success"]))
        assert not bool(jnp.asarray(next_state.info["collision"]))
        assert int(jnp.asarray(next_state.info["gates_crossed"])) == 0
        assert int(jnp.asarray(next_state.info["current_gate"])) == 0

    def test_collide_gate_penalizes_and_terminates(self):
        env = GateCrossingEnv(max_steps_in_episode=2)
        state = env.reset(jax.random.PRNGKey(0))

        miss_state = env.step(
            _with_pose(state, [1.0, 2.0, 1.0], vel=[0.0, 0.0, 0.0]),
            HOVER_ACTION,
        )
        collision_state = env.step(
            _with_pose(state, [1.99, 0.0, 1.7], vel=[3.0, 0.0, 0.0]),
            HOVER_ACTION,
        )

        assert bool(jnp.asarray(collision_state.info["collision"]))
        assert not bool(jnp.asarray(collision_state.info["gate_success"]))
        assert float(jnp.asarray(collision_state.done)) == 1.0
        assert float(jnp.asarray(collision_state.reward)) < float(
            jnp.asarray(miss_state.reward)
        )

    def test_timeout_truncates_episode(self):
        env = GateCrossingEnv(max_steps_in_episode=1)
        state = env.reset(jax.random.PRNGKey(0))

        next_state = env.step(
            _with_pose(state, [1.0, 2.0, 1.0], vel=[0.0, 0.0, 0.0]),
            HOVER_ACTION,
        )

        assert not bool(jnp.asarray(next_state.info["gate_success"]))
        assert not bool(jnp.asarray(next_state.info["collision"]))
        assert float(jnp.asarray(next_state.info["truncation"])) == 1.0
        assert float(jnp.asarray(next_state.done)) == 1.0


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

    def test_seed_stable_tree_layout(self):
        env = ForestNavigationEnv()
        rng = jax.random.PRNGKey(42)

        state1 = env.reset(rng)
        state2 = env.reset(rng)

        assert jnp.allclose(state1.info["tree_positions"], state2.info["tree_positions"])
        assert jnp.allclose(state1.info["tree_radii"], state2.info["tree_radii"])
        assert jnp.allclose(state1.info["goal"], state2.info["goal"])

    def test_tree_collision_failure(self):
        env = ForestNavigationEnv()
        state = env.reset(jax.random.PRNGKey(0))
        tree_pos = state.info["tree_positions"][0]
        state = _with_pose(state, tree_pos, vel=[0.0, 0.0, 0.0])

        next_state = env.step(state, HOVER_ACTION)

        assert bool(jnp.asarray(next_state.info["collision"]))
        assert not bool(jnp.asarray(next_state.info["success"]))

    def test_goal_reach_success(self):
        env = ForestNavigationEnv()
        state = env.reset(jax.random.PRNGKey(0))
        state = _with_pose(state, state.info["goal"], vel=[0.0, 0.0, 0.0])

        next_state = env.step(state, HOVER_ACTION)

        assert bool(jnp.asarray(next_state.info["success"]))
        assert bool(jnp.asarray(next_state.done))

    def test_rangefinder_hit_detects_controlled_tree_geometry(self):
        env = ForestNavigationEnv(sensor_mode="rangefinder")
        state = env.reset(jax.random.PRNGKey(0))

        # Place a single cylinder directly ahead of the drone so the raycast
        # hit is deterministic and independent of the random forest layout.
        info = dict(state.info)
        info["tree_positions"] = jnp.array([[2.0, 0.0, 1.0]])
        info["tree_radii"] = jnp.array([0.5])

        qpos = state.pipeline_state.qpos.at[0:3].set(jnp.array([0.0, 0.0, 1.0]))
        qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = state.pipeline_state.qvel.at[0:6].set(jnp.zeros(6))

        data = env.pipeline_init(qpos, qvel)
        obs = env._get_obs(data, info)

        tree_hit = float(jnp.asarray(obs[-1]))

        assert obs.shape == (25,)
        assert tree_hit == pytest.approx(0.15, abs=0.05)
