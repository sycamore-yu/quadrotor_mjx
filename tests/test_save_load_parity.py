"""Tests for checkpoint save/load parity across different algorithms."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.algorithms import trainer
from dva_quadrotor_mjx.envs.base import State
from dva_quadrotor_mjx.envs import load
from dva_quadrotor_mjx.learning import shac_checkpoint
from dva_quadrotor_mjx.policies import create_train_state, make_network
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


class _DummySpace:
    shape = (4,)


class _TinyBpttEnv:
    action_space = _DummySpace()
    observation_size = (23,)
    max_steps_in_episode = 2
    model = SimpleNamespace(nq=1, nv=1)

    def reset(self, _key):
        return State(
            pipeline_state=jnp.zeros((1,), dtype=jnp.float32),
            obs=jnp.zeros((23,), dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            done=jnp.asarray(False),
            metrics={},
            info={},
        )

    def step(self, state, action):
        # Keep the state simple but make the reward depend on the policy output
        # so the real BPTT backend has a nontrivial gradient path.
        reward = jnp.sum(action)
        return State(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=reward,
            done=jnp.asarray(False),
            metrics={},
            info={},
        )


def _assert_params_match(actual: TrainState, expected: TrainState) -> None:
    assert actual.step == expected.step
    matches = jax.tree_util.tree_map(
        lambda left, right: jnp.allclose(left, right),
        actual.params,
        expected.params,
    )
    assert all(bool(leaf) for leaf in jax.tree_util.tree_leaves(matches))


def test_bptt_save_load_parity(tmp_path):
    """Verify compiled BPTT checkpoints preserve params and action parity."""

    env = _TinyBpttEnv()
    network = make_network("bptt", env.action_space.shape[0])

    result = trainer.train(
        env,
        algo="bptt",
        network=network,
        num_epochs=2,
        num_steps_per_epoch=1,
        num_envs=2,
        seed=42,
    )

    assert result.metrics["backend"] == "jax_bptt"
    assert result.metrics["num_envs"] == 2
    assert isinstance(result.metrics["loss"], list)
    assert len(result.metrics["loss"]) == 2
    assert result.metrics["compile_time_seconds"] >= 0.0
    assert result.metrics["warmup_time_seconds"] >= 0.0
    assert result.metrics["train_time_seconds"] >= 0.0
    assert result.metrics["sps"] > 0.0
    assert result.train_state is not None
    assert result.train_state.step == 2

    ckpt_path = tmp_path / "bptt_test.ckpt"
    trainer.save_checkpoint(ckpt_path, result.train_state)

    state_loaded = create_train_state(
        network,
        env.observation_size,
        seed=0,
        learning_rate=3e-4,
    )
    state_loaded = trainer.load_checkpoint(ckpt_path, state_loaded)

    _assert_params_match(result.train_state, state_loaded)

    policy_orig = trainer.make_policy(result.train_state, env)
    policy_loaded = trainer.make_policy(ckpt_path, env)

    obs = jnp.zeros(env.observation_size)
    act_orig, _ = policy_orig(obs)
    act_loaded, _ = policy_loaded(obs)

    np.testing.assert_allclose(act_orig, act_loaded, rtol=1e-5, atol=1e-5)


def test_shac_save_load_parity(tmp_path):
    """Verify SHAC pickle checkpointing and reconstruction succeeds."""

    import types as _types

    env = NormalizeActionWrapper(load("quadrotor_hover"))

    from dva_quadrotor_mjx.algorithms.shac.networks import make_shac_networks
    from brax.training.acme import running_statistics
    from brax.training import types as training_types

    obs_size = env.observation_size
    action_size = env.action_space.shape[0]
    shac_nets = make_shac_networks(
        obs_size, action_size,
        preprocess_observations_fn=running_statistics.normalize)
    key = jax.random.PRNGKey(42)
    policy_params = shac_nets.policy_network.init(key)

    # Build proper RunningStatisticsState for normalizer_params
    normalizer_params = running_statistics.RunningStatisticsState(
        mean=jnp.zeros(obs_size),
        std=jnp.ones(obs_size),
        count=training_types.UInt64(hi=0, lo=0),
        summed_variance=jnp.zeros(obs_size),
    )

    training_state = _types.SimpleNamespace(
        policy_params=policy_params,
        normalizer_params=normalizer_params,
    )
    ckpt_dict = {
        "epoch_key": key,
        "local_key": key,
        "training_state": training_state,
        "env_state": None,
    }

    ckpt_path = tmp_path / "checkpoint.pkl"
    trainer.save_checkpoint(ckpt_path, ckpt_dict)

    # Load and execute SHAC policy
    policy = trainer.make_policy(ckpt_path, env)
    obs = jnp.zeros(obs_size)
    act, _ = policy(obs, key)
    assert act.shape == (action_size,)
    assert jnp.all(act >= -1.0) and jnp.all(act <= 1.0)
