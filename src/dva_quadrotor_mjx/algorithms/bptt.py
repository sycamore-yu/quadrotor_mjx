"""BPTT (Backpropagation Through Time) algorithm for MJX environments.

Reference:
- third_party/rpg_flightning/flightning/algos/bptt.py
- third_party/mujoco/mjx/training_apg.ipynb
"""

from functools import partial
import time
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.envs.base import State
from dva_quadrotor_mjx.wrappers.logging import LogWrapper
from dva_quadrotor_mjx.wrappers.vectorization import VecEnv


class TrajectoryState(PyTreeNode):
    reward: jnp.array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: State
    key: chex.PRNGKey
    epoch_idx: int


def _checkpoint(fn):
    """Applies rematerialization using the best available JAX checkpoint policy."""

    policy = getattr(
        jax.checkpoint_policies,
        "dots_with_no_batch_dims_saveable",
        None,
    )
    if policy is None:
        return jax.checkpoint(fn)
    return jax.checkpoint(fn, policy=policy)


def train(
    env,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    key: chex.PRNGKey,
):
    """Train a deterministic policy using BPTT.

    Args:
        env: Brax-compatible environment
        train_state: Flax TrainState with deterministic policy network
        num_epochs: Number of training epochs
        num_steps_per_epoch: Steps per epoch per env
        num_envs: Number of parallel environments
        key: JAX PRNGKey

    Returns:
        dict with "runner_state" and "metrics"
    """
    env = LogWrapper(env)
    env = VecEnv(env)

    step_env = _checkpoint(env.step)
    warmup_step_env = jax.jit(step_env)

    def rollout(runner_state: RunnerState, params):
        reward_dtype = runner_state.env_state.reward.dtype

        def step_fn(carry, _unused):
            old_runner_state, total_reward = carry
            train_state, env_state, key, epoch_idx = old_runner_state

            action = train_state.apply_fn(params, env_state.obs)
            next_state = step_env(env_state, action)

            runner_state = RunnerState(train_state, next_state, key, epoch_idx)
            total_reward = total_reward + jnp.sum(next_state.reward)
            return (runner_state, total_reward), None

        init = (runner_state, jnp.zeros((), dtype=reward_dtype))
        (runner_state, total_reward), _ = jax.lax.scan(
            step_fn, init, None, num_steps_per_epoch
        )
        return runner_state, TrajectoryState(reward=total_reward)

    def train_epoch(epoch_state: RunnerState, _unused):
        @partial(jax.value_and_grad, has_aux=True)
        def loss_fn(params, runner_state: RunnerState):
            runner_state, trajectory = rollout(runner_state, params)
            loss = -trajectory.reward / num_envs
            return loss, runner_state

        train_state = epoch_state.train_state
        (loss, epoch_state), grad = loss_fn(train_state.params, epoch_state)
        train_state = train_state.apply_gradients(grads=grad)
        epoch_state = epoch_state._replace(
            train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
        )
        return epoch_state, loss

    @partial(jax.jit, donate_argnums=(0,))
    def training_updates(runner_state: RunnerState):
        return jax.lax.scan(train_epoch, runner_state, None, length=num_epochs)

    def training_updates_with_timing(runner_state: RunnerState):
        compile_start = time.time()
        compiled_updates = training_updates.lower(runner_state).compile()
        compile_time_seconds = time.time() - compile_start

        train_start = time.time()
        runner_state, losses = compiled_updates(runner_state)
        losses = jax.block_until_ready(losses)
        train_time_seconds = time.time() - train_start
        return runner_state, losses, compile_time_seconds, train_time_seconds

    # Initialize environments
    key, key_ = jax.random.split(key)
    key_reset = jax.random.split(key_, num_envs)
    env_state = env.reset(key_reset)
    obs = env_state.obs
    runner_state = RunnerState(train_state, env_state, key, epoch_idx=0)

    warmup_time_seconds = 0.0
    losses = jnp.zeros((0,), dtype=jnp.float32)
    compile_time_seconds = 0.0
    execute_time_seconds = 0.0
    if num_epochs > 0:
        warmup_action = jnp.zeros(
            (num_envs,) + tuple(env.action_space.shape),
            dtype=obs.dtype,
        )
        warmup_start = time.time()
        warmup_state = warmup_step_env(env_state, warmup_action)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_state)
        warmup_time_seconds = time.time() - warmup_start

        runner_state, losses, compile_time_seconds, execute_time_seconds = (
            training_updates_with_timing(runner_state)
        )

    return {
        "runner_state": runner_state,
        "metrics": losses,
        "compile_time_seconds": jnp.asarray(compile_time_seconds, dtype=jnp.float32),
        "warmup_time_seconds": jnp.asarray(warmup_time_seconds, dtype=jnp.float32),
        "execute_time_seconds": jnp.asarray(execute_time_seconds, dtype=jnp.float32),
    }
