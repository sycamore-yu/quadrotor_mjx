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
    last_obs: jax.Array
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

    def rollout(runner_state: RunnerState, params):
        def step_fn(old_runner_state: RunnerState, _unused):
            train_state, env_state, last_obs, key, epoch_idx = old_runner_state

            action = train_state.apply_fn(params, last_obs)
            next_state = step_env(env_state, action)

            runner_state = RunnerState(
                train_state, next_state, next_state.obs, key, epoch_idx
            )
            return runner_state, TrajectoryState(reward=next_state.reward)

        return jax.lax.scan(step_fn, runner_state, None, num_steps_per_epoch)

    @jax.jit
    def train_epoch(epoch_state: RunnerState):
        @partial(jax.value_and_grad, has_aux=True)
        def loss_fn(params, runner_state: RunnerState):
            runner_state, trajectory = rollout(runner_state, params)
            loss = -trajectory.reward.sum() / num_envs
            return loss, runner_state

        train_state = epoch_state.train_state
        (loss, epoch_state), grad = loss_fn(train_state.params, epoch_state)
        train_state = train_state.apply_gradients(grads=grad)
        epoch_state = epoch_state._replace(
            train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
        )
        return epoch_state, loss

    # Initialize environments
    key, key_ = jax.random.split(key)
    key_reset = jax.random.split(key_, num_envs)
    env_state = env.reset(key_reset)
    obs = env_state.obs
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    losses = []
    compile_time_seconds = 0.0
    if num_epochs > 0:
        compile_start = time.time()
        runner_state, first_loss = train_epoch(runner_state)
        first_loss = jax.block_until_ready(first_loss)
        compile_time_seconds = time.time() - compile_start
        losses.append(first_loss)

        for _ in range(1, num_epochs):
            runner_state, loss = train_epoch(runner_state)
            losses.append(loss)

    return {
        "runner_state": runner_state,
        "metrics": jnp.stack(losses) if losses else jnp.zeros((0,), dtype=jnp.float32),
        "compile_time_seconds": jnp.asarray(compile_time_seconds, dtype=jnp.float32),
    }
