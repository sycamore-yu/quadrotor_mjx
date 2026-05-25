"""BPTT (Backpropagation Through Time) algorithm for MJX environments.

Reference: third_party/rpg_flightning/flightning/algos/bptt.py
"""

from functools import partial
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

    def _train(runner_state: RunnerState):
        def epoch_fn(epoch_state: RunnerState, _unused):

            @partial(jax.value_and_grad, has_aux=True)
            def loss_fn(params, runner_state: RunnerState):

                def rollout(runner_state: RunnerState):
                    def step_fn(old_runner_state: RunnerState, _unsused):
                        train_state, env_state, last_obs, key, epoch_idx = (
                            old_runner_state
                        )

                        # Deterministic action
                        action = train_state.apply_fn(params, last_obs)

                        # Env step
                        key, key_ = jax.random.split(key)
                        key_step = jax.random.split(key_, num_envs)
                        next_state = env.step(env_state, action, key_step)

                        runner_state = RunnerState(
                            train_state, next_state, next_state.obs, key, epoch_idx
                        )

                        return (
                            runner_state,
                            TrajectoryState(reward=next_state.reward),
                        )

                    runner_state, trajectory = jax.lax.scan(
                        step_fn, runner_state, None, num_steps_per_epoch
                    )
                    return runner_state, trajectory

                # Collect data
                runner_state, trajectory = rollout(runner_state)
                loss = -trajectory.reward.sum() / num_envs
                return loss, runner_state

            # Compute reward
            train_state = epoch_state.train_state
            (loss, epoch_state), grad = loss_fn(
                train_state.params, epoch_state
            )
            # Update params
            train_state = train_state.apply_gradients(grads=grad)

            epoch_state = epoch_state._replace(
                train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
            )

            return epoch_state, loss

        # Run epochs
        runner_state_final, losses = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )

        return {"runner_state": runner_state_final, "metrics": losses}

    # Initialize environments
    key, key_ = jax.random.split(key)
    key_reset = jax.random.split(key_, num_envs)
    env_state = env.reset(key_reset)
    obs = env_state.obs
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    return jax.jit(_train)(runner_state)
