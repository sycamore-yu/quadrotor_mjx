"""PPO (Proximal Policy Optimization) algorithm for MJX environments.

Reference: third_party/rpg_flightning/flightning/algos/ppo.py
"""

from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.envs.base import State
from dva_quadrotor_mjx.wrappers.logging import LogWrapper
from dva_quadrotor_mjx.wrappers.vectorization import VecEnv


class Config(NamedTuple):
    """PPO hyperparameters."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    num_minibatches: int = 4
    logging_freq: int = 10
    logging: bool = True


class PPOSample(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: State
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


class UpdateState(NamedTuple):
    train_state: TrainState
    trajectory: PPOSample
    advantages: jax.Array
    targets: jax.Array
    key: chex.PRNGKey


class LossInfo(NamedTuple):
    value_loss: jax.Array
    actor_loss: jax.Array
    entropy: jax.Array
    total_loss: jax.Array


def train(
    env,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    key: chex.PRNGKey,
    config: Config,
):
    """Train using PPO.

    Args:
        env: Brax-compatible environment
        train_state: Flax TrainState with Actor-Critic network
        num_epochs: Number of training epochs
        num_steps_per_epoch: Steps per epoch per env
        num_envs: Number of parallel environments
        key: JAX PRNGKey
        config: PPO hyperparameters

    Returns:
        dict with "runner_state" and "metrics"
    """
    env = LogWrapper(env)
    env = VecEnv(env)

    minibatch_size = num_envs * num_steps_per_epoch // config.num_minibatches

    def _train(runner_state: RunnerState):
        def epoch_fn(runner_state: RunnerState, _unused):
            def step_fn(runner_state: RunnerState, _unused):
                train_state, env_state, last_obs, key, epoch_idx = runner_state

                # Select action
                key, key_ = jax.random.split(key)
                pi, value = train_state.apply_fn(train_state.params, last_obs)
                action = pi.sample(seed=key_)
                log_prob = pi.log_prob(action)

                # Step env
                key, key_ = jax.random.split(key)
                key_step = jax.random.split(key_, num_envs)
                next_state = env.step(env_state, action, key_step)
                done = next_state.done

                transition = PPOSample(
                    done, action, value, next_state.reward, log_prob, last_obs, next_state.info
                )
                runner_state = RunnerState(
                    train_state, next_state, next_state.obs, key, epoch_idx
                )
                return runner_state, transition

            runner_state, samples = jax.lax.scan(
                step_fn, runner_state, None, num_steps_per_epoch
            )

            # Calculate advantage
            train_state, env_state, last_obs, key, epoch_idx = runner_state
            _, last_value = train_state.apply_fn(train_state.params, last_obs)

            def calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, sample):
                    gae, next_value = gae_and_next_value
                    done = sample.done
                    value = sample.value
                    reward = sample.reward

                    delta = (
                        reward + config.gamma * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = calculate_gae(samples, last_value)

            # Update network
            def update_epoch(update_state: UpdateState, _unused):
                def update_minibatch(
                    train_state: TrainState,
                    batch_info: Tuple[PPOSample, jax.Array, jax.Array],
                ):
                    traj_batch, advantages, targets = batch_info

                    @partial(jax.value_and_grad, has_aux=True)
                    def loss_fn(params, traj_batch, gae, targets):
                        # Rerun network
                        pi, value = train_state.apply_fn(
                            params, traj_batch.obs
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Calculate value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )

                        # Calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )

                        loss_info = LossInfo(
                            value_loss, loss_actor, entropy, total_loss
                        )

                        return total_loss, loss_info

                    (_, loss_info), grads = loss_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, key = (
                    update_state
                )
                key, key_ = jax.random.split(key)
                batch_size = minibatch_size * config.num_minibatches
                permutation = jax.random.permutation(key_, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    update_minibatch, train_state, minibatches
                )
                update_state = UpdateState(
                    train_state, traj_batch, advantages, targets, key
                )
                return update_state, loss_info

            update_state = UpdateState(
                train_state, samples, advantages, targets, key
            )
            update_state, loss_info = jax.lax.scan(
                update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state.train_state
            metric = samples.info
            key = update_state.key

            runner_state = RunnerState(
                train_state, env_state, last_obs, key, epoch_idx + 1
            )
            return runner_state, metric

        # Run epochs
        runner_state, metric = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )
        return {"runner_state": runner_state, "metrics": metric}

    # Initialize environments
    key, key_ = jax.random.split(key)
    reset_rng = jax.random.split(key_, num_envs)
    env_state = env.reset(reset_rng)
    obs = env_state.obs
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    return jax.jit(_train)(runner_state)
