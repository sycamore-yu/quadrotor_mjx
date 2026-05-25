"""Unified trainer for smoke rollouts and Brax-backed baseline workflows.

The real DVA/APG work remains in ``openspec/changes/dva-quadrotor-mjx``.  This
module provides a small, stable interface for the rpg-flightning MJX platform
so every environment/algorithm pair can be launched, checkpointed, evaluated,
and visualized without each script reimplementing policy plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.policies import create_train_state, select_action
from dva_quadrotor_mjx.wrapper import wrap_for_brax_training


@dataclass
class TrainResult:
    """Result of a training or smoke rollout."""

    params: Any
    metrics: Dict[str, Any]
    train_state: Optional[TrainState] = None
    checkpoint_path: Optional[Path] = None


@dataclass
class EvalResult:
    """Result of an evaluation run."""

    mean_reward: float
    mean_episode_length: float
    success_rate: Optional[float] = None
    collision_rate: Optional[float] = None


def _run_episode(env, train_state: TrainState, key: jax.Array, steps: int):
    state = env.reset(key)
    total_reward = 0.0
    episode_length = 0
    success = False
    collision = False

    for _ in range(steps):
        action = select_action(train_state, state.obs)
        state = env.step(state, action)
        total_reward += float(jnp.asarray(state.reward))
        episode_length += 1
        success = success or bool(jnp.asarray(state.info.get("success", False)))
        collision = collision or bool(jnp.asarray(state.info.get("collision", False)))
        if bool(jnp.asarray(state.done)):
            break

    return {
        "reward": total_reward,
        "length": episode_length,
        "success": success,
        "collision": collision,
    }


def train(
    env,
    algo: str,
    network,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 100,
    num_envs: int = 128,
    learning_rate: float = 3e-4,
    seed: int = 0,
    **kwargs,
) -> TrainResult:
    """Runs a trainer with a common algorithm interface.

    ``ppo`` uses the Brax PPO trainer and vectorized training wrappers, matching
    the MuJoCo Playground training chain.  ``bptt`` and ``shac`` remain light
    baseline rollouts until their dedicated algorithm work lands.
    """

    if algo == "ppo":
        return _train_brax_ppo(
            env,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_envs=num_envs,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs,
        )

    train_state = create_train_state(
        network,
        env.observation_size,
        seed=seed,
        learning_rate=learning_rate,
    )

    rewards = []
    lengths = []
    key = jax.random.PRNGKey(seed)
    for epoch in range(num_epochs):
        key, episode_key = jax.random.split(key)
        episode = _run_episode(env, train_state, episode_key, num_steps_per_epoch)
        rewards.append(episode["reward"])
        lengths.append(episode["length"])

    metrics = {
        "algo": algo,
        "epochs": num_epochs,
        "steps_per_epoch": num_steps_per_epoch,
        "requested_num_envs": num_envs,
        "episode_reward": rewards,
        "episode_length": lengths,
        "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
        "mean_episode_length": float(sum(lengths) / max(len(lengths), 1)),
        "note": "Smoke/baseline rollout; long-run learning gates are separate OpenSpec tasks.",
    }
    return TrainResult(params=train_state.params, metrics=metrics, train_state=train_state)


def _train_brax_ppo(
    env,
    *,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    learning_rate: float,
    seed: int,
    **kwargs,
) -> TrainResult:
    """Trains PPO through Brax's vectorized PPO agent."""

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    num_timesteps = int(
        kwargs.pop("num_timesteps", num_epochs * num_steps_per_epoch * num_envs)
    )
    checkpoint_dir = kwargs.pop("checkpoint_dir", None)
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    action_repeat = int(kwargs.pop("action_repeat", 1))
    unroll_length = int(kwargs.pop("unroll_length", min(10, num_steps_per_epoch)))
    unroll_length = max(1, unroll_length)
    num_minibatches = int(kwargs.pop("num_minibatches", 1 if num_timesteps <= num_envs else 8))
    batch_size = int(kwargs.pop("batch_size", max(num_envs, num_envs * unroll_length // num_minibatches)))
    if batch_size * num_minibatches % num_envs != 0:
        batch_size = int(np.ceil(batch_size * num_minibatches / num_envs) * num_envs // num_minibatches)

    run_evals = bool(kwargs.pop("run_evals", num_timesteps > num_envs))
    num_evals = int(kwargs.pop("num_evals", 1))
    num_eval_envs = int(kwargs.pop("num_eval_envs", min(num_envs, 128)))
    normalize_observations = bool(kwargs.pop("normalize_observations", True))

    progress_metrics: Dict[str, Any] = {}

    def progress(_num_steps, metrics):
        progress_metrics.update(metrics)

    make_policy, params, metrics = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        num_envs=num_envs,
        episode_length=num_steps_per_epoch,
        action_repeat=action_repeat,
        wrap_env_fn=wrap_for_brax_training,
        learning_rate=learning_rate,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=int(kwargs.pop("num_updates_per_batch", 1)),
        normalize_observations=normalize_observations,
        num_evals=num_evals,
        num_eval_envs=num_eval_envs,
        run_evals=run_evals,
        network_factory=kwargs.pop("network_factory", ppo_networks.make_ppo_networks),
        seed=seed,
        progress_fn=progress,
        save_checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        **kwargs,
    )

    del make_policy
    merged_metrics = dict(metrics)
    merged_metrics.update(progress_metrics)
    merged_metrics.update(
        {
            "algo": "ppo",
            "trainer": "brax_ppo",
            "num_timesteps": num_timesteps,
            "num_envs": num_envs,
            "episode_length": num_steps_per_epoch,
            "unroll_length": unroll_length,
            "batch_size": batch_size,
            "num_minibatches": num_minibatches,
            "checkpoint_dir": str(checkpoint_path) if checkpoint_path else None,
        }
    )
    mean_reward = merged_metrics.get("eval/episode_reward")
    if mean_reward is not None:
        merged_metrics["mean_reward"] = float(np.asarray(mean_reward))
    else:
        merged_metrics["mean_reward"] = float("nan")

    return TrainResult(
        params=params,
        metrics=merged_metrics,
        train_state=None,
        checkpoint_path=checkpoint_path,
    )


def eval(
    checkpoint: TrainState,
    env,
    episodes: int = 10,
    seed: int = 0,
    max_steps: int | None = None,
) -> EvalResult:
    """Evaluates a TrainState checkpoint with the shared policy interface."""

    steps = env.max_steps_in_episode if max_steps is None else max_steps
    rewards = []
    lengths = []
    successes = []
    collisions = []
    key = jax.random.PRNGKey(seed)
    for _ in range(episodes):
        key, episode_key = jax.random.split(key)
        episode = _run_episode(env, checkpoint, episode_key, steps)
        rewards.append(episode["reward"])
        lengths.append(episode["length"])
        successes.append(episode["success"])
        collisions.append(episode["collision"])

    return EvalResult(
        mean_reward=float(sum(rewards) / max(len(rewards), 1)),
        mean_episode_length=float(sum(lengths) / max(len(lengths), 1)),
        success_rate=float(sum(successes) / max(len(successes), 1)),
        collision_rate=float(sum(collisions) / max(len(collisions), 1)),
    )


def make_policy(checkpoint: TrainState) -> Callable:
    """Creates a normalized-action policy from a TrainState checkpoint."""

    def policy(obs):
        return select_action(checkpoint, obs)

    return policy


def save_checkpoint(path: str | Path, train_state: TrainState) -> None:
    """Saves a Flax TrainState checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(train_state))


def load_checkpoint(path: str | Path, train_state: TrainState) -> TrainState:
    """Loads a Flax TrainState checkpoint using ``train_state`` as template."""

    return flax.serialization.from_bytes(train_state, Path(path).read_bytes())
