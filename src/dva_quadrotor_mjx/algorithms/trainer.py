"""Unified trainer seam for smoke and baseline CLI workflows.

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
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.policies import create_train_state, select_action


@dataclass
class TrainResult:
    """Result of a training or smoke rollout."""

    params: Any
    metrics: Dict[str, Any]
    train_state: Optional[TrainState] = None


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
    """Runs a deterministic baseline trainer with a common algorithm interface.

    ``bptt`` uses a deterministic actor. ``ppo``/``shac`` use the actor
    head of an actor-critic module.  The current implementation is deliberately
    small: it validates environment/algorithm/checkpoint wiring and leaves
    long-run learning gates unchecked until the dedicated algorithm work lands.
    """

    del kwargs
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
