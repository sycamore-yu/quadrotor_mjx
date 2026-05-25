"""Shared policy networks and checkpoint helpers for CLI entry points."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax


SUPPORTED_ALGOS = ("bptt", "ppo", "shac")


class DeterministicMLP(nn.Module):
    """Deterministic normalized-action policy used by BPTT smoke paths."""

    hidden_dims: tuple[int, ...] = (128, 128)
    action_dim: int = 4

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        return nn.tanh(nn.Dense(self.action_dim)(x))


class ActorCriticMLP(nn.Module):
    """Small actor-critic network for PPO/SHAC smoke paths."""

    hidden_dims: tuple[int, ...] = (128, 128)
    action_dim: int = 4

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        actor = nn.tanh(nn.Dense(self.action_dim)(x))
        value = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        return actor, value


def make_network(algo: str, action_dim: int) -> nn.Module:
    """Creates the default policy module for an algorithm name."""

    if algo == "bptt":
        return DeterministicMLP(action_dim=action_dim)
    if algo in ("ppo", "shac"):
        return ActorCriticMLP(action_dim=action_dim)
    raise ValueError(f"Unknown algorithm {algo!r}. Expected one of {SUPPORTED_ALGOS}.")


def create_train_state(
    network: nn.Module,
    obs_shape: tuple[int, ...],
    *,
    seed: int,
    learning_rate: float,
) -> TrainState:
    """Initializes a TrainState template shared by train/eval/render scripts."""

    params = network.init(jax.random.PRNGKey(seed), jnp.zeros(obs_shape))
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )


def select_action(train_state: TrainState, obs: jax.Array) -> jax.Array:
    """Returns a normalized action in [-1, 1] from either policy interface."""

    out: Any = train_state.apply_fn(train_state.params, obs)
    if isinstance(out, tuple):
        out = out[0]
    return jnp.clip(out, -1.0, 1.0)
