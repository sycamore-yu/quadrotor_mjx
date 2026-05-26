"""Helpers for loading vendored SHAC checkpoints as inference policies."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from dva_quadrotor_mjx.algorithms import shac as shac_backend


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Returns a concrete SHAC pickle checkpoint path."""

    return shac_backend.resolve_checkpoint_path(path)


def load_policy(path: str | Path, env, *, deterministic: bool = True) -> Callable:
    """Loads a callable policy from a SHAC checkpoint.

    The vendored SHAC checkpoint stores the algorithm state in pickle form,
    not a Flax ``TrainState``.  This helper reconstructs the matching SHAC
    networks and returns a policy with the same ``policy(obs, key) -> (action,
    extras)`` contract used by Brax policies.
    """

    checkpoint = resolve_checkpoint_path(path)
    return shac_backend.load_policy(checkpoint, env, deterministic=deterministic)
