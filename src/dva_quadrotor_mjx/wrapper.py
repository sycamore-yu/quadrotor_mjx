"""MuJoCo Playground-style wrappers for Brax training agents."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, Wrapper
from brax.envs.wrappers import training as brax_training


def wrap_for_brax_training(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
    **_: Any,
) -> Wrapper:
    """Wraps an env with the same public contract as MuJoCo Playground.

    Our State uses Brax's ``pipeline_state`` field, so the official Brax
    wrappers match this project better than Playground's MJX-specific auto reset
    wrapper, which expects a ``data`` field.
    """

    return brax_training.wrap(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )
