"""Project-owned SHAC backend facade.

This file is the canonical runtime boundary for the vendored ``jax_shac``
adapter. The package ``dva_quadrotor_mjx.algorithms.shac`` re-exports these
symbols so callers can keep using the package import path while the actual
backend logic stays here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolves a SHAC checkpoint file or directory to a concrete pickle."""

    path = Path(path).resolve()
    if path.is_file():
        return path

    checkpoint = path / "checkpoint.pkl"
    if checkpoint.exists():
        return checkpoint

    candidates = sorted(path.glob("checkpoint_*.pkl"))
    if candidates:
        return candidates[-1]

    raise ValueError(f"No SHAC checkpoint found under {path}")


def load_policy(
    path: str | Path,
    env,
    *,
    deterministic: bool = True,
    scalar_var: bool = False,
) -> Callable:
    """Loads a deterministic policy callable from a SHAC checkpoint."""

    from dva_quadrotor_mjx.algorithms.shac import eval_utils as shac_eval_utils
    from dva_quadrotor_mjx.algorithms.shac import networks as shac_networks

    checkpoint = resolve_checkpoint_path(path)
    make_policy, params = shac_eval_utils.init_SHAC_policy_from_checkpoint(
        env,
        checkpoint,
        shac_networks.make_shac_networks,
        scalar_var=scalar_var,
    )
    return make_policy(params, deterministic=deterministic)


def make_policy(checkpoint: str | Path, env, *, deterministic: bool = True) -> Callable:
    """Compatibility helper returning ``policy(obs, key) -> (action, extras)``."""

    return load_policy(checkpoint, env, deterministic=deterministic)


def train(
    env,
    policy_network,
    value_network,
    num_timesteps: int = 100000,
    episode_length: int = 1000,
    num_envs: int = 128,
    actor_learning_rate: float = 1e-3,
    critic_learning_rate: float = 1e-3,
    unroll_length: int = 10,
    discounting: float = 0.99,
    seed: int = 0,
    progress_fn: Callable[[int, Dict[str, Any]], None] = lambda *args: None,
    **kwargs,
):
    """Trains SHAC through the vendored ``jax_shac`` implementation."""

    from dva_quadrotor_mjx.algorithms.shac.train import SHAC

    checkpoint_dir = Path(kwargs.get("checkpoint_dir", "artifacts/shac_checkpoints"))
    trainer = SHAC(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_envs=num_envs,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        unroll_length=unroll_length,
        discounting=discounting,
        seed=seed,
        progress_fn=progress_fn,
        **kwargs,
    )

    make_policy_fn, policy_params, value_params, metrics = trainer.train()
    raw_metrics = dict(metrics)
    compact_metrics = _compact_training_metrics(raw_metrics)
    if "training/policy_loss" in raw_metrics:
        compact_metrics["actor_loss"] = float(np.mean(np.asarray(raw_metrics["training/policy_loss"])))
    if "training/v_loss" in raw_metrics:
        compact_metrics["value_loss"] = float(np.mean(np.asarray(raw_metrics["training/v_loss"])))
    if "eval/episode_reward" in raw_metrics:
        compact_metrics["mean_reward"] = float(np.asarray(raw_metrics["eval/episode_reward"]))

    compact_metrics.update(
        {
            "algo": "shac",
            "backend": "jax_shac",
            "num_timesteps": int(num_timesteps),
            "num_envs": int(num_envs),
            "episode_length": int(episode_length),
            "unroll_length": int(unroll_length),
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    checkpoint_path = _resolve_existing_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        compact_metrics["checkpoint_path"] = str(checkpoint_path)

    return {
        "params": policy_params,
        "value_params": value_params,
        "metrics": compact_metrics,
        "make_policy": make_policy_fn,
        "checkpoint_path": checkpoint_path,
    }


def _resolve_existing_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if checkpoint_dir.is_file():
        return checkpoint_dir
    if (checkpoint_dir / "checkpoint.pkl").exists():
        return checkpoint_dir / "checkpoint.pkl"
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    if candidates:
        return candidates[-1]
    return None


def _compact_training_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, bool, int, float)):
            compact[key] = value
            continue

        array = None
        if hasattr(value, "__array__"):
            array = np.asarray(value)
        elif isinstance(value, np.ndarray):
            array = value

        if array is None:
            continue
        if array.ndim == 0:
            compact[key] = array.item()
        elif array.size <= 4:
            compact[key] = array.tolist()
        elif key.endswith(("loss", "reward", "sps", "walltime")):
            compact[f"{key}_mean"] = float(np.mean(array))
    return compact
