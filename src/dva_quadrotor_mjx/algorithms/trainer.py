"""Unified trainer for smoke rollouts and Brax-backed baseline workflows.

The real DVA/APG work remains in ``openspec/changes/dva-quadrotor-mjx``.  This
module provides a small, stable interface for the rpg-flightning MJX platform
so every environment/algorithm pair can be launched, checkpointed, evaluated,
and visualized without each script reimplementing policy plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, Optional

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.policies import create_train_state, select_action
from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime
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

    ``ppo`` uses the Brax PPO trainer and vectorized training wrappers.
    ``bptt`` and ``shac`` dispatch to their real JAX backends; unsupported
    algorithms fail loudly instead of falling back to diagnostic rollouts.
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
    if algo == "bptt":
        return _train_jax_bptt(
            env,
            network=network,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_envs=num_envs,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs,
        )
    if algo == "shac":
        return _train_jax_shac(
            env,
            network=network,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_envs=num_envs,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs,
        )

    raise ValueError(f"Unsupported algorithm backend {algo!r}")


def _train_jax_bptt(
    env,
    *,
    network,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    learning_rate: float,
    seed: int,
    **kwargs,
) -> TrainResult:
    """Trains BPTT through the differentiable JAX scan backend."""

    configure_jax_runtime()

    from dva_quadrotor_mjx.algorithms import bptt

    train_state = create_train_state(
        network,
        env.observation_size,
        seed=seed,
        learning_rate=learning_rate,
    )
    start = time.time()
    result = bptt.train(
        env,
        train_state=train_state,
        num_epochs=num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        num_envs=num_envs,
        key=jax.random.PRNGKey(seed),
    )
    losses = jax.block_until_ready(result["metrics"])
    runner_state = result["runner_state"]
    final_train_state = runner_state.train_state
    train_time = time.time() - start
    compile_time = float(np.asarray(result.get("compile_time_seconds", 0.0)))
    warmup_time = float(np.asarray(result.get("warmup_time_seconds", 0.0)))
    execute_time = float(np.asarray(result.get("execute_time_seconds", 0.0)))
    loss_values = np.asarray(losses)
    mean_reward = float(-loss_values[-1]) if loss_values.size else float("nan")
    metrics = {
        "algo": "bptt",
        "backend": "jax_bptt",
        "epochs": num_epochs,
        "steps_per_epoch": num_steps_per_epoch,
        "num_envs": num_envs,
        "loss": loss_values.tolist(),
        "mean_reward": mean_reward,
        "compile_time_seconds": compile_time,
        "warmup_time_seconds": warmup_time,
        "execute_time_seconds": execute_time,
        "train_time_seconds": train_time,
        "sps": float(num_epochs * num_steps_per_epoch * num_envs / max(train_time, 1e-9)),
    }
    return TrainResult(
        params=final_train_state.params,
        metrics=metrics,
        train_state=final_train_state,
    )


def _train_jax_shac(
    env,
    *,
    network,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    learning_rate: float,
    seed: int,
    **kwargs,
) -> TrainResult:
    """Trains SHAC through the vendored ``jax_shac`` backend."""

    configure_jax_runtime()

    from dva_quadrotor_mjx.algorithms.shac._impl import (
        train as _shac_train,
        resolve_checkpoint_path as _shac_resolve_checkpoint_path,
    )

    num_timesteps = int(
        kwargs.pop("num_timesteps", num_epochs * num_steps_per_epoch * num_envs)
    )
    checkpoint_dir = Path(kwargs.pop("checkpoint_dir", "artifacts/shac_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if num_timesteps == 0:
        return _init_jax_shac_checkpoint(
            env,
            network=network,
            checkpoint_dir=checkpoint_dir,
            num_steps_per_epoch=num_steps_per_epoch,
            num_envs=num_envs,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs,
        )
    start = time.time()
    result = _shac_train(
        env,
        policy_network=network,
        value_network=network,
        num_timesteps=num_timesteps,
        episode_length=num_steps_per_epoch,
        num_envs=num_envs,
        num_eval_envs=int(kwargs.pop("num_eval_envs", min(num_envs, 128))),
        actor_learning_rate=learning_rate,
        critic_learning_rate=float(kwargs.pop("critic_learning_rate", learning_rate)),
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        eval_env=kwargs.pop("eval_env", env),
        num_evals=num_epochs + 1,
        **kwargs,
    )
    train_time = time.time() - start
    metrics = dict(result["metrics"])
    metrics["train_time_seconds"] = train_time
    resolved_checkpoint = result.get("checkpoint_path")
    if resolved_checkpoint is None:
        resolved_checkpoint = _shac_resolve_checkpoint_path(checkpoint_dir)
    return TrainResult(
        params=result["params"],
        metrics=metrics,
        train_state=None,
        checkpoint_path=resolved_checkpoint,
    )


def _init_jax_shac_checkpoint(
    env,
    *,
    network,
    checkpoint_dir: Path,
    num_steps_per_epoch: int,
    num_envs: int,
    learning_rate: float,
    seed: int,
    **kwargs,
) -> TrainResult:
    """Initializes SHAC state and writes a checkpoint without optimization.

    This is the SHAC smoke path: it exercises the real vendored SHAC network
    and state initialization while avoiding the full short-horizon gradient
    compile.  Full acceptance gates must use ``num_timesteps > 0``.
    """

    from dva_quadrotor_mjx.algorithms.shac.train import SHAC

    trainer = SHAC(
        environment=env,
        num_timesteps=0,
        episode_length=num_steps_per_epoch,
        num_envs=num_envs,
        num_eval_envs=int(kwargs.pop("num_eval_envs", min(num_envs, 128))),
        actor_learning_rate=learning_rate,
        critic_learning_rate=float(kwargs.pop("critic_learning_rate", learning_rate)),
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        eval_env=kwargs.pop("eval_env", env),
        **kwargs,
    )
    key = jax.random.PRNGKey(seed)
    key_env, key_state = jax.random.split(key)
    env_state = trainer.reset_fn(jax.random.split(key_env, num_envs))
    training_state, key_state = trainer.init_training_state(key_state)
    checkpoint_path = checkpoint_dir / "checkpoint.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(
            {
                "epoch_key": key_state,
                "local_key": key_state,
                "training_state": training_state,
                "env_state": env_state,
            },
            handle,
        )
    metrics = {
        "algo": "shac",
        "backend": "jax_shac",
        "num_timesteps": 0,
        "num_envs": num_envs,
        "episode_length": num_steps_per_epoch,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "mean_reward": float("nan"),
        "note": "SHAC smoke initialization only; reward gates require num_timesteps > 0.",
    }
    return TrainResult(
        params=training_state.policy_params,
        metrics=metrics,
        train_state=None,
        checkpoint_path=checkpoint_path,
    )


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

    configure_jax_runtime()

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
            "backend": "brax_ppo",
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


def _run_episode(env, policy_fn: Callable, key: jax.Array, steps: int):
    state = env.reset(key)
    total_reward = 0.0
    episode_length = 0
    success = False
    collision = False

    for _ in range(steps):
        key, action_key = jax.random.split(key)
        action, _ = policy_fn(state.obs, action_key)
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


def eval(
    checkpoint: Any,
    env,
    episodes: int = 10,
    seed: int = 0,
    max_steps: int | None = None,
) -> EvalResult:
    """Evaluates a checkpoint with the shared policy interface."""

    policy_fn = make_policy(checkpoint, env)
    steps = env.max_steps_in_episode if max_steps is None else max_steps
    rewards = []
    lengths = []
    successes = []
    collisions = []
    key = jax.random.PRNGKey(seed)
    for _ in range(episodes):
        key, episode_key = jax.random.split(key)
        episode = _run_episode(env, policy_fn, episode_key, steps)
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


def _compact_training_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps scalar trainer metrics and drops large gradient tensors."""

    compact: Dict[str, Any] = {}
    for key, value in metrics.items():
        array = None
        if isinstance(value, jax.Array):
            array = np.asarray(value)
        elif isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, np.generic):
            compact[key] = value.item()
            continue
        elif isinstance(value, (int, float, str, bool)):
            compact[key] = value
            continue

        if array is None:
            continue
        if array.ndim == 0:
            compact[key] = array.item()
        elif array.size <= 4:
            compact[key] = array.tolist()
        elif key.endswith(("loss", "reward", "sps", "walltime")):
            compact[f"{key}_mean"] = float(np.mean(array))
    return compact


def make_policy(checkpoint: Any, env: Any = None) -> Callable:
    """Creates a unified policy function: policy(obs, key) -> (action, extras)."""

    if isinstance(checkpoint, TrainState):
        def policy_fn(obs, key=None):
            return select_action(checkpoint, obs), {}
        return policy_fn

    path = Path(checkpoint)

    # 1. PPO Checkpoint
    is_ppo = False
    p_path = path if path.is_dir() else path.parent
    if (p_path / "ppo_network_config.json").exists():
        is_ppo = True
    elif "ppo" in str(path).lower():
        is_ppo = True

    if is_ppo:
        from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
        ppo_p = path if path.is_dir() else path.parent
        return load_ppo_policy(ppo_p)

    # 2. SHAC Checkpoint
    is_shac = False
    if "shac" in str(path).lower() or path.suffix == ".pkl" or (path.is_dir() and (path / "checkpoint.pkl").exists()):
        is_shac = True

    if is_shac:
        from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
        return load_shac_policy(path, env)

    # 3. BPTT Checkpoint (Flax TrainState)
    from dva_quadrotor_mjx.policies import create_train_state, make_network
    network = make_network("bptt", env.action_space.shape[0])
    template = create_train_state(
        network,
        env.observation_size,
        seed=0,
        learning_rate=3e-4,
    )
    loaded_state = load_checkpoint(path, template)
    def bptt_policy_fn(obs, key=None):
        return select_action(loaded_state, obs), {}
    return bptt_policy_fn


def save_checkpoint(path: str | Path, checkpoint: Any) -> None:
    """Saves a checkpoint (Flax TrainState or SHAC params)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(checkpoint, TrainState):
        path.write_bytes(flax.serialization.to_bytes(checkpoint))
    else:
        with path.open("wb") as handle:
            pickle.dump(checkpoint, handle)


def load_checkpoint(path: str | Path, train_state: Optional[TrainState] = None) -> Any:
    """Loads a checkpoint (Flax TrainState or SHAC params)."""

    path = Path(path)
    if train_state is not None:
        return flax.serialization.from_bytes(train_state, path.read_bytes())
    with path.open("rb") as handle:
        return pickle.load(handle)
