"""Acceptance gate runner for OpenSpec rpg-flightning-mujoco-platform."""

from __future__ import annotations

from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime
configure_jax_runtime()

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from dva_quadrotor_mjx.configs.acceptance import (
    AcceptanceConfig,
    AcceptanceMetricGate,
    AcceptanceThresholds,
    _threshold_is_weaker,
    apply_threshold_overrides,
    expected_backend_for_algo,
    load_acceptance_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate an acceptance gate")
    parser.add_argument("--config", required=True, help="YAML acceptance config")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for acceptance artifacts. Defaults to config.output_dir.",
    )
    parser.add_argument(
        "--min-reward-delta",
        type=float,
        help="Higher-only override for min_reward_delta.",
    )
    parser.add_argument(
        "--primary-metric-threshold",
        type=float,
        help="Higher-only override for the primary metric threshold.",
    )
    parser.add_argument(
        "--primary-metric-vs-random-delta",
        type=float,
        help="Higher-only override for the delta vs random baseline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a plan artifact without training or evaluation.",
    )
    parser.add_argument(
        "--validate-artifact",
        type=str,
        help="Validate an existing acceptance artifact without training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_acceptance_config(args.config)
    thresholds = apply_threshold_overrides(
        config.thresholds,
        min_reward_delta=args.min_reward_delta,
        primary_metric_threshold=args.primary_metric_threshold,
        primary_metric_vs_random_delta=args.primary_metric_vs_random_delta,
    )
    config = replace(config, thresholds=thresholds)

    output_dir = Path(args.output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_artifact:
        artifact = json.loads(Path(args.validate_artifact).read_text())
        validate_acceptance_artifact(artifact, config, thresholds)
        print(f"Validated acceptance artifact: {args.validate_artifact}")
        return 0

    if args.dry_run:
        plan = build_dry_run_artifact(config, thresholds, output_dir)
        plan_path = output_dir / f"{config.env}_{config.algo}_acceptance_plan.json"
        plan_path.write_text(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
        print(f"Dry-run plan written to: {plan_path}")
        return 0

    artifact = run_acceptance(config, thresholds=thresholds, output_dir=output_dir)
    artifact_path = output_dir / f"{config.env}_{config.algo}_acceptance.json"
    artifact_path.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True))

    print(f"Acceptance artifact: {artifact_path}")
    print(f"Pass: {artifact['pass']}")
    return 0 if artifact["pass"] else 1


def run_acceptance(
    config: AcceptanceConfig,
    *,
    thresholds: AcceptanceThresholds | None = None,
    output_dir: Path | None = None,
    train_fn: Callable[..., Any] | None = None,
    create_train_state_fn: Callable[..., Any] | None = None,
    make_network_fn: Callable[..., Any] | None = None,
    save_checkpoint_fn: Callable[..., Any] | None = None,
    load_checkpoint_fn: Callable[..., Any] | None = None,
    load_ppo_policy_fn: Callable[..., Any] | None = None,
    load_env_fn: Callable[..., Any] | None = None,
    wrap_env_fn: Callable[..., Any] | None = None,
    eval_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Runs the acceptance gate and returns a JSON-serialisable artifact."""

    from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint, save_checkpoint, train
    from dva_quadrotor_mjx.envs.registry import load
    from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
    from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
    from dva_quadrotor_mjx.policies import create_train_state, make_network, select_action
    from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper

    thresholds = thresholds or config.thresholds
    output_dir = Path(output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fn = train_fn or train
    create_train_state_fn = create_train_state_fn or create_train_state
    make_network_fn = make_network_fn or make_network
    save_checkpoint_fn = save_checkpoint_fn or save_checkpoint
    load_checkpoint_fn = load_checkpoint_fn or load_checkpoint
    load_ppo_policy_fn = load_ppo_policy_fn or load_ppo_policy
    load_env_fn = load_env_fn or load
    wrap_env_fn = wrap_env_fn or NormalizeActionWrapper
    eval_fn = eval_fn or _evaluate_policy_on_seeds

    expected_backend = config.expected_backend or expected_backend_for_algo(config.algo)
    metric_names = [
        config.thresholds.primary_metric_name,
        *[gate.name for gate in config.thresholds.additional_metrics],
    ]
    runs: list[dict[str, Any]] = []

    for train_seed in config.train_seeds:
        run_dir = output_dir / f"seed_{train_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        env = wrap_env_fn(load_env_fn(config.env))
        network = make_network_fn(config.algo, env.action_space.shape[0])
        template_state = create_train_state_fn(
            network,
            env.observation_size,
            seed=train_seed,
            learning_rate=config.learning_rate,
        )

        initial_metrics = eval_fn(
            env,
            _deterministic_policy(template_state, select_action),
            config.eval_seeds,
            episodes=config.eval_episodes,
            steps=_resolve_eval_steps(config, env),
            primary_metric_name=thresholds.primary_metric_name,
            metric_names=metric_names,
        )
        random_metrics = eval_fn(
            env,
            _random_policy(env.action_space.shape[0]),
            config.eval_seeds,
            episodes=config.eval_episodes,
            steps=_resolve_eval_steps(config, env),
            primary_metric_name=thresholds.primary_metric_name,
            metric_names=metric_names,
        )

        train_kwargs = dict(config.train_kwargs or {})
        if config.algo == "ppo":
            train_kwargs.setdefault("checkpoint_dir", str((run_dir / "checkpoints").resolve()))
        if config.algo == "shac":
            train_kwargs.setdefault("checkpoint_dir", str((run_dir / "checkpoints").resolve()))
        result = train_fn(
            env,
            algo=config.algo,
            network=network,
            num_epochs=config.num_epochs,
            num_steps_per_epoch=config.num_steps_per_epoch,
            num_envs=config.num_envs,
            learning_rate=config.learning_rate,
            seed=train_seed,
            **train_kwargs,
        )

        checkpoint_path = _resolve_checkpoint_path(
            result,
            run_dir=run_dir,
            config=config,
            train_seed=train_seed,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        actual_backend = _training_backend(result.metrics)
        final_metrics = _evaluate_final_checkpoint(
            env,
            config,
            checkpoint_path,
            template_state,
            metric_names=metric_names,
            eval_fn=eval_fn,
            load_checkpoint_fn=load_checkpoint_fn,
            load_ppo_policy_fn=load_ppo_policy_fn,
            load_shac_policy_fn=load_shac_policy,
        )

        run_pass, failures = _evaluate_run(
            actual_backend=actual_backend,
            expected_backend=expected_backend,
            thresholds=thresholds,
            initial_metrics=initial_metrics,
            random_metrics=random_metrics,
            final_metrics=final_metrics,
        )

        runs.append(
            {
                "train_seed": train_seed,
                "eval_seeds": list(config.eval_seeds),
                "backend": actual_backend,
                "expected_backend": expected_backend,
                "checkpoint_path": str(checkpoint_path),
                "training_metrics": _jsonable(result.metrics),
                "initial_policy_metrics": initial_metrics,
                "random_policy_baseline_metrics": random_metrics,
                "final_checkpoint_metrics": final_metrics,
                "thresholds": thresholds.to_dict(),
                "pass": run_pass,
                "failures": failures,
            }
        )

    artifact = {
        "schema_version": 1,
        "config_path": None,
        "env": config.env,
        "algo": config.algo,
        "expected_backend": expected_backend,
        "backend": runs[0]["backend"] if runs else expected_backend,
        "train_seeds": list(config.train_seeds),
        "eval_seeds": list(config.eval_seeds),
        "thresholds": thresholds.to_dict(),
        "runs": runs,
        "pass": all(run["pass"] for run in runs),
        "failures": [failure for run in runs for failure in run["failures"]],
    }
    return artifact


def build_dry_run_artifact(
    config: AcceptanceConfig,
    thresholds: AcceptanceThresholds | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Builds a plan artifact without executing training."""

    thresholds = thresholds or config.thresholds
    output_dir = Path(output_dir or config.output_dir)
    return {
        "schema_version": 1,
        "dry_run": True,
        "config_path": None,
        "env": config.env,
        "algo": config.algo,
        "expected_backend": config.expected_backend,
        "train_seeds": list(config.train_seeds),
        "eval_seeds": list(config.eval_seeds),
        "output_dir": str(output_dir),
        "thresholds": thresholds.to_dict(),
        "planned_runs": [
            {
                "train_seed": train_seed,
                "eval_seeds": list(config.eval_seeds),
                "expected_backend": config.expected_backend,
            }
            for train_seed in config.train_seeds
        ],
    }


def validate_acceptance_artifact(
    artifact: Mapping[str, Any],
    config: AcceptanceConfig,
    thresholds: AcceptanceThresholds | None = None,
) -> None:
    """Validates an acceptance artifact against the gate contract."""

    thresholds = thresholds or config.thresholds
    _require_mapping(artifact, "artifact")
    _require_keys(
        artifact,
        [
            "schema_version",
            "env",
            "algo",
            "expected_backend",
            "backend",
            "train_seeds",
            "eval_seeds",
            "thresholds",
            "runs",
            "pass",
        ],
    )
    if int(artifact["schema_version"]) != 1:
        raise ValueError(f"Unsupported schema_version {artifact['schema_version']!r}")
    if artifact["env"] != config.env:
        raise ValueError(f"Artifact env {artifact['env']!r} does not match config {config.env!r}")
    if artifact["algo"] != config.algo:
        raise ValueError(f"Artifact algo {artifact['algo']!r} does not match config {config.algo!r}")
    if artifact["expected_backend"] != config.expected_backend:
        raise ValueError(
            "Artifact expected_backend "
            f"{artifact['expected_backend']!r} does not match config {config.expected_backend!r}"
        )

    _validate_thresholds(artifact["thresholds"], thresholds)
    runs = artifact["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("Artifact runs must be a non-empty list.")
    for index, run in enumerate(runs):
        _validate_run_artifact(run, thresholds, config, index=index)


def _validate_run_artifact(
    run: Mapping[str, Any],
    thresholds: AcceptanceThresholds,
    config: AcceptanceConfig,
    *,
    index: int,
) -> None:
    _require_mapping(run, f"runs[{index}]")
    _require_keys(
        run,
        [
            "train_seed",
            "eval_seeds",
            "backend",
            "expected_backend",
            "checkpoint_path",
            "initial_policy_metrics",
            "random_policy_baseline_metrics",
            "final_checkpoint_metrics",
            "thresholds",
            "pass",
            "failures",
        ],
    )
    if run["expected_backend"] != config.expected_backend:
        raise ValueError(
            f"Run {index} expected_backend {run['expected_backend']!r} "
            f"does not match config {config.expected_backend!r}"
        )
    _validate_thresholds(run["thresholds"], thresholds)
    for field_name in (
        "initial_policy_metrics",
        "random_policy_baseline_metrics",
        "final_checkpoint_metrics",
    ):
        _validate_metrics_block(run[field_name], field_name, index)
    if not isinstance(run["checkpoint_path"], str) or not run["checkpoint_path"]:
        raise ValueError(f"runs[{index}].checkpoint_path must be a non-empty string.")
    if not isinstance(run["pass"], bool):
        raise ValueError(f"runs[{index}].pass must be a boolean.")
    if not isinstance(run["failures"], list):
        raise ValueError(f"runs[{index}].failures must be a list.")


def _validate_metrics_block(block: Mapping[str, Any], name: str, index: int) -> None:
    _require_mapping(block, f"runs[{index}].{name}")
    _require_keys(
        block,
        [
            "mean_reward",
            "mean_episode_length",
            "primary_metric_name",
            "primary_metric_value",
            "episodes",
        ],
    )


def _validate_thresholds(
    actual: Mapping[str, Any],
    expected: AcceptanceThresholds,
) -> None:
    _require_mapping(actual, "thresholds")
    _require_keys(
        actual,
        [
            "min_reward_delta",
            "primary_metric_name",
            "primary_metric_direction",
            "primary_metric_threshold",
            "primary_metric_vs_random_delta",
            "additional_metrics",
        ],
    )
    if float(actual["min_reward_delta"]) < expected.min_reward_delta:
        raise ValueError("Artifact min_reward_delta is weaker than config.")
    if actual["primary_metric_name"] != expected.primary_metric_name:
        raise ValueError("Artifact primary_metric_name does not match config.")
    if actual["primary_metric_direction"] != expected.primary_metric_direction:
        raise ValueError("Artifact primary_metric_direction does not match config.")
    if _threshold_is_weaker(
        float(actual["primary_metric_threshold"]),
        expected.primary_metric_threshold,
        expected.primary_metric_direction,
    ):
        raise ValueError("Artifact primary_metric_threshold is weaker than config.")
    if float(actual["primary_metric_vs_random_delta"]) < expected.primary_metric_vs_random_delta:
        raise ValueError("Artifact primary_metric_vs_random_delta is weaker than config.")
    actual_additional = actual.get("additional_metrics", [])
    if len(actual_additional) != len(expected.additional_metrics):
        raise ValueError("Artifact additional_metrics do not match config.")
    for actual_gate, expected_gate in zip(actual_additional, expected.additional_metrics, strict=True):
        _validate_additional_metric_gate(actual_gate, expected_gate)


def _validate_additional_metric_gate(
    actual: Mapping[str, Any],
    expected: AcceptanceMetricGate,
) -> None:
    _require_mapping(actual, "additional_metrics[]")
    _require_keys(actual, ["name", "direction", "threshold", "vs_random_delta"])
    if actual["name"] != expected.name:
        raise ValueError("Artifact additional metric name does not match config.")
    if actual["direction"] != expected.direction:
        raise ValueError("Artifact additional metric direction does not match config.")
    if _threshold_is_weaker(
        float(actual["threshold"]),
        expected.threshold,
        expected.direction,
    ):
        raise ValueError("Artifact additional metric threshold is weaker than config.")
    if float(actual["vs_random_delta"]) < expected.vs_random_delta:
        raise ValueError("Artifact additional metric vs_random_delta is weaker than config.")


def _evaluate_run(
    *,
    actual_backend: str,
    expected_backend: str,
    thresholds: AcceptanceThresholds,
    initial_metrics: Mapping[str, Any],
    random_metrics: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if actual_backend != expected_backend:
        failures.append(f"backend mismatch: actual={actual_backend!r} expected={expected_backend!r}")

    initial_reward = float(initial_metrics["mean_reward"])
    final_reward = float(final_metrics["mean_reward"])
    if final_reward - initial_reward < thresholds.min_reward_delta:
        failures.append(
            "reward delta below threshold: "
            f"{final_reward - initial_reward:.6f} < {thresholds.min_reward_delta:.6f}"
        )

    final_primary = float(final_metrics["primary_metric_value"])
    random_primary = float(random_metrics["primary_metric_value"])
    failures.extend(
        _evaluate_metric_gate(
            name=thresholds.primary_metric_name,
            direction=thresholds.primary_metric_direction,
            threshold=thresholds.primary_metric_threshold,
            vs_random_delta=thresholds.primary_metric_vs_random_delta,
            final_value=final_primary,
            random_value=random_primary,
        )
    )

    for gate in thresholds.additional_metrics:
        final_value = _metric_summary_value(final_metrics, gate.name)
        random_value = _metric_summary_value(random_metrics, gate.name)
        failures.extend(
            _evaluate_metric_gate(
                name=gate.name,
                direction=gate.direction,
                threshold=gate.threshold,
                vs_random_delta=gate.vs_random_delta,
                final_value=final_value,
                random_value=random_value,
            )
        )

    return (not failures), failures


def _evaluate_metric_gate(
    *,
    name: str,
    direction: str,
    threshold: float,
    vs_random_delta: float,
    final_value: float,
    random_value: float,
) -> list[str]:
    import math

    if math.isnan(final_value):
        return [f"{name} is NaN"]

    failures: list[str] = []
    if direction == "max":
        if final_value < threshold:
            failures.append(f"{name} below threshold: {final_value:.6f} < {threshold:.6f}")
        if final_value < random_value + vs_random_delta:
            failures.append(
                f"{name} did not beat random baseline: "
                f"{final_value:.6f} < {random_value + vs_random_delta:.6f}"
            )
        return failures
    if direction == "min":
        if final_value > threshold:
            failures.append(f"{name} above threshold: {final_value:.6f} > {threshold:.6f}")
        if final_value > random_value - vs_random_delta:
            failures.append(
                f"{name} did not beat random baseline: "
                f"{final_value:.6f} > {random_value - vs_random_delta:.6f}"
            )
        return failures
    return [f"unknown metric direction {direction!r} for {name!r}"]


def _resolve_checkpoint_path(
    result: Any,
    *,
    run_dir: Path,
    config: AcceptanceConfig,
    train_seed: int,
    save_checkpoint_fn: Callable[..., Any],
) -> Path:
    checkpoint_path = getattr(result, "checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_path = run_dir / f"{config.env}_{config.algo}_seed{train_seed}.ckpt"
    checkpoint_path = Path(checkpoint_path)
    if getattr(result, "train_state", None) is not None and not checkpoint_path.exists():
        save_checkpoint_fn(checkpoint_path, result.train_state)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Acceptance checkpoint was not written: {checkpoint_path}")
    return checkpoint_path


def _evaluate_final_checkpoint(
    env,
    config: AcceptanceConfig,
    checkpoint_path: Path,
    template_state: Any,
    *,
    metric_names: list[str] | tuple[str, ...],
    eval_fn: Callable[..., dict[str, Any]],
    load_checkpoint_fn: Callable[..., Any],
    load_ppo_policy_fn: Callable[..., Any],
    load_shac_policy_fn: Callable[..., Any],
) -> dict[str, Any]:
    if config.algo == "ppo":
        policy = load_ppo_policy_fn(checkpoint_path)

        def policy_fn(obs, key):
            return policy(obs, key)[0]

    elif config.algo == "shac":
        policy = load_shac_policy_fn(checkpoint_path, env)

        def policy_fn(obs, key):
            return policy(obs, key)[0]

    else:
        checkpoint = load_checkpoint_fn(checkpoint_path, template_state)
        from dva_quadrotor_mjx.policies import select_action

        policy_fn = _deterministic_policy(checkpoint, select_action)

    return eval_fn(
        env,
        policy_fn,
        config.eval_seeds,
        episodes=config.eval_episodes,
        steps=_resolve_eval_steps(config, env),
        primary_metric_name=config.thresholds.primary_metric_name,
        metric_names=metric_names,
    )


def _resolve_eval_steps(config: AcceptanceConfig, env: Any) -> int:
    if config.eval_steps is not None:
        return int(config.eval_steps)
    return int(getattr(env, "max_steps_in_episode", config.num_steps_per_epoch))


def _deterministic_policy(train_state: Any, select_action_fn: Callable[[Any, Any], Any]):
    def policy(obs, _key):
        return select_action_fn(train_state, obs)

    return policy


def _random_policy(action_dim: int):
    def policy(_obs, key):
        import jax

        return jax.random.uniform(key, shape=(action_dim,), minval=-1.0, maxval=1.0)

    return policy


def _training_backend(metrics: Mapping[str, Any]) -> str:
    backend = metrics.get("backend") or metrics.get("trainer") or "unknown"
    return str(backend)


def _evaluate_policy_on_seeds(
    env,
    policy_fn: Callable[[Any, Any], Any],
    eval_seeds,
    *,
    episodes: int,
    steps: int,
    primary_metric_name: str,
    metric_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dva_quadrotor_mjx.envs.base import rollout

    # JIT-compile the rollout function. Redefinitions are cached by closure.
    @jax.jit
    def run_one_episode(key):
        return rollout(env, key, policy_fn, num_steps=steps)

    episode_records: list[dict[str, Any]] = []
    all_successes: list[bool] = []
    all_collisions: list[bool] = []
    for seed in eval_seeds:
        for episode_index in range(max(int(episodes), 1)):
            episode_seed = int(seed) + episode_index
            key = jax.random.PRNGKey(episode_seed)
            
            # Execute compiled rollout on GPU
            states = run_one_episode(key)
            
            # Convert JAX arrays to NumPy arrays ONCE to avoid multiple round-trips
            dones = np.asarray(states.done)
            done_indices = np.where(dones)[0]
            length = int(done_indices[0]) if done_indices.size > 0 else int(steps)
            
            # Extract terminal state info and metrics
            terminal_state = jax.tree_util.tree_map(lambda x: x[length], states)
            terminal_info = _jsonable(getattr(terminal_state, "info", {}))
            terminal_metrics = _jsonable(getattr(terminal_state, "metrics", {}))
            
            rewards = np.asarray(states.reward)
            total_reward = float(np.sum(rewards[1:length+1]))
            
            successes = states.info.get("success")
            if successes is not None:
                successes = np.asarray(successes)
                if successes.ndim > 0:
                    success = bool(np.any(successes[:length+1]))
                else:
                    success = bool(successes)
            else:
                success = False
                
            collisions = states.info.get("collision")
            if collisions is not None:
                collisions = np.asarray(collisions)
                if collisions.ndim > 0:
                    collision = bool(np.any(collisions[:length+1]))
                else:
                    collision = bool(collisions)
            else:
                collision = False
                
            metric_values = {
                name: _extract_episode_metric_value(
                    terminal_state,
                    metric_name=name,
                    fallback_reward=total_reward,
                    success=success,
                    collision=collision,
                    episode_length=length,
                )
                for name in dict.fromkeys(metric_names)
            }
            primary_metric_value = metric_values[primary_metric_name]
            record = {
                "seed": episode_seed,
                "reward": total_reward,
                "length": length,
                "success": success,
                "collision": collision,
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "metric_values": metric_values,
                "terminal_info": terminal_info,
                "terminal_metrics": terminal_metrics,
            }
            episode_records.append(record)
            all_successes.append(success)
            all_collisions.append(collision)

    primary_values = [record["primary_metric_value"] for record in episode_records]
    metric_summaries = {
        name: float(np.mean([record["metric_values"][name] for record in episode_records]))
        for name in dict.fromkeys(metric_names)
    }
    return {
        "mean_reward": float(np.mean([record["reward"] for record in episode_records])),
        "mean_episode_length": float(np.mean([record["length"] for record in episode_records])),
        "success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
        "collision_rate": float(np.mean(all_collisions)) if all_collisions else 0.0,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": float(np.mean(primary_values)) if primary_values else float("nan"),
        "metric_summaries": metric_summaries,
        "episodes": episode_records,
    }


def _extract_episode_metric_value(
    state: Any,
    *,
    metric_name: str,
    fallback_reward: float,
    success: bool,
    collision: bool,
    episode_length: int,
) -> float:
    if metric_name in {"mean_reward", "reward"}:
        return float(fallback_reward)
    if metric_name in {"mean_episode_length", "episode_length", "length"}:
        return float(episode_length)
    if metric_name in {"success", "success_rate"}:
        return float(success)
    if metric_name in {"collision", "collision_rate"}:
        return float(collision)
    return _extract_metric_value(
        state,
        primary_metric_name=metric_name,
        fallback_reward=fallback_reward,
    )


def _metric_summary_value(metrics_block: Mapping[str, Any], metric_name: str) -> float:
    if metric_name == metrics_block.get("primary_metric_name"):
        return float(metrics_block["primary_metric_value"])
    if metric_name in {"mean_reward", "reward"}:
        return float(metrics_block["mean_reward"])
    if metric_name in {"mean_episode_length", "episode_length", "length"}:
        return float(metrics_block["mean_episode_length"])
    if metric_name == "success_rate":
        return float(metrics_block["success_rate"])
    if metric_name == "collision_rate":
        return float(metrics_block["collision_rate"])
    if metric_name == "success":
        return float(metrics_block.get("metric_summaries", {}).get("success", metrics_block["success_rate"]))
    if metric_name == "collision":
        return float(metrics_block.get("metric_summaries", {}).get("collision", metrics_block["collision_rate"]))
    summaries = metrics_block.get("metric_summaries", {})
    if metric_name not in summaries:
        raise KeyError(f"Metric summary {metric_name!r} not found in acceptance metrics block.")
    return float(summaries[metric_name])


def _extract_metric_value(state: Any, *, primary_metric_name: str, fallback_reward: float) -> float:
    info = getattr(state, "info", {})
    metrics = getattr(state, "metrics", {})
    value: Any | None = None
    if isinstance(info, Mapping) and primary_metric_name in info:
        value = info[primary_metric_name]
    elif isinstance(metrics, Mapping) and primary_metric_name in metrics:
        value = metrics[primary_metric_name]
    elif primary_metric_name in {"mean_reward", "reward"}:
        value = fallback_reward

    if value is None:
        raise KeyError(
            f"Primary metric {primary_metric_name!r} not found in terminal info or metrics."
        )

    import jax
    import numpy as np

    if isinstance(value, jax.Array):
        value = np.asarray(value)
    if hasattr(value, "shape") and getattr(value, "shape", ()) == ():
        value = float(value)
    return float(value)


def _validate_metrics_value(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}")


def _require_mapping(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}")


def _require_keys(mapping: Mapping[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing required keys: {', '.join(missing)}")


def _jsonable(value: Any) -> Any:
    import jax
    import numpy as np

    if isinstance(value, jax.Array):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    sys.exit(main())
