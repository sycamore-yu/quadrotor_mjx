"""MuJoCo Playground-style training CLI for quadrotor MJX tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    _xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags.strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

ALGORITHMS = ("bptt", "ppo", "shac")
FULL_DEFAULT_EPOCHS = 100
FULL_DEFAULT_STEPS = 1000
FULL_DEFAULT_NUM_ENVS = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a quadrotor policy")
    parser.add_argument("--env", "--env_name", dest="env", required=True)
    parser.add_argument("--algo", choices=ALGORITHMS, required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the long baseline defaults instead of the short safe default.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int)
    parser.add_argument("--steps", "--episode_length", dest="steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_timesteps", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser.parse_args(argv)


def _resolve_training_budget(args: argparse.Namespace) -> str:
    """Resolves Playground-style flags to this trainer's epoch/step budget."""

    if args.smoke:
        args.epochs = min(args.epochs or 1, 1)
        args.steps = min(args.steps or 1, 1)
        args.num_envs = min(args.num_envs or 4, 4)
        return "smoke"

    if args.num_timesteps is not None and args.steps is None:
        epochs = args.epochs or 1
        args.steps = max(1, args.num_timesteps // max(epochs, 1))

    if args.full:
        args.epochs = args.epochs or FULL_DEFAULT_EPOCHS
        args.steps = args.steps or FULL_DEFAULT_STEPS
        args.num_envs = args.num_envs or FULL_DEFAULT_NUM_ENVS
        return "full"

    args.epochs = args.epochs or 1
    args.steps = args.steps or 1
    args.num_envs = args.num_envs or 1
    return "short-default"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = _resolve_training_budget(args)

    from dva_quadrotor_mjx.algorithms.trainer import save_checkpoint, train
    from dva_quadrotor_mjx.envs.registry import load
    from dva_quadrotor_mjx.policies import make_network
    from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper

    config = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text()) or {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.env}_{args.algo}_seed{args.seed}"
    checkpoint_path = output_dir / f"{stem}.ckpt"
    metrics_path = output_dir / f"{stem}_metrics.json"
    if args.algo == "ppo":
        config.setdefault("checkpoint_dir", str((output_dir / f"{stem}_checkpoints").resolve()))
        if args.smoke:
            config.setdefault("num_timesteps", 0)
            config.setdefault("run_evals", False)
        if args.num_timesteps is not None:
            config.setdefault("num_timesteps", args.num_timesteps)
    if args.algo == "shac":
        config.setdefault("checkpoint_dir", str((output_dir / f"{stem}_checkpoints").resolve()))
        if args.smoke:
            config.setdefault("num_timesteps", 0)
            config.setdefault("unroll_length", 1)
            config.setdefault("critic_epochs", 1)
            config.setdefault("critic_batch_size", max(args.num_envs, 1))
            config.setdefault("num_evals", 1)
            config.setdefault("num_eval_envs", min(args.num_envs, 4))
        if args.num_timesteps is not None:
            config.setdefault("num_timesteps", args.num_timesteps)

    env = NormalizeActionWrapper(load(args.env))
    network = make_network(args.algo, env.action_space.shape[0])

    print(f"Training {args.algo} on {args.env}")
    print(
        "  "
        f"mode={mode} epochs={args.epochs} steps={args.steps} "
        f"num_envs={args.num_envs} seed={args.seed}"
    )
    if mode == "short-default":
        print("  pass --full or explicit --epochs/--steps for a longer baseline run")

    result = train(
        env,
        algo=args.algo,
        network=network,
        num_epochs=args.epochs,
        num_steps_per_epoch=args.steps,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        seed=args.seed,
        **config,
    )

    if result.train_state is not None:
        save_checkpoint(checkpoint_path, result.train_state)
    metrics_path.write_text(json.dumps(_jsonable(result.metrics), indent=2, sort_keys=True))

    print(f"Checkpoint: {result.checkpoint_path or checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    mean_reward = result.metrics.get("mean_reward")
    if isinstance(mean_reward, (int, float)) and np.isfinite(mean_reward):
        print(f"Mean reward: {mean_reward:.6f}")
    else:
        print("Mean reward: n/a (evaluation disabled for this run)")
    return 0


def _jsonable(value: Any) -> Any:
    import jax

    if isinstance(value, jax.Array):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    sys.exit(main())
