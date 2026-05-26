"""Offline MuJoCo rendering for random or checkpoint policies."""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import gc
import subprocess
import sys
from pathlib import Path

from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime

configure_jax_runtime()

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint
from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
from dva_quadrotor_mjx.learning.render_worker import render_rollout
from dva_quadrotor_mjx.policies import SUPPORTED_ALGOS, create_train_state, make_network, select_action
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a quadrotor rollout")
    parser.add_argument("--env", required=True)
    parser.add_argument("--algo", choices=SUPPORTED_ALGOS, default="bptt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=str, default="artifacts/render.mp4")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """In-process render path kept for tests and direct module calls."""

    args = _parse_args(argv)
    env = NormalizeActionWrapper(load(args.env))
    base_env = getattr(env, "env", env)
    loaded_policy = None
    if args.checkpoint and args.algo == "ppo":
        loaded_policy = load_ppo_policy(args.checkpoint)
    if args.checkpoint and args.algo == "shac":
        loaded_policy = load_shac_policy(args.checkpoint, env)
    train_state = None
    if loaded_policy is None:
        network = make_network(args.algo, env.action_space.shape[0])
        train_state = create_train_state(
            network,
            env.observation_size,
            seed=args.seed,
            learning_rate=3e-4,
        )
        if args.checkpoint:
            train_state = load_checkpoint(args.checkpoint, train_state)

    key = jax.random.PRNGKey(args.seed)
    state = env.reset(key)
    mj_model = base_env.model
    mj_data = mujoco.MjData(mj_model)
    frames = []
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    try:
        for _ in range(args.steps):
            key, action_key = jax.random.split(key)
            if args.random_policy:
                action = jax.random.uniform(
                    action_key,
                    (env.action_space.shape[0],),
                    minval=-1.0,
                    maxval=1.0,
                )
            elif loaded_policy is not None:
                action = loaded_policy(state.obs, action_key)[0]
            else:
                action = select_action(train_state, state.obs)
            state = env.step(state, action)
            mj_data.qpos[:] = np.asarray(state.pipeline_state.qpos)
            mj_data.qvel[:] = np.asarray(state.pipeline_state.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data)
            frames.append(renderer.render())
            if bool(jnp.asarray(state.done)):
                state = env.reset(action_key)
    finally:
        close = getattr(renderer, "close", None)
        if callable(close):
            close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(output, frames, fps=30)
    except Exception as exc:
        raise RuntimeError(
            "MP4 video writer unavailable; install an imageio ffmpeg backend "
            "and rerun render.py. Frame-dump fallback is intentionally disabled "
            "by the OpenSpec render acceptance contract."
        ) from exc
    del frames, mj_data, state, train_state, loaded_policy, env, base_env, mj_model
    gc.collect()
    print(f"Rendered video: {output}")
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """Headless-stable CLI path that isolates rendering in a worker process."""

    args = _parse_args(argv)
    worker_argv = [
        sys.executable,
        "-m",
        "dva_quadrotor_mjx.learning.render_cli_worker",
        "--env",
        args.env,
        "--algo",
        args.algo,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--output",
        args.output,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.checkpoint:
        worker_argv.extend(["--checkpoint", args.checkpoint])
    if args.random_policy:
        worker_argv.append("--random-policy")
    completed = subprocess.run(worker_argv, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(cli_main())
