"""Offline MuJoCo rendering for random or checkpoint policies."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    _xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags.strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint
from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
from dva_quadrotor_mjx.policies import SUPPORTED_ALGOS, create_train_state, make_network, select_action
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

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
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)

    frames = []
    for step in range(args.steps):
        key, action_key = jax.random.split(key)
        if args.random_policy:
            action = jax.random.uniform(action_key, (env.action_space.shape[0],), minval=-1.0, maxval=1.0)
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    try:
        imageio.mimsave(output, frames, fps=30)
    except Exception as exc:
        raise RuntimeError(
            "MP4 video writer unavailable; install an imageio ffmpeg backend "
            "and rerun render.py. Frame-dump fallback is intentionally disabled "
            "by the OpenSpec render acceptance contract."
        ) from exc
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Render output was not written or is empty: {output}")
    print(f"Rendered video: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
