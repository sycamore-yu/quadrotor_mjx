"""Live mjviser play script for quadrotor MJX scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import mujoco
import numpy as np

from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint
from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
from dva_quadrotor_mjx.policies import SUPPORTED_ALGOS, create_train_state, make_network, select_action
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play a quadrotor scene in mjviser")
    parser.add_argument("--env", required=True)
    parser.add_argument("--mode", choices=("scene", "random", "checkpoint"), default="scene")
    parser.add_argument("--algo", choices=SUPPORTED_ALGOS, default="bptt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--dry-run",
        "--check-scene",
        action="store_true",
        dest="dry_run",
        help="Construct the scene and policy path without starting a viser server; "
        "prints a machine-readable dry-run plan.",
    )
    args = parser.parse_args(argv)

    env = NormalizeActionWrapper(load(args.env))
    base_env = getattr(env, "env", env)
    model = base_env.model
    data = mujoco.MjData(model)
    key = jax.random.PRNGKey(args.seed)
    state_box = {"state": env.reset(key), "key": key, "steps": 0}

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
        if loaded_policy is None:
            train_state = load_checkpoint(Path(args.checkpoint), train_state)
        args.mode = "checkpoint"

    def copy_state_to_mujoco() -> None:
        state = state_box["state"]
        data.qpos[:] = np.asarray(state.pipeline_state.qpos)
        data.qvel[:] = np.asarray(state.pipeline_state.qvel)
        mujoco.mj_forward(model, data)

    def step_fn(_model: mujoco.MjModel, _data: mujoco.MjData) -> None:
        if args.mode == "scene":
            copy_state_to_mujoco()
            return

        state_box["key"], action_key = jax.random.split(state_box["key"])
        if args.mode == "random":
            action = jax.random.uniform(action_key, (env.action_space.shape[0],), minval=-1.0, maxval=1.0)
        elif loaded_policy is not None:
            action = loaded_policy(state_box["state"].obs, action_key)[0]
        else:
            action = select_action(train_state, state_box["state"].obs)
        state_box["state"] = env.step(state_box["state"], action)
        state_box["steps"] += 1
        if state_box["steps"] >= args.steps or bool(state_box["state"].done):
            state_box["steps"] = 0
            state_box["state"] = env.reset(action_key)
        copy_state_to_mujoco()

    copy_state_to_mujoco()
    if args.dry_run:
        plan = {
            "algo": args.algo,
            "checkpoint": args.checkpoint,
            "env": args.env,
            "mode": args.mode,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "port": args.port,
            "seed": args.seed,
            "steps": args.steps,
            "viewer_started": False,
        }
        print("Dry-run scene check passed (viewer not started)")
        print(json.dumps(plan, sort_keys=True))
        return 0

    import viser
    try:
        from mjviser import Viewer
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "mjviser" / "src"))
        from mjviser import Viewer

    server = viser.ViserServer(port=args.port)
    bound_port = server.get_port() if hasattr(server, "get_port") else args.port
    print(f"mjviser URL: http://localhost:{bound_port}")
    print("Use SSH port forwarding from the headless server if needed.")
    Viewer(model, data, step_fn=step_fn, server=server).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
