"""Standalone render worker used by the render CLI."""

from __future__ import annotations

import gc
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint
from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
from dva_quadrotor_mjx.policies import create_train_state, make_network, select_action
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


def render_rollout(
    *,
    env_name: str,
    algo: str,
    checkpoint: str | None,
    random_policy: bool,
    seed: int,
    steps: int,
    output: str,
    width: int,
    height: int,
) -> Path:
    """Renders a rollout to MP4 and returns the output path."""

    env = NormalizeActionWrapper(load(env_name))
    base_env = getattr(env, "env", env)

    loaded_policy = None
    network = None
    train_state = None
    if checkpoint and algo == "ppo":
        loaded_policy = load_ppo_policy(checkpoint)
    if checkpoint and algo == "shac":
        loaded_policy = load_shac_policy(checkpoint, env)
    if loaded_policy is None:
        network = make_network(algo, env.action_space.shape[0])
        train_state = create_train_state(
            network,
            env.observation_size,
            seed=seed,
            learning_rate=3e-4,
        )
        if checkpoint:
            train_state = load_checkpoint(checkpoint, train_state)

    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    mj_model = base_env.model
    mj_data = mujoco.MjData(mj_model)
    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        for _ in range(steps):
            key, action_key = jax.random.split(key)
            if random_policy:
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

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Render output was not written or is empty: {output_path}")

    del frames, mj_data, state, train_state, loaded_policy, network, env, base_env, mj_model
    gc.collect()
    return output_path
