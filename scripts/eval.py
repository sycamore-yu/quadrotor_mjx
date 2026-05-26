"""Evaluate a quadrotor MJX checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp

from dva_quadrotor_mjx.algorithms.trainer import eval as eval_policy
from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint
from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.learning.ppo_checkpoint import load_policy as load_ppo_policy
from dva_quadrotor_mjx.learning.shac_checkpoint import load_policy as load_shac_policy
from dva_quadrotor_mjx.policies import SUPPORTED_ALGOS, create_train_state, make_network
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument("--env", required=True)
    parser.add_argument("--algo", choices=SUPPORTED_ALGOS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    env = NormalizeActionWrapper(load(args.env))
    if args.algo in ("ppo", "shac"):
        policy = (
            load_ppo_policy(args.checkpoint)
            if args.algo == "ppo"
            else load_shac_policy(args.checkpoint, env)
        )
        result = _eval_brax_policy(policy, env, args.episodes, args.seed, args.steps)
        print(f"Mean reward: {result['mean_reward']:.6f}")
        print(f"Mean episode length: {result['mean_episode_length']:.2f}")
        print(f"Success rate: {result['success_rate']:.3f}")
        print(f"Collision rate: {result['collision_rate']:.3f}")
        return 0

    network = make_network(args.algo, env.action_space.shape[0])
    template = create_train_state(
        network,
        env.observation_size,
        seed=args.seed,
        learning_rate=3e-4,
    )
    checkpoint = load_checkpoint(Path(args.checkpoint), template)
    result = eval_policy(
        checkpoint,
        env,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.steps,
    )

    print(f"Mean reward: {result.mean_reward:.6f}")
    print(f"Mean episode length: {result.mean_episode_length:.2f}")
    print(f"Success rate: {result.success_rate:.3f}")
    print(f"Collision rate: {result.collision_rate:.3f}")
    return 0


def _eval_brax_policy(policy, env, episodes: int, seed: int, steps: int) -> dict[str, float]:
    rewards = []
    lengths = []
    successes = []
    collisions = []
    key = jax.random.PRNGKey(seed)
    for _ in range(episodes):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        total_reward = 0.0
        episode_length = 0
        success = False
        collision = False
        for _ in range(steps):
            key, action_key = jax.random.split(key)
            action = policy(state.obs, action_key)[0]
            state = env.step(state, action)
            total_reward += float(jnp.asarray(state.reward))
            episode_length += 1
            success = success or bool(jnp.asarray(state.info.get("success", False)))
            collision = collision or bool(jnp.asarray(state.info.get("collision", False)))
            if bool(jnp.asarray(state.done)):
                break
        rewards.append(total_reward)
        lengths.append(episode_length)
        successes.append(success)
        collisions.append(collision)
    return {
        "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
        "mean_episode_length": float(sum(lengths) / max(len(lengths), 1)),
        "success_rate": float(sum(successes) / max(len(successes), 1)),
        "collision_rate": float(sum(collisions) / max(len(collisions), 1)),
    }


if __name__ == "__main__":
    sys.exit(main())
