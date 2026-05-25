"""Evaluation script for trained policies."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.algorithms.trainer import eval, load_checkpoint
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


class DeterministicMLP(nn.Module):
    """Deterministic policy network."""
    hidden_dims: tuple = (128, 128)
    action_dim: int = 4

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class ActorCriticMLP(nn.Module):
    """Actor-Critic network."""
    hidden_dims: tuple = (128, 128)
    action_dim: int = 4

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        actor = nn.Dense(self.action_dim)(x)
        actor = nn.tanh(actor)
        critic = nn.Dense(1)(x)
        critic = jnp.squeeze(critic, axis=-1)
        return actor, critic


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--algo", type=str, required=True, choices=["bptt", "ppo", "shac"], help="Algorithm")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Create environment
    env = load(args.env)
    env = NormalizeActionWrapper(env)

    # Create network
    action_dim = env.action_space.shape[0]

    if args.algo == "bptt":
        network = DeterministicMLP(action_dim=action_dim)
    else:
        network = ActorCriticMLP(action_dim=action_dim)

    # Initialize template train state
    import optax
    dummy_obs = jnp.zeros(env.observation_size)
    params = network.init(jax.random.PRNGKey(0), dummy_obs)
    tx = optax.adam(1e-3)
    template_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, template_state)

    # Evaluate
    print(f"Evaluating on {args.env} for {args.episodes} episodes")
    result = eval(checkpoint, env, episodes=args.episodes, seed=args.seed)

    print(f"Results:")
    print(f"  Mean reward: {result.mean_reward:.2f}")
    print(f"  Mean episode length: {result.mean_episode_length:.2f}")
    if result.success_rate is not None:
        print(f"  Success rate: {result.success_rate:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
