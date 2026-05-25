"""Unified training script for all environments and algorithms."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import yaml

from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.algorithms.trainer import train, save_checkpoint
from dva_quadrotor_mjx.wrappers.normalize import NormalizeActionWrapper


class DeterministicMLP(nn.Module):
    """Deterministic policy network for BPTT."""
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
    """Actor-Critic network for PPO."""
    hidden_dims: tuple = (128, 128)
    action_dim: int = 4

    @nn.compact
    def __call__(self, x):
        # Shared backbone
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Actor head
        actor = nn.Dense(self.action_dim)(x)
        # Use tanh for bounded actions
        actor = nn.tanh(actor)

        # Critic head
        critic = nn.Dense(1)(x)
        critic = jnp.squeeze(critic, axis=-1)

        return actor, critic


def main():
    parser = argparse.ArgumentParser(description="Train a policy")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--algo", type=str, required=True, choices=["bptt", "ppo", "shac"], help="Algorithm")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (minimal steps)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Smoke test overrides
    if args.smoke:
        args.epochs = 2
        args.steps = 10
        args.num_envs = 4
        print("Running smoke test with minimal configuration")

    # Create environment
    env = load(args.env)
    env = NormalizeActionWrapper(env)

    # Create network
    obs_dim = env.observation_size[0]
    action_dim = env.action_space.shape[0]

    if args.algo == "bptt":
        network = DeterministicMLP(action_dim=action_dim)
    elif args.algo in ["ppo", "shac"]:
        network = ActorCriticMLP(action_dim=action_dim)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Train
    print(f"Training {args.algo} on {args.env}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {args.steps}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Learning rate: {args.lr}")

    result = train(
        env,
        algo=args.algo,
        network=network,
        num_epochs=args.epochs,
        num_steps_per_epoch=args.steps,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        seed=args.seed,
        **config
    )

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.env}_{args.algo}_seed{args.seed}.ckpt"

    if result.train_state is not None:
        save_checkpoint(str(checkpoint_path), result.train_state)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
