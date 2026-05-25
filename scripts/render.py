"""Render script for trained policies using MuJoCo viewer."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import mujoco
import mujoco.viewer

from dva_quadrotor_mjx.envs.registry import load
from dva_quadrotor_mjx.algorithms.trainer import load_checkpoint
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
    parser = argparse.ArgumentParser(description="Render a trained policy")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--algo", type=str, required=True, choices=["bptt", "ppo", "shac"], help="Algorithm")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--duration", type=float, default=10.0, help="Render duration in seconds")
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

    # Create policy function
    def policy(obs):
        if args.algo == "bptt":
            return checkpoint.apply_fn(checkpoint.params, obs)
        else:
            actor, _ = checkpoint.apply_fn(checkpoint.params, obs)
            return actor

    # Reset environment
    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)

    # Get MuJoCo model and data for rendering
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)

    # Copy initial state
    mj_data.qpos[:] = jnp.array(state.pipeline_state.qpos)
    mj_data.qvel[:] = jnp.array(state.pipeline_state.qvel)

    print(f"Rendering {args.env} for {args.duration} seconds")
    print("Close the viewer window to stop")

    # Run simulation with viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        start_time = mj_data.time
        while viewer.is_running() and mj_data.time - start_time < args.duration:
            # Get action from policy
            obs = jnp.array(state.obs)
            action = policy(obs)
            action = jnp.array(action)

            # Step environment
            state = env.step(state, action)

            # Update MuJoCo data
            mj_data.qpos[:] = jnp.array(state.pipeline_state.qpos)
            mj_data.qvel[:] = jnp.array(state.pipeline_state.qvel)
            mujoco.mj_forward(mj_model, mj_data)

            # Sync viewer
            viewer.sync()

    print("Render complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
