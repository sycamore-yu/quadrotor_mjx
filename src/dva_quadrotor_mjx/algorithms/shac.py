"""SHAC (Short-Horizon Actor Critic) wrapper for jax_shac.

Clean wrapper without runtime monkey patches.
"""

from typing import Any, Callable, Dict, Optional
import jax
import jax.numpy as jnp
import optax


def train(
    env,
    policy_network,
    value_network,
    num_timesteps: int = 100000,
    episode_length: int = 1000,
    num_envs: int = 128,
    actor_learning_rate: float = 1e-3,
    critic_learning_rate: float = 1e-3,
    unroll_length: int = 10,
    discounting: float = 0.99,
    seed: int = 0,
    progress_fn: Callable[[int, Dict[str, Any]], None] = lambda *args: None,
    **kwargs
):
    """Train using SHAC algorithm.

    This is a simplified wrapper that provides a clean interface.
    For full SHAC functionality, use jax_shac directly.

    Args:
        env: Brax-compatible environment
        policy_network: Policy network (flax.linen.Module)
        value_network: Value network (flax.linen.Module)
        num_timesteps: Total training timesteps
        episode_length: Max episode length
        num_envs: Number of parallel environments
        actor_learning_rate: Policy learning rate
        critic_learning_rate: Value learning rate
        unroll_length: Unroll length for gradient estimation
        discounting: Reward discount factor
        seed: Random seed
        progress_fn: Progress callback

    Returns:
        dict with trained parameters and metrics
    """
    try:
        from jax_shac.shac.train import SHAC
    except ImportError:
        raise ImportError(
            "jax_shac not installed. Install with: "
            "pip install -e third_party/jax_shac"
        )

    # Create SHAC trainer
    trainer = SHAC(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_envs=num_envs,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        unroll_length=unroll_length,
        discounting=discounting,
        seed=seed,
        progress_fn=progress_fn,
        **kwargs
    )

    # Train
    params, metrics = trainer.train()

    return {
        "params": params,
        "metrics": metrics,
        "make_policy": trainer.make_policy,
    }
