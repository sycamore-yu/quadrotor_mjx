"""Unified training interface for all algorithms."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import flax.serialization


@dataclass
class TrainResult:
    """Result of a training run."""
    params: Any
    metrics: Dict[str, Any]
    train_state: Optional[TrainState] = None


@dataclass
class EvalResult:
    """Result of an evaluation run."""
    mean_reward: float
    mean_episode_length: float
    success_rate: Optional[float] = None


def train(
    env,
    algo: str,
    network,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 100,
    num_envs: int = 128,
    learning_rate: float = 3e-4,
    seed: int = 0,
    **kwargs
) -> TrainResult:
    """Train a policy using the specified algorithm.

    Args:
        env: Brax-compatible environment
        algo: Algorithm name ('bptt', 'ppo', 'shac')
        network: Neural network module
        num_epochs: Number of training epochs
        num_steps_per_epoch: Steps per epoch
        num_envs: Number of parallel environments
        learning_rate: Learning rate
        seed: Random seed
        **kwargs: Additional algorithm-specific arguments

    Returns:
        TrainResult with trained parameters and metrics
    """
    key = jax.random.PRNGKey(seed)

    # Initialize network parameters
    obs_shape = env.observation_size
    dummy_obs = jnp.zeros(obs_shape)
    params = network.init(key, dummy_obs)

    # Create optimizer
    tx = jax.example_libraries.optimizers.adam(learning_rate)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    if algo == "bptt":
        from dva_quadrotor_mjx.algorithms.bptt import train as bptt_train
        result = bptt_train(
            env, train_state, num_epochs, num_steps_per_epoch, num_envs, key
        )
        return TrainResult(
            params=result["runner_state"].train_state.params,
            metrics={"losses": result["metrics"]},
            train_state=result["runner_state"].train_state,
        )
    elif algo == "ppo":
        from dva_quadrotor_mjx.algorithms.ppo import train as ppo_train, Config
        config = Config(**kwargs.get("ppo_config", {}))
        result = ppo_train(
            env, train_state, num_epochs, num_steps_per_epoch, num_envs, key, config
        )
        return TrainResult(
            params=result["runner_state"].train_state.params,
            metrics={"returns": result["metrics"]},
            train_state=result["runner_state"].train_state,
        )
    elif algo == "shac":
        from dva_quadrotor_mjx.algorithms.shac import train as shac_train
        result = shac_train(
            env,
            policy_network=network,
            value_network=kwargs.get("value_network"),
            num_timesteps=num_epochs * num_steps_per_epoch * num_envs,
            episode_length=num_steps_per_epoch,
            num_envs=num_envs,
            seed=seed,
            **kwargs
        )
        return TrainResult(
            params=result["params"],
            metrics=result["metrics"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def eval(
    checkpoint: Any,
    env,
    episodes: int = 10,
    seed: int = 0,
) -> EvalResult:
    """Evaluate a trained policy.

    Args:
        checkpoint: Trained parameters or TrainState
        env: Brax-compatible environment
        episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        EvalResult with evaluation metrics
    """
    key = jax.random.PRNGKey(seed)

    # Extract params from checkpoint
    if isinstance(checkpoint, TrainState):
        params = checkpoint.params
        apply_fn = checkpoint.apply_fn
    else:
        params = checkpoint
        # Need to reconstruct apply_fn
        raise ValueError("Checkpoint must be a TrainState for eval")

    def policy(obs, key):
        return apply_fn(params, obs)

    # Run evaluation episodes
    from dva_quadrotor_mjx.envs.base import rollout
    states = rollout(env, key, policy, num_steps=env.max_steps_in_episode)

    mean_reward = jnp.mean(states.reward)
    mean_length = jnp.mean(jnp.sum(1 - states.done))

    return EvalResult(
        mean_reward=float(mean_reward),
        mean_episode_length=float(mean_length),
    )


def make_policy(checkpoint: Any) -> Callable:
    """Create a policy function from a checkpoint.

    Args:
        checkpoint: Trained parameters or TrainState

    Returns:
        Policy function: obs -> action
    """
    if isinstance(checkpoint, TrainState):
        params = checkpoint.params
        apply_fn = checkpoint.apply_fn
    else:
        params = checkpoint
        raise ValueError("Checkpoint must be a TrainState")

    def policy(obs):
        return apply_fn(params, obs)

    return policy


def save_checkpoint(path: str, train_state: TrainState):
    """Save training checkpoint.

    Args:
        path: File path to save checkpoint
        train_state: Flax TrainState
    """
    bytes_data = flax.serialization.to_bytes(train_state)
    with open(path, "wb") as f:
        f.write(bytes_data)


def load_checkpoint(path: str, train_state: TrainState) -> TrainState:
    """Load training checkpoint.

    Args:
        path: File path to load checkpoint
        train_state: Template TrainState

    Returns:
        Loaded TrainState
    """
    with open(path, "rb") as f:
        bytes_data = f.read()
    return flax.serialization.from_bytes(train_state, bytes_data)
