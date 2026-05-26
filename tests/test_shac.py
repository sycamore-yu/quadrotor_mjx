# Patch deprecated jax.tree_map (removed in JAX v0.6.0+)
import jax
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

import jax.numpy as jnp
import pytest

from dva_quadrotor_mjx.envs import load
from dva_quadrotor_mjx.algorithms.shac.train import SHAC

def test_m3_shac_integration() -> None:
    """M3.1 & M3.2: Verify HoveringStateEnv can be loaded into SHAC and complete one training epoch."""
    env = load("quadrotor_hover", max_steps_in_episode=20)
    
    # Initialize SHAC with tiny parameters for fast test check
    shac = SHAC(
        environment=env,
        num_timesteps=40,
        episode_length=20,
        num_envs=4,
        num_eval_envs=2,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        seed=42,
        unroll_length=5,
        critic_batch_size=4, # (num_envs * unroll_length) = 4 * 5 = 20.
        critic_epochs=2,
        num_evals=2,
    )
    
    key = jax.random.PRNGKey(0)
    key_env, key_state = jax.random.split(key)
    
    # 1. Reset batched envs
    env_state = shac.reset_fn(jax.random.split(key_env, 4))
    assert env_state.obs.shape == (4, 23)
    
    # 2. Init training state
    training_state, key_state = shac.init_training_state(key_state)
    assert training_state.env_steps == 0
    
    # 3. Execute one epoch
    epoch_key, _ = jax.random.split(key_state)
    new_tr_state, new_env_state, loss_metrics, err = shac.training_epoch(
        training_state, env_state, epoch_key
    )
    
    err = err.get()
    if err:
        raise ValueError(f"SHAC epoch checkify error: {err}")
        
    assert int(new_tr_state.env_steps) > 0
