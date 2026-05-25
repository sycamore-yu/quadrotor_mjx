import jax
import jax.numpy as jnp
import pytest

from dva_quadrotor_mjx.envs import load

def test_m2_env_reset_and_properties() -> None:
    """M2.2 & M2.4: Test reset state observation shapes, episode properties and registries."""
    env = load("quadrotor_hover")
    
    # Check dimensions
    assert env.action_size == 4
    
    # State observation length: 3 (pos) + 9 (rot matrix) + 3 (vel) + 8 (action queue) = 23
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    assert state.obs.shape == (23,)
    assert state.reward.shape == ()
    assert state.done.shape == ()
    
    # Confirm randomize behavior: reset twice with different keys
    state2 = env.reset(jax.random.PRNGKey(43))
    # Position should be different
    assert not jnp.allclose(state.obs[0:3], state2.obs[0:3])

def test_m2_env_step_evolution() -> None:
    """M2.1 & M2.3: Test stepping transitions and the P-control dual-frequency mechanism."""
    env = load("quadrotor_hover")
    rng = jax.random.PRNGKey(101)
    
    state = env.reset(rng)
    
    # Send hovering command: mass * 9.81 = 7.3575, no angular rate command
    action = jnp.array([7.3575, 0.0, 0.0, 0.0])
    
    # Run step
    next_state = env.step(state, action)
    
    assert next_state.obs.shape == (23,)
    assert next_state.reward.shape == ()
    assert next_state.done.shape == ()
    assert int(next_state.info["steps"]) == 1
    
    # Check that position has updated (gravity vs hover force)
    # Since hover action exactly counters gravity, the quadrotor vertical position should remain extremely close to original position
    z_diff = next_state.obs[2] - state.obs[2]
    # The actual vertical motion should be tiny, e.g. < 0.01
    assert abs(z_diff) < 0.01

def test_m2_collision_done_trigger() -> None:
    """M2.1: Test that done flag is correctly set to 1.0 when the quadrotor goes out of bounds."""
    env = load("quadrotor_hover")
    rng = jax.random.PRNGKey(202)
    state = env.reset(rng)
    
    # Manually move the quadrotor out of bounds (Z coord < 0.0)
    qpos = state.pipeline_state.qpos.at[2].set(-0.1)
    pipeline_state_oob = state.pipeline_state.replace(qpos=qpos)
    state_oob = state.replace(pipeline_state=pipeline_state_oob)
    
    # Re-evaluate observation based on oob state
    obs = env._get_obs(pipeline_state_oob, state_oob.info)
    state_oob = state_oob.replace(obs=obs)
    
    # Take a step with any action, it should detect collision and set done = 1.0
    next_state = env.step(state_oob, jnp.zeros(4))
    assert float(next_state.done) == 1.0
    # Reward should incorporate collision cost
    assert float(next_state.reward) < -1.0
