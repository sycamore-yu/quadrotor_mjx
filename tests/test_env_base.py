import jax
import jax.numpy as jnp
import pytest
from dva_quadrotor_mjx.envs import load
from dva_quadrotor_mjx.envs.base import AutoResetWrapper, rollout, rollout_recurrent

def test_rollout_and_autoreset() -> None:
    env = load("quadrotor_hover")
    env = AutoResetWrapper(env)
    
    # Check reset
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    assert "rng" in state.info
    
    # Check step
    action = jnp.array([7.3575, 0.0, 0.0, 0.0])
    next_state = env.step(state, action)
    assert next_state.obs.shape == (23,)
    assert int(next_state.info["steps"]) == 1
    
    # Define a simple policy
    def policy(obs, key):
        return jnp.array([7.3575, 0.0, 0.0, 0.0])
        
    # Test rollout
    states = rollout(env, rng, policy, num_steps=10)
    # The returned states should have length 11 (init + 10 steps)
    assert states.obs.shape == (11, 23)
    assert jnp.array_equal(states.info["steps"], jnp.arange(11))
    
    # Test rollout recurrent
    def policy_recurrent(carry, obs, key):
        return carry + 1, jnp.array([7.3575, 0.0, 0.0, 0.0])
        
    states, carries = rollout_recurrent(env, rng, policy_recurrent, carry_init=jnp.array(0), num_steps=10)
    assert states.obs.shape == (11, 23)
    assert carries.shape == (10,)

def test_wrappers() -> None:
    from dva_quadrotor_mjx.wrappers import LogWrapper, NormalizeActionWrapper, MinMaxObservationWrapper, VecEnv
    
    env = load("quadrotor_hover")
    
    # 1. NormalizeActionWrapper
    env_norm_action = NormalizeActionWrapper(env)
    assert jnp.allclose(env_norm_action.map_action(jnp.array([-1.0, -1.0, -1.0, -1.0])), env.action_space_low)
    assert jnp.allclose(env_norm_action.map_action(jnp.array([1.0, 1.0, 1.0, 1.0])), env.action_space_high)
    
    # 2. MinMaxObservationWrapper
    obs_min = jnp.zeros(23)
    obs_max = jnp.ones(23) * 10.0
    env_norm_obs = MinMaxObservationWrapper(env, obs_min, obs_max)
    
    # 3. LogWrapper
    env_logged = LogWrapper(env)
    rng = jax.random.PRNGKey(0)
    state = env_logged.reset(rng)
    assert "episode_returns" in state.info
    next_state = env_logged.step(state, jnp.array([7.3575, 0.0, 0.0, 0.0]))
    assert "returned_episode_returns" in next_state.info
    
    # 4. VecEnv
    env_vec = VecEnv(env)
    rngs = jax.random.split(rng, 4)
    state_vec = env_vec.reset(rngs)
    assert state_vec.obs.shape == (4, 23)
    
    actions = jnp.tile(jnp.array([7.3575, 0.0, 0.0, 0.0]), (4, 1))
    next_state_vec = env_vec.step(state_vec, actions)
    assert next_state_vec.obs.shape == (4, 23)
