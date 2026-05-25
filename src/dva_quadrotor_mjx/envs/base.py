from brax.base import Base
from brax.envs.base import Env
from brax.envs.base import Wrapper
from flax import struct
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import Any, Dict, Optional, Tuple

@struct.dataclass
class State(Base):
    """Environment state for DVA quadrotor training and inference."""
    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax/SHAC."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
        **kwargs
    ):
        """Initializes MjxEnv.

        Args:
            mj_model: mujoco.MjModel
            physics_steps_per_control_step: the number of times to step the physics
                pipeline for each environment step
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.put_model(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.make_data(self.sys)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Takes physics steps using the physics pipeline."""
        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return mjx.step(self.sys, data), None
        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def action_size(self) -> int:
        """Size of the action space."""
        return self.sys.nu

    @property
    def backend(self) -> str:
        return 'mjx'


class AutoResetWrapper(Wrapper):
    """Automatically resets envs that are done, using a split key from state.info['rng']."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['rng'] = rng
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng = state.info.get('rng', jax.random.PRNGKey(0))
        rng_step, rng_reset = jax.random.split(rng)
        
        # Step environment
        next_state = self.env.step(state, action)
        
        # Reset environment
        reset_state = self.env.reset(rng_reset)
        
        def where_done(x, y):
            done = next_state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)
            
        pipeline_state = jax.tree_util.tree_map(where_done, reset_state.pipeline_state, next_state.pipeline_state)
        obs = jax.tree_util.tree_map(where_done, reset_state.obs, next_state.obs)
        
        next_info = dict(next_state.info)
        next_info['rng'] = rng_step
        if 'steps' in next_info:
            steps = next_info['steps']
            if next_state.done.shape:
                steps = jnp.where(next_state.done, jnp.zeros_like(steps), steps)
            else:
                steps = jnp.where(next_state.done, 0, steps)
            next_info['steps'] = steps
            
        return next_state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            info=next_info
        )


def rollout(
    env,
    key: jax.Array,
    policy,
    state: Optional[State] = None,
    *,
    real_step: bool = True,
    num_steps: Optional[int] = None,
) -> State:
    """Rollouts the environment with a policy."""
    if num_steps is None:
        num_steps = env.max_steps_in_episode
    
    key_reset, key_rollout = jax.random.split(key)
    if state is None:
        state = env.reset(key_reset)
        
    def step_fn(state, key_step):
        key_policy, _ = jax.random.split(key_step)
        action = policy(state.obs, key_policy)
        if real_step:
            next_state = env.step(state, action)
        else:
            next_state = getattr(env, "unwrapped", env).step(state, action)
        return next_state, next_state

    keys_steps = jax.random.split(key_rollout, num_steps)
    _, states = jax.lax.scan(step_fn, state, keys_steps)
    
    states = jax.tree_util.tree_map(
        lambda l0, l1: jnp.concatenate([l0[None], l1]), state, states
    )
    return states


def rollout_recurrent(
    env,
    key: jax.Array,
    policy,
    carry_init,
    state: Optional[State] = None,
    *,
    num_steps: Optional[int] = None,
) -> Tuple[State, Any]:
    """Rollouts the environment with a recurrent policy."""
    if num_steps is None:
        num_steps = env.max_steps_in_episode
        
    key_reset, key_rollout = jax.random.split(key)
    if state is None:
        state = env.reset(key_reset)
        
    def step_fn(carry_state, key_step):
        carry, state = carry_state
        key_policy, _ = jax.random.split(key_step)
        carry, action = policy(carry, state.obs, key_policy)
        next_state = env.step(state, action)
        return (carry, next_state), (next_state, carry)

    keys_steps = jax.random.split(key_rollout, num_steps)
    (carry_final, _), (states, carries) = jax.lax.scan(
        step_fn, (carry_init, state), keys_steps
    )
    
    states = jax.tree_util.tree_map(
        lambda l0, l1: jnp.concatenate([l0[None], l1]), state, states
    )
    return states, carries
