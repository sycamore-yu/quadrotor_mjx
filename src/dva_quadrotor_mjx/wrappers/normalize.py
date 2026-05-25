import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper, State

class NormalizeActionWrapper(Wrapper):
    """Normalize the actions of the environment to [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space_low = getattr(env, "action_space_low", jnp.array([-1.0]))
        self.action_space_high = getattr(env, "action_space_high", jnp.array([1.0]))

    def map_action(self, action: jax.Array) -> jax.Array:
        # map action from [-1, 1] to action bounds
        return (action + 1.0) / 2.0 * (self.action_space_high - self.action_space_low) + self.action_space_low

    def step(self, state: State, action: jax.Array) -> State:
        mapped_action = self.map_action(action)
        return self.env.step(state, mapped_action)


class MinMaxObservationWrapper(Wrapper):
    """Normalize the observations of the environment to [-1, 1]."""

    def __init__(self, env, obs_min: jax.Array, obs_max: jax.Array):
        super().__init__(env)
        self._obs_min = obs_min
        self._obs_max = obs_max

    def _normalize(self, obs: jax.Array) -> jax.Array:
        # Avoid division by zero
        denom = self._obs_max - self._obs_min
        denom = jnp.where(denom == 0, 1.0, denom)
        # Normalize to [-1, 1]
        return 2.0 * (obs - self._obs_min) / denom - 1.0

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state.replace(obs=self._normalize(state.obs))

    def step(self, state: State, action: jax.Array) -> State:
        next_state = self.env.step(state, action)
        return next_state.replace(obs=self._normalize(next_state.obs))
