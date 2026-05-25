import jax
from brax.envs.base import Wrapper, State

class VecEnv(Wrapper):
    """Vectorize an environment for parallelization using jax.vmap."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_vmap = jax.vmap(self.env.reset)
        self._step_vmap = jax.vmap(self.env.step)

    def reset(self, rng: jax.Array) -> State:
        # rng is shape [num_envs, ...]
        return self._reset_vmap(rng)

    def step(self, state: State, action: jax.Array) -> State:
        # state and action are batched
        return self._step_vmap(state, action)
