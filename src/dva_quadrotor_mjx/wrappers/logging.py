import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper, State

class LogWrapper(Wrapper):
    """Adds logging to the environment."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        
        info = dict(state.info)
        info.update({
            "episode_returns": jnp.zeros(()),
            "episode_lengths": jnp.zeros((), dtype=jnp.int32),
            "returned_episode_returns": jnp.zeros(()),
            "returned_episode_lengths": jnp.zeros((), dtype=jnp.int32),
            "returned_episode": jnp.zeros(()),
        })
        
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        next_state = self.env.step(state, action)
        
        done = next_state.done
        
        episode_returns = state.info["episode_returns"] + next_state.reward
        episode_lengths = state.info["episode_lengths"] + 1
        
        returned_episode_returns = jnp.where(
            done > 0.5, episode_returns, state.info["returned_episode_returns"]
        )
        returned_episode_lengths = jnp.where(
            done > 0.5, episode_lengths, state.info["returned_episode_lengths"]
        )
        
        info = dict(next_state.info)
        info.update({
            "episode_returns": jnp.where(done > 0.5, 0.0, episode_returns),
            "episode_lengths": jnp.where(done > 0.5, 0, episode_lengths),
            "returned_episode_returns": returned_episode_returns,
            "returned_episode_lengths": returned_episode_lengths,
            "returned_episode": done,
        })
        
        metrics = dict(next_state.metrics)
        metrics.update({
            "episode_returns": returned_episode_returns,
            "episode_lengths": returned_episode_lengths,
        })
        
        return next_state.replace(info=info, metrics=metrics)
