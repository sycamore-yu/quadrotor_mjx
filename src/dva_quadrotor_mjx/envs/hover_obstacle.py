"""Hovering environment with static obstacle avoidance."""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
from dva_quadrotor_mjx.utils.math import smooth_l1


class HoverObstacleEnv(HoveringStateEnv):
    """Hovering with static obstacle.

    Adds a box obstacle at origin and collision penalty.
    """

    def __init__(
        self,
        *,
        max_steps_in_episode=1000,
        dt=0.02,
        delay=0.02,
        yaw_scale=0.1,
        pitch_roll_scale=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        reward_sharpness=1.0,
        action_penalty_weight=1.0,
        margin=0.5,
        **kwargs
    ):
        super().__init__(
            max_steps_in_episode=max_steps_in_episode,
            dt=dt,
            delay=delay,
            yaw_scale=yaw_scale,
            pitch_roll_scale=pitch_roll_scale,
            velocity_std=velocity_std,
            omega_std=omega_std,
            reward_sharpness=reward_sharpness,
            action_penalty_weight=action_penalty_weight,
            margin=margin,
            **kwargs
        )

    def step(self, state, action):
        next_state = super().step(state, action)

        # Check collision with obstacle (body ID 2 is obstacle)
        p = next_state.pipeline_state.qpos[0:3]
        obstacle_pos = jnp.array([0.0, 0.0, 0.5])
        obstacle_size = jnp.array([0.2, 0.2, 0.5])

        # Simple box collision check
        dist = jnp.abs(p - obstacle_pos)
        collided = jnp.all(dist < obstacle_size + 0.1)

        # Collision penalty
        collision_penalty = jax.lax.select(
            collided,
            -10.0,
            0.0
        )

        reward = next_state.reward + collision_penalty

        info = dict(next_state.info)
        info["collision"] = collided

        return next_state.replace(
            reward=reward,
            info=info
        )
