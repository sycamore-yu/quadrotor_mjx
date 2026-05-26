"""Forest navigation environment with procedural obstacles."""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
from dva_quadrotor_mjx.envs.sensors.raycasting import raycast_cylinders
from dva_quadrotor_mjx.utils.math import quat_to_rot_matrix


class ForestNavigationEnv(HoveringStateEnv):
    """Forest navigation with procedural cylindrical obstacles.

    Uses deterministic procedural generation based on PRNG seed.
    """

    def __init__(
        self,
        *,
        max_steps_in_episode=2000,
        dt=0.02,
        delay=0.02,
        yaw_scale=0.1,
        pitch_roll_scale=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        reward_sharpness=1.0,
        action_penalty_weight=1.0,
        margin=0.5,
        num_trees=10,
        forest_size=10.0,
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
        self.num_trees = num_trees
        self.forest_size = forest_size

    def reset(self, rng):
        key_forest = jax.random.split(rng, 5)[4]

        # Generate forest obstacles
        tree_positions = jax.random.uniform(
            key_forest,
            shape=(self.num_trees, 3),
            minval=jnp.array([-self.forest_size/2, -self.forest_size/2, 0.0]),
            maxval=jnp.array([self.forest_size/2, self.forest_size/2, 3.0])
        )
        tree_radii = jax.random.uniform(
            key_forest,
            shape=(self.num_trees,),
            minval=0.1,
            maxval=0.5
        )

        # Sample start and goal ensuring they are not inside trees
        start = jnp.array([-self.forest_size/2 + 1.0, 0.0, 1.0])
        goal = jnp.array([self.forest_size/2 - 1.0, 0.0, 1.0])

        state = super().reset(rng)

        info = dict(state.info)
        info["tree_positions"] = tree_positions
        info["tree_radii"] = tree_radii
        info["goal"] = goal
        info["start"] = start
        info["collision"] = jnp.array(False)
        info["success"] = jnp.array(False)
        dist_to_goal = jnp.linalg.norm(state.pipeline_state.qpos[0:3] - goal)
        start_distance = jnp.linalg.norm(start - goal)
        info["dist_to_goal"] = dist_to_goal
        info["goal_progress"] = 1.0 - dist_to_goal / jnp.maximum(start_distance, 1e-6)
        info["rangefinder_clearance"] = jnp.array(1.0)

        obs = self._get_obs(state.pipeline_state, info)
        metrics = dict(state.metrics)
        metrics["dist_to_goal"] = dist_to_goal
        metrics["goal_progress"] = info["goal_progress"]
        metrics["rangefinder_clearance"] = info["rangefinder_clearance"]
        return state.replace(obs=obs, info=info, metrics=metrics)

    def step(self, state, action):
        next_state = super().step(state, action)

        p = next_state.pipeline_state.qpos[0:3]
        goal = state.info["goal"]
        tree_positions = state.info["tree_positions"]
        tree_radii = state.info["tree_radii"]

        # Distance to goal reward
        dist_to_goal = jnp.linalg.norm(p - goal)
        goal_reward = -0.01 * dist_to_goal
        start_distance = jnp.linalg.norm(state.info["start"] - goal)
        goal_progress = 1.0 - dist_to_goal / jnp.maximum(start_distance, 1e-6)

        # Check collision with trees
        dists = jnp.linalg.norm(tree_positions - p, axis=1)
        collided = jnp.any(dists < tree_radii + 0.1)

        quat = next_state.pipeline_state.qpos[3:7]
        R = quat_to_rot_matrix(quat)
        pos_2d = p[0:2]
        dir_world = R[:, 0]
        dir_2d = dir_world[0:2]
        dir_2d_norm = dir_2d / jnp.maximum(jnp.linalg.norm(dir_2d), 1e-6)
        rangefinder_clearance = raycast_cylinders(
            pos_2d,
            dir_2d_norm,
            tree_positions,
            tree_radii,
        )

        collision_penalty = jax.lax.select(
            collided,
            -10.0,
            0.0
        )

        # Success if close to goal
        success = dist_to_goal < 0.5
        success_bonus = jax.lax.select(success, 100.0, 0.0)

        reward = next_state.reward + goal_reward + collision_penalty + success_bonus

        done = jnp.where(success, 1.0, next_state.done)

        info = dict(next_state.info)
        info["collision"] = collided
        info["success"] = success
        info["dist_to_goal"] = dist_to_goal
        info["goal_progress"] = goal_progress
        info["rangefinder_clearance"] = rangefinder_clearance
        metrics = dict(next_state.metrics)
        metrics["dist_to_goal"] = dist_to_goal
        metrics["goal_progress"] = goal_progress
        metrics["rangefinder_clearance"] = rangefinder_clearance

        return next_state.replace(
            reward=reward,
            done=done,
            info=info,
            metrics=metrics,
        )
