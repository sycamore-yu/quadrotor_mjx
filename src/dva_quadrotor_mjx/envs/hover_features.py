import jax
import jax.numpy as jnp

from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
from dva_quadrotor_mjx.envs.sensors.feature_camera import create_example_camera
from dva_quadrotor_mjx.utils.math import quat_to_rot_matrix


class HoveringFeaturesEnv(HoveringStateEnv):
    """Feature-based hovering environment with landmark projection.

    Observation schema (dim = 2 * 7 + num_last_actions * 4):
        - projected landmarks (14): 7 feature points projected to 2D image plane
          normalized to [-1, 1]
        - action history (num_last_actions * 4): normalized queued actions

    The 7 feature points are fixed world coordinates at z=0 plane:
        [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0], [-0.5, 0, 0.0], [0, 0.5, 0.0], [0.5, 0, 0.0]
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
        skip_frames=1,
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
        self.cam = create_example_camera()
        self.cam.pitch = -90.0
        self.skip_frames = skip_frames

        # Fixed feature points in world frame (z=0 plane)
        self.feature_pos = jnp.array([
            [0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ])

    @property
    def observation_size(self):
        num_frames = 1  # skip_frames=1 means only current frame
        return (2 * 7 * num_frames + 4 * self.num_last_actions,)

    def _get_obs(self, data, info):
        p = data.qpos[0:3]
        quat = data.qpos[3:7]
        R = quat_to_rot_matrix(quat)

        # Project feature points to image plane
        projected = self.cam.project_points_with_pose(
            self.feature_pos, p, R
        )
        # Remove validity flag, take only u,v
        points_C = projected[:, :2]

        # Normalize to [-1, 1]
        points_normalized = points_C / jnp.array(
            [self.cam.width, self.cam.height], dtype=float
        ) * jnp.array([2.0, 2.0]) - jnp.array([1.0, 1.0])

        # Normalize action history to [-1, 1]
        last_actions = info["last_actions"].flatten()
        actions_low = jnp.concatenate(
            [self.action_space_low] * self.num_last_actions
        )
        actions_high = jnp.concatenate(
            [self.action_space_high] * self.num_last_actions
        )
        last_action_normalized = 2.0 * (
            last_actions - actions_low
        ) / (actions_high - actions_low) - 1.0

        return jnp.concatenate([
            points_normalized.flatten(),
            last_action_normalized
        ])
