from brax.envs.base import Env
from flax import struct
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import Any, Dict, Tuple

from dva_quadrotor_mjx.envs.base import MjxEnv, State
from dva_quadrotor_mjx.controllers.low_level import LowLevelController
from dva_quadrotor_mjx.controllers.motor_model import MotorModel
from dva_quadrotor_mjx.dynamics.quadrotor import QuadrotorDynamics
from dva_quadrotor_mjx.utils.math import quat_to_rot_matrix, smooth_l1


class Box:
    """Minimal Box space compatible with rpg_flightning's spaces.Box."""

    def __init__(self, low, high, shape=None):
        self.low = jnp.array(low)
        self.high = jnp.array(high)
        if shape is not None:
            self.shape = shape
        else:
            self.shape = self.low.shape

    def sample(self, key):
        return jax.random.uniform(
            key, shape=self.shape, minval=self.low, maxval=self.high
        )


class HoveringStateEnv(MjxEnv):
    """MJX implementation of rpg_flightning's HoveringStateEnv.

    Observation schema (dim = 15 + num_last_actions * 4):
        - position (3): [x, y, z] in world frame
        - rotation matrix (9): flattened 3x3 rotation matrix R_WB
        - velocity (3): [vx, vy, vz] in world frame
        - action history (num_last_actions * 4): queued actions for delay handling

    Action schema (dim = 4):
        - total thrust (1): cumulative thrust [N]
        - body rates (3): [omega_x, omega_y, omega_z] commanded [rad/s]
    """

    def __init__(
        self,
        *,
        max_steps_in_episode: int = 1000,
        dt: float = 0.02,
        delay: float = 0.02,
        yaw_scale: float = 0.1,
        pitch_roll_scale: float = 0.1,
        velocity_std: float = 0.1,
        omega_std: float = 0.1,
        reward_sharpness: float = 1.0,
        action_penalty_weight: float = 1.0,
        margin: float = 0.5,
        sensor_mode: str = "state",
        **kwargs
    ):
        spec = kwargs.get("spec", None)
        if spec is None:
            from dva_quadrotor_mjx.envs import get_env_spec
            spec = get_env_spec("quadrotor_hover")

        mj_model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))

        physics_steps_per_control_step = int(round(dt / mj_model.opt.timestep))
        super().__init__(
            mj_model=mj_model,
            physics_steps_per_control_step=physics_steps_per_control_step
        )
        self.sensor_mode = sensor_mode

        self.max_steps_in_episode = max_steps_in_episode
        self.dt_control = dt
        self.sim_dt = mj_model.opt.timestep
        self.yaw_scale = yaw_scale
        self.pitch_roll_scale = pitch_roll_scale
        self.velocity_std = velocity_std
        self.omega_std = omega_std
        self.reward_sharpness = reward_sharpness
        self.action_penalty_weight = action_penalty_weight
        self.margin = margin
        self.goal = jnp.array([0.0, 0.0, 1.0])

        # World boundaries
        self.world_min = jnp.array([-5.0, -5.0, 0.0])
        self.world_max = jnp.array([5.0, 5.0, 3.0])

        # Actuator parameters
        self.mass = 0.75
        self.hovering_action = jnp.array([self.mass * 9.81, 0.0, 0.0, 0.0])
        self.motor_tau = 0.033

        # Action space bounds
        self.action_space_low = jnp.array([0.0, -6.0, -6.0, -4.0])
        self.action_space_high = jnp.array([12.0, 6.0, 6.0, 4.0])

        self.delay = delay
        self.num_last_actions = int(jnp.ceil(delay / dt)) + 1

        self.low_level = LowLevelController(
            inertia=jnp.array([0.0023, 0.0023, 0.0040]),
            k_p=jnp.array([20.0, 20.0, 41.0]),
            ctrl_min=0.0,
            ctrl_max=3.0
        )
        self.motor_model = MotorModel(
            motor_tau=self.motor_tau,
            ctrl_min=0.0,
            ctrl_max=3.0
        )
        self.dynamics = QuadrotorDynamics(
            sys=self.sys,
            low_level=self.low_level,
            motor_model=self.motor_model,
            sim_dt=self.sim_dt,
            physics_steps_per_control_step=self._physics_steps_per_control_step,
        )

    @property
    def observation_size(self) -> Tuple[int, ...]:
        n = self.num_last_actions
        size = 15 + n * 4
        if self.sensor_mode == "rangefinder":
            if self.__class__.__name__ == "ForestNavigationEnv":
                size += 2
            else:
                size += 1
        return (size,)

    @property
    def action_space(self) -> Box:
        """Action space: [thrust, omega_x, omega_y, omega_z]."""
        return Box(
            low=self.action_space_low,
            high=self.action_space_high,
            shape=(4,),
        )

    @property
    def observation_space(self) -> Box:
        """Observation space."""
        n = self.num_last_actions
        action_high = jnp.concatenate([self.action_space_high] * n)
        action_low = jnp.concatenate([self.action_space_low] * n)
        low = jnp.concatenate([
            self.world_min,
            -jnp.ones(9),
            jnp.array([-10.0, -10.0, -10.0]),
            action_low,
        ])
        high = jnp.concatenate([
            self.world_max,
            jnp.ones(9),
            jnp.array([10.0, 10.0, 10.0]),
            action_high,
        ])
        if self.sensor_mode == "rangefinder":
            if self.__class__.__name__ == "ForestNavigationEnv":
                low = jnp.concatenate([low, jnp.zeros(2)])
                high = jnp.concatenate([high, jnp.ones(2)])
            else:
                low = jnp.concatenate([low, jnp.zeros(1)])
                high = jnp.concatenate([high, jnp.ones(1)])
        return Box(
            low=low,
            high=high,
            shape=(self.observation_size[0],),
        )

    def reset(self, rng: jax.Array) -> State:
        key_p, key_R, key_v, key_omega = jax.random.split(rng, 4)

        # Randomize position within margins
        p = jax.random.uniform(
            key_p,
            shape=(3,),
            minval=self.world_min + self.margin,
            maxval=self.world_max - self.margin
        )

        # Randomize orientation Euler angles (yaw, pitch, roll)
        yaw = self.yaw_scale * jax.random.uniform(
            key_R, minval=-jnp.pi, maxval=jnp.pi
        )
        key_p_angle, key_r_angle = jax.random.split(key_R)
        pitch = self.pitch_roll_scale * jax.random.uniform(
            key_p_angle, minval=-jnp.pi, maxval=jnp.pi
        )
        roll = self.pitch_roll_scale * jax.random.uniform(
            key_r_angle, minval=-jnp.pi, maxval=jnp.pi
        )

        # Build quaternion (w, x, y, z)
        cy = jnp.cos(yaw * 0.5)
        sy = jnp.sin(yaw * 0.5)
        cp = jnp.cos(pitch * 0.5)
        sp = jnp.sin(pitch * 0.5)
        cr = jnp.cos(roll * 0.5)
        sr = jnp.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        quat = jnp.array([qw, qx, qy, qz])
        quat = quat / jnp.linalg.norm(quat)

        # Randomize velocities
        v = self.velocity_std * jax.random.normal(key_v, shape=(3,))
        omega_body = self.omega_std * jax.random.normal(key_omega, shape=(3,))

        R = quat_to_rot_matrix(quat)
        omega_world = R @ omega_body

        qpos = jnp.concatenate([p, quat])
        qvel = jnp.concatenate([v, omega_world])

        # Init physics data
        data = self.pipeline_init(qpos, qvel)

        # Initialize action queue with hover actions
        last_actions = jnp.tile(self.hovering_action, (self.num_last_actions, 1))

        # Motor forces starts at hovering force
        motor_force = jnp.array([self.mass * 9.81 / 4.0] * 4)

        info = {
            "steps": jnp.zeros((), dtype=jnp.int32),
            "truncation": jnp.zeros(()),
            "reward_tuple": {
                "pos": jnp.zeros(()),
                "vel": jnp.zeros(()),
                "omega": jnp.zeros(()),
                "action": jnp.zeros(()),
                "acc": jnp.zeros(()),
            },
            "last_actions": last_actions,
            "motor_force": motor_force,
            "collision": jnp.array(False),
            "target_distance": jnp.linalg.norm(p - self.goal),
        }

        obs = self._get_obs(data, info)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {
            "pos_error": jnp.linalg.norm(p - self.goal),
            "target_distance": jnp.linalg.norm(p - self.goal),
            "vel_norm": jnp.linalg.norm(v),
            "reward": reward,
        }

        return State(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info
        )

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        p = data.qpos[0:3]
        quat = data.qpos[3:7]
        R = quat_to_rot_matrix(quat)
        v = data.qvel[0:3]
        last_actions = info["last_actions"]

        obs = jnp.concatenate([
            p,
            R.flatten(),
            v,
            last_actions.flatten()
        ])

        if self.sensor_mode == "rangefinder":
            from dva_quadrotor_mjx.envs.sensors.rangefinder import get_rangefinder_reading, normalize_rangefinder
            raw_reading = get_rangefinder_reading(data)
            norm_reading = normalize_rangefinder(raw_reading)

            if self.__class__.__name__ == "ForestNavigationEnv":
                pos_2d = p[0:2]
                dir_world = R[:, 0]
                dir_2d = dir_world[0:2]
                dir_2d_norm = dir_2d / jnp.maximum(jnp.linalg.norm(dir_2d), 1e-6)

                from dva_quadrotor_mjx.envs.sensors.raycasting import raycast_cylinders
                tree_positions = info.get("tree_positions", jnp.zeros((1, 3)))
                tree_radii = info.get("tree_radii", jnp.zeros((1,)))

                ray_reading = raycast_cylinders(pos_2d, dir_2d_norm, tree_positions, tree_radii)
                obs = jnp.concatenate([obs, jnp.array([norm_reading, ray_reading])])
            else:
                obs = jnp.concatenate([obs, jnp.array([norm_reading])])
        return obs

    def step(self, state: State, action: jax.Array) -> State:
        # Clip action to valid range
        action = jnp.clip(action, self.action_space_low, self.action_space_high)

        # Roll actions in queue
        last_actions = state.info["last_actions"]
        last_actions = jnp.roll(last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # Apply delayed action
        active_action = last_actions[0]
        f_T = active_action[0]
        omega_cmd = active_action[1:]

        data = state.pipeline_state
        motor_force = state.info["motor_force"]

        # Step quadrotor dynamics
        data, motor_force_new, acc = self.dynamics.step(
            data, motor_force, f_T, omega_cmd
        )

        # Computations after substeps
        p = data.qpos[0:3]
        v = data.qvel[0:3]
        quat = data.qpos[3:7]
        R = quat_to_rot_matrix(quat)
        omega = R.T @ data.qvel[3:6]

        # Termination conditions (out of bounds or NaN)
        is_colliding = jnp.logical_or(
            jnp.any(p < self.world_min),
            jnp.any(p > self.world_max)
        )
        is_nan = (
            jnp.any(jnp.isnan(p))
            | jnp.any(jnp.isnan(v))
            | jnp.any(jnp.isnan(omega))
        )

        terminated = jnp.where(is_colliding | is_nan, 1.0, 0.0)

        # Step count and truncation
        steps = state.info["steps"] + 1
        truncated = jnp.where(steps >= self.max_steps_in_episode, 1.0, 0.0)

        done = jnp.where(terminated > 0.5, 1.0, truncated)

        # Compute rewards (matching rpg_flightning)
        pos_cost = smooth_l1(
            self.reward_sharpness * (p - self.goal)
        ) / self.reward_sharpness
        vel_cost = 0.1 * smooth_l1(v)
        omega_cost = 0.1 * smooth_l1(omega)
        acc_cost = 0.1 * smooth_l1(acc)
        goal_cost = pos_cost + vel_cost + omega_cost + acc_cost

        action_cost = self.action_penalty_weight * smooth_l1(
            action - self.hovering_action
        )

        # Smoothness penalty
        action_last = state.info["last_actions"][-1]
        smoothness_cost = 0.01 * jnp.inner(action, action_last)

        cost = goal_cost + action_cost + smoothness_cost

        # Collision penalty
        time_left = self.max_steps_in_episode - steps
        collision_cost = jax.lax.select(terminated > 0.5, time_left * cost, 0.0)

        cost = cost + jax.lax.stop_gradient(collision_cost)
        reward = -self.dt_control * cost

        info = dict(state.info)
        info.update({
            "steps": steps,
            "truncation": truncated,
            "reward_tuple": {
                "pos": pos_cost,
                "vel": vel_cost,
                "omega": omega_cost,
                "action": action_cost,
                "acc": acc_cost,
            },
            "last_actions": last_actions,
            "motor_force": motor_force_new,
            "target_distance": jnp.linalg.norm(p - self.goal),
        })

        obs = self._get_obs(data, info)

        metrics = {
            "pos_error": jnp.linalg.norm(p - self.goal),
            "target_distance": jnp.linalg.norm(p - self.goal),
            "vel_norm": jnp.linalg.norm(v),
            "reward": reward,
        }

        return State(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info
        )
