import jax
import jax.numpy as jnp
from mujoco import mjx

from dva_quadrotor_mjx.controllers.low_level import LowLevelController
from dva_quadrotor_mjx.controllers.motor_model import MotorModel
from dva_quadrotor_mjx.dynamics.drag import compute_drag_force
from dva_quadrotor_mjx.utils.math import quat_to_rot_matrix


class QuadrotorDynamics:
    """Combines motor model, rate controller, aerodynamic drag, and MJX simulation step."""

    def __init__(
        self,
        sys: mjx.Model,
        low_level: LowLevelController,
        motor_model: MotorModel,
        sim_dt: float,
        physics_steps_per_control_step: int,
    ):
        self.sys = sys
        self.low_level = low_level
        self.motor_model = motor_model
        self.sim_dt = sim_dt
        self.physics_steps_per_control_step = physics_steps_per_control_step

    def step(
        self,
        data: mjx.Data,
        motor_force: jax.Array,
        f_T: jax.Array,
        omega_cmd: jax.Array,
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        """Steps the quadrotor dynamics for one control step.

        Returns:
            data: Updated MJX data
            motor_force_new: Updated motor forces
            acc: World-frame linear acceleration
        """
        def substep_fn(carry, _):
            d, mf = carry

            # Current posture and rate
            qpos = d.qpos
            qvel = d.qvel
            quat = qpos[3:7]
            R = quat_to_rot_matrix(quat)
            omega = R.T @ qvel[3:6]

            # Run low-level controller
            f_cmd = self.low_level.compute_ctrl(omega, omega_cmd, f_T)

            # Update motor delay dynamics
            mf_new = self.motor_model.step(mf, f_cmd, self.sim_dt)

            # Set actuator control force
            d = d.replace(ctrl=mf_new)

            # Apply quadratic aerodynamic drag
            v = qvel[0:3]
            f_drag_world = compute_drag_force(R, v)

            # Body ID 1 is the quadrotor body
            xfrc = d.xfrc_applied
            xfrc = xfrc.at[1, :3].set(f_drag_world)
            d = d.replace(xfrc_applied=xfrc)

            # Compute acceleration for reward (matching rpg_flightning)
            mass = 0.75
            gravity = jnp.array([0.0, 0.0, -9.81])
            thrust_body = jnp.array([0.0, 0.0, jnp.sum(mf_new)])
            f_drag_body = R.T @ f_drag_world
            acc = gravity + R @ (thrust_body + f_drag_body) / mass

            # Advance physics step
            d = mjx.step(self.sys, d)
            return (d, mf_new), acc

        (data, motor_force_new), acc_seq = jax.lax.scan(
            substep_fn,
            (data, motor_force),
            None,
            length=self.physics_steps_per_control_step
        )
        # Return the last acceleration
        acc = acc_seq[-1] if acc_seq.shape[0] > 0 else jnp.zeros(3)
        return data, motor_force_new, acc
