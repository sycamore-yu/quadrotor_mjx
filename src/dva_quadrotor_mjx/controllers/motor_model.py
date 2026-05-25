import jax
import jax.numpy as jnp

class MotorModel:
    """First-order motor delay model for quadrotor actuators."""

    def __init__(self, motor_tau: float = 0.033, ctrl_min: float = 0.0, ctrl_max: float = 3.0):
        self.motor_tau = motor_tau
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max

    def step(self, motor_force: jax.Array, f_cmd: jax.Array, dt: float) -> jax.Array:
        """Updates motor forces using first-order delay dynamics."""
        mf_new = (motor_force - f_cmd) * jnp.exp(-dt / self.motor_tau) + f_cmd
        return jnp.clip(mf_new, self.ctrl_min, self.ctrl_max)
