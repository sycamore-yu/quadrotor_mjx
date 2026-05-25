import jax
import jax.numpy as jnp

class LowLevelController:
    """Low-level P-controller for quadrotor angular rate tracking."""

    def __init__(
        self,
        inertia: jax.Array = jnp.array([0.0023, 0.0023, 0.0040]),
        k_p: jax.Array = jnp.array([20.0, 20.0, 41.0]),
        ctrl_min: float = 0.0,
        ctrl_max: float = 3.0,
    ):
        self.J = inertia
        self.K_p = k_p
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max

        # Construct allocation matrix
        # Rotor order: fl (rotor_fl), fr (rotor_fr), rl (rotor_rl), rr (rotor_rr)
        # Coordinates and yaw direction (kappa) from quadrotor_hover.xml
        x = jnp.array([0.13, 0.13, -0.13, -0.13])
        y = jnp.array([0.13, -0.13, 0.13, -0.13])
        kappa = jnp.array([0.008, -0.008, -0.008, 0.008])

        self.A = jnp.stack([
            jnp.ones(4),
            y,
            -x,
            kappa
        ])
        self.A_inv = jnp.linalg.inv(self.A)

    def compute_ctrl(self, omega: jax.Array, omega_cmd: jax.Array, f_T: jax.Array) -> jax.Array:
        """Computes low level motor controls.
        
        Args:
            omega: Current body-frame angular velocity (shape: (3,))
            omega_cmd: Commanded body-frame angular velocity (shape: (3,))
            f_T: Commanded total thrust (scalar or shape: (1,))
        
        Returns:
            ctrl: Motor commands (shape: (4,))
        """
        omega_err = omega_cmd - omega
        
        # P control on rates with Coriolis compensation
        # torque = J * K_p * error + omega x (J * omega)
        torques_cmd = self.J * self.K_p * omega_err + jnp.cross(omega, self.J * omega)
        
        alpha = jnp.concatenate([jnp.atleast_1d(f_T), torques_cmd])
        ctrl = self.A_inv @ alpha
        
        return jnp.clip(ctrl, self.ctrl_min, self.ctrl_max)
