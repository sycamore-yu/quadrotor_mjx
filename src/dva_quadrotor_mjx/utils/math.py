import jax
import jax.numpy as jnp

def quat_to_rot_matrix(q: jax.Array) -> jax.Array:
    """Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.stack([
        jnp.stack([1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - w*z), 2.0*(x*z + w*y)]),
        jnp.stack([2.0*(x*y + w*z), 1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - w*x)]),
        jnp.stack([2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x**2 + y**2)])
    ])

def smooth_l1(x: jax.Array) -> jax.Array:
    """Smoothed L1 norm matching rpg_flightning's implementation."""
    delta = 1.0
    abs_errors = jnp.linalg.norm(x + 1e-6)
    quadratic = jnp.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear
