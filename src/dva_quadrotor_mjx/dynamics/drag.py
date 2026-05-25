import jax
import jax.numpy as jnp

def compute_drag_force(
    R: jax.Array,
    v: jax.Array,
    key: jax.Array = None,
    horizontal_drag_coeff: float = 1.04,
    vertical_drag_coeff: float = 1.04,
    frontarea_x: float = 1.0e-3,
    frontarea_y: float = 1.0e-3,
    frontarea_z: float = 1.0e-2,
    air_density: float = 1.2,
) -> jax.Array:
    """Computes quadratic drag force in the body frame, then rotates it to world frame.
    
    This matches the rpg_flightning drag model equation:
    f_drag_body = -0.5 * air_density * drag_coeff * area * v_body * abs(v_body)
    f_drag_world = R @ f_drag_body
    """
    v_body = R.T @ v
    area = jnp.array([frontarea_x, frontarea_y, frontarea_z])
    drag_coeff = jnp.array([horizontal_drag_coeff, horizontal_drag_coeff, vertical_drag_coeff])
    
    if key is not None:
        drag_coeff = jax.random.uniform(
            key, drag_coeff.shape, minval=0.5 * drag_coeff, maxval=1.5 * drag_coeff
        )
        
    f_drag_body = -0.5 * air_density * drag_coeff * area * v_body * jnp.abs(v_body)
    f_drag_world = R @ f_drag_body
    return f_drag_world
