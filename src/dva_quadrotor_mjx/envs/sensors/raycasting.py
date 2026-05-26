"""JAX-differentiable cylinder raycasting for obstacle range sensors."""

import jax
import jax.numpy as jnp


def raycast_cylinders(
    pos_2d: jax.Array,
    dir_2d: jax.Array,
    tree_positions: jax.Array,
    tree_radii: jax.Array,
    max_range: float = 10.0,
) -> jax.Array:
    """Computes JAX-differentiable 2D Cylinder Raycast intersection.

    Args:
        pos_2d: (2,) array containing quadrotor 2D position.
        dir_2d: (2,) normalized array containing forward 2D heading.
        tree_positions: (N, 3) or (N, 2) array containing cylinder centers.
        tree_radii: (N,) array containing cylinder radii.
        max_range: Maximum range of the raycaster.

    Returns:
        Closest normalized distance in [0, 1].
    """
    c = tree_positions[:, 0:2]  # (N, 2)
    r = tree_radii  # (N,)

    # Vector from quadrotor to cylinder center
    v = c - pos_2d  # (N, 2)

    # Equation: t^2 + 2 B t + C = 0
    # A = 1 (since dir_2d is normalized)
    # B = (pos_2d - c) . dir_2d = - v . dir_2d
    # C = ||pos_2d - c||^2 - r^2 = ||v||^2 - r^2
    B = -jnp.dot(v, dir_2d)  # (N,)
    C = jnp.sum(v**2, axis=-1) - r**2  # (N,)

    discriminant = B**2 - C  # (N,)
    has_intersection = discriminant >= 0

    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t1 = -B - sqrt_disc
    t2 = -B + sqrt_disc

    # Nearest positive intersection
    t_valid = jnp.where(t1 >= 0.0, t1, jnp.where(t2 >= 0.0, t2, max_range))
    dist = jnp.where(has_intersection, t_valid, max_range)

    # Minimum distance across all obstacles
    min_dist = jnp.min(dist)

    # Return normalized distance in [0, 1]
    return jnp.clip(min_dist / max_range, 0.0, 1.0)
