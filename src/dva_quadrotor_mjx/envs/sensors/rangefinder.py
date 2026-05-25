"""Rangefinder sensor utilities for obstacle detection."""

import jax
import jax.numpy as jnp
from mujoco import mjx


def get_rangefinder_reading(data: mjx.Data, sensor_name: str = "rangefinder") -> jax.Array:
    """Extract rangefinder reading from MJX sensor data.

    Args:
        data: MJX data containing sensor readings
        sensor_name: Name of the rangefinder sensor

    Returns:
        Distance reading (scalar), -1.0 if no hit
    """
    # In MJX, sensor data is in data.sensordata
    # For a single rangefinder, it's the first element
    if hasattr(data, "sensordata") and data.sensordata is not None:
        return data.sensordata[0]
    return jnp.array(-1.0)


def normalize_rangefinder(reading: jax.Array, max_range: float = 10.0) -> jax.Array:
    """Normalize rangefinder reading to [0, 1].

    Args:
        reading: Raw distance reading
        max_range: Maximum expected range

    Returns:
        Normalized reading in [0, 1], 0.0 if no hit (-1.0)
    """
    valid = reading >= 0.0
    normalized = jnp.where(
        valid,
        jnp.clip(reading / max_range, 0.0, 1.0),
        0.0
    )
    return normalized
