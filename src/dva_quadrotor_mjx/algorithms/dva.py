"""DVA algorithm placeholder for ``openspec/changes/dva-quadrotor-mjx``.

Do not remove this module while completing the rpg-flightning migration.  The
DVA change is currently at M5 and will fill this seam with a vision/lidar actor,
privileged-state critic, rollout buffer, and JAX actor update.
"""


class DvaNotImplemented(RuntimeError):
    """Raised when callers try to use DVA before the M5 implementation lands."""


def train(*args, **kwargs):
    """Reserved DVA training entry point."""

    del args, kwargs
    raise DvaNotImplemented(
        "DVA training belongs to openspec/changes/dva-quadrotor-mjx M5 and is not implemented yet."
    )


__all__ = ["DvaNotImplemented", "train"]
