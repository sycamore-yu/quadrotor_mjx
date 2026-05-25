"""Small environment registry following the MuJoCo Playground pattern."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from brax.envs.base import Env


@dataclass(frozen=True)
class EnvSpec:
    """Metadata needed to construct an environment without importing trainers."""

    name: str
    mjcf_path: Path
    description: str
    stage: str


def _asset_path(filename: str) -> Path:
    return Path(resources.files("dva_quadrotor_mjx.envs").joinpath("xmls", filename))


_ENV_SPECS = {
    "quadrotor_hover": EnvSpec(
        name="quadrotor_hover",
        mjcf_path=_asset_path("quadrotor_hover.xml"),
        description="Minimal MJCF quadrotor skeleton for M0/M1 bootstrap checks.",
        stage="M0",
    ),
    "hover_state": EnvSpec(
        name="hover_state",
        mjcf_path=_asset_path("quadrotor_hover.xml"),
        description="State-based hovering environment.",
        stage="M1",
    ),
    "hover_features": EnvSpec(
        name="hover_features",
        mjcf_path=_asset_path("quadrotor_hover.xml"),
        description="Feature-based hovering environment with landmark projection.",
        stage="M1",
    ),
    "hover_obstacle": EnvSpec(
        name="hover_obstacle",
        mjcf_path=_asset_path("hover_obstacle.xml"),
        description="Hovering with static obstacle avoidance.",
        stage="M2",
    ),
    "gate_crossing": EnvSpec(
        name="gate_crossing",
        mjcf_path=_asset_path("gate_crossing.xml"),
        description="Gate crossing navigation task.",
        stage="M2",
    ),
    "forest_navigation": EnvSpec(
        name="forest_navigation",
        mjcf_path=_asset_path("quadrotor_hover.xml"),
        description="Forest navigation with procedural obstacles.",
        stage="M2",
    ),
}


def list_envs() -> tuple[str, ...]:
    """Returns registered environment names in stable order."""

    return tuple(sorted(_ENV_SPECS))


def get_env_spec(name: str) -> EnvSpec:
    """Returns environment metadata by name."""

    try:
        return _ENV_SPECS[name]
    except KeyError as exc:
        available = ", ".join(list_envs())
        raise KeyError(f"Unknown environment {name!r}. Available: {available}") from exc


def load(name: str, **kwargs) -> Env:
    """Instantiates environment class by name."""
    if name == "quadrotor_hover":
        from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
        spec = get_env_spec(name)
        return HoveringStateEnv(spec=spec, **kwargs)
    elif name == "hover_state":
        from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
        spec = get_env_spec(name)
        return HoveringStateEnv(spec=spec, **kwargs)
    elif name == "hover_features":
        from dva_quadrotor_mjx.envs.hover_features import HoveringFeaturesEnv
        spec = get_env_spec(name)
        return HoveringFeaturesEnv(spec=spec, **kwargs)
    elif name == "hover_obstacle":
        from dva_quadrotor_mjx.envs.hover_obstacle import HoverObstacleEnv
        spec = get_env_spec(name)
        return HoverObstacleEnv(spec=spec, **kwargs)
    elif name == "gate_crossing":
        from dva_quadrotor_mjx.envs.gate_crossing import GateCrossingEnv
        spec = get_env_spec(name)
        return GateCrossingEnv(spec=spec, **kwargs)
    elif name == "forest_navigation":
        from dva_quadrotor_mjx.envs.forest_navigation import ForestNavigationEnv
        spec = get_env_spec(name)
        return ForestNavigationEnv(spec=spec, **kwargs)
    else:
        raise ValueError(f"Unknown environment {name!r}")
