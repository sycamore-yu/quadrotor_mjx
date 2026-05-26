"""Acceptance gate configuration and threshold helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


EXPECTED_BACKENDS = {
    "bptt": "jax_bptt",
    "ppo": "brax_ppo",
    "shac": "jax_shac",
}


def expected_backend_for_algo(algo: str) -> str:
    """Returns the backend name expected for an algorithm."""

    try:
        return EXPECTED_BACKENDS[algo]
    except KeyError as exc:
        available = ", ".join(sorted(EXPECTED_BACKENDS))
        raise ValueError(f"Unknown algorithm {algo!r}. Expected one of: {available}") from exc


@dataclass(slots=True)
class AcceptanceMetricGate:
    """Auxiliary metric gate for non-random scene behaviour checks."""

    name: str
    direction: str
    threshold: float
    vs_random_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AcceptanceThresholds:
    """Thresholds used by the acceptance gate."""

    min_reward_delta: float
    primary_metric_name: str
    primary_metric_direction: str
    primary_metric_threshold: float
    primary_metric_vs_random_delta: float = 0.0
    additional_metrics: tuple[AcceptanceMetricGate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["additional_metrics"] = [gate.to_dict() for gate in self.additional_metrics]
        return data


@dataclass(slots=True)
class AcceptanceConfig:
    """Parsed configuration for a gate run."""

    env: str
    algo: str
    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    num_epochs: int
    num_steps_per_epoch: int
    num_envs: int
    learning_rate: float
    thresholds: AcceptanceThresholds
    eval_episodes: int = 8
    eval_steps: int | None = None
    output_dir: str = "artifacts/acceptance"
    expected_backend: str | None = None
    train_kwargs: dict[str, Any] | None = None
    eval_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AcceptanceConfig":
        """Creates a config from a YAML-compatible mapping."""

        train_seeds = _coerce_int_sequence(mapping.get("train_seeds", mapping.get("train_seed", (0,))))
        eval_seeds = _coerce_int_sequence(mapping.get("eval_seeds", mapping.get("eval_seed", (0,))))
        algo = str(mapping["algo"])
        thresholds = _parse_thresholds(mapping)

        return cls(
            env=str(mapping["env"]),
            algo=algo,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            num_epochs=int(mapping.get("num_epochs", 1)),
            num_steps_per_epoch=int(mapping.get("num_steps_per_epoch", 1)),
            num_envs=int(mapping.get("num_envs", 1)),
            learning_rate=float(mapping.get("learning_rate", 3e-4)),
            thresholds=thresholds,
            eval_episodes=int(mapping.get("eval_episodes", len(eval_seeds) or 1)),
            eval_steps=_optional_int(mapping.get("eval_steps")),
            output_dir=str(mapping.get("output_dir", "artifacts/acceptance")),
            expected_backend=str(
                mapping.get("expected_backend", expected_backend_for_algo(algo))
            ),
            train_kwargs=dict(mapping.get("train_kwargs", {})),
            eval_kwargs=dict(mapping.get("eval_kwargs", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON/YAML serialisable representation."""

        data = asdict(self)
        data["thresholds"] = self.thresholds.to_dict()
        return data


def load_acceptance_config(path: str | Path) -> AcceptanceConfig:
    """Loads an acceptance config from YAML."""

    mapping = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(mapping, Mapping):
        raise TypeError(f"Acceptance config must be a mapping, got {type(mapping).__name__}")
    return AcceptanceConfig.from_mapping(mapping)


def apply_threshold_overrides(
    thresholds: AcceptanceThresholds,
    *,
    min_reward_delta: float | None = None,
    primary_metric_threshold: float | None = None,
    primary_metric_vs_random_delta: float | None = None,
) -> AcceptanceThresholds:
    """Applies threshold overrides without allowing weaker acceptance criteria."""

    if min_reward_delta is not None and min_reward_delta < thresholds.min_reward_delta:
        raise ValueError(
            "Refusing to lower min_reward_delta "
            f"from {thresholds.min_reward_delta} to {min_reward_delta}."
        )
    if (
        primary_metric_threshold is not None
        and _threshold_is_weaker(
            primary_metric_threshold,
            thresholds.primary_metric_threshold,
            thresholds.primary_metric_direction,
        )
    ):
        raise ValueError(
            "Refusing to lower primary_metric_threshold "
            f"from {thresholds.primary_metric_threshold} to {primary_metric_threshold}."
        )
    if (
        primary_metric_vs_random_delta is not None
        and primary_metric_vs_random_delta < thresholds.primary_metric_vs_random_delta
    ):
        raise ValueError(
            "Refusing to lower primary_metric_vs_random_delta "
            f"from {thresholds.primary_metric_vs_random_delta} "
            f"to {primary_metric_vs_random_delta}."
        )

    return replace(
        thresholds,
        min_reward_delta=(
            thresholds.min_reward_delta
            if min_reward_delta is None
            else float(min_reward_delta)
        ),
        primary_metric_threshold=(
            thresholds.primary_metric_threshold
            if primary_metric_threshold is None
            else float(primary_metric_threshold)
        ),
        primary_metric_vs_random_delta=(
            thresholds.primary_metric_vs_random_delta
            if primary_metric_vs_random_delta is None
            else float(primary_metric_vs_random_delta)
        ),
    )


def _parse_thresholds(mapping: Mapping[str, Any]) -> AcceptanceThresholds:
    raw = mapping.get("thresholds", mapping)
    if not isinstance(raw, Mapping):
        raise TypeError(f"thresholds must be a mapping, got {type(raw).__name__}")

    additional_metrics = raw.get("additional_metrics", raw.get("auxiliary_metrics", ()))
    if not isinstance(additional_metrics, Sequence) or isinstance(
        additional_metrics, (str, bytes)
    ):
        raise TypeError("additional_metrics must be a list of mappings.")

    return AcceptanceThresholds(
        min_reward_delta=float(raw.get("min_reward_delta", mapping.get("min_reward_delta", 0.0))),
        primary_metric_name=str(
            raw.get("primary_metric_name", mapping.get("primary_metric_name", "mean_reward"))
        ),
        primary_metric_direction=str(
            raw.get("primary_metric_direction", mapping.get("primary_metric_direction", "max"))
        ),
        primary_metric_threshold=float(
            raw.get("primary_metric_threshold", mapping.get("primary_metric_threshold", 0.0))
        ),
        primary_metric_vs_random_delta=float(
            raw.get(
                "primary_metric_vs_random_delta",
                mapping.get("primary_metric_vs_random_delta", 0.0),
            )
        ),
        additional_metrics=tuple(
            AcceptanceMetricGate(
                name=str(item["name"]),
                direction=str(item["direction"]),
                threshold=float(item["threshold"]),
                vs_random_delta=float(item.get("vs_random_delta", 0.0)),
            )
            for item in additional_metrics
        ),
    )


def _threshold_is_weaker(candidate: float, current: float, direction: str) -> bool:
    if direction == "min":
        return candidate > current
    if direction == "max":
        return candidate < current
    raise ValueError(
        f"Unknown primary_metric_direction {direction!r}; expected 'min' or 'max'."
    )


def _coerce_int_sequence(value: Any) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
