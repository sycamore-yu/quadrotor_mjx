"""Helpers for Brax PPO checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Returns a concrete Brax PPO checkpoint step directory."""

    path = Path(path).resolve()
    if (path / "ppo_network_config.json").exists():
        return path

    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "ppo_network_config.json").exists()
    ]
    if not candidates:
        raise ValueError(f"No Brax PPO checkpoint found under {path}")
    return sorted(candidates, key=lambda child: int(child.name))[-1]


def load_policy(path: str | Path) -> Callable:
    """Loads a deterministic Brax PPO policy from a checkpoint directory."""

    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training import networks as training_networks
    from brax.training.acme import running_statistics

    path = resolve_checkpoint_path(path)
    config = json.loads((path / "ppo_network_config.json").read_text())
    kwargs = config.get("network_factory_kwargs", {})
    if isinstance(kwargs.get("activation"), str):
        kwargs["activation"] = training_networks.ACTIVATION[kwargs["activation"]]
    for key in (
        "policy_network_kernel_init_fn",
        "value_network_kernel_init_fn",
        "mean_kernel_init_fn",
    ):
        if isinstance(kwargs.get(key), str):
            kwargs[key] = training_networks.KERNEL_INITIALIZER[kwargs[key]]

    observation_size = config["observation_size"]
    if isinstance(observation_size, dict) and "shape" in observation_size:
        shape = tuple(observation_size["shape"])
        observation_size = shape[0] if len(shape) == 1 else shape

    preprocess = (
        running_statistics.normalize
        if config.get("normalize_observations", False)
        else lambda x, y: x
    )
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size,
        config["action_size"],
        preprocess_observations_fn=preprocess,
        **kwargs,
    )
    params = ppo_checkpoint.load(path)
    return ppo_networks.make_inference_fn(ppo_network)(params, deterministic=True)
