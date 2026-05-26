"""Acceptance gate configuration and artifact contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dva_quadrotor_mjx.configs.acceptance import (
    AcceptanceConfig,
    AcceptanceThresholds,
    apply_threshold_overrides,
    expected_backend_for_algo,
    load_acceptance_config,
)
from scripts.check_acceptance import (
    build_dry_run_artifact,
    validate_acceptance_artifact,
)


def test_acceptance_config_loads_expected_backend(tmp_path):
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env: hover_state",
                "algo: bptt",
                "train_seeds: [0, 1]",
                "eval_seeds: [10, 11]",
                "thresholds:",
                "  min_reward_delta: 1.0",
                "  primary_metric_name: mean_reward",
                "  primary_metric_direction: max",
                "  primary_metric_threshold: 0.0",
                "  primary_metric_vs_random_delta: 0.5",
            ]
        )
    )

    config = load_acceptance_config(config_path)

    assert config.expected_backend == "jax_bptt"
    assert config.train_seeds == (0, 1)
    assert config.eval_seeds == (10, 11)
    assert config.thresholds.min_reward_delta == 1.0


def test_threshold_overrides_cannot_weaken_gate():
    thresholds = AcceptanceThresholds(
        min_reward_delta=1.0,
        primary_metric_name="mean_reward",
        primary_metric_direction="max",
        primary_metric_threshold=10.0,
        primary_metric_vs_random_delta=2.0,
    )

    with pytest.raises(ValueError, match="min_reward_delta"):
        apply_threshold_overrides(thresholds, min_reward_delta=0.5)
    with pytest.raises(ValueError, match="primary_metric_threshold"):
        apply_threshold_overrides(thresholds, primary_metric_threshold=9.0)
    with pytest.raises(ValueError, match="primary_metric_vs_random_delta"):
        apply_threshold_overrides(thresholds, primary_metric_vs_random_delta=1.0)


def test_dry_run_artifact_has_backend_contract(tmp_path):
    config = AcceptanceConfig.from_mapping(
        {
            "env": "gate_crossing",
            "algo": "shac",
            "train_seeds": [0],
            "eval_seeds": [10],
            "thresholds": {
                "min_reward_delta": 0.0,
                "primary_metric_name": "mean_reward",
                "primary_metric_direction": "max",
                "primary_metric_threshold": 0.0,
                "primary_metric_vs_random_delta": 0.0,
            },
        }
    )

    artifact = build_dry_run_artifact(config, output_dir=tmp_path)

    assert artifact["expected_backend"] == expected_backend_for_algo("shac")
    assert artifact["planned_runs"][0]["expected_backend"] == "jax_shac"


def test_acceptance_artifact_validation_rejects_weak_backend(tmp_path):
    config = AcceptanceConfig.from_mapping(
        {
            "env": "hover_state",
            "algo": "ppo",
            "thresholds": {
                "min_reward_delta": 0.0,
                "primary_metric_name": "mean_reward",
                "primary_metric_direction": "max",
                "primary_metric_threshold": 0.0,
                "primary_metric_vs_random_delta": 0.0,
            },
        }
    )
    valid_metrics = {
        "mean_reward": 1.0,
        "mean_episode_length": 1.0,
        "primary_metric_name": "mean_reward",
        "primary_metric_value": 1.0,
        "episodes": [],
    }
    artifact = {
        "schema_version": 1,
        "env": "hover_state",
        "algo": "ppo",
        "expected_backend": "brax_ppo",
        "backend": "smoke_rollout",
        "train_seeds": [0],
        "eval_seeds": [0],
        "thresholds": config.thresholds.to_dict(),
        "runs": [
            {
                "train_seed": 0,
                "eval_seeds": [0],
                "backend": "smoke_rollout",
                "expected_backend": "smoke_rollout",
                "checkpoint_path": "checkpoint.pkl",
                "initial_policy_metrics": valid_metrics,
                "random_policy_baseline_metrics": valid_metrics,
                "final_checkpoint_metrics": valid_metrics,
                "thresholds": config.thresholds.to_dict(),
                "pass": True,
                "failures": [],
            }
        ],
        "pass": True,
    }

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="expected_backend"):
        validate_acceptance_artifact(json.loads(artifact_path.read_text()), config)


def test_acceptance_config_loads_additional_metrics(tmp_path):
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env: hover_obstacle",
                "algo: bptt",
                "thresholds:",
                "  min_reward_delta: 0.0",
                "  primary_metric_name: target_distance",
                "  primary_metric_direction: min",
                "  primary_metric_threshold: 1.0",
                "  primary_metric_vs_random_delta: 0.0",
                "  additional_metrics:",
                "    - name: obstacle_clearance",
                "      direction: max",
                "      threshold: 0.1",
                "      vs_random_delta: 0.0",
            ]
        )
    )

    config = load_acceptance_config(config_path)

    assert len(config.thresholds.additional_metrics) == 1
    assert config.thresholds.additional_metrics[0].name == "obstacle_clearance"


def test_checked_in_acceptance_configs_support_dry_run():
    config_paths = sorted(
        Path("src/dva_quadrotor_mjx/configs/acceptance_runs").glob("*.yaml")
    )

    assert len(config_paths) == 13

    for config_path in config_paths:
        config = load_acceptance_config(config_path)
        artifact = build_dry_run_artifact(config)
        assert artifact["env"] == config.env
        assert artifact["algo"] == config.algo
        assert artifact["planned_runs"]
