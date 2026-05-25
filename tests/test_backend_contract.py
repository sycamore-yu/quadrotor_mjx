"""Backend completion contract tests for training dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from dva_quadrotor_mjx.algorithms import trainer


class _DummySpace:
    shape = (4,)


class _DummyEnv:
    action_space = _DummySpace()
    observation_size = (23,)


def test_train_dispatch_never_returns_smoke_rollout(monkeypatch):
    """BPTT/SHAC/PPO dispatch must expose concrete backend names."""

    def fake_bptt(env, **kwargs):
        return trainer.TrainResult(params={}, metrics={"backend": "jax_bptt"})

    def fake_shac(env, **kwargs):
        return trainer.TrainResult(
            params={},
            metrics={"backend": "jax_shac"},
            checkpoint_path=Path("checkpoint.pkl"),
        )

    def fake_ppo(env, **kwargs):
        return trainer.TrainResult(params={}, metrics={"backend": "brax_ppo"})

    monkeypatch.setattr(trainer, "_train_jax_bptt", fake_bptt)
    monkeypatch.setattr(trainer, "_train_jax_shac", fake_shac)
    monkeypatch.setattr(trainer, "_train_brax_ppo", fake_ppo)

    for algo, expected_backend in {
        "bptt": "jax_bptt",
        "ppo": "brax_ppo",
        "shac": "jax_shac",
    }.items():
        result = trainer.train(
            _DummyEnv(),
            algo=algo,
            network=object(),
            num_epochs=1,
            num_steps_per_epoch=1,
            num_envs=1,
        )
        assert result.metrics["backend"] == expected_backend
        assert result.metrics["backend"] != "smoke_rollout"


def test_unknown_training_backend_fails_loudly():
    """Unsupported algorithms must not silently become diagnostic rollouts."""

    with pytest.raises(ValueError, match="Unsupported algorithm backend"):
        trainer.train(
            _DummyEnv(),
            algo="apg",
            network=object(),
            num_epochs=1,
            num_steps_per_epoch=1,
            num_envs=1,
        )


def test_trainer_source_does_not_accept_smoke_backend():
    """Regression guard for the previous smoke_rollout fallback path."""

    source = Path(trainer.__file__).read_text()
    assert "smoke_rollout" not in source
