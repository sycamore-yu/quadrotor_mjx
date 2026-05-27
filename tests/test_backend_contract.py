"""Backend completion contract tests for training dispatch."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

import scripts.eval as eval_script
import scripts.render as render_script
import scripts.visualize_mjviser as play_script
from dva_quadrotor_mjx.algorithms import trainer
from dva_quadrotor_mjx.learning import shac_checkpoint
from dva_quadrotor_mjx.policies import create_train_state, make_network


class _DummySpace:
    shape = (4,)


@dataclass
class _PipelineState:
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass
class _DummyState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, bool]
    pipeline_state: _PipelineState


class _DummyEnv:
    action_space = _DummySpace()
    observation_size = (23,)
    max_steps_in_episode = 2
    model = SimpleNamespace(nq=2, nv=2)

    def reset(self, _key):
        return self._make_state()

    def step(self, _state, _action):
        return self._make_state()

    @staticmethod
    def _make_state() -> _DummyState:
        return _DummyState(
            obs=jnp.zeros((23,)),
            reward=jnp.asarray(1.0),
            done=jnp.asarray(False),
            info={"success": False, "collision": False},
            pipeline_state=_PipelineState(
                qpos=np.zeros(2, dtype=np.float32),
                qvel=np.zeros(2, dtype=np.float32),
            ),
        )


def _make_train_state(*, seed: int = 0, step: int = 7) -> TrainState:
    network = make_network("bptt", _DummySpace.shape[0])
    train_state = create_train_state(
        network,
        _DummyEnv.observation_size,
        seed=seed,
        learning_rate=1e-3,
    )
    return train_state.replace(step=step)


def _assert_params_match(actual: TrainState, expected: TrainState) -> None:
    assert actual.step == expected.step
    matches = jax.tree_util.tree_map(
        lambda left, right: jnp.allclose(left, right),
        actual.params,
        expected.params,
    )
    assert all(bool(leaf) for leaf in jax.tree_util.tree_leaves(matches))


def _patch_dummy_env(monkeypatch, module):
    env = _DummyEnv()
    monkeypatch.setattr(module, "load", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(module, "NormalizeActionWrapper", lambda wrapped: wrapped)
    return env


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


def test_bptt_checkpoint_is_loaded_by_eval(monkeypatch, tmp_path):
    """Eval must reload a TrainState checkpoint for the non-PPO path."""

    expected_state = _make_train_state()
    checkpoint_path = tmp_path / "bptt.ckpt"
    trainer.save_checkpoint(checkpoint_path, expected_state)
    env = _patch_dummy_env(monkeypatch, eval_script)

    seen = {}

    def fake_eval_policy(checkpoint, env_arg, episodes, seed, max_steps):
        seen["checkpoint"] = checkpoint
        assert env_arg is env
        _assert_params_match(checkpoint, expected_state)
        return trainer.EvalResult(
            mean_reward=1.0,
            mean_episode_length=2.0,
            success_rate=0.0,
            collision_rate=0.0,
        )

    monkeypatch.setattr(eval_script, "eval_policy", fake_eval_policy)

    exit_code = eval_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "bptt",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            "1",
            "--steps",
            "2",
        ]
    )

    assert exit_code == 0
    assert isinstance(seen["checkpoint"], TrainState)


def test_bptt_checkpoint_is_loaded_by_render(monkeypatch, tmp_path):
    """Render must reload a TrainState checkpoint for the non-PPO path."""

    expected_state = _make_train_state()
    checkpoint_path = tmp_path / "bptt.ckpt"
    trainer.save_checkpoint(checkpoint_path, expected_state)
    output_path = tmp_path / "render.mp4"
    _patch_dummy_env(monkeypatch, render_script)

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    class _FakeRenderer:
        def __init__(self, model, height, width):
            self.model = model
            self.height = height
            self.width = width

        def update_scene(self, data):
            self.data = data

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_select_action(train_state, obs):
        _assert_params_match(train_state, expected_state)
        assert obs.shape == (23,)
        return jnp.zeros((4,), dtype=jnp.float32)

    def fake_mimsave(path, frames, fps):
        Path(path).write_bytes(b"fake-mp4")
        assert frames
        assert fps == 30

    monkeypatch.setattr(render_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(render_script.mujoco, "Renderer", _FakeRenderer)
    monkeypatch.setattr(render_script.mujoco, "mj_forward", lambda model, data: None)
    monkeypatch.setattr(render_script, "select_action", fake_select_action)

    import imageio.v2 as imageio_v2

    monkeypatch.setattr(imageio_v2, "mimsave", fake_mimsave)

    exit_code = render_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "bptt",
            "--checkpoint",
            str(checkpoint_path),
            "--steps",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_bptt_checkpoint_is_loaded_by_play_dry_run(monkeypatch, tmp_path, capsys):
    """Play must accept a TrainState checkpoint before starting mjviser."""

    expected_state = _make_train_state()
    checkpoint_path = tmp_path / "bptt.ckpt"
    trainer.save_checkpoint(checkpoint_path, expected_state)
    _patch_dummy_env(monkeypatch, play_script)

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    monkeypatch.setattr(play_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(play_script.mujoco, "mj_forward", lambda model, data: None)

    exit_code = play_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "bptt",
            "--checkpoint",
            str(checkpoint_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dry-run scene check passed" in captured.out
    assert '"mode": "checkpoint"' in captured.out
    assert '"viewer_started": false' in captured.out


def test_shac_pickle_checkpoint_uses_dedicated_loader(monkeypatch, tmp_path):
    """SHAC pickle checkpoints use shac_checkpoint, not the TrainState loader."""

    expected_state = _make_train_state()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint_path.write_bytes(pickle.dumps({"backend": "jax_shac"}))

    with pytest.raises(Exception, match="extra data"):
        trainer.load_checkpoint(checkpoint_path, expected_state)

    def fake_load_policy(checkpoint, env, deterministic=True, scalar_var=False):
        assert deterministic is True
        assert env is _DummyEnv
        assert Path(checkpoint) == checkpoint_path
        assert scalar_var is False

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    monkeypatch.setattr(
        shac_checkpoint.shac_backend,
        "load_policy",
        fake_load_policy,
    )

    policy = shac_checkpoint.load_policy(checkpoint_path, _DummyEnv)
    action, extras = policy(jnp.zeros((23,)), jax.random.PRNGKey(0))
    assert action.shape == (4,)
    assert extras == {}


def test_shac_checkpoint_directory_resolves_latest_file(tmp_path):
    """Directory checkpoints should resolve to the newest SHAC pickle."""

    (tmp_path / "checkpoint_1.pkl").write_bytes(b"first")
    (tmp_path / "checkpoint_9.pkl").write_bytes(b"second")
    (tmp_path / "checkpoint.pkl").write_bytes(b"canonical")

    assert shac_checkpoint.resolve_checkpoint_path(tmp_path) == tmp_path / "checkpoint.pkl"


def test_shac_eval_cli_uses_shac_loader(monkeypatch, tmp_path, capsys):
    """Eval must dispatch SHAC checkpoints through the SHAC loader path."""

    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint_path.write_bytes(pickle.dumps({"training_state": "placeholder"}))
    env = _patch_dummy_env(monkeypatch, eval_script)

    seen = {}

    def fake_load_policy(path, env_arg, deterministic=True):
        seen["path"] = Path(path)
        seen["env"] = env_arg
        assert deterministic is True
        assert env_arg is env

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    monkeypatch.setattr(eval_script, "load_shac_policy", fake_load_policy)

    exit_code = eval_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "shac",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            "1",
            "--steps",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["path"] == checkpoint_path
    assert seen["env"] is env
    assert "Mean reward:" in captured.out


def test_shac_render_cli_uses_shac_loader(monkeypatch, tmp_path):
    """Render must dispatch SHAC checkpoints through the SHAC loader path."""

    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint_path.write_bytes(pickle.dumps({"training_state": "placeholder"}))
    output_path = tmp_path / "render.mp4"
    env = _patch_dummy_env(monkeypatch, render_script)

    def fake_load_policy(path, env_arg, deterministic=True):
        assert Path(path) == checkpoint_path
        assert env_arg is env
        assert deterministic is True

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    class _FakeRenderer:
        def __init__(self, model, height, width):
            self.model = model
            self.height = height
            self.width = width

        def update_scene(self, data):
            self.data = data

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_mimsave(path, frames, fps):
        Path(path).write_bytes(b"fake-mp4")
        assert frames
        assert fps == 30

    monkeypatch.setattr(render_script, "load_shac_policy", fake_load_policy)
    monkeypatch.setattr(render_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(render_script.mujoco, "Renderer", _FakeRenderer)
    monkeypatch.setattr(render_script.mujoco, "mj_forward", lambda model, data: None)

    import imageio.v2 as imageio_v2

    monkeypatch.setattr(imageio_v2, "mimsave", fake_mimsave)

    exit_code = render_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "shac",
            "--checkpoint",
            str(checkpoint_path),
            "--steps",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_shac_play_dry_run_uses_shac_loader(monkeypatch, tmp_path, capsys):
    """Play dry-run must dispatch SHAC checkpoints through the SHAC loader path."""

    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint_path.write_bytes(pickle.dumps({"training_state": "placeholder"}))
    env = _patch_dummy_env(monkeypatch, play_script)

    def fake_load_policy(path, env_arg, deterministic=True):
        assert Path(path) == checkpoint_path
        assert env_arg is env
        assert deterministic is True

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    monkeypatch.setattr(play_script, "load_shac_policy", fake_load_policy)
    monkeypatch.setattr(play_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(play_script.mujoco, "mj_forward", lambda model, data: None)

    exit_code = play_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "shac",
            "--checkpoint",
            str(checkpoint_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dry-run scene check passed" in captured.out
    assert '"mode": "checkpoint"' in captured.out


def test_render_fails_loudly_when_mp4_writer_is_unavailable(monkeypatch, tmp_path):
    """Render must not silently downgrade MP4 acceptance to NPZ frames."""

    expected_state = _make_train_state()
    checkpoint_path = tmp_path / "bptt.ckpt"
    trainer.save_checkpoint(checkpoint_path, expected_state)
    output_path = tmp_path / "render.mp4"
    _patch_dummy_env(monkeypatch, render_script)

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    class _FakeRenderer:
        def __init__(self, model, height, width):
            pass

        def update_scene(self, data):
            pass

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(render_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(render_script.mujoco, "Renderer", _FakeRenderer)
    monkeypatch.setattr(render_script.mujoco, "mj_forward", lambda model, data: None)

    import imageio.v2 as imageio_v2

    def fail_mimsave(*_args, **_kwargs):
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(imageio_v2, "mimsave", fail_mimsave)

    with pytest.raises(RuntimeError, match="Frame-dump fallback is intentionally disabled"):
        render_script.main(
            [
                "--env",
                "dummy",
                "--algo",
                "bptt",
                "--checkpoint",
                str(checkpoint_path),
                "--steps",
                "1",
                "--output",
                str(output_path),
            ]
        )
    assert not output_path.with_suffix(".npz").exists()


def test_ppo_eval_cli_uses_ppo_loader(monkeypatch, tmp_path, capsys):
    """Eval must dispatch PPO checkpoints through the PPO loader path."""

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "ppo_network_config.json").write_text(
        '{"observation_size": 23, "action_size": 4, "normalize_observations": false}'
    )
    env = _patch_dummy_env(monkeypatch, eval_script)

    seen = {}

    def fake_load_policy(path):
        seen["path"] = Path(path)

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    monkeypatch.setattr(eval_script, "load_ppo_policy", fake_load_policy)

    exit_code = eval_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "ppo",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            "1",
            "--steps",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["path"] == checkpoint_path
    assert "Mean reward:" in captured.out


def test_ppo_render_cli_uses_ppo_loader(monkeypatch, tmp_path):
    """Render must dispatch PPO checkpoints through the PPO loader path."""

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "ppo_network_config.json").write_text(
        '{"observation_size": 23, "action_size": 4, "normalize_observations": false}'
    )
    output_path = tmp_path / "render.mp4"
    env = _patch_dummy_env(monkeypatch, render_script)

    def fake_load_policy(path):
        assert Path(path) == checkpoint_path

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    class _FakeRenderer:
        def __init__(self, model, height, width):
            self.model = model
            self.height = height
            self.width = width

        def update_scene(self, data):
            self.data = data

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_mimsave(path, frames, fps):
        Path(path).write_bytes(b"fake-mp4")
        assert frames
        assert fps == 30

    monkeypatch.setattr(render_script, "load_ppo_policy", fake_load_policy)
    monkeypatch.setattr(render_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(render_script.mujoco, "Renderer", _FakeRenderer)
    monkeypatch.setattr(render_script.mujoco, "mj_forward", lambda model, data: None)

    import imageio.v2 as imageio_v2

    monkeypatch.setattr(imageio_v2, "mimsave", fake_mimsave)

    exit_code = render_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "ppo",
            "--checkpoint",
            str(checkpoint_path),
            "--steps",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_ppo_play_dry_run_uses_ppo_loader(monkeypatch, tmp_path, capsys):
    """Play dry-run must dispatch PPO checkpoints through the PPO loader path."""

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "ppo_network_config.json").write_text(
        '{"observation_size": 23, "action_size": 4, "normalize_observations": false}'
    )
    env = _patch_dummy_env(monkeypatch, play_script)

    def fake_load_policy(path):
        assert Path(path) == checkpoint_path

        def policy(obs, key):
            assert obs.shape == (23,)
            return jnp.zeros((4,), dtype=jnp.float32), {}

        return policy

    class _FakeMjData:
        def __init__(self, model):
            self.qpos = np.zeros(int(model.nq), dtype=np.float32)
            self.qvel = np.zeros(int(model.nv), dtype=np.float32)

    monkeypatch.setattr(play_script, "load_ppo_policy", fake_load_policy)
    monkeypatch.setattr(play_script.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(play_script.mujoco, "mj_forward", lambda model, data: None)

    exit_code = play_script.main(
        [
            "--env",
            "dummy",
            "--algo",
            "ppo",
            "--checkpoint",
            str(checkpoint_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dry-run scene check passed" in captured.out
    assert '"mode": "checkpoint"' in captured.out

