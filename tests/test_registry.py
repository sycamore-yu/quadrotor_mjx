from pathlib import Path

import mujoco
import pytest

from dva_quadrotor_mjx.envs import get_env_spec, list_envs


def test_quadrotor_hover_is_registered() -> None:
    assert "quadrotor_hover" in list_envs()

    spec = get_env_spec("quadrotor_hover")

    assert spec.stage == "M0"
    assert spec.mjcf_path.name == "quadrotor_hover.xml"
    assert Path(spec.mjcf_path).exists()


def test_minimal_quadrotor_mjcf_loads() -> None:
    spec = get_env_spec("quadrotor_hover")

    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    quad_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")

    assert quad_body > 0
    assert model.njnt == 1
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nu == 4
    assert model.ncam == 1
    assert float(model.body_mass[quad_body]) == pytest.approx(0.75)
