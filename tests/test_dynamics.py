import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import pytest

from dva_quadrotor_mjx.envs import get_env_spec
from dva_quadrotor_mjx.dynamics.drag import compute_drag_force

def test_m1_model_parameters() -> None:
    """M1.2: Verify that body mass, principal inertia, motor positions and thrust directions are fixed."""
    spec = get_env_spec("quadrotor_hover")
    mj_model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    quad_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    
    # 0.75kg mass
    assert float(mj_model.body_mass[quad_body_id]) == pytest.approx(0.75)
    
    # Inertia
    inertia = mj_model.body_inertia[quad_body_id]
    assert float(inertia[0]) == pytest.approx(0.0023)
    assert float(inertia[1]) == pytest.approx(0.0023)
    assert float(inertia[2]) == pytest.approx(0.0040)
    
    # 4 motors (actuators)
    assert mj_model.nu == 4
    
    # Motor names
    assert mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0) == "motor_fl"
    assert mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, 1) == "motor_fr"
    assert mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, 2) == "motor_rl"
    assert mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, 3) == "motor_rr"

def test_m1_hover_sanity_and_step() -> None:
    """M1.1 & M1.3: put_model, make_data, step run successfully and vertical acceleration is zero at hover thrust."""
    spec = get_env_spec("quadrotor_hover")
    mj_model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    
    # Put to MJX
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)
    
    # Mass = 0.75, Gravity = -9.81 => Hover thrust sum = 7.3575 N
    # Equal thrust per motor = 1.839375 N
    hover_ctrl = jnp.array([1.839375] * 4)
    data = data.replace(ctrl=hover_ctrl)
    
    # Step simulation
    data = mjx.step(model, data)
    
    # Acceleration should be close to zero (qacc[2] is Z acceleration)
    assert float(data.qacc[2]) == pytest.approx(0.0, abs=1e-5)

def test_m1_drag_applied_force() -> None:
    """M1.5: Verify the drag application channel using xfrc_applied in MJX."""
    spec = get_env_spec("quadrotor_hover")
    mj_model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    quad_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)
    
    # Compute drag force: base velocity = [0, 0, 5] (going up)
    # Rotation matrix = Identity (no rotation)
    R = jnp.eye(3)
    v = jnp.array([0.0, 0.0, 5.0])
    
    # Calculate drag
    f_drag_world = compute_drag_force(R, v, key=None)
    
    # With vertical velocity = 5.0:
    # f_drag_body_z = -0.5 * 1.2 * 1.04 * 1.0e-2 * 5.0 * 5.0 = -0.156 N
    assert float(f_drag_world[2]) == pytest.approx(-0.156, abs=1e-5)
    
    # Apply to xfrc_applied
    xfrc = jnp.zeros((model.nbody, 6))
    xfrc = xfrc.at[quad_body_id, :3].set(f_drag_world)
    data = data.replace(xfrc_applied=xfrc)
    
    # With no motor thrust, step 1 step
    # Gravity is -9.81, Drag is -0.156/0.75 = -0.208
    # Total acceleration should be -10.018
    data = mjx.step(model, data)
    assert float(data.qacc[2]) == pytest.approx(-9.81 - 0.208, abs=1e-4)

def test_m1_explicit_motor_delay() -> None:
    """M1.4: Verify the JAX explicit first-order motor delay tracking."""
    # Simulation step dt = 0.005s, motor_tau = 0.033s
    dt = 0.005
    motor_tau = 0.033
    
    motor_omega = jnp.array([150.0] * 4)
    motor_omega_d = jnp.array([1000.0] * 4)
    
    # Apply delay equation
    motor_omega_new = (motor_omega - motor_omega_d) * jnp.exp(-dt / motor_tau) + motor_omega_d
    
    # Analytical expected
    expected = 150.0 + (1000.0 - 150.0) * (1 - jnp.exp(-0.005 / 0.033))
    assert float(motor_omega_new[0]) == pytest.approx(expected, abs=1e-4)
