from dva_quadrotor_mjx.envs import load
from dva_quadrotor_mjx.algorithms.shac.train import SHAC


def test_shac_fast_adapter_disables_debug_instrumentation() -> None:
    """Default SHAC path keeps jacobian and checkify helpers off the hot loop."""
    env = load("quadrotor_hover", max_steps_in_episode=20)

    shac = SHAC(
        environment=env,
        num_timesteps=2,
        episode_length=2,
        num_envs=1,
        num_eval_envs=1,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        seed=42,
        unroll_length=1,
        critic_batch_size=1,
        critic_epochs=1,
        num_evals=2,
    )

    assert shac.debug_mode is False
    assert shac.enable_runtime_checks is False
    assert shac.enable_jacobian_debug is False
    assert shac.jac_env_step is None
    assert shac.jac_rollout is None
    assert shac._compiled_training_epoch is None


def test_shac_debug_adapter_retains_runtime_check_tools() -> None:
    """Debug SHAC path keeps checkify and jacobian helpers available."""
    env = load("quadrotor_hover", max_steps_in_episode=20)

    shac = SHAC(
        environment=env,
        num_timesteps=2,
        episode_length=2,
        num_envs=1,
        num_eval_envs=1,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        seed=7,
        unroll_length=1,
        critic_batch_size=1,
        critic_epochs=1,
        num_evals=2,
        debug_mode=True,
    )

    assert shac.debug_mode is True
    assert shac.enable_runtime_checks is True
    assert shac.enable_jacobian_debug is True
    assert callable(shac.jac_env_step)
    assert callable(shac.jac_rollout)
    assert callable(shac._checked_training_step)
