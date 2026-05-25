# Design: Complete rpg_flightning On MuJoCo MJX

## Current State

当前仓库已经具备可用但不完整的基础：

- 有 `src/dva_quadrotor_mjx/envs/hover.py`，实现了一个 `HoveringStateEnv`。
- 有 `src/dva_quadrotor_mjx/controllers/low_level.py` 和 `src/dva_quadrotor_mjx/dynamics/drag.py`。
- 有 `scripts/train.py`，可通过 `PYTHONPATH=src:third_party` 跑 state SHAC demo。
- 有 `tests/test_dynamics.py`、`tests/test_hover_env.py`、`tests/test_shac.py`、`tests/test_perception.py`。

但还不能称为完整 `rpg_flightning` MJX 迁移：

- `scripts/eval.py`、`scripts/render.py` 为空。
- `src/dva_quadrotor_mjx/algorithms/{bptt,ppo,shac,dva}.py` 基本为空或缺少正式封装。
- `src/dva_quadrotor_mjx/envs/{obstacle_avoidance,ring_navigation,forest_navigation}.py` 为空。
- `src/dva_quadrotor_mjx/envs/sensors/{camera,lidar,raycasting}.py` 为空。
- `test_perception.py` 允许 renderer fallback 为零图，不能证明真实渲染。
- `scripts/train.py` 使用 `brax.v1`、`jaxtyping`、`typeguard` monkey patch，不是干净依赖集成。

## Architecture Target

### Package Layout

```text
src/dva_quadrotor_mjx/
  algorithms/
    bptt.py
    ppo.py
    shac.py
  configs/
    tasks/
      hover_state.yaml
      hover_features.yaml
      hover_obstacle.yaml
      gate_crossing.yaml
      forest_navigation.yaml
    algorithms/
      bptt.yaml
      ppo.yaml
      shac.yaml
  controllers/
    motor_model.py
    low_level.py
    rate_controller.py
  dynamics/
    quadrotor.py
    drag.py
  envs/
    base.py
    hover.py
    hover_features.py
    obstacle_avoidance.py
    ring_navigation.py
    forest_navigation.py
    registry.py
    sensors/
      feature_camera.py
      rangefinder.py
      render_camera.py
    xmls/
      quadrotor_base.xml
      hover_state.xml
      hover_obstacle.xml
      gate_crossing.xml
      forest_navigation.xml
  wrappers/
    normalize.py
    logging.py
    vectorization.py
scripts/
  train.py
  eval.py
  render.py
  visualize_mjviser.py
```

### Environment Semantics

#### Base Env

Reference:

- `third_party/rpg_flightning/flightning/envs/env_base.py`
- `third_party/rpg_flightning/flightning/envs/wrappers.py`
- `third_party/jax_shac/envs/mjx_envs.py`

Required work:

- Define a single `State` type with `pipeline_state`, `obs`, `reward`, `done`, `metrics`, `info`.
- Add deterministic `reset`, `step`, `rollout`, `auto_reset`, `vmap` support.
- Remove algorithm-specific monkey patches from environment code.
- Add wrappers equivalent to `LogWrapper`, `VecEnv`, `MinMaxObservationWrapper`.

Acceptance:

- `pytest tests/test_env_base.py -q` passes.
- `jax.jit(env.reset)` and `jax.jit(env.step)` pass for every registered env.
- `jax.vmap(env.reset)` and `jax.vmap(env.step)` pass for every registered env.

#### Hover State

Reference:

- `third_party/rpg_flightning/flightning/envs/hovering_state_env.py`
- `third_party/rpg_flightning/flightning/objects/quadrotor_obj.py`

Required work:

- Lock observation semantics: position, attitude, velocity, angular rate or action history exactly documented.
- Lock action semantics: total thrust + angular-rate command, or expose a compatibility mode matching `rpg_flightning`.
- Match delay, low-level controller, drag, reward, termination, episode length.

Acceptance:

- Numeric parity tests compare reset/step/reward against `rpg_flightning` reference for fixed seeds and fixed actions.
- Error tolerances are explicit:
  - state observation max absolute error <= `1e-4` for shared quantities.
  - reward max absolute error <= `1e-3` for shared reward terms.
  - termination behavior exactly matches for tested boundary cases.

#### Hover Features

Reference:

- `third_party/rpg_flightning/flightning/envs/hovering_features_env.py`
- `third_party/rpg_flightning/flightning/sensors/double_sphere_camera.py`

Required work:

- Recreate `rpg_flightning` feature vision as projected 2D landmarks, not RGB.
- Keep it separate from real RGB/depth perception.
- Implement camera model, feature points, clipping/out-of-view semantics, action history concatenation.

Acceptance:

- `tests/test_hover_features_env.py` verifies landmark projection against `rpg_flightning` for fixed states.
- `scripts/train.py --env hover_features --algo bptt --smoke` runs without error.
- Feature observation shape and action history shape are fixed in spec.

### Three Scene Tasks

#### 1. hover_obstacle

Reference:

- `third_party/rpg_flightning/flightning/objects/world_box_obj.py`
- `third_party/mujoco/doc/overview.rst`
- `third_party/mujoco/python/mujoco/specs_test.py` rangefinder examples

Required work:

- Static obstacle assets in MJCF.
- Goal hover point.
- Collision or clearance reward.
- Optional rangefinder observation.

Acceptance:

- `scripts/train.py --env hover_obstacle --algo bptt --smoke` no error.
- `scripts/train.py --env hover_obstacle --algo ppo --smoke` no error.
- `scripts/train.py --env hover_obstacle --algo shac --smoke` no error.
- `scripts/render.py --env hover_obstacle --random-policy --steps 200 --output artifacts/hover_obstacle.mp4` produces non-empty video or mjviser trajectory.

#### 2. gate_crossing

Reference:

- `rpg_flightning` env/reward patterns from `hovering_state_env.py`.
- MuJoCo XML assets and procedural body placement from Playground-style envs.

Required work:

- Gate geometry in MJCF or procedural scene builder.
- Ordered waypoints.
- Success when passing gate plane within opening.
- Failure on collision or timeout.

Acceptance:

- `scripts/train.py --env gate_crossing --algo bptt --smoke` no error.
- `scripts/train.py --env gate_crossing --algo ppo --smoke` no error.
- `scripts/train.py --env gate_crossing --algo shac --smoke` no error.
- `tests/test_gate_crossing.py` verifies waypoint advancement, gate miss, gate hit, collision, timeout.

#### 3. forest_navigation

Reference:

- `DiffPhysDrone` obstacle field ideas:
  - `third_party/DiffPhysDrone/env_cuda.py`
  - `third_party/DiffPhysDrone/main_cuda.py`
- MuJoCo rangefinder docs:
  - `third_party/mujoco/doc/overview.rst`
  - `third_party/mujoco/python/mujoco/specs_test.py`

Required work:

- Deterministic procedural forest generator keyed by PRNG seed.
- Cylinder/tree obstacles.
- Start/goal sampling.
- Rangefinder and optional RGB/depth observation.

Acceptance:

- `scripts/train.py --env forest_navigation --algo bptt --smoke` no error.
- `scripts/train.py --env forest_navigation --algo ppo --smoke` no error.
- `scripts/train.py --env forest_navigation --algo shac --smoke` no error.
- `tests/test_forest_navigation.py` verifies seed-stable obstacle layout, collision, goal success, rangefinder hit distances.

### Three Algorithms

#### BPTT

Reference:

- `third_party/rpg_flightning/flightning/algos/bptt.py`
- `third_party/rpg_flightning/examples/train_bptt_state.ipynb`
- `third_party/rpg_flightning/examples/train_bptt_vision.ipynb`

Required work:

- JAX rollout through MJX dynamics.
- Deterministic policy network equivalent to `rpg_flightning` MLP.
- Stop-gradient strategy documented for sensors.
- Support state hover and feature hover.

Acceptance:

- `scripts/train.py --algo bptt --env hover_state --smoke` no error.
- `scripts/train.py --algo bptt --env hover_features --smoke` no error.
- `scripts/train.py --algo bptt --env hover_obstacle --smoke` no error.
- `scripts/train.py --algo bptt --env gate_crossing --smoke` no error.
- `scripts/train.py --algo bptt --env forest_navigation --smoke` no error.
- Full run improves evaluation reward by a configured minimum over initial policy for at least 2 fixed seeds.

#### PPO

Reference:

- `third_party/rpg_flightning/flightning/algos/ppo.py`
- `third_party/rpg_flightning/examples/train_ppo_state.ipynb`
- MuJoCo Playground `learning/train_jax_ppo.py`

Required work:

- Clean PPO implementation or clean adapter to Brax PPO.
- No notebook-only path.
- Configurable env/algo YAML.
- Checkpoint save/load.
- Use the MuJoCo Playground training shape: registry-loaded env, `wrap_for_brax_training`, Brax PPO vectorized trainer, and Brax/Orbax checkpoint directories.

Implementation note:

- The earlier CLI implementation was a smoke/baseline rollout adapter. It matched Playground naming and package entry points, but not the actual Playground training backend.
- That compromise existed to keep the 3x3 smoke matrix runnable while env semantics and package structure were still moving.
- This OpenSpec now treats that as insufficient for PPO parity: PPO must run through Brax PPO and the vectorized training wrapper chain. BPTT and SHAC keep explicit pending tasks until their real trainers replace the baseline adapter.

Acceptance:

- `scripts/train.py --algo ppo --env hover_state --smoke` no error.
- `scripts/train.py --algo ppo --env hover_obstacle --smoke` no error.
- `scripts/train.py --algo ppo --env gate_crossing --smoke` no error.
- `scripts/train.py --algo ppo --env forest_navigation --smoke` no error.
- `scripts/eval.py --checkpoint <ppo_ckpt> --episodes 4` no error.

#### SHAC

Reference:

- `third_party/jax_shac/shac/train.py`
- `third_party/jax_shac/envs/mjx_envs.py`
- `third_party/jax_shac/envs/register_framed_hopper.py`

Required work:

- Move current SHAC integration out of `scripts/train.py` monkey patch path.
- Vendor or adapt `jax_shac` dependencies cleanly.
- Provide stable checkpoint path outside `third_party`.
- Add smoke and short-run test modes.

Acceptance:

- `scripts/train.py --algo shac --env hover_state --smoke` no error without manual `PYTHONPATH=src:third_party`.
- `scripts/train.py --algo shac --env hover_obstacle --smoke` no error.
- `scripts/train.py --algo shac --env gate_crossing --smoke` no error.
- `scripts/train.py --algo shac --env forest_navigation --smoke` no error.
- `scripts/eval.py --checkpoint <shac_ckpt> --episodes 4` no error.

### Visualization And Inspection

Reference:

- `third_party/mjviser/README.md`
- `third_party/mjviser/examples/passive_viewer.py`
- `third_party/mjviser/examples/active_viewer_with_controller.py`

Required work:

- Add `scripts/visualize_mjviser.py`.
- Support static scene load:
  - `--env hover_obstacle`
  - `--env gate_crossing`
  - `--env forest_navigation`
- Support rollout replay from checkpoint or recorded trajectory.
- Support random policy/manual hover controller for debugging.

Acceptance:

- `scripts/visualize_mjviser.py --env hover_obstacle --mode scene --port 8080` launches viewer.
- `scripts/visualize_mjviser.py --env gate_crossing --mode random --steps 500 --port 8081` launches viewer and advances simulation.
- `scripts/visualize_mjviser.py --env forest_navigation --checkpoint <ckpt> --steps 500 --port 8082` launches viewer and replays policy.

## Known Risks

- MJX/MJWarp rendering is not a dependable differentiable renderer. Use it as observation generation unless a separate differentiable renderer branch is created.
- SHAC currently depends on ad hoc monkey patches. This must be fixed before claiming production-quality training.
- `rpg_flightning` feature vision is landmark projection, not RGB/LiDAR. It should be implemented as compatibility mode, not confused with final perception tasks.
- Exact numeric parity with `rpg_flightning` may be limited by RK integration vs MJX/MuJoCo solver differences. Tests must separate semantic parity from bitwise parity.
