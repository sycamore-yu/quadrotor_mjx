## Why

当前 `dva_quadrotor_mjx` 已有最小 MJX hover 环境和 state SHAC 演示，但距离“完整实现 `rpg_flightning` under MuJoCo MJX”仍有明显缺口：

- `rpg_flightning` 的 state hover、feature-vision hover、BPTT、PPO、save/load 示例尚未被完整复现。
- 当前仓库只有 `quadrotor_hover` 一个任务，`obstacle_avoidance.py`、`ring_navigation.py`、`forest_navigation.py` 仍为空。
- 当前训练入口只有 state SHAC demo；缺少三场景 x 三算法的标准训练脚本。
- 当前视觉/rangefinder 测试更多是 shape smoke test，尚未验证真实场景语义、命中距离、训练可用性和回放可视化。

本变更定义后续工作：在 MuJoCo MJX 下完成 `rpg_flightning` 等价平台，并扩展为论文实验平台需要的三个场景、三个算法、可视化和可复现实验验收。

## What Changes

1. **补齐 rpg_flightning 等价核心**
   - 对齐 `HoveringStateEnv`、`HoveringFeaturesEnv`、`EnvBase`、wrapper、rollout、policy save/load。
   - 对齐 quadrotor 动力学参数、低级控制、delay、drag、action history、reward、termination。

2. **完成三个 MuJoCo MJX 场景**
   - `hover_obstacle`: 悬停避障。
   - `gate_crossing`: 穿环/门洞导航。
   - `forest_navigation`: 森林柱状障碍穿越。

3. **完成三个算法训练路径**
   - `bptt`: 对齐 `rpg_flightning` 的 differentiable rollout / BPTT。
   - `ppo`: 对齐 `rpg_flightning` 的 state PPO baseline。
   - `shac`: 基于 `jax_shac` 的 MJX state SHAC baseline。

4. **建立标准脚本矩阵**
   - 每个场景和算法组合都必须有 CLI 入口、配置、最小 smoke run、正式训练 run、checkpoint、metrics 和可视化/回放路径。

5. **建立验收门禁**
   - 不再用“脚本能启动”作为完成标准。
   - 每个阶段必须明确通过测试、无 monkey patch 依赖、无 fallback 假通过、训练指标可复现。

## Non-Goals

- 本阶段不直接迁移 `DiffPhysDrone` CUDA op。
- 本阶段不要求 MJX/MJWarp RGB/depth 渲染可微。
- 本阶段不要求 D.VA 完整训练；D.VA 属于后续 `dva-quadrotor-mjx` 主线。
- 本阶段不以 Notebook 作为唯一入口；Notebook 可以存在，但 CLI 必须可运行。

## Evidence And References

- `rpg_flightning` references:
  - `third_party/rpg_flightning/flightning/envs/env_base.py`
  - `third_party/rpg_flightning/flightning/envs/hovering_state_env.py`
  - `third_party/rpg_flightning/flightning/envs/hovering_features_env.py`
  - `third_party/rpg_flightning/flightning/envs/wrappers.py`
  - `third_party/rpg_flightning/flightning/objects/quadrotor_obj.py`
  - `third_party/rpg_flightning/flightning/objects/world_box_obj.py`
  - `third_party/rpg_flightning/flightning/sensors/double_sphere_camera.py`
  - `third_party/rpg_flightning/flightning/algos/bptt.py`
  - `third_party/rpg_flightning/flightning/algos/ppo.py`
  - `third_party/rpg_flightning/examples/train_bptt_state.ipynb`
  - `third_party/rpg_flightning/examples/train_bptt_vision.ipynb`
  - `third_party/rpg_flightning/examples/train_ppo_state.ipynb`
  - `third_party/rpg_flightning/examples/save_load_policy.ipynb`
- MJX / Playground references:
  - `third_party/jax_shac/envs/mjx_envs.py`
  - `third_party/jax_shac/envs/register_framed_hopper.py`
  - `third_party/jax_shac/shac/train.py`
  - `third_party/mujoco_playground/learning/README.md`
  - `third_party/mujoco_playground/learning/train_jax_ppo.py`
  - `third_party/mujoco_playground/mujoco_playground/_src/dm_control_suite/cartpole.py`
  - `third_party/mujoco/doc/overview.rst`
  - `third_party/mujoco/doc/mjx.rst`
  - `third_party/mujoco/doc/mjwarp/index.rst`

Context7 note: `npx ctx7@latest library MuJoCo ...` failed with `fetch failed` in this session, and sandbox escalation was unavailable. The references above are local vendored docs/source checked into this workspace.
