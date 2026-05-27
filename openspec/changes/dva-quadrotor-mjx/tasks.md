# Current Stage

Active stage: `M5 - JAX D.VA`.

Stage advancement rule: only move to the next stage when every task in the current stage is checked and the verification command in that stage has been run without skipped required checks.

## M0. Proposal Hardening + Skeleton Bootstrap

- [x] M0.1 收紧 `proposal.md`、`design.md`、`spec.md`、`tasks.md`，明确阶段边界和 MJX/MJWarp 渲染不可导风险
- [x] M0.2 新增 `pyproject.toml`，让项目能以 `src/` layout 被 pytest/import 工具识别
- [x] M0.3 新增根目录 `tests/`，把后续测试入口从包内测试迁到仓库级测试
- [x] M0.4 新增环境 registry，提供可查询的最小环境元数据与 MJCF 路径
- [x] M0.5 新增最小 quadrotor hover MJCF skeleton，并验证 `mujoco.MjModel.from_xml_path(...)` 可加载
- [x] M0.6 运行 `openspec validate dva-quadrotor-mjx --strict`
- [x] M0.7 运行 `.venv/bin/python -m pytest tests -q`

## M1. MJX Quadrotor Dynamics Bootstrap

- [x] M1.1 将最小 MJCF 放入 MJX：`mjx.put_model`、`mjx.make_data`、`mjx.step` 均可运行
- [x] M1.2 固定 0.75kg 质量、主轴惯量、电机位置和推力方向，并用测试锁定模型参数
- [x] M1.3 实现 hover action sanity check：四电机等推力时竖直加速度接近 0
- [x] M1.4 验证一阶电机延迟方案；若 MuJoCo actuator filter 与目标控制语义不一致，退回 JAX 显式 motor state
- [x] M1.5 验证 drag 施力通道；不得在未测试 `qfrc_applied/xfrc_applied` 行为前声称已对齐 `rpg_flightning`

## M2. rpg_flightning Hover State Env Semantics

- [x] M2.1 复现 `HoveringStateEnv` 的 reset/step/reward/done 语义
- [x] M2.2 保持 state observation、action scaling、episode length、auto-reset 行为可测试
- [x] M2.3 将低级 200Hz 控制与 50Hz policy step 的双频机制写成显式测试
- [x] M2.4 只迁移任务语义和接口，不直接复用原源码作为运行底座

## M3. State SHAC Baseline

- [x] M3.1 以 `third_party/jax_shac` 的 `MjxEnv` 模式接入 hover state env
- [x] M3.2 跑通 state SHAC 训练脚本
- [x] M3.3 记录收敛指标、seed、配置文件和 checkpoint 路径
- [x] M3.4 state SHAC 未收敛前，不进入 pixel SHAC 或 D.VA 主线

## M4. Perception Observations

- [x] M4.1 按 Playground 模式接入 `mjx.create_render_context`、`mjx.refit_bvh`、`mjx.render`、`mjx.get_rgb/get_depth`
- [x] M4.2 接入 MuJoCo `rangefinder`/`sensordata`，优先做 site fan，其次 camera-based rangefinder
- [x] M4.3 对 RGB/depth/rangefinder 写 shape、dtype、frame stack 和 enable/disable 测试
- [x] M4.4 明确这些感知观测在本阶段是 observation，不是可微物理渲染链路

## M5. JAX D.VA

- [ ] M5.1 实现 actor 输入为 vision/lidar、critic 输入为 privileged state 的网络分解
- [ ] M5.2 在 actor 输入端对高维感知观测执行 `jax.lax.stop_gradient`
- [ ] M5.3 实现 D.VA rollout buffer、critic bootstrap 和 actor update 的 JAX 版本
- [ ] M5.4 与 state SHAC baseline 使用同一 MJX 动力学环境对比

## M6. Optional Differentiable Vision Baselines

- [ ] M6.1 评估 `third_party/jaxrenderer` 是否能承接简化几何场景的可微 RGB/depth baseline
- [ ] M6.2 评估 DiffPhysDrone-style 自定义深度/碰撞渲染是否值得作为论文额外 baseline
- [ ] M6.3 仅当 M3/M5 已形成可复现实验结果后，再投入 vision SHAC
