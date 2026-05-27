# Design: DVA Quadrotor MJX Platform

## Context

目标是在独立项目中实现一个 MuJoCo MJX 四旋翼实验平台，并复用 MuJoCo Playground 的设计模式：环境 registry、MJCF assets、JAX-friendly env state、批量训练接口、vision observation pipeline。`rpg_flightning`、`jax_shac`、`D.VA`、`DiffPhysDrone` 都是参考源，不是直接运行底座。

关键约束：

- `rpg_flightning` 提供任务语义、控制接口、hover reward、BPTT 设计参考。
- `jax_shac` 提供 MJX env 和 state SHAC baseline 的最短路径。
- `D.VA` 提供 `vision/lidar actor + privileged state critic` 的算法结构。
- `DiffPhysDrone` 提供可微视觉飞行 baseline 的论文对照参考，但其实现是 PyTorch + 自定义 CUDA，不纳入 MJX 主线。
- MJX/MJWarp render 可生成 RGB/depth observation，但当前不能当作可依赖的可微渲染链路。

## Architecture

### 独立项目 + Playground 风格

项目主包为 `src/dva_quadrotor_mjx`。核心组织方式：

- `envs/`: MJX 环境、任务语义、registry。
- `envs/xmls/`: MJCF assets。
- `controllers/`: motor model、rate/attitude controller。
- `dynamics/`: drag model 和动力学辅助计算。
- `algorithms/`: SHAC、D.VA、APG 等训练入口。
- `networks/`: actor、critic、encoder。
- `wrappers/`: Brax/Playground 兼容层。
- `tests/`: 仓库根目录测试，验证阶段门禁。

不新增 `decision.md`。关键阶段决策写入 `spec.md`，执行状态写入 `tasks.md`。

### 阶段门禁

当前阶段由 `tasks.md` 中最早未完成阶段决定，不由 agent 自述决定。每个阶段必须有：

- 明确的可运行验证命令。
- 明确的通过/失败标准。
- 明确的不能跳过的依赖。

推荐阶段：

1. `M0`: 提案收敛和最小骨架。
2. `M1`: MJX quadrotor dynamics bootstrap。
3. `M2`: `rpg_flightning` hover state env 语义。
4. `M3`: state SHAC baseline。
5. `M4`: RGB/depth/rangefinder observation。
6. `M5`: JAX D.VA。
7. `M6`: 可选 differentiable vision baseline。

### 动力学路线

第一阶段动力学目标不是完整无人机论文环境，而是可验证的 MJX quadrotor core：

- MJCF 中固定机体质量、惯量、freejoint、四个 motor sites/actuators。
- 用测试验证模型参数、actuator 数量、MJX step 可运行。
- hover force sanity check 作为第一个物理验收。
- motor delay 和 drag 都必须先有测试再宣称对齐。

drag 方案保持开放：

- 优先验证 `qfrc_applied` / `xfrc_applied` 在 MJX 中是否可用且影响 `mjx.step`。
- 如果该路径不稳定，改用 JAX 显式状态更新或 actuator force mapping。
- 未通过测试前，不写“100% 对齐 rpg_flightning drag”。

### 感知路线

感知分两层：

- 主线 observation：MJX/MJWarp render 生成 RGB/depth，MuJoCo rangefinder/sensordata 生成 LiDAR。
- 可选研究分支：`jaxrenderer` 或 DiffPhysDrone-style custom renderer 生成可微视觉 baseline。

主线要求：

- RGB/depth/rangefinder 有稳定 shape、dtype、frame stack、enable/disable 配置。
- 在 actor 输入端默认 `jax.lax.stop_gradient`。
- 不把 MJX/MJWarp render 梯度作为训练正确性的依赖。

依据：

- MuJoCo Playground 使用 MJWarp Batch Renderer 作为 vision support。
- Playground vision env 调用 `mjx.create_render_context`、`mjx.refit_bvh`、`mjx.render`、`mjx.get_rgb`。
- MuJoCo/MJWarp 文档说明 MJWarp 当前不支持自动微分。
- `jaxrenderer` 是 JAX differentiable renderer，但需要单独做 MJCF-to-scene 转换，不适合压进 M0-M5 主线。

### 算法路线

算法顺序必须保守：

1. `state SHAC`: 使用 privileged state observation，验证动力学、reward、termination、训练接口。
2. `D.VA`: actor 看 vision/lidar，critic 看 privileged state，复现结构而不是迁移 PyTorch/DFlex 代码。
3. `vision SHAC` 或 DiffPhysDrone-style differentiable vision：只作为可选论文补充。

state SHAC 未收敛前，不推进 D.VA 或 vision SHAC。

## Risks / Trade-offs

- **渲染不可导风险**：MJX render 可 JIT/vmap 生成 observation，但 MJWarp 自动微分不可用。解决：主线默认 stop_gradient，另开可微视觉分支。
- **范围膨胀风险**：同时迁移 `rpg_flightning`、复现 `DiffPhysDrone`、实现 D.VA 会失控。解决：阶段门禁。
- **过早算法开发风险**：环境动力学未验证时训练结果不可解释。解决：M1/M2/M3 必须先过。
- **代码图不可用风险**：当前 code-review-graph 对本仓库返回 0 nodes/0 edges。解决：agent 不得声称已完成依赖关系审查，直到索引建立并通过检查。

## Verification

M0 验证命令：

```bash
openspec validate dva-quadrotor-mjx --strict
.venv/bin/python -m pytest tests -q
```

M1 以后每个阶段必须在 `tasks.md` 增加或保持对应验证命令。没有验证命令的阶段不得标记完成。
