## Why

为后续论文构建一个长期可复现的无人机实验平台：用 MuJoCo MJX 承接四旋翼动力学、低级控制、任务语义和批量训练，用 Playground 的环境组织与视觉观测模式作为参考，而不是直接把 `rpg_flightning`、`D.VA` 或 `DiffPhysDrone` 整包迁入。

当前阶段的核心风险是范围失控：`rpg_flightning` 的 vision 是几何特征投影，不是真实 RGB/LiDAR；`DiffPhysDrone` 是 PyTorch + 自定义 CUDA 自动微分链路；MJX/MJWarp 的渲染适合高吞吐视觉观测，但不是当前可依赖的可微渲染链路。因此需要先建立可验证的 MJX 动力学平台和 state baseline，再进入视觉/D.VA。

## What Changes

1. **独立项目骨架**
   - 新增可安装 Python 项目配置、根目录测试、环境 registry 和最小 MJCF quadrotor skeleton。
   - 保留 `src/dva_quadrotor_mjx` 作为主包，Playground 只作为设计模式参考。

2. **阶段化迁移路线**
   - `M0`: 提案收敛与最小骨架。
   - `M1`: MJX 四旋翼动力学与 hover sanity check。
   - `M2`: 复现 `rpg_flightning` 的 hover state env 语义和 API。
   - `M3`: 复现 state-based SHAC baseline。
   - `M4`: 接入 RGB/depth/rangefinder 观测，但不承诺感知渲染可导。
   - `M5`: 实现 JAX 版 D.VA，采用 vision/lidar actor + privileged state critic。
   - `M6`: 需要论文对照时，再评估 DiffPhysDrone-style differentiable vision 或 vision SHAC。

3. **渲染与梯度边界**
   - MJX/MJWarp 渲染 SHALL 用作 observation 生成与评估可视化。
   - 一阶段 SHALL NOT 把 MJX/MJWarp RGB/depth 渲染作为可微物理链路的一部分。
   - 若需要可微视觉 baseline，后续单独评估 `third_party/jaxrenderer` 或 DiffPhysDrone-style 自定义渲染分支。

4. **防偷懒执行机制**
   - 当前阶段 SHALL 由 `spec.md` 的 stage gate 和 `tasks.md` 的可验证检查项决定。
   - Agent 不得用“已分析/已计划/已写草稿”替代阶段完成证明。

## Non-Goals

- 不直接源码级迁移 `rpg_flightning`。
- 不直接 port `third_party/D.VA` 的 PyTorch/DFlex 训练器。
- 不把 `DiffPhysDrone` CUDA op 作为 MJX 主线依赖。
- 不在 state SHAC 未收敛前推进 pixel SHAC。
- 不新增单独 `decision.md`；阶段和关键决策写入现有 spec/tasks/design。

## Impact

- 新增独立项目最小运行骨架：`pyproject.toml`、根目录 `tests/`、环境 registry、最小 MJCF。
- 修改 OpenSpec 提案、设计、任务和 spec，明确阶段边界、渲染不可导风险和执行验收标准。
- 后续实现必须从 `M1` 动力学 sanity check 开始推进，不得跳到 D.VA 或视觉 baseline。
