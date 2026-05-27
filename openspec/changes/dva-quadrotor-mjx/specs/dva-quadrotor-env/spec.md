## ADDED Requirements

### Requirement: Stage-Gated Execution
项目 SHALL 使用阶段门禁而不是 agent 自述来判定当前进度。当前阶段由 `tasks.md` 中最早未完成的阶段和本 spec 的验收场景共同决定。

#### Scenario: Determine Current Stage
- **GIVEN** `tasks.md` 存在 `Current Stage` 与 `M0..M6` 阶段检查项
- **WHEN** agent 汇报当前进度
- **THEN** agent SHALL 引用已完成检查项、验证命令和剩余阻塞项
- **AND** agent SHALL NOT 仅用“已分析”“已计划”“已基本完成”作为阶段完成依据

#### Scenario: Prevent Stage Skipping
- **GIVEN** 当前阶段仍有未完成检查项
- **WHEN** agent 试图推进下一阶段实现
- **THEN** agent SHALL 先说明跳阶段原因、风险和缺失验证
- **AND** 未经用户明确确认，agent SHALL NOT 跳过 state dynamics 或 state SHAC 直接进入 D.VA/vision SHAC

#### Scenario: Report Missing Code Graph Evidence
- **GIVEN** 任务涉及代码查看或依赖关系判断
- **WHEN** code-review-graph 返回空图、过期索引或不可用结果
- **THEN** agent SHALL 明确报告该限制
- **AND** agent SHALL NOT 声称已完成真实依赖关系审查

### Requirement: Independent Project Skeleton
项目 SHALL 作为独立 Python 项目实现，使用 `src/dva_quadrotor_mjx` 主包、根目录 `tests/`、可安装项目配置、环境 registry 和 MJCF assets；Playground 组件只作为设计模式与适配参考。

#### Scenario: Bootstrap Project Import
- **GIVEN** 仓库根目录存在 `pyproject.toml`
- **WHEN** 在项目虚拟环境中运行 pytest
- **THEN** `dva_quadrotor_mjx` 包 SHALL 可被导入
- **AND** 环境 registry SHALL 能返回最小 quadrotor hover 环境元数据

#### Scenario: Load Minimal MJCF
- **GIVEN** registry 返回 quadrotor hover MJCF 路径
- **WHEN** 使用 `mujoco.MjModel.from_xml_path(...)` 加载该 XML
- **THEN** 模型 SHALL 至少包含一个 freejoint、四个电机 actuator、0.75kg 机体质量和基础相机定义

### Requirement: Quadrotor Dynamics Simulation
环境模型 SHALL 准确重现 `rpg_flightning` 的关键物理语义，包括质量、转动惯量、电机排布、电机响应延迟、机身空气阻力、低级控制频率和 policy 控制频率。动力学主链路 SHALL 优先保持在 JAX/MJX 中可微。

#### Scenario: Verify Physical Alignment
- **GIVEN** 环境初始化为静止悬停状态
- **WHEN** 对四轴施加相等的 hover 推力
- **THEN** 无人机的短时垂直加速度 SHALL 接近 0
- **AND** 测试 SHALL 明确记录质量、重力、电机推力映射和控制步长

#### Scenario: Verify Force Injection Before Claiming Drag Parity
- **GIVEN** 实现机身 quadratic drag
- **WHEN** drag 通过 `qfrc_applied`、`xfrc_applied` 或外层 JAX 状态更新施加到 MJX data
- **THEN** 测试 SHALL 验证该施力通道影响 `mjx.step`
- **AND** 未通过该测试前，agent SHALL NOT 声称已对齐 `rpg_flightning` drag

### Requirement: Dual Frequency Loop Integration
物理模拟频率 SHALL 为 200Hz (`sim_dt=0.005s`)，策略控制频率 SHALL 为 50Hz (`ctrl_dt=0.02s`)。环境每次 policy step SHALL 执行 4 个物理 substep，并在每个 substep 中更新低级控制器或 motor state。

#### Scenario: Run Simulation Steps
- **GIVEN** 上层策略输出一个低级控制指令
- **WHEN** 环境执行一次 `step`
- **THEN** 系统 SHALL 执行 4 次 MJX 物理积分
- **AND** 每个物理 substep SHALL 使用当前高频状态更新低级控制输入

### Requirement: Perceptual Modality Separation
环境观测 SHALL 支持根据配置独立或联合开启 `state`、`vision`、`depth` 和 `lidar`。视觉与 LiDAR 观测 SHALL 有稳定的 shape、dtype、frame stack 约定和 enable/disable 行为。

#### Scenario: Visual Observation
- **GIVEN** `vision_enabled=True`
- **WHEN** 环境 reset 或 step
- **THEN** 观测 SHALL 包含像素键，例如 `pixels/view_0`
- **AND** 图像来源 MAY 使用 MJX/MJWarp render pipeline

#### Scenario: LiDAR Observation
- **GIVEN** `lidar_enabled=True`
- **WHEN** 环境 reset 或 step
- **THEN** 观测 SHALL 包含 rangefinder 测距数据
- **AND** 测距数据 SHALL 来自 MuJoCo sensor data 或经过验证的 JAX raycast 实现

### Requirement: Rendering Gradient Boundary
MJX/MJWarp RGB/depth/rangefinder SHALL 在一阶段作为 observation 生成链路，不作为可微渲染链路。训练算法不得依赖 MJX/MJWarp rendering 对几何、相机或物理状态提供有效自动微分梯度。

#### Scenario: Backpropagation Through Perception
- **GIVEN** 训练执行 BPTT 或 SHAC/D.VA actor update
- **WHEN** loss 对轨迹进行反向传播
- **THEN** 物理 state/action 主链路 MAY 参与梯度传播
- **AND** RGB/depth/rangefinder observation SHALL 默认在输入端 `stop_gradient`
- **AND** 若后续要使用 `jaxrenderer` 或 DiffPhysDrone-style 可微渲染，必须创建单独阶段任务和测试，不得混入 M1-M5 主线验收

### Requirement: State SHAC Before Vision SHAC
系统 SHALL 先完成 state-based SHAC baseline，再进入 D.VA 或 vision SHAC。state SHAC 是动力学环境、reward、termination 和训练接口正确性的基础门禁。

#### Scenario: Block Premature Vision Baselines
- **GIVEN** state SHAC 尚未在 hover state env 中收敛
- **WHEN** agent 准备实现 pixel SHAC、D.VA 或 DiffPhysDrone-style baseline
- **THEN** agent SHALL 报告该跳跃违反阶段门禁
- **AND** 除非用户明确覆盖，agent SHALL 返回 M3 继续完成 state baseline

### Requirement: D.VA Architecture
D.VA 阶段 SHALL 复现算法结构而不是直接迁移 PyTorch/DFlex 代码：actor 使用 vision/lidar observation，critic 使用 privileged state observation。

#### Scenario: Actor-Critic Observation Split
- **GIVEN** D.VA 训练配置启用视觉或 LiDAR
- **WHEN** actor 和 critic 前向传播
- **THEN** actor SHALL 接收高维感知观测
- **AND** critic SHALL 接收 privileged state observation
- **AND** 两者 SHALL 共享同一 MJX 动力学环境

### Requirement: Three Obstacle Avoidance Env Tasks
环境模型 SHALL 在完成 state hover baseline 后，提供悬停避障、穿环和森林穿越三种任务场景。

#### Scenario: Forest Navigation Evaluation
- **GIVEN** 无人机运行在森林避障环境中
- **WHEN** 无人机撞击树木或越界
- **THEN** 环境 SHALL 结束 episode 或给予明确负值惩罚
