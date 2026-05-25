# `rpg_flightning` 迁移到 MuJoCo 的探索结论

## Task List

1. 梳理 `third_party/rpg_flightning` 的环境、观测、动力学和训练接口
2. 梳理 `third_party/jax_shac` 的 MJX 环境抽象和 SHAC 训练器
3. 梳理 `third_party/D.VA` 的视觉策略、状态 critic 和渲染耦合方式
4. 用 MuJoCo / MuJoCo Playground 文档确认视觉与 rangefinder 能力
5. 判断应该“迁移”还是“重实现”
6. 给出一个可执行的论文实验平台路线

## Assumptions

- 目标是做后续论文的长期实验平台，不是最短时间复刻 `rpg_flightning` 论文。
- “复现 SHAC baseline” 默认至少包含 `state-based SHAC`。
- 如果你说的 baseline 指的是 `pixel-based SHAC`，那是单独的高风险项，不应和 state baseline 混在一起推进。
- 优先级应是：统一动力学平台 > 统一任务定义 > 统一观测接口 > 最大化复用旧代码。

## 结论

不建议做源码级“直接迁移”。

建议做的是：

1. 用 MuJoCo 重新实现 `rpg_flightning` 的任务语义和控制接口
2. 用 `jax_shac` 作为 `state SHAC baseline` 的第一落点
3. 用 MuJoCo / MuJoCo Playground 的传感器和视觉接口补齐 RGB / LiDAR 观测
4. 参考 `D.VA` 的算法结构，在 JAX / MJX 栈里重做一个 `vision-policy + state-critic` 版本

一句话判断：

`rpg_flightning` 应该作为“任务与梯度设计参考”，不是“代码底座”；真正的平台底座应该是 `MuJoCo model + MJX/Warp env + JAX trainer`。

## 关键依据

### 1. `rpg_flightning` 的 “vision” 不是原始图像，也不是 LiDAR

`HoveringFeaturesEnv` 的观测不是 RGB，而是把 7 个固定特征点投影到相机平面后得到的 2D 坐标，再拼接 action history：

- `third_party/rpg_flightning/flightning/envs/hovering_features_env.py:54-61`
- `third_party/rpg_flightning/flightning/envs/hovering_features_env.py:118-165`

这意味着如果你的论文目标是“基于视觉 / 激光雷达输入的无人机飞行”，那么 `rpg_flightning` 的现有观测接口不能直接当成最终平台接口，它更接近“几何特征输入”。

### 2. `rpg_flightning` 的核心贡献之一是“前向全模型，反向简化模型”

`Quadrotor.step` 用了 `jax.custom_jvp`：

- 前向走全动力学和低层控制
- 反向用 `simple_dynamics` 做梯度传播

代码位置：

- `third_party/rpg_flightning/flightning/objects/quadrotor_obj.py:239-308`

这和 README 里“simple surrogate model for gradient computation”的描述一致。也就是说：

- 如果你把环境换成 MuJoCo/MJX 的原生可微动力学，梯度路径就已经变了
- 所以“直接迁移源码”并不能严格复现 `rpg_flightning` 的训练机制

这也是我不建议做 1:1 port 的最重要原因。

### 3. `rpg_flightning` 的环境语义本身很值得保留

它的环境 API 很干净：

- `Env.step` 负责自动 reset
- `rollout` 用 `lax.scan`
- 整个接口天然适合 JAX 批量训练

代码位置：

- `third_party/rpg_flightning/flightning/envs/env_base.py:31-56`
- `third_party/rpg_flightning/flightning/envs/env_base.py:98-141`

而且 `HoveringStateEnv` 已经定义了你后续需要继承的任务语义：

- 随机初始姿态
- action delay
- hover reward
- collision / truncation

代码位置：

- `third_party/rpg_flightning/flightning/envs/hovering_state_env.py:40-90`
- `third_party/rpg_flightning/flightning/envs/hovering_state_env.py:92-134`
- `third_party/rpg_flightning/flightning/envs/hovering_state_env.py:146-237`

所以该保留的是：

- 任务定义
- reward 结构
- control semantics
- delay 机制

而不是原始实现细节。

### 4. `D.VA` 本质上就是 `vision policy + state critic`

`D.VA` 的 actor 在视觉模式下会启用 CNN encoder：

- `third_party/D.VA/models/actor.py:15-20`
- `third_party/D.VA/models/actor.py:64-69`
- `third_party/D.VA/models/actor.py:103-118`

但 critic 仍然只吃 state observation。`DVA.__init__` 创建 critic 时输入维度始终是 `num_state_obs`，而不是图像：

- `third_party/D.VA/algorithms/dva.py:111-118`

`compute_actor_loss` 里也能看出来：

- actor 用 `vis_obs.detach().clone()`
- critic bootstrap 用 `target_critic(state_obs)`

代码位置：

- `third_party/D.VA/algorithms/dva.py:185-229`
- `third_party/D.VA/algorithms/dva.py:241-257`

这非常关键，因为它说明 D.VA 不是“end-to-end image critic”，而是：

- 策略看图像
- value 函数看 privileged state

这条结构在 MuJoCo/JAX 里是非常值得保留的，因为它比“纯像素 SHAC”稳定得多。

### 5. `D.VA` 环境接口和你要做的平台接口其实已经很接近

`DFlexEnv` / `DFlexDiffRenderEnv` 里环境同时维护：

- `state_obs_buf`
- `vis_obs_buf`
- `reset_buf`
- `progress_buf`

代码位置：

- `third_party/D.VA/envs/dflex_env.py:18-66`
- `third_party/D.VA/envs/hopper_diff_render_env.py:143-196`
- `third_party/D.VA/envs/hopper_diff_render_env.py:275-329`

这给了一个很明确的平台接口目标：

- 同一个 MuJoCo quadrotor 环境，应该同时支持
  - privileged state obs
  - RGB obs
  - LiDAR obs
- 算法层按需要消费不同 observation head

### 6. `jax_shac` 已经提供了最适合你的 MuJoCo/MJX baseline 骨架

`MjxEnv` 很薄，只做三件事：

- 持有 `mujoco.MjModel`
- `pipeline_init(qpos, qvel)`
- `pipeline_step(data, ctrl)`

代码位置：

- `third_party/jax_shac/envs/mjx_envs.py:33-74`

自定义环境也很直接，`FramedHopper` 就是标准例子：

- 从 MJCF 载入 `MjModel`
- 写 `reset`
- 写 `step`
- 写 `_get_obs`

代码位置：

- `third_party/jax_shac/envs/register_framed_hopper.py:9-39`
- `third_party/jax_shac/envs/register_framed_hopper.py:41-121`

这对无人机环境非常合适，因为你完全可以照这个模板做一个 `QuadrotorHoverMJX`。

### 7. `jax_shac` 甚至已经有 D.VA 需要的网络分解

`make_shac_networks` 已经支持三种模式：

- `None`: actor/critic 都看 state
- `vision`: actor/critic 都看 vision
- `vPsC`: vision-based policy and state-based critic

代码位置：

- `third_party/jax_shac/shac/networks.py:101-157`

尤其是：

- `third_party/jax_shac/shac/networks.py:121-126`

这几乎就是 D.VA 的结构。

所以从研究工程角度看，最合理的路线不是“把 `third_party/D.VA` 整包搬过来”，而是：

- 用 `jax_shac` 的 trainer 和网络分解
- 在 MuJoCo 环境上重新实现 D.VA 的训练逻辑

### 8. MuJoCo 本身已经能支持 LiDAR / rangefinder / RGB / depth

MuJoCo 官方文档明确说：

- 传感器数据在 `mjData.sensordata`
- 支持 `rangefinder`
- 支持 user sensor
- 颜色/深度相机可以通过程序化 off-screen rendering 获得

文档位置：

- `third_party/mujoco/doc/overview.rst:640-648`

而且在 Python 绑定测试里，已经直接演示了：

- site-based rangefinder
- camera-based rangefinder

代码位置：

- `third_party/mujoco/python/mujoco/specs_test.py:1881-1979`

尤其是 camera-based rangefinder：

- `third_party/mujoco/python/mujoco/specs_test.py:1954-1979`

这意味着 LiDAR 路线不是“自己手写 ray cast”，而是可以优先考虑：

1. site fan rangefinder
2. camera-based rangefinder sensor

### 9. MuJoCo Playground 已经把 vision 路线跑通了

Playground README 直接说明：

- 基于 MuJoCo MJX
- 视觉支持通过 MJWarp Batch Renderer

文档位置：

- `third_party/mujoco_playground/README.md:7-19`

学习脚本也已经有 vision 入口：

- `third_party/mujoco_playground/learning/README.md:21-26`

更重要的是，`CartpoleBalance` 的 vision env 展示了完整模式：

- `vision_config`
- `mjx.create_render_context`
- `mjx.render`
- `mjx.get_rgb`
- frame stack 存在 `obs["pixels/view_0"]`

代码位置：

- `third_party/mujoco_playground/mujoco_playground/_src/dm_control_suite/cartpole.py:32-56`
- `third_party/mujoco_playground/mujoco_playground/_src/dm_control_suite/cartpole.py:73-107`
- `third_party/mujoco_playground/mujoco_playground/_src/dm_control_suite/cartpole.py:182-227`

同时 Playground 还提供了按名字读 sensor 的 helper：

- `third_party/mujoco_playground/mujoco_playground/_src/mjx_env.py:366-373`

这对你后续做 RGB + LiDAR 混合观测非常有用。

## 为什么不建议“直接迁移源码”

### 原因 1：后端不一致

- `rpg_flightning` 是 JAX + 自写动力学
- `D.VA` 是 PyTorch + DFlex
- 你的目标平台是 MuJoCo

这三者不是简单替换 import 就能统一的。

### 原因 2：观测语义不一致

`rpg_flightning` 现有 vision 其实是 feature projection，不是像素输入。

如果你想做论文平台，观测接口从 day 1 就应该设计成：

- state
- RGB
- depth
- LiDAR
- 可选特征

而不是继承一个只服务于 feature projection 的接口。

### 原因 3：梯度语义不一致

`rpg_flightning` 的 backward pass 是自定义 surrogate，不是 MuJoCo 原生梯度。

如果你不重新设计这件事，就会得到一个“看起来像迁移，实际上训练机理已经变掉”的系统。

### 原因 4：跨框架复用 `D.VA` 不划算

`third_party/D.VA` 的算法、环境和 buffer 全是 PyTorch/DFlex 风格。

如果硬接到 MJX 上，最后你会得到：

- JAX physics
- PyTorch policy/optimizer
- 中间频繁做 device / array 桥接

这对论文平台是坏架构，不利于后续扩展、复现实验和稳定调试。

## 推荐架构

## A. 平台主干

建议选：

- **环境层**: MuJoCo model + MJX/Warp env
- **baseline 训练层**: `jax_shac`
- **视觉层**: MuJoCo Playground 风格 render context
- **LiDAR 层**: MuJoCo `rangefinder` / `sensordata`

这条路线的优点：

1. 一个物理平台
2. 一个 observation API
3. state / vision / lidar 都能统一
4. SHAC 和 D.VA 都能在同一动力学环境上比较

## B. 观测分层

建议从一开始就把 observation 设计成结构体，而不是单一数组：

- `obs["state"]`
- `obs["rgb/front"]`
- `obs["depth/front"]`
- `obs["lidar/front"]`
- `obs["history"]`

训练时再由 wrapper 决定：

- SHAC-state 只拿 `state`
- D.VA 拿 `rgb` 作为 actor 输入、`state` 作为 critic 输入
- LiDAR 版 D.VA 拿 `lidar` 作为 actor 输入、`state` 作为 critic 输入

## C. 任务分层

建议先只做一个最小任务：

- `QuadrotorHover`

先不要一上来做障碍穿越、竞速、复杂视觉泛化。

因为论文平台最先要解决的是：

1. 动力学可微链路是否稳定
2. SHAC 是否能在 state 下收敛
3. RGB / LiDAR 观测 head 是否工作

## 迁移与实现建议

## 第一阶段：重建 `state-based hover` 任务

目标：

- 在 MuJoCo 中重做 `HoveringStateEnv`
- 尽量匹配 `rpg_flightning` 的 reward、termination、action delay

保留的语义：

- 随机初始位姿
- world box / collision
- hover reward
- action smoothness / action penalty
- control delay

不必强行保留的东西：

- `DoubleSphereCamera`
- 7-point feature projection 观测

## 第二阶段：复现 `state SHAC baseline`

直接复用：

- `third_party/jax_shac/envs/mjx_envs.py`
- `third_party/jax_shac/shac/train.py`

建议先不改 trainer，只写一个新 env，先验证：

1. reset / step 正常
2. reward 曲线正常
3. SHAC 能在 state observation 下学会 hover

如果这一步都不稳，后面 vision / lidar 只会更难。

## 第三阶段：加 RGB observation

建议参考 Playground 的方式，不要自己发明接口：

1. 在 env config 里显式加 `vision=True`
2. 初始化 render context
3. 在 `reset/step` 里维护 frame stack

实现目标应当对齐 D.VA：

- actor 输入是视觉 frame stack
- critic 输入仍然是 privileged state

## 第四阶段：加 LiDAR observation

优先顺序建议如下：

1. **site fan rangefinder**
2. **camera-based rangefinder**
3. 必要时再自己做更复杂的 beam packing

原因：

- MuJoCo 已经原生支持
- 数据直接进 `sensordata`
- 更容易和 state/RGB 共存

## 第五阶段：做 JAX 版 D.VA

这里我建议你不要直接迁 `third_party/D.VA/algorithms/dva.py`，而是复现其算法结构：

1. actor 吃 vision / lidar
2. critic 吃 state
3. image / lidar observation 不走物理梯度
4. 物理梯度只通过状态 rollout 和 bootstrap 值传播

你手头已经有两个非常好的参照：

- `third_party/D.VA/algorithms/dva.py`
- `third_party/jax_shac/shac/networks.py` 里的 `vPsC`

这条路比跨框架桥接稳得多。

## SHAC baseline 的具体建议

如果你说的 baseline 只是“SHAC baseline”，我建议拆成两个层次：

### Baseline 1: state SHAC

这是必须先做的。

价值：

- 校验 MuJoCo quadrotor env 是否可训练
- 给 D.VA 提供 privileged upper bound
- 最便于和 `rpg_flightning` 的 state-BPTT 作 sanity check

### Baseline 2: vision SHAC

这是可选的高风险 baseline。

原因：

- 你需要确认视觉链路是否真的适合 SHAC 反传
- 当前 `jax_shac` 虽然有 vision / `vPsC` 网络支持，但没有现成的无人机 vision env
- 这一步不应该阻塞 state baseline 和 D.VA 主线

所以正确顺序应该是：

1. `state SHAC`
2. `D.VA (vision actor, state critic)`
3. 再决定要不要补 `vision SHAC`

## 风险清单

## 高风险

### 1. 想“直接迁 `D.VA` 代码到 MJX”

这是高风险低收益。

原因：

- 框架错位
- buffer 设计错位
- autograd 语义错位

### 2. 想“严格复刻 `rpg_flightning` 梯度语义”

如果要严格做，你最终需要给 MuJoCo 版 quadrotor step 重新定义 backward rule，等价于重新实现 surrogate-gradient 机制。

这不是迁移，是新研究问题。

### 3. 一上来做视觉 SHAC

如果 state SHAC 还没收敛，就不应该进视觉 SHAC。

## 中风险

### 4. LiDAR 表达形式选错

如果一开始就做过于真实的 spinning lidar、复杂 beam timing、复杂遮挡模型，工程成本会显著上升。

建议先做：

- 固定扇形 beam
- 固定朝向
- 低维 range vector

## 低风险

### 5. MuJoCo 自定义 quadrotor env 实现本身

这一部分反而是最可控的，因为：

- `jax_shac` 的 env 模板已经够薄
- MuJoCo / Playground 的感知接口已经成熟

## 我建议你现在就按这个顺序推进

### Phase 0

写一个明确的 task spec：

- task 名称
- 动作语义
- reward
- termination
- state obs
- RGB obs
- LiDAR obs

### Phase 1

实现 `QuadrotorHoverMJX`，只开 state obs。

### Phase 2

用 `jax_shac` 跑通 `state SHAC`。

### Phase 3

加入 RGB frame stack，先只做 actor vision / critic state。

### Phase 4

做 JAX 版 D.VA 主线。

### Phase 5

补 LiDAR variant。

### Phase 6

如果论文需要，再决定是否补 `vision SHAC`。

## 最后判断

如果你的目标是“做论文平台”，我给的明确建议是：

- **不要直接 port `rpg_flightning`**
- **要在 MuJoCo 上重做一个统一平台**
- **先复现 state SHAC**
- **再做 D.VA 的 vision/lidar 版本**
- **把 `rpg_flightning` 当成任务定义和梯度设计参考，不要当成代码底座**

这条路线技术债最少，也最适合后面写论文、做 ablation、扩展新观测模态。

## 文档与代码依据

### 本地代码

- `third_party/rpg_flightning/flightning/envs/env_base.py`
- `third_party/rpg_flightning/flightning/envs/hovering_state_env.py`
- `third_party/rpg_flightning/flightning/envs/hovering_features_env.py`
- `third_party/rpg_flightning/flightning/objects/quadrotor_obj.py`
- `third_party/jax_shac/envs/mjx_envs.py`
- `third_party/jax_shac/envs/register_framed_hopper.py`
- `third_party/jax_shac/shac/train.py`
- `third_party/jax_shac/shac/networks.py`
- `third_party/D.VA/algorithms/dva.py`
- `third_party/D.VA/envs/dflex_env.py`
- `third_party/D.VA/envs/hopper_diff_render_env.py`
- `third_party/D.VA/models/actor.py`
- `third_party/mujoco/doc/overview.rst`
- `third_party/mujoco/python/mujoco/specs_test.py`
- `third_party/mujoco_playground/README.md`
- `third_party/mujoco_playground/learning/README.md`
- `third_party/mujoco_playground/mujoco_playground/_src/dm_control_suite/cartpole.py`
- `third_party/mujoco_playground/mujoco_playground/_src/mjx_env.py`

### 文档确认

- Context7 `MuJoCo` library lookup: `/google-deepmind/mujoco`
- Context7 `MuJoCo` docs query: MJX 安装、`mujoco.mjx` API、tutorial reset / observation 示例

