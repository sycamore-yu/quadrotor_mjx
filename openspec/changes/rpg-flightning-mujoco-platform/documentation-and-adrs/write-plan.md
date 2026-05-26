# Write-Plan: rpg-flightning-mujoco-platform 剩余工作

> **日期**: 2026-05-26
> **状态**: documentation-and-adrs 阶段后的后续执行计划

---

## 一、已遇到的问题总结

### 问题 1: SHAC `SyntaxError` — `@jaxtyped` / `typeguard` 不兼容

**描述**: `third_party/jax_shac` 大量使用了 `@jaxtyped(typechecker=beartype)` 装饰器和
`jaxtyping` 的符号化形状字符串（如 `"bdim obs_size"`）。新版 `typeguard` 把这些字符串当作
Python 表达式进行 AST 解析，导致 `SyntaxError`。

**已完成的修复**: 将 `jax_shac` 核心文件复制到 `src/dva_quadrotor_mjx/algorithms/shac/`，
设置 `typechecker = None` 并在环境变量中禁用 `JAXTYPING_DISABLE=1`。

**参考文件**:
- [train.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/algorithms/shac/train.py#L43-L50) — 禁用 jaxtyping 的代码
- [third_party/jax_shac/shac/train.py](file:///home/tong/tongworkspace/mjx_diff/third_party/jax_shac/shac/train.py) — 原始参考

**遗留**: `train.py` 第 44 行仍 `from jaxtyping import ...`，虽然运行时不再报错，但属于冗余依赖。

---

### 问题 2: BPTT `lax.scan` Pytree 结构不匹配 — `collision` 键

**描述**: 子类环境（`HoverObstacleEnv`、`GateCrossingEnv`、`ForestNavigationEnv`）在
`step()` 中向 `info` dict 添加了 `"collision"` 键，但基类 `HoveringStateEnv.reset()`
返回的 `info` 不包含该键。当 BPTT 使用 `jax.lax.scan` 扫描 step 时，第一次迭代的 carry
（来自 reset 的 state）与后续迭代的 carry（来自 step 的 state，多了 `collision` 键）
pytree 结构不一致，JAX 报 `TypeError: ... has 10 children but output has 11 children`。

**根因**: `lax.scan` 要求 carry 的 pytree 结构在所有迭代中保持恒定。

**修复策略**: 在 `HoveringStateEnv.reset()` 的 `info` 中初始化占位键
`"collision": jnp.array(False)`。同理，对 `GateCrossingEnv` 需要初始化
`"current_gate"`, `"gates_crossed"`, `"gate_success"`, `"collision"` 等键；
`ForestNavigationEnv` 需要初始化 `"collision"`, `"success"`, `"dist_to_goal"` 等键。

**参考文件**:
- [hover.py reset](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/hover.py#L175-L262) — 基类 reset，info 缺少 collision
- [hover_obstacle.py step](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/hover_obstacle.py#L47-L74) — step 中新增 collision 键
- [bptt.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/algorithms/bptt.py#L83-L85) — lax.scan 调用点

---

### 问题 3: SHAC 后端仅完成 smoke 初始化，未验证完整训练

**描述**: 当前 SHAC smoke path 通过 `_init_jax_shac_checkpoint()` 仅初始化网络和
状态然后写入 checkpoint，不执行任何梯度更新。这满足 smoke 测试但不满足
reward-improvement gate 要求。

**遗留**: 完整的 `num_timesteps > 0` 训练路径已连通但未在所有环境上验证。

**参考文件**:
- [trainer.py _init_jax_shac_checkpoint](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/algorithms/trainer.py#L253-L317)
- [trainer.py _train_jax_shac](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/algorithms/trainer.py#L179-L250)

---

### 问题 4: 子环境 info pytree 静态/动态键不一致

**描述**: `ForestNavigationEnv.reset()` 添加了 `tree_positions`、`tree_radii`、
`goal`、`start` 到 info dict，但 `GateCrossingEnv.reset()` 只添加了
`current_gate` 和 `gates_crossed`。各子环境的 info 结构在 reset vs step 之间不一致，
导致 `lax.scan` 和 `VecEnv(jax.vmap)` 在某些情况下出错。

**参考文件**:
- [forest_navigation.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/forest_navigation.py#L51-L81)
- [gate_crossing.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/gate_crossing.py#L58-L63)

---

### 问题 5: Acceptance Gate Runner 框架存在但未运行完整矩阵

**描述**: `scripts/check_acceptance.py` 和 gate runner 的 metrics 记录逻辑已实现
（task 6.26 ✓），但 13 个 full learning gate（6.14–6.25）和 feature BPTT gate（6.14a）
全部为 `[ ]` 未完成。

**参考文件**:
- [tasks.md](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/tasks.md#L108-L123)
- [check_acceptance.py](file:///home/tong/tongworkspace/mjx_diff/scripts/check_acceptance.py)

---

### 问题 6: mjviser 可视化在 headless 服务器上未验证

**描述**: 三个 mjviser scene 命令（7.1–7.3）和 render/eval 命令（7.4–7.5）标记为
`[ ]`。headless 模式下是否能启动 mjviser 以及 EGL 渲染是否能生成 MP4 还未确认。

**参考文件**:
- [visualize_mjviser.py](file:///home/tong/tongworkspace/mjx_diff/scripts/visualize_mjviser.py)
- [render.py](file:///home/tong/tongworkspace/mjx_diff/scripts/render.py)

---

### 问题 7: 场景 success/failure 测试缺失

**描述**: task 3.7 的 success/failure 测试覆盖不完整：
- `hover_obstacle`: clearance、collision、target hold 测试缺
- `gate_crossing`: pass gate、miss gate、collide gate、timeout 测试缺
- `forest_navigation`: rangefinder hit 测试缺

**参考文件**:
- [tasks.md 3.7](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/tasks.md#L35-L38)
- [test_scene_envs.py](file:///home/tong/tongworkspace/mjx_diff/tests/test_scene_envs.py)

---

### 问题 8: 未提交的代码变更 (17 文件)

**描述**: 上一代理完成了大量修改但未 git commit。当前有 17 个文件的 diff 涉及：
SHAC 迁移、trainer 统一接口、sensor mode 支持、raycasting 模块、backend contract 测试等。

---

## 二、实施计划

### 阶段 A: 基础修复与提交（预计 30 分钟）

| # | 任务 | 目标文件 | 验证 |
|---|------|----------|------|
| A1 | 在 `HoveringStateEnv.reset()` 的 info 中添加 `"collision": jnp.array(False)` 占位符 | [hover.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/hover.py) | BPTT smoke 不再报 pytree mismatch |
| A2 | 在 `GateCrossingEnv.reset()` 中初始化 `"gate_success"` 和 `"collision"` | [gate_crossing.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/gate_crossing.py) | gate_crossing + bptt smoke pass |
| A3 | 在 `ForestNavigationEnv.reset()` 中初始化 `"collision"`, `"success"`, `"dist_to_goal"` | [forest_navigation.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/forest_navigation.py) | forest + bptt smoke pass |
| A4 | 在 `HoverObstacleEnv.reset()` 中初始化 `"collision"` | [hover_obstacle.py](file:///home/tong/tongworkspace/mjx_diff/src/dva_quadrotor_mjx/envs/hover_obstacle.py) | hover_obstacle + bptt smoke pass |
| A5 | 运行 `pytest tests -q` 确认全部通过 | — | 0 failures |
| A6 | 提交所有未提交更改 | — | clean git status |

### 阶段 B: Smoke 矩阵全通（预计 60 分钟）

| # | 任务 | 验证命令 |
|---|------|----------|
| B1 | 运行所有 13 条 smoke 命令（task 6.1–6.13）| `scripts/train.py --env X --algo Y --smoke` |
| B2 | 修复发现的任何 smoke 失败 | 无错误退出 |
| B3 | 提交修复 | clean git status |

### 阶段 C: 场景 Success/Failure 测试（task 3.7, 预计 45 分钟）

| # | 任务 | 文件 |
|---|------|------|
| C1 | 为 hover_obstacle 添加 clearance/collision/target-hold 测试 | [test_scene_envs.py](file:///home/tong/tongworkspace/mjx_diff/tests/test_scene_envs.py) |
| C2 | 为 gate_crossing 添加 pass/miss/collide/timeout 测试 | 同上 |
| C3 | 为 forest_navigation 添加 seed-stable trees/tree collision/goal reach/rangefinder hit 测试 | 同上 |
| C4 | 运行 pytest 验证 | `pytest tests -q` |

### 阶段 D: Backend Completion Tests（task 4.5, 4.9, 预计 30 分钟）

| # | 任务 | 文件 |
|---|------|------|
| D1 | 完善 save/load parity test (task 4.5) | [tests/test_save_load_parity.py](file:///home/tong/tongworkspace/mjx_diff/tests/test_save_load_parity.py) |
| D2 | 实现 backend completion contract test (task 4.9): 验证每个 backend 的 metrics->checkpoint->eval->render 全链路 | [tests/test_backend_contract.py](file:///home/tong/tongworkspace/mjx_diff/tests/test_backend_contract.py) |

### 阶段 E: Full Learning Gates（task 6.14–6.25, 6.14a, 预计 4-8 小时训练时间）

> 这是最耗时的阶段，每个 env-algo 对需要非 smoke 训练并验证 reward 提升。

| # | Env + Algo | Task | 关键指标 |
|---|-----------|------|---------|
| E1 | hover_state + bptt (×2 seeds) | 6.14 | eval_reward > initial + delta |
| E2 | hover_features + bptt (×2 seeds) | 6.14a | 使用 landmark obs，非 RGB |
| E3 | hover_state + ppo (×2 seeds) | 6.15 | eval_reward > initial + delta |
| E4 | hover_state + shac (×2 seeds) | 6.16 | eval_reward > initial + delta |
| E5 | hover_obstacle + {bptt,ppo,shac} | 6.17-6.19 | target-dist + collision beat random |
| E6 | gate_crossing + {bptt,ppo,shac} | 6.20-6.22 | waypoint progress + collision beat random |
| E7 | forest_navigation + {bptt,ppo,shac} | 6.23-6.25 | goal-progress + tree-collision beat random |

**执行策略**: 使用 `scripts/check_acceptance.py` 自动化 gate runner，每个 pair 记录
seeds/thresholds/baseline/final 到 `artifacts/acceptance_results/`。

### 阶段 F: Visualization & Replay（task 7.1–7.5, 预计 30 分钟）

| # | 任务 | 验证命令 |
|---|------|----------|
| F1 | 验证 mjviser scene 模式 | `python scripts/visualize_mjviser.py --env hover_obstacle --mode scene --port 8080` |
| F2 | 验证 mjviser random 模式 | `python scripts/visualize_mjviser.py --env gate_crossing --mode random --steps 500 --port 8081` |
| F3 | 验证 mjviser forest 模式 | `python scripts/visualize_mjviser.py --env forest_navigation --mode random --steps 500 --port 8082` |
| F4 | 验证 render MP4 | `python scripts/render.py --env hover_obstacle --checkpoint <ckpt> --output artifacts/hover_obstacle.mp4` |
| F5 | 验证 eval | `python scripts/eval.py --env gate_crossing --algo shac --checkpoint <ckpt> --episodes 8` |

> 注意: headless 服务器上 mjviser 的 `--port` 需通过端口转发或 headless mode 验证。

### 阶段 G: Final Acceptance（task 8, 预计 15 分钟）

| # | 任务 |
|---|------|
| G1 | 运行 `openspec validate rpg-flightning-mujoco-platform --strict` |
| G2 | 运行 `python -m pytest tests -q` |
| G3 | 确认所有 smoke 命令通过 |
| G4 | 确认三个 mjviser scene 命令通过 |
| G5 | 更新 `delivery-status-and-grill.md` 确认无 open blocking grill |
| G6 | 更新 README 最终训练/eval/render 命令 |

---

## 三、依赖与风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| BPTT 在复杂场景（forest/gate）编译时间过长 | 高 | 训练 gate 延迟 | 使用 `jax.checkpoint` 减少内存；缩短 horizon |
| SHAC 完整训练在某些 env 上不收敛 | 中 | gate 失败 | 调参；降低 threshold；记录并上报 |
| headless 服务器缺少 ffmpeg → MP4 生成失败 | 低 | task 7.4 阻塞 | 安装 ffmpeg 或使用 imageio-ffmpeg |
| mjviser headless 模式下不可用 | 低 | task 7.1-7.3 阻塞 | 验证 EGL backend；或以 offscreen render 代替 |

---

## 四、参考文件索引

| 文档 | 路径 |
|------|------|
| OpenSpec 规格 | [spec.md](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/specs/rpg-flightning-mujoco-platform/spec.md) |
| 设计文档 | [design.md](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/design.md) |
| 任务清单 | [tasks.md](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/tasks.md) |
| 交付状态 | [delivery-status-and-grill.md](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/delivery-status-and-grill.md) |
| 提案 | [proposal.md](file:///home/tong/tongworkspace/mjx_diff/openspec/changes/rpg-flightning-mujoco-platform/proposal.md) |
| BPTT 参考 | [training_apg.ipynb](file:///home/tong/tongworkspace/mjx_diff/third_party/mujoco/mjx/training_apg.ipynb) |
| SHAC 参考 | [jax_shac/train.py](file:///home/tong/tongworkspace/mjx_diff/third_party/jax_shac/shac/train.py) |
| rpg_flightning 参考 | [hovering_state_env.py](file:///home/tong/tongworkspace/mjx_diff/third_party/rpg_flightning/flightning/envs/hovering_state_env.py) |
| Playground 参考 | [mujoco_playground](file:///home/tong/tongworkspace/mjx_diff/third_party/mujoco_playground) |
