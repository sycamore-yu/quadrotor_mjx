# DVA Quadrotor MJX

MJX-based quadrotor simulation environment migrated from RPG Flightning.

## Project Structure

```
src/dva_quadrotor_mjx/
├── envs/              # Environments
│   ├── base.py        # MjxEnv base class
│   ├── hover.py       # HoveringStateEnv
│   ├── hover_features.py    # HoveringFeaturesEnv
│   ├── hover_obstacle.py    # Obstacle avoidance
│   ├── gate_crossing.py     # Gate crossing navigation
│   ├── forest_navigation.py # Procedural forest
│   ├── registry.py    # Environment registry
│   └── sensors/       # Sensors
│       ├── feature_camera.py   # Double sphere camera
│       └── rangefinder.py      # Rangefinder utilities
├── algorithms/        # RL algorithms
│   ├── bptt.py        # Backpropagation through time
│   ├── ppo.py         # Proximal policy optimization
│   ├── shac.py        # Short-horizon actor-critic
│   └── trainer.py     # Unified training interface
├── dynamics/          # Physics
│   ├── quadrotor.py   # Quadrotor dynamics
│   └── drag.py        # Aerodynamic drag
├── controllers/       # Control
│   ├── low_level.py   # Rate controller
│   └── motor_model.py # Motor delay model
├── wrappers/          # Environment wrappers
│   ├── normalize.py   # Action normalization
│   ├── logging.py     # Episode logging
│   └── vectorization.py # Parallel envs
└── configs/           # YAML configs
    ├── tasks/         # Task configs
    └── algorithms/    # Algorithm configs

scripts/
├── train.py           # Training CLI
├── eval.py            # Evaluation CLI
├── render.py          # Offline rendering
└── visualize_mjviser.py # Video rendering

tests/
├── test_env_base.py
├── test_hover_env.py
├── test_hover_state_env.py
├── test_scene_envs.py
├── test_save_load.py
└── test_numeric_parity.py
```

## Environments

| Environment | Description | Observation | Action |
|-------------|-------------|-------------|--------|
| `hover_state` | State-based hovering | position (3) + rotation (9) + velocity (3) + action history (8) = 23 | thrust + body rates (4) |
| `hover_features` | Feature-based hovering | projected landmarks (14) + action history (8) = 22 | thrust + body rates (4) |
| `hover_obstacle` | Hovering with static obstacle | same as hover_state | thrust + body rates (4) |
| `gate_crossing` | Gate crossing navigation (3 gates) | same as hover_state | thrust + body rates (4) |
| `forest_navigation` | Forest with procedural trees | same as hover_state | thrust + body rates (4) |

## Quick Start

### Training

```bash
# Script form
python scripts/train.py --env hover_state --algo bptt --smoke
python scripts/train.py --env hover_state --algo ppo --smoke
python scripts/train.py --env hover_state --algo shac --smoke

# MuJoCo Playground-style console scripts after package install
train-jax-bptt --env_name hover_obstacle --smoke
train-jax-ppo --env_name gate_crossing --smoke
train-jax-shac --env_name forest_navigation --smoke
```

The console scripts are registered in `pyproject.toml` as `dva_quadrotor_mjx.learning.*:run` entry points, matching the same pattern as MuJoCo Playground's `train-jax-ppo = "learning.train_jax_ppo:run"`.
APG is folded into the BPTT track, so the public training matrix is `bptt`, `ppo`, and `shac`.

PPO uses the MuJoCo Playground-style chain: `envs.registry.load(...)`, `dva_quadrotor_mjx.wrapper.wrap_for_brax_training`, and Brax PPO's vectorized trainer. BPTT and SHAC still use the current baseline rollout path until their dedicated algorithm tasks are completed.

Why this is explicit: earlier implementation work only aligned the console-script naming, package entry points, output directories, and smoke checks. It did not use Playground's actual vectorized trainer path, so it was a temporary baseline rollout rather than full training parity. This is now corrected for PPO; BPTT and SHAC remain documented follow-up work instead of being presented as Playground-parity trainers.

Default console runs are intentionally short:

```bash
train-jax-bptt --env quadrotor_hover
```

That command runs `1 epoch / 1 step / 1 env` so CLI and MJX compilation can be checked quickly. Use `--full` for the normal baseline defaults:

```bash
train-jax-bptt --env quadrotor_hover --full
# epochs=100 steps=1000 num_envs=128
```

### 3x3 Scene Matrix

```bash
for env in hover_obstacle gate_crossing forest_navigation; do
  train-jax-bptt --env_name "$env" --smoke
  train-jax-ppo --env_name "$env" --smoke
  train-jax-shac --env_name "$env" --smoke
done
```

### Evaluation

```bash
python scripts/eval.py --env gate_crossing --algo shac --checkpoint artifacts/gate_crossing_shac_seed0.ckpt --episodes 1 --steps 1
```

### Interactive Visualization (MuJoCo Playground Style)

```bash
# Live mjviser scene inspection on a headless server
play-dva-quadrotor --env hover_obstacle --mode scene --port 8080
play-dva-quadrotor --env gate_crossing --mode random --steps 500 --port 8081
play-dva-quadrotor --env forest_navigation --mode checkpoint --algo shac --checkpoint artifacts/forest_navigation_shac_seed0.ckpt --steps 500 --port 8082

# Offline frames/video
python scripts/render.py --env hover_obstacle --algo bptt --checkpoint artifacts/hover_obstacle_bptt_seed0.ckpt --output artifacts/hover_obstacle.mp4 --steps 200
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_hover_state_env.py -v
python -m pytest tests/test_scene_envs.py -v
python -m pytest tests/test_save_load.py -v
```

## Observation / Action Schema

### HoveringStateEnv

**Observation (dim=23):**
- `obs[0:3]` — position `[x, y, z]` in world frame
- `obs[3:12]` — flattened rotation matrix `R_WB` (9 elements)
- `obs[12:15]` — velocity `[vx, vy, vz]` in world frame
- `obs[15:23]` — action history (2 previous actions x 4 dims)

**Action (dim=4):**
- `action[0]` — total thrust [N]
- `action[1:4]` — body rates `[omega_x, omega_y, omega_z]` [rad/s]

### HoveringFeaturesEnv

**Observation (dim=22):**
- `obs[0:14]` — 7 feature points projected to 2D image plane (normalized to [-1, 1])
- `obs[14:22]` — normalized action history

## Configuration

Task configs: `src/dva_quadrotor_mjx/configs/tasks/*.yaml`
Algorithm configs: `src/dva_quadrotor_mjx/configs/algorithms/*.yaml`

## References

- Original implementation: `third_party/rpg_flightning/`
- MuJoCo Playground: `third_party/mujoco_playground/`
- jax_shac: `third_party/jax_shac/`
