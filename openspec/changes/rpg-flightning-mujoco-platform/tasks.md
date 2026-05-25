# Implementation Tasks

## 0. Preconditions And Hygiene

- [x] 0.1 Remove manual `PYTHONPATH=src:third_party` requirement by making local package and selected third-party adapters importable through project config.
- [x] 0.2 Remove `brax.v1`, `jaxtyping`, and `typeguard` monkey patches from `scripts/train.py` and tests.
- [x] 0.3 Ensure `python -m pytest tests -q` works from a clean shell.
- [x] 0.4 Keep generated `__pycache__`, checkpoints, logs, and videos out of source control.
- [x] 0.5 Add a one-command smoke suite: `scripts/check_smoke.sh` or equivalent Python runner.

## 1. rpg_flightning Core Parity

- [x] 1.1 Implement `envs/base.py` parity with `rpg_flightning/flightning/envs/env_base.py`: rollout, auto-reset, metrics, wrapper compatibility.
- [x] 1.2 Implement `wrappers/normalize.py`, `wrappers/logging.py`, `wrappers/vectorization.py` based on `rpg_flightning/flightning/envs/wrappers.py`.
- [x] 1.3 Implement `dynamics/quadrotor.py` and `controllers/motor_model.py` as explicit, tested components rather than embedding all logic in `hover.py`.
- [x] 1.4 Add numeric parity tests against `rpg_flightning` fixed seeds/actions for shared state quantities.
- [x] 1.5 Document accepted tolerance where MJX integration differs from original Runge-Kutta implementation.

## 2. Hover State And Hover Feature Environments

- [x] 2.1 Refactor current `HoveringStateEnv` into a clean state env with documented observation/action schema.
- [x] 2.2 Implement `HoveringFeaturesEnv` compatibility mode using landmark projection from `hovering_features_env.py`.
- [x] 2.3 Implement `sensors/feature_camera.py` using `double_sphere_camera.py` as reference.
- [x] 2.4 Add `hover_state` and `hover_features` to registry.
- [x] 2.5 Add tests for state observation, feature observation, action history, reset, reward, done, auto-reset.

## 3. Three Scene Environments

- [x] 3.1 Implement `hover_obstacle` MJCF/assets and env class.
- [x] 3.2 Implement `gate_crossing` MJCF/assets and env class.
- [x] 3.3 Implement `forest_navigation` MJCF/assets/procedural generator and env class.
- [x] 3.4 Register all three envs in `envs/registry.py`.
- [ ] 3.5 Add sensor configs for state-only, feature, rangefinder, RGB/depth modes.
- [ ] 3.6 Add deterministic reset tests for every env.
- [ ] 3.7 Add success/failure tests for every env:
  - hover obstacle: clearance, collision, target hold.
  - gate crossing: pass gate, miss gate, collide gate, timeout.
  - forest navigation: seed-stable trees, tree collision, goal reach, rangefinder hit.

## 4. Algorithms

- [ ] 4.1 Implement `algorithms/bptt.py` as a real `jax_bptt` backend:
  - vectorized reset/step over `num_envs`
  - differentiable `lax.scan` unroll through MJX dynamics
  - trajectory loss over reward/cost terms
  - `jax.value_and_grad` + Optax policy update
  - metrics with backend=`jax_bptt`, loss, reward, SPS, compile/train time
  - checkpoint/eval/render compatibility
- [x] 4.2 Implement `algorithms/ppo.py` or a clean Brax PPO adapter.
- [ ] 4.3 Implement `algorithms/shac.py` as a real `jax_shac` backend:
  - clean imports from `third_party/jax_shac`
  - no runtime monkey patches
  - explicit actor/value config mapping from CLI/YAML
  - metrics with backend=`jax_shac`, actor loss, value loss, reward, SPS
  - checkpoint/eval/render compatibility
- [x] 4.4 Add common trainer interface:
  - `train(config) -> TrainResult`
  - `eval(checkpoint, env, episodes) -> EvalResult`
  - `make_policy(checkpoint) -> callable`
- [ ] 4.5 Add save/load parity test based on `rpg_flightning/examples/save_load_policy.ipynb`.
- [x] 4.6 Fold APG into the BPTT public algorithm track while keeping APG placeholders reserved for `dva-quadrotor-mjx`.
- [ ] 4.7 Remove fallback smoke/baseline rollout as an accepted training backend for `bptt` and `shac`.
- [ ] 4.8 Add backend assertion tests: metrics for `bptt`, `ppo`, and `shac` SHALL report `jax_bptt`, `brax_ppo`, and `jax_shac` respectively.
- [ ] 4.9 Add backend completion contract tests for each claimed-complete backend:
  non-smoke training writes concrete backend metrics, writes a reloadable
  checkpoint, works with `eval.py`, `render.py`, and `play-dva-quadrotor`, and
  passes the configured reward-improvement gate on every acceptance env.

## 5. CLI Scripts And Configs

- [x] 5.1 Replace `scripts/train.py` with generic CLI:
  - `--env`
  - `--algo`
  - `--config`
  - `--smoke`
  - `--seed`
  - `--num-envs`
  - `--steps`
  - `--output-dir`
- [x] 5.2 Implement `scripts/eval.py`.
- [x] 5.3 Implement `scripts/render.py`.
- [x] 5.4 Implement `scripts/visualize_mjviser.py`.
- [x] 5.5 Add YAML configs for each env and each algorithm.
- [x] 5.6 Ensure all outputs go under `artifacts/` or configurable run directory, not `third_party/`.

## 6. Training Script Matrix

Every command below must run without errors in smoke mode:

- [x] 6.1 `python scripts/train.py --env hover_state --algo bptt --smoke`
- [x] 6.2 `python scripts/train.py --env hover_state --algo ppo --smoke`
- [x] 6.3 `python scripts/train.py --env hover_state --algo shac --smoke`
- [x] 6.4 `python scripts/train.py --env hover_features --algo bptt --smoke`
- [x] 6.5 `python scripts/train.py --env hover_obstacle --algo bptt --smoke`
- [x] 6.6 `python scripts/train.py --env hover_obstacle --algo ppo --smoke`
- [x] 6.7 `python scripts/train.py --env hover_obstacle --algo shac --smoke`
- [x] 6.8 `python scripts/train.py --env gate_crossing --algo bptt --smoke`
- [x] 6.9 `python scripts/train.py --env gate_crossing --algo ppo --smoke`
- [x] 6.10 `python scripts/train.py --env gate_crossing --algo shac --smoke`
- [x] 6.11 `python scripts/train.py --env forest_navigation --algo bptt --smoke`
- [x] 6.12 `python scripts/train.py --env forest_navigation --algo ppo --smoke`
- [x] 6.13 `python scripts/train.py --env forest_navigation --algo shac --smoke`

Full training gates:

- [ ] 6.14 `hover_state + bptt` improves eval reward over initial policy for 2 seeds.
- [ ] 6.15 `hover_state + ppo` improves eval reward over initial policy for 2 seeds.
- [ ] 6.16 `hover_state + shac` improves eval reward over initial policy for 2 seeds.
- [ ] 6.17 `hover_obstacle + bptt` beats random-policy baseline on target-distance and collision/clearance metrics.
- [ ] 6.18 `hover_obstacle + ppo` beats random-policy baseline on target-distance and collision/clearance metrics.
- [ ] 6.19 `hover_obstacle + shac` beats random-policy baseline on target-distance and collision/clearance metrics.
- [ ] 6.20 `gate_crossing + bptt` beats random-policy baseline on waypoint/gate progress and collision metrics.
- [ ] 6.21 `gate_crossing + ppo` beats random-policy baseline on waypoint/gate progress and collision metrics.
- [ ] 6.22 `gate_crossing + shac` beats random-policy baseline on waypoint/gate progress and collision metrics.
- [ ] 6.23 `forest_navigation + bptt` beats random-policy baseline on goal-progress, tree-collision, and rangefinder-clearance metrics.
- [ ] 6.24 `forest_navigation + ppo` beats random-policy baseline on goal-progress, tree-collision, and rangefinder-clearance metrics.
- [ ] 6.25 `forest_navigation + shac` beats random-policy baseline on goal-progress, tree-collision, and rangefinder-clearance metrics.
- [ ] 6.26 Implement the gate runner that records fixed train/eval seeds, initial/random baseline metrics, final checkpoint metrics, thresholds, checkpoint path, and pass/fail result.

## 7. Visualization And Replay

- [ ] 7.1 `python scripts/visualize_mjviser.py --env hover_obstacle --mode scene --port 8080`
- [ ] 7.2 `python scripts/visualize_mjviser.py --env gate_crossing --mode random --steps 500 --port 8081`
- [ ] 7.3 `python scripts/visualize_mjviser.py --env forest_navigation --mode random --steps 500 --port 8082`
- [ ] 7.4 `python scripts/render.py --env hover_obstacle --checkpoint <ckpt> --output artifacts/hover_obstacle.mp4`
- [ ] 7.5 `python scripts/eval.py --env gate_crossing --algo shac --checkpoint <ckpt> --episodes 8`

## 8. Final Acceptance

- [x] 8.1 `openspec validate rpg-flightning-mujoco-platform --strict` passes.
- [ ] 8.2 `python -m pytest tests -q` passes without manual environment variables.
- [x] 8.3 All smoke train commands in section 6 pass.
- [ ] 8.4 Three scene visualization commands in section 7 pass.
- [x] 8.5 Final README section documents exact train/eval/render commands.
- [ ] 8.6 `delivery-status-and-grill.md` has no open blocking grill decision for
  any acceptance item being claimed complete.
