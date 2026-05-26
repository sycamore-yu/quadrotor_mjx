## ADDED Requirements

### Requirement: Complete rpg_flightning Core Semantics
The platform SHALL reproduce the core `rpg_flightning` environment semantics under MuJoCo MJX, including reset, step, rollout, auto-reset, wrappers, quadrotor dynamics, action delay, drag, reward, termination, and policy save/load.

#### Scenario: Fixed-Seed State Parity
- **GIVEN** a fixed seed, fixed initial state, and fixed action sequence
- **WHEN** running the MJX hover state environment and the `rpg_flightning` reference environment for shared state quantities
- **THEN** the test SHALL compare position, attitude, velocity, reward terms, and done flags
- **AND** the accepted tolerance SHALL be documented in the test

#### Scenario: Clean Project Imports
- **GIVEN** a fresh shell in the repository root
- **WHEN** running `python -m pytest tests -q`
- **THEN** tests SHALL NOT require manual `PYTHONPATH=src:third_party`
- **AND** training scripts SHALL NOT monkey patch `brax.v1`, `jaxtyping`, or `typeguard`

### Requirement: Hover State And Hover Feature Tasks
The platform SHALL provide both state-based hover and `rpg_flightning` feature-vision hover tasks.

#### Scenario: Hover State Training
- **GIVEN** the `hover_state` environment is registered
- **WHEN** running `python scripts/train.py --env hover_state --algo bptt --smoke`
- **THEN** the command SHALL finish without error and write metrics to the configured output directory

#### Scenario: Hover Feature Training
- **GIVEN** the `hover_features` environment is registered
- **WHEN** running `python scripts/train.py --env hover_features --algo bptt --smoke`
- **THEN** the command SHALL finish without error
- **AND** feature observations SHALL be projected landmark coordinates, not RGB pixels

#### Scenario: Hover Feature BPTT Learning
- **GIVEN** `hover_features` uses projected landmark coordinates and action history
- **WHEN** running non-smoke `bptt` training for the configured acceptance budget
- **THEN** the run SHALL write metrics with `backend="jax_bptt"`
- **AND** it SHALL improve evaluation reward over the initial policy for two fixed training seeds
- **AND** the evidence SHALL NOT depend on RGB/depth rendering availability

### Requirement: Three MJX Scene Tasks
The platform SHALL provide three scene tasks: `hover_obstacle`, `gate_crossing`, and `forest_navigation`.

#### Scenario: Hover Obstacle Scene
- **GIVEN** the `hover_obstacle` environment is registered
- **WHEN** the environment is reset with a fixed seed
- **THEN** obstacle layout, target point, observation shape, and reward terms SHALL be deterministic
- **AND** collision and clearance tests SHALL pass

#### Scenario: Gate Crossing Scene
- **GIVEN** the `gate_crossing` environment is registered
- **WHEN** a trajectory passes through a gate opening
- **THEN** waypoint progress SHALL advance
- **AND** collisions with gate geometry SHALL terminate or penalize the episode

#### Scenario: Forest Navigation Scene
- **GIVEN** the `forest_navigation` environment is registered
- **WHEN** the environment is reset with a fixed seed
- **THEN** tree/cylinder obstacle placement SHALL be deterministic
- **AND** rangefinder readings SHALL reflect obstacle distance for a controlled geometry case

### Requirement: Three Algorithm Training Matrix
The platform SHALL support `bptt`, `ppo`, and `shac` training through one common CLI.
APG SHALL be treated as folded into the BPTT track and SHALL NOT be exposed as a separate public training command for this platform change.
PPO SHALL use the MuJoCo Playground-style training chain: registry-loaded environment, `wrap_for_brax_training`, and Brax PPO vectorized trainer.
BPTT SHALL use a real differentiable training backend: vectorized JAX rollout through MJX state, loss accumulation over unroll windows, Optax policy updates, checkpoint/eval/render compatibility, and reward-improvement gates.
SHAC SHALL use a real `jax_shac` adapter backend: clean dependency imports, no runtime monkey patches, explicit actor/value update configuration, checkpoint/eval/render compatibility, and reward-improvement gates.
Smoke-only or baseline rollout adapters SHALL NOT satisfy this requirement for any algorithm.

#### Scenario: BPTT Reference Semantics
- **GIVEN** the project uses `bptt` as the public name for the differentiable policy-gradient track
- **WHEN** implementing or modifying the `bptt` backend
- **THEN** the implementation SHALL preserve the short-horizon FoPG or APG semantics shown in `third_party/mujoco/mjx/training_apg.ipynb`
- **AND** it SHALL expose horizon-like configuration rather than hiding all rollout length decisions behind only generic epoch counters
- **AND** compile-time mitigations SHALL prefer reference-consistent techniques such as rematerialization, fixed shapes, and shorter differentiable horizons rather than substituting a non-differentiable fallback

#### Scenario: BPTT Reward And Initialization Contract
- **GIVEN** a task is accepted as `bptt`-trainable for this platform
- **WHEN** defining its reward and initialization strategy
- **THEN** the task SHALL use dense or otherwise information-rich reward terms appropriate for first-order policy gradients
- **AND** any use of residual learning or baseline policy initialization SHALL be explicit in config, metrics, or task documentation
- **AND** a task that fundamentally depends on sparse or discontinuous reward SHALL document that mismatch instead of being silently counted as a backend failure

#### Scenario: SHAC Reference Semantics
- **GIVEN** the project uses `third_party/jax_shac/shac` as the reference implementation for `shac`
- **WHEN** implementing or modifying `src/dva_quadrotor_mjx/algorithms/shac.py`
- **THEN** the backend SHALL preserve separate actor and value parameter or optimizer states
- **AND** it SHALL preserve short-horizon actor updates plus TD-lambda or equivalent critic target computation
- **AND** it SHALL preserve explicit truncation-versus-termination handling in rollout or loss computation
- **AND** it SHALL keep vectorized reset and rollout semantics compatible with checkpoint, eval, render, and play flows

#### Scenario: Algorithm Reference Disclosure
- **GIVEN** a backend is claimed complete or substantially revised
- **WHEN** reviewing the implementation
- **THEN** the design or code review record SHALL state which reference implementation was followed:
  `third_party/mujoco/mjx/training_apg.ipynb` for `bptt` and
  `third_party/jax_shac/shac` for `shac`
- **AND** any intentional semantic deviation from those references SHALL be documented explicitly

#### Scenario: Smoke Training Matrix
- **GIVEN** `scripts/train.py` supports `--env`, `--algo`, and `--smoke`
- **WHEN** running the smoke matrix for `hover_obstacle`, `gate_crossing`, and `forest_navigation` across `bptt`, `ppo`, and `shac`
- **THEN** all nine commands SHALL finish without errors
- **AND** each run SHALL write metrics and a checkpoint or explicit smoke artifact

#### Scenario: Baseline Learning Signal
- **GIVEN** a non-smoke training config for each acceptance env (`hover_state`, `hover_obstacle`, `gate_crossing`, and `forest_navigation`) and for `hover_features + bptt`
- **WHEN** training `bptt`, `ppo`, and `shac` for the configured acceptance budget
- **THEN** every algorithm-env pair SHALL improve evaluation reward over the initial policy for two fixed training seeds
- **AND** `hover_features + bptt` SHALL improve evaluation reward over the initial policy for two fixed training seeds
- **AND** every scene algorithm-env pair SHALL beat its random-policy baseline on the scene primary metric

#### Scenario: Reward Improvement Gate Calculation
- **GIVEN** an algorithm-env pair is evaluated for delivery
- **WHEN** the gate runner starts
- **THEN** it SHALL evaluate the untrained initial policy and random-policy baseline on the same fixed evaluation seeds used for the trained checkpoint
- **AND** it SHALL run non-smoke training for the fixed acceptance budget declared in config
- **AND** it SHALL evaluate the final checkpoint with `scripts/eval.py`
- **AND** it SHALL pass when `final_mean_reward - initial_mean_reward` meets or exceeds the configured `min_reward_delta`
- **AND** it SHALL pass when the scene primary metric meets or exceeds the configured scene threshold
- **AND** it SHALL write the baseline metrics, final metrics, thresholds, seeds, checkpoint path, and pass/fail result to the metrics artifact

#### Scenario: Scene Non-Random Behavior Matrix
- **GIVEN** the scene envs are `hover_obstacle`, `gate_crossing`, and `forest_navigation`
- **WHEN** running acceptance training for `bptt`, `ppo`, and `shac`
- **THEN** all nine scene algorithm-env pairs SHALL beat their random-policy baselines
- **AND** `hover_obstacle` SHALL report target-distance improvement and collision or clearance metrics
- **AND** `gate_crossing` SHALL report waypoint/gate progress and collision metrics
- **AND** `forest_navigation` SHALL report goal-progress, tree-collision, and rangefinder-clearance metrics

#### Scenario: No Compromise Algorithm Backend
- **GIVEN** an algorithm CLI reports success for `bptt`, `ppo`, or `shac`
- **WHEN** inspecting the produced metrics artifact
- **THEN** it SHALL identify the concrete backend used (`jax_bptt`, `brax_ppo`, or `jax_shac`)
- **AND** it SHALL NOT use smoke-only rollout metrics as proof of training parity

#### Scenario: Backend Completion Contract
- **GIVEN** an algorithm is claimed complete for this change
- **WHEN** running its non-smoke training command on every acceptance env
- **THEN** the command SHALL write metrics with the concrete backend name
- **AND** it SHALL write a reloadable checkpoint
- **AND** the checkpoint SHALL work with `scripts/eval.py`, `scripts/render.py`, and `play-dva-quadrotor`
- **AND** the run SHALL pass the configured reward-improvement gate
- **AND** a smoke-only CLI run SHALL NOT satisfy backend completion

### Requirement: Evaluation, Rendering, And mjviser Visualization
The platform SHALL provide CLI tools for evaluation, offline rendering, and live mjviser inspection.

#### Scenario: Evaluate Checkpoint
- **GIVEN** a checkpoint produced by `scripts/train.py`
- **WHEN** running `python scripts/eval.py --env <env> --algo <algo> --checkpoint <path> --episodes 8`
- **THEN** the command SHALL finish without error
- **AND** it SHALL report mean reward, episode length, success rate when defined, and collision rate when defined

#### Scenario: Render Checkpoint
- **GIVEN** a checkpoint produced by `scripts/train.py`
- **WHEN** running `python scripts/render.py --env <env> --algo <algo> --checkpoint <path> --output artifacts/render.mp4`
- **THEN** the command SHALL produce a non-empty `.mp4` video
- **AND** `.npz` frame sequences or image-directory frame dumps SHALL NOT satisfy render acceptance
- **AND** missing video encoder dependencies SHALL fail loudly with a documented error rather than silently downgrading the artifact

#### Scenario: Visualize Scene With mjviser
- **GIVEN** `mjviser` is installed in the active environment
- **WHEN** running `python scripts/visualize_mjviser.py --env forest_navigation --mode random --steps 500 --port 8082`
- **THEN** the script SHALL launch a web viewer and advance the MuJoCo scene
- **AND** the printed URL SHALL be usable on the headless server through the configured port forwarding path

### Requirement: Rendering And Sensor Honesty
The platform SHALL distinguish `rpg_flightning` feature observations, MuJoCo rangefinder observations, and MJX/MJWarp RGB/depth rendering.

#### Scenario: No Fake Render Pass
- **GIVEN** a perception test invokes MJX/MJWarp rendering
- **WHEN** rendering is unavailable
- **THEN** tests SHALL skip or fail explicitly
- **AND** tests SHALL NOT silently substitute zero images and claim render success

#### Scenario: Rangefinder Geometry Validation
- **GIVEN** a controlled obstacle is placed at a known distance along a rangefinder ray
- **WHEN** the environment reads `sensordata`
- **THEN** measured range SHALL match expected distance within documented tolerance
