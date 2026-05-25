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
- **AND** rangefinder readings SHALL reflect obstacle distance for at least one controlled geometry case

### Requirement: Three Algorithm Training Matrix
The platform SHALL support `bptt`, `ppo`, and `shac` training through one common CLI.
APG SHALL be treated as folded into the BPTT track and SHALL NOT be exposed as a separate public training command for this platform change.

#### Scenario: Smoke Training Matrix
- **GIVEN** `scripts/train.py` supports `--env`, `--algo`, and `--smoke`
- **WHEN** running the smoke matrix for `hover_obstacle`, `gate_crossing`, and `forest_navigation` across `bptt`, `ppo`, and `shac`
- **THEN** all nine commands SHALL finish without errors
- **AND** each run SHALL write metrics and a checkpoint or explicit smoke artifact

#### Scenario: Baseline Learning Signal
- **GIVEN** a non-smoke training config for `hover_state`
- **WHEN** training `bptt`, `ppo`, and `shac` for the configured short-run budget
- **THEN** each algorithm SHALL improve evaluation reward over the initial policy for at least two fixed seeds

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
- **THEN** the command SHALL produce a non-empty video or documented frame sequence

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
