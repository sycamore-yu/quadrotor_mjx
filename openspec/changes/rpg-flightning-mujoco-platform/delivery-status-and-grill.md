# Delivery Status And Grill Log

Date: 2026-05-25

This file is the handoff point for `rpg-flightning-mujoco-platform`. It records
what is actually delivered, what remains before the change is shippable, and the
grill decisions that prevent smoke-only or compromise implementations from being
mistaken for completed work.

## Current Delivery Status

### Source Of Truth

- This file records implementation truth for this change. If README, tasks, or
  code comments appear to imply stronger completion than this file, this file is
  authoritative until the contradiction is fixed.
- `tasks.md` remains the executable checklist.
- `specs/rpg-flightning-mujoco-platform/spec.md` remains the normative
  acceptance contract.
- `design.md` remains the implementation rationale and reference map.

### Completed

- Project packaging supports editable install from the repository root.
- Playground-style console scripts exist for `train-jax-bptt`, `train-jax-ppo`,
  `train-jax-shac`, and `play-dva-quadrotor`.
- APG is folded into the BPTT track and is not exposed as a public training
  command for this change.
- Environment registry includes `quadrotor_hover`, `hover_state`,
  `hover_features`, `hover_obstacle`, `gate_crossing`, and
  `forest_navigation`.
- PPO is connected to the real Playground-style training chain:
  registry-loaded env, `wrap_for_brax_training`, Brax PPO vectorized trainer,
  and Brax/Orbax checkpoint directories.
- PPO checkpoint loading works for `eval`, `render`, and `play` through
  `src/dva_quadrotor_mjx/learning/ppo_checkpoint.py`.
- OpenSpec validation passes for `rpg-flightning-mujoco-platform`.
- Documentation now states that smoke-only rollout adapters are not algorithm
  parity.
- Grill consensus now rejects "pending real trainer" as a shippable state:
  incomplete BPTT/SHAC backends must remain explicit open work, not a hidden
  compromise behind successful smoke commands.
- BPTT dispatch now uses the real `jax_bptt` scan/gradient backend for smoke
  and short runs, and its metrics report `backend="jax_bptt"`.
- SHAC dispatch now uses the vendored `jax_shac` backend without the previous
  `typeguard`/`jaxtyping` runtime failure, writes checkpoints under
  `artifacts/`, and its smoke metrics report `backend="jax_shac"`.
- Backend contract tests now reject `smoke_rollout` as a completion backend and
  assert `jax_bptt`, `brax_ppo`, and `jax_shac` backend names.
- The acceptance gate runner now records seeds, initial/random/final metrics,
  thresholds, checkpoint path, backend, and pass/fail, with tests for stronger
  threshold overrides and artifact validation.
- Scene task configs now include `state`, `feature`, `rangefinder`, and
  `rgb_depth` sensor mode metadata.
- Scene deterministic reset tests now cover all three scene envs.

### Not Yet Delivered

- `jax_bptt` full completion remains pending: reward-improvement gates and
  eval/render/play checkpoint compatibility are not verified for every
  acceptance env.
- `jax_shac` full completion remains pending: it still needs the project-owned
  adapted backend contract from Decision 5, plus reward-improvement gates and
  eval/render/play checkpoint compatibility for every acceptance env.
- Full `python -m pytest tests -q` passes in the current state
  (`51 passed, 1 warning`, 2026-05-25).
- Full success/failure tests for all three scene envs remain incomplete:
  target hold, gate pass/collide/timeout, and forest rangefinder-hit cases are
  still open.
- Full learning gates are not complete:
  all `bptt`, `ppo`, and `shac` combinations across `hover_state`,
  `hover_obstacle`, `gate_crossing`, and `forest_navigation`, plus
  `hover_features + bptt`.
- Three mjviser scene commands are not yet verified as final acceptance.

## Delivery Gates

The change is not ready to archive until all gates below pass:

- `openspec validate rpg-flightning-mujoco-platform --strict`
- `python -m pytest tests -q`
- Smoke matrix for all three scene envs across `bptt`, `ppo`, and `shac`
- Metrics backend assertions:
  - BPTT metrics report `backend="jax_bptt"`
  - PPO metrics report `backend="brax_ppo"`
  - SHAC metrics report `backend="jax_shac"`
- Full learning gates from `tasks.md` section 6.14 to 6.26
- The feature-observation BPTT gate from `tasks.md` section 6.14a
- mjviser visualization gates from `tasks.md` section 7.1 to 7.5
- No metric artifact with `backend="smoke_rollout"` may be used as evidence of
  training parity.

## Compromise Risk Scan

Known compromise patterns and required handling:

- Smoke rollout adapters: allowed as diagnostics, never as backend
  completion.
- DVA/APG placeholders: belong to `openspec/changes/dva-quadrotor-mjx`; they
  must not be counted as implemented algorithms in this change.
- RGB/depth rendering fallback: tests must skip or fail loudly when rendering is
  unavailable; zero images or fake render output are not acceptable.
- Disabled XML physical collisions: acceptable when paired with explicit
  JAX semantic collision tests and documented scene semantics.
- Missing ffmpeg/pyav video writer: not acceptable for final render acceptance.
  `render.py` must write a non-empty `.mp4` or fail loudly with a documented
  encoder dependency error.

## Grill Decisions

### Decision 1: Backend Completion Definition

Status: accepted

Question:

What exactly counts as "algorithm backend complete" for this change?

Decision:

An algorithm backend is complete when its training command produces a
metrics artifact with the correct backend name, writes a reloadable checkpoint,
works with `eval.py`, `render.py`, and `play-dva-quadrotor`, and passes the
configured reward-improvement gate. A smoke command that checks CLI wiring is
not enough.

Spec updates:

- `spec.md` has a normative backend completion scenario.
- `tasks.md` has backend completion contract tests.
- `design.md` references this definition in the algorithm section.

### Decision 2: Scene Learning Scope

Status: accepted

Question:

Should a backend be considered complete after `hover_state` learning, or must
it also show non-random behavior on each of the three scene envs?

Decision:

Every public algorithm (`bptt`, `ppo`, `shac`) must pass the reward-improvement
gate on `hover_state`, `hover_obstacle`, `gate_crossing`, and
`forest_navigation`. Every scene algorithm-env pair must beat its random-policy
baseline on the scene primary metric. The platform is not deliverable while any
cell in this matrix remains unverified.

### Decision 3: Reward-Improvement Gate

Status: accepted

Question:

What is the concrete reward-improvement gate?

Decision:

For every algorithm-env pair, the gate runner evaluates an untrained initial
policy and a random-policy baseline on fixed evaluation seeds, trains for the
configured acceptance budget, reloads the final checkpoint, and evaluates the
checkpoint on the same evaluation seeds. The pair passes when
`final_mean_reward - initial_mean_reward >= min_reward_delta` and the scene
primary metric meets the configured threshold. The metrics artifact must record
the seeds, thresholds, baseline metrics, final metrics, checkpoint path, and
pass/fail result.

### Decision 4: BPTT Feature Scope

Status: accepted

Question:

Should `BPTT` target state and feature observations in the same milestone, or
state first and feature second?

Decision:

`BPTT` must target state and feature observations in the same milestone. In this
change, `hover_features` means `rpg_flightning`-style projected landmark
features plus action history, not RGB/depth rendering. Therefore feature BPTT is
part of the `rpg-flightning-mujoco-platform` milestone and must pass a non-smoke
reward-improvement gate. It must not be split out as a rendering task. Real
RGB/depth rendered perception remains separate from feature-observation parity
and must keep explicit capability checks.

Spec updates:

- `CONTEXT.md` defines `Feature Observation` separately from rendered RGB/depth
  observations.
- `design.md` already states that `hover_features` uses projected landmarks,
  not RGB pixels.
- `spec.md` already requires `hover_features` observations to be projected
  landmark coordinates, not RGB pixels, and now includes the feature BPTT
  learning gate.
- `tasks.md` now includes `hover_features + bptt` in the non-smoke learning
  gates.

### Decision 5: SHAC Backend Ownership

Status: accepted

Question:

Should `SHAC` be required to use `third_party/jax_shac` directly, or may it
copy/adapt the algorithm into `src/dva_quadrotor_mjx/algorithms/shac.py`?

Decision:

`SHAC` SHALL be implemented as an adapted backend inside
`src/dva_quadrotor_mjx/algorithms/shac.py`, using `third_party/jax_shac` as the
reference implementation rather than as the direct runtime entry point. The
adapted backend must preserve the algorithm semantics needed for this platform:
short-horizon actor update, value/critic update, explicit actor/value config,
checkpoint/eval/render/play compatibility, backend-identifying metrics, and the
reward-improvement gates. It must remove `third_party/jax_shac` assumptions that
conflict with this package boundary, including manual `PYTHONPATH`, runtime
monkey patches, output paths outside `artifacts/`, or hidden notebook-only
entry points.

Spec updates:

- `README.md` now references `third_party/jax_shac/shac/train.py` as the SHAC
  reference implementation and states that the project-owned backend lives in
  `src/dva_quadrotor_mjx/algorithms/shac.py`.
- `design.md` now states that SHAC is a project-owned adapted backend derived
  from `third_party/jax_shac`.
- `tasks.md` now requires the adapted backend to live in
  `src/dva_quadrotor_mjx/algorithms/shac.py`.

### Decision 6: Render Acceptance Artifact

Status: accepted

Question:

Should render/video acceptance require ffmpeg-backed `.mp4`, or is a non-empty
`.npz` frame sequence acceptable on headless machines?

Decision:

Final render acceptance SHALL require a non-empty `.mp4` video artifact. A
non-empty `.npz` frame sequence, image directory, or other frame dump is not an
acceptable substitute for this change. If the headless server lacks the required
video writer, `render.py` must fail loudly with a clear encoder dependency
error; it must not silently downgrade the evidence artifact.

Spec updates:

- `spec.md` now requires `scripts/render.py` to produce a non-empty `.mp4`
  video and explicitly disallows `.npz` frame sequence fallback.
- `design.md` now records MP4 as the render artifact contract.
- `README.md` now labels the render command as an MP4 video path.

## Open Grill Questions

None.
