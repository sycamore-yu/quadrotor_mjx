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

### Not Yet Delivered

- `jax_bptt` backend is not implemented.
- `jax_shac` backend is not implemented.
- Backend assertion tests are not implemented.
- Full `python -m pytest tests -q` is not yet verified in the current state.
- Deterministic reset and success/failure tests for all three scene envs remain
  incomplete.
- Sensor configs for state-only, feature, rangefinder, RGB/depth modes remain
  incomplete.
- Full learning gates are not complete:
  `hover_state + bptt`, `hover_state + ppo`, `hover_state + shac`, and at least
  one successful short-run learner for each scene env.
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
- Full learning gates from `tasks.md` section 6.14 to 6.17
- mjviser visualization gates from `tasks.md` section 7.1 to 7.5
- No metric artifact with `backend="smoke_rollout"` may be used as evidence of
  training parity.

## Compromise Risk Scan

Known compromise patterns and required handling:

- Smoke rollout adapters: allowed only as diagnostics, never as backend
  completion.
- DVA/APG placeholders: belong to `openspec/changes/dva-quadrotor-mjx`; they
  must not be counted as implemented algorithms in this change.
- RGB/depth rendering fallback: tests must skip or fail loudly when rendering is
  unavailable; zero images or fake render output are not acceptable.
- Disabled XML physical collisions: acceptable only when paired with explicit
  JAX semantic collision tests and documented scene semantics.
- Missing ffmpeg/pyav video writer: acceptable only when `render.py` writes a
  documented non-empty frame sequence fallback.

## Grill Decisions

### Decision 1: Backend Completion Definition

Status: accepted

Question:

What exactly counts as "algorithm backend complete" for this change?

Decision:

An algorithm backend is complete only when its training command produces a
metrics artifact with the correct backend name, writes a reloadable checkpoint,
works with `eval.py`, `render.py`, and `play-dva-quadrotor`, and passes the
configured short-run reward-improvement gate. A smoke command that only checks
CLI wiring is not enough.

Spec updates:

- `spec.md` has a normative backend completion scenario.
- `tasks.md` has backend completion contract tests.
- `design.md` references this definition in the algorithm section.

### Decision 2: Scene Learning Scope

Status: open

Question:

Should a backend be considered complete after `hover_state` learning only, or
must it also show at least one non-random behavior on each of the three scene
envs?

Recommended answer:

Backend-specific completion should require `hover_state` reward improvement for
that backend, while platform-level delivery should require each scene env to
show non-random behavior with at least one algorithm. Requiring every backend to
learn every scene before backend completion would block BPTT/SHAC integration on
scene-specific reward tuning instead of proving the backend is real.

## Open Grill Questions

- Should `BPTT` target state and feature observations in the same milestone, or
  state first and feature second?
- Should `SHAC` be required to use `third_party/jax_shac` directly, or may it
  copy/adapt the algorithm into `src/dva_quadrotor_mjx/algorithms/shac.py`?
- Should render/video acceptance require ffmpeg-backed `.mp4`, or is a non-empty
  `.npz` frame sequence acceptable on headless machines?
