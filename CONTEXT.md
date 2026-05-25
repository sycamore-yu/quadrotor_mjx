# MJX Quadrotor Research Platform

This context defines the language for the MuJoCo MJX quadrotor migration work.
It keeps algorithm, observation, and rendering terms distinct so delivery gates
do not mix smoke wiring, geometric features, and rendered perception.

## Language

**Feature Observation**:
Projected 2D landmark coordinates from a camera model, optionally with action history.
_Avoid_: RGB observation, depth observation, rendered image

**Rendered Observation**:
RGB or depth data produced by a MuJoCo/MJX/MJWarp renderer.
_Avoid_: Feature observation, visual feature

**Rangefinder Observation**:
Distance readings from MuJoCo sensor data or an equivalent explicit range sensor path.
_Avoid_: LiDAR image, rendered depth

**Backend Completion**:
An algorithm backend is complete only when non-smoke training writes backend-identifying metrics, a reloadable checkpoint, and passes the configured acceptance gates.
_Avoid_: Smoke success, CLI wiring

## Relationships

- A **Feature Observation** is a geometric perception input, not a **Rendered Observation**.
- A **Rendered Observation** depends on rendering capability; a **Feature Observation** does not.
- **Backend Completion** requires training evidence, not only smoke execution.

## Example Dialogue

> **Dev:** "Does `hover_features` require RGB rendering?"
> **Domain expert:** "No. `hover_features` uses projected landmarks. RGB/depth rendering belongs to rendered perception work."

## Flagged Ambiguities

- "feature vision" was used near render/RGB/depth work. Resolved: in the
  `rpg_flightning` migration, feature vision means **Feature Observation**.
