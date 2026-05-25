## OVERVIEW
当前是在远程服务器headless模式下运行的
中文回答
测试环境使用uv环境/home/tong/tongworkspace/mjx_diff/.venv/bin
Think Before Coding,每次写代码前列一个task list
涉及代码查看时使用code-review-graph工具查看代码关系
分析问题解决问题给出解决方案时给出文档依据
总是使用context7或deepwiki查询文档进行确认
尽量参考已有实现，避免重复造轮子
每次修改完代码git add 你修改的代码并提交以保存你当前的工作
在修改 mjx_diff 前，必须先用 GitNexus 查询相关源码。优先查询 third_party下的仓库
需要先找入口、调用链、数据结构和相似实现，再给修改方案。
禁止只凭记忆回答。
给用户脚本前应该自己运行一遍确保没有报错后提交
These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

## Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 — Surgical Changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor what isn't broken. Match existing style.

## Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

## Rule 5 — Use the model only for judgment calls
Use me for: classification, drafting, summarization, extraction.
Do NOT use me for: routing, retries, deterministic transforms.
If code can answer, code answers.

## Rule 6 — Token budgets are not advisory
Per-task: 4,000 tokens. Per-session: 30,000 tokens.
If approaching budget, summarize and start fresh.
Surface the breach. Do not silently overrun.

## Rule 7 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

## Rule 8 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

## Rule 9 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

## Rule 10 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

## Rule 11 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

## Rule 12 — Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## What Done Means
For this project, "done" means the requested outcome is implemented, verified,
documented where needed, and not represented by a smoke-only or placeholder
path. If any required verification is skipped, blocked, or substituted, the work
is not done; report the gap explicitly.

OpenSpec-backed work is done when:
- The relevant change under `openspec/changes/<change-id>/` has proposal,
  design, specs, and tasks aligned with the implementation.
- `openspec validate <change-id> --strict` passes.
- Every task claimed complete has direct evidence: test output, training
  artifact, metrics file, render/play output, or a documented reviewer decision.
- No pending, placeholder, fallback, or diagnostic path is counted as delivery.

Code work is done when:
- The implementation is committed to the intended project files only.
- The relevant tests pass in `/home/tong/tongworkspace/mjx_diff/.venv/bin`
  without manual `PYTHONPATH` unless the task explicitly changes packaging.
- GitNexus has been used before editing and `npx gitnexus detect-changes` has
  been run before commit for the edited scope.
- Existing user changes are not reverted or hidden inside unrelated commits.

Training work is done when:
- The training command uses the intended backend, and metrics identify the
  concrete backend name.
- Smoke commands are treated as wiring checks, not learning evidence.
- Non-smoke training writes a reloadable checkpoint and metrics artifact.
- `scripts/eval.py`, `scripts/render.py`, and `play-dva-quadrotor` can consume
  the checkpoint or the limitation is recorded as an open task.

For `rpg-flightning-mujoco-platform`, delivery requires:
- `bptt`, `ppo`, and `shac` each use real backends: `jax_bptt`, `brax_ppo`, and
  `jax_shac`.
- Every algorithm-env pair across `hover_state`, `hover_obstacle`,
  `gate_crossing`, and `forest_navigation` passes the configured
  reward-improvement gate.
- Every scene algorithm-env pair beats its random-policy baseline on the scene
  primary metrics.
- Three mjviser scene commands and the final evaluation/render paths are
  verified on the headless server.

## How To Verify Work
Use the smallest verification set that proves the actual claim. Do not present
a broader claim than the commands you ran can prove.

Minimum verification commands for OpenSpec changes:
- `openspec validate <change-id> --strict`
- `rg -n "pending|placeholder|fake|smoke-only|baseline rollout|not implemented|TODO|FIXME" openspec/changes/<change-id>`

Minimum verification commands for Python/package changes:
- `.venv/bin/python -m compileall -q src/dva_quadrotor_mjx scripts`
- `.venv/bin/python -m pytest tests -q`

Minimum verification commands for training CLI changes:
- Run the exact user-facing command before giving it to the user.
- Inspect the metrics artifact and confirm the backend field and checkpoint path.
- Run eval/render/play against the produced checkpoint when the task claims
  checkpoint usability.

Reward-improvement gate verification records:
- train seeds and eval seeds
- initial-policy metrics
- random-policy baseline metrics
- final checkpoint metrics
- configured thresholds
- checkpoint path
- pass/fail result
