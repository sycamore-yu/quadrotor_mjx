"""Shared pytest environment defaults for headless, deterministic test runs."""

from __future__ import annotations

import os


# Tests primarily validate semantics and artifact contracts.  Defaulting to CPU
# avoids flaky CUDA plugin initialization and OOM during JIT-heavy smoke cases
# while still allowing explicit GPU overrides from the shell.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
