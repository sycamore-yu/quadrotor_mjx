"""Shared pytest environment defaults for headless, deterministic test runs."""

from __future__ import annotations

import os

from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime


# Tests primarily validate semantics and artifact contracts.  Defaulting to CPU
# avoids flaky CUDA plugin initialization and OOM during JIT-heavy smoke cases
# while still allowing explicit GPU overrides from the shell.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
configure_jax_runtime(cache_subdir="jax_test_cache")
