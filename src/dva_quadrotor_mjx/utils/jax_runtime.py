from __future__ import annotations

import os
from pathlib import Path


def configure_jax_runtime(*, cache_subdir: str = "jax_cache") -> Path | None:
    """Applies shared JAX/MuJoCo runtime settings before first compilation."""

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
        xla_flags = f"{xla_flags} --xla_gpu_triton_gemm_any=True".strip()
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")

    if os.environ.get("DVA_JAX_ENABLE_COMPILATION_CACHE", "1") == "0":
        return None

    cache_dir = Path(
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            Path.cwd() / "artifacts" / cache_subdir,
        )
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    return cache_dir
