"""SHAC backend package facade.

The public package import re-exports the project-owned implementation from the
adjacent ``src/dva_quadrotor_mjx/algorithms/shac.py`` module so callers keep a
stable ``dva_quadrotor_mjx.algorithms.shac`` import path while the runtime
boundary stays centralized in one file.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

_impl_path = Path(__file__).resolve().parents[1] / "shac.py"
_spec = spec_from_file_location(f"{__name__}._impl", _impl_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load SHAC backend implementation from {_impl_path}")

_impl: ModuleType = module_from_spec(_spec)
sys.modules[_spec.name] = _impl
_spec.loader.exec_module(_impl)

train = _impl.train
load_policy = _impl.load_policy
make_policy = _impl.make_policy
resolve_checkpoint_path = _impl.resolve_checkpoint_path

__all__ = [
    "train",
    "load_policy",
    "make_policy",
    "resolve_checkpoint_path",
]
