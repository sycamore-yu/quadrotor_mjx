"""Console entry points matching MuJoCo Playground naming style."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _script_main(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(repo_root / "scripts" / script_name))
    return namespace["main"]


def _train(algo: str) -> int:
    main = _script_main("train.py")
    return main(["--algo", algo, *sys.argv[1:]])


def train_jax_bptt() -> int:
    return _train("bptt")


def train_jax_apg() -> int:
    return _train("bptt")


def train_jax_ppo() -> int:
    return _train("ppo")


def train_jax_shac() -> int:
    return _train("shac")


def play() -> int:
    main = _script_main("visualize_mjviser.py")
    return main(sys.argv[1:])
