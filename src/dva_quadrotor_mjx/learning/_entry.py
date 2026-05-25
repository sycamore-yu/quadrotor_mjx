"""Shared console entry helpers for Playground-style scripts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def script_main(script_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    namespace = runpy.run_path(str(repo_root / "scripts" / script_name))
    return namespace["main"]


def train_with_algo(algo: str) -> int:
    main = script_main("train.py")
    return main(["--algo", algo, *sys.argv[1:]])


def play() -> int:
    main = script_main("visualize_mjviser.py")
    return main(sys.argv[1:])

