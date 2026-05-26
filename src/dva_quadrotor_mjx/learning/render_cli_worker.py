"""Package-module render entry point used by the render CLI wrapper."""

from __future__ import annotations

import argparse
import sys

from dva_quadrotor_mjx.learning.render_worker import render_rollout
from dva_quadrotor_mjx.policies import SUPPORTED_ALGOS
from dva_quadrotor_mjx.utils.jax_runtime import configure_jax_runtime

configure_jax_runtime()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a quadrotor rollout")
    parser.add_argument("--env", required=True)
    parser.add_argument("--algo", choices=SUPPORTED_ALGOS, default="bptt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=str, default="artifacts/render.mp4")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args(argv)

    try:
        output = render_rollout(
            env_name=args.env,
            algo=args.algo,
            checkpoint=args.checkpoint,
            random_policy=args.random_policy,
            seed=args.seed,
            steps=args.steps,
            output=args.output,
            width=args.width,
            height=args.height,
        )
    except Exception as exc:
        raise RuntimeError(
            "MP4 video writer unavailable; install an imageio ffmpeg backend "
            "and rerun render.py. Frame-dump fallback is intentionally disabled "
            "by the OpenSpec render acceptance contract."
        ) from exc

    print(f"Rendered video: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
