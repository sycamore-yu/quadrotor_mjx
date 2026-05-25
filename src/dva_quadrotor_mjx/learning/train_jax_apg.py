"""Compatibility target for the removed ``train-jax-apg`` console script."""

from dva_quadrotor_mjx.learning._entry import train_with_algo


def run() -> int:
    return train_with_algo("bptt")
