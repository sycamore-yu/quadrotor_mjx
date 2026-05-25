"""Reserved APG module for the DVA OpenSpec stage plan.

APG was merged into the BPTT user-facing algorithm name.  Keep this module so
older imports do not fail while new CLI/config paths use ``bptt`` only.
"""

from dva_quadrotor_mjx.algorithms.bptt import train

__all__ = ["train"]
