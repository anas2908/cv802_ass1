#!/usr/bin/env python3
"""Repository entry point; mutable outputs are fixed below DATA_ROOT/evaluation."""

from __future__ import annotations

import sys

# Source checkout bytecode is never a valid evaluation artifact.  Runtime
# commands also set PYTHONPYCACHEPREFIX to the data root for imported packages.
sys.dont_write_bytecode = True

from cv802_evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

