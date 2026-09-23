#!/usr/bin/env python3
"""Thin repository-local entry point; implementation lives in vggsfm_engine."""

import sys

# The code root must remain GitHub-only. Official child inference gets a
# PYTHONPYCACHEPREFIX below DATA_ROOT; prevent this small launcher from creating
# bytecode beside source before the engine constructs that environment.
sys.dont_write_bytecode = True

from vggsfm_engine.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
