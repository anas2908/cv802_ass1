#!/usr/bin/env python3
"""Repository-local entry point for one PyCOLMAP dense stage."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv802_mvs.pycolmap_backend import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
