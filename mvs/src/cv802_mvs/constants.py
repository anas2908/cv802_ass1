"""Storage constants shared by the MVS command-line tools."""

import os
from pathlib import Path

VERSION = "0.1.0"
SCHEMA_VERSION = 1

# This assignment deliberately has a fixed storage contract.  Keeping the
# canonical paths here makes an accidental run in $HOME fail before COLMAP can
# create a large dense workspace.
DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
DATA_ROOT = Path(os.environ.get("CV802_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
if not DATA_ROOT.is_absolute():
    raise RuntimeError("CV802_DATA_ROOT must be an absolute path")
DATA_ROOT = DATA_ROOT.resolve()
METHOD_ROOT = DATA_ROOT / "mvs"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
