"""Immutable project locations and the reviewed upstream revision."""

import os
from pathlib import Path

DEFAULT_PROJECT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
PROJECT_DATA_ROOT = Path(
    os.environ.get("CV802_DATA_ROOT", str(DEFAULT_PROJECT_DATA_ROOT))
).expanduser()
if not PROJECT_DATA_ROOT.is_absolute():
    raise RuntimeError("CV802_DATA_ROOT must be an absolute path")
PROJECT_DATA_ROOT = PROJECT_DATA_ROOT.resolve()
METHOD_DATA_ROOT = PROJECT_DATA_ROOT / "vggsfm"

# Reviewed on 2026-09-19.  Updating this SHA is an explicit provenance change:
# update PROVENANCE.md, run the unit tests, and create a new experiment run id.
OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/vggsfm.git"
OFFICIAL_COMMIT = "e1d9d2eb2b3575525792206fb94b2c749c58dc50"
LIGHTGLUE_REPOSITORY = "https://github.com/jytime/LightGlue.git"
LIGHTGLUE_COMMIT = "2f23ca2ea9638cecad7f7220795210fc6b8353c3"

MODEL_NAME = "vggsfm_v2_0_0"
MODEL_REPOSITORY = "facebook/VGGSfM"

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
)
REQUIRED_COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")
OPTIONAL_COLMAP_FILES = ("frames.bin", "rigs.bin")
