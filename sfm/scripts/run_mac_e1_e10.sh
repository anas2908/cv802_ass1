#!/usr/bin/env bash
set -euo pipefail

# Self-contained macOS launcher for the E1-E10 reconstruction UI.
# It deliberately uses an explicit, managed Python so an activated system
# Python 3.9 environment cannot leak into the reconstruction runtime.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_PARENT="$(cd "$REPO_ROOT/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: this launcher is for macOS." >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: the pinned PyCOLMAP 4.2 macOS package requires an Apple-silicon Mac." >&2
    exit 1
fi

MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if [[ ! "$MACOS_MAJOR" =~ ^[0-9]+$ ]] || (( MACOS_MAJOR < 14 )); then
    echo "ERROR: PyCOLMAP 4.2 requires macOS 14 or newer on Apple silicon." >&2
    exit 1
fi

: "${CV802_DATA_ROOT:=$REPO_PARENT/cv802-data}"
: "${CV802_UI_PORT:=8770}"
if [[ ! "$CV802_UI_PORT" =~ ^[0-9]+$ ]] || (( CV802_UI_PORT < 1024 || CV802_UI_PORT > 65535 )); then
    echo "ERROR: CV802_UI_PORT must be an integer from 1024 to 65535." >&2
    exit 1
fi
case "$CV802_DATA_ROOT" in
    /*) ;;
    *) CV802_DATA_ROOT="$REPO_ROOT/$CV802_DATA_ROOT" ;;
esac
export CV802_DATA_ROOT

# Keep every downloaded tool, Python installation, package cache, environment,
# and generated result in the selected data root rather than in the Git clone.
export UV_CACHE_DIR="$CV802_DATA_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$CV802_DATA_ROOT/python"
UV_BIN_DIR="$CV802_DATA_ROOT/tools/uv-bin"
ENV_DIR="$CV802_DATA_ROOT/sfm/mac-env-py311-v1"
UV="$UV_BIN_DIR/uv"
UV_VERSION="0.12.18"

mkdir -p "$UV_BIN_DIR" "$UV_CACHE_DIR" "$CV802_DATA_ROOT/sfm"

if [[ ! -x "$UV" ]] || [[ "$("$UV" --version 2>/dev/null || true)" != "uv $UV_VERSION"* ]]; then
    command -v curl >/dev/null 2>&1 || {
        echo "ERROR: curl is required to install the Python environment." >&2
        exit 1
    }
    echo "Installing the uv environment manager under CV802_DATA_ROOT..."
    curl --proto '=https' --tlsv1.2 -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
        | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" UV_NO_MODIFY_PATH=1 sh
fi

# Ignore any environment (for example the old Python 3.9 mac-env) that was
# activated before this script. All commands below use explicit executables.
unset VIRTUAL_ENV

echo "Installing/locating managed Python 3.11..."
"$UV" python install 3.11
PYTHON_311="$("$UV" python find --managed-python 3.11)"

if [[ ! -x "$ENV_DIR/bin/python" ]] \
        || ! "$ENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    "$UV" venv --clear --managed-python --python "$PYTHON_311" "$ENV_DIR"
fi

echo "Installing the pinned reconstruction dependencies..."
"$UV" pip install --python "$ENV_DIR/bin/python" \
    --only-binary :all: \
    -r "$REPO_ROOT/sfm/requirements-mac-e1-e10.txt"

echo "Checking the reconstruction environment..."
"$ENV_DIR/bin/python" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, found {sys.version.split()[0]}")

import matplotlib
import imageio_ffmpeg
import numpy
import open3d
import PIL
import pycolmap
import scipy

print(f"Python {sys.version.split()[0]} ({platform.machine()})")
print(f"Open3D {open3d.__version__}; PyCOLMAP {pycolmap.__version__}")
PY

if ! command -v xcrun >/dev/null 2>&1 || ! xcrun --find swiftc >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: Apple's Swift compiler is required for the E4/E5/E7/E9/E10 masks.
Run `xcode-select --install`, finish that installation, then rerun this script.
EOF
    exit 1
fi

export CV802_ALLOW_NON_SLURM=1

echo "Opening the local reconstruction UI at http://127.0.0.1:$CV802_UI_PORT/"
exec "$ENV_DIR/bin/python" "$REPO_ROOT/sfm/mac_reconstruction_ui.py" ui --port "$CV802_UI_PORT"
