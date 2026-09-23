#!/usr/bin/env bash
set -euo pipefail

# One-command macOS launcher for the bundled, read-only result viewer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_PARENT="$(cd "$REPO_ROOT/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: this launcher is for macOS." >&2
    exit 1
fi

: "${CV802_DATA_ROOT:=$REPO_PARENT/cv802-data}"
case "$CV802_DATA_ROOT" in
    /*) ;;
    *) CV802_DATA_ROOT="$REPO_ROOT/$CV802_DATA_ROOT" ;;
esac
mkdir -p "$CV802_DATA_ROOT"
CV802_DATA_ROOT="$(cd "$CV802_DATA_ROOT" && pwd -P)"
if [[ "$CV802_DATA_ROOT" == "$REPO_ROOT" || "$CV802_DATA_ROOT" == "$REPO_ROOT/"* ]]; then
    echo "Ignoring the old data path inside the Git clone."
    CV802_DATA_ROOT="$REPO_PARENT/cv802-data"
    mkdir -p "$CV802_DATA_ROOT"
    CV802_DATA_ROOT="$(cd "$CV802_DATA_ROOT" && pwd -P)"
fi
export CV802_DATA_ROOT

MANAGED_PYTHON="$CV802_DATA_ROOT/sfm/mac-env-py311-v1/bin/python"
if [[ -x "$MANAGED_PYTHON" ]]; then
    PYTHON="$MANAGED_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "ERROR: python3 was not found. Run sfm/scripts/run_mac_e1_e10.sh once to install it." >&2
    exit 1
fi

"$PYTHON" -c 'import sys; raise SystemExit("Python 3.9 or newer is required" if sys.version_info < (3, 9) else 0)'

ARCHIVE="$REPO_ROOT/saved_models/CV802_Project1_SavedResults_v1.zip"
if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: the bundled saved-model archive is missing: $ARCHIVE" >&2
    exit 1
fi

# Repeated launches reuse an already installed, valid result set.
if "$PYTHON" "$REPO_ROOT/sfm/browse_historical.py" \
        --data-root "$CV802_DATA_ROOT" validate >/dev/null 2>&1; then
    echo "Using the verified saved results already in $CV802_DATA_ROOT"
else
    echo "Installing and verifying the bundled saved results..."
    "$PYTHON" "$REPO_ROOT/scripts/package_saved_results.py" install \
        --archive "$ARCHIVE"
fi

"$PYTHON" "$REPO_ROOT/sfm/browse_historical.py" \
    --data-root "$CV802_DATA_ROOT" validate >/dev/null

START_PORT="${CV802_VIEWER_PORT:-8767}"
if [[ ! "$START_PORT" =~ ^[0-9]+$ ]] || (( START_PORT < 1024 || START_PORT > 65535 )); then
    echo "ERROR: CV802_VIEWER_PORT must be an integer from 1024 to 65535." >&2
    exit 1
fi

# Select the requested port, or the next free one, without asking the user.
PORT="$("$PYTHON" - "$START_PORT" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, min(start + 50, 65536)):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit("No free local viewer port was found")
PY
)"

URL="http://127.0.0.1:$PORT/"
echo "Opening the saved-result viewer at $URL"
echo "Keep this terminal open; press Control-C to stop the viewer."
(sleep 1; open "$URL") &
exec env CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$REPO_ROOT/sfm/browse_historical.py" \
    --data-root "$CV802_DATA_ROOT" serve --port "$PORT"
