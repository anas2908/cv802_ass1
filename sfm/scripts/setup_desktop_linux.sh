#!/usr/bin/env bash
# Isolated real Open3D GUI: host packages are downloaded/extracted, never installed.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use the existing compute allocation.' >&2; exit 2; }
CV802_GUI_CODE=/home/anas.khan/cv802_project/project1/ass1/sfm
CV802_GUI_DATA=/l/users/anas.khan/cv_802_ass1/sfm
[[ -d "$CV802_GUI_DATA" && -w "$CV802_GUI_DATA" ]] || exit 2
export TMPDIR="$CV802_GUI_DATA/tmp/desktop"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export XDG_CACHE_HOME="$CV802_GUI_DATA/cache/desktop/xdg"
export PIP_CACHE_DIR="$CV802_GUI_DATA/cache/desktop/pip"
export PYTHONPYCACHEPREFIX="$CV802_GUI_DATA/cache/desktop/pycache"
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export APT_CONFIG="$CV802_GUI_CODE/configs/desktop-apt.conf"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" \
  "$CV802_GUI_DATA/logs/desktop" "$CV802_GUI_DATA/downloads/desktop/debs" \
  "$CV802_GUI_DATA/runtime/desktop/apt/var/lib/apt/lists/partial" \
  "$CV802_GUI_DATA/runtime/desktop/sysroot"
exec > >(tee -a "$CV802_GUI_DATA/logs/desktop/setup.log") 2>&1
/usr/bin/python3.12 -m venv "$CV802_GUI_DATA/envs/open3d-desktop"
"$CV802_GUI_DATA/envs/open3d-desktop/bin/python" -B -m pip install \
  'numpy==1.26.4' 'open3d-cpu==0.19.0' 'Pillow==11.3.0' 'websockify==0.13.0'
"$CV802_GUI_DATA/envs/open3d-desktop/bin/python" -B -m pip freeze \
  > "$CV802_GUI_DATA/logs/desktop/python-freeze.txt"
# APT_CONFIG redirects all state and excludes system configuration hooks.
apt-config dump
apt-get update
cd "$CV802_GUI_DATA/downloads/desktop/debs"
apt-get download xvfb x11vnc libxfont2 libvncclient1 libvncserver1 x11-xkb-utils \
  xfonts-base xfonts-encodings xfonts-utils xserver-common novnc libfontenc1 proot libtalloc2
for cv802_gui_deb in "$CV802_GUI_DATA/downloads/desktop/debs/"*.deb; do
  dpkg-deb -x "$cv802_gui_deb" "$CV802_GUI_DATA/runtime/desktop/sysroot"
done
sha256sum "$CV802_GUI_DATA/downloads/desktop/debs/"*.deb \
  > "$CV802_GUI_DATA/logs/desktop/deb-checksums.sha256"
echo 'Desktop dependencies extracted into DATA_ROOT only.'
