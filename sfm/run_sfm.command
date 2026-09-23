#!/bin/zsh
set -e
sfm_dir="${0:A:h}"
cd "$sfm_dir/assignment1"
exec "$sfm_dir/.venv/bin/python" main.py
