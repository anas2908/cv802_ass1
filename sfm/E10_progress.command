#!/bin/zsh
sfm_dir="${0:A:h}"
cd "$sfm_dir"
exec "$sfm_dir/.venv/bin/python" "$sfm_dir/work/E10-exhaustive-guided/show_progress.py"
