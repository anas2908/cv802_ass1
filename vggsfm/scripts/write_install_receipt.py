#!/usr/bin/env python3
"""Write the immutable-ish install inventory after a successful GPU import check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_PREFIX = (
    Path(os.environ.get("CV802_DATA_ROOT", "/l/users/anas.khan/cv_802_ass1"))
    .expanduser()
    .resolve()
    / "vggsfm"
)


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout.strip()


def within(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".install-receipt.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--lightglue-root", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve(strict=False)
    output.parent.resolve(strict=True).relative_to(EXPECTED_PREFIX.resolve(strict=True))
    official = within(args.official_root, EXPECTED_PREFIX)
    lightglue = within(args.lightglue_root, EXPECTED_PREFIX)
    python = within(args.environment_python, EXPECTED_PREFIX)
    gpu_probe = command_output(
        [
            str(python),
            "-c",
            (
                "import json, torch; print(json.dumps({"
                "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
            ),
        ]
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": command_output(["hostname"]),
        "official": {
            "path": str(official),
            "commit": command_output(["git", "-C", str(official), "rev-parse", "HEAD"]),
            "remote": command_output(
                ["git", "-C", str(official), "remote", "get-url", "origin"]
            ),
        },
        "lightglue": {
            "path": str(lightglue),
            "commit": command_output(["git", "-C", str(lightglue), "rev-parse", "HEAD"]),
            "remote": command_output(
                ["git", "-C", str(lightglue), "remote", "get-url", "origin"]
            ),
        },
        "environment_python": str(python),
        "python_version": command_output([str(python), "--version"]),
        "gpu_probe": json.loads(gpu_probe),
        "pip_freeze": command_output([str(python), "-m", "pip", "freeze"]).splitlines(),
        "cache_environment": {
            key: os.environ.get(key)
            for key in (
                "CONDA_PKGS_DIRS",
                "CONDA_ENVS_PATH",
                "CONDARC",
                "PIP_CACHE_DIR",
                "PIP_CONFIG_FILE",
                "TORCH_HOME",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_DATASETS_CACHE",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "CUDA_CACHE_PATH",
                "MPLCONFIGDIR",
                "PYTHONPYCACHEPREFIX",
                "TORCH_EXTENSIONS_DIR",
                "TRITON_CACHE_DIR",
                "NUMBA_CACHE_DIR",
                "GRADIO_TEMP_DIR",
                "TMPDIR",
                "PYTHONUSERBASE",
            )
        },
    }
    atomic_json(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
