#!/usr/bin/env python3
"""Record the reviewed VGGSfM environment without installing anything.

The pinned LightGlue package declares ``opencv-python`` by distribution name,
while this headless server intentionally installs ``opencv-python-headless``.
Both provide the same ``cv2`` import, but pip's metadata checker does not treat
the headless distribution as a provider for the GUI distribution.  This audit
accepts only that one exact warning (or a genuinely clean future environment),
proves ``cv2`` imports from the fixed DATA_ROOT environment, and hashes a fresh
package inventory.  It never invokes pip install/uninstall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import atomic_write_json, file_record
from vggsfm_engine.paths import production_layout


AUDIT_NAME = "environment-audit-v1"
KNOWN_HEADLESS_METADATA_WARNING = (
    "lightglue 0.0 requires opencv-python, which is not installed."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_capture(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        env=env, check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def classify_pip_check(result: dict[str, Any]) -> dict[str, Any]:
    lines = [
        line.strip()
        for stream in (result["stdout"], result["stderr"])
        for line in stream.splitlines()
        if line.strip()
    ]
    if result["returncode"] == 0 and not lines:
        return {
            "classification": "clean",
            "only_known_headless_distribution_metadata_warning": False,
            "lines": [],
        }
    if result["returncode"] == 1 and lines == [KNOWN_HEADLESS_METADATA_WARNING]:
        return {
            "classification": "known_headless_distribution_metadata_warning_only",
            "only_known_headless_distribution_metadata_warning": True,
            "lines": lines,
        }
    raise RuntimeError(
        "pip check reported an unreviewed dependency problem: "
        f"returncode={result['returncode']}, lines={lines!r}"
    )


def audit(*, replace: bool = False) -> dict[str, Any]:
    layout = production_layout()
    layout.create_runtime_directories()
    expected_python = layout.python.resolve(strict=True)
    running_python = Path(sys.executable).resolve(strict=True)
    if running_python != expected_python:
        raise RuntimeError(
            f"Run this audit with {expected_python}; current interpreter is {running_python}"
        )
    destination = layout.root / f"{AUDIT_NAME}.json"
    inventory_path = layout.logs / AUDIT_NAME / "pip-freeze-all.txt"
    install_receipt_path = layout.root / "install_receipt.json"
    layout.assert_member(destination, label="environment audit")
    layout.assert_member(inventory_path, label="package inventory")
    layout.assert_member(install_receipt_path, label="original install receipt", must_exist=True)
    if not replace and (destination.exists() or inventory_path.exists()):
        raise RuntimeError(
            "Environment audit already exists; use --replace only to record a fresh read-only audit"
        )
    install_receipt_before = file_record(install_receipt_path)

    env = layout.runtime_environment()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pip_check = run_capture([str(expected_python), "-m", "pip", "check"], env)
    check_classification = classify_pip_check(pip_check)
    cv2_code = """
import importlib.metadata as metadata
import json
import cv2
def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
print(json.dumps({
    "cv2_version": cv2.__version__,
    "cv2_file": cv2.__file__,
    "opencv_python_distribution_version": version("opencv-python"),
    "opencv_python_headless_distribution_version": version("opencv-python-headless"),
    "lightglue_distribution_version": version("lightglue"),
}, sort_keys=True))
""".strip()
    cv2_import = run_capture([str(expected_python), "-B", "-c", cv2_code], env)
    if cv2_import["returncode"] != 0 or cv2_import["stderr"].strip():
        raise RuntimeError(f"cv2 import audit failed: {cv2_import!r}")
    try:
        cv2_details = json.loads(cv2_import["stdout"])
    except json.JSONDecodeError as exc:
        raise RuntimeError("cv2 import audit did not return JSON") from exc
    env_prefix = str(layout.env_prefix.resolve(strict=True)) + os.sep
    if not str(Path(cv2_details["cv2_file"]).resolve(strict=True)).startswith(env_prefix):
        raise RuntimeError("cv2 imported from outside the fixed VGGSfM environment")
    if (
        not cv2_details.get("cv2_version")
        or not cv2_details.get("opencv_python_headless_distribution_version")
        or cv2_details.get("opencv_python_distribution_version") is not None
        or cv2_details.get("lightglue_distribution_version") != "0.0"
    ):
        raise RuntimeError(f"Unexpected cv2/LightGlue distribution state: {cv2_details!r}")

    pip_freeze = run_capture(
        [str(expected_python), "-m", "pip", "freeze", "--all"], env
    )
    if pip_freeze["returncode"] != 0 or pip_freeze["stderr"].strip():
        raise RuntimeError(f"pip freeze failed: {pip_freeze!r}")
    if not pip_freeze["stdout"].endswith("\n"):
        raise RuntimeError("pip freeze inventory is unexpectedly empty or unterminated")
    atomic_write_text(inventory_path, pip_freeze["stdout"])
    inventory_sha256 = hashlib.sha256(pip_freeze["stdout"].encode("utf-8")).hexdigest()
    inventory_record = file_record(inventory_path)
    if inventory_record["sha256"] != inventory_sha256:
        raise RuntimeError("Persisted package inventory hash differs from captured stdout")
    install_receipt_after = file_record(install_receipt_path)
    if install_receipt_after != install_receipt_before:
        raise RuntimeError("Original installation receipt changed during read-only audit")

    report = {
        "schema_version": 1,
        "status": "complete",
        "audited_at": utc_now(),
        "scope": "Read-only package/import audit; no environment mutation and no inference.",
        "environment_prefix": str(layout.env_prefix),
        "python": str(expected_python),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "pip_check": {**pip_check, **check_classification},
        "cv2_import": {**cv2_import, "parsed": cv2_details},
        "pip_freeze": {
            "command": pip_freeze["command"],
            "returncode": pip_freeze["returncode"],
            "stderr": pip_freeze["stderr"],
            "line_count": len(pip_freeze["stdout"].splitlines()),
            "stdout_sha256": inventory_sha256,
            "inventory_file": inventory_record,
        },
        "interpretation": {
            "runtime_cv2_available": True,
            "provider": "opencv-python-headless",
            "metadata_mismatch_is_not_an_import_or_inference_failure": True,
            "why_pip_check_warns": (
                "LightGlue 0.0 declares the opencv-python distribution name. "
                "pip metadata does not consider opencv-python-headless an interchangeable "
                "provider even though both expose cv2."
            ),
            "safety_decision": (
                "Keep the reviewed headless provider. Do not install opencv-python beside it: "
                "dual distributions can overwrite the same cv2 package files."
            ),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "audit_source": file_record(Path(__file__).resolve()),
        "original_install_receipt_before": install_receipt_before,
        "original_install_receipt_after": install_receipt_after,
        "original_install_receipt_before_after_equal": True,
        "original_install_receipt_modified": False,
    }
    atomic_write_json(destination, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace", action="store_true",
        help="replace only this read-only audit and its package inventory",
    )
    args = parser.parse_args()
    result = audit(replace=args.replace)
    print(json.dumps({
        "status": result["status"],
        "pip_check_classification": result["pip_check"]["classification"],
        "cv2": result["cv2_import"]["parsed"],
        "pip_freeze": result["pip_freeze"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
