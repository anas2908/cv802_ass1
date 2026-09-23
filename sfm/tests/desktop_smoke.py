"""Real Open3D GUI smoke test for all preserved E1-E10 display clouds.

This is an opt-in display test, not a discoverable ``unittest`` module.  Run it
inside the dedicated CPU-only X/VNC step.  Every generated receipt and image is
constrained to ``DATA_ROOT/sfm/logs/desktop``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from historical_browser.catalog import HistoricalCatalog, SUBJECTS
from historical_desktop import DesktopSelection, build_window_class


DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ticks(application: Any, count: int = 8) -> None:
    for _ in range(count):
        if not application.run_one_tick():
            raise RuntimeError("Open3D application stopped during the smoke test")
        time.sleep(0.03)


def _capture_display(path: Path) -> None:
    from PIL import ImageGrab

    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError("DISPLAY is unset; the native desktop test needs X/VNC")
    image = ImageGrab.grab(xdisplay=display)
    image.save(path, format="PNG")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    command.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "sfm/logs/desktop/validation",
    )
    command.add_argument("--width", type=int, default=1280)
    command.add_argument("--height", type=int, default=800)
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    data_root = arguments.data_root.resolve(strict=True)
    allowed_output = (data_root / "sfm/logs/desktop").resolve(strict=False)
    output = arguments.output_dir.resolve(strict=False)
    if not _inside(output, allowed_output):
        raise SystemExit(f"output must remain below {allowed_output}")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "receipt.json"

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output),
        "display": os.environ.get("DISPLAY"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "libgl_always_software": os.environ.get("LIBGL_ALWAYS_SOFTWARE"),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        },
        "results": [],
        "screenshots": [],
    }
    _write_json(receipt_path, receipt)

    application = None
    window = None
    try:
        import numpy as np
        import open3d as o3d
        import open3d.visualization.gui as gui

        catalog = HistoricalCatalog(data_root=data_root)
        selection = DesktopSelection(catalog)
        gui_module, window_type = build_window_class()
        if gui_module is not gui:
            raise RuntimeError("launcher and test imported different Open3D GUI modules")

        application = gui.Application.instance
        application.initialize()
        window = window_type(arguments.width, arguments.height, catalog, "E1", "light")
        _ticks(application)

        receipt["runtime"] = {
            "open3d_version": o3d.__version__,
            "open3d_distribution": "open3d-cpu",
            "open3d_distribution_version": metadata.version("open3d-cpu"),
            "numpy_version": np.__version__,
            "native_scene_widget": type(window._scene) is gui.SceneWidget,
            "widget_type": f"{type(window._scene).__module__}.{type(window._scene).__name__}",
            "build_cuda_module": bool(o3d._build_config.get("BUILD_CUDA_MODULE", False)),
            "controls": "Open3D SceneWidget ROTATE_CAMERA (native orbit, pan, wheel zoom)",
        }
        if not receipt["runtime"]["native_scene_widget"]:
            raise RuntimeError("the desktop window is not using the native Open3D SceneWidget")
        if receipt["runtime"]["build_cuda_module"]:
            raise RuntimeError("desktop validation must use the CPU-only Open3D build")

        for experiment_index, experiment_id in enumerate(selection.experiment_ids):
            window._historical_experiment.selected_index = experiment_index
            _ticks(application, 4)
            if window._selected_experiment != experiment_id:
                window._on_historical_experiment(
                    selection.experiment_label(experiment_id), experiment_index
                )
            available = selection.available_subjects(experiment_id)
            for subject_index, subject in enumerate(available):
                window._historical_subject.selected_index = subject_index
                _ticks(application, 4)
                if window._selected_subject != subject:
                    window._on_historical_subject(subject, subject_index)
                window.load_selected_result()
                # llvmpipe can present one or two frames behind the Python
                # callbacks.  Let the native widget finish drawing before a
                # count check or full-display capture.
                _ticks(application, 60 if experiment_id == "MVS" else 20)
                expected = selection.resolve(experiment_id, subject)["result"].point_count
                loaded = len(window._loaded_cloud.points) if window._loaded_cloud else 0
                if loaded != expected:
                    raise RuntimeError(
                        f"{experiment_id} {subject}: loaded {loaded}, expected {expected}"
                    )
                receipt["results"].append(
                    {
                        "experiment": experiment_id,
                        "subject": subject,
                        "method": "mvs" if experiment_id == "MVS" else "sfm",
                        "expected_points": expected,
                        "loaded_points": loaded,
                        "status": "loaded",
                    }
                )
                if (experiment_id, subject) in {
                    ("E1", "light"),
                    ("E4", "dark"),
                    ("E10", "light"),
                    ("MVS", "light"),
                }:
                    screenshot = output / f"{experiment_id}-{subject}-whole-window.png"
                    _capture_display(screenshot)
                    receipt["screenshots"].append(_file_record(screenshot))

        if selection.available_subjects("E10") != ("light",):
            raise RuntimeError("E10 subject selector must expose light only")
        if selection.available_subjects("MVS") != ("light",):
            raise RuntimeError("MVS subject selector must expose only the validated Light result")
        if len(receipt["results"]) != 20:
            raise RuntimeError("desktop validation must load 19 SfM clouds and one MVS cloud")

        receipt["status"] = "complete"
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["summary"] = {
            "clouds_loaded": len(receipt["results"]),
            "sfm_clouds_loaded": 19,
            "mvs_clouds_loaded": 1,
            "selection_slots": 22,
            "available_clouds": 20,
            "e10_dark": "intentionally not_run and absent from subject selector",
            "mvs_dark": "not_complete; no validated published fused PLY",
            "geometry_files_modified": 0,
        }
        _write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2))
        return 0
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(receipt_path, receipt)
        raise
    finally:
        if window is not None:
            window.window.close()
        if application is not None:
            for _ in range(2):
                try:
                    application.run_one_tick()
                except Exception:
                    break


if __name__ == "__main__":
    raise SystemExit(main())
