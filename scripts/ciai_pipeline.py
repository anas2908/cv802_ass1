#!/usr/bin/env python3
"""Portable CIAI orchestration for COLMAP MVS and VGGSfM.

This module contains no web code.  It validates the active Slurm/GPU/storage
contract, stages repository photographs into the current user's Lustre area,
and invokes the already-tested method engines with resumable experiment names.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterable


CODE_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
Progress = Callable[[int, str], None]
Output = Callable[[str], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_datasets() -> list[str]:
    result = []
    datasets = CODE_ROOT / "datasets"
    if not datasets.is_dir():
        return result
    for child in sorted(datasets.iterdir()):
        images = child / "images"
        if child.is_dir() and SAFE_NAME.fullmatch(child.name) and any(
            item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            for item in images.rglob("*")
        ):
            result.append(child.name)
    return result


def validate_request(dataset: str, method: str) -> None:
    if dataset not in available_datasets():
        raise ValueError(f"Unknown or empty dataset: {dataset}")
    if method not in {"sfm", "mvs", "vggsfm"}:
        raise ValueError("Method must be 'sfm', 'mvs' or 'vggsfm'")


def validate_runtime(data_root: Path) -> dict[str, str]:
    """Fail before downloads if this is not a one-GPU CIAI compute job."""

    data_root = data_root.resolve()
    user = os.environ.get("USER", "")
    expected = Path("/l/users") / user if user else None
    if not user or expected is None:
        raise RuntimeError("USER is unavailable")
    try:
        data_root.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(
            f"CV802_DATA_ROOT must be inside your own Lustre folder {expected}; got {data_root}"
        ) from error
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Start this through scripts/run_ciai_gpu_ui.sh; no Slurm job is active")
    data_root.mkdir(parents=True, exist_ok=True)
    probe = data_root / f".cv802-write-test-{os.getpid()}"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    mount = subprocess.run(
        ["findmnt", "--noheadings", "--output", "TARGET,FSTYPE", "--target", str(data_root)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    disk = shutil.disk_usage(data_root)
    if disk.free < 20 * 1024**3:
        raise RuntimeError(
            f"At least 20 GiB free is required below {data_root}; only {disk.free / 1024**3:.1f} GiB is free"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and "," in visible:
        raise RuntimeError(f"Exactly one allocated GPU is required; CUDA_VISIBLE_DEVICES={visible!r}")
    gpu_command = ["nvidia-smi"]
    if visible:
        gpu_command.append(f"--id={visible}")
    gpu_command.extend(
        ["--query-gpu=name,memory.total,uuid", "--format=csv,noheader"]
    )
    gpu = subprocess.run(
        gpu_command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip().splitlines()
    if len(gpu) != 1:
        raise RuntimeError(
            "Exactly one allocated GPU is required; the allocation-aware "
            f"nvidia-smi query reported {len(gpu)}"
        )
    return {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": os.uname().nodename,
        "mount": mount,
        "free_gib": f"{disk.free / 1024**3:.1f}",
        "gpu": gpu[0],
        "cuda_visible_devices": visible or "not-set",
    }


def _manifest_records(dataset: str) -> list[dict[str, object]]:
    manifest = json.loads((CODE_ROOT / "datasets" / "images_manifest.json").read_text())
    records = manifest.get("datasets", {}).get(dataset, {}).get("files")
    if not isinstance(records, list) or len(records) < 2:
        raise RuntimeError(f"The repository image manifest has no valid entry for {dataset}")
    return records


def stage_images(dataset: str, destination: Path, output: Output) -> Path:
    """Copy and verify one dataset without ever symlinking into the checkout."""

    source_images = (CODE_ROOT / "datasets" / dataset / "images").resolve(strict=True)
    records = _manifest_records(dataset)
    expected: list[tuple[Path, int, str]] = []
    prefix = Path(dataset) / "images"
    for record in records:
        relative = Path(str(record["name"]))
        try:
            image_relative = relative.relative_to(prefix)
        except ValueError as error:
            raise RuntimeError(f"Unsafe image-manifest path: {relative}") from error
        if image_relative.is_absolute() or ".." in image_relative.parts:
            raise RuntimeError(f"Unsafe image-manifest path: {relative}")
        expected.append((image_relative, int(record["bytes"]), str(record["sha256"])))
    if len({item[0] for item in expected}) != len(expected):
        raise RuntimeError(f"Duplicate image paths in the manifest for {dataset}")

    def verify(root: Path) -> None:
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"Staged image root must be a physical directory: {root}")
        resolved_root = root.resolve(strict=True)
        actual = sorted(
            item.relative_to(root)
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        wanted = sorted(item[0] for item in expected)
        if actual != wanted:
            raise RuntimeError(f"Image inventory mismatch in {root}")
        for relative, size, digest in expected:
            path = root / relative
            try:
                path.resolve(strict=True).relative_to(resolved_root)
            except ValueError as error:
                raise RuntimeError(f"Staged image escapes its data directory: {path}") from error
            if path.is_symlink():
                raise RuntimeError(f"Staged images must be physical files: {path}")
            if path.stat().st_size != size or _sha256(path) != digest:
                raise RuntimeError(f"Image checksum mismatch: {path}")

    if destination.exists():
        verify(destination)
        output(f"Reused {len(expected)} checksum-verified images in {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-images-", dir=destination.parent) as temp:
        staged = Path(temp) / "images"
        staged.mkdir()
        for index, (relative, size, digest) in enumerate(expected, 1):
            source = source_images / relative
            try:
                source.resolve(strict=True).relative_to(source_images)
            except ValueError as error:
                raise RuntimeError(f"Repository image escapes its dataset directory: {source}") from error
            if source.is_symlink():
                raise RuntimeError(f"Repository images must be physical files: {source}")
            if source.stat().st_size != size or _sha256(source) != digest:
                raise RuntimeError(f"Repository image does not match its manifest: {source}")
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.stat().st_size != size or _sha256(target) != digest:
                raise RuntimeError(f"Copied image failed checksum verification: {target}")
            if index % 25 == 0 or index == len(expected):
                output(f"Staged images {index}/{len(expected)}")
        staged.rename(destination)
    return destination


def _runtime_environment(data_root: Path) -> dict[str, str]:
    cache = data_root / "runtime" / "cache"
    temporary = data_root / "runtime" / "tmp"
    for path in (cache / "pip", cache / "xdg", cache / "huggingface", cache / "torch", temporary):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CV802_DATA_ROOT": str(data_root),
            "PYTHONUNBUFFERED": "1",
            "PIP_CACHE_DIR": str(cache / "pip"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
            "TMPDIR": str(temporary),
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "PYTHONNOUSERSITE": "1",
            "MPLBACKEND": "Agg",
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    return env


def run_command(
    command: Iterable[object],
    *,
    cwd: Path,
    env: dict[str, str],
    output: Output,
    on_line: Callable[[str], None] | None = None,
) -> None:
    argv = [str(item) for item in command]
    output("$ " + " ".join(argv))
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        output(line)
        if on_line:
            on_line(line)
    status = process.wait()
    if status:
        raise RuntimeError(f"Command failed with status {status}: {' '.join(argv)}")


def _write_mvs_config(data_root: Path, experiment: str) -> Path:
    template = json.loads((CODE_ROOT / "mvs" / "configs" / "ciai_template.json").read_text())
    template["experiment"] = experiment
    config_dir = data_root / "mvs" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    destination = config_dir / f"{experiment}.json"
    encoded = json.dumps(template, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text() != encoded:
        raise RuntimeError(f"Existing MVS recipe differs; preserved at {destination}")
    destination.write_text(encoded, encoding="utf-8")
    return destination


def ensure_sparse_sfm(
    dataset: str,
    data_root: Path,
    progress: Progress,
    output: Output,
) -> tuple[Path, Path, Path]:
    """Create or checksum-validate the sparse calibration shared with MVS."""

    env = _runtime_environment(data_root)
    sfm_root = data_root / "sfm"
    sfm_root.mkdir(parents=True, exist_ok=True)
    progress(4, "Copying and verifying input photographs")
    images = stage_images(dataset, sfm_root / "inputs" / dataset / "images", output)

    progress(12, "Installing or checking the CUDA SfM environment")
    run_command(["bash", CODE_ROOT / "sfm" / "scripts" / "setup_linux_cuda.sh"], cwd=CODE_ROOT, env=env, output=output)
    sfm_python = sfm_root / "envs" / "headless-cuda" / "bin" / "python"
    sfm_experiment = f"ciai-{dataset}-mvs-sfm-v1"

    progress(22, "Estimating calibrated cameras and a sparse SfM model")
    run_command(
        [
            sfm_python,
            CODE_ROOT / "sfm" / "run_headless.py",
            "run",
            "--experiment",
            sfm_experiment,
            "--images",
            images,
            "--camera-model",
            "SIMPLE_RADIAL",
            "--camera-mode",
            "per_folder",
            "--matcher",
            "exhaustive",
            "--device",
            "cuda",
            "--max-image-size",
            "1600",
            "--max-num-features",
            "8192",
            "--num-threads",
            "16",
        ],
        cwd=CODE_ROOT / "sfm",
        env=env,
        output=output,
    )
    model = sfm_root / "experiments" / sfm_experiment / "outputs" / "colmap" / "sparse" / "0"
    raw_ply = sfm_root / "experiments" / sfm_experiment / "outputs" / "sparse_colored.ply"
    return images, model, raw_ply


def run_sfm(dataset: str, data_root: Path, progress: Progress, output: Output) -> Path:
    _images, model, _raw_ply = ensure_sparse_sfm(dataset, data_root, progress, output)
    env = _runtime_environment(data_root)
    sfm_python = data_root / "sfm" / "envs" / "headless-cuda" / "bin" / "python"
    cleaned_root = data_root / "sfm" / "derived" / f"ciai-{dataset}-sfm-clean-v1"
    cleaned = cleaned_root / "sparse_colored_cleaned.ply"
    receipt = cleaned_root / "cleanup_receipt.json"
    cleaned_root.mkdir(parents=True, exist_ok=True)
    progress(88, "Creating a non-destructive sparse quality cleanup")
    run_command(
        [
            sfm_python,
            CODE_ROOT / "sfm" / "clean_sparse.py",
            "--model",
            model,
            "--output",
            cleaned,
            "--receipt",
            receipt,
        ],
        cwd=CODE_ROOT / "sfm",
        env=env,
        output=output,
    )
    return cleaned


def run_mvs(dataset: str, data_root: Path, progress: Progress, output: Output) -> Path:
    env = _runtime_environment(data_root)
    images, model, _raw_ply = ensure_sparse_sfm(dataset, data_root, progress, output)

    (data_root / "mvs").mkdir(parents=True, exist_ok=True)
    progress(42, "Installing or checking the CUDA MVS environment")
    run_command(["bash", CODE_ROOT / "mvs" / "scripts" / "bootstrap_python_environment.sh"], cwd=CODE_ROOT, env=env, output=output)
    mvs_python = data_root / "mvs" / "envs" / "mvs-engine" / "bin" / "python"
    mvs_experiment = f"ciai-{dataset}-mvs1600-v1"

    progress(50, "Copying the registered images and calibrated cameras into MVS")
    run_command(
        [
            mvs_python,
            CODE_ROOT / "mvs" / "run_mvs.py",
            "stage-inputs",
            "--experiment",
            mvs_experiment,
            "--images-source",
            images,
            "--model-source",
            model,
            "--resume",
        ],
        cwd=CODE_ROOT / "mvs",
        env=env,
        output=output,
    )
    config = _write_mvs_config(data_root, mvs_experiment)

    progress(56, "Validating the MVS inputs and CUDA runtime")
    run_command(
        [mvs_python, CODE_ROOT / "mvs" / "run_mvs.py", "validate", "--config", config, "--runtime"],
        cwd=CODE_ROOT / "mvs",
        env=env,
        output=output,
    )

    stage_seen: set[str] = set()

    def mvs_stage(line: str) -> None:
        lower = line.lower()
        for marker, percent, label in (
            ("undistort", 62, "Undistorting registered photographs"),
            ("patch_match", 70, "Estimating dense depth maps with PatchMatch Stereo"),
            ("patchmatch", 70, "Estimating dense depth maps with PatchMatch Stereo"),
            ("stereo_fusion", 91, "Fusing consistent depth maps into a coloured cloud"),
            ("fusion", 91, "Fusing consistent depth maps into a coloured cloud"),
        ):
            if marker in lower and marker not in stage_seen:
                stage_seen.add(marker)
                progress(percent, label)

    progress(60, "Running COLMAP dense MVS")
    run_command(
        [mvs_python, CODE_ROOT / "mvs" / "run_mvs.py", "run", "--config", config, "--resume"],
        cwd=CODE_ROOT / "mvs",
        env=env,
        output=output,
        on_line=mvs_stage,
    )
    result = data_root / "mvs" / "experiments" / mvs_experiment / "outputs" / "fused.ply"
    run_command(
        [mvs_python, CODE_ROOT / "mvs" / "run_mvs.py", "validate", "--ply", result],
        cwd=CODE_ROOT / "mvs",
        env=env,
        output=output,
    )
    return result


def run_vggsfm(dataset: str, data_root: Path, progress: Progress, output: Output) -> Path:
    env = _runtime_environment(data_root)
    method_root = data_root / "vggsfm"
    method_root.mkdir(parents=True, exist_ok=True)
    progress(5, "Copying and verifying input photographs")
    stage_images(dataset, method_root / "inputs" / dataset / "images", output)

    python = method_root / "envs" / "vggsfm" / "bin" / "python"
    receipt = method_root / "install_receipt.json"
    if not python.is_file() or not receipt.is_file():
        progress(15, "Installing pinned VGGSfM and its CUDA environment")
        run_command(["bash", CODE_ROOT / "vggsfm" / "scripts" / "install_linux.sh"], cwd=CODE_ROOT, env=env, output=output)
    else:
        progress(15, "Checking the existing pinned VGGSfM environment")

    progress(30, "Validating the VGGSfM environment and upstream revisions")
    run_command(
        [python, CODE_ROOT / "vggsfm" / "run_vggsfm.py", "doctor", "--require-ready"],
        cwd=CODE_ROOT / "vggsfm",
        env=env,
        output=output,
    )
    profile = CODE_ROOT / "vggsfm" / "configs" / (
        "dark_shirt.json" if "dark" in dataset.lower() else "light_shirt.json"
    )
    run_id = f"ciai-{dataset}-vggsfm-v1"
    progress(35, "Planning VGGSfM inference")
    run_command(
        [python, CODE_ROOT / "vggsfm" / "run_vggsfm.py", "plan", "--dataset", dataset, "--run-id", run_id, "--profile", profile],
        cwd=CODE_ROOT / "vggsfm",
        env=env,
        output=output,
    )
    progress(40, "Downloading cached weights if needed, then running VGGSfM")

    def vgg_stage(line: str) -> None:
        lower = line.lower()
        if "download" in lower or "checkpoint" in lower:
            progress(45, "Preparing pretrained model weights in Lustre")
        elif "bundle" in lower or "ba " in lower:
            progress(78, "Refining cameras and 3D points")
        elif "export" in lower or "point_cloud" in lower:
            progress(92, "Exporting the coloured point cloud")

    run_command(
        [python, CODE_ROOT / "vggsfm" / "run_vggsfm.py", "run", "--dataset", dataset, "--run-id", run_id, "--profile", profile, "--resume"],
        cwd=CODE_ROOT / "vggsfm",
        env=env,
        output=output,
        on_line=vgg_stage,
    )
    result = method_root / "outputs" / run_id / "point_cloud.ply"
    if not result.is_file() or result.stat().st_size < 100:
        raise RuntimeError(f"VGGSfM did not produce a usable point cloud at {result}")
    return result


def run_pipeline(dataset: str, method: str, data_root: Path, progress: Progress, output: Output) -> Path:
    validate_request(dataset, method)
    runtime = validate_runtime(data_root)
    output("Runtime: " + json.dumps(runtime, sort_keys=True))
    runners = {"sfm": run_sfm, "mvs": run_mvs, "vggsfm": run_vggsfm}
    result = runners[method](dataset, data_root, progress, output)
    progress(100, f"Complete: {result.name}")
    return result.resolve(strict=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: ciai_pipeline.py DATASET {mvs|vggsfm} /absolute/data/root")
    result_path = run_pipeline(sys.argv[1], sys.argv[2], Path(sys.argv[3]), lambda p, s: print(f"[{p:3d}%] {s}", flush=True), lambda s: print(s, flush=True))
    print(result_path)
