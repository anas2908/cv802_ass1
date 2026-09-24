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

# These are the two reviewed recipes recorded by the completed assignment
# runs.  Light MVS starts from E10 and uses its foreground masks.  Dark MVS
# starts from E3; VGGSfM remains independent and never consumes either model.
MVS_REFERENCE_RECIPES: dict[str, dict[str, object]] = {
    "light_shirt": {
        "experiment": "light_e10_colmap_mvs_1600",
        "reference_experiment": "light_e10_colmap_mvs_1600",
        "sfm_label": "E10 quality exhaustive guided consensus90",
        "use_masks": True,
        "max_image_size": 1600,
        "num_threads": 16,
        "patch_match": {
            "cache_size_gb": 16,
            "filter_min_num_consistent": 2,
            "geom_consistency": True,
            "num_iterations": 5,
            "num_samples": 15,
            "source_images_per_view": 10,
            "window_radius": 5,
            "window_step": 1,
        },
        "fusion": {
            "cache_size_gb": 16,
            "check_num_images": 50,
            "max_depth_error": 0.01,
            "max_normal_error": 10.0,
            "max_reproj_error": 2.0,
            "min_num_pixels": 5,
        },
    },
    "dark_shirt": {
        "experiment": "dark_e3_colmap_mvs_1024_raw_v1",
        "reference_experiment": "dark_e3_colmap_mvs_1024_raw_v1",
        "sfm_label": "E3 selective-pair quality model",
        "use_masks": False,
        "max_image_size": 1024,
        "num_threads": 4,
        "patch_match": {
            "cache_size_gb": 8,
            "filter_min_num_consistent": 2,
            "geom_consistency": True,
            "num_iterations": 3,
            "num_samples": 15,
            "source_images_per_view": 5,
            "window_radius": 5,
            "window_step": 1,
        },
        "fusion": {
            "cache_size_gb": 8,
            "check_num_images": 50,
            "max_depth_error": 0.01,
            "max_normal_error": 10.0,
            "max_reproj_error": 2.0,
            "min_num_pixels": 5,
        },
    },
}


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
    if method not in {"sfm", "mvs", "vggsfm", "vggsfm_cleanup", "vggsfm_mask_cleanup"}:
        raise ValueError("Unknown reconstruction or cleanup method")


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


def _write_mvs_config(
    data_root: Path, experiment: str, recipe: dict[str, object] | None = None
) -> Path:
    template = json.loads((CODE_ROOT / "mvs" / "configs" / "ciai_template.json").read_text())
    template["experiment"] = experiment
    if recipe is not None:
        template["max_image_size"] = int(recipe["max_image_size"])
        template["execution"]["num_threads"] = int(recipe["num_threads"])
        template["patch_match"] = dict(recipe["patch_match"])
        template["fusion"] = dict(recipe["fusion"])
        if bool(recipe["use_masks"]):
            template["masking"] = {
                "mode": "black_background",
                "masks": "inputs/masks",
                "manifest": "inputs/mask_manifest.json",
                "threshold": 128,
                "dilation_pixels": 3,
            }
        else:
            template["masking"] = {"mode": "none"}
    config_dir = data_root / "mvs" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    destination = config_dir / f"{experiment}.json"
    encoded = json.dumps(template, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text() != encoded:
        raise RuntimeError(f"Existing MVS recipe differs; preserved at {destination}")
    destination.write_text(encoded, encoding="utf-8")
    return destination


def _reference_mvs_inputs(dataset: str, data_root: Path) -> Path | None:
    """Locate the verified inputs from the successful assignment MVS run.

    Existing staged inputs in the new data root win.  A prior assignment data
    root can be supplied explicitly with CV802_REFERENCE_DATA_ROOT.  For the
    original CIAI layout we also recognize the sibling ``cv_802_ass1`` root.
    No home-directory path or user name is hard-coded.
    """

    recipe = MVS_REFERENCE_RECIPES[dataset]
    experiment = str(recipe["experiment"])
    current = data_root / "mvs" / "experiments" / experiment
    if (current / "manifests" / "input_provenance.json").is_file():
        return current / "inputs"

    roots: list[Path] = []
    configured = os.environ.get("CV802_REFERENCE_DATA_ROOT", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise RuntimeError("CV802_REFERENCE_DATA_ROOT must be an absolute path")
        roots.append(candidate.resolve())
    roots.append(data_root.parent / "cv_802_ass1")

    for root in roots:
        inputs = (
            root
            / "mvs"
            / "experiments"
            / str(recipe["reference_experiment"])
            / "inputs"
        )
        if (
            (inputs / "images").is_dir()
            and (inputs / "sparse" / "cameras.bin").is_file()
            and (inputs / "sparse" / "images.bin").is_file()
            and (inputs / "sparse" / "points3D.bin").is_file()
        ):
            if bool(recipe["use_masks"]) and not (
                (inputs / "masks").is_dir()
                and (inputs / "mask_manifest.json").is_file()
            ):
                continue
            return inputs
    return None


def _prepare_reference_mvs_sources(
    dataset: str,
    reference: Path,
    data_root: Path,
    output: Output,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Import reviewed calibration/masks and prove repository photos match.

    The MVS engine deliberately accepts staging sources only from the active
    data root.  Legacy assignment results may be in a sibling data root, so we
    copy just the model and required masks into a checksum-verified reference
    area.  Photographs are staged from the repository and checked against the
    successful run's provenance instead of being duplicated from the legacy
    experiment.
    """

    recipe = MVS_REFERENCE_RECIPES[dataset]
    images = stage_images(
        dataset, data_root / "sfm" / "inputs" / dataset / "images", output
    )
    provenance_path = reference.parent / "manifests" / "input_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"Reviewed MVS input provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    image_records = [
        record for record in provenance.get("files", [])
        if isinstance(record, dict) and record.get("role") == "registered_image"
    ]
    for record in image_records:
        relative = Path(str(record["destination_relative"])).relative_to("inputs/images")
        image = images / relative
        if (
            not image.is_file()
            or image.stat().st_size != int(record["size_bytes"])
            or _sha256(image) != record["sha256"]
        ):
            raise RuntimeError(
                f"Repository photograph differs from the reviewed MVS input: {relative}"
            )
    if len(image_records) < 2:
        raise RuntimeError("Reviewed MVS provenance contains no usable image inventory")
    output(f"Verified {len(image_records)} photographs against the saved MVS provenance")

    try:
        reference.resolve(strict=True).relative_to(data_root.resolve(strict=True))
        inside_active_root = True
    except ValueError:
        inside_active_root = False
    if inside_active_root:
        masks = reference / "masks" if bool(recipe["use_masks"]) else None
        manifest = reference / "mask_manifest.json" if masks else None
        return images, reference / "sparse", masks, manifest

    destination = data_root / "mvs" / "reference_inputs" / dataset
    wanted: list[tuple[Path, Path]] = [
        (source, Path("sparse") / source.name)
        for source in sorted((reference / "sparse").iterdir())
        if source.is_file()
    ]
    if bool(recipe["use_masks"]):
        wanted.extend(
            (source, Path("masks") / source.relative_to(reference / "masks"))
            for source in sorted((reference / "masks").rglob("*"))
            if source.is_file()
        )
        wanted.append((reference / "mask_manifest.json", Path("mask_manifest.json")))

    def verify_import(root: Path) -> None:
        actual = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
        expected = sorted(relative for _source, relative in wanted)
        if actual != expected:
            raise RuntimeError(f"Reviewed MVS import inventory differs at {root}")
        for source, relative in wanted:
            copied = root / relative
            if copied.stat().st_size != source.stat().st_size or _sha256(copied) != _sha256(source):
                raise RuntimeError(f"Reviewed MVS import checksum mismatch: {copied}")

    if destination.exists():
        verify_import(destination)
        output(f"Reused checksum-verified {recipe['sfm_label']} calibration")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{dataset}-reference-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "reference"
            staged.mkdir()
            for source, relative in wanted:
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                before = source.stat()
                digest = _sha256(source)
                shutil.copy2(source, target)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Reviewed MVS source changed during import: {source}")
                if target.stat().st_size != before.st_size or _sha256(target) != digest:
                    raise RuntimeError(f"Reviewed MVS import checksum mismatch: {target}")
            verify_import(staged)
            staged.rename(destination)
        output(f"Imported and verified {recipe['sfm_label']} calibration")

    masks = destination / "masks" if bool(recipe["use_masks"]) else None
    manifest = destination / "mask_manifest.json" if masks else None
    return images, destination / "sparse", masks, manifest


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
    recipe = MVS_REFERENCE_RECIPES.get(dataset)
    if recipe is not None:
        reference = _reference_mvs_inputs(dataset, data_root)
        if reference is None:
            raise RuntimeError(
                f"The reviewed {recipe['sfm_label']} MVS inputs are unavailable. "
                "Set CV802_REFERENCE_DATA_ROOT to the assignment data root that "
                f"contains mvs/experiments/{recipe['reference_experiment']}/inputs."
            )
        images, model, masks, mask_manifest = _prepare_reference_mvs_sources(
            dataset, reference, data_root, output
        )
        mvs_experiment = str(recipe["experiment"])
        output(f"MVS calibration: {recipe['sfm_label']} ({reference})")
    else:
        images, model, _raw_ply = ensure_sparse_sfm(dataset, data_root, progress, output)
        masks = mask_manifest = None
        mvs_experiment = f"ciai-{dataset}-mvs1600-v1"

    (data_root / "mvs").mkdir(parents=True, exist_ok=True)
    progress(42, "Installing or checking the CUDA MVS environment")
    run_command(["bash", CODE_ROOT / "mvs" / "scripts" / "bootstrap_python_environment.sh"], cwd=CODE_ROOT, env=env, output=output)
    mvs_python = data_root / "mvs" / "envs" / "mvs-engine" / "bin" / "python"

    progress(50, "Copying the reviewed registered images and calibrated cameras into MVS")
    stage_command: list[object] = [
        mvs_python,
        CODE_ROOT / "mvs" / "run_mvs.py",
        "stage-inputs",
        "--experiment",
        mvs_experiment,
        "--images-source",
        images,
        "--model-source",
        model,
    ]
    if recipe is not None and bool(recipe["use_masks"]):
        stage_command.extend(
            [
                "--masks-source",
                masks,
                "--mask-manifest-source",
                mask_manifest,
            ]
        )
    stage_command.append("--resume")
    run_command(
        stage_command,
        cwd=CODE_ROOT / "mvs",
        env=env,
        output=output,
    )
    config = _write_mvs_config(data_root, mvs_experiment, recipe)

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


def run_vggsfm_cleanup(dataset: str, data_root: Path, progress: Progress, output: Output) -> Path:
    """Clean an existing VGGSfM cloud without rerunning learned inference."""
    run_id = f"ciai-{dataset}-vggsfm-v1"
    method_root = data_root / "vggsfm"
    raw = method_root / "outputs" / run_id / "point_cloud.ply"
    if not raw.is_file():
        raise FileNotFoundError(f"No raw VGGSfM cloud at {raw}; run VGGSfM first")
    cleaned = method_root / "outputs" / f"{run_id}-geometry-clean-v1" / "point_cloud.ply"
    python = method_root / "envs" / "vggsfm" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"VGGSfM environment is missing: {python}")
    progress(15, "Checking raw VGGSfM output")
    progress(35, "Removing isolated spatial outliers (CPU; no inference rerun)")
    run_command(
        [python, CODE_ROOT / "vggsfm" / "scripts" / "clean_point_cloud.py",
         "--source", raw, "--output", cleaned],
        cwd=CODE_ROOT / "vggsfm", env=_runtime_environment(data_root), output=output,
    )
    progress(95, "Verifying the separate cleaned cloud")
    if not cleaned.is_file() or cleaned.stat().st_size < 100:
        raise RuntimeError(f"Cleanup did not produce a usable cloud at {cleaned}")
    return cleaned


def run_vggsfm_mask_cleanup(dataset: str, data_root: Path, progress: Progress, output: Output) -> Path:
    """Apply the historical Mac masks in this VGGSfM run's own camera frame."""
    if dataset not in {"light_shirt", "dark_shirt"}:
        raise ValueError("Historical Mac masks exist only for light_shirt and dark_shirt")
    method_root = data_root / "vggsfm"
    run_id = f"ciai-{dataset}-vggsfm-v1"
    raw = method_root / "outputs" / run_id / "point_cloud.ply"
    if not raw.is_file():
        raise FileNotFoundError(f"No raw VGGSfM cloud at {raw}; run VGGSfM first")
    python = method_root / "envs" / "vggsfm" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"VGGSfM environment is missing: {python}")
    reference = os.environ.get("CV802_REFERENCE_DATA_ROOT", "").strip()
    if reference and not Path(reference).expanduser().is_absolute():
        raise ValueError("CV802_REFERENCE_DATA_ROOT must be an absolute path")
    reference_root = Path(reference).expanduser().resolve() if reference else data_root.parent / "cv_802_ass1"
    progress(10, "Verifying and copying the historical Mac masks into VGGSfM storage")
    result = method_root / "outputs" / f"{run_id}-mac-mask-clean-v1" / "point_cloud.ply"

    def mask_stage(line: str) -> None:
        match = re.search(r"Mask projection (\d+)/(\d+) views", line)
        if match:
            progress(20 + int(65 * int(match.group(1)) / int(match.group(2))),
                     f"Projecting points into Mac masks: {match.group(1)}/{match.group(2)} views")

    run_command(
        [python, CODE_ROOT / "vggsfm" / "scripts" / "clean_with_mac_masks.py",
         "--dataset", dataset, "--data-root", data_root,
         "--reference-root", reference_root],
        cwd=CODE_ROOT / "vggsfm", env=_runtime_environment(data_root),
        output=output, on_line=mask_stage,
    )
    progress(95, "Verifying the separate mask-cleaned point cloud")
    if not result.is_file() or result.stat().st_size < 100:
        raise RuntimeError(f"Mac-mask cleanup did not produce a usable cloud at {result}")
    return result


def run_pipeline(dataset: str, method: str, data_root: Path, progress: Progress, output: Output) -> Path:
    validate_request(dataset, method)
    runtime = validate_runtime(data_root)
    output("Runtime: " + json.dumps(runtime, sort_keys=True))
    runners = {"sfm": run_sfm, "mvs": run_mvs, "vggsfm": run_vggsfm,
               "vggsfm_cleanup": run_vggsfm_cleanup,
               "vggsfm_mask_cleanup": run_vggsfm_mask_cleanup}
    result = runners[method](dataset, data_root, progress, output)
    progress(100, f"Complete: {result.name}")
    return result.resolve(strict=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: ciai_pipeline.py DATASET METHOD /absolute/data/root")
    result_path = run_pipeline(sys.argv[1], sys.argv[2], Path(sys.argv[3]), lambda p, s: print(f"[{p:3d}%] {s}", flush=True), lambda s: print(s, flush=True))
    print(result_path)
