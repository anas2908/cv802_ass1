"""Reproducible, headless orchestration of the pinned official VGGSfM demo."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from .colmap import (
    convert_points3d_to_ply,
    discover_model,
    validate_binary_model,
)
from .constants import (
    IMAGE_EXTENSIONS,
    LIGHTGLUE_COMMIT,
    LIGHTGLUE_REPOSITORY,
    MODEL_NAME,
    MODEL_REPOSITORY,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    OPTIONAL_COLMAP_FILES,
    REQUIRED_COLMAP_FILES,
)
from .errors import (
    ConfigurationError,
    ExecutionError,
    IncompleteRunError,
    OutputValidationError,
    ProvenanceError,
    StoragePolicyError,
)
from .io_utils import (
    atomic_write_json,
    file_record,
    manifest_for_files,
    object_sha256,
    read_json,
    validate_identifier,
)
from .paths import Layout
from .profile import InferenceProfile
from .resources import ResourceMonitor


COMPATIBILITY_RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "official_demo_compat.py"


def _python_function_record(
    source: Path, *, class_name: str, function_name: str, relative_to: Path
) -> dict[str, Any]:
    """Hash both an upstream file and one exact function within it."""

    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == function_name
                ),
                None,
            )
            break
    if target is None or target.end_lineno is None:
        raise ProvenanceError(
            f"Cannot locate {class_name}.{function_name} in pinned source {source}"
        )
    lines = text.splitlines(keepends=True)
    function_bytes = "".join(lines[target.lineno - 1 : target.end_lineno]).encode(
        "utf-8"
    )
    return {
        "source": file_record(source, relative_to=relative_to),
        "qualified_name": f"{class_name}.{function_name}",
        "first_line": target.lineno,
        "last_line": target.end_lineno,
        "function_sha256": hashlib.sha256(function_bytes).hexdigest(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExecutionOutcome:
    returncode: int
    elapsed_seconds: float
    resource_metrics: dict[str, Any] | None = None


class Executor(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        label: str,
    ) -> ExecutionOutcome:
        """Execute one argv (never a shell command), appending output to a log."""


class SubprocessExecutor:
    """Line-buffered executor: output is both visible and persisted under DATA_ROOT."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        label: str,
    ) -> ExecutionOutcome:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {label} @ {utc_now()} =====\n")
            log.write(json.dumps(list(command), ensure_ascii=False) + "\n")
            log.flush()
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
            monitor = ResourceMonitor(
                process.pid, log_path.parent / f"resources-{process.pid}.jsonl"
            )
            monitor.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                metrics = monitor.finish()
            elapsed = time.monotonic() - started
            log.write(
                f"===== {label} exit={returncode} elapsed={elapsed:.3f}s @ {utc_now()} =====\n"
            )
            log.flush()
        return ExecutionOutcome(returncode=returncode, elapsed_seconds=elapsed, resource_metrics=metrics)


@dataclass(frozen=True)
class RunPlan:
    dataset: str
    run_id: str
    state: str
    next_attempt: int | None
    image_count: int
    mask_count: int
    request_fingerprint: str
    official_command: tuple[str, ...] | None
    experiment_root: Path
    output_root: Path
    readiness: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "state": self.state,
            "next_attempt": self.next_attempt,
            "image_count": self.image_count,
            "mask_count": self.mask_count,
            "request_fingerprint": self.request_fingerprint,
            "official_command": list(self.official_command) if self.official_command else None,
            "experiment_root": str(self.experiment_root),
            "output_root": str(self.output_root),
            "readiness": self.readiness,
        }


@dataclass(frozen=True)
class RunResult:
    status: str
    run_id: str
    request_fingerprint: str
    output_root: Path
    attempt: int | None
    point_count: int | None
    registered_image_count: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "request_fingerprint": self.request_fingerprint,
            "output_root": str(self.output_root),
            "attempt": self.attempt,
            "point_count": self.point_count,
            "registered_image_count": self.registered_image_count,
        }


@dataclass(frozen=True)
class _PreparedRequest:
    dataset: str
    run_id: str
    images: tuple[Path, ...]
    masks: tuple[Path, ...]
    official_image_names: tuple[str, ...]
    profile: InferenceProfile
    descriptor: dict[str, Any]
    fingerprint: str


class VGGSfMEngine:
    """Owns staging, official execution, validation, normalization and receipts."""

    def __init__(
        self,
        layout: Layout,
        *,
        executor: Executor | None = None,
        verify_source: bool = True,
    ) -> None:
        self.layout = layout
        self.executor = executor or SubprocessExecutor()
        self.verify_source = verify_source

    def _source_revision(self) -> str | None:
        root = self.layout.official_root
        if not (root / ".git").exists():
            return None
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode:
            return None
        return completed.stdout.strip()

    def runtime_readiness(self) -> dict[str, Any]:
        root = self.layout.official_root
        revision = self._source_revision()
        return {
            "method_data_root_exists": self.layout.root.is_dir(),
            "method_data_root_writable": os.access(self.layout.root, os.W_OK)
            if self.layout.root.exists()
            else False,
            "environment_python": str(self.layout.python),
            "environment_python_exists": self.layout.python.is_file(),
            "official_root": str(root),
            "official_demo_exists": (root / "demo.py").is_file(),
            "official_revision": revision,
            "expected_official_revision": OFFICIAL_COMMIT,
            "official_revision_matches": revision == OFFICIAL_COMMIT,
        }

    def verify_runtime(self) -> None:
        self.layout.assert_member(self.layout.python, label="environment Python")
        self.layout.assert_member(self.layout.official_root, label="official checkout")
        if not self.layout.python.is_file() or not os.access(self.layout.python, os.X_OK):
            raise ProvenanceError(
                f"VGGSfM environment is not installed at {self.layout.env_prefix}. "
                "Run scripts/install_linux.sh inside the allocated GPU job."
            )
        if not (self.layout.official_root / "demo.py").is_file():
            raise ProvenanceError(
                f"Official VGGSfM checkout is missing at {self.layout.official_root}"
            )
        if self.verify_source:
            revision = self._source_revision()
            if revision != OFFICIAL_COMMIT:
                raise ProvenanceError(
                    "Official VGGSfM revision mismatch: "
                    f"expected {OFFICIAL_COMMIT}, found {revision or 'unreadable'}"
                )

    def _input_files(self, dataset: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        dataset_root = self.layout.dataset_root(dataset)
        self.layout.assert_member(dataset_root, label="dataset")
        images_dir = dataset_root / "images"
        if not images_dir.is_dir():
            raise ConfigurationError(
                f"Expected independent VGGSfM input images at {images_dir}"
            )
        image_candidates = (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        # The transferred capture uses camera-group subdirectories but embeds
        # one global, zero-padded capture index in every basename.  Sort by
        # basename first so interleaved cameras retain capture order.  The
        # relative path is a deterministic tie-breaker for general datasets
        # that contain equal basenames in different directories.
        images = tuple(
            sorted(
                image_candidates,
                key=lambda path: (
                    path.name.casefold(),
                    path.relative_to(images_dir).as_posix().casefold(),
                    path.relative_to(images_dir).as_posix(),
                ),
            )
        )
        if len(images) < 3:
            raise ConfigurationError(
                f"VGGSfM requires at least 3 images; found {len(images)} in {images_dir}"
            )
        image_relatives = tuple(path.relative_to(images_dir) for path in images)
        lowered = [path.as_posix().casefold() for path in image_relatives]
        if len(lowered) != len(set(lowered)):
            raise ConfigurationError("Input image paths collide under case-insensitive comparison")
        for image in images:
            self.layout.assert_member(image, label="input image", must_exist=True)

        masks_dir = dataset_root / "masks"
        masks: tuple[Path, ...] = ()
        if masks_dir.exists():
            if not masks_dir.is_dir():
                raise ConfigurationError(f"Masks path is not a directory: {masks_dir}")
            mask_by_relative = {
                path.relative_to(masks_dir).as_posix(): path
                for path in masks_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            expected = {path.as_posix() for path in image_relatives}
            if set(mask_by_relative) != expected:
                missing = sorted(expected - set(mask_by_relative))
                extra = sorted(set(mask_by_relative) - expected)
                raise ConfigurationError(
                    "Masks must have exactly the same relative paths as images; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            masks = tuple(mask_by_relative[path.as_posix()] for path in image_relatives)
            for mask in masks:
                self.layout.assert_member(mask, label="input mask", must_exist=True)
        return images, masks

    @staticmethod
    def _official_image_names(images: tuple[Path, ...], images_dir: Path) -> tuple[str, ...]:
        """Create collision-proof flat names for the official demo scene.

        The assignment groups images by camera/capture directory, whereas the
        official demo consumes one flat ``scene/images`` directory.  The short
        digest is derived from the complete relative source path, so equal
        basenames in different camera groups remain distinct.  The exact map
        is included in the immutable request receipt.
        """

        names: list[str] = []
        for index, image in enumerate(images):
            relative = image.relative_to(images_dir).as_posix()
            prefix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
            # DemoLoader sorts the flat directory lexicographically.  Keeping
            # the sequence index first therefore preserves the reviewed input
            # order; the path digest still makes equal basenames collision
            # proof and the complete mapping is recorded in request.json.
            names.append(f"{index:06d}__{prefix}__{image.name}")
        if len(names) != len(set(names)):
            raise ConfigurationError("Generated official image names are not unique")
        return tuple(names)

    def _prepare_request(
        self, dataset: str, run_id: str, profile: InferenceProfile
    ) -> _PreparedRequest:
        validate_identifier(dataset, label="dataset")
        validate_identifier(run_id, label="run id")
        images, masks = self._input_files(dataset)
        profile.validate(image_count=len(images))
        dataset_root = self.layout.dataset_root(dataset)
        images_dir = dataset_root / "images"
        official_names = self._official_image_names(images, images_dir)
        input_manifest = {
            "images": manifest_for_files(images, relative_to=dataset_root),
            "masks": manifest_for_files(masks, relative_to=dataset_root),
        }
        descriptor: dict[str, Any] = {
            "schema_version": 1,
            "method": "official_vggsfm_v2",
            "dataset": dataset,
            "input_manifest": input_manifest,
            "official_image_name_map": [
                {
                    "source": image.relative_to(dataset_root).as_posix(),
                    "official": official_name,
                }
                for image, official_name in zip(images, official_names, strict=True)
            ],
            "profile": profile.as_dict(),
            "independence": {
                "raw_images_and_optional_masks_only": True,
                "external_camera_poses_used": False,
                "sfm_e10_poses_used": False,
                "alignment_allowed_only_after_reconstruction": True,
            },
            "upstream": {
                "repository": OFFICIAL_REPOSITORY,
                "commit": OFFICIAL_COMMIT,
                "model_repository": MODEL_REPOSITORY,
                "model_name": MODEL_NAME,
            },
        }
        if profile.img_size != 1024:
            lightglue_utils = (
                self.layout.downloads / "source" / "lightglue" / "lightglue" / "utils.py"
            )
            self.layout.assert_member(
                lightglue_utils, label="pinned LightGlue source", must_exist=True
            )
            descriptor["compatibility_adapter"] = {
                "kind": "resize_lightglue_invalid_mask_to_extractor_resolution",
                "reason": (
                    "Pinned LightGlue query extractors resize images to 1024 but otherwise "
                    "leave VGGSfM's img_size-dependent invalid mask unchanged."
                ),
                "applies_only_when_img_size_is_not_1024": True,
                "mask_resize": "nearest_only_when_spatial_shape_differs",
                "mask_polarity_changed": False,
                "keypoint_coordinate_mapping_changed": False,
                "wrapper_source": file_record(COMPATIBILITY_RUNNER),
                "upstream_extractor": {
                    "repository": LIGHTGLUE_REPOSITORY,
                    "commit": LIGHTGLUE_COMMIT,
                    **_python_function_record(
                        lightglue_utils,
                        class_name="Extractor",
                        function_name="extract",
                        relative_to=self.layout.root,
                    ),
                },
                "pinned_checkouts_modified": False,
            }
        return _PreparedRequest(
            dataset=dataset,
            run_id=run_id,
            images=images,
            masks=masks,
            official_image_names=official_names,
            profile=profile,
            descriptor=descriptor,
            fingerprint=object_sha256(descriptor),
        )

    def _request_state(self, request: _PreparedRequest) -> tuple[str, int | None]:
        experiment = self.layout.experiment_root(request.run_id)
        output = self.layout.output_root(request.run_id)
        request_file = experiment / "request.json"
        if request_file.exists():
            stored = read_json(request_file)
            if stored.get("request_fingerprint") != request.fingerprint:
                raise ConfigurationError(
                    f"Run id {request.run_id!r} already belongs to a different request. "
                    "Choose a new run id; existing provenance will not be overwritten."
                )
        elif experiment.exists() and any(experiment.iterdir()):
            raise IncompleteRunError(
                f"Experiment directory exists without a valid request receipt: {experiment}"
            )

        if output.exists():
            manifest_file = output / "manifest.json"
            if not manifest_file.is_file():
                raise OutputValidationError(
                    f"Output directory exists without manifest: {output}"
                )
            manifest = read_json(manifest_file)
            if manifest.get("request_fingerprint") != request.fingerprint:
                raise OutputValidationError(
                    f"Published output fingerprint mismatch at {output}"
                )
            self._verify_published_files(output, manifest)
            return "complete", None

        attempts = experiment / "attempts"
        numbers: list[int] = []
        if attempts.is_dir():
            for item in attempts.iterdir():
                if item.is_dir() and item.name.isdigit():
                    numbers.append(int(item.name))
        next_attempt = max(numbers, default=0) + 1
        return ("incomplete" if request_file.exists() else "new"), next_attempt

    def _command(self, scene: Path, profile: InferenceProfile) -> tuple[str, ...]:
        entrypoint = self.layout.official_root / "demo.py"
        prefix = [str(self.layout.python), str(entrypoint)]
        if profile.img_size != 1024:
            if not COMPATIBILITY_RUNNER.is_file():
                raise ProvenanceError(
                    f"Non-1024 compatibility runner is missing: {COMPATIBILITY_RUNNER}"
                )
            prefix = [str(self.layout.python), str(COMPATIBILITY_RUNNER), str(entrypoint)]
        command = [
            *prefix,
            f"SCENE_DIR={scene}",
            f"model_name={MODEL_NAME}",
            "auto_download_ckpt=True",
            "load_gt=False",
            "save_to_disk=True",
            "dense_depth=False",
            "viz_visualize=False",
            "gr_visualize=False",
            "make_reproj_video=False",
            "visual_tracks=False",
            "visual_query_points=False",
            "visual_dense_point_cloud=False",
            *profile.hydra_overrides(),
        ]
        return tuple(command)

    def plan(
        self, dataset: str, run_id: str, profile: InferenceProfile
    ) -> RunPlan:
        request = self._prepare_request(dataset, run_id, profile)
        state, attempt = self._request_state(request)
        command: tuple[str, ...] | None = None
        if attempt is not None:
            scene = (
                self.layout.experiment_root(run_id)
                / "attempts"
                / f"{attempt:04d}"
                / "scene"
            )
            command = self._command(scene, profile)
        return RunPlan(
            dataset=dataset,
            run_id=run_id,
            state=state,
            next_attempt=attempt,
            image_count=len(request.images),
            mask_count=len(request.masks),
            request_fingerprint=request.fingerprint,
            official_command=command,
            experiment_root=self.layout.experiment_root(run_id),
            output_root=self.layout.output_root(run_id),
            readiness=self.runtime_readiness(),
        )

    def _stage_inputs(self, request: _PreparedRequest, scene: Path) -> None:
        images_dir = scene / "images"
        images_dir.mkdir(parents=True, exist_ok=False)
        for source, official_name in zip(
            request.images, request.official_image_names, strict=True
        ):
            self.layout.assert_member(source, label="staged input source", must_exist=True)
            os.symlink(source, images_dir / official_name)
        if request.masks:
            masks_dir = scene / "masks"
            masks_dir.mkdir(parents=True, exist_ok=False)
            for source, official_name in zip(
                request.masks, request.official_image_names, strict=True
            ):
                self.layout.assert_member(source, label="staged mask source", must_exist=True)
                os.symlink(source, masks_dir / official_name)

    def _preflight_command(self) -> tuple[str, ...]:
        program = (
            "import json, os, torch; "
            "assert torch.cuda.is_available(), 'CUDA is not available'; "
            "i=torch.cuda.current_device(); "
            "print(json.dumps({'cuda_available': True, 'device_index': i, "
            "'device_name': torch.cuda.get_device_name(i), "
            "'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
            "'slurm_job_id': os.environ.get('SLURM_JOB_ID')}))"
        )
        return (str(self.layout.python), "-c", program)

    def _write_request_if_new(self, request: _PreparedRequest) -> None:
        experiment = self.layout.experiment_root(request.run_id)
        experiment.mkdir(parents=True, exist_ok=True)
        request_file = experiment / "request.json"
        if not request_file.exists():
            payload = dict(request.descriptor)
            payload.update(
                {
                    "request_fingerprint": request.fingerprint,
                    "created_at": utc_now(),
                }
            )
            atomic_write_json(request_file, payload)

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> str:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"

    def _model_artifacts(self) -> list[dict[str, Any]]:
        """Locate and hash the exact checkpoint material used by upstream."""

        candidates: list[Path] = []
        huggingface = self.layout.cache / "huggingface" / "hub"
        if huggingface.is_dir():
            candidates.extend(
                huggingface.glob(
                    "models--facebook--VGGSfM/snapshots/*/vggsfm_v2_0_0.bin"
                )
            )
        torch_checkpoint = (
            self.layout.cache / "torch" / "hub" / "checkpoints" / "vggsfm_v2_0_0.bin"
        )
        if torch_checkpoint.is_file():
            candidates.append(torch_checkpoint)
        # Some Torch versions use TORCH_HOME/checkpoints rather than hub/checkpoints.
        alternate_torch = self.layout.cache / "torch" / "checkpoints" / "vggsfm_v2_0_0.bin"
        if alternate_torch.is_file():
            candidates.append(alternate_torch)

        # These pretrained components are initialized by the official runner
        # before the VGGSfM checkpoint is loaded. Record their exact bytes too.
        auxiliary = self.layout.cache / "torch" / "hub" / "checkpoints"
        for filename in ("dinov2_vitb14_reg4_pretrain.pth", "aliked-n16.pth"):
            if (auxiliary / filename).is_file():
                candidates.append(auxiliary / filename)

        artifacts: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for candidate in sorted(candidates):
            resolved = self.layout.assert_member(
                candidate, label="model checkpoint", must_exist=True
            )
            if resolved in seen:
                continue
            seen.add(resolved)
            record = file_record(candidate, relative_to=self.layout.root)
            record["resolved_path"] = str(resolved)
            artifacts.append(record)
        if not any(Path(item["path"]).name == "vggsfm_v2_0_0.bin" for item in artifacts):
            raise OutputValidationError(
                "Official inference completed but no vggsfm_v2_0_0.bin was found in "
                f"the redirected Hugging Face/Torch caches below {self.layout.cache}"
            )
        return artifacts

    def _publish(
        self,
        request: _PreparedRequest,
        *,
        attempt: int,
        attempt_root: Path,
        official_elapsed_seconds: float,
        started_at: str,
        resource_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        official_model = discover_model(attempt_root / "scene")
        model_validation = validate_binary_model(official_model)
        registered = int(model_validation["registered_image_count"])
        cameras = int(model_validation["camera_count"])
        if registered < 3 or registered > len(request.images):
            raise OutputValidationError(
                f"Official model registers {registered} images; expected between "
                f"3 and {len(request.images)} input views"
            )
        # Upstream may reject poorly constrained views, but every retained name
        # must map exactly to a staged input: a count alone cannot prove this.
        unexpected_names = set(model_validation["image_names"]) - set(
            request.official_image_names
        )
        if unexpected_names:
            raise OutputValidationError(
                f"Official model contains images absent from the input manifest: "
                f"{sorted(unexpected_names)}"
            )

        final = self.layout.output_root(request.run_id)
        if final.exists():
            raise OutputValidationError(f"Refusing to replace published output {final}")
        staging = self.layout.outputs / (
            f".{request.run_id}.{attempt:04d}.{os.getpid()}.{uuid.uuid4().hex}.staging"
        )
        self.layout.assert_member(staging, label="output staging directory")
        staging.mkdir(parents=True, exist_ok=False)
        normalized_model = staging / "colmap" / "sparse" / "0"
        normalized_model.mkdir(parents=True)
        transfer_modes: dict[str, str] = {}
        for name in (*REQUIRED_COLMAP_FILES, *OPTIONAL_COLMAP_FILES):
            source = official_model / name
            if source.is_file():
                transfer_modes[name] = self._link_or_copy(source, normalized_model / name)

        ply_path = staging / "point_cloud.ply"
        point_metrics = convert_points3d_to_ply(
            normalized_model / "points3D.bin", ply_path
        )
        model_artifacts = self._model_artifacts()
        output_paths = [
            path for path in normalized_model.iterdir() if path.is_file()
        ] + [ply_path]
        output_records = manifest_for_files(output_paths, relative_to=staging)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "method": "official_vggsfm_v2",
            "run_id": request.run_id,
            "dataset": request.dataset,
            "request_fingerprint": request.fingerprint,
            "attempt": attempt,
            "started_at": started_at,
            "completed_at": utc_now(),
            "official_elapsed_seconds": official_elapsed_seconds,
            "resource_metrics": resource_metrics,
            "input_image_count": len(request.images),
            "input_mask_count": len(request.masks),
            "registered_image_count": registered,
            "camera_count": cameras,
            "model_validation": model_validation,
            "point_cloud": point_metrics,
            "profile": request.profile.as_dict(),
            "upstream": request.descriptor["upstream"],
            "model_artifacts": model_artifacts,
            "auxiliary_source_snapshots": self._auxiliary_source_snapshots(),
            "independence": request.descriptor["independence"],
            "official_model_source": str(official_model),
            "normalization_transfer_modes": transfer_modes,
            "files": output_records,
            "experiment_receipt": str(attempt_root / "receipt.json"),
        }
        if "compatibility_adapter" in request.descriptor:
            manifest["compatibility_adapter"] = request.descriptor[
                "compatibility_adapter"
            ]
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, final)
        return manifest

    def _auxiliary_source_snapshots(self) -> list[dict[str, Any]]:
        """Record Torch Hub's transitive DINO source, not just its weights.

        Upstream requests DINO's main branch through Torch Hub. That cache has
        no Git metadata, so inventing a commit would be misleading. Hash and
        preserve its exact files, explicitly recording this limitation.
        """
        snapshots = []
        hub = self.layout.cache / "torch" / "hub"
        for directory in sorted(hub.glob("facebookresearch_dinov2_*")):
            self.layout.assert_member(directory, label="DINO source cache", must_exist=True)
            if not directory.is_dir():
                continue
            files = []
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                self.layout.assert_member(path, label="DINO source file", must_exist=True)
                files.append(path)
            records = manifest_for_files(files, relative_to=directory)
            snapshots.append({
                "path": str(directory),
                "repository": "https://github.com/facebookresearch/dinov2",
                "cache_directory": directory.name,
                "git_commit": None,
                "limitation": "Upstream Torch Hub main-branch cache has no Git metadata; exact local content is preserved and hashed.",
                "file_count": len(records), "files_digest": object_sha256(records), "files": records,
            })
        return snapshots

    @staticmethod
    def _verify_published_files(output: Path, manifest: dict[str, Any]) -> None:
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise OutputValidationError(f"Published manifest has no file records: {output}")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise OutputValidationError(f"Malformed file record in {output / 'manifest.json'}")
            path = output / record["path"]
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(output.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise OutputValidationError(
                    f"Published file escapes or is missing from output: {path}"
                ) from exc
            current = file_record(path, relative_to=output)
            if current["bytes"] != record.get("bytes") or current["sha256"] != record.get(
                "sha256"
            ):
                raise OutputValidationError(f"Published output checksum mismatch: {path}")

    def run(
        self,
        dataset: str,
        run_id: str,
        profile: InferenceProfile,
        *,
        resume: bool = False,
    ) -> RunResult:
        request = self._prepare_request(dataset, run_id, profile)
        state, next_attempt = self._request_state(request)
        output_root = self.layout.output_root(run_id)
        if state == "complete":
            manifest = read_json(output_root / "manifest.json")
            return RunResult(
                status="cached",
                run_id=run_id,
                request_fingerprint=request.fingerprint,
                output_root=output_root,
                attempt=manifest.get("attempt"),
                point_count=manifest.get("point_cloud", {}).get("point_count"),
                registered_image_count=manifest.get("registered_image_count"),
            )
        if state == "incomplete" and not resume:
            raise IncompleteRunError(
                f"Run {run_id!r} has an incomplete prior attempt. Re-run with --resume "
                "to create a new immutable attempt; no partial data will be overwritten."
            )
        assert next_attempt is not None

        self.verify_runtime()
        self.layout.create_runtime_directories()
        self._write_request_if_new(request)
        attempt_root = (
            self.layout.experiment_root(run_id)
            / "attempts"
            / f"{next_attempt:04d}"
        )
        self.layout.assert_member(attempt_root, label="experiment attempt")
        attempt_root.mkdir(parents=True, exist_ok=False)
        scene = attempt_root / "scene"
        scene.mkdir()
        self._stage_inputs(request, scene)
        log_path = attempt_root / "official.log"
        command = self._command(scene, profile)
        started_at = utc_now()
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "run_id": run_id,
            "dataset": dataset,
            "attempt": next_attempt,
            "request_fingerprint": request.fingerprint,
            "started_at": started_at,
            "command": list(command),
            "cwd": str(self.layout.official_root),
            "log": str(log_path),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "job_name": os.environ.get("SLURM_JOB_NAME"),
                "node_list": os.environ.get("SLURM_JOB_NODELIST"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "cache_and_temp": {
                key: value
                for key, value in self.layout.runtime_environment().items()
                if key
                in {
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
                }
            },
        }
        if "compatibility_adapter" in request.descriptor:
            receipt["compatibility_adapter"] = request.descriptor[
                "compatibility_adapter"
            ]
        receipt_path = attempt_root / "receipt.json"
        atomic_write_json(receipt_path, receipt)
        env = self.layout.runtime_environment()

        try:
            preflight = self.executor.run(
                self._preflight_command(),
                cwd=self.layout.official_root,
                env=env,
                log_path=log_path,
                label="CUDA preflight",
            )
            if preflight.returncode:
                raise ExecutionError(
                    f"CUDA preflight failed with exit code {preflight.returncode}; see {log_path}"
                )
            outcome = self.executor.run(
                command,
                cwd=self.layout.official_root,
                env=env,
                log_path=log_path,
                label="official VGGSfM inference",
            )
            # Keep successful inference measurements even if subsequent file
            # validation/publication fails. A failed wrapper is not necessarily
            # a failed neural run, and recovery must not invent its timings.
            receipt.update({
                "official_returncode": outcome.returncode,
                "official_elapsed_seconds": outcome.elapsed_seconds,
                "resource_metrics": outcome.resource_metrics,
                "cuda_preflight_elapsed_seconds": preflight.elapsed_seconds,
            })
            atomic_write_json(receipt_path, receipt)
            if outcome.returncode:
                raise ExecutionError(
                    f"Official VGGSfM failed with exit code {outcome.returncode}; see {log_path}"
                )
            manifest = self._publish(
                request,
                attempt=next_attempt,
                attempt_root=attempt_root,
                official_elapsed_seconds=outcome.elapsed_seconds,
                started_at=started_at,
                resource_metrics=outcome.resource_metrics,
            )
            receipt.update(
                {
                    "status": "complete",
                    "completed_at": utc_now(),
                    "cuda_preflight_elapsed_seconds": preflight.elapsed_seconds,
                    "official_elapsed_seconds": outcome.elapsed_seconds,
                    "resource_metrics": outcome.resource_metrics,
                    "output_manifest": str(output_root / "manifest.json"),
                    "point_count": manifest["point_cloud"]["point_count"],
                    "registered_image_count": manifest["registered_image_count"],
                    "model_artifacts": manifest["model_artifacts"],
                }
            )
            atomic_write_json(receipt_path, receipt)
            return RunResult(
                status="complete",
                run_id=run_id,
                request_fingerprint=request.fingerprint,
                output_root=output_root,
                attempt=next_attempt,
                point_count=manifest["point_cloud"]["point_count"],
                registered_image_count=manifest["registered_image_count"],
            )
        except Exception as exc:
            receipt.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            atomic_write_json(receipt_path, receipt)
            raise
