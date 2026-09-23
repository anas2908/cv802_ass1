"""Transactional, resumable execution of the headless COLMAP MVS stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from .commands import StageCommand, build_commands, set_auto_source_count
from .config import MVSConfig
from .errors import CommandExecutionError, InputValidationError, StageConflictError
from .io_utils import atomic_json, stable_digest, utc_now
from .lifecycle import archive_run_artifacts, assert_clean_run_area
from .masking import create_masked_images
from .paths import PathPolicy
from .validation import (
    stage_output_summary,
    validate_colored_ply,
    validate_inputs,
    validate_runtime,
)


@dataclass(frozen=True)
class ExecutionMetrics:
    return_code: int
    elapsed_seconds: float
    child_peak_rss_kib: int
    peak_gpu_memory_mib: int | None
    peak_gpu_utilization_percent: int | None


class _GpuMonitor:
    def __init__(self, environment: Mapping[str, str], interval_seconds: float = 2.0):
        self.environment = dict(environment)
        self.interval_seconds = interval_seconds
        self.peak_memory: int | None = None
        self.peak_utilization: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuMonitor":
        self._thread = threading.Thread(target=self._run, name="mvs-gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 2)

    def _run(self) -> None:
        # nvidia-smi can expose every physical card even when CUDA is scoped
        # by Slurm. Never attribute a different job's card to this method.
        selected = (self.environment.get("SLURM_STEP_GPUS")
                    or self.environment.get("SLURM_JOB_GPUS")
                    or self.environment.get("CUDA_VISIBLE_DEVICES"))
        if not selected or selected in {"-1", "NoDevFiles"}:
            return
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={selected}",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    env=self.environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) == 2:
                            memory, utilization = int(fields[0]), int(fields[1])
                            self.peak_memory = max(self.peak_memory or 0, memory)
                            self.peak_utilization = max(
                                self.peak_utilization or 0, utilization
                            )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_seconds)


def execute_command(
    command: StageCommand,
    *,
    environment: Mapping[str, str],
    log_path: Path,
    popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
) -> ExecutionMetrics:
    """Run one argv vector without a shell while teeing combined output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{utc_now()}] argv={json.dumps(command.argv)}\n")
        with _GpuMonitor(environment) as monitor:
            process = popen_factory(
                list(command.argv),
                env=dict(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
            except BaseException:
                # Do not orphan a GPU process when Slurm or the user interrupts
                # the controller.  The partial PatchMatch maps remain resumable.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except (AttributeError, ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (AttributeError, ProcessLookupError):
                        pass
                raise
        elapsed = time.monotonic() - started
        after_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        log.write(
            f"[{utc_now()}] return_code={return_code} elapsed_seconds={elapsed:.3f}\n"
        )
    return ExecutionMetrics(
        return_code=return_code,
        elapsed_seconds=elapsed,
        child_peak_rss_kib=max(before_rss, after_rss),
        peak_gpu_memory_mib=monitor.peak_memory,
        peak_gpu_utilization_percent=monitor.peak_utilization,
    )


class MVSRunner:
    def __init__(
        self,
        config: MVSConfig,
        policy: PathPolicy,
        *,
        command_executor: Callable[..., ExecutionMetrics] = execute_command,
    ) -> None:
        self.config = config
        self.policy = policy
        self.paths = config.paths(policy)
        self.commands = build_commands(config, self.paths)
        self.command_executor = command_executor

    def dry_run(self) -> dict[str, Any]:
        """Validate staged inputs and return commands without runtime writes."""
        input_report = validate_inputs(self.config, self.paths)
        return {
            "dry_run": True,
            "experiment": self.config.experiment,
            "config_digest": self.config.digest,
            "input_validation": input_report,
            "internal_stages": (
                ["mask_inputs"] if self.config.masking_mode == "black_background" else []
            ),
            "commands": [
                {"stage": command.name, "argv": list(command.argv), "display": command.display()}
                for command in self.commands
            ],
            "writes_performed": False,
        }

    def run(self, *, resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
        if resume and overwrite:
            raise StageConflictError("--resume and --overwrite are mutually exclusive")
        if not (self.paths.root / "inputs").is_dir():
            raise InputValidationError(
                f"Inputs are not staged for {self.config.experiment}; run the stage-inputs command first"
            )
        if overwrite:
            archive_run_artifacts(self.paths.root)
        elif not resume:
            assert_clean_run_area(self.paths.root)
        self._validate_resume_identity(resume)

        # Completed results are read-only artifacts. In particular, do not
        # replace their original runtime with cache-validation time or require
        # a new CUDA allocation merely to load them.
        if resume and (cached := self._completed_run_cache()) is not None:
            return cached

        self.paths.receipts.mkdir(parents=True, exist_ok=True)
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        self.paths.manifests.mkdir(parents=True, exist_ok=True)
        self.paths.outputs.mkdir(parents=True, exist_ok=True)
        # COLMAP's undistorter creates dense/ but requires its parent to exist.
        # Foreground masking used to create work/ incidentally; raw/unmasked
        # runs must establish the same directory precondition independently.
        self.paths.work.mkdir(parents=True, exist_ok=True)

        started_monotonic = time.monotonic()
        input_report = validate_inputs(self.config, self.paths)
        atomic_json(self.paths.manifests / "input_validation.json", input_report)
        runtime_report, environment = validate_runtime(
            self.config, self.paths, self.policy, self.commands
        )
        atomic_json(self.paths.manifests / "runtime_validation.json", runtime_report)
        run_manifest = {
            "schema_version": 1,
            "started_at": utc_now(),
            "experiment": self.config.experiment,
            "config": self.config.raw,
            "config_digest": self.config.digest,
            "input_validation": input_report,
            "runtime_validation": runtime_report,
            "commands": [
                {"stage": command.name, "argv": list(command.argv)}
                for command in self.commands
            ],
            "resume_requested": resume,
            "overwrite_requested": overwrite,
        }
        atomic_json(self.paths.manifests / "run_manifest.json", run_manifest)
        self._write_state("running", None)

        completed: list[dict[str, Any]] = []
        try:
            if self.config.masking_mode == "black_background":
                completed.append(self._run_mask_stage(resume=resume))
            for command in self.commands:
                completed.append(
                    self._run_command_stage(command, environment=environment, resume=resume)
                )
            final = {
                "schema_version": 1,
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "experiment": self.config.experiment,
                "method": "COLMAP CUDA PatchMatch Stereo and stereo fusion",
                "config_digest": self.config.digest,
                "registered_images": input_report["registered_images"],
                "colored_dense_point_cloud": validate_colored_ply(self.paths.fused),
                "mesh": (
                    validate_colored_ply(self.paths.mesh)
                    if self.config.meshing != "none"
                    else None
                ),
                "stage_receipts": completed,
                "limitations": [
                    "No geometric ground truth is available; completeness and metric accuracy are not claimed.",
                    "Black-background masking, when enabled, is not the historical sparse 90% consensus filter.",
                    "Dense reconstruction quality depends on texture, view overlap, motion, and calibration.",
                ],
            }
            atomic_json(self.paths.manifests / "final_validation.json", final)
            self._write_state("complete", None)
            return final
        except BaseException as error:
            self._write_state("failed", f"{type(error).__name__}: {error}")
            raise

    def _completed_run_cache(self) -> dict[str, Any] | None:
        """Validate a completed run without writing or probing the GPU runtime.

        A complete marker is a promise: missing/corrupt evidence fails closed,
        rather than silently falling through to reconstruction. Incomplete runs
        continue through the existing stage-resume path.
        """
        state_path = self.paths.root / "run_state.json"
        final_path = self.paths.manifests / "final_validation.json"

        def read_object(path: Path, label: str) -> dict[str, Any]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise StageConflictError(f"Cannot reload {label}: {error}") from error
            if not isinstance(value, dict):
                raise StageConflictError(f"Cannot reload {label}: expected a JSON object")
            return value

        if not state_path.exists():
            if final_path.exists():
                raise StageConflictError("Completed final validation has no run-state record")
            return None
        state = read_object(state_path, "run state")
        if state.get("status") != "complete":
            return None

        final = read_object(final_path, "completed final validation")
        manifest = read_object(self.paths.manifests / "run_manifest.json", "run manifest")
        for label, record in (("run state", state), ("final validation", final), ("run manifest", manifest)):
            if record.get("config_digest") != self.config.digest or record.get("experiment") != self.config.experiment:
                raise StageConflictError(f"Completed {label} belongs to different settings")

        inputs = validate_inputs(self.config, self.paths)
        original_inputs = manifest.get("input_validation")
        if not isinstance(original_inputs, dict):
            raise StageConflictError("Completed run has no original input validation")
        # Validation timestamps naturally change on a read. Everything else
        # (including provenance digest, camera/image counts and dimensions)
        # must be identical to the original run's validated input identity.
        without_time = lambda value: {key: item for key, item in value.items() if key != "validated_at"}
        if without_time(inputs) != without_time(original_inputs):
            raise StageConflictError("Completed run input identity changed")
        if final.get("registered_images") != inputs.get("registered_images"):
            raise StageConflictError("Completed final validation has a different image count")

        expected = [(command.name, command.argv) for command in self.commands]
        if self.config.masking_mode == "black_background":
            expected.insert(0, ("mask_inputs", None))
        saved_receipts = final.get("stage_receipts")
        if not isinstance(saved_receipts, list) or any(not isinstance(row, dict) for row in saved_receipts):
            raise StageConflictError("Completed final validation has malformed stage receipts")
        saved_stages = [row.get("stage") for row in saved_receipts]
        if saved_stages != [stage for stage, _ in expected]:
            raise StageConflictError("Completed final validation is missing expected stage receipts")
        for (stage, argv), saved in zip(expected, saved_receipts):
            current = self._completed_receipt(stage, argv)
            if current is None:
                raise StageConflictError(f"Completed run has no complete {stage} receipt")
            # _completed_receipt marks only its in-memory dictionary resumed.
            # Preserve the original final report's stage flags and timings.
            unchanged = lambda value: {key: item for key, item in value.items() if key != "resumed"}
            if unchanged(current) != unchanged(saved):
                raise StageConflictError(f"Completed {stage} receipt differs from final validation")

        cloud = validate_colored_ply(self.paths.fused)
        if cloud != final.get("colored_dense_point_cloud"):
            raise StageConflictError("Completed fused PLY differs from final validation")
        mesh = validate_colored_ply(self.paths.mesh) if self.config.meshing != "none" else None
        if mesh != final.get("mesh"):
            raise StageConflictError("Completed mesh differs from final validation")
        return {**final, "cache_hit": True}

    def _validate_resume_identity(self, resume: bool) -> None:
        manifest_path = self.paths.manifests / "run_manifest.json"
        if not resume or not manifest_path.exists():
            return
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StageConflictError(f"Cannot resume malformed run manifest: {error}") from error
        if previous.get("config_digest") != self.config.digest:
            raise StageConflictError(
                "Configuration changed since the previous run; use a new experiment or --overwrite"
            )

    def _write_state(self, status: str, error: str | None) -> None:
        atomic_json(
            self.paths.root / "run_state.json",
            {
                "schema_version": 1,
                "updated_at": utc_now(),
                "experiment": self.config.experiment,
                "status": status,
                "error": error,
                "config_digest": self.config.digest,
            },
        )

    def _receipt_path(self, stage: str) -> Path:
        return self.paths.receipts / f"{stage}.json"

    def _command_digest(self, stage: str, argv: tuple[str, ...] | None) -> str:
        return stable_digest(
            {
                "stage": stage,
                "argv": list(argv) if argv else None,
                "config_digest": self.config.digest,
            }
        )

    def _completed_receipt(
        self, stage: str, argv: tuple[str, ...] | None
    ) -> dict[str, Any] | None:
        path = self._receipt_path(stage)
        if not path.exists():
            return None
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StageConflictError(f"Malformed receipt for {stage}: {error}") from error
        if receipt.get("status") != "complete":
            return None
        if receipt.get("command_digest") != self._command_digest(stage, argv):
            raise StageConflictError(
                f"Completed {stage} receipt belongs to different settings; use --overwrite"
            )
        current = stage_output_summary(stage, self.config, self.paths)
        if receipt.get("output_digest") != stable_digest(current):
            raise StageConflictError(
                f"Completed {stage} artifacts changed after validation; use --overwrite"
            )
        receipt["resumed"] = True
        return receipt

    def _run_mask_stage(self, *, resume: bool) -> dict[str, Any]:
        if resume and (receipt := self._completed_receipt("mask_inputs", None)):
            return receipt
        started = time.monotonic()
        try:
            create_masked_images(self.config, self.paths)
            summary = stage_output_summary("mask_inputs", self.config, self.paths)
            receipt = self._make_receipt(
                "mask_inputs", None, summary, time.monotonic() - started, None
            )
            atomic_json(self._receipt_path("mask_inputs"), receipt)
            return receipt
        except BaseException as error:
            self._failure_receipt("mask_inputs", None, started, error)
            raise

    def _run_command_stage(
        self,
        command: StageCommand,
        *,
        environment: Mapping[str, str],
        resume: bool,
    ) -> dict[str, Any]:
        if resume and (receipt := self._completed_receipt(command.name, command.argv)):
            return receipt
        self._prepare_incomplete_stage(command.name, resume=resume)
        started = time.monotonic()
        try:
            metrics = self.command_executor(
                command,
                environment=environment,
                log_path=self.paths.logs / f"{command.name}.log",
            )
            if metrics.return_code != 0:
                raise CommandExecutionError(
                    f"COLMAP stage {command.name} exited with code {metrics.return_code}; "
                    f"see {self.paths.logs / (command.name + '.log')}"
                )
            if command.name == "undistort":
                configured = set_auto_source_count(
                    self.paths.workspace / "stereo" / "patch-match.cfg",
                    self.config.source_images_per_view,
                )
                model_count = len(load_model_records_for_runner(self.paths.model))
                if configured != model_count:
                    raise InputValidationError(
                        f"patch-match.cfg contains {configured} views; expected {model_count}"
                    )
            summary = stage_output_summary(command.name, self.config, self.paths)
            receipt = self._make_receipt(
                command.name,
                command.argv,
                summary,
                metrics.elapsed_seconds,
                {
                    "child_peak_rss_kib": metrics.child_peak_rss_kib,
                    "peak_gpu_memory_mib": metrics.peak_gpu_memory_mib,
                    "peak_gpu_utilization_percent": metrics.peak_gpu_utilization_percent,
                },
            )
            atomic_json(self._receipt_path(command.name), receipt)
            return receipt
        except BaseException as error:
            self._failure_receipt(command.name, command.argv, started, error)
            raise

    def _prepare_incomplete_stage(self, stage: str, *, resume: bool) -> None:
        """Preserve non-resumable partial artifacts before retrying."""
        if not resume:
            return
        targets: list[Path] = []
        if stage == "undistort" and self.paths.workspace.exists():
            targets.append(self.paths.workspace)
        elif stage == "fusion" and self.paths.fused.exists():
            targets.append(self.paths.fused)
        elif stage == "mesh" and self.paths.mesh.exists():
            targets.append(self.paths.mesh)
        # PatchMatch is intentionally not moved: COLMAP's documented resume
        # behavior skips already completed depth/normal maps.
        if targets:
            archive = self.paths.root / "archive" / f"partial-{stage}-{utc_now().replace(':', '')}-{uuid.uuid4().hex[:8]}"
            archive.mkdir(parents=True, exist_ok=False)
            for target in targets:
                os.replace(target, archive / target.name)

    def _make_receipt(
        self,
        stage: str,
        argv: tuple[str, ...] | None,
        output: dict[str, Any],
        elapsed_seconds: float,
        metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": stage,
            "status": "complete",
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "argv": list(argv) if argv else None,
            "command_digest": self._command_digest(stage, argv),
            "config_digest": self.config.digest,
            "output": output,
            "output_digest": stable_digest(output),
            "metrics": metrics,
            "resumed": False,
        }

    def _failure_receipt(
        self,
        stage: str,
        argv: tuple[str, ...] | None,
        started: float,
        error: BaseException,
    ) -> None:
        atomic_json(
            self._receipt_path(stage),
            {
                "schema_version": 1,
                "stage": stage,
                "status": "failed",
                "failed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "argv": list(argv) if argv else None,
                "command_digest": self._command_digest(stage, argv),
                "config_digest": self.config.digest,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def load_model_records_for_runner(model_path: Path) -> tuple[Any, ...]:
    """Small indirection kept mockable in unit tests."""
    from .colmap_model import load_model_records

    return load_model_records(model_path).images
